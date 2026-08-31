"""BFS Head Swap Sampler (LTX) — crop, mask and temporal loop in one node.

Everything is optional, so the node degrades to whatever you connect:

    guide_video + identity_image                  -> plain head swap, one pass
    + subject_mask                                -> native inpainting: only the
                                                     masked region is denoised
    + crop_mode                                   -> the swap runs inside a stable
                                                     box around the subject and is
                                                     pasted back afterwards
    + temporal_tile_size < frame count            -> chunked sampling with overlap

The LoRA is never asked to understand a mask. It keeps doing its job (guide on
source_id 1, identity on source_id 2) and the mask acts through ComfyUI's own
inpainting path: `latent["noise_mask"]` reaches the guider as `denoise_mask`, so
pixels outside the mask keep the guide's own content and the original face is
never hidden from the model.

Why crop matters: at 512x288 a person filling a fifth of the frame leaves a face
about 25 px tall. No LoRA recovers identity from that. Cropping the head region
and sampling it full-frame gives the same face 200-300 px, then the result is
feathered back into the untouched frames.

The crop planner comes from drozbay's MaskVidExperiments when that pack is
installed (https://github.com/drozbay/MaskVidExperiments) — its boxes hold still
through mask noise and occlusion, which naive per-frame crops do not, and a
jittering crop reads to a video model as camera motion. Without the pack the node
falls back to one static box around the subject's whole travel.
"""

import logging

import torch

import comfy.model_management
import comfy.utils

log = logging.getLogger("BFS.HeadSwapMasked")

CATEGORY = "BFS/video"


# ─────────────────────────────────────────────────────────────────────────────
# optional dependency: drozbay/MaskVidExperiments crop planner
# ─────────────────────────────────────────────────────────────────────────────

def _mvex_planner():
    """`_plan_and_crop` from MaskVidExperiments, or None when it is not installed."""
    import importlib
    import os
    import sys

    import folder_paths

    for root in folder_paths.get_folder_paths("custom_nodes"):
        path = os.path.join(root, "MaskVidExperiments", "nodes_subject_crop.py")
        if not os.path.isfile(path):
            continue
        try:
            spec = importlib.util.spec_from_file_location("_bfs_mvex_crop", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_bfs_mvex_crop"] = mod
            spec.loader.exec_module(mod)
            return mod._plan_and_crop
        except Exception as exc:  # pragma: no cover - depends on their internals
            log.warning("MaskVidExperiments found but not usable (%s); using the static box", exc)
            return None
    return None


def _static_box(masks, images, crop_scale, divisible_by, thresh=0.1):
    """Fallback crop: one box around the subject's whole travel, held for the clip."""
    m = masks if masks.ndim == 3 else masks.squeeze(-1)
    hits = (m > thresh).any(dim=0)
    ys, xs = torch.where(hits)
    H, W = m.shape[-2], m.shape[-1]
    if ys.numel() == 0:
        return 0, 0, W, H
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    h = max(1.0, (y1 - y0) * crop_scale)
    w = max(1.0, (x1 - x0) * crop_scale)
    h = min(H, (int(h) + divisible_by - 1) // divisible_by * divisible_by)
    w = min(W, (int(w) + divisible_by - 1) // divisible_by * divisible_by)
    y0 = int(min(max(0, cy - h / 2), H - h))
    x0 = int(min(max(0, cx - w / 2), W - w))
    return x0, y0, int(w), int(h)


# ─────────────────────────────────────────────────────────────────────────────
# mask helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grow_blur(masks, grow, blur):
    """Dilate then soften a mask stack, in pixels."""
    m = masks.unsqueeze(1).float()  # (N,1,H,W)
    if grow > 0:
        k = 2 * int(grow) + 1
        m = torch.nn.functional.max_pool2d(m, k, stride=1, padding=k // 2)
    if blur > 0:
        k = 2 * int(blur) + 1
        m = torch.nn.functional.avg_pool2d(m, k, stride=1, padding=k // 2, count_include_pad=False)
    return m.squeeze(1).clamp(0, 1)


def _mask_to_latent(masks, vae, latent_t, latent_h, latent_w):
    """Pixel masks -> a latent-grid mask, reduced with MAX.

    ComfyUI would trilinearly resize the pixel mask instead, which blurs it
    across frames and lets the original content bleed through the edit -- the
    failure MaskVidExperiments' Mask To Latent Space node was written to fix.
    Max keeps a latent cell that any masked pixel touches fully editable.
    """
    m = masks.unsqueeze(1).float()  # (N,1,H,W)
    m = torch.nn.functional.adaptive_max_pool2d(m, (latent_h, latent_w))  # (N,1,h,w)
    n = m.shape[0]
    # frames per latent frame: LTX keeps frame 0 alone, then groups by t_sf
    t_sf = int(vae.downscale_index_formula[0]) if hasattr(vae, "downscale_index_formula") else 8
    groups, start = [], 0
    for i in range(latent_t):
        span = 1 if i == 0 else t_sf
        end = min(n, start + span)
        if start >= n:
            groups.append(m[-1:])
        else:
            groups.append(m[start:end].amax(dim=0, keepdim=True))
        start = end
    out = torch.cat(groups, dim=0)          # (latent_t,1,h,w)
    return out.permute(1, 0, 2, 3).unsqueeze(0)  # (1,1,latent_t,h,w)


# ─────────────────────────────────────────────────────────────────────────────
# node
# ─────────────────────────────────────────────────────────────────────────────

class BFSHeadSwapMaskedSampler:
    """Head swap with an optional stable crop, native mask inpainting and looping."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "noise": ("NOISE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "guider": ("GUIDER", {"tooltip": "Provides CFG/STG settings; its conds are replaced per chunk."}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "guide_video": ("IMAGE", {"tooltip": "Source clip: body, motion, camera, scene. Output geometry follows it."}),
                "identity_image": ("IMAGE", {"tooltip": "Head/face reference. Crop to the head."}),
            },
            "optional": {
                "subject_mask": ("MASK", {"tooltip":
                    "Per-frame mask of the region to edit (head, with margin). Drives the crop box and, "
                    "with inpaint_with_mask on, restricts denoising to it. Leave unconnected for a plain swap."}),

                "crop_mode": (["off", "combined", "tracked", "zoomed"], {"default": "off", "tooltip":
                    "Sample inside a box around the subject instead of the whole frame -- the fix for faces "
                    "that are too small to carry identity. combined: one static box for the clip. "
                    "tracked/zoomed: MaskVidExperiments' planner (installed separately); without it these "
                    "fall back to combined."}),
                "crop_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip":
                    "Box size as a multiple of the subject. 1.5 leaves a third as margin. Keep neck and "
                    "shoulders in: a face-tight crop is a framing the LoRA never saw in training."}),
                "crop_divisible_by": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
                "uncrop_feather": ("INT", {"default": 16, "min": 0, "max": 256, "tooltip":
                    "Blend width when pasting the crop back, in pixels."}),

                "inpaint_with_mask": ("BOOLEAN", {"default": True, "tooltip":
                    "Send the mask to the sampler as a denoise mask, so only the masked region changes and "
                    "everything else stays the guide's own pixels. Native ComfyUI inpainting -- the LoRA "
                    "never sees the mask and the original face stays visible to the model."}),
                "mask_grow": ("INT", {"default": 8, "min": 0, "max": 256, "tooltip":
                    "Dilate the mask before use, in pixels. The new head can be bigger than the old one."}),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 256, "tooltip":
                    "Soften the mask edge, in pixels, to avoid a hard seam."}),

                "temporal_tile_size": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 8, "tooltip":
                    "Frames per chunk. 0 samples the whole clip in one pass. Use the length the LoRA trained "
                    "at (73 for the LTX head-swap recipe) for clips longer than that."}),
                "temporal_overlap": ("INT", {"default": 16, "min": 0, "max": 256, "step": 8}),

                "guide_source_id": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 1.0}),
                "identity_source_id": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 1.0}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "STRING")
    RETURN_NAMES = ("images", "latent", "debug")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    # -- internals ----------------------------------------------------------

    def _crop(self, guide, masks, mode, scale, div):
        """Returns (cropped guide, cropped masks, paste-back fn, note)."""
        if mode == "off" or masks is None:
            return guide, masks, None, "crop: off"

        planner = _mvex_planner() if mode in ("tracked", "zoomed") else None
        if planner is not None:
            try:
                p = {"crop_scale": scale, "aspect_ratio": 0.0, "padding": "firm",
                     "prefer": "stillness", "seamless_loop": False,
                     "pad_surplus_tol": 16, "zoom_step": 1.0}
                out = planner(guide, masks, mode, p, div, 0.1, 0.0)
                cropped, cropped_masks, bboxes = out[0], out[1], out[2]
                return cropped, cropped_masks, ("mvex", bboxes), f"crop: mvex/{mode}"
            except Exception as exc:
                log.warning("MaskVidExperiments planner failed (%s); using the static box", exc)

        x0, y0, w, h = _static_box(masks, guide, scale, div)
        cropped = guide[:, y0:y0 + h, x0:x0 + w, :]
        cropped_masks = masks[:, y0:y0 + h, x0:x0 + w]
        return cropped, cropped_masks, ("static", (x0, y0, w, h)), f"crop: static {w}x{h} @({x0},{y0})"

    def _paste_back(self, result, original, ctx, feather):
        if ctx is None:
            return result
        kind, box = ctx
        if kind == "mvex":
            try:
                import importlib
                mod = importlib.import_module("_bfs_mvex_crop")
                return mod.MVEx_SubjectUncropNode.execute(result, original, box, feather).result[0]
            except Exception as exc:
                log.warning("MaskVidExperiments uncrop failed (%s); pasting the static way", exc)
                return result
        x0, y0, w, h = box
        out = original.clone()[: result.shape[0]]
        patch = result[: out.shape[0]]
        if feather > 0:
            ramp = torch.ones(h, w, device=patch.device)
            f = min(feather, h // 2, w // 2)
            if f > 0:
                edge = torch.linspace(0, 1, f, device=patch.device)
                ramp[:f, :] *= edge[:, None]
                ramp[-f:, :] *= edge.flip(0)[:, None]
                ramp[:, :f] *= edge[None, :]
                ramp[:, -f:] *= edge.flip(0)[None, :]
            a = ramp[None, :, :, None]
        else:
            a = 1.0
        out[:, y0:y0 + h, x0:x0 + w, :] = (
            a * patch + (1 - a) * out[:, y0:y0 + h, x0:x0 + w, :]
        )
        return out

    # -- entry point --------------------------------------------------------

    def execute(self, model, vae, noise, sampler, sigmas, guider, positive, negative,
                guide_video, identity_image, subject_mask=None,
                crop_mode="off", crop_scale=1.5, crop_divisible_by=32, uncrop_feather=16,
                inpaint_with_mask=True, mask_grow=8, mask_blur=4,
                temporal_tile_size=0, temporal_overlap=16,
                guide_source_id=1.0, identity_source_id=2.0, debug_log=False):
        from .ltx_multiple_controls import LTXMultipleControls
        from .ltxv_editanything import LTXVEditAnythingLoopingSampler as _Loop

        notes = []
        masks = subject_mask
        if masks is not None:
            if masks.ndim == 4:
                masks = masks.squeeze(-1)
            if mask_grow or mask_blur:
                masks = _grow_blur(masks, mask_grow, mask_blur)
                notes.append(f"mask: grow {mask_grow}px, blur {mask_blur}px")

        guide, masks, crop_ctx, note = self._crop(
            guide_video, masks, crop_mode, crop_scale, crop_divisible_by)
        notes.append(note)

        n_frames = guide.shape[0]
        tile = temporal_tile_size if 0 < temporal_tile_size < n_frames else n_frames
        overlap = min(temporal_overlap, max(0, tile - 8)) if tile < n_frames else 0
        stride = max(1, tile - overlap)
        notes.append(f"frames {n_frames}, tile {tile}, overlap {overlap}")

        _, w_sf, h_sf = vae.downscale_index_formula
        lat_h, lat_w = guide.shape[1] // h_sf, guide.shape[2] // w_sf

        chunks, pos = [], 0
        while pos < n_frames:
            end = min(n_frames, pos + tile)
            chunks.append((pos, end))
            if end >= n_frames:
                break
            pos += stride

        mc = LTXMultipleControls()
        out_latents = []
        for idx, (a, b) in enumerate(chunks):
            g = guide[a:b]
            lat_t = (g.shape[0] - 1) // vae.downscale_index_formula[0] + 1
            empty = {"samples": torch.zeros(
                [1, 128, lat_t, lat_h, lat_w],
                device=comfy.model_management.intermediate_device())}

            m, p, n, latent, _dbg = mc.apply(
                model, positive, negative, vae, empty,
                guide_video=g, guide_source_id=guide_source_id,
                identity_image=identity_image, identity_source_id=identity_source_id,
                auto_mask_guide=False, debug_log=debug_log,
            )

            if inpaint_with_mask and masks is not None:
                latent = dict(latent)
                latent["noise_mask"] = _mask_to_latent(
                    masks[a:b], vae, latent["samples"].shape[2], lat_h, lat_w
                ).to(latent["samples"].device)

            # CRITICAL: _sample_chunk samples through guider.model_patcher, so the
            # patched clone from LTXMultipleControls has to replace it. Without the
            # swap the reference specs live in transformer_options the forward never
            # reads, and guide + identity are silently inert -- it samples happily
            # and ignores both. _set_guider_conds does the conds and the swap.
            gd = _Loop._set_guider_conds(guider, p, n, model_patcher=m)
            chunk = _Loop._sample_chunk(m, noise, sampler, sigmas, gd, latent, seed_offset=idx)
            out_latents.append(chunk["samples"])

        samples = out_latents[0] if len(out_latents) == 1 else torch.cat(out_latents, dim=2)
        images = vae.decode(samples)  # the tensor, not a latent dict
        if isinstance(images, dict):
            images = images.get("samples")
        if images.ndim == 5:  # (B,T,H,W,C) -> frames batch
            images = images.reshape(-1, *images.shape[2:])

        final = self._paste_back(images, guide_video, crop_ctx, uncrop_feather)
        debug = " | ".join(notes)
        if debug_log:
            print("[BFS Head Swap Masked Sampler]", debug)
        return (final, {"samples": samples}, debug)


NODE_CLASS_MAPPINGS = {"BFSHeadSwapMaskedSampler": BFSHeadSwapMaskedSampler}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BFSHeadSwapMaskedSampler": "BFS Head Swap Sampler (crop · mask · loop)"
}
