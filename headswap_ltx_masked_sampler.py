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

The crop planner is vendored from drozbay's MaskVidExperiments (GPL-3.0, same as
this pack) in `bfs_subject_crop.py`, so nothing here depends on that pack being
installed. Its boxes hold still through mask noise and occlusion, which naive
per-frame crops do not, and a jittering crop reads to a video model as camera
motion.
"""

import logging

import torch

import comfy.model_management
import comfy.utils

log = logging.getLogger("BFS.HeadSwapMasked")

CATEGORY = "BFS/video"


# ─────────────────────────────────────────────────────────────────────────────
# crop planning (vendored from drozbay/MaskVidExperiments, GPL-3.0)
# ─────────────────────────────────────────────────────────────────────────────

def _planner():
    """The vendored crop planner (bfs_subject_crop), or None if it fails to import."""
    try:
        from .bfs_subject_crop import _plan_and_crop
        return _plan_and_crop
    except Exception as exc:  # pragma: no cover
        log.warning("vendored crop planner unavailable (%s); using the static box", exc)
        return None


def _feather_ramp(h, w, feather, device, touches=(False, False, False, False)):
    """Blend weights fading in from the crop border, as (1,h,w,1).

    ``touches`` marks the (top, bottom, left, right) sides that sit on the image
    edge. Those are NOT feathered: there is nothing outside to blend into, and
    fading there leaves a visible washed band along the frame border.
    """
    if feather <= 0:
        return torch.ones(1, h, w, 1, device=device)
    ramp = torch.ones(h, w, device=device)
    f = min(int(feather), h // 2, w // 2)
    if f > 0:
        edge = torch.linspace(0, 1, f, device=device)
        top, bottom, left, right = touches
        if not top:
            ramp[:f, :] *= edge[:, None]
        if not bottom:
            ramp[-f:, :] *= edge.flip(0)[:, None]
        if not left:
            ramp[:, :f] *= edge[None, :]
        if not right:
            ramp[:, -f:] *= edge.flip(0)[None, :]
    return ramp[None, :, :, None]


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
    failure drozbay's Mask To Latent Space node was written to fix.
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

def _inject_transformer_options(guider, model_patcher, debug=False):
    """Copy the patched model's transformer_options INTO the guider's own dict.

    CFGGuider captures `self.model_options = model_patcher.model_options` by
    reference when it is constructed (comfy/samplers.py:937), so replacing
    guider.model_patcher afterwards does not propagate: the guider keeps
    sampling with the dict it captured, and the reference specs that
    LTXMultipleControls wrote on the clone never reach the forward. The keys
    have to be mutated into the existing dict in place.
    """
    src = (getattr(model_patcher, "model_options", None) or {}).get("transformer_options", {})
    if not src:
        return []
    target = None
    if isinstance(getattr(guider, "model_options", None), dict):
        target = guider.model_options
    else:
        mp = getattr(guider, "model_patcher", None)
        if mp is not None and isinstance(getattr(mp, "model_options", None), dict):
            target = mp.model_options
    if target is None:
        return []
    to = target.setdefault("transformer_options", {})
    for k, v in src.items():
        to[k] = v
    if debug:
        print(f"[BFS HeadSwap] injected into guider: {sorted(src.keys())}")
    return sorted(src.keys())


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
                "latent": ("LATENT", {"tooltip":
                    "Empty latent from EmptyLTXVLatentVideo, sized to the CROP when cropping is on. "
                    "Strongly recommended: LTX-2.5 latents are AV (video+audio) and this node cannot "
                    "fabricate that structure -- without it a plain video latent is built and the model "
                    "may ignore the guide entirely."}),
                "subject_mask": ("MASK", {"tooltip":
                    "Per-frame mask of the region to edit (head, with margin). Drives the crop box and, "
                    "with inpaint_with_mask on, restricts denoising to it. Leave unconnected for a plain swap."}),

                "crop_mode": (["off", "combined", "tracked", "zoomed"], {"default": "off", "tooltip":
                    "Sample inside a box around the subject instead of the whole frame -- the fix for faces "
                    "that are too small to carry identity. combined: one static box for the clip. "
                    "tracked: a constant-size box that stays still until the subject would leave it. "
                    "zoomed: the box follows the subject's size too, planned over the whole clip."}),
                "crop_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip":
                    "Box size as a multiple of the subject. 1.5 leaves a third as margin. Keep neck and "
                    "shoulders in: a face-tight crop is a framing the LoRA never saw in training."}),
                "crop_divisible_by": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
                "uncrop_feather": ("INT", {"default": 16, "min": 0, "max": 256, "tooltip":
                    "Blend width when pasting the crop back, in pixels. Sides sitting on the image "
                    "edge are never feathered -- there is nothing outside to blend into."}),
                "paste_back": ("BOOLEAN", {"default": True, "tooltip":
                    "Composite the crop into the original frames before returning. Turn OFF for a "
                    "second pass: images/latent then stay in the crop's own space, so you can "
                    "upscale and refine the crop -- where the face actually has pixels -- and "
                    "composite at the end with the Head Swap Paste Back node, feeding it the "
                    "crop_bboxes output."}),
                "paste_confine_to_mask": ("BOOLEAN", {"default": True, "tooltip":
                    "Paste only inside the mask instead of the whole crop rectangle, so anything the "
                    "model changed in the crop's background never reaches the frame. Off pastes the "
                    "full box."}),

                "inpaint_with_mask": ("BOOLEAN", {"default": True, "tooltip":
                    "Send the mask to the sampler as a denoise mask, so only the masked region changes and "
                    "everything else stays the guide's own pixels. Native ComfyUI inpainting -- the LoRA "
                    "never sees the mask and the original face stays visible to the model."}),
                "mask_grow": ("INT", {"default": 8, "min": 0, "max": 256, "tooltip":
                    "Dilate the mask before use, in pixels. The new head can be bigger than the old one."}),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 256, "tooltip":
                    "Soften the mask edge, in pixels, to avoid a hard seam."}),
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                    "How completely the masked region is replaced. 1.0 = fully regenerated. Below that "
                    "the original latent is blended back, which keeps the original geometry and "
                    "expression in pixels -- and drags the original identity back with them, so the "
                    "result becomes an average of both faces. Expression does NOT need this: it "
                    "reaches the model through the aligned guide, which carries the whole source "
                    "performance regardless of the mask."}),

                "temporal_tile_size": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 8, "tooltip":
                    "Frames per chunk. 0 samples the whole clip in one pass. Use the length the LoRA trained "
                    "at (73 for the LTX head-swap recipe) for clips longer than that."}),
                "temporal_overlap": ("INT", {"default": 16, "min": 0, "max": 256, "step": 8}),

                "guide_source_id": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 1.0}),
                "identity_source_id": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 1.0}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "LATENT", "BOUNDING_BOX", "STRING")
    RETURN_NAMES = ("images", "mask_over_source", "cropped_guide", "crop_mask",
                    "latent_mask", "latent", "crop_bboxes", "debug")
    OUTPUT_TOOLTIPS = (
        "Final frames, with the crop pasted back when cropping is on.",
        "The mask painted over the ORIGINAL frames, plus the crop box outline. The "
        "quickest way to see whether the mask is where you think it is, at the scale "
        "you think it is, on the frames it belongs to.",
        "Exactly what the model was fed as the guide: cropped and resized. If this "
        "does not look like the region you meant to edit, nothing downstream will.",
        "The mask after grow/blur (and cropping), in the crop's pixel space.",
        "The mask as the sampler actually sees it: reduced to the latent grid with "
        "max and upsampled back for viewing. One frame per latent frame -- this is "
        "the real resolution of the edit, and where a too-thin mask disappears.",
        "Sampled latent. With cropping on this is the CROP's latent, which is what you "
        "want to upscale and refine in a second pass.",
        "The crop boxes, one per frame. Feed to Head Swap Paste Back after a second pass.",
        "What the node decided: crop mode and box, mask ops, tiling.",
    )
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    # -- internals ----------------------------------------------------------

    def _crop(self, guide, masks, mode, scale, div):
        """Returns (cropped guide, cropped masks, paste-back fn, note)."""
        if mode == "off" or masks is None:
            return guide, masks, None, "crop: off"

        planner = _planner() if mode in ("tracked", "zoomed") else None
        if planner is not None:
            try:
                p = {"crop_scale": scale, "aspect_ratio": 0.0, "padding": "firm",
                     "prefer": "stillness", "seamless_loop": False,
                     "pad_surplus_tol": 16, "zoom_step": 1.0}
                out = planner(guide, masks, mode, p, div, 0.1, 0.0)
                cropped, cropped_masks, bboxes = out[0], out[1], out[2]
                return cropped, cropped_masks, ("planned", bboxes), f"crop: planned/{mode}"
            except Exception as exc:
                log.warning("crop planner failed (%s); using the static box", exc)

        x0, y0, w, h = _static_box(masks, guide, scale, div)
        cropped = guide[:, y0:y0 + h, x0:x0 + w, :]
        cropped_masks = masks[:, y0:y0 + h, x0:x0 + w]
        return cropped, cropped_masks, ("static", (x0, y0, w, h)), f"crop: static {w}x{h} @({x0},{y0})"

    def _paste_back(self, result, original, ctx, feather, confine=None):
        if ctx is None:
            return result
        kind, box = ctx
        if kind == "planned":
            # one box per frame: paste each crop into its own box
            out = original.clone()
            n = min(result.shape[0], out.shape[0], len(box))
            for i in range(n):
                b = box[i][0] if isinstance(box[i], list) else box[i]
                if isinstance(b, dict):  # planner boxes: {"x","y","width","height"}
                    x0, y0, w, h = int(b["x"]), int(b["y"]), int(b["width"]), int(b["height"])
                else:                    # a plain (x, y, w, h) sequence
                    x0, y0, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                patch = result[i:i + 1]
                if patch.shape[1] != h or patch.shape[2] != w:
                    patch = comfy.utils.common_upscale(
                        patch.movedim(-1, 1), w, h, "lanczos", "disabled").movedim(1, -1)
                H, W = out.shape[1], out.shape[2]
                a = _feather_ramp(h, w, feather, patch.device,
                                  (y0 <= 0, y0 + h >= H, x0 <= 0, x0 + w >= W))
                if confine is not None:
                    cm = confine[min(i, confine.shape[0] - 1)]
                    if cm.shape != (h, w):
                        cm = torch.nn.functional.interpolate(
                            cm[None, None], size=(h, w), mode="bilinear")[0, 0]
                    a = a * cm[None, :, :, None].to(a.device)
                out[i:i + 1, y0:y0 + h, x0:x0 + w, :] = (
                    a * patch + (1 - a) * out[i:i + 1, y0:y0 + h, x0:x0 + w, :])
            return out
        x0, y0, w, h = box
        out = original.clone()[: result.shape[0]]
        patch = result[: out.shape[0]]
        # The sampled crop does not have to match the box: the connected latent
        # sets the sampled size, and the zoomed crop mode rescales
        # crops by design. Bring it back to the box before blending.
        if patch.shape[1] != h or patch.shape[2] != w:
            log.info("paste-back: resizing the sampled crop %dx%d -> box %dx%d",
                     patch.shape[2], patch.shape[1], w, h)
            patch = comfy.utils.common_upscale(
                patch.movedim(-1, 1), w, h, "lanczos", "disabled").movedim(1, -1)
        H, W = out.shape[1], out.shape[2]
        a = _feather_ramp(h, w, feather, patch.device,
                          (y0 <= 0, y0 + h >= H, x0 <= 0, x0 + w >= W))
        if confine is not None:
            cm = confine[:1] if confine.shape[0] == 1 else confine[: out.shape[0]]
            if cm.shape[-2:] != (h, w):
                cm = torch.nn.functional.interpolate(
                    cm.unsqueeze(1), size=(h, w), mode="bilinear").squeeze(1)
            a = a * cm.unsqueeze(-1).to(a.device)
        out[:, y0:y0 + h, x0:x0 + w, :] = (
            a * patch + (1 - a) * out[:, y0:y0 + h, x0:x0 + w, :]
        )
        return out

    # -- entry point --------------------------------------------------------

    def execute(self, model, vae, noise, sampler, sigmas, guider, positive, negative,
                guide_video, identity_image, latent=None, subject_mask=None,
                crop_mode="off", crop_scale=1.5, crop_divisible_by=32, uncrop_feather=16,
                inpaint_with_mask=True, mask_grow=8, mask_blur=4, mask_strength=1.0,
                paste_back=True, paste_confine_to_mask=True,
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
                notes.append(f"mask: grow {mask_grow}px, blur {mask_blur}px, "
                             f"strength {mask_strength}")

        masks_full = masks
        guide, masks, crop_ctx, note = self._crop(
            guide_video, masks, crop_mode, crop_scale, crop_divisible_by)
        notes.append(note)

        n_frames = guide.shape[0]
        tile = temporal_tile_size if 0 < temporal_tile_size < n_frames else n_frames
        overlap = min(temporal_overlap, max(0, tile - 8)) if tile < n_frames else 0
        stride = max(1, tile - overlap)
        notes.append(f"frames {n_frames}, tile {tile}, overlap {overlap}")
        notes.append(f"sample size {guide.shape[2]}x{guide.shape[1]} "
                     f"(connect EmptyLTXVLatentVideo at this size to avoid a resize)")

        _, w_sf, h_sf = vae.downscale_index_formula
        lat_h, lat_w = guide.shape[1] // h_sf, guide.shape[2] // w_sf

        lat_t_total = (n_frames - 1) // vae.downscale_index_formula[0] + 1
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
            if latent is not None:
                empty = dict(latent)
                sm = empty["samples"]
                if getattr(sm, "is_nested", False):
                    if len(chunks) > 1:
                        raise ValueError(
                            "chunked sampling with an AV (nested) latent is not supported yet: "
                            "set temporal_tile_size to 0, or feed a video-only latent")
                elif sm.shape[2] != lat_t:
                    empty["samples"] = sm[:, :, :lat_t]
            else:
                # last resort: a plain video latent. On LTX-2.5 the real thing is an
                # AV (video+audio) latent, so connect EmptyLTXVLatentVideo instead.
                log.warning("no latent connected: building a plain video latent, "
                            "which is wrong for AV models like LTX-2.5")
                empty = {"samples": torch.zeros(
                    [1, 128, lat_t, lat_h, lat_w],
                    device=comfy.model_management.intermediate_device())}
            if debug_log:
                _s = empty["samples"]
                print(f"[BFS HeadSwap] chunk {idx}: guide {tuple(g.shape)} -> latent "
                      f"{'nested AV' if getattr(_s,'is_nested',False) else tuple(_s.shape)}")

            m, p, n, latent, _dbg = mc.apply(
                model, positive, negative, vae, empty,
                guide_video=g, guide_source_id=guide_source_id,
                identity_image=identity_image, identity_source_id=identity_source_id,
                auto_mask_guide=False, debug_log=debug_log,
            )

            if inpaint_with_mask and masks is not None:
                latent = dict(latent)
                # Inpainting keeps the INITIAL latent outside the mask, so it has to be
                # the guide -- not the empty latent, which would leave grey where the
                # video should be. This is the encode a normal inpaint graph does before
                # Set Latent Noise Mask.
                g_lat = vae.encode(g)
                base = latent["samples"]
                if getattr(base, "is_nested", False):
                    # keep the audio stream from the connected AV latent, swap the video
                    import comfy.nested_tensor
                    streams = list(base.tensors) if hasattr(base, "tensors") else list(base.unbind())
                    streams[0] = g_lat.to(streams[0].device, streams[0].dtype)
                    latent["samples"] = comfy.nested_tensor.NestedTensor(tuple(streams))
                else:
                    latent["samples"] = g_lat.to(base.device, base.dtype)
                if debug_log:
                    print(f"[BFS HeadSwap] inpaint: latent seeded from the guide "
                          f"{tuple(g_lat.shape)}")
                nm = _mask_to_latent(
                    masks[a:b], vae, latent["samples"].shape[2], lat_h, lat_w)
                if mask_strength < 1.0:
                    nm = nm * float(mask_strength)
                latent["noise_mask"] = nm.to(latent["samples"].device)

            # CRITICAL: _sample_chunk samples through guider.model_patcher, so the
            # patched clone from LTXMultipleControls has to replace it. Without the
            # swap the reference specs live in transformer_options the forward never
            # reads, and guide + identity are silently inert -- it samples happily
            # and ignores both. _set_guider_conds does the conds and the swap.
            gd = _Loop._set_guider_conds(guider, p, n, model_patcher=m)
            _inject_transformer_options(gd, m, debug=debug_log)
            chunk = _Loop._sample_chunk(m, noise, sampler, sigmas, gd, latent, seed_offset=idx)
            out_latents.append(chunk["samples"])

        samples = out_latents[0] if len(out_latents) == 1 else torch.cat(out_latents, dim=2)
        video = samples
        if getattr(video, "is_nested", False):
            # AV latent: the video VAE only takes the video stream (this is what
            # LTXVSeparateAVLatent does in the stock graphs)
            video = video.tensors[0] if hasattr(video, "tensors") else video.unbind()[0]
        images = vae.decode(video)  # the tensor, not a latent dict
        if isinstance(images, dict):
            images = images.get("samples")
        if images.ndim == 5:  # (B,T,H,W,C) -> frames batch
            images = images.reshape(-1, *images.shape[2:])

        confine = masks if (paste_confine_to_mask and masks is not None) else None
        if paste_back:
            final = self._paste_back(images, guide_video, crop_ctx, uncrop_feather, confine)
        else:
            final = images
            notes.append("paste_back off: images/latent stay in crop space")

        # --- inspection outputs ------------------------------------------
        h_px, w_px = guide.shape[1], guide.shape[2]
        if masks is not None:
            crop_mask_out = masks
            lat_mask = _mask_to_latent(masks, vae, lat_t_total, lat_h, lat_w)[0, 0]
            latent_mask_out = torch.nn.functional.interpolate(
                lat_mask.unsqueeze(1), size=(h_px, w_px), mode="nearest"
            ).squeeze(1)
        else:
            crop_mask_out = torch.zeros(1, h_px, w_px)
            latent_mask_out = torch.zeros(1, h_px, w_px)

        overlay = guide_video.clone()
        if masks_full is not None:
            n = min(overlay.shape[0], masks_full.shape[0])
            a = masks_full[:n].unsqueeze(-1).clamp(0, 1) * 0.45
            tint = torch.tensor([1.0, 0.15, 0.15], device=overlay.device)
            overlay[:n] = overlay[:n] * (1 - a) + tint * a
        if crop_ctx is not None and crop_ctx[0] == "planned":
            g = torch.tensor([0.1, 1.0, 0.2], device=overlay.device)
            t = 2
            for i in range(min(overlay.shape[0], len(crop_ctx[1]))):
                b = crop_ctx[1][i][0] if isinstance(crop_ctx[1][i], list) else crop_ctx[1][i]
                x0, y0, w, h = int(b["x"]), int(b["y"]), int(b["width"]), int(b["height"])
                overlay[i, y0:y0 + t, x0:x0 + w] = g
                overlay[i, y0 + h - t:y0 + h, x0:x0 + w] = g
                overlay[i, y0:y0 + h, x0:x0 + t] = g
                overlay[i, y0:y0 + h, x0 + w - t:x0 + w] = g
        elif crop_ctx is not None and crop_ctx[0] == "static":
            x0, y0, w, h = crop_ctx[1]
            g = torch.tensor([0.1, 1.0, 0.2], device=overlay.device)
            t = 2
            overlay[:, y0:y0 + t, x0:x0 + w] = g
            overlay[:, y0 + h - t:y0 + h, x0:x0 + w] = g
            overlay[:, y0:y0 + h, x0:x0 + t] = g
            overlay[:, y0:y0 + h, x0 + w - t:x0 + w] = g

        debug = " | ".join(notes)
        if debug_log:
            print("[BFS Head Swap Masked Sampler]", debug)
        boxes_out = crop_ctx[1] if crop_ctx is not None else []
        if crop_ctx is not None and crop_ctx[0] == "static":
            x0, y0, w, h = crop_ctx[1]
            boxes_out = [[{"x": x0, "y": y0, "width": w, "height": h}]] * guide_video.shape[0]
        return (final, overlay, guide, crop_mask_out, latent_mask_out,
                {"samples": samples}, boxes_out, debug)


class BFSHeadSwapPasteBack:
    """Composite processed crops back into the original frames.

    The other half of `paste_back: off`: run the sampler in crop space, upscale
    and refine there, then bring the result home. Sides of a crop that sit on the
    image edge are not feathered, and the paste can be confined to a mask so only
    the subject travels back.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cropped_images": ("IMAGE", {"tooltip": "Processed crops, at any resolution."}),
                "original_images": ("IMAGE", {"tooltip": "The frames to paste into."}),
                "crop_bboxes": ("BOUNDING_BOX", {"tooltip": "From the sampler's crop_bboxes output."}),
                "feather": ("INT", {"default": 16, "min": 0, "max": 256}),
            },
            "optional": {
                "confine_mask": ("MASK", {"tooltip":
                    "Confines the paste to the mask instead of the whole box, in the crop's space."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, cropped_images, original_images, crop_bboxes, feather, confine_mask=None):
        return (BFSHeadSwapMaskedSampler()._paste_back(
            cropped_images, original_images, ("planned", crop_bboxes), feather, confine_mask),)


NODE_CLASS_MAPPINGS = {
    "BFSHeadSwapMaskedSampler": BFSHeadSwapMaskedSampler,
    "BFSHeadSwapPasteBack": BFSHeadSwapPasteBack,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BFSHeadSwapMaskedSampler": "BFS Head Swap Sampler (crop · mask · loop)",
    "BFSHeadSwapPasteBack": "BFS Head Swap Paste Back",
}
