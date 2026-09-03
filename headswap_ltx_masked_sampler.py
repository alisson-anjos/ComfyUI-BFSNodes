"""BFS Head Swap Sampler (LTX) — crop, mask and temporal loop in one node.

Everything is optional, so the node degrades to whatever you connect:

    guide_video                                   -> the LoRA over the whole clip, one pass
    + identity_image                              -> for a LoRA that takes a reference
                                                     (head swap, identity transfer)
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


def _subject_stats(masks, thresh=0.1):
    """Measure the subject in the mask: size, travel, size change, per-frame step.

    Everything the auto config decides is derived from these, because they are
    the only numbers that make a pixel amount meaningful. ``mask_grow: 8`` is
    generous on a 90 px head and invisible on a 900 px one; six percent of the
    head's width is the same amount of slack in both.
    """
    m = masks if masks.ndim == 3 else masks.squeeze(-1)
    n, H, W = m.shape[0], m.shape[-2], m.shape[-1]
    hit = m > thresh
    boxes = []
    for i in range(n):
        ys, xs = torch.where(hit[i])
        if ys.numel() == 0:
            boxes.append(None)
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    seen = [b for b in boxes if b is not None]
    if not seen:
        return None

    def pct(vals, q):
        v = sorted(vals)
        return float(v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))])

    ws = [b[2] for b in seen]
    hs = [b[3] for b in seen]
    head_w, head_h = pct(ws, 0.5), pct(hs, 0.5)          # median: one bad frame cannot move it
    cx = [b[0] + b[2] / 2.0 for b in seen]
    cy = [b[1] + b[3] / 2.0 for b in seen]
    travel = max(max(cx) - min(cx), max(cy) - min(cy))
    size_var = (pct(ws, 0.9) - pct(ws, 0.1)) / max(1.0, head_w)
    step, prev = 0.0, None
    for b in boxes:
        if b is None:
            prev = None
            continue
        c = (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)
        if prev is not None:
            step = max(step, abs(c[0] - prev[0]), abs(c[1] - prev[1]))
        prev = c
    return {"head_w": head_w, "head_h": head_h, "travel": travel, "size_var": size_var,
            "step": step, "seen": len(seen), "frames": n, "H": H, "W": W}


def _auto_config(st, cell, ref_aspect=None, headroom=1.15):
    """Turn the subject stats into every pixel-space knob, in one place.

    Two rules behind all of it.

    **A pixel amount only means something relative to the head it acts on.** The
    same 8 px of grow is generous on a 90 px head and invisible on a 900 px one,
    so every amount here is a fraction of the measured head.

    **The mask is the OLD head; the new one does not have to fit inside it.** A
    wider face, more hair, a taller cut -- if the mask does not cover where the
    new head lands, the swap is clipped at the mask edge, which is the seam this
    mode exists to remove. Absolute size cannot be recovered from a cropped
    reference (a head crop carries no scale), but the reference's *proportions*
    can: ``ref_aspect`` against the mask's own aspect says whether the new head
    is relatively wider or taller, and ``headroom`` covers the part that cannot
    be measured. Growth is anisotropic on purpose -- hair overflows upward and
    sideways, never down into the neck and collar, which must stay the guide's.

    The margin between the mask and the crop border is then a budget spent in
    order: the grow takes what it needs, the paste feather gets half of what is
    left. A ramp wider than its share fades the head itself.
    """
    hw, hh, W, H = st["head_w"], st["head_h"], st["W"], st["H"]
    occ = hw / max(1.0, W)
    fit = min(W / max(1.0, hw), H / max(1.0, hh))       # largest box the frame still holds

    # ── how much bigger the new head may be, per axis ────────────────────────
    src_aspect = hw / max(1.0, hh)
    ref = float(ref_aspect) if ref_aspect else src_aspect
    headroom = max(1.0, float(headroom))
    # headroom carries the MAGNITUDE -- it is the part no measurement can give,
    # since a cropped reference head has no scale. The aspect only tilts how
    # that slack is split between sideways and up: a relatively wider reference
    # needs it at the sides, a taller one above.
    tilt = min(1.6, max(0.6, ref / src_aspect))
    delta = headroom - 1.0

    # every slack is a fraction of the head's WIDTH, capped. Width is the stable
    # dimension of a head mask -- height swings with how much neck the mask took,
    # and scaling the upward slack by it once produced holes far larger than the
    # head, with the new head floating inside the regenerated area.
    base = 0.06 * hw
    want_x = base + min(0.25 * hw, 0.5 * hw * delta * tilt)
    want_up = 1.5 * base + min(0.30 * hw, hw * delta / tilt)  # hair goes up
    want_down = base                                     # never eat into the neck

    # ── the box has to be big enough to hold all of that plus a ramp ─────────
    need_x = 1.0 + 2.0 * (want_x + 4.0) / hw
    need_y = 1.0 + 2.0 * (max(want_up, want_down) + 4.0) / hh
    scale = round(min(fit, max(1.2, 1.8, need_x, need_y)), 2)

    if occ >= 0.45 or fit < 1.15:
        mode, why = "off", f"head is {occ:.0%} of the frame width, a crop would gain nothing"
    elif st["size_var"] > 0.35:
        mode, why = "zoomed", f"the subject's size changes {st['size_var']:.0%} across the clip"
    elif st["travel"] > 0.25 * hw * scale:
        mode, why = "tracked", f"the subject travels {st['travel']:.0f}px, more than a quarter of the box"
    else:
        mode, why = "combined", f"the subject travels {st['travel']:.0f}px, it fits one static box"

    # ── spend the margin: grow first, then what is left is the ramp's ────────
    margin_x = hw * (scale - 1.0) / 2.0
    margin_y = hh * (scale - 1.0) / 2.0
    grow_x = int(min(96, max(2, int(min(want_x, 0.7 * margin_x)))))
    grow_up = int(min(160, max(2, int(min(want_up, 0.7 * margin_y)))))
    grow_down = int(min(96, max(2, int(min(want_down, 0.7 * margin_y)))))
    left_x = margin_x - grow_x
    left_y = margin_y - max(grow_up, grow_down)
    # floor, never round: a cap that rounds up is a ramp that reaches past the margin
    feather = int(min(64, max(4, int(0.5 * min(left_x, left_y)))))
    blur = int(min(48, max(2, round(0.03 * hw))))       # paste-back softness, not denoise
    # NOT derived from the grow: the pixel grow above already carries the slack,
    # and _mask_to_latent reduces with MAX, so any cell the mask touches is
    # already fully editable. Dilating here on top of that added a whole 32 px
    # cell on every side for nothing.
    cells = 0
    frames = 1 if st["step"] > 0.15 * hw else 0         # a fast head needs slack along time

    ref_note = (f"reference aspect {ref:.2f} vs mask {src_aspect:.2f} (tilt {tilt:.2f}), "
                f"headroom {headroom:.2f}")
    note = (f"auto: head {hw:.0f}x{hh:.0f}px in {W}x{H} ({occ:.0%} of the width), "
            f"seen in {st['seen']}/{st['frames']} frames; {ref_note} -> "
            f"crop {mode} @ {scale} ({why}); grow {grow_x}px sideways, {grow_up}px up, "
            f"{grow_down}px down, blur {blur}px, feather {feather}px, "
            f"latent dilate {cells} cell(s), {frames} frame(s)")
    return {"crop_mode": mode, "crop_scale": scale,
            "mask_grow": (grow_up, grow_down, grow_x, grow_x), "mask_blur": blur,
            "uncrop_feather": feather, "latent_mask_dilate": cells,
            "latent_mask_dilate_frames": frames, "note": note}


# ─────────────────────────────────────────────────────────────────────────────
# mask helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grow_blur(masks, grow, blur):
    """Dilate then soften a mask stack, in pixels.

    ``grow`` is either one amount for every side, or ``(up, down, left, right)``.
    The sides matter: a new head overflows the old one upward and sideways --
    hair, a wider face -- while growing downward only eats into the neck and
    collar, which have to stay the guide's own pixels.
    """
    F = torch.nn.functional
    m = masks.unsqueeze(1).float()  # (N,1,H,W)
    if isinstance(grow, (tuple, list)):
        up, down, left, right = (max(0, int(g)) for g in grow)
        if up or down or left or right:
            # pad by what each side grows, then pool the padding away. The pad
            # is mirrored on purpose: padding the TOP shifts the window down, so
            # the top pad is what the mask grows DOWNWARD by.
            m = F.pad(m, (right, left, down, up), mode="replicate")
            if up or down:
                m = F.max_pool2d(m, (up + down + 1, 1), stride=1)
            if left or right:
                m = F.max_pool2d(m, (1, left + right + 1), stride=1)
    elif grow > 0:
        k = 2 * int(grow) + 1
        m = F.max_pool2d(m, k, stride=1, padding=k // 2)
    if blur > 0:
        k = 2 * int(blur) + 1
        m = F.avg_pool2d(m, k, stride=1, padding=k // 2, count_include_pad=False)
    return m.squeeze(1).clamp(0, 1)


def _dilate_latent(mask, cells, frames=0):
    """Grow a latent-grid mask by whole cells (and latent frames).

    Growing here is not the same as growing in pixels: the latent grid is coarse
    (one cell per 32 px, one frame per 8), so this is the knob that guarantees the
    head lands inside editable blocks instead of clipping at a cell boundary.
    """
    if cells > 0:
        k = 2 * int(cells) + 1
        m = mask.squeeze(0)                      # (1,t,h,w)
        m = torch.nn.functional.max_pool3d(m.unsqueeze(0), (1, k, k), stride=1,
                                           padding=(0, k // 2, k // 2))
        mask = m
    if frames > 0:
        k = 2 * int(frames) + 1
        mask = torch.nn.functional.max_pool3d(mask, (k, 1, 1), stride=1,
                                              padding=(k // 2, 0, 0))
    return mask


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

def _keep_new_frames(samples, produced, total):
    """Drop the head a chunk shares with the one before it.

    Chunks are sampled with an overlap for continuity, so consecutive chunks
    cover some of the same frames. Concatenating them whole makes the clip
    longer than it is and every frame after the first seam lands on the wrong
    moment -- and, with per-frame crop boxes, in the wrong box. Keeping only
    what each chunk adds makes the concatenation exactly `total` frames.
    """
    have = samples.shape[2]
    new = max(0, min(have, total - produced))
    if new < have:
        samples = samples[:, :, have - new:]
    return samples, produced + samples.shape[2]


def _boxes_per_frame(crop_ctx, n_frames):
    """One box per frame, whatever the crop mode produced.

    The planner already returns per-frame boxes; a static box is repeated so the
    paste-back sees the same shape either way and never has to branch.
    """
    if crop_ctx is None:
        return []
    if crop_ctx[0] == "static":
        x0, y0, w, h = crop_ctx[1]
        return [[{"x": x0, "y": y0, "width": w, "height": h}]] * n_frames
    return crop_ctx[1]


def _fit_latent_to_crop(samples, lat_t, lat_h, lat_w):
    """Rebuild the video stream at the crop's latent shape, keeping the audio.

    The crop's size comes from the mask, so it cannot be known when the empty
    latent is built upstream -- and a latent that does not match it is not an
    error anywhere: the model samples at the latent's shape and the paste-back
    squeezes the result into the box. A face crop is portrait; sampled into a
    landscape latent it comes back visibly deformed, and under crop_mode zoomed
    the deformation changes frame to frame.

    Nothing is lost by rebuilding. At denoise 1.0 the initial latent contributes
    nothing (x = sigma*noise + (1-sigma)*latent), and the inpainting path
    overwrites the video stream with the encoded guide anyway. Returns the
    samples and a note, or None when they already fit.
    """
    def _shape_of(v):
        return (int(v.shape[2]), int(v.shape[3]), int(v.shape[4]))

    if getattr(samples, "is_nested", False):
        streams = list(samples.tensors) if hasattr(samples, "tensors") else list(samples.unbind())
        v = streams[0]
        if _shape_of(v) == (lat_t, lat_h, lat_w):
            return None, None
        was = _shape_of(v)
        streams[0] = torch.zeros(v.shape[0], v.shape[1], lat_t, lat_h, lat_w,
                                 device=v.device, dtype=v.dtype)
        from comfy import nested_tensor as _nested
        return _nested.NestedTensor(tuple(streams)), was
    if _shape_of(samples) == (lat_t, lat_h, lat_w):
        return None, None
    was = _shape_of(samples)
    return torch.zeros(samples.shape[0], samples.shape[1], lat_t, lat_h, lat_w,
                       device=samples.device, dtype=samples.dtype), was


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
        return _Injection({}, [], [])
    target = None
    if isinstance(getattr(guider, "model_options", None), dict):
        target = guider.model_options
    else:
        mp = getattr(guider, "model_patcher", None)
        if mp is not None and isinstance(getattr(mp, "model_options", None), dict):
            target = mp.model_options
    if target is None:
        return _Injection({}, [], [])
    to = target.setdefault("transformer_options", {})
    undo = [(k, k in to, to.get(k)) for k in src]
    for k, v in src.items():
        to[k] = v
    if debug:
        print(f"[BFS HeadSwap] injected into guider: {sorted(src.keys())}")
    return _Injection(to, undo, sorted(src.keys()))


class _Injection:
    """What was written into the guider, and how to take it back out.

    The guider is an object ComfyUI hands us from another node, and it survives
    between runs in the execution cache. Writing reference specs into its dict
    and leaving them there does two bad things: the reference LATENTS stay
    reachable, so the VRAM they occupy is never freed while that guider is
    cached, and the next run that does not overwrite the same keys -- a graph
    with no references, an aborted run, another sampler sharing the guider --
    samples with the previous clip's specs, whose shapes no longer fit. Undoing
    the write leaves the guider exactly as it was found.
    """

    def __init__(self, options, undo, keys):
        self.options, self._undo, self.keys = options, undo, keys

    def undo(self):
        for k, had, old in self._undo:
            if had:
                self.options[k] = old
            else:
                self.options.pop(k, None)
        self._undo = []


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
            },
            "optional": {
                "identity_image": ("IMAGE", {"tooltip":
                    "Reference image for the LoRA that wants one — a head crop for a head swap, "
                    "a subject for identity transfer. Leave it unconnected for any IC-LoRA that "
                    "works from the guide alone (an instruction edit, a sharpener, a restyler): "
                    "the slot is simply not packed and the guide is the only reference."}),
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
                    "Composite only inside the mask, so anything the model changed outside it never "
                    "reaches the frame. With a crop this confines the paste to the mask instead of "
                    "the whole box; WITHOUT a crop it is what keeps the untouched pixels the "
                    "source's own — otherwise the whole frame is the VAE's round trip of it, "
                    "softer everywhere the edit never went. Off pastes the full frame or box."}),

                "inpaint_with_mask": ("BOOLEAN", {"default": True, "tooltip":
                    "Send the mask to the sampler as a denoise mask, so only the masked region changes and "
                    "everything else stays the guide's own pixels. Native ComfyUI inpainting -- the LoRA "
                    "never sees the mask and the original face stays visible to the model."}),
                "mask_grow": ("INT", {"default": 8, "min": 0, "max": 256, "tooltip":
                    "Dilate the mask before use, in pixels. The new head can be bigger than the old one."}),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 256, "tooltip":
                    "Soften the mask edge, in pixels, to avoid a hard seam."}),
                "mask_hard_for_inpaint": ("BOOLEAN", {"default": True, "tooltip":
                    "Binarise the mask before it becomes the denoise mask. Blur belongs to the "
                    "paste-back, where a soft edge hides the seam; in the DENOISE mask a soft edge "
                    "means partial denoising, which blends the original latent -- and the original "
                    "identity -- back in exactly at the edge of the head. Off passes the soft mask "
                    "through to the sampler as well."}),
                "latent_mask_dilate": ("INT", {"default": 0, "min": 0, "max": 16, "tooltip":
                    "Grow the mask by whole LATENT cells after the reduction. One cell is 32 px, so "
                    "this is much coarser than mask_grow -- and it is what guarantees the head sits "
                    "inside editable blocks instead of clipping at a cell boundary. Check the "
                    "latent_mask output to see the effect."}),
                "latent_mask_dilate_frames": ("INT", {"default": 0, "min": 0, "max": 8, "tooltip":
                    "Same, along time: grow by whole latent frames (one covers 8 video frames)."}),
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                    "How completely the masked region is replaced. 1.0 = fully regenerated. Below that "
                    "the original latent is blended back, which keeps the original geometry and "
                    "expression in pixels -- and drags the original identity back with them, so the "
                    "result becomes an average of both faces. Expression does NOT need this: it "
                    "reaches the model through the aligned guide, which carries the whole source "
                    "performance regardless of the mask."}),

                "decode": (["full", "tiled", "none"], {"default": "full", "tooltip":
                    "How to turn the sampled latent into frames. full: one shot, fine at the size "
                    "the sampler ran at. tiled: for a big latent (a second pass after a 2x "
                    "upscaler), where a full decode thrashes VRAM and looks like a hang. none: "
                    "skip decoding and return the latent only -- use your own VAE Decode (Tiled) "
                    "downstream. With none there is nothing to paste back, so images comes out empty."}),
                "decode_tile_size": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
                "decode_overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 16}),
                "decode_temporal_size": ("INT", {"default": 32, "min": 4, "max": 4096, "step": 4, "tooltip":
                    "Frames decoded at once in tiled mode."}),
                "decode_temporal_overlap": ("INT", {"default": 4, "min": 0, "max": 256, "step": 1}),

                "temporal_tile_size": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 8, "tooltip":
                    "Frames per chunk. 0 samples the whole clip in one pass. Use the length the LoRA trained "
                    "at (73 for the LTX head-swap recipe) for clips longer than that."}),
                "temporal_overlap": ("INT", {"default": 16, "min": 0, "max": 256, "step": 8}),

                "guide_source_id": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 1.0}),
                "identity_source_id": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 1.0}),
                "debug_log": ("BOOLEAN", {"default": False}),

                # ── appended, never inserted ──────────────────────────────────
                # ComfyUI stores widgets_values positionally, so a widget added
                # anywhere but the end shifts every later value in every saved
                # workflow: crop_mode lands in this slot, crop_scale in the next,
                # and the graph fails validation with "input out of range".
                "auto_config": ("BOOLEAN", {"default": False, "tooltip":
                    "Measure the subject in the mask and set crop mode, crop scale, mask grow "
                    "and blur, paste feather and latent dilation from its size -- ignoring those "
                    "widgets. Needs subject_mask. Every amount below is in pixels, which only "
                    "means something relative to how big the head is in frame: the same 8 px is "
                    "generous on a distant head and invisible on a close one, and that mismatch "
                    "is what leaves a seam. The debug output prints what it chose."}),

                "identity_headroom": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05,
                    "tooltip":
                    "auto_config only. How much bigger the reference head may be than the head in "
                    "the guide. The mask is the OLD head, so a wider face or more hair lands "
                    "outside it and gets clipped -- the seam. The reference's PROPORTIONS are "
                    "measured from identity_image against the mask; its absolute size cannot be, "
                    "because a cropped head carries no scale. Raise it for big hair or a visibly "
                    "larger head; 1.0 assumes the two heads match."}),
            },
        }

    DESCRIPTION = ("Sampler for guide-driven IC-LoRAs, with an optional stable crop around the "
                   "subject, native mask inpainting and temporal chunking. Connect only what you "
                   "need: a guide alone runs the LoRA over the whole clip; add an identity image "
                   "for a LoRA that takes one; add a mask to restrict the edit; add a crop mode to "
                   "sample the subject full-frame when it is too small to carry detail. "
                   "auto_config measures the subject in the mask and derives every pixel amount "
                   "from its size.")
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
            if confine is None:
                return result
            # No crop, but a mask: composite the full frame through it. Without
            # this the whole frame is whatever came out of the VAE, so every
            # pixel the edit never touched still went through encode/decode and
            # came back softer. Confining here keeps them the source's own.
            out = original.clone()[: result.shape[0]]
            patch = result[: out.shape[0]]
            H, W = out.shape[1], out.shape[2]
            if patch.shape[1] != H or patch.shape[2] != W:
                patch = comfy.utils.common_upscale(
                    patch.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
            cm = confine[:1] if confine.shape[0] == 1 else confine[: out.shape[0]]
            if cm.shape[-2:] != (H, W):
                cm = torch.nn.functional.interpolate(
                    cm.unsqueeze(1), size=(H, W), mode="bilinear").squeeze(1)
            a = cm.unsqueeze(-1).to(patch.device, patch.dtype)
            return a * patch + (1 - a) * out
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
                guide_video, identity_image=None, latent=None, subject_mask=None,
                auto_config=False, identity_headroom=1.15,
                crop_mode="off", crop_scale=1.5, crop_divisible_by=32, uncrop_feather=16,
                inpaint_with_mask=True, mask_grow=8, mask_blur=4,
                mask_hard_for_inpaint=True, latent_mask_dilate=0, latent_mask_dilate_frames=0,
                mask_strength=1.0,
                paste_back=True, paste_confine_to_mask=True,
                decode="full", decode_tile_size=768, decode_overlap=64,
                decode_temporal_size=32, decode_temporal_overlap=4,
                temporal_tile_size=0, temporal_overlap=16,
                guide_source_id=1.0, identity_source_id=2.0, debug_log=False):
        from .ltx_multiple_controls import LTXMultipleControls
        from .ltxv_editanything import LTXVEditAnythingLoopingSampler as _Loop

        notes = []
        masks = subject_mask
        if masks is not None:
            if masks.ndim == 4:
                masks = masks.squeeze(-1)
            if auto_config:
                # measure the RAW mask: grow/blur below would inflate the head we measure
                st = _subject_stats(masks)
                if st is None:
                    notes.append("auto: the mask is empty, keeping the widget values")
                else:
                    try:
                        cell = int(vae.downscale_index_formula[1])
                    except Exception:
                        cell = 32
                    ref_aspect = None
                    if identity_image is not None and identity_image.ndim == 4:
                        ref_aspect = float(identity_image.shape[2]) / max(1.0, float(identity_image.shape[1]))
                    auto = _auto_config(st, cell, ref_aspect, identity_headroom)
                    crop_mode = auto["crop_mode"]
                    crop_scale = auto["crop_scale"]
                    mask_grow = auto["mask_grow"]
                    mask_blur = auto["mask_blur"]
                    uncrop_feather = auto["uncrop_feather"]
                    latent_mask_dilate = auto["latent_mask_dilate"]
                    latent_mask_dilate_frames = auto["latent_mask_dilate_frames"]
                    notes.append(auto["note"])
            if mask_grow or mask_blur:
                masks = _grow_blur(masks, mask_grow, mask_blur)
                g = (f"{mask_grow[0]}/{mask_grow[1]}/{mask_grow[2]}px up/down/side"
                     if isinstance(mask_grow, (tuple, list)) else f"{mask_grow}px")
                notes.append(f"mask: grow {g}, blur {mask_blur}px, strength {mask_strength}")

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
                     f"(the latent is rebuilt to this if it does not match)")

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
        injections = []
        produced = 0
        try:
            for idx, (a, b) in enumerate(chunks):
                g = guide[a:b]
                lat_t = (g.shape[0] - 1) // vae.downscale_index_formula[0] + 1
                if latent is not None:
                    empty = dict(latent)
                    sm = empty["samples"]
                    if getattr(sm, "is_nested", False) and len(chunks) > 1:
                        raise ValueError(
                            "chunked sampling with an AV (nested) latent is not supported yet: "
                            "set temporal_tile_size to 0, or feed a video-only latent")
                    fitted, was = _fit_latent_to_crop(sm, lat_t, lat_h, lat_w)
                    if fitted is not None:
                        empty["samples"] = fitted
                        if idx == 0:
                            notes.append(
                                f"latent rebuilt {was[2]*w_sf}x{was[1]*h_sf} -> "
                                f"{lat_w*w_sf}x{lat_h*h_sf} to match the crop")
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
                        # keep the audio stream from the connected AV latent, swap the video.
                        # Bound to its own name on purpose: `import comfy.nested_tensor`
                        # here would make `comfy` a LOCAL of this whole function, and
                        # every other comfy.* use in it would raise UnboundLocalError.
                        from comfy import nested_tensor as _nested
                        streams = list(base.tensors) if hasattr(base, "tensors") else list(base.unbind())
                        streams[0] = g_lat.to(streams[0].device, streams[0].dtype)
                        latent["samples"] = _nested.NestedTensor(tuple(streams))
                    else:
                        latent["samples"] = g_lat.to(base.device, base.dtype)
                    if debug_log:
                        print(f"[BFS HeadSwap] inpaint: latent seeded from the guide "
                              f"{tuple(g_lat.shape)}")
                    src_mask = masks[a:b]
                    if mask_hard_for_inpaint:
                        # hard for the sampler, soft only for compositing
                        src_mask = (src_mask > 0.5).float()
                    nm = _mask_to_latent(
                        src_mask, vae, latent["samples"].shape[2], lat_h, lat_w)
                    nm = _dilate_latent(nm, latent_mask_dilate, latent_mask_dilate_frames)
                    if mask_strength < 1.0:
                        nm = nm * float(mask_strength)
                    latent["noise_mask"] = nm.to(latent["samples"].device)

                # CRITICAL: _sample_chunk samples through guider.model_patcher, so the
                # patched clone from LTXMultipleControls has to replace it. Without the
                # swap the reference specs live in transformer_options the forward never
                # reads, and guide + identity are silently inert -- it samples happily
                # and ignores both. _set_guider_conds does the conds and the swap.
                gd = _Loop._set_guider_conds(guider, p, n, model_patcher=m)
                injections.append(_inject_transformer_options(gd, m, debug=debug_log))
                chunk = _Loop._sample_chunk(m, noise, sampler, sigmas, gd, latent, seed_offset=idx)
                got = chunk["samples"]
                if len(chunks) > 1 and not getattr(got, "is_nested", False):
                    got, produced = _keep_new_frames(got, produced, lat_t_total)
                    # a long clip is many chunks: holding them all on the sampling
                    # device grows VRAM with the clip while the model is still loaded
                    got = got.to(comfy.model_management.intermediate_device())
                out_latents.append(got)

        finally:
            # hand the guider back the way it was found, whatever happened above:
            # an exception here is exactly the case that would poison the next run
            # reverse order: with several chunks each injection captured the state
            # the previous one left, so undoing forwards would restore chunk 1's
            # specs onto the guider instead of clearing them
            for inj in reversed(injections):
                inj.undo()

        samples = out_latents[0] if len(out_latents) == 1 else torch.cat(out_latents, dim=2)
        video = samples
        if getattr(video, "is_nested", False):
            # AV latent: the video VAE only takes the video stream (this is what
            # LTXVSeparateAVLatent does in the stock graphs)
            video = video.tensors[0] if hasattr(video, "tensors") else video.unbind()[0]
        if decode == "none":
            images = torch.zeros(1, 64, 64, 3)
            notes.append("decode skipped: use your own VAE Decode on the latent output")
        else:
            if decode == "tiled":
                images = vae.decode_tiled(
                    video, tile_x=decode_tile_size, tile_y=decode_tile_size,
                    overlap=decode_overlap, tile_t=decode_temporal_size,
                    overlap_t=decode_temporal_overlap)
            else:
                images = vae.decode(video)  # the tensor, not a latent dict
            if isinstance(images, dict):
                images = images.get("samples")
            if images.ndim == 5:  # (B,T,H,W,C) -> frames batch
                images = images.reshape(-1, *images.shape[2:])

        confine = masks if (paste_confine_to_mask and masks is not None) else None
        if paste_back and decode != "none":
            final = self._paste_back(images, guide_video, crop_ctx, uncrop_feather, confine)
        else:
            final = images
            notes.append("paste_back off: images/latent stay in crop space")

        # --- inspection outputs ------------------------------------------
        h_px, w_px = guide.shape[1], guide.shape[2]
        if masks is not None:
            crop_mask_out = masks
            shown = (masks > 0.5).float() if mask_hard_for_inpaint else masks
            lat_mask = _dilate_latent(
                _mask_to_latent(shown, vae, lat_t_total, lat_h, lat_w),
                latent_mask_dilate, latent_mask_dilate_frames)[0, 0]
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
        boxes_out = _boxes_per_frame(crop_ctx, guide_video.shape[0])
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
                "crop_bboxes": ("BOUNDING_BOX", {"forceInput": True, "tooltip":
                    "Connect the sampler's crop_bboxes output -- one box per frame. This is a "
                    "socket, not something to fill in: typing a single box by hand would paste "
                    "every frame into the same fixed rectangle."}),
                "feather": ("INT", {"default": 16, "min": 0, "max": 256}),
            },
            "optional": {
                "confine_mask": ("MASK", {"tooltip":
                    "Confines the paste to the mask instead of the whole box, in the crop's space."}),
            },
        }

    DESCRIPTION = ("Composites processed crops back into the original frames, with edge-aware "
                   "feathering and optional mask confinement. The other half of the sampler's "
                   "paste_back: off, for refining a crop before it goes home.")
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, cropped_images, original_images, crop_bboxes, feather, confine_mask=None):
        return (BFSHeadSwapMaskedSampler()._paste_back(
            cropped_images, original_images, ("planned", crop_bboxes), feather, confine_mask),)



class BFSCropSize:
    """The crop's size, before anything is sampled.

    The sampler derives its box from the mask, so the size to build the empty
    latent at is only knowable after a run -- which meant sampling a whole clip
    to read one number off the debug string, and doing it again after every
    change to the mask or the crop. This runs the same planner on the same
    inputs and hands the number over up front.

    Feed `width` and `height` straight into EmptyLTXVLatentVideo. Keep these
    three widgets identical to the sampler's: the planner is deterministic, so
    equal inputs give the same box, and a mismatch here means the sampler
    silently resizes the crop to a latent that does not fit it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide_video": ("IMAGE", {"tooltip": "The same clip you feed the sampler."}),
                "subject_mask": ("MASK", {"tooltip": "The same mask you feed the sampler."}),
                "crop_mode": (["combined", "tracked", "zoomed"], {"default": "tracked"}),
                "crop_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05}),
                "crop_divisible_by": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
            },
        }

    DESCRIPTION = ("The size the sampler will crop to, computed from the mask before sampling, "
                   "so EmptyLTXVLatentVideo can be built at exactly that size instead of being "
                   "guessed and silently resized. Keep the three crop widgets identical to the "
                   "sampler's.")
    RETURN_TYPES = ("INT", "INT", "BOUNDING_BOX", "STRING")
    RETURN_NAMES = ("width", "height", "crop_bboxes", "info")
    OUTPUT_TOOLTIPS = (
        "Crop width -- into EmptyLTXVLatentVideo's width.",
        "Crop height -- into EmptyLTXVLatentVideo's height.",
        "One box per frame, the same the sampler will use. Feeds BFS Paste Back.",
        "What the planner decided, for a PreviewAny.",
    )
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, guide_video, subject_mask, crop_mode, crop_scale, crop_divisible_by):
        masks = subject_mask.squeeze(-1) if subject_mask.ndim == 4 else subject_mask
        cropped, _cm, ctx, note = BFSHeadSwapMaskedSampler()._crop(
            guide_video, masks, crop_mode, crop_scale, crop_divisible_by)
        h, w = int(cropped.shape[1]), int(cropped.shape[2])
        boxes = _boxes_per_frame(ctx, guide_video.shape[0])
        info = (f"{note} | sample size {w}x{h} "
                f"(build EmptyLTXVLatentVideo at this size) | {len(boxes)} box(es)")
        return (w, h, boxes, info)


NODE_CLASS_MAPPINGS = {
    "BFSHeadSwapMaskedSampler": BFSHeadSwapMaskedSampler,
    "BFSHeadSwapPasteBack": BFSHeadSwapPasteBack,
    "BFSCropSize": BFSCropSize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BFSHeadSwapMaskedSampler": "BFS Sampler (crop · mask · loop)",
    "BFSHeadSwapPasteBack": "BFS Paste Back",
    "BFSCropSize": "BFS Crop Size",
}
