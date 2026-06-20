"""Head-Swap Bernini Conditioning — standalone node for the BFS pack.

Self-contained: bundles the WanModel patch (vendored `bernini_patches.py`) that
makes the model consume `context_latents`. NO dependency on ComfyUI-RH-Bernini.

Fixed to our head-swap training case:
    guide video -> context[0] -> source_id 1   (scene/body/motion kept)
    head image  -> context[1] -> source_id 2   (identity)
    denoised output (source_id 0) = the empty latent

Sizes follow the inputs: output = guide video's native resolution (snapped to a
/16 grid), head reference = its own native resolution (snapped). Only `length`
(frame count) is a knob.

Classic ComfyUI API (NODE_CLASS_MAPPINGS) so it drops into any pack. In BFS's
__init__, either `from .headswap_node import NODE_CLASS_MAPPINGS as X; ...update`,
or merge NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS below.
"""
import logging

import torch

import comfy.model_management
import comfy.utils
import node_helpers

# Apply the bundled WanModel patch on import (idempotent; no-ops if core already
# supports Bernini context_latents).
try:
    from .bernini_patches import apply_bernini_patches
except Exception:  # pragma: no cover - flat import fallback
    from bernini_patches import apply_bernini_patches

log = logging.getLogger("BFS.HeadSwapBernini")
try:
    apply_bernini_patches()
except Exception as e:
    log.warning("HeadSwap-Bernini: WanModel patch not applied: %s", e)

STRIDE = 16


def _snap(v, stride=STRIDE):
    return max(stride, round(v / stride) * stride)


def _snap_frames(n):
    # Wan VAE compresses time by 4 -> 4k+1 frame counts.
    return max(1, ((max(1, int(n)) - 1) // 4) * 4 + 1)


def _encode_native(vae, frames):
    """VAE-encode [T,H,W,C] at native size, snapped to the /16 grid."""
    h, w = frames.shape[1], frames.shape[2]
    nh, nw = _snap(h), _snap(w)
    if (nh, nw) != (h, w):
        frames = comfy.utils.common_upscale(
            frames[:, :, :, :3].movedim(-1, 1), nw, nh, "area", "disabled"
        ).movedim(1, -1)
    return vae.encode(frames[:, :, :, :3]), nh, nw


def _tokens(lat):
    _, _, t, h, w = lat.shape
    return t * (h // 2) * (w // 2)


class HeadSwapBerniniConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "guide_video": ("IMAGE", {"tooltip": "Source video (scene/body/motion kept). Output size = this."}),
                "head_image": ("IMAGE", {"tooltip": "Reference head/face image (identity). Crop to head only."}),
                "length": ("INT", {"default": 73, "min": 1, "max": 1000, "step": 4,
                                   "tooltip": "Frame count (snapped to 4k+1). Trained at 73."}),
                "amplify_reference": ("BOOLEAN", {"default": True, "tooltip":
                    "ON: head reference goes on positive only, so the sampler's CFG amplifies "
                    "the identity (guide stays on both as a stable canvas). Use CFG ~3-5, no "
                    "LightX2V (CFG 1.0 = no amplification). OFF: reference on both (legacy, weaker)."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "debug")
    FUNCTION = "execute"
    CATEGORY = "BFS/video"

    def execute(self, positive, negative, vae, guide_video, head_image, length, amplify_reference=True):
        length = _snap_frames(length)
        guide = guide_video[:length]

        guide_lat, gh, gw = _encode_native(vae, guide)          # source_id 1
        head_lat, hh, hw = _encode_native(vae, head_image[:1])  # source_id 2

        # positive always carries guide + head. With amplify_reference, the negative
        # carries the guide ONLY (no head) -> CFG (cond-uncond) isolates and amplifies
        # the identity, while the guide stays as a stable canvas on both sides. This
        # approximates Bernini's omega_img reference guidance using ComfyUI's plain CFG.
        ctx_pos = [guide_lat, head_lat]
        ctx_neg = [guide_lat] if amplify_reference else [guide_lat, head_lat]
        positive = node_helpers.conditioning_set_values(positive, {"context_latents": ctx_pos})
        negative = node_helpers.conditioning_set_values(negative, {"context_latents": ctx_neg})

        latent = torch.zeros(
            [1, 16, ((length - 1) // 4) + 1, gh // 8, gw // 8],
            device=comfy.model_management.intermediate_device(),
        )

        dbg = (
            "=== Head-Swap Bernini Conditioning ===\n"
            f"OUTPUT (source_id 0 / target): {gw}x{gh}, {length}f -> latent {tuple(latent.shape)}\n"
            f"GUIDE  (source_id 1, kept):    native {guide_video.shape[2]}x{guide_video.shape[1]} -> {gw}x{gh} "
            f"-> latent {tuple(guide_lat.shape)} (~{_tokens(guide_lat)} tokens)\n"
            f"HEAD   (source_id 2, identity):native {head_image.shape[2]}x{head_image.shape[1]} -> {hw}x{hh} "
            f"-> latent {tuple(head_lat.shape)} (~{_tokens(head_lat)} tokens)\n"
            f"amplify_reference: {amplify_reference}  (positive ctx={len(ctx_pos)}, negative ctx={len(ctx_neg)})\n"
            + ("-> set CFG ~3-5 and DISABLE LightX2V (CFG 1.0 won't amplify).\n" if amplify_reference else "")
            + "Body of the head image bleeding in? crop head_image to head/shoulders only."
        )
        log.info("\n" + dbg)
        return (positive, negative, {"samples": latent}, dbg)


class HeadSwapLoRADebug:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "debug")
    FUNCTION = "execute"
    CATEGORY = "BFS/video"

    def execute(self, model):
        patches = getattr(model, "patches", {}) or {}
        n_keys = len(patches)
        total = sum(len(v) for v in patches.values())
        block_keys = [k for k in patches if "blocks" in k]
        sample = list(patches.keys())[:10]
        dbg = (
            "=== Head-Swap LoRA Debug ===\n"
            f"patched weight keys: {n_keys}\n"
            f"total patch entries: {total}\n"
            f"keys touching transformer blocks: {len(block_keys)}\n"
            + ("!! ZERO patches -> LoRA not loaded or keys didn't match.\n" if n_keys == 0 else "")
            + "sample keys:\n" + ("\n".join(f"  {k}" for k in sample) if sample else "  (none)")
        )
        log.info("\n" + dbg)
        return (model, dbg)


NODE_CLASS_MAPPINGS = {
    "BFSHeadSwapBerniniConditioning": HeadSwapBerniniConditioning,
    "BFSHeadSwapLoRADebug": HeadSwapLoRADebug,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BFSHeadSwapBerniniConditioning": "Head-Swap Bernini Conditioning",
    "BFSHeadSwapLoRADebug": "Head-Swap LoRA Debug",
}
