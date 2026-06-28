"""Anime2Real Bernini Conditioning — standalone node for the BFS pack.

Pure v2v (NO reference). Mirrors the anime2real training case exactly: the LoRA
was trained with context `[src(source_id 1) | noisy(source_id 0)]` and the
reference (source_id 2) OMITTED. Feeding a source_id 2 token here (as the
head-swap node does) would be out-of-distribution, so this node never builds one.

    guide video -> context[0] -> source_id 1   (anime, style/content kept as canvas)
    denoised output (source_id 0) = the empty latent   (photorealistic result)

There is no `amplify_reference` knob: that trick amplifies a reference identity
via CFG, and here there is no reference. Only plain text CFG (the fixed
"photorealistic" caption) drives the style shift, so the guide goes on both the
positive and negative conditioning.

Bundles the same WanModel patch (`bernini_patches.py`) so it drops in without a
dependency on ComfyUI-RH-Bernini.
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

log = logging.getLogger("BFS.Anime2RealBernini")
try:
    apply_bernini_patches()
except Exception as e:
    log.warning("Anime2Real-Bernini: WanModel patch not applied: %s", e)

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


class Anime2RealBerniniConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "guide_video": ("IMAGE", {"tooltip": "Anime source video (style/content kept). Output size = this."}),
                "length": ("INT", {"default": 41, "min": 1, "max": 1000, "step": 4,
                                   "tooltip": "Frame count (snapped to 4k+1). Trained at 41."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "debug")
    FUNCTION = "execute"
    CATEGORY = "BFS/video"
    DESCRIPTION = (
        "Pure v2v anime->real conditioning for the Bernini-R anime2real LoRA.\n"
        "guide_video -> source_id 1 (anime canvas) | output -> source_id 0 (photoreal). NO reference.\n\n"
        "The guide is placed on BOTH positive and negative, so CFG only amplifies the text prompt "
        "(the photorealistic style), not a reference. Use a moderate CFG; the guide keeps motion/layout.\n\n"
        "Do NOT use the head-swap node with this LoRA: it injects a source_id 2 token the anime2real "
        "LoRA never saw in training."
    )

    def execute(self, positive, negative, vae, guide_video, length):
        length = _snap_frames(length)
        guide = guide_video[:length]

        guide_lat, gh, gw = _encode_native(vae, guide)          # source_id 1

        # No reference: guide is the only context, on both sides. Plain text CFG.
        ctx = [guide_lat]
        positive = node_helpers.conditioning_set_values(positive, {"context_latents": ctx})
        negative = node_helpers.conditioning_set_values(negative, {"context_latents": ctx})

        latent = torch.zeros(
            [1, 16, ((length - 1) // 4) + 1, gh // 8, gw // 8],
            device=comfy.model_management.intermediate_device(),
        )

        dbg = (
            "=== Anime2Real Bernini Conditioning ===\n"
            f"OUTPUT (source_id 0 / target): {gw}x{gh}, {length}f -> latent {tuple(latent.shape)}\n"
            f"GUIDE  (source_id 1, kept):    native {guide_video.shape[2]}x{guide_video.shape[1]} -> {gw}x{gh} "
            f"-> latent {tuple(guide_lat.shape)} (~{_tokens(guide_lat)} tokens)\n"
            "reference: NONE (pure v2v) | guide on positive AND negative (text-only CFG)\n"
            "Trained at 41f/640; staying near that resolution is safest."
        )
        log.info("\n" + dbg)
        return (positive, negative, {"samples": latent}, dbg)


NODE_CLASS_MAPPINGS = {
    "BFSAnime2RealBerniniConditioning": Anime2RealBerniniConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BFSAnime2RealBerniniConditioning": "Anime2Real Bernini Conditioning",
}
