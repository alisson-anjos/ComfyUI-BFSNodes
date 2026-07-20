"""LTX Trim First Frames — drop the first N frames of an IMAGE batch.

Palliative for the overlap reference "frame-0 leak": with layout=overlap the reference
sits at the target's frame-0 RoPE positions, so the generated first frame tends to pull
toward the reference image before the sequence follows the prompt. Dropping the first
frame(s) removes that artifact from the final video.
"""
import torch


class LTXTrimFirstFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "trim": ("INT", {"default": 1, "min": 0, "max": 16,
                     "tooltip": "Number of leading frames to drop (the overlap frame-0 leak is usually 1 frame)."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"
    CATEGORY = "LTX/identity"
    DESCRIPTION = "Drop the first N frames of a video (IMAGE batch) — removes the overlap reference frame-0 leak."

    def apply(self, images, trim=1):
        n = int(trim)
        if n <= 0 or images.shape[0] <= n:
            return (images,)
        return (images[n:],)


NODE_CLASS_MAPPINGS = {"LTXTrimFirstFrames": LTXTrimFirstFrames}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXTrimFirstFrames": "LTX Trim First Frames"}
