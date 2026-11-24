"""BFS H3 Face Swap • Auto Subjects (experimental): the multi face swap LoRA in one node. Reads the subjects you picked in
the BFS Video Segment Picker ('sparse face contour' mode), takes one face picture per picked color, writes the training
prompt (image prompt for a 1-frame guide, video prompt otherwise) and builds the H3 Ref2VA conditioning with the marked
guide as the guide latent. Pictures are numbered in color order of the picked subjects (red, blue, green, yellow)."""
import json

from .bfs_h3_multi_swap_prompt import COLOR_NAMES, FACE_CAPTIONS, face_swap_prompt
from .bfs_h3_swap_conditioning import BFSH3KeyframeConditioning, SPARSE_FACE

FACE_COLORS = COLOR_NAMES[:4]
FACE_CAPTION = FACE_CAPTIONS[1]   # the face swap LoRAs are trained on the v2 caption (color-shuffled datasets)


def picked_subjects(report):
    try:
        r = json.loads(report or "{}")
    except Exception:
        r = {}
    ids = sorted({int(s) for s in (r.get("active_subjects") or r.get("subjects") or []) if 1 <= int(s) <= 4})
    return ids, r.get("mode", "")


class BFSH3FaceSwapAutoSubjects:
    DESCRIPTION = __doc__.split("\n\n")[0] if "\n\n" in __doc__ else __doc__

    @classmethod
    def INPUT_TYPES(cls):
        refs = {f"{c}_reference": ("IMAGE", {"lazy": True, "tooltip": f"Face picture for the {c} subject."}) for c in FACE_COLORS}
        return {"required": {
                    "clip": ("CLIP", {"tooltip": "MiniMax H3's text encoder (Qwen3-VL)."}),
                    "vae": ("VAE",),
                    "guide": ("IMAGE", {"tooltip": "The Video Segment Picker `hybrid` output in 'sparse face contour' mode. "
                                                   "One frame = image face swap, more = video."}),
                    "picker_report": ("STRING", {"forceInput": True, "tooltip": "The Video Segment Picker report."}),
                    "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 32, "tooltip": "0 = the guide's width (multiple of 32)."}),
                    "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 32, "tooltip": "0 = the guide's height (multiple of 32)."}),
                    "prompt_override": ("STRING", {"multiline": True, "default": "", "tooltip": "Non-empty: used instead of the generated prompt."})},
                "optional": {"model": ("MODEL", {"tooltip": "Passed through (with the VRAM estimate patch)."}), **refs}}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "STRING", "IMAGE")
    RETURN_NAMES = ("model", "positive", "latent", "prompt", "pictures")
    OUTPUT_IS_LIST = (False, False, False, False, True)
    FUNCTION = "encode"
    CATEGORY = "BFS/MiniMax H3"

    def check_lazy_status(self, picker_report=None, **kw):
        ids, _ = picked_subjects(picker_report)
        return [f"{FACE_COLORS[s - 1]}_reference" for s in ids if kw.get(f"{FACE_COLORS[s - 1]}_reference") is None]

    def encode(self, clip, vae, guide, picker_report, width=0, height=0, prompt_override="", model=None, **kw):
        ids, mode = picked_subjects(picker_report)
        if not ids:
            raise ValueError("Pick at least one person in the Video Segment Picker (red, blue, green or yellow), then run again.")
        if mode and mode != "sparse face contour":
            print(f"[BFS H3 Face Swap] warning: the picker ran in mode {mode!r}; the face swap LoRA expects 'sparse face contour'.", flush=True)
        pictures = [kw.get(f"{FACE_COLORS[s - 1]}_reference") for s in ids]
        if any(p is None for p in pictures):
            raise ValueError("Connect a face picture for every picked color: " + ", ".join(FACE_COLORS[s - 1] for s in ids))
        length = int(guide.shape[0])
        h, w = int(guide.shape[1]), int(guide.shape[2])
        width = int(width) or max(32, w // 32 * 32); height = int(height) or max(32, h // 32 * 32)
        prompt = prompt_override.strip() or face_swap_prompt(range(1, len(ids) + 1), image=length == 1,
                                                              colors=[FACE_COLORS[s - 1] for s in ids], caption=FACE_CAPTION)
        node = BFSH3KeyframeConditioning()
        model, positive, latent, prompt, _, pictures = node._guide(clip, vae, guide, width, height, length, pictures, prompt,
                                                                   model, {}, mode=SPARSE_FACE)
        return model, positive, latent, prompt, pictures


NODE_CLASS_MAPPINGS = {"BFSH3FaceSwapAutoSubjects": BFSH3FaceSwapAutoSubjects}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSH3FaceSwapAutoSubjects": "BFS H3 Face Swap • Auto Subjects (experimental)"}
