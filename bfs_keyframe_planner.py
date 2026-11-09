"""BFS Keyframe Planner: pick keyframe times on the guide video, drop in edited frames and reference pictures.

The node shows the guide on a small player with a timeline (frames come from its last run: queue it once to load
them). "+ Keyframe" marks the current frame; each keyframe card holds the source frame and, optionally, the edited
frame for that time (drop an image on the card or upload it). Reference pictures (up to 4) are dropped or uploaded in
the node too, or connected as inputs.

Two ways to use it:
  - with edited frames: wire keyframes + keyframe_spec + pictures into BFS H3 Keyframe Swap Conditioning;
  - without: use it to choose the times; source_frames are the original frames there, ready for an image edit
    (e.g. Segment Picker + the Qwen 2.1 magenta body swap), and the edits come back into the conditioning node.

All state lives in the hidden `plan` STRING widget (JSON), saved with the workflow:
    {"keyframes": [{"frame": 53, "edit": "bfs_keyframes/abc.png"}, {"frame": 96, "edit": null}],
     "pictures": ["bfs_keyframes/ref1.png"]}
Image paths are relative to ComfyUI's input folder (the panel uploads there).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except Exception:  # tests run without ComfyUI
    folder_paths = None

MAX_PICTURES = 4
STRIP_MAX_FRAMES = 480      # the panel's filmstrip: at most this many frames (evenly spread over longer guides)
STRIP_WIDTH = 384


def parse_plan(text):
    try:
        p = json.loads(text or "{}") or {}
    except Exception:
        p = {}
    kfs = []
    for k in p.get("keyframes", []) if isinstance(p.get("keyframes"), list) else []:
        try:
            kfs.append({"frame": int(k["frame"]), "edit": k.get("edit") or None})
        except Exception:
            continue
    pics = [x for x in p.get("pictures", []) if isinstance(x, str) and x] if isinstance(p.get("pictures"), list) else []
    return {"keyframes": kfs, "pictures": pics[:MAX_PICTURES]}


def _input_path(rel):
    base = folder_paths.get_input_directory() if folder_paths else "."
    path = os.path.normpath(os.path.join(base, rel))
    if not path.startswith(os.path.normpath(base) + os.sep):
        raise ValueError(f"image path outside the input folder: {rel}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"keyframe planner image not found in input/: {rel}")
    return path


def load_image(rel):
    im = Image.open(_input_path(rel)).convert("RGB")
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0)[None]


def fit(image, h, w):
    """[1,H,W,C] -> [1,h,w,3]: center crop to the guide's aspect, then resize."""
    import comfy.utils
    x = image[:1, ..., :3].movedim(-1, 1)
    return comfy.utils.common_upscale(x, w, h, "lanczos", "center").movedim(1, -1).clamp(0, 1)


def filmstrip(video, fps):
    """Small JPEGs of the guide in the temp folder, for the panel. Same video -> same files (hashed)."""
    n = video.shape[0]
    idx = np.unique(np.linspace(0, n - 1, min(n, STRIP_MAX_FRAMES)).round().astype(int))
    h, w = video.shape[1:3]
    sw = min(STRIP_WIDTH, w); sh = max(1, round(h * sw / w))
    sample = video[:: max(1, n // 16), ::16, ::16, :3].float().cpu().numpy()
    key = hashlib.sha1(np.ascontiguousarray(sample).tobytes() + f"{n}x{h}x{w}".encode()).hexdigest()[:16]
    temp = folder_paths.get_temp_directory() if folder_paths else "."
    sub = os.path.join("bfs_kfplan", key); out = os.path.join(temp, sub); os.makedirs(out, exist_ok=True)
    files = []
    for i in idx:
        name = f"f{i:05d}.jpg"; path = os.path.join(out, name)
        if not os.path.exists(path):
            arr = (video[int(i), ..., :3].float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).resize((sw, sh), Image.BILINEAR).save(path, quality=82)
        files.append(name)
    return {"subfolder": sub, "type": "temp", "files": files, "frames": [int(i) for i in idx],
            "count": int(n), "fps": float(fps), "width": int(w), "height": int(h)}


class BFSKeyframePlanner:
    DESCRIPTION = __doc__.split("\n\n")[0]

    @classmethod
    def INPUT_TYPES(cls):
        opt = {f"picture_{j}": ("IMAGE", {"tooltip": f"Reference picture {j}; overrides the one dropped in the panel."})
               for j in range(1, MAX_PICTURES + 1)}
        return {
            "required": {
                "guide_video": ("IMAGE", {"tooltip": "The guide frames (e.g. BFS H3 Prepare Native Video)."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "plan": ("STRING", {"default": "{}", "multiline": True, "tooltip": "Set by the panel: keyframe frames, edited frames and pictures."}),
            },
            "optional": opt,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING") + ("IMAGE",) * MAX_PICTURES
    RETURN_NAMES = ("guide_video", "source_frames", "keyframes", "keyframe_spec") + tuple(f"picture_{j}" for j in range(1, MAX_PICTURES + 1))
    OUTPUT_TOOLTIPS = ("The guide, unchanged.",
                       "The original guide frames at the keyframe times (one per keyframe), e.g. to edit them.",
                       "The edited frame of every keyframe, at the guide size; a keyframe without one gives its source frame.",
                       "JSON: frames, times (s), fps and which keyframes have an edited frame.") + \
                      tuple(f"Reference picture {j} (None when not set)." for j in range(1, MAX_PICTURES + 1))
    FUNCTION = "plan_keyframes"
    CATEGORY = "BFS/MiniMax H3"
    OUTPUT_NODE = True

    def plan_keyframes(self, guide_video, fps, plan, unique_id=None, **kw):
        n, h, w = guide_video.shape[0], guide_video.shape[1], guide_video.shape[2]
        state = parse_plan(plan)
        ui = {"bfs_kfplan": [filmstrip(guide_video, fps)]}
        kfs = sorted({max(0, min(n - 1, k["frame"])): k for k in state["keyframes"]}.items())
        src, edit, spec = [], [], {"frames": [], "times": [], "edited": [], "fps": float(fps), "length": int(n)}
        for f, k in kfs:
            frame = guide_video[f:f + 1, ..., :3]
            src.append(frame)
            edit.append(fit(load_image(k["edit"]), h, w).to(frame) if k["edit"] else frame)
            spec["frames"].append(int(f)); spec["times"].append(round(f / float(fps), 4)); spec["edited"].append(bool(k["edit"]))
        if not kfs:
            print("[BFS Keyframe Planner] no keyframes yet: add them in the panel (outputs hold the first frame)", flush=True)
            src = edit = [guide_video[:1, ..., :3]]
        pictures = []
        for j in range(1, MAX_PICTURES + 1):
            p = kw.get(f"picture_{j}")
            if p is None and j <= len(state["pictures"]):
                p = load_image(state["pictures"][j - 1])
            pictures.append(p)
        return {"ui": ui, "result": (guide_video, torch.cat(src), torch.cat(edit), json.dumps(spec), *pictures)}


NODE_CLASS_MAPPINGS = {"BFSKeyframePlanner": BFSKeyframePlanner}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSKeyframePlanner": "BFS Keyframe Planner"}
