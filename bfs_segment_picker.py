"""BFS Segment Picker: segment an image into objects with SAM 3.1, pick them with the mouse, get their masks.

Connect SAM 3.1 from a Load Checkpoint node (sam3.1_multiplex_fp16.safetensors gives its MODEL and CLIP). The node
segments the first image of its input automatically (one SAM 3 text prompt per concept: "person, animal" by
default) and shows every object on a canvas in the node. Click objects to select them; the outputs are the mask of the
selection, one mask per selected object, the image with the selection tinted (the Colored Mark convention: magenta at
30 % marks the subject a body swap replaces) and a numbered overlay of everything found.

When the automatic pass misses something, add it in the panel: type what to find (a SAM 3 text prompt, e.g. "red
umbrella"), place keep / exclude points on it, or paint it with the brush (to grow the mark past what SAM found, e.g.
over hair). Added objects appear at once (the panel asks the server through
/bfs/segpick/*), and the workflow stores how each one was made, so a reload or another machine segments them again.

All state lives in the hidden `selection` STRING widget (JSON):
    {"selected": ["a3", "t1.0"], "added": [{"id": "t1", "text": "red umbrella"},
                                             {"id": "p2", "points": [{"x": .41, "y": .62, "label": 1}]},
                                             {"id": "b3", "image": "9f2c...", "strokes": [{"r": .02, "pts": [[.4, .3], [.45, .32]]}]}], "next": 4}
An entry with "remove": true (r<n> points, e<n> eraser strokes) is cut out of the selection instead of added to it.
Points are relative to the image (0..1), a stroke radius to its width. A brush entry belongs to the image it was painted
on (`image`, see image_key): on another image it is dropped, while text and point entries are segmented again. Ids: a<k> automatic objects (left to right),
t<n>.<k> the k-th match of a text entry, p<n> a point entry, b<n> a brush entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    import folder_paths
except Exception:  # tests run without ComfyUI
    folder_paths = None

PALETTE = [(230, 70, 60), (45, 125, 225), (50, 180, 90), (235, 205, 35), (205, 55, 180), (30, 200, 205),
           (240, 130, 35), (135, 75, 210), (120, 200, 60), (225, 95, 150), (80, 160, 230), (190, 150, 90)]
DEFAULT_CONCEPTS = "person, animal"
PREVIEW_SIDE = 1024          # long side of the images the panel shows (outputs keep full resolution)
MAX_SESSIONS = 6
_SESSIONS: "OrderedDict[str, dict]" = OrderedDict()


# ---------------------------------------------------------------------------------------------------------------- state

def parse_state(text: str | None) -> dict:
    """The selection widget's JSON, tolerant of empty or hand-edited values."""
    try:
        s = json.loads(text) if text and text.strip() else {}
    except (TypeError, ValueError):
        s = {}
    if not isinstance(s, dict):
        s = {}
    added = []
    for e in s.get("added") or []:
        if not isinstance(e, dict) or not e.get("id"):
            continue
        n = len(added)
        if str(e.get("text") or "").strip():
            added.append({"id": str(e["id"]), "text": str(e["text"]).strip()})
        elif e.get("points"):
            pts = [{"x": float(p["x"]), "y": float(p["y"]), "label": 1 if p.get("label", 1) else 0}
                   for p in e["points"] if isinstance(p, dict) and "x" in p and "y" in p]
            if any(p["label"] for p in pts):
                added.append({"id": str(e["id"]), "points": pts})
        elif e.get("strokes"):
            strokes = [{"r": float(st["r"]), "pts": [[float(x), float(y)] for x, y in st["pts"]], **({"erase": True} if st.get("erase") else {})}
                       for st in e["strokes"] if isinstance(st, dict) and st.get("pts")]
            if strokes:
                added.append({"id": str(e["id"]), "image": str(e.get("image") or ""), "strokes": strokes})
        if e.get("remove") and len(added) > n:
            added[-1]["remove"] = True
    selected = [str(i) for i in s.get("selected") or [] if isinstance(i, (str, int))]
    return {"selected": list(dict.fromkeys(selected)), "added": added, "next": int(s.get("next") or len(added) + 1)}


def split_concepts(text: str) -> list[str]:
    return list(dict.fromkeys(c.strip() for c in str(text or "").replace("\n", ",").split(",") if c.strip()))


def entry_key(entry: dict) -> str:
    """What an added entry asks SAM for: an entry whose text or points change is segmented again."""
    return json.dumps({k: entry[k] for k in ("text", "points", "strokes", "remove") if k in entry}, sort_keys=True)


# ------------------------------------------------------------------------------------------------------------ geometry

def box_of(mask: torch.Tensor) -> list[int]:
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return [x0, y0, x1 - x0, y1 - y0]


def dedupe(found: list[tuple[torch.Tensor, dict]], overlap: float = .8) -> list[tuple[torch.Tensor, dict]]:
    """Drop a mask that mostly repeats an earlier one (the animal prompt can find a person again, and the reverse)."""
    kept: list[tuple[torch.Tensor, dict]] = []
    for m, info in found:
        a = int(m.sum())
        if any(int((m & o).sum()) > overlap * min(a, int(o.sum())) for o, _ in kept):
            continue
        kept.append((m, info))
    return kept


def compose(masks: list[torch.Tensor], h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(union [1, h, w], individual [n, h, w]) as float masks; an empty selection gives zero masks."""
    if not masks:
        z = torch.zeros(1, h, w)
        return z, z.clone()
    stack = torch.stack([m.float() for m in masks])
    return (stack.amax(0, keepdim=True)), stack


def paint(strokes: list[dict], h: int, w: int) -> torch.Tensor:
    """Brush strokes (relative points, radius relative to the width) as a bool mask [h, w]; an "erase" stroke clears what
    the strokes before it painted (shift while painting in the panel)."""
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for st in strokes:
        r = max(1., st["r"] * w)
        pts = [(x * w, y * h) for x, y in st["pts"]]
        fill = 0 if st.get("erase") else 255
        if len(pts) > 1:
            draw.line(pts, fill=fill, width=round(2 * r), joint="curve")
        for x, y in pts:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
    return torch.from_numpy(np.array(img) > 127)


def tint(image: torch.Tensor, mask: torch.Tensor, color: tuple[int, int, int], opacity: float) -> torch.Tensor:
    """image [1, h, w, 3] with the mask's pixels blended toward the color."""
    c = torch.tensor(color, dtype=image.dtype) / 255.
    a = (mask[0].to(image.dtype) * float(opacity))[..., None]
    out = image.clone()
    out[0, ..., :3] = image[0, ..., :3] * (1 - a) + c * a
    return out


def parse_color(text: str) -> tuple[int, int, int]:
    s = str(text or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except (ValueError, IndexError):
        return 255, 0, 255


# ----------------------------------------------------------------------------------------------------------------- SAM

def _text_cond(clip, text: str, max_detections: int):
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
    # SAM3_Detect keeps one detection per text prompt unless the prompt asks for more (the multi-prompt form)
    return [[cond[0][0], {**cond[0][1], "sam3_multi_cond": [{"cond": cond[0][0], "attention_mask": cond[0][1].get("attention_mask"),
                                                             "max_detections": int(max_detections)}]}]]


def detect_text(sam: tuple, frame: torch.Tensor, text: str, threshold: float, max_detections: int) -> list[tuple[torch.Tensor, dict]]:
    from comfy_extras.nodes_sam3 import SAM3_Detect
    model, clip = sam
    h, w = frame.shape[1:3]
    # ComfyUI can materialize SAM weights as inference tensors. Its detector must then
    # run under inference_mode too; otherwise autograd asks those weights for a version counter.
    with torch.inference_mode():
        masks, boxes = SAM3_Detect.execute(model=model, image=frame, conditioning=_text_cond(clip, text, max_detections),
                                           threshold=float(threshold), refine_iterations=1, individual_masks=True).args
    masks = masks.cpu() > .5 if masks.numel() else torch.zeros(0, h, w, dtype=torch.bool)
    boxes = boxes[0] if boxes else []
    return [(m, {"score": round(float(b.get("score", 0)), 3)}) for m, b in zip(masks, boxes)]


def detect_points(sam: tuple, frame: torch.Tensor, points: list[dict], threshold: float) -> torch.Tensor:
    from comfy_extras.nodes_sam3 import SAM3_Detect
    model, _ = sam
    h, w = frame.shape[1:3]
    pos = [{"x": p["x"] * w, "y": p["y"] * h} for p in points if p["label"]]
    neg = [{"x": p["x"] * w, "y": p["y"] * h} for p in points if not p["label"]]
    with torch.inference_mode():
        masks = SAM3_Detect.execute(model=model, image=frame, positive_coords=json.dumps(pos), negative_coords=json.dumps(neg),
                                    threshold=float(threshold), refine_iterations=2).args[0]
    return (masks[0].cpu() > .5) if masks.numel() else torch.zeros(h, w, dtype=torch.bool)


# ------------------------------------------------------------------------------------------------------------- session

def image_key(frame: torch.Tensor) -> str:
    """Which image this is, from a 64x64 thumbnail and the size: what a brush entry is tied to."""
    small = torch.nn.functional.interpolate(frame[..., :3].movedim(-1, 1).float(), size=(64, 64), mode="area")
    digest = hashlib.sha1((small * 255).round().byte().numpy().tobytes())
    digest.update(str(tuple(frame.shape)).encode())
    return digest.hexdigest()[:12]


def _session_key(frame: torch.Tensor, spec: dict, sam: tuple) -> str:
    digest = hashlib.sha1(image_key(frame).encode())
    digest.update(json.dumps(spec, sort_keys=True).encode())
    digest.update(str(id(getattr(sam[0], "model", sam[0]))).encode())    # another SAM checkpoint segments again
    return digest.hexdigest()[:16]


def _remember(sid: str, sess: dict) -> dict:
    _SESSIONS[sid] = sess
    _SESSIONS.move_to_end(sid)
    while len(_SESSIONS) > MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    return sess


def auto_objects(sam: tuple, frame: torch.Tensor, concepts: list[str], threshold: float, max_per_concept: int, min_area: float) -> list[dict]:
    h, w = frame.shape[1:3]
    found = []
    for concept in concepts:
        for m, info in detect_text(sam, frame, concept, threshold, max_per_concept):
            area = int(m.sum())
            if area < max(64, min_area * h * w):
                continue
            # Generic "objects" may segment the entire scene. Such a mask is
            # unsafe as an auto subject: it paints the background into the guide.
            if concept.strip().casefold() in {"object", "objects"} and area >= .25 * h * w:
                _, _, bw, bh = box_of(m)
                if bw >= .98 * w and bh >= .98 * h:
                    continue
            found.append((m, {**info, "label": concept}))
    found = dedupe(found)
    found.sort(key=lambda f: box_of(f[0])[0])         # left to right, so "a1" reads naturally
    return [{"id": f"a{k + 1}", "label": info["label"], "source": "auto", "score": info["score"], "mask": m}
            for k, (m, info) in enumerate(found)]


def entry_objects(sam: tuple, frame: torch.Tensor, entry: dict, threshold: float, max_per_concept: int) -> list[dict]:
    objs = _entry_objects(sam, frame, entry, threshold, max_per_concept)
    if entry.get("remove"):
        for o in objs:
            o["remove"], o["label"] = True, "eraser" if entry.get("strokes") else "remove"
    return objs


def _entry_objects(sam: tuple, frame: torch.Tensor, entry: dict, threshold: float, max_per_concept: int) -> list[dict]:
    if entry.get("text"):
        hits = dedupe(detect_text(sam, frame, entry["text"], threshold, max_per_concept))
        hits.sort(key=lambda f: box_of(f[0])[0])
        return [{"id": f"{entry['id']}.{k}", "label": entry["text"], "source": "text", "entry": entry["id"],
                 "score": info["score"], "mask": m} for k, (m, info) in enumerate(hits)]
    if entry.get("strokes"):
        m = paint(entry["strokes"], *frame.shape[1:3])
        return [{"id": entry["id"], "label": "brush", "source": "brush", "entry": entry["id"], "score": None, "mask": m}] if m.any() else []
    m = detect_points(sam, frame, entry["points"], threshold)
    if int(m.sum()) < 16:
        return []
    return [{"id": entry["id"], "label": "points", "source": "points", "entry": entry["id"], "score": None, "mask": m}]


def session_for(sam: tuple, frame: torch.Tensor, spec: dict) -> tuple[str, dict]:
    sid = _session_key(frame, spec, sam)
    if sid in _SESSIONS:
        _SESSIONS.move_to_end(sid)
        _SESSIONS[sid]["sam"] = sam
        return sid, _SESSIONS[sid]
    objs = auto_objects(sam, frame, split_concepts(spec["concepts"]), spec["threshold"], spec["max_per_concept"], spec["min_area"])
    # the panel's /bfs/segpick/add segments with the SAM the node last ran with
    return sid, _remember(sid, {"frame": frame.cpu(), "image_key": image_key(frame), "spec": spec, "auto": objs, "entries": {}, "sam": sam})


def ensure_entry(sess: dict, entry: dict) -> list[dict]:
    """The objects of an added entry, segmented once per (entry id, what it asks for)."""
    key = entry_key(entry)
    have = sess["entries"].get(entry["id"])
    if have is None or have["key"] != key:
        spec = sess["spec"]
        have = {"key": key, "objects": entry_objects(sess["sam"], sess["frame"], entry, spec["threshold"], spec["max_per_concept"])}
        sess["entries"][entry["id"]] = have
    return have["objects"]


def all_objects(sess: dict, state: dict) -> list[dict]:
    objs = list(sess["auto"])
    for e in state["added"]:
        if e.get("strokes") and e["image"] and e["image"] != sess["image_key"]:
            continue                                   # painted on another image
        objs += ensure_entry(sess, e)
    return objs


# --------------------------------------------------------------------------------------------------------- panel files

def _temp_dir() -> str:
    base = folder_paths.get_temp_directory() if folder_paths else os.path.join(os.path.dirname(__file__), ".temp")
    d = os.path.join(base, "bfs_segpick")
    os.makedirs(d, exist_ok=True)
    return d


def _preview_size(h: int, w: int) -> tuple[int, int]:
    s = min(1.0, PREVIEW_SIDE / max(h, w))
    return max(1, round(w * s)), max(1, round(h * s))


def _save(img: Image.Image, prefix: str) -> dict:
    name = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
    img.save(os.path.join(_temp_dir(), name), compress_level=1)
    return {"filename": name, "subfolder": "bfs_segpick", "type": "temp"}


def panel_object(o: dict, size: tuple[int, int], h: int, w: int) -> dict:
    """What the panel needs about one object: its mask as a small PNG (white on transparent) and where it is."""
    m = Image.fromarray(o["mask"].numpy().astype(np.uint8) * 255).resize(size, Image.NEAREST)
    rgba = Image.new("RGBA", size, (255, 255, 255, 0))
    rgba.putalpha(m)
    x, y, bw, bh = box_of(o["mask"])
    if "_file" not in o:
        o["_file"] = _save(rgba, f"m_{o['id']}")
    return {"id": o["id"], "label": o["label"], "source": o["source"], "entry": o.get("entry"), "score": o["score"],
            "remove": bool(o.get("remove")), "area": round(float(o["mask"].float().mean()), 5), "box": [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)],
            "mask": o["_file"]}


def panel_data(sid: str, sess: dict, objects: list[dict]) -> dict:
    frame = sess["frame"]
    h, w = frame.shape[1:3]
    size = _preview_size(h, w)
    if "_image" not in sess:
        rgb = (frame[0, ..., :3].clamp(0, 1) * 255).byte().numpy()
        sess["_image"] = _save(Image.fromarray(rgb).resize(size, Image.LANCZOS), "img")
    return {"session": sid, "image_key": sess["image_key"], "width": int(w), "height": int(h), "image": sess["_image"],
            "objects": [panel_object(o, size, h, w) for o in objects]}


def overlay(frame: torch.Tensor, objects: list[dict], selected: set[str]) -> torch.Tensor:
    rgb = (frame[0, ..., :3].clamp(0, 1) * 255).byte().numpy().astype(np.float32)
    h, w = rgb.shape[:2]
    for n, o in enumerate(objects):
        m = o["mask"].numpy()
        a = .6 if o["id"] in selected else .3
        rgb[m] = rgb[m] * (1 - a) + np.array(PALETTE[n % len(PALETTE)], np.float32) * a
    canvas = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    size = max(14, h // 28)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    for n, o in enumerate(objects):
        x, y, bw, bh = box_of(o["mask"])
        color = PALETTE[n % len(PALETTE)]
        if o["id"] in selected:
            draw.rectangle([x, y, x + bw, y + bh], outline=color, width=max(2, h // 300))
        draw.text((x + 4, y + 4), o["id"], fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return torch.from_numpy(np.asarray(canvas, np.float32) / 255.)[None]


# ---------------------------------------------------------------------------------------------------------------- node

class BFSSegmentPicker:
    DESCRIPTION = __doc__.split("\n\n")[0] + " Click objects on the canvas to select them; add missed ones by text, points or the brush."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sam3_model": ("MODEL", {"tooltip": "SAM 3.1: MODEL of a Load Checkpoint node with sam3.1_multiplex_fp16.safetensors."}),
                "sam3_clip": ("CLIP", {"tooltip": "SAM 3.1: CLIP of the same Load Checkpoint node; turns the concepts and Find texts into SAM prompts."}),
                "concepts": ("STRING", {"default": DEFAULT_CONCEPTS, "tooltip": "What the automatic pass looks for, comma separated (one SAM 3 text prompt each), e.g. person, animal, car, chair. Empty = no automatic pass, only what you add."}),
                "threshold": ("FLOAT", {"default": .4, "min": .05, "max": .95, "step": .05, "tooltip": "SAM 3 detection score threshold: lower finds more (and more false hits)."}),
                "max_per_concept": ("INT", {"default": 8, "min": 1, "max": 32, "tooltip": "Most objects one concept or text prompt can return."}),
                "min_area": ("FLOAT", {"default": .002, "min": 0., "max": .5, "step": .001, "tooltip": "Automatic objects smaller than this share of the image are dropped."}),
                "mark_color": ("STRING", {"default": "#FF00FF", "tooltip": "Tint of the selection in the marked output (hex)."}),
                "mark_opacity": ("FLOAT", {"default": .3, "min": 0., "max": 1., "step": .05, "tooltip": "How strongly the marked output tints the selection (0.3 = the Colored Mark default)."}),
                "selection": ("STRING", {"default": "{}", "multiline": True, "tooltip": "Set by the canvas: the selected objects and the ones added by text or points."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "masks", "marked", "overlay", "objects")
    OUTPUT_TOOLTIPS = ("Union of the selected objects, minus the selected remove / eraser ones (one mask).", "One mask per selected object (remove ones already cut out), in selection order.",
                       "The image with the selection tinted (mark_color, mark_opacity).", "Every object found, numbered; the selection stronger.",
                       "JSON: size, selected ids and every object's id, label, score, box (relative x, y, w, h) and area.")
    FUNCTION = "pick"
    CATEGORY = "BFS/Mask"
    OUTPUT_NODE = True

    def pick(self, image, sam3_model, sam3_clip, concepts, threshold, max_per_concept, min_area, mark_color, mark_opacity, selection, unique_id=None):
        frame = image[:1, ..., :3].float().cpu()
        h, w = frame.shape[1:3]
        spec = {"concepts": ", ".join(split_concepts(concepts)), "threshold": round(float(threshold), 4),
                "max_per_concept": int(max_per_concept), "min_area": round(float(min_area), 5)}
        state = parse_state(selection)
        sid, sess = session_for((sam3_model, sam3_clip), frame, spec)
        objects = all_objects(sess, state)
        by_id = {o["id"]: o for o in objects}
        chosen = [by_id[i] for i in state["selected"] if i in by_id]
        cut = [o["mask"] for o in chosen if o.get("remove")]
        cut = torch.stack(cut).any(0) if cut else torch.zeros(h, w, dtype=torch.bool)
        mask, masks = compose([o["mask"] & ~cut for o in chosen if not o.get("remove")], h, w)
        marked = tint(frame, mask, parse_color(mark_color), mark_opacity)
        info = {"width": int(w), "height": int(h), "selected": [o["id"] for o in chosen],
                "objects": [{k: v for k, v in panel_object(o, _preview_size(h, w), h, w).items() if k != "mask"} for o in objects]}
        return {"ui": {"bfs_segpick": [panel_data(sid, sess, objects)]},
                "result": (mask, masks, marked, overlay(frame, objects, set(info["selected"])), json.dumps(info))}


# -------------------------------------------------------------------------------------------------------------- routes

def _routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return

    @PromptServer.instance.routes.post("/bfs/segpick/add")
    async def add(request):
        """Segment one added entry ({session, entry: {id, text} | {id, points}}) on the session's image now."""
        body = await request.json()
        sess = _SESSIONS.get(str(body.get("session") or ""))
        if sess is None:
            return web.json_response({"error": "This image is no longer in memory: run the node once, then add again."}, status=404)
        state = parse_state(json.dumps({"added": [body.get("entry") or {}]}))
        if not state["added"]:
            return web.json_response({"error": "Give a text prompt, at least one keep point or a brush stroke."}, status=400)
        try:
            objs = ensure_entry(sess, state["added"][0])
        except Exception as e:  # noqa: BLE001 - shown in the panel
            return web.json_response({"error": f"SAM 3 failed: {e}"}, status=500)
        h, w = sess["frame"].shape[1:3]
        size = _preview_size(h, w)
        return web.json_response({"objects": [panel_object(o, size, h, w) for o in objs]})


_routes()

NODE_CLASS_MAPPINGS = {"BFSSegmentPicker": BFSSegmentPicker}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSSegmentPicker": "BFS Segment Picker (SAM 3)"}
