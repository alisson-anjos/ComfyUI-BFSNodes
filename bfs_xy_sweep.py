"""BFS XY Sweep: run any workflow over a grid of widget values and get one labeled grid back.

Connect the IMAGE you want to compare (a decoded video or an image) to `result`. In the panel, pick for X (and
optionally Y) any widget of any node upstream of that input (the panel lists them, like TinyTerra's advanced xyPlot)
and type the values. On run the node does not compute `result` itself (lazy input): it expands one copy of the part of
the graph that depends on the swept widgets per combination (everything else, models included, is shared and runs
once), then a collector builds the grid.

All state lives in the hidden `sweep` STRING widget (JSON), saved with the workflow:
    {"x": {"node": "44", "widget": "lora_name", "values": ["a.safetensors", "b.safetensors"]},
     "y": {"node": "101", "widget": "keyframe_until", "values": [0.5, 0.7, 0.9]}}
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

try:
    import folder_paths
except Exception:  # tests run without ComfyUI
    folder_paths = None

CATEGORY = "BFS/Loop"
MAX_CELLS = 64


def parse_sweep(text):
    try:
        s = json.loads(text or "{}") or {}
    except Exception:
        s = {}
    axes = []
    for key in ("x", "y"):
        a = s.get(key) or {}
        if a.get("node") not in (None, "") and a.get("widget") and isinstance(a.get("values"), list) and a["values"]:
            axes.append({"node": str(a["node"]), "widget": str(a["widget"]), "values": list(a["values"]), "title": a.get("title", "")})
    return axes


def _cast(value, like):
    """A typed value from the panel, cast to the type the widget had in the prompt."""
    if isinstance(like, bool):
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(like, int) and not isinstance(like, bool):
        return int(round(float(value)))
    if isinstance(like, float):
        return float(value)
    return value if isinstance(value, str) else str(value)


def _short(v):
    s = str(v)
    if "/" in s or "\\" in s:
        s = s.replace("\\", "/").rsplit("/", 1)[-1]
    for ext in (".safetensors", ".ckpt", ".pt", ".png", ".jpg", ".mp4"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    return s if len(s) <= 40 else "…" + s[-39:]


def _font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


class BFSXYSweep:
    DESCRIPTION = __doc__.split("\n\n")[0]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "result": ("IMAGE", {"lazy": True, "tooltip": "The image or decoded video to compare; the sweep runs everything upstream of it."}),
            "frames_per_cell": ("INT", {"default": 4, "min": 1, "max": 16, "tooltip": "For videos: how many frames (evenly spread) each cell shows."}),
            "cell_height": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 16}),
            "save_videos": ("BOOLEAN", {"default": True, "tooltip": "Save every multi-frame result as an mp4 in output/bfs_sweep/, named by its values."}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            "sweep": ("STRING", {"default": "{}", "multiline": True, "tooltip": "Set by the panel: the swept widgets and their values."}),
        }, "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("grid", "results", "labels")
    OUTPUT_TOOLTIPS = ("The labeled grid.", "Every result (the cells' frames), resized to one size.", "JSON: the values of every cell, in order.")
    FUNCTION = "sweep_run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def check_lazy_status(self, result=None, **kw):
        return []   # never evaluate `result` here: the expansion evaluates one copy per combination

    def sweep_run(self, result=None, frames_per_cell=4, cell_height=256, save_videos=True, fps=24.0, sweep="{}",
                  dynprompt=None, unique_id=None):
        axes = parse_sweep(sweep)
        if not axes:
            raise ValueError("BFS XY Sweep: choose at least the X widget and its values in the panel.")
        if dynprompt is None:
            raise ValueError("BFS XY Sweep needs ComfyUI's dynamic graph execution")
        from comfy_execution.graph_utils import GraphBuilder, is_link
        me = str(unique_id)
        link = dynprompt.get_node(me)["inputs"].get("result")
        if not is_link(link):
            raise ValueError("Connect the image to compare to `result`.")
        root = str(link[0])
        # nodes upstream of `result` and who feeds whom inside that set
        up, feeds, stack = set(), {}, [root]
        while stack:
            nid = stack.pop()
            if nid in up:
                continue
            up.add(nid)
            for v in dynprompt.get_node(nid).get("inputs", {}).values():
                if is_link(v):
                    p = str(v[0]); feeds.setdefault(p, set()).add(nid); stack.append(p)
        for a in axes:
            if a["node"] not in up:
                raise ValueError(f"Swept node {a['node']} is not upstream of `result` (pick it again in the panel).")
            if a["widget"] not in dynprompt.get_node(a["node"]).get("inputs", {}):
                raise ValueError(f"Node {a['node']} has no widget '{a['widget']}' in this prompt.")
        # the body: what depends on a swept node and leads to `result` (cloned per cell); the rest is shared
        body, stack = set(), [a["node"] for a in axes]
        while stack:
            nid = stack.pop()
            if nid in body:
                continue
            body.add(nid)
            stack.extend(c for c in feeds.get(nid, ()) if c in up)
        xs = axes[0]["values"]; ys = axes[1]["values"] if len(axes) > 1 else [None]
        cells = [(x, y) for y in ys for x in xs]
        if len(cells) > MAX_CELLS:
            raise ValueError(f"{len(cells)} combinations; at most {MAX_CELLS}.")
        graph = GraphBuilder()
        outs, labels = [], []
        for ci, (x, y) in enumerate(cells):
            over = {(axes[0]["node"], axes[0]["widget"]): x}
            if y is not None:
                over[(axes[1]["node"], axes[1]["widget"])] = y
            clones = {n: graph.node(dynprompt.get_node(n)["class_type"], f"c{ci}_{n}") for n in body}
            for nid, node in clones.items():
                node.set_override_display_id(dynprompt.get_display_node_id(nid))
                for key, value in dynprompt.get_node(nid).get("inputs", {}).items():
                    if (nid, key) in over:
                        node.set_input(key, _cast(over[(nid, key)], value))
                    elif is_link(value) and str(value[0]) in body:
                        node.set_input(key, clones[str(value[0])].out(value[1]))
                    else:
                        node.set_input(key, value)
            outs.append(clones[root].out(link[1]) if root in clones else link)
            labels.append({"x": x, "y": y})
        meta = {"axes": [{k: a[k] for k in ("node", "widget", "title")} for a in axes], "cells": labels,
                "xs": xs, "ys": ys if len(axes) > 1 else [], "frames_per_cell": frames_per_cell, "cell_height": cell_height,
                "save_videos": save_videos, "fps": fps}
        coll = graph.node("BFSXYSweepCollect", "collect")
        coll.set_input("meta", json.dumps(meta))
        for i, o in enumerate(outs):
            coll.set_input(f"image_{i}", o)
        return {"result": (coll.out(0), coll.out(1), coll.out(2)), "expand": graph.finalize()}


class BFSXYSweepCollect:
    """Internal: receives every cell's result from BFS XY Sweep and draws the grid."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"meta": ("STRING", {"default": "{}"})},
                "optional": {f"image_{i}": ("IMAGE",) for i in range(MAX_CELLS)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("grid", "results", "labels")
    FUNCTION = "collect"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def collect(self, meta, **kw):
        m = json.loads(meta)
        cells = m["cells"]; k = int(m["frames_per_cell"]); ch = int(m["cell_height"])
        imgs = [kw.get(f"image_{i}") for i in range(len(cells))]
        strips, saved = [], []
        for i, (img, lab) in enumerate(zip(imgs, cells)):
            idx = np.unique(np.linspace(0, img.shape[0] - 1, min(k, img.shape[0])).round().astype(int))
            fr = img[idx, ..., :3].float()
            w = max(1, round(fr.shape[2] * ch / fr.shape[1]))
            fr = F.interpolate(fr.movedim(-1, 1), size=(ch, w), mode="bilinear", align_corners=False).movedim(1, -1)
            strips.append(torch.cat(list(fr), dim=1))
            if m.get("save_videos") and img.shape[0] > 1 and folder_paths:
                saved.append(self._save(img, lab, m, i))
        sw = max(s.shape[1] for s in strips)
        strips = [F.pad(s, (0, 0, 0, sw - s.shape[1])) for s in strips]
        xs, ys = m["xs"], m["ys"] or [None]
        ax = m["axes"]
        xl = f"{ax[0]['title'] or ax[0]['node']}.{ax[0]['widget']}"
        yl = f"{ax[1]['title'] or ax[1]['node']}.{ax[1]['widget']}" if len(ax) > 1 else ""
        pad, top, left = 6, 34, (170 if len(ax) > 1 else 0)
        H = top + len(ys) * (ch + pad); W = left + len(xs) * (sw + pad)
        canvas = Image.new("RGB", (W, H), (24, 24, 24)); d = ImageDraw.Draw(canvas); f = _font(16)
        for c, x in enumerate(xs):
            d.text((left + c * (sw + pad) + 4, 8), f"{xl} = {_short(x)}", fill=(230, 230, 230), font=f)
        for r, y in enumerate(ys):
            if y is not None:
                d.text((6, top + r * (ch + pad) + ch // 2 - 10), f"{yl} =\n{_short(y)}", fill=(230, 230, 230), font=f)
            for c in range(len(xs)):
                s = strips[r * len(xs) + c]
                arr = (s.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                canvas.paste(Image.fromarray(arr), (left + c * (sw + pad), top + r * (ch + pad)))
        grid = torch.from_numpy(np.asarray(canvas, np.float32) / 255.0)[None]
        results = torch.stack(strips)
        info = {"cells": cells, "videos": saved}
        ui = {}
        if folder_paths:
            out = os.path.join(folder_paths.get_output_directory(), "bfs_sweep"); os.makedirs(out, exist_ok=True)
            name = f"grid_{len(os.listdir(out)):05d}.png"; canvas.save(os.path.join(out, name))
            ui = {"images": [{"filename": name, "subfolder": "bfs_sweep", "type": "output"}]}
        return {"ui": ui, "result": (grid, results, json.dumps(info))}

    @staticmethod
    def _save(img, lab, m, i):
        import av
        out = os.path.join(folder_paths.get_output_directory(), "bfs_sweep"); os.makedirs(out, exist_ok=True)
        tag = "_".join(_short(v).replace(" ", "") for v in (lab["x"], lab["y"]) if v is not None)
        tag = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in tag)[:80]
        path = os.path.join(out, f"{i:02d}_{tag}.mp4")
        frames = (img[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        h, w = frames.shape[1] // 2 * 2, frames.shape[2] // 2 * 2
        with av.open(path, "w") as c:
            st = c.add_stream("libx264", rate=int(round(float(m.get("fps", 24)))))
            st.width, st.height, st.pix_fmt = w, h, "yuv420p"; st.options = {"crf": "16"}
            for fr in frames:
                for p in st.encode(av.VideoFrame.from_ndarray(fr[:h, :w], format="rgb24")):
                    c.mux(p)
            for p in st.encode():
                c.mux(p)
        return path


NODE_CLASS_MAPPINGS = {"BFSXYSweep": BFSXYSweep, "BFSXYSweepCollect": BFSXYSweepCollect}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSXYSweep": "BFS XY Sweep", "BFSXYSweepCollect": "BFS XY Sweep Collect (internal)"}
