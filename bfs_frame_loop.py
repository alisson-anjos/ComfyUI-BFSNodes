"""BFS Frame Loop: run any part of a workflow once per frame, one after another, and collect the results.

Wire a batch of frames (e.g. the Keyframe Planner's source_frames, or BFS Keyframe Marks' marked_frames) into
BFS Frame Loop Start; it gives one frame (plus its mask, index and count). Build the per-frame work after it (an image
edit: Qwen 2.1 body swap, Klein, ...), and wire its result into BFS Frame Loop End. The End node expands the body
between Start and End again for the next frame until all are done, then returns them as one batch (same order).
Models stay outside the loop body. Results are kept on disk between iterations; no global state.

The expansion follows the BFS H3 Native Shot Loop pattern (start/end graph expansion).
"""
import json
import os
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import folder_paths
except Exception:  # tests run without ComfyUI
    folder_paths = None

CATEGORY = "BFS/Loop"


class BFSFrameLoopStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE", {"tooltip": "The frames to process one by one."})},
                "optional": {"masks": ("MASK", {"tooltip": "One mask per frame (optional), handed out with its frame."}),
                             "loop_state": ("BFS_FRAME_LOOP", {"tooltip": "Set by Frame Loop End; leave unconnected."})}}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "BFS_FRAME_LOOP")
    RETURN_NAMES = ("image", "mask", "index", "count", "loop_state")
    OUTPUT_TOOLTIPS = ("The current frame.", "Its mask (zeros when no masks are given).", "Index of the current frame (0-based).",
                       "How many frames the loop runs.", "Connect to BFS Frame Loop End.")
    FUNCTION = "start"
    CATEGORY = CATEGORY
    DESCRIPTION = "Hands out one frame per loop iteration; pair it with BFS Frame Loop End."

    def start(self, images, masks=None, loop_state=None):
        n = int(images.shape[0])
        if loop_state is None:
            temp = folder_paths.get_temp_directory() if folder_paths else "."
            folder = os.path.join(temp, "bfs_frame_loop", uuid.uuid4().hex[:12])
            os.makedirs(folder, exist_ok=True)
            loop_state = {"folder": folder, "index": 0, "count": n, "done": []}
        state = dict(loop_state)
        if state["count"] != n:
            raise ValueError(f"The loop started with {state['count']} frames but now gets {n}.")
        i = int(state["index"])
        h, w = images.shape[1:3]
        if masks is not None:
            m = masks if masks.ndim == 3 else masks[None]
            if m.shape[0] != n:
                raise ValueError(f"{m.shape[0]} masks for {n} frames.")
            mask = m[i:i + 1]
            if mask.shape[1:] != (h, w):
                mask = F.interpolate(mask[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        else:
            mask = torch.zeros(1, h, w)
        print(f"[BFS Frame Loop] frame {i + 1}/{n}", flush=True)
        return images[i:i + 1], mask, i, n, state


class BFSFrameLoopEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE", {"tooltip": "The processed frame of this iteration."}),
                             "loop_state": ("BFS_FRAME_LOOP", {"tooltip": "From BFS Frame Loop Start."})},
                "optional": {"resize_to_first": ("BOOLEAN", {"default": True,
                                                             "tooltip": "Resize every result to the first result's size so they form one batch."})},
                "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    OUTPUT_TOOLTIPS = ("All processed frames, in order, as one batch.", "How many frames.")
    FUNCTION = "finish"
    CATEGORY = CATEGORY
    DESCRIPTION = "Collects one processed frame per iteration and expands the loop body until all frames are done."

    def finish(self, image, loop_state, resize_to_first=True, dynprompt=None, unique_id=None):
        state = dict(loop_state)
        i = int(state["index"])
        path = Path(state["folder"]) / f"{i:05d}.pt"
        tmp = path.with_suffix(".tmp"); torch.save(image[:1].detach().cpu().contiguous(), tmp); tmp.replace(path)
        state["done"] = list(state["done"]) + [str(path)]
        state["index"] = i + 1
        if state["index"] < state["count"]:
            if dynprompt is None:
                raise ValueError("BFS Frame Loop needs ComfyUI's dynamic graph execution")
            return self.expand(dynprompt, str(unique_id), state)
        parts = [torch.load(p, map_location="cpu", weights_only=True) for p in state["done"]]
        if resize_to_first:
            h, w = parts[0].shape[1:3]
            parts = [p if p.shape[1:3] == (h, w) else
                     F.interpolate(p.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1) for p in parts]
        if len({tuple(p.shape[1:]) for p in parts}) != 1:
            raise ValueError("The processed frames have different sizes; turn on resize_to_first.")
        (Path(state["folder"]) / "manifest.json").write_text(json.dumps(state, indent=1))
        return torch.cat(parts), len(parts)

    @staticmethod
    def expand(dynprompt, end, state):
        from comfy_execution.graph_utils import GraphBuilder, is_link
        feeds, seen = {}, set()

        def visit(nid):
            if nid in seen:
                return
            seen.add(nid)
            for value in dynprompt.get_node(nid).get("inputs", {}).values():
                if is_link(value):
                    parent = str(value[0]); feeds.setdefault(parent, []).append(nid); visit(parent)
        visit(end)
        starts = [n for n in seen if dynprompt.get_node(n)["class_type"] == "BFSFrameLoopStart"]
        if len(starts) != 1:
            raise ValueError("Connect exactly one BFS Frame Loop Start upstream of this Frame Loop End")
        start = starts[0]; body = {start}; pending = [start]
        while pending:
            for child in feeds.get(pending.pop(), []):
                if child not in body:
                    body.add(child); pending.append(child)
        graph = GraphBuilder()
        clones = {n: graph.node(dynprompt.get_node(n)["class_type"], "Next" if n == end else n) for n in body}
        for nid, node in clones.items():
            node.set_override_display_id(dynprompt.get_display_node_id(nid))
            for key, value in dynprompt.get_node(nid).get("inputs", {}).items():
                node.set_input(key, clones[str(value[0])].out(value[1]) if is_link(value) and str(value[0]) in body else value)
        clones[start].set_input("loop_state", state)
        return {"result": tuple(clones[end].out(i) for i in range(2)), "expand": graph.finalize()}


NODE_CLASS_MAPPINGS = {"BFSFrameLoopStart": BFSFrameLoopStart, "BFSFrameLoopEnd": BFSFrameLoopEnd}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSFrameLoopStart": "BFS Frame Loop Start", "BFSFrameLoopEnd": "BFS Frame Loop End"}
