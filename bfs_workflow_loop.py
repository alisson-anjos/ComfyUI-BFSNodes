"""BFS Workflow Loop: run any part of any workflow once per item, injecting the item (and per-iteration parameters)
into nodes you pick, and collect the results in order.

Connect the IMAGE the loop should produce to `result`, and the batch to iterate over to `items` (frames, keyframes,
marked frames...) and optionally `item_masks`. In the panel pick, among the nodes upstream of `result`:
  - inject: a node whose output is replaced, on iteration i, by item i (e.g. a Load Image that stands for "the frame",
    or a mask loader for "its mask"),
  - params: widgets that take one value per iteration (a seed, a prompt line, a strength...), cycled if shorter.
On run the node does not compute `result` itself (lazy input): it expands one copy per iteration of the part of the graph
that depends on the injected nodes or the changed widgets; models and everything else are shared and run once.

State lives in the hidden `loop` STRING widget (JSON), saved with the workflow:
    {"inject": [{"node": "6", "slot": 0, "source": "items"}],
     "params": [{"node": "34", "widget": "noise_seed", "values": [1, 2, 3]}]}
"""
import json

import torch
import torch.nn.functional as F

from .bfs_xy_sweep import _cast

CATEGORY = "BFS/Loop"
MAX_ITEMS = 256


def parse_loop(text):
    try:
        s = json.loads(text or "{}") or {}
    except Exception:
        s = {}
    inject = [{"node": str(i["node"]), "slot": int(i.get("slot", 0)), "source": i.get("source", "items")}
              for i in s.get("inject", []) if isinstance(i, dict) and i.get("node") not in (None, "")]
    params = [{"node": str(p["node"]), "widget": str(p["widget"]), "values": list(p["values"])}
              for p in s.get("params", []) if isinstance(p, dict) and p.get("node") not in (None, "") and p.get("widget")
              and isinstance(p.get("values"), list) and p["values"]]
    return inject, params


class BFSWorkflowLoop:
    DESCRIPTION = __doc__.split("\n\n")[0]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "result": ("IMAGE", {"lazy": True, "tooltip": "What one iteration produces; everything upstream of it can be looped."}),
            "loop": ("STRING", {"default": "{}", "multiline": True, "tooltip": "Set by the panel: injected nodes and per-iteration widgets."}),
        }, "optional": {
            "items": ("IMAGE", {"tooltip": "The batch to iterate over: item i is injected on iteration i."}),
            "items_b": ("IMAGE", {"tooltip": "A second batch, same length (e.g. the marked frames next to the plain ones)."}),
            "item_masks": ("MASK", {"tooltip": "One mask per item, injected where the panel says (source 'item_masks')."}),
            "count": ("INT", {"default": 0, "min": 0, "max": MAX_ITEMS,
                              "tooltip": "Iterations when there are no items (params only); 0 = as many as the longest value list."}),
        }, "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    OUTPUT_TOOLTIPS = ("The result of every iteration, in order, as one batch (resized to the first one's size).", "Iterations run.")
    FUNCTION = "run_loop"
    CATEGORY = CATEGORY

    def check_lazy_status(self, result=None, **kw):
        return []

    def run_loop(self, loop="{}", result=None, items=None, items_b=None, item_masks=None, count=0, dynprompt=None, unique_id=None):
        inject, params = parse_loop(loop)
        if not inject and not params:
            raise ValueError("BFS Workflow Loop: pick a node to inject the items into, or a widget to change, in the panel.")
        if dynprompt is None:
            raise ValueError("BFS Workflow Loop needs ComfyUI's dynamic graph execution")
        from comfy_execution.graph_utils import GraphBuilder, is_link
        me = str(unique_id); mine = dynprompt.get_node(me)["inputs"]
        link = mine.get("result")
        if not is_link(link):
            raise ValueError("Connect what one iteration produces to `result`.")
        n = int(items.shape[0]) if items is not None else (int(count) or max(len(p["values"]) for p in params) if params else 0)
        if any(i["source"] == "items" for i in inject) and items is None:
            raise ValueError("The panel injects `items`, but nothing is connected to items.")
        if any(i["source"] == "items_b" for i in inject):
            if items_b is None:
                raise ValueError("The panel injects `items_b`, but nothing is connected to items_b.")
            if items is not None and items_b.shape[0] != n:
                raise ValueError(f"{items_b.shape[0]} items_b for {n} items.")
            n = n or int(items_b.shape[0])
        if any(i["source"] == "item_masks" for i in inject):
            if item_masks is None:
                raise ValueError("The panel injects `item_masks`, but nothing is connected to item_masks.")
            if items is not None and item_masks.shape[0] != n:
                raise ValueError(f"{item_masks.shape[0]} masks for {n} items.")
        if n < 1:
            raise ValueError("Nothing to iterate over: connect items or give parameter values.")
        if n > MAX_ITEMS:
            raise ValueError(f"{n} iterations; at most {MAX_ITEMS}.")
        root = str(link[0])
        up, feeds, stack = set(), {}, [root]
        while stack:
            nid = stack.pop()
            if nid in up:
                continue
            up.add(nid)
            for v in dynprompt.get_node(nid).get("inputs", {}).values():
                if is_link(v):
                    p = str(v[0]); feeds.setdefault(p, set()).add(nid); stack.append(p)
        for i in inject:
            if i["node"] not in up:
                raise ValueError(f"Injected node {i['node']} is not upstream of `result` (pick it again in the panel).")
        for p in params:
            if p["node"] not in up:
                raise ValueError(f"Node {p['node']} is not upstream of `result` (pick it again in the panel).")
            if p["widget"] not in dynprompt.get_node(p["node"]).get("inputs", {}):
                raise ValueError(f"Node {p['node']} has no widget '{p['widget']}' in this prompt.")
        injected = {(i["node"], i["slot"]): i["source"] for i in inject}
        # the body: everything that depends on an injected output or a changed widget and leads to `result`
        seeds = [p["node"] for p in params] + [c for i in inject for c in feeds.get(i["node"], ()) if c in up]
        body, stack = set(), list(seeds)
        while stack:
            nid = stack.pop()
            if nid in body:
                continue
            body.add(nid)
            stack.extend(c for c in feeds.get(nid, ()) if c in up)
        body -= {i["node"] for i in inject if i["node"] not in {p["node"] for p in params}}
        graph = GraphBuilder()
        outs = []
        for it in range(n):
            picks = {}
            for src in set(injected.values()):
                pk = graph.node("BFSLoopPick", f"pick{it}_{src}")
                if src == "item_masks":
                    pk.set_input("masks", mine["item_masks"])
                else:
                    pk.set_input("images", mine[src])
                pk.set_input("index", it)
                picks[src] = pk
            over = {(p["node"], p["widget"]): p["values"][it % len(p["values"])] for p in params}
            clones = {nid: graph.node(dynprompt.get_node(nid)["class_type"], f"i{it}_{nid}") for nid in body}
            for nid, node in clones.items():
                node.set_override_display_id(dynprompt.get_display_node_id(nid))
                for key, value in dynprompt.get_node(nid).get("inputs", {}).items():
                    if (nid, key) in over:
                        node.set_input(key, _cast(over[(nid, key)], value))
                    elif is_link(value) and (str(value[0]), int(value[1])) in injected:
                        src = injected[(str(value[0]), int(value[1]))]
                        node.set_input(key, picks[src].out(1 if src == "item_masks" else 0))
                    elif is_link(value) and str(value[0]) in body:
                        node.set_input(key, clones[str(value[0])].out(value[1]))
                    else:
                        node.set_input(key, value)
            if root in clones:
                outs.append(clones[root].out(link[1]))
            elif (root, int(link[1])) in injected:
                src = injected[(root, int(link[1]))]; outs.append(picks[src].out(1 if src == "item_masks" else 0))
            else:
                outs.append(link)
        coll = graph.node("BFSLoopCollect", "collect")
        for i, o in enumerate(outs):
            coll.set_input(f"image_{i}", o)
        coll.set_input("n", n)
        return {"result": (coll.out(0), coll.out(1)), "expand": graph.finalize()}


class BFSLoopPick:
    """Internal: item `index` of a batch (image and/or mask)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"index": ("INT", {"default": 0, "min": 0, "max": 100000})},
                "optional": {"images": ("IMAGE",), "masks": ("MASK",)}}

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "pick"
    CATEGORY = CATEGORY

    def pick(self, index, images=None, masks=None):
        img = images[index:index + 1] if images is not None else torch.zeros(1, 64, 64, 3)
        if masks is not None:
            m = masks if masks.ndim == 3 else masks[None]
            msk = m[index:index + 1]
        else:
            msk = torch.zeros(1, img.shape[1], img.shape[2])
        return img, msk


class BFSLoopCollect:
    """Internal: every iteration's result, in order, as one batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"n": ("INT", {"default": 1, "min": 1, "max": MAX_ITEMS})},
                "optional": {f"image_{i}": ("IMAGE",) for i in range(MAX_ITEMS)}}

    RETURN_TYPES = ("IMAGE", "INT")
    FUNCTION = "collect"
    CATEGORY = CATEGORY

    def collect(self, n, **kw):
        parts = [kw[f"image_{i}"] for i in range(n)]
        h, w = parts[0].shape[1:3]
        parts = [p if p.shape[1:3] == (h, w) else
                 F.interpolate(p.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1) for p in parts]
        return torch.cat(parts), n


NODE_CLASS_MAPPINGS = {"BFSWorkflowLoop": BFSWorkflowLoop, "BFSLoopPick": BFSLoopPick, "BFSLoopCollect": BFSLoopCollect}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSWorkflowLoop": "BFS Workflow Loop", "BFSLoopPick": "BFS Loop Pick (internal)",
                              "BFSLoopCollect": "BFS Loop Collect (internal)"}


class BFSLoopSlot:
    """A stand-in for "the current item" inside a BFS Workflow Loop body: pick it in the loop's Inject list and it gets
    item i on iteration i. Outside a loop it gives a small gray image (no file to load, so validation never fails)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"label": ("STRING", {"default": "item", "tooltip": "Only a name, to tell slots apart in the loop's list."})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "slot"
    CATEGORY = CATEGORY

    def slot(self, label="item"):
        return torch.full((1, 64, 64, 3), 0.5), torch.zeros(1, 64, 64)


NODE_CLASS_MAPPINGS["BFSLoopSlot"] = BFSLoopSlot
NODE_DISPLAY_NAME_MAPPINGS["BFSLoopSlot"] = "BFS Loop Slot"
