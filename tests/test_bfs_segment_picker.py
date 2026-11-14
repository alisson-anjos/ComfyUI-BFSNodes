import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bfs_segment_picker", ROOT / "bfs_segment_picker.py")
SP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SP)
ORIGINAL_DETECT_TEXT = SP.detect_text
ORIGINAL_DETECT_POINTS = SP.detect_points


def rect(h, w, x0, y0, x1, y1):
    m = torch.zeros(h, w, dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


class StateTest(unittest.TestCase):
    def test_empty_and_broken_values(self):
        for v in (None, "", "{}", "not json", "[1, 2]"):
            self.assertEqual(SP.parse_state(v), {"selected": [], "added": [], "next": 1})

    def test_keeps_valid_entries_only(self):
        s = SP.parse_state(json.dumps({
            "selected": ["a1", "a1", "t1.0", 3],
            "added": [{"id": "t1", "text": " red car "}, {"id": "p2", "points": [{"x": .5, "y": .5, "label": 1}]},
                      {"id": "p3", "points": [{"x": .1, "y": .1, "label": 0}]},   # only exclude points: nothing to keep
                      {"text": "no id"}, {"id": "t4", "text": "  "}],
            "next": 5}))
        self.assertEqual(s["selected"], ["a1", "t1.0", "3"])
        self.assertEqual(s["added"], [{"id": "t1", "text": "red car"}, {"id": "p2", "points": [{"x": .5, "y": .5, "label": 1}]}])
        self.assertEqual(s["next"], 5)

    def test_concepts(self):
        self.assertEqual(SP.split_concepts("person, animal,\ncar, person ,,"), ["person", "animal", "car"])
        self.assertEqual(SP.split_concepts(""), [])

    def test_entry_key_changes_with_what_it_asks(self):
        a = SP.entry_key({"id": "t1", "text": "car"})
        self.assertEqual(a, SP.entry_key({"id": "t9", "text": "car"}))
        self.assertNotEqual(a, SP.entry_key({"id": "t1", "text": "bus"}))


class GeometryTest(unittest.TestCase):
    def test_box(self):
        self.assertEqual(SP.box_of(rect(10, 20, 3, 2, 8, 6)), [3, 2, 5, 4])
        self.assertEqual(SP.box_of(torch.zeros(4, 4, dtype=torch.bool)), [0, 0, 0, 0])

    def test_dedupe_drops_repeats_keeps_neighbours(self):
        a = rect(20, 20, 0, 0, 10, 10)
        repeat = rect(20, 20, 0, 0, 10, 9)           # 90 % inside a
        other = rect(20, 20, 8, 8, 18, 18)           # touches a a little
        kept = SP.dedupe([(a, {"n": 1}), (repeat, {"n": 2}), (other, {"n": 3})])
        self.assertEqual([i["n"] for _, i in kept], [1, 3])

    def test_compose(self):
        u, each = SP.compose([], 4, 5)
        self.assertEqual(tuple(u.shape), (1, 4, 5)); self.assertEqual(float(u.sum()), 0.)
        a, b = rect(4, 5, 0, 0, 2, 2), rect(4, 5, 1, 1, 3, 3)
        u, each = SP.compose([a, b], 4, 5)
        self.assertEqual(tuple(each.shape), (2, 4, 5))
        self.assertEqual(float(u.sum()), float((a | b).sum()))

    def test_tint_only_inside(self):
        img = torch.zeros(1, 4, 4, 3)
        m = rect(4, 4, 0, 0, 2, 2).float()[None]
        out = SP.tint(img, m, (255, 0, 255), .3)
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), .3, places=5)
        self.assertAlmostEqual(float(out[0, 0, 0, 1]), 0., places=5)
        self.assertEqual(float(out[0, 3, 3].sum()), 0.)

    def test_color(self):
        self.assertEqual(SP.parse_color("#FF00FF"), (255, 0, 255))
        self.assertEqual(SP.parse_color("0f0"), (0, 255, 0))
        self.assertEqual(SP.parse_color("nope"), (255, 0, 255))


class InferenceModeTest(unittest.TestCase):
    def test_text_and_point_sam_calls_run_in_inference_mode(self):
        calls = []

        class FakeDetect:
            @staticmethod
            def execute(**kwargs):
                calls.append(torch.is_inference_mode_enabled())
                return types.SimpleNamespace(args=(torch.ones(1, 4, 4), [[{"score": 0.9}]]))

        parent = types.ModuleType("comfy_extras")
        parent.__path__ = []
        child = types.ModuleType("comfy_extras.nodes_sam3")
        child.SAM3_Detect = FakeDetect
        frame = torch.zeros(1, 4, 4, 3)
        with patch.dict(sys.modules, {"comfy_extras": parent, "comfy_extras.nodes_sam3": child}), \
             patch.object(SP, "_text_cond", return_value=[]):
            self.assertEqual(len(ORIGINAL_DETECT_TEXT((None, None), frame, "person", 0.4, 1)), 1)
            self.assertTrue(ORIGINAL_DETECT_POINTS((None, None), frame, [{"x": 0.5, "y": 0.5, "label": 1}], 0.4).any())
        self.assertEqual(calls, [True, True])


class AutoObjectsTest(unittest.TestCase):
    def test_rejects_scene_sized_generic_object_but_keeps_person(self):
        h, w = 20, 40
        scene = torch.zeros(h, w, dtype=torch.bool)
        scene[::2, :] = True
        scene[-1, :] = True
        person = rect(h, w, 10, 2, 28, 18)

        def detect_text(sam, frame, concept, threshold, max_detections):
            mask = person if concept == "person" else scene
            return [(mask, {"score": .9})]

        frame = torch.zeros(1, h, w, 3)
        with patch.object(SP, "detect_text", side_effect=detect_text):
            objects = SP.auto_objects((None, None), frame, ["objects", "person"], .4, 8, 0.)
        with patch.object(SP, "detect_text", return_value=[(scene, {"score": .9})]):
            large_person = SP.auto_objects((None, None), frame, ["person"], .4, 8, 0.)
        self.assertEqual([(o["id"], o["label"]) for o in objects], [("a1", "person")])
        self.assertEqual(len(large_person), 1)


class SessionTest(unittest.TestCase):
    """The node with SAM replaced: what is segmented once, what is segmented again, what the outputs hold."""

    def setUp(self):
        self.calls = []
        h, w = 20, 40
        people = [rect(h, w, 25, 2, 35, 18), rect(h, w, 2, 2, 12, 18)]

        def detect_text(sam, frame, text, threshold, max_detections):
            self.calls.append(("text", text))
            if text == "person":
                return [(m, {"score": .9}) for m in people]
            if text == "dog":
                return [(rect(h, w, 15, 10, 20, 15), {"score": .7})]
            return []

        def detect_points(sam, frame, points, threshold):
            self.calls.append(("points", len(points)))
            return rect(h, w, 14, 0, 19, 6)

        SP.detect_text, SP.detect_points = detect_text, detect_points
        SP._SESSIONS.clear()
        self.frame = torch.rand(1, h, w, 3)
        self.sam = (object(), object())
        self.tmp = Path(__file__).resolve().parent / ".segpick_tmp"
        SP._temp_dir = lambda: str(self.tmp.mkdir(exist_ok=True) or self.tmp)

    def tearDown(self):
        for f in self.tmp.glob("*.png"):
            f.unlink()
        if self.tmp.exists():
            self.tmp.rmdir()

    def run_node(self, selection, concepts="person"):
        return SP.BFSSegmentPicker().pick(self.frame, self.sam[0], self.sam[1], concepts, .4, 8, 0., "#FF00FF", .3, json.dumps(selection))

    def test_auto_ids_left_to_right_and_selection(self):
        out = self.run_node({"selected": ["a2"]})
        mask, masks, marked, overlay, info = out["result"]
        info = json.loads(info)
        self.assertEqual([o["id"] for o in info["objects"]], ["a1", "a2"])
        self.assertLess(info["objects"][0]["box"][0], info["objects"][1]["box"][0])
        self.assertEqual(info["selected"], ["a2"])
        self.assertEqual(float(mask[0, 10, 30]), 1.)      # a2 is the person on the right
        self.assertEqual(float(mask[0, 10, 5]), 0.)
        self.assertEqual(tuple(masks.shape), (1, 20, 40))
        self.assertGreater(float(marked[0, 10, 30, 0]), float(self.frame[0, 10, 30, 0]) * .7 - 1e-6)
        panel = out["ui"]["bfs_segpick"][0]
        self.assertEqual(len(panel["objects"]), 2)
        self.assertTrue((self.tmp / panel["objects"][0]["mask"]["filename"]).exists())

    def test_segments_once_then_only_new_entries(self):
        self.run_node({"selected": ["a1"]})
        self.run_node({"selected": ["a2"]})                # only the selection changed
        self.assertEqual(self.calls, [("text", "person")])
        sel = {"selected": ["t1.0", "p2"], "added": [{"id": "t1", "text": "dog"},
                                                     {"id": "p2", "points": [{"x": .37, "y": .05, "label": 1}]}]}
        _, masks, *_ , info = self.run_node(sel)["result"]
        self.assertEqual(json.loads(info)["selected"], ["t1.0", "p2"])
        self.assertEqual(tuple(masks.shape), (2, 20, 40))
        self.run_node(sel)
        self.assertEqual(self.calls, [("text", "person"), ("text", "dog"), ("points", 1)])
        sel["added"][0]["text"] = "cat"                     # an edited entry is segmented again
        _, _, _, _, info = self.run_node(sel)["result"]
        self.assertEqual(self.calls[-1], ("text", "cat"))
        self.assertEqual(json.loads(info)["selected"], ["p2"])   # "cat" found nothing, so t1.0 is gone

    def test_brush_is_dropped_on_another_image(self):
        key = SP.image_key(self.frame)
        sel = {"selected": ["a1", "b1"], "added": [{"id": "b1", "image": key, "strokes": [{"r": .05, "pts": [[.5, .5]]}]}]}
        out = self.run_node(sel)
        self.assertEqual(out["ui"]["bfs_segpick"][0]["image_key"], key)
        self.assertEqual(json.loads(out["result"][4])["selected"], ["a1", "b1"])
        self.frame = torch.rand(1, 20, 40, 3)                  # another image: the strokes no longer apply
        mask, _, _, _, info = self.run_node(sel)["result"]
        self.assertEqual(json.loads(info)["selected"], ["a1"])
        self.assertEqual(float(mask[0, 10, 20]), 0.)

    def test_remove_entries_cut_the_selection(self):
        key = SP.image_key(self.frame)
        sel = {"selected": ["a1", "e1", "r2"], "added": [
            {"id": "e1", "image": key, "remove": True, "strokes": [{"r": .05, "pts": [[.175, .5]]}]},
            {"id": "r2", "remove": True, "points": [{"x": .4, "y": .1, "label": 1}]}]}
        out = self.run_node(sel)
        mask, masks, _, _, info = out["result"]
        self.assertEqual(float(mask[0, 10, 7]), 0.)              # erased out of a1
        self.assertEqual(float(mask[0, 10, 3]), 1.)              # the rest of a1 stays
        self.assertEqual(tuple(masks.shape), (1, 20, 40))       # remove entries are not masks of their own
        self.assertEqual(float(masks[0, 10, 7]), 0.)
        self.assertEqual(json.loads(info)["selected"], ["a1", "e1", "r2"])
        panel = {o["id"]: o for o in out["ui"]["bfs_segpick"][0]["objects"]}
        self.assertEqual((panel["e1"]["label"], panel["e1"]["remove"]), ("eraser", True))
        self.assertEqual((panel["r2"]["label"], panel["r2"]["remove"]), ("remove", True))
        self.assertFalse(panel["a1"]["remove"])
        sel["selected"] = ["a1", "r2"]                           # unselecting the eraser gives a1 back whole
        self.assertEqual(float(self.run_node(sel)["result"][0][0, 10, 7]), 1.)

    def test_unknown_ids_are_ignored(self):
        mask, masks, *_ = self.run_node({"selected": ["a9", "t5.0"]})["result"]
        self.assertEqual(float(mask.sum()), 0.)
        self.assertEqual(tuple(masks.shape), (1, 20, 40))

    def test_another_sam_is_another_session(self):
        self.run_node({})
        self.sam = (object(), object())
        self.run_node({})
        self.assertEqual(self.calls, [("text", "person"), ("text", "person")])

    def test_other_concepts_are_another_session(self):
        self.run_node({}, concepts="person")
        self.run_node({}, concepts="person, dog")
        self.assertEqual(self.calls, [("text", "person"), ("text", "person"), ("text", "dog")])


class BrushTest(unittest.TestCase):
    def test_state_keeps_strokes(self):
        s = SP.parse_state(json.dumps({"added": [{"id": "b1", "strokes": [{"r": .1, "pts": [[.2, .5], [.8, .5]]}]},
                                                 {"id": "b2", "strokes": [{"r": .1, "pts": []}]}]}))
        self.assertEqual(s["added"], [{"id": "b1", "image": "", "strokes": [{"r": .1, "pts": [[.2, .5], [.8, .5]]}]}])
        self.assertNotEqual(SP.entry_key(s["added"][0]), SP.entry_key({"id": "b1", "strokes": [{"r": .2, "pts": [[.2, .5]]}]}))
        r = SP.parse_state(json.dumps({"added": [{"id": "r1", "remove": True, "points": [{"x": .5, "y": .5, "label": 1}]},
                                                 {"id": "p2", "remove": False, "points": [{"x": .5, "y": .5, "label": 1}]}]}))
        self.assertEqual([e.get("remove") for e in r["added"]], [True, None])

    def test_paint_covers_the_stroke_only(self):
        m = SP.paint([{"r": .1, "pts": [[.2, .5], [.8, .5]]}], 50, 100)
        self.assertTrue(m[25, 20] and m[25, 50] and m[25, 80])
        self.assertFalse(m[5, 50] or m[25, 5] or m[25, 95])

    def test_brush_entry_needs_no_sam(self):
        frame = torch.zeros(1, 50, 100, 3)
        objs = SP.entry_objects(None, frame, {"id": "b3", "strokes": [{"r": .05, "pts": [[.5, .5]]}]}, .4, 8)
        self.assertEqual([(o["id"], o["source"]) for o in objs], [("b3", "brush")])
        self.assertTrue(objs[0]["mask"][25, 50])


if __name__ == "__main__":
    unittest.main()
