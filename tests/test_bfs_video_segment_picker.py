import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("bfsnodes")
pkg.__path__ = [str(ROOT)]
sys.modules["bfsnodes"] = pkg

VSP = importlib.import_module("bfsnodes.bfs_video_segment_picker")


class MaskQualityTest(unittest.TestCase):
    def test_rendered_edge_is_soft_and_stays_inside_selection(self):
        mask = torch.zeros(20, 20, dtype=torch.bool)
        mask[3:17, 3:17] = True
        alpha = VSP.feather_inside_mask(mask, 4)
        self.assertEqual(float(alpha[2, 10]), 0.0)
        self.assertGreater(float(alpha[10, 10]), 0.95)
        self.assertGreater(float(alpha[3, 10]), 0.0)
        self.assertLess(float(alpha[3, 10]), 0.5)

    def test_anchor_windows_cover_each_frame_once_and_keep_earlier_ties(self):
        windows = list(VSP.anchor_windows([0, 53, 69, 125], 0, 209))
        assigned = {i: a for a, lo, hi in windows for i in range(lo, hi)}
        self.assertEqual(len(assigned), 209)
        self.assertEqual(assigned[26], 0)
        self.assertEqual(assigned[27], 53)
        self.assertEqual(assigned[61], 53)
        self.assertEqual(assigned[62], 69)

    def test_fill_empty_tracks_only_gaps_and_preserves_existing_masks(self):
        frames = torch.zeros(8, 4, 4, 3)
        masks = torch.zeros(8, 4, 4, dtype=torch.bool)
        masks[[0, 3, 7], 1, 1] = True
        report = [{"frame": i, "source": "empty"} for i in range(8)]
        calls = []

        class FakeTracker:
            def track(self, video, model, mask, anchor_frame):
                calls.append((len(video), anchor_frame))
                out = torch.zeros(len(video), 4, 4)
                out[:, 0, 0] = 1
                return out, anchor_frame

        with patch.object(VSP, "BFSTrackFromFrame", FakeTracker):
            filled = VSP.fill_empty_mask_runs(frames, None, masks, report, [(0, 8)], active_frame=3)
        self.assertEqual(filled, 5)
        self.assertEqual(calls, [(3, 2), (4, 0)])
        self.assertTrue(bool(masks[3, 1, 1]))
        self.assertFalse(bool(masks[3, 0, 0]))
        self.assertTrue(bool(masks[5, 0, 0]))

    def test_fill_scope_does_not_request_molmo_when_masks_are_cached(self):
        selection = '{"frames":{"1":{"1":{"selected":["a1"]}}}}'
        prompt = {"7": {"inputs": {"molmo_clip": ["6", 0]}}}
        with patch.object(VSP, "video_key", return_value="video"), \
             patch.object(VSP, "_LAST_TRACK_CACHE", {("video", 1): (None, None, None)}):
            need = VSP.BFSVideoSegmentPicker().check_lazy_status(
                selection=selection, mark_style="flat color", track=True, image=torch.zeros(2, 4, 4, 3),
                molmo_clip=None, propagation_method="Molmo2 (classic)",
                propagate_scope="fill only empty masks", unique_id="7", prompt=prompt)
        self.assertEqual(need, [])


class AutoUnmarkingTest(unittest.TestCase):
    def test_find_connected_picture_count(self):
        # 1. 2 pictures connected downstream
        mock_prompt = {
            "100": {
                "class_type": "BFSH3KeyframeConditioning",
                "inputs": {
                    "guide_video": ["10", 4],
                    "picture_1": ["20", 0],
                    "picture_2": ["21", 0],
                    "picture_3": None,
                }
            },
            "10": {
                "class_type": "BFSVideoSegmentPicker",
                "inputs": {}
            }
        }
        self.assertEqual(VSP.find_connected_picture_count(mock_prompt, "10"), 2)

        # 2. 3 pictures connected downstream
        mock_prompt["100"]["inputs"]["picture_3"] = ["22", 0]
        self.assertEqual(VSP.find_connected_picture_count(mock_prompt, "10"), 3)

        # 3. Only 1 picture connected
        mock_prompt["100"]["inputs"]["picture_2"] = None
        mock_prompt["100"]["inputs"]["picture_3"] = None
        self.assertEqual(VSP.find_connected_picture_count(mock_prompt, "10"), 1)

        # 4. No downstream node connected
        mock_prompt["100"]["inputs"]["guide_video"] = ["99", 0]
        self.assertIsNone(VSP.find_connected_picture_count(mock_prompt, "10"))

        # 5. Empty prompt or None
        self.assertIsNone(VSP.find_connected_picture_count(None, "10"))
        self.assertIsNone(VSP.find_connected_picture_count({}, "10"))

    def test_input_types_controls(self):
        t = VSP.BFSVideoSegmentPicker.INPUT_TYPES()
        opt = t["optional"]
        self.assertIn("max_subjects", opt)
        self.assertEqual(opt["max_subjects"][0], ["auto", "1", "2", "3"])
        self.assertEqual(opt["max_subjects"][1]["default"], "auto")

        self.assertIn("molmo_descriptions", opt)
        self.assertEqual(opt["molmo_descriptions"][0], "STRING")
        self.assertFalse(opt["molmo_descriptions"][1].get("multiline"))
        self.assertEqual(opt["protect_mask"][0], "MASK")
        self.assertIn(VSP.PROTECTED_NORMAL_STYLE, opt["mark_style"][0])

    def test_locked_objects_survive_selection_round_trip(self):
        import json
        state = {"frames": {"7": {"1": {"selected": ["a1"]},
                                  "lock": {"selected": ["lock_p1"], "added": [
                                      {"id": "lock_p1", "points": [{"x": 0.4, "y": 0.5, "label": 1}]}]}}}}
        frames = VSP.parse_frames(json.dumps(state))
        self.assertEqual(frames["7"]["1"]["selected"], ["a1"])
        self.assertEqual(frames["7"]["lock"]["selected"], ["lock_p1"])


if __name__ == "__main__":
    unittest.main()
