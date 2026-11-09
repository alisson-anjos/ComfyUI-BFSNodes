import importlib
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("bfsnodes")
pkg.__path__ = [str(ROOT)]
sys.modules["bfsnodes"] = pkg

VSP = importlib.import_module("bfsnodes.bfs_video_segment_picker")


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
        self.assertTrue(opt["molmo_descriptions"][1].get("multiline"))


if __name__ == "__main__":
    unittest.main()
