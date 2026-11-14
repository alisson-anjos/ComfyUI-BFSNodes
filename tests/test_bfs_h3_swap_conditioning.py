import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bfs_h3_swap_conditioning", ROOT / "bfs_h3_swap_conditioning.py")
SC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SC)
NODE = SC.BFSH3KeyframeConditioning


class InputsTest(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(SC.MODES, ["keyframe pins", "normal map guide", "DWPose guide", "translucent overlay guide", "outline guide"])

    def test_pictures_grouped_first(self):
        t = NODE.INPUT_TYPES()
        self.assertEqual(list(t["required"])[:7], ["clip", "vae", "guide_video", "width", "height", "length", "picture_1"])
        self.assertEqual(list(t["optional"])[:5], ["picture_2", "picture_3", "model", "keyframes", "keyframe_spec"])

    def test_outputs(self):
        self.assertEqual(len(NODE.RETURN_TYPES), len(NODE.RETURN_NAMES))
        self.assertEqual(len(NODE.OUTPUT_IS_LIST), len(NODE.RETURN_TYPES))
        self.assertEqual(NODE.RETURN_NAMES[-1], "pictures")
        self.assertTrue(NODE.OUTPUT_IS_LIST[-1])

    def test_unknown_mode(self):
        import torch
        with self.assertRaises(ValueError):
            NODE().encode(None, None, torch.zeros(1, 8, 8, 3), 8, 8, 1, torch.zeros(1, 8, 8, 3), mode="gray video")


class PromptTest(unittest.TestCase):
    def test_guide_prompts_name_each_subject_once(self):
        for fn, word in ((SC.normals_caption, "normal map"), (SC.pose_caption, "pose skeleton")):
            for k in range(1, SC.MAX_PICTURES + 1):
                p = fn(k)
                self.assertTrue(p.startswith("character_swap:\nsubject_definitions:\n"))
                for j in range(1, k + 1):
                    self.assertIn(f"<Subject {j}> is the subject marked by the {SC.SUBJECT_COLORS[j - 1]} {word}", p)
                self.assertNotIn(f"<Picture {k + 1}>", p)

    def test_pins_prompt(self):
        p = SC.pins_caption(2, [0.0, 1.5])
        self.assertIn("2 keyframes of the target video are pinned at 0.0s, 1.5s", p)
        self.assertIn("<Picture 2>", p)
        self.assertIn("1 keyframe of the target video is pinned at 2.0s", SC.pins_caption(1, [2.0]))

    def test_overlay_prompt(self):
        prompt = SC.overlay_caption(2)
        self.assertIn("translucent red overlay", prompt)
        self.assertIn("translucent blue overlay", prompt)
        self.assertIn("original appearance visible beneath it is not the target appearance", prompt)
        self.assertIn("The tint must disappear in the result", prompt)
        self.assertNotIn("<Picture 3>", prompt)

    def test_outline_prompt_preserves_props(self):
        prompt = SC.outline_caption(2)
        self.assertIn("red outline", prompt)
        self.assertIn("blue outline", prompt)
        self.assertIn("held or interacting objects", prompt)
        self.assertIn("foreground occlusions", prompt)
        self.assertNotIn("<Picture 3>", prompt)

    def test_normal_map_prompt_preserves_occluders(self):
        prompt = SC.normals_caption(1)
        self.assertIn("held objects and foreground occluders", prompt)
        self.assertIn("original positions, shapes and colors", prompt)


if __name__ == "__main__":
    unittest.main()
