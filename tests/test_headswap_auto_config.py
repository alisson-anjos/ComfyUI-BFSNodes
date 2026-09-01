"""Auto config: every pixel amount is derived from the subject's own size.

The knobs the node exposes (mask_grow, mask_blur, uncrop_feather, the latent
dilation) are absolute pixel amounts, and a pixel only means something relative
to how big the head is in frame. These tests pin the derivation: same head, two
distances, and the amounts must scale with it.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

for name in ("comfy", "comfy.model_management", "comfy.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))

SPEC = importlib.util.spec_from_file_location(
    "headswap_ltx_masked_sampler", ROOT / "headswap_ltx_masked_sampler.py")
assert SPEC is not None and SPEC.loader is not None
NODE = importlib.util.module_from_spec(SPEC)
sys.modules["headswap_ltx_masked_sampler"] = NODE
SPEC.loader.exec_module(NODE)


def masks(boxes, h=1080, w=1920):
    m = torch.zeros(len(boxes), h, w)
    for i, (x, y, bw, bh) in enumerate(boxes):
        m[i, y:y + bh, x:x + bw] = 1.0
    return m


def auto(boxes, cell=32, ref_aspect=None, headroom=1.15, **kw):
    st = NODE._subject_stats(masks(boxes, **kw))
    return NODE._auto_config(st, cell, ref_aspect, headroom)


def sides(a):
    """(up, down, left, right) growth."""
    return a["mask_grow"]


class SubjectStatsTest(unittest.TestCase):
    def test_median_ignores_a_broken_frame(self):
        boxes = [(900, 300, 100, 130)] * 20
        boxes[7] = (900, 300, 900, 900)          # one blown-up mask frame
        st = NODE._subject_stats(masks(boxes))
        self.assertEqual(st["head_w"], 100.0)

    def test_empty_mask_reports_nothing(self):
        self.assertIsNone(NODE._subject_stats(torch.zeros(8, 64, 64)))

    def test_frames_without_the_subject_do_not_break_travel(self):
        boxes = [(100, 300, 100, 130), (0, 0, 0, 0), (900, 300, 100, 130)]
        st = NODE._subject_stats(masks(boxes))
        self.assertEqual(st["seen"], 2)
        self.assertAlmostEqual(st["travel"], 800.0)


class AutoConfigTest(unittest.TestCase):
    def test_amounts_scale_with_the_head(self):
        near = auto([(700, 200, 600, 760)] * 12)
        far = auto([(900, 300, 100, 130)] * 12)
        self.assertGreater(sides(near)[2], sides(far)[2] * 4)
        self.assertGreater(near["mask_blur"], far["mask_blur"] * 4)
        self.assertGreater(near["uncrop_feather"], far["uncrop_feather"])

    def test_feather_fits_inside_the_margin_it_has(self):
        # a ramp wider than the gap between the mask and the crop border would
        # fade the head itself -- this is the seam the auto mode exists to avoid
        for w in (60, 100, 240, 600, 900):
            a = auto([(400, 200, w, int(w * 1.3))] * 8)
            margin = w * (a["crop_scale"] - 1.0) / 2.0
            left = margin - sides(a)[2]
            self.assertLessEqual(a["uncrop_feather"], max(4.0, 0.5 * left) + 1e-6,
                                 f"feather spills past the margin at head width {w}")

    def test_a_head_that_fills_the_frame_turns_cropping_off(self):
        self.assertEqual(auto([(300, 100, 1150, 950)] * 8)["crop_mode"], "off")

    def test_a_still_subject_gets_one_static_box(self):
        self.assertEqual(auto([(900, 300, 120, 150)] * 8)["crop_mode"], "combined")

    def test_a_travelling_subject_is_tracked(self):
        boxes = [(200 + i * 18, 300, 120, 150) for i in range(40)]
        self.assertEqual(auto(boxes)["crop_mode"], "tracked")

    def test_a_subject_changing_size_is_zoomed(self):
        boxes = [(900 - i * 3, 300, 100 + i * 4, 130 + i * 5) for i in range(40)]
        self.assertEqual(auto(boxes)["crop_mode"], "zoomed")

    def test_the_box_never_asks_for_more_than_the_frame(self):
        a = auto([(100, 100, 1400, 900)] * 8)
        self.assertLessEqual(1400 * a["crop_scale"], 1920 + 1e-6)
        self.assertLessEqual(900 * a["crop_scale"], 1080 + 1e-6)

    def test_the_reference_head_may_be_bigger_than_the_mask(self):
        # the mask is the OLD head: a swap clipped at its edge is the seam
        tight = auto([(900, 300, 120, 160)] * 8, headroom=1.0)
        roomy = auto([(900, 300, 120, 160)] * 8, headroom=1.5)
        self.assertGreater(sides(roomy)[0], sides(tight)[0])
        self.assertGreater(sides(roomy)[2], sides(tight)[2])

    def test_growth_never_goes_down_into_the_neck(self):
        for hr in (1.0, 1.25, 1.5, 2.0):
            up, down, left, right = sides(auto([(900, 300, 120, 160)] * 8, headroom=hr))
            self.assertLess(down, up, "growing down eats the neck and collar")
            self.assertLessEqual(down, left + 1)

    def test_a_wider_reference_pushes_the_slack_sideways(self):
        # below the caps, where the tilt still has room to act
        wide = auto([(900, 300, 120, 160)] * 8, ref_aspect=1.4, headroom=1.15)
        tall = auto([(900, 300, 120, 160)] * 8, ref_aspect=0.5, headroom=1.15)
        self.assertGreater(sides(wide)[2], sides(tall)[2])
        self.assertGreater(sides(tall)[0], sides(wide)[0])

    def test_the_box_always_holds_the_grow_plus_the_ramp(self):
        # the real invariant: whatever the mask grew by, the crop still has room
        # for it AND for the feather, or the paste fades the head
        for hr in (1.0, 1.15, 1.5, 2.0):
            for ref in (0.5, 1.0, 1.6):
                a = auto([(900, 300, 120, 160)] * 8, ref_aspect=ref, headroom=hr)
                up, down, left, _ = sides(a)
                margin_x = 120 * (a["crop_scale"] - 1.0) / 2.0
                margin_y = 160 * (a["crop_scale"] - 1.0) / 2.0
                self.assertLessEqual(left + a["uncrop_feather"], margin_x + 1)
                self.assertLessEqual(max(up, down) + a["uncrop_feather"], margin_y + 1)

    def test_latent_dilation_is_not_added_on_top_of_the_pixel_grow(self):
        # _mask_to_latent reduces with MAX, so a cell the mask touches is already
        # editable; dilating here too added a whole 32 px cell on every side and
        # blew the hole up far past the head
        for w in (40, 100, 600):
            self.assertEqual(auto([(400, 200, w, w)] * 8)["latent_mask_dilate"], 0)

    def test_growth_stays_a_sane_fraction_of_the_head(self):
        # the failure this pins: slack scaled by the mask's HEIGHT produced a
        # hole much larger than the head, with the new head floating inside it
        for hw, hh in ((120, 160), (300, 400), (600, 900)):
            for ref, hr in ((0.5, 1.15), (1.0, 1.15), (0.75, 1.5), (1.4, 2.0)):
                up, down, left, _ = sides(auto([(300, 100, hw, hh)] * 8,
                                               ref_aspect=ref, headroom=hr))
                self.assertLessEqual(up, 0.45 * hw, f"up runs away at {hw}x{hh} ref {ref} hr {hr}")
                self.assertLessEqual(left, 0.40 * hw, f"sides run away at {hw}x{hh} ref {ref} hr {hr}")

    def test_a_fast_head_gets_slack_along_time(self):
        still = auto([(800, 300, 100, 130)] * 20)
        jumpy = auto([(800 + (60 if i % 2 else 0), 300, 100, 130) for i in range(20)])
        self.assertEqual(still["latent_mask_dilate_frames"], 0)
        self.assertEqual(jumpy["latent_mask_dilate_frames"], 1)

    def test_the_note_says_what_it_chose(self):
        note = auto([(900, 300, 120, 150)] * 8)["note"]
        for token in ("head", "crop", "grow", "blur", "feather", "latent dilate",
                      "headroom", "reference aspect"):
            self.assertIn(token, note)


class DirectionalGrowTest(unittest.TestCase):
    def test_each_side_grows_by_exactly_its_own_amount(self):
        m = torch.zeros(1, 40, 40)
        m[0, 20, 20] = 1.0
        out = NODE._grow_blur(m, (5, 1, 3, 2), 0)[0]
        ys, xs = torch.where(out > 0.5)
        self.assertEqual((int(ys.min()), int(ys.max())), (15, 21))   # up 5, down 1
        self.assertEqual((int(xs.min()), int(xs.max())), (17, 22))   # left 3, right 2

    def test_a_single_number_still_grows_every_side(self):
        m = torch.zeros(1, 40, 40)
        m[0, 20, 20] = 1.0
        out = NODE._grow_blur(m, 4, 0)[0]
        ys, xs = torch.where(out > 0.5)
        self.assertEqual((int(ys.min()), int(ys.max())), (16, 24))
        self.assertEqual((int(xs.min()), int(xs.max())), (16, 24))

    def test_zero_growth_leaves_the_mask_alone(self):
        m = torch.rand(2, 16, 16).round()
        self.assertTrue(torch.equal(NODE._grow_blur(m, (0, 0, 0, 0), 0), m))


class WiringTest(unittest.TestCase):
    def test_the_widget_exists_and_defaults_to_off(self):
        opt = NODE.BFSHeadSwapMaskedSampler.INPUT_TYPES()["optional"]
        self.assertIn("auto_config", opt)
        self.assertEqual(opt["auto_config"][1]["default"], False)
        self.assertIn("identity_headroom", opt)
        self.assertEqual(opt["identity_headroom"][1]["default"], 1.15)

    def test_execute_accepts_it(self):
        import inspect
        sig = inspect.signature(NODE.BFSHeadSwapMaskedSampler.execute)
        self.assertIn("auto_config", sig.parameters)
        self.assertIs(sig.parameters["auto_config"].default, False)
        self.assertEqual(sig.parameters["identity_headroom"].default, 1.15)


if __name__ == "__main__":
    unittest.main()


class GuiderInjectionTest(unittest.TestCase):
    """The guider belongs to another node and survives between runs.

    Whatever is written into its options has to come back out, or the reference
    latents stay reachable (VRAM that is never freed) and the next run samples
    with the previous clip's specs.
    """

    class _Guider:
        def __init__(self, options=None):
            self.model_options = options if options is not None else {}

    class _Patcher:
        def __init__(self, opts):
            self.model_options = {"transformer_options": opts}

    def test_injection_is_undone(self):
        g = self._Guider()
        p = self._Patcher({"_id_ref_specs": ["latents"]})
        inj = NODE._inject_transformer_options(g, p)
        self.assertEqual(g.model_options["transformer_options"]["_id_ref_specs"], ["latents"])
        inj.undo()
        self.assertNotIn("_id_ref_specs", g.model_options["transformer_options"])

    def test_undo_restores_a_pre_existing_value_instead_of_deleting_it(self):
        g = self._Guider({"transformer_options": {"_id_ref_specs": "theirs"}})
        p = self._Patcher({"_id_ref_specs": "ours"})
        NODE._inject_transformer_options(g, p).undo()
        self.assertEqual(g.model_options["transformer_options"]["_id_ref_specs"], "theirs")

    def test_undo_is_idempotent(self):
        g = self._Guider()
        inj = NODE._inject_transformer_options(g, self._Patcher({"k": 1}))
        inj.undo()
        inj.undo()
        self.assertEqual(g.model_options["transformer_options"], {})

    def test_nothing_to_inject_still_returns_something_undoable(self):
        NODE._inject_transformer_options(self._Guider(), self._Patcher({})).undo()

    def test_the_loop_undoes_in_a_finally(self):
        import inspect
        src = inspect.getsource(NODE.BFSHeadSwapMaskedSampler.execute)
        self.assertIn("finally:", src)
        self.assertIn("inj.undo()", src)
