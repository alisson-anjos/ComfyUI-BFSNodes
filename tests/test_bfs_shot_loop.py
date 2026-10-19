from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="bfs_shotloop_test_")

# The module only needs a few folder_paths calls; stub them so the tests run without ComfyUI.
_fp = types.ModuleType("folder_paths")
_fp.get_temp_directory = lambda: _TMP
_fp.get_input_directory = lambda: _TMP
_fp.get_annotated_filepath = lambda name: str(Path(_TMP) / name)
sys.modules.setdefault("folder_paths", _fp)


class _Blocker:
    def __init__(self, message):
        self.message = message


_ce = types.ModuleType("comfy_execution")
_gu = types.ModuleType("comfy_execution.graph_utils")
_gu.ExecutionBlocker = _Blocker
sys.modules.setdefault("comfy_execution", _ce)
sys.modules.setdefault("comfy_execution.graph_utils", _gu)

SPEC = importlib.util.spec_from_file_location("bfs_shot_loop", ROOT / "bfs_shot_loop.py")
assert SPEC is not None and SPEC.loader is not None
SL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SL)
H3 = "H3 (17n+5)"


def _shot(i, start, end, cut=False, gen=None, count=1, queue=False, run_id=""):
    length = end - start
    return {"index": i, "count": count, "start": start, "end": end, "length": length,
            "gen_length": gen or SL.snap_up(length, H3), "fps": 24.0, "cut_before": cut,
            "queue": queue, "run_id": run_id}


class GridTest(unittest.TestCase):
    def test_h3_grid(self):
        self.assertEqual([SL.snap_up(x, H3) for x in (1, 5, 6, 22, 23, 107, 108)], [5, 5, 22, 22, 39, 107, 124])
        self.assertEqual([SL.snap_down(x, H3) for x in (4, 5, 21, 22, 106, 107, 110)], [5, 5, 5, 22, 90, 107, 107])

    def test_other_grids(self):
        self.assertEqual(SL.snap_up(50, "LTX / Wan (8n+1)"), 57)
        self.assertEqual(SL.snap_down(50, "LTX / Wan (8n+1)"), 49)
        self.assertEqual(SL.snap_up(6, "Wan (4n+1)"), 9)
        self.assertEqual(SL.snap_up(37, "any"), 37)

    def test_generation_size_is_on_the_multiple(self):
        w, h = SL.generation_size(1280, 720, 0.15, 32)
        self.assertEqual((w % 32, h % 32), (0, 0))
        self.assertGreater(w, h)

    def test_planner_defaults_to_half_megapixel(self):
        self.assertEqual(SL.DEFAULT_PLAN["megapixels"], 0.5)


class TextGenerateCompatibilityTest(unittest.TestCase):
    def _generate(self, execute):
        calls = []

        def wrapped(*args, **kwargs):
            result = execute(*args, **kwargs)
            calls.append(kwargs)
            return result

        textgen = types.ModuleType("comfy_extras.nodes_textgen")
        textgen.TextGenerate = type("TextGenerate", (), {"execute": staticmethod(wrapped)})
        comfy_extras = types.ModuleType("comfy_extras")
        comfy_extras.__path__ = []
        modules = {"comfy_extras": comfy_extras, "comfy_extras.nodes_textgen": textgen}
        with patch.dict(sys.modules, modules):
            result = SL.vlm_generate(object(), "describe this", torch.zeros(1, 2, 2, 3), 32)
        self.assertEqual(result, "generated")
        return calls[0]

    def test_each_textgenerate_call_cleans_up_before_the_next(self):
        import bfs_agent_inference as inference
        records = []
        def execute(**kwargs):
            self.assertTrue(torch.is_inference_mode_enabled())
            records.append("generate")
            return types.SimpleNamespace(args=("generated",))
        generator = types.SimpleNamespace(execute=execute)
        with patch.object(inference, "cleanup_inference", lambda: records.append("cleanup")):
            for _ in range(2):
                SL._textgenerate_execute(generator, clip=object(), prompt="inspect", max_length=32,
                                         sampling_mode={"sampling_mode": "off"})
        self.assertEqual(records, ["generate", "cleanup", "generate", "cleanup"])

    def test_old_textgenerate_signature_does_not_receive_mtp(self):
        def execute(clip, prompt, max_length, sampling_mode, image=None):
            return types.SimpleNamespace(args=("generated",))

        kwargs = self._generate(execute)
        self.assertNotIn("mtp", kwargs)

    def test_new_textgenerate_signature_keeps_mtp_disabled(self):
        def execute(clip, prompt, max_length, sampling_mode, image=None, mtp="auto"):
            return types.SimpleNamespace(args=("generated",))

        kwargs = self._generate(execute)
        self.assertEqual(kwargs["mtp"], "off")


class PlanTest(unittest.TestCase):
    def test_fixed_mode_splits_evenly_under_the_limit(self):
        segs = SL.plan_segments(300, [], "fixed", 107, 24)
        self.assertEqual([s["end"] - s["start"] for s in segs], [100, 100, 100])

    def test_camera_cut_mode_splits_a_continuous_range_at_max_length(self):
        segs = SL.plan_segments(360, [], "shots", 107, 1)
        self.assertEqual(len(segs), 4)
        self.assertTrue(all(s["end"] - s["start"] <= 107 for s in segs))
        self.assertEqual(segs[-1]["end"], 360)

    def test_shots_split_at_cuts_and_long_shots_are_divided(self):
        segs = SL.plan_segments(300, [50, 80], "shots", 107, 10)
        bounds = [(s["start"], s["end"]) for s in segs]
        self.assertEqual(bounds[:2], [(0, 50), (50, 80)])
        self.assertTrue(all(e - s <= 107 for s, e in bounds))
        self.assertEqual(bounds[-1][1], 300)
        self.assertTrue(segs[1]["cut_before"])

    def test_short_shots_merge_into_a_neighbour_that_still_fits(self):
        segs = SL.plan_segments(200, [10, 100], "shots", 107, 24)
        self.assertEqual([(s["start"], s["end"]) for s in segs], [(0, 100), (100, 200)])

    def test_limits_and_manual_bounds(self):
        self.assertEqual(len(SL.plan_segments(1000, [], "fixed", 100, 10, max_parts=3)), 3)
        self.assertEqual(SL.plan_segments(1000, [], "fixed", 100, 10, max_total=250)[-1]["end"], 250)
        manual = SL.plan_segments(100, [], "manual", 107, 10, manual=[30, 70])
        self.assertEqual([(s["start"], s["end"]) for s in manual], [(0, 30), (30, 70), (70, 100)])

    def test_builtin_detector_finds_spikes_and_respects_the_gap(self):
        score = [1.0] * 100
        raw = [0.01] * 100
        for i in (20, 22, 60):
            score[i], raw[i] = 50.0, 0.5
        self.assertEqual(SL.detect_cuts(score, raw, 0.5, 24.0), [20, 60])


class JoinTest(unittest.TestCase):
    def _frames(self, n, value):
        return torch.full((n, 8, 8, 3), float(value))

    def test_trims_to_true_lengths_in_order(self):
        shots = [_shot(1, 30, 50, cut=True, gen=22), _shot(0, 0, 30, gen=39)]
        imgs = [self._frames(22, 0.8), self._frames(39, 0.2)]
        video, audio, fps = SL.BFSShotJoin()._join_all(imgs, shots, [0])
        self.assertEqual(video.shape[0], 50)
        self.assertAlmostEqual(float(video[0, 0, 0, 0]), 0.2)
        self.assertAlmostEqual(float(video[-1, 0, 0, 0]), 0.8)
        self.assertEqual(fps, 24.0)

    def test_crossfade_only_across_soft_joins(self):
        imgs = [self._frames(39, 0.0), self._frames(39, 1.0)]
        soft = SL.BFSShotJoin()._join_all(imgs, [_shot(0, 0, 30, gen=39), _shot(1, 30, 60, gen=39)], [4])[0]
        hard = SL.BFSShotJoin()._join_all(imgs, [_shot(0, 0, 30, gen=39), _shot(1, 30, 60, cut=True, gen=39)], [4])[0]
        self.assertGreater(float(soft[31, 0, 0, 0]), 0.0)
        self.assertLess(float(soft[31, 0, 0, 0]), 1.0)
        self.assertEqual(float(hard[31, 0, 0, 0]), 1.0)

    def test_audio_is_trimmed_to_the_video(self):
        audio = {"waveform": torch.zeros(1, 2, 48000 * 10), "sample_rate": 48000}
        imgs = [self._frames(39, 0.5)]
        _, a, _ = SL.BFSShotJoin()._join_all(imgs, [_shot(0, 0, 24, gen=39)], [0], [audio])
        self.assertEqual(a["waveform"].shape[-1], 48000)

    def test_queue_mode_blocks_until_every_shot_is_stored(self):
        join = SL.BFSShotJoin()
        shots = [_shot(0, 0, 20, count=2, queue=True, run_id="t1"), _shot(1, 20, 45, cut=True, count=2, queue=True, run_id="t1")]
        first = join.join([self._frames(22, 0.1)], [shots[0]], [0])
        self.assertIsInstance(first[0], _Blocker)
        video, _, _, _, _, _ = join.join([self._frames(39, 0.9)], [shots[1]], [0])
        self.assertEqual(video.shape[0], 45)
        self.assertAlmostEqual(float(video[0, 0, 0, 0]), 0.1, places=2)

    def test_run_again_with_every_shot_stored_generates_nothing(self):
        join = SL.BFSShotJoin()
        shots = [_shot(0, 0, 20, count=2, queue=True, run_id="t2"), _shot(1, 20, 45, cut=True, count=2, queue=True, run_id="t2")]
        join.join([self._frames(22, 0.1)], [shots[0]], [0])
        join.join([self._frames(39, 0.9)], [shots[1]], [0])
        again = dict(shots[1], assemble_only=True)       # what the planner hands out when nothing is left to run
        self.assertEqual(join.check_lazy_status(images=None, shots=[again]), [])     # the sampler is not pulled
        self.assertEqual(join.check_lazy_status(images=None, shots=[shots[1]]), ["images"])
        video = SL.BFSShotJoin().join(shots=[again], crossfade=[0])[0]
        self.assertEqual(video.shape[0], 45)
        self.assertAlmostEqual(float(video[0, 0, 0, 0]), 0.1, places=2)
        self.assertAlmostEqual(float(video[-1, 0, 0, 0]), 0.9, places=2)

    def test_comparison_turned_on_later_reads_the_originals_from_the_source(self):
        import cv2
        import numpy as np
        src = os.path.join(_TMP, "cmp_src.mp4")
        vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 24, (32, 32))
        for _ in range(50):
            vw.write(np.full((32, 32, 3), 200, np.uint8))
        vw.release()
        tl = {"path": src, "fps": 24.0, "width": 32, "height": 32, "fill": "original", "audio": None,
              "segs": [{"start": 0, "end": 20, "run": True, "cut_before": False, "run_index": 0},
                       {"start": 20, "end": 45, "run": True, "cut_before": True, "run_index": 1}],
              "src": list(range(50)), "n": 50}
        shots = [_shot(0, 0, 20, count=2, queue=True, run_id="t3"), _shot(1, 20, 45, cut=True, count=2, queue=True, run_id="t3")]
        join = SL.BFSShotJoin()
        frames = lambda n, v: torch.full((n, 32, 32, 3), float(v))
        join.join([frames(22, 0.1)], [shots[0]], [0], timeline=[tl])          # stored with the comparison off
        join.join([frames(39, 0.9)], [shots[1]], [0], timeline=[tl])
        out = SL.BFSShotJoin().join(shots=[dict(shots[1], assemble_only=True)], crossfade=[0], timeline=[tl],
                                    comparison=[True], comparison_height=[32])
        comp, guide = out[3], out[5]
        self.assertEqual(guide.shape[0], 45)                         # the guide output is there again
        self.assertAlmostEqual(float(guide[10].mean()), 200 / 255, places=1)   # and it is the source video
        self.assertGreater((comp - 200 / 255).abs().lt(0.05).float().mean().item(), 0.05)   # original column shown


class FilterTest(unittest.TestCase):
    STATS = {"persons": 1, "person_area": 0.2, "person_frames": 6, "faces": 1, "brightness": 0.4, "motion": 0.02, "sampled": 6}

    def test_no_filter_keeps_everything(self):
        self.assertEqual(SL.skip_reason(self.STATS, 50, dict(SL.DEFAULT_FILTERS)), "")
        self.assertFalse(SL.filters_active(dict(SL.DEFAULT_FILTERS)))

    def test_each_filter(self):
        f = dict(SL.DEFAULT_FILTERS)
        self.assertEqual(SL.skip_reason(dict(self.STATS, persons=0), 50, dict(f, person=True)), "no person")
        self.assertIn("smaller", SL.skip_reason(dict(self.STATS, person_area=0.01), 50, dict(f, person=True, min_person_area=0.03)))
        self.assertIn("more than", SL.skip_reason(dict(self.STATS, persons=5), 50, dict(f, max_persons=2)))
        self.assertEqual(SL.skip_reason(dict(self.STATS, faces=0), 50, dict(f, face=True)), "no face")
        self.assertEqual(SL.skip_reason(dict(self.STATS, brightness=0.02), 50, dict(f, skip_dark=True)), "dark / fade")
        self.assertEqual(SL.skip_reason(dict(self.STATS, motion=0.0), 50, dict(f, skip_static=True)), "static")
        self.assertIn("shorter", SL.skip_reason(self.STATS, 5, dict(f, min_frames=10)))


class TimelineJoinTest(unittest.TestCase):
    def _tl(self, fill):
        return {"path": "", "fps": 24.0, "width": 8, "height": 8, "fill": fill, "audio": None, "n": 60,
                "src": list(range(60)),
                "segs": [{"start": 0, "end": 20, "run": True, "cut_before": False, "run_index": 0},
                         {"start": 20, "end": 40, "run": False, "cut_before": True, "run_index": -1},
                         {"start": 40, "end": 60, "run": True, "cut_before": True, "run_index": 1}]}

    def test_drop_removes_skipped_shots(self):
        shots = [_shot(0, 0, 20, gen=22), _shot(1, 40, 60, cut=True, gen=22)]
        imgs = [torch.zeros(22, 8, 8, 3), torch.ones(22, 8, 8, 3)]
        video = SL.BFSShotJoin()._join_all(imgs, shots, [0], None, self._tl("drop"))[0]
        self.assertEqual(video.shape[0], 40)
        self.assertEqual(float(video[25, 0, 0, 0]), 1.0)

    def test_original_fill_reads_the_source_frames(self):
        shots = [_shot(0, 0, 20, gen=22), _shot(1, 40, 60, cut=True, gen=22)]
        imgs = [torch.zeros(22, 8, 8, 3), torch.ones(22, 8, 8, 3)]
        orig = SL._read_frames
        SL._read_frames = lambda path, idx, size: [__import__("numpy").full((8, 8, 3), 128, "uint8") for _ in idx]
        try:
            video = SL.BFSShotJoin()._join_all(imgs, shots, [0], None, self._tl("original"))[0]
        finally:
            SL._read_frames = orig
        self.assertEqual(video.shape[0], 60)
        self.assertAlmostEqual(float(video[30, 0, 0, 0]), 128 / 255, places=3)

    def test_skipped_shot_keeps_its_sound(self):
        # 60 frames at 24 fps, sample rate 240 -> 10 samples per frame; each sample holds its timeline frame number
        sr = 240
        timeline_audio = {"waveform": torch.arange(600, dtype=torch.float32).div(10).floor().view(1, 1, 600), "sample_rate": sr}
        runs_only = {"waveform": torch.full((1, 1, 400), -1.0), "sample_rate": sr, "bfs_runs_only": True}
        tl = dict(self._tl("original"), audio=timeline_audio)
        shots = [_shot(0, 0, 20, gen=22), _shot(1, 40, 60, cut=True, gen=22)]
        imgs = [torch.zeros(22, 8, 8, 3), torch.ones(22, 8, 8, 3)]
        orig = SL._read_frames
        SL._read_frames = lambda path, idx, size: [__import__("numpy").full((8, 8, 3), 128, "uint8") for _ in idx]
        try:
            _, a, _ = SL.BFSShotJoin()._join_all(imgs, shots, [0], [runs_only], tl)
            # a soundtrack from elsewhere is still used as given
            _, other, _ = SL.BFSShotJoin()._join_all(imgs, shots, [0], [dict(runs_only, bfs_runs_only=False)], tl)
        finally:
            SL._read_frames = orig
        wf = a["waveform"][0, 0]
        self.assertEqual(wf.shape[0], 600)
        self.assertEqual([float(wf[i * 10]) for i in (0, 25, 30, 45, 59)], [0.0, 25.0, 30.0, 45.0, 59.0])  # in sync, skipped shot not silent
        self.assertNotIn("bfs_runs_only", a)
        self.assertEqual(float(other["waveform"][0, 0, 0]), -1.0)


class CastTest(unittest.TestCase):
    # person 0 on frames 0-47 (big face), person 1 on 48-95, both small on 96-119
    CAST = {"step": 6, "people": [{"id": 0}, {"id": 1}], "samples":
            [{"f": f, "faces": [{"pid": 0, "area": 0.1}]} for f in range(0, 48, 6)]
            + [{"f": f, "faces": [{"pid": 1, "area": 0.1}]} for f in range(48, 96, 6)]
            + [{"f": f, "faces": []} for f in range(96, 120, 6)]}

    def test_shot_people_and_main(self):
        self.assertEqual(SL.shot_people(self.CAST, 0, 60), {"people": [0, 1], "main": 0})
        self.assertEqual(SL.shot_people(self.CAST, 96, 120), {"people": [], "main": -1})

    def test_change_points_need_a_long_enough_run(self):
        self.assertEqual(SL.person_change_points(self.CAST, 0, 120, 24), [48])
        self.assertEqual(SL.person_change_points(self.CAST, 0, 60, 24), [])

    def test_plan_splits_assigns_and_skips_by_person(self):
        path = str(Path(_TMP) / "cast.mp4"); Path(path).write_bytes(b"x")
        a = {"fps": 24.0, "n": 120, "score": [0.0] * 120, "raw": [0.0] * 120}
        SL._CAST_CACHE[SL._cast_key(path, a)] = self.CAST
        orig = SL.find_cuts
        SL.find_cuts = lambda *args, **kw: ([], "test")
        try:
            plan = dict(SL.DEFAULT_PLAN, max_s=10, min_s=0.5, cast={"0": {"ref": "a.png"}}, cast_split=True, cast_only=True)
            segs = SL.apply_filters(plan, a, path, SL.resolve_plan(plan, a, path))
        finally:
            SL.find_cuts = orig
        self.assertEqual([(s["start"], s["end"]) for s in segs], [(0, 48), (48, 120)])
        self.assertEqual(segs[0]["ref"], "a.png")
        self.assertTrue(segs[0]["run"])
        self.assertEqual(segs[1]["skip_reason"], "no linked person")


class ComparisonTest(unittest.TestCase):
    def test_join_returns_a_labelled_side_by_side(self):
        shots = [dict(_shot(0, 0, 20, gen=22), frames=torch.zeros(22, 32, 48, 3), ref=torch.ones(1, 40, 30, 3),
                      ref2=None, prompt="a test prompt"),
                 dict(_shot(1, 20, 40, cut=True, gen=22), frames=torch.zeros(22, 32, 48, 3), ref=None, ref2=None,
                      prompt="")]
        imgs = [torch.full((22, 32, 48, 3), 0.5), torch.full((22, 32, 48, 3), 0.7)]
        video, _, _, comp, _, guide = SL.BFSShotJoin().join(imgs, shots, [0], comparison=[True], label=["steps 20"])
        self.assertEqual(video.shape[0], 40)
        self.assertEqual(tuple(guide.shape), (40, 32, 48, 3))                   # the originals, frame for frame
        self.assertEqual(float(guide.abs().max()), 0.0)
        self.assertEqual(comp.shape[0], 40)
        self.assertEqual(comp.shape[2] % 16, 0)
        self.assertEqual(comp.shape[1] % 16, 0)
        self.assertGreaterEqual(comp.shape[2], 48 * 2 + 32)
        self.assertGreater(comp.shape[1], 32)
        off = SL.BFSShotJoin().join(imgs, shots, [0])
        self.assertEqual(off[3].shape[0], 1)
        self.assertIsInstance(off[5], _Blocker)                                 # no comparison: no originals kept
        self.assertIn("comparison", off[5].message)


class ContinuityTest(unittest.TestCase):
    def test_pick_frame_ignores_the_overlap(self):
        frames = torch.arange(30, dtype=torch.float32).view(30, 1, 1, 1).expand(30, 2, 2, 3)
        self.assertEqual(float(SL.pick_frame(frames, 20, "first")[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(SL.pick_frame(frames, 20, "middle")[0, 0, 0, 0]), 10.0)
        self.assertEqual(float(SL.pick_frame(frames, 20, "last")[0, 0, 0, 0]), 19.0)

    def test_chain_uses_the_previous_result_only_when_asked(self):
        frames = torch.arange(30, dtype=torch.float32).view(30, 1, 1, 1).expand(30, 2, 2, 3)
        SL.remember_result({"index": 0, "count": 2}, frames)
        shot = {"index": 1, "count": 2, "chain": "reference", "chain_frame": "last", "prev_length": 20}
        self.assertEqual(float(SL.chain_image(shot)[0, 0, 0, 0]), 19.0)
        self.assertIsNone(SL.chain_image(dict(shot, chain="off")))
        self.assertIsNone(SL.chain_image(dict(shot, index=0)))
        self.assertIsNone(SL.chain_image(dict(shot, index=3, count=4)))   # not the shot right after
        stored = torch.ones(1, 2, 2, 3)
        self.assertIs(SL.chain_image(dict(shot, chain_image=stored)), stored)

    def test_plan_carries_the_per_shot_setting(self):
        a = {"fps": 24.0, "n": 100, "score": [0.0] * 100, "raw": [0.0] * 100}
        plan = dict(SL.DEFAULT_PLAN, mode="fixed", max_s=2, bounds=[50],
                    segs=[{}, {"chain": "first frame", "chain_frame": "middle"}])
        segs = SL.resolve_plan(plan, a)
        self.assertEqual((segs[0]["chain"], segs[1]["chain"], segs[1]["chain_frame"]), ("head", "first frame", "middle"))   # head is the default


class MaskCropTest(unittest.TestCase):
    def test_spec_takes_target_from_shot_and_settings_from_global(self):
        spec = SL.mask_spec({"text": "person", "padding": 0.9}, {"padding": 0.3, "text": "ignored"})
        self.assertEqual((spec["text"], spec["padding"], spec["fill_holes"]), ("person", 0.3, True))

    def test_shape_fills_holes_and_holds_in_time(self):
        m = torch.zeros(5, 9, 9, dtype=torch.uint8)
        m[2, 2:7, 2:7] = 1
        m[2, 4, 4] = 0                                 # a hole
        out = SL.shape_mask(m, dict(SL.DEFAULT_MASK, temporal_expand=1))
        self.assertEqual(int(out[2, 4, 4]), 1)
        self.assertEqual(int(out[1, 3, 3]), 1)          # held one frame back
        self.assertEqual(int(out[0, 3, 3]), 0)
        blocks = SL.shape_mask(m, dict(SL.DEFAULT_MASK, temporal_expand=0, blockify=4))
        self.assertEqual(tuple(blocks.shape), (5, 9, 9))

    def test_box_is_the_union_plus_padding(self):
        m = torch.zeros(2, 10, 10, dtype=torch.uint8)
        m[0, 2:4, 2:4] = 1
        m[1, 6:8, 6:8] = 1
        self.assertEqual(SL.crop_box(m, 0.0), [0.2, 0.2, 0.8, 0.8])
        self.assertIsNone(SL.crop_box(torch.zeros(2, 10, 10), 0.1))

    def test_uncrop_pastes_only_inside_the_mask(self):
        full = torch.zeros(3, 20, 20, 3)
        mask = torch.zeros(3, 20, 20)
        mask[:, 5:15, 5:15] = 1
        shot = {"full_frames": full, "crop": {"box": [0.25, 0.25, 0.75, 0.75], "mask": mask, "paste": "mask",
                                              "expand": 0, "feather": 0}}
        out = SL.uncrop(torch.ones(3, 8, 8, 3), shot)
        self.assertEqual(tuple(out.shape), (3, 20, 20, 3))
        self.assertEqual(float(out[0, 10, 10, 0]), 1.0)
        self.assertEqual(float(out[0, 1, 1, 0]), 0.0)


class DuetPanelCropTest(unittest.TestCase):
    def test_join_side_cuts_the_panel_off(self):
        shot = {"width": 64, "height": 32, "panel": {"position": "left", "h": 2, "w": 4, "strip_h": 0, "strip_w": 4}}
        canvas = torch.zeros(3, 32, 128, 3)
        canvas[:, :, 64:] = 1.0
        out = SL.crop_panel(canvas, shot)
        self.assertEqual(tuple(out.shape), (3, 32, 64, 3))
        self.assertEqual(float(out.min()), 1.0)
        self.assertIs(SL.crop_panel(out, shot), out)            # already the video size
        self.assertIs(SL.crop_panel(canvas, {"width": 64}), canvas)  # no panel


class SilentAudioTest(unittest.TestCase):
    def test_join_without_audio_returns_a_silent_track(self):
        imgs = [torch.full((39, 8, 8, 3), 0.5)]
        video, a, fps = SL.BFSShotJoin()._join_all(imgs, [_shot(0, 0, 24, gen=39)], [0])
        self.assertEqual(tuple(a["waveform"].shape[:2]), (1, 2))
        self.assertEqual(a["waveform"].shape[-1], 44100)        # 24 frames at 24 fps = 1 s
        self.assertEqual(float(a["waveform"].abs().max()), 0.0)
        self.assertFalse(SL._usable({"waveform": torch.zeros(1, 2, 0), "sample_rate": 44100}))


class RepackTest(unittest.TestCase):
    def test_replaces_only_connected_pieces_and_fits_the_guide(self):
        shot = dict(_shot(0, 0, 20, gen=22), width=8, height=8, frames=torch.zeros(22, 8, 8, 3),
                    ref=torch.zeros(1, 4, 4, 3), ref2=None, prompt="old", audio=None)
        ref = torch.ones(1, 6, 6, 3)
        out = SL.BFSShotRepack().repack([shot], refs=[ref])[0][0]
        self.assertTrue(torch.equal(out["ref"], ref))
        self.assertEqual(out["prompt"], "old")
        self.assertIs(out["frames"], shot["frames"])
        out = SL.BFSShotRepack().repack([shot], guide_frames=[torch.ones(30, 16, 16, 3)], prompt=["new"])[0][0]
        self.assertEqual(tuple(out["frames"].shape), (22, 8, 8, 3))
        self.assertEqual(out["prompt"], "new")
        self.assertEqual(out["length"], 20)

    def test_unpack_lists_the_pictures_and_repack_puts_them_back_per_shot(self):
        a = [torch.full((1, 4, 4, 3), k / 10) for k in range(3)]
        base = dict(_shot(0, 0, 20, gen=22), width=8, height=8, frames=torch.zeros(22, 8, 8, 3), prompt="",
                    audio=None)
        s1, s2, s3 = dict(base, refs=a[:2]), dict(base, index=1, refs=[]), dict(base, index=2, refs=a[2:])
        flat = []
        for s in (s1, s2, s3):   # what ComfyUI hands the next node: the lists of every Unpack call, in order
            out = SL.BFSShotUnpack().unpack(s)
            flat += out[1]
            self.assertEqual(out[2], len(s["refs"]))
        edited = [x + 1 for x in flat]   # e.g. a background remover, once per picture
        s1n, s2n, s3n = SL.BFSShotRepack().repack([s1, s2, s3], refs=edited)[0]
        self.assertEqual([len(s1n["refs"]), len(s2n["refs"]), len(s3n["refs"])], [2, 0, 1])
        self.assertTrue(torch.equal(s3n["refs"][0], a[2] + 1) and torch.equal(s1n["ref2"], a[1] + 1))


class ConditioningPromptTest(unittest.TestCase):
    """BFS Shot H3 Conditioning passes the planner's prompt and adds the setting picture (no models)."""

    def setUp(self):
        self.seen = seen = {}

        class R2V:
            @staticmethod
            def execute(**kw):
                seen.update(kw)
                return types.SimpleNamespace(args=("pos", "lat"))

        mod = types.ModuleType("comfy_extras.nodes_minimax_h3")
        mod.MiniMaxH3ReferenceToVideo, mod.MiniMaxH3AddGuide = R2V, R2V
        self._old = {k: sys.modules.get(k) for k in ("comfy_extras", "comfy_extras.nodes_minimax_h3")}
        sys.modules["comfy_extras"] = types.ModuleType("comfy_extras")
        sys.modules["comfy_extras.nodes_minimax_h3"] = mod

    def tearDown(self):
        for k, v in self._old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def _shot(self):
        return dict(prompt="subject_definitions: <Subject 1> is the man in <Picture 1>.", width=448, height=800,
                    gen_length=22, frames=torch.full((22, 8, 8, 3), 0.5), ref=torch.zeros(1, 4, 4, 3), ref2=None,
                    audio=None)

    def test_the_planner_prompt_goes_as_it_is_and_the_model_passes_through(self):
        out = SL.BFSShotH3Conditioning().condition(self._shot(), "clip", "vae", "none", "none", "match", model="m")
        self.assertEqual(out[3], self._shot()["prompt"])
        self.assertEqual(self.seen["prompt"], out[3])
        self.assertEqual(out[2], "m")
        self.assertNotIn("duet", SL.BFSShotH3Conditioning.INPUT_TYPES()["optional"])

    def test_setting_picture_is_the_last_reference_with_static(self):
        mask = torch.zeros(22, 8, 8)
        mask[:, 2:4, 2:4] = 1
        out = SL.BFSShotH3Conditioning().condition(self._shot(), "clip", "vae", "none", "none", "match",
                                                    setting_ref="on (generation size)", setting_mask=mask)
        refs = list(self.seen["ref_images"].values())
        self.assertEqual(len(refs), 2)
        pic = refs[-1][0]
        self.assertTrue(((pic[3, 3] == 0) | (pic[3, 3] == 1)).all())      # static inside the (grown) mask
        self.assertTrue(torch.allclose(pic[7, 7], torch.tensor(0.5)))   # the place outside it
        self.assertIn("<Picture 2> shows the setting", out[3])
        self.assertEqual(SL.add_setting("a {setting} b", 3, True), "a <Picture 3> b")


class SimpleConditioningTest(unittest.TestCase):
    def test_exposes_only_the_common_controls(self):
        inputs = SL.BFSShotH3ConditioningSimple.INPUT_TYPES()
        names = set(inputs["required"]) | set(inputs["optional"])
        self.assertIn("with_audio", names)
        self.assertIn("inpaint", names)
        self.assertNotIn("duet", names)
        self.assertNotIn("model", names)
        self.assertNotIn("rope_gap", names)
        self.assertNotIn("task", names)

    def test_forwards_safe_advanced_defaults_and_hides_model_output(self):
        seen = {}
        original = SL.BFSShotH3Conditioning.condition

        def fake(_self, *args, **kwargs):
            seen["args"], seen["kwargs"] = args, kwargs
            return "positive", "latent", "unchanged model", "prompt"

        SL.BFSShotH3Conditioning.condition = fake
        try:
            out = SL.BFSShotH3ConditioningSimple().condition(
                "shot", "clip", "vae", SL.BFSShotH3ConditioningSimple.GUIDE_MODES[0], "match",
                audio_vae="audio vae", with_audio=True)
        finally:
            SL.BFSShotH3Conditioning.condition = original

        self.assertEqual(out, ("positive", "latent", "prompt"))
        self.assertEqual(seen["args"][4], "none")
        self.assertTrue(seen["kwargs"]["with_audio"])
        self.assertEqual(seen["kwargs"]["inpaint"], SL.INPAINT_MODES[0])


class TargetTest(unittest.TestCase):
    def test_target_placeholder(self):
        self.assertEqual(SL.fill_target("Replace {target} with <Subject 1>", "the man in a red coat"),
                         "Replace the man in a red coat with <Subject 1>")
        self.assertEqual(SL.fill_target("Replace {target} with <Subject 1>", ""), "Replace the person with <Subject 1>")
        self.assertEqual(SL.fill_target("no placeholder", "x"), "no placeholder")


class MaskOverlayTest(unittest.TestCase):
    def test_overlay_marks_mask_and_box(self):
        o = torch.zeros(3, 32, 32, 3)
        m = torch.zeros(3, 32, 32); m[:, 8:16, 8:16] = 1
        out = SL._mask_overlay(o, {"mask": m, "box": [0.1, 0.1, 0.9, 0.9]}, 3)
        self.assertGreater(float(out[0, 12, 12, 0]), 0.3)          # red inside the mask
        self.assertEqual(float(out[0, 20, 20].sum()), 0.0)          # untouched outside mask and box
        self.assertGreater(float(out[0, 3, 16, 1]), 0.5)            # yellow box edge


class StitchFinishTest(unittest.TestCase):
    def _shot(self):
        full = torch.full((4, 64, 64, 3), 0.5)
        mask = torch.zeros(4, 64, 64); mask[:, 20:44, 24:40] = 1
        return {"full_frames": full, "crop": {"box": [0.125, 0.125, 0.875, 0.875], "mask": mask, "paste": "mask",
                                              "expand": 0, "feather": 2}}

    def test_ring_colour_match_removes_drift_and_keeps_subject_contrast(self):
        shot = self._shot()
        gen = torch.full((4, 48, 48, 3), 0.62)          # the model brightened the whole crop (+0.12)
        gen[:, 12:36, 16:32] = 0.2                       # a darker new subject
        plain = SL.uncrop(gen, shot)
        fixed = SL.uncrop(gen, shot, match_colors=1.0)
        # around the subject the drift is gone, inside it the new subject stays darker than the scene
        self.assertGreater(abs(float(plain[0, 21, 23].mean()) - 0.5), abs(float(fixed[0, 21, 23].mean()) - 0.5))
        self.assertLess(float(fixed[0, 30, 30].mean()), 0.45)

    def test_edge_hardness(self):
        a = torch.tensor([[[0.1, 0.5, 0.9]]])
        h = SL._alpha_hardness(a, 1.0)
        self.assertEqual([round(float(x), 2) for x in h[0, 0]], [0.0, 0.5, 1.0])
        self.assertTrue(torch.equal(SL._alpha_hardness(a, 0.0), a))


class InpaintInCropTest(unittest.TestCase):
    def test_latent_starts_from_the_shot_and_only_the_mask_is_generated(self):
        sys.path.insert(0, str(ROOT))
        nested = sys.modules.get("comfy.nested_tensor")
        if nested is None:
            class _N:
                is_nested = True

                def __init__(self, t):
                    self.tensors = list(t)
            comfy = sys.modules.setdefault("comfy", types.ModuleType("comfy"))
            nested = types.ModuleType("comfy.nested_tensor"); nested.NestedTensor = _N
            comfy.nested_tensor = nested
            sys.modules["comfy.nested_tensor"] = nested
            sys.modules.setdefault("comfy.utils", types.ModuleType("comfy.utils"))

        class VAE:
            def encode(self, px):
                t = ((px.shape[0] - 5) // 17) * 5 + 2
                x = px.mean(-1)[None, None]
                return torch.nn.functional.interpolate(x, size=(t, px.shape[1] // 16, px.shape[2] // 16)).repeat(1, 24, 1, 1, 1)

        lat = {"samples": nested.NestedTensor((torch.zeros(1, 24, 7, 8, 8), torch.zeros(1, 32, 2, 37)))}
        frames = torch.full((22, 128, 128, 3), 0.4)
        mask = torch.zeros(22, 128, 128); mask[:, 48:80, 48:80] = 1
        pos, out = SL.inpaint_latent(lat, [], VAE(), frames, mask)
        video = out["samples"].tensors[0]
        self.assertAlmostEqual(float(video.mean()), 0.4, places=4)        # starts from the shot's own frames
        vm, am = out["noise_mask"].tensors
        self.assertEqual(tuple(vm.shape), (1, 1, 7, 8, 8))
        self.assertEqual(float(vm[0, 0, 3, 4, 4]), 1.0)                   # the person: generated
        self.assertEqual(float(vm[0, 0, 3, 0, 0]), 0.0)                   # the rest of the crop: kept
        self.assertEqual(float(vm[0, 0, 3, 2, 4]), 1.0)                   # grown by one latent cell
        self.assertTrue(bool((am == 1).all()))                            # audio fully generated
        _, out2 = SL.inpaint_latent(lat, [], VAE(), frames, mask, strength=0.8)
        vm2 = out2["noise_mask"].tensors[0]
        self.assertAlmostEqual(float(vm2[0, 0, 3, 4, 4]), 0.8, places=5)     # partly regenerated inside the mask
        self.assertEqual(float(vm2[0, 0, 3, 0, 0]), 0.0)                       # still kept outside


    def test_generation_mask_with_and_without_crop(self):
        m = torch.zeros(5, 64, 64); m[:, 20:40, 20:40] = 1
        cropped = SL.generation_mask({"crop": {"crop_mask": m, "expand": 4}, "frames": torch.zeros(5, 64, 64, 3)})
        self.assertEqual(float(cropped[0, 17, 30]), 1.0)                  # grown by expand
        self.assertEqual(float(cropped[0, 10, 30]), 0.0)
        with self.assertRaises(ValueError):
            SL.generation_mask({"crop": None, "mask_src": None, "frames": torch.zeros(5, 64, 64, 3)})
        orig = SL.shot_mask
        SL.shot_mask = lambda *a: {"masks": torch.nn.functional.interpolate(m[:, None], size=(32, 32))[:, 0].to(torch.uint8)}
        try:
            full = SL.generation_mask({"crop": None, "frames": torch.zeros(5, 64, 64, 3),
                                       "mask_src": {"path": "", "analysis": {}, "start": 0, "length": 5,
                                                    "spec": {"expand": 0}}})
        finally:
            SL.shot_mask = orig
        self.assertEqual(tuple(full.shape), (5, 64, 64))
        self.assertTrue(torch.equal(full, m))


    def test_comparison_overlay_for_mask_only_shots(self):
        m = torch.zeros(5, 32, 32); m[:, 8:24, 8:24] = 1
        ms = {"path": __file__, "analysis": {"fps": 24.0}, "start": 0, "length": 5, "spec": dict(SL.DEFAULT_MASK, expand=0)}
        shot = {"crop": None, "frames": torch.zeros(5, 64, 64, 3), "mask_src": ms, "inpaint": True}
        orig = SL.shot_mask
        SL.shot_mask = lambda *a: {"masks": m.to(torch.uint8)}
        try:
            ov = SL.overlay_of(shot)
            self.assertEqual(tuple(ov["mask"].shape), (5, 64, 64))
            self.assertIsNone(SL.overlay_of(dict(shot, inpaint=False)))      # not used and not segmented yet: no SAM 3
            self.assertIsNone(SL.overlay_of({"crop": None, "frames": shot["frames"]}))
        finally:
            SL.shot_mask = orig
        o = SL._mask_overlay(torch.full((5, 32, 32, 3), 0.5), ov, 5)
        self.assertGreater(float(o[0, 16, 16, 0]), float(o[0, 16, 16, 1]))     # red inside the mask
        self.assertAlmostEqual(float(o[0, 2, 2, 0]), 0.5)                        # untouched outside


class ExternalMaskTest(unittest.TestCase):
    def _video(self, name, n=48, fps=24.0, w=64, h=36, box=(10, 5, 30, 25)):
        import cv2
        import numpy as np
        path = str(Path(_TMP) / name)
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i in range(n):
            f = np.zeros((h, w, 3), np.uint8)
            x0, y0, x1, y1 = box
            f[y0:y1, x0 + i % 4:x1 + i % 4] = 255
            vw.write(f)
        vw.release()
        return path

    def setUp(self):
        self.an = {"fps": 24.0, "n": 48, "n_src": 48, "fps_src": 24.0, "width": 64, "height": 36}

    def test_precedence(self):
        p = {"mask_video": "global.mp4"}
        self.assertEqual(SL.plan_spec(p, {"video": "shot.mp4", "text": "man"})["ext"]["scope"], "shot")
        self.assertNotIn("ext", SL.plan_spec(p, {"text": "man"}))                  # the shot's own SAM 3 text wins
        self.assertEqual(SL.plan_spec(p, {})["ext"], {"file": "global.mp4", "scope": "video"})
        SL._EXT_MASK["v.mp4"] = torch.ones(48, 36, 64)
        try:
            self.assertEqual(SL.plan_spec(p, {}, "v.mp4")["ext"], {"tensor": "v.mp4"})   # the node input beats the panel file
        finally:
            SL._EXT_MASK.pop("v.mp4")
        self.assertFalse(SL.spec_has_mask(SL.plan_spec({}, {})))

    def test_file_and_tensor_masks_replace_sam(self):
        path = self._video("roto_whole.mp4")
        self._video("roto_shot.mp4", n=12)
        orig = SL.segment_frames
        SL.segment_frames = lambda *a, **k: self.fail("SAM 3 must not run with an external mask")
        try:
            r = SL.shot_mask(path, self.an, 24, 12, SL.plan_spec({"mask_video": "roto_whole.mp4"}, {}))
            self.assertEqual(r["masks"].shape[0], 12)
            self.assertTrue(r["box"] is not None)
            cov = float(r["masks"].float().mean())
            self.assertTrue(0.1 < cov < 0.4, cov)
            r2 = SL.shot_mask(path, self.an, 24, 12, SL.plan_spec({}, {"video": "roto_shot.mp4"}))
            self.assertEqual(r2["masks"].shape[0], 12)
            t = torch.zeros(48, 36, 64); t[:, 0:18, :] = 1                         # top half
            SL._EXT_MASK[path] = t
            r3 = SL.shot_mask(path, self.an, 0, 10, SL.plan_spec({}, {}, path))
            m = r3["masks"].float()
            self.assertGreater(float(m[:, :8].mean()), 0.9)
            self.assertLess(float(m[:, -8:].mean()), 0.1)
        finally:
            SL.segment_frames = orig
            SL._EXT_MASK.pop(path, None)


class FramePasteTest(unittest.TestCase):
    def test_paste_back_keeps_the_background_and_takes_the_new_outline(self):
        base = torch.full((5, 64, 64, 3), 0.2)                       # the original frames
        result = torch.full((5, 64, 64, 3), 0.9)                     # generated: everything changed
        old = torch.zeros(5, 64, 64); old[:, 20:44, 20:36] = 1       # the original person
        new = torch.zeros(5, 64, 64); new[:, 16:48, 18:44] = 1       # the new one is bigger
        ms = {"path": "", "analysis": {}, "start": 0, "length": 5, "spec": dict(SL.DEFAULT_MASK, expand=0, feather=0)}
        shot = {"index": 0, "frames": base, "mask_src": ms, "paste": True}
        o1, o2 = SL.shot_mask, SL.segment_frames
        SL.shot_mask = lambda *a: {"masks": old.to(torch.uint8)}
        SL.segment_frames = lambda imgs, sp: torch.nn.functional.interpolate(new[:, None], size=imgs.shape[1:3])[:, 0]
        try:
            out = SL.paste_back(result, shot, True)
            only_old = SL.paste_back(result, shot, False)
        finally:
            SL.shot_mask, SL.segment_frames = o1, o2
        self.assertAlmostEqual(float(out[0, 2, 2, 0]), 0.2, places=4)      # background: the original
        self.assertAlmostEqual(float(out[0, 30, 28, 0]), 0.9, places=4)    # the person: generated
        self.assertAlmostEqual(float(out[0, 30, 41, 0]), 0.9, places=4)    # the bigger new outline is pasted too
        self.assertAlmostEqual(float(only_old[0, 30, 41, 0]), 0.2, places=4)


class MaskGuideTest(unittest.TestCase):
    def test_masked_only_greys_everything_outside_the_mask(self):
        fr = torch.full((4, 32, 32, 3), 0.9)
        m = torch.zeros(4, 16, 16); m[:, 4:12, 4:12] = 1
        g = SL.masked_only(fr, m)
        self.assertEqual(tuple(g.shape), (4, 32, 32, 3))
        self.assertAlmostEqual(float(g[0, 16, 16, 0]), 0.9)      # the masked region
        self.assertAlmostEqual(float(g[0, 2, 2, 0]), 0.5)        # grey elsewhere
        self.assertEqual(SL.MASK_GUIDES[0], "off")
        fr2 = torch.rand(3, 64, 64, 3)
        m2 = torch.zeros(3, 64, 64); m2[:, 10:50, 10:50] = 1
        old_det = dict(SL._DET); SL._DET["pose"] = None                  # no YOLO pose model: a clear error
        try:
            with self.assertRaisesRegex(ValueError, "YOLO pose model"):
                SL.masked_only(fr2, m2, look="pose (people)")
        finally:
            SL._DET.clear(); SL._DET.update(old_det)
        for look in [x for x in SL.MASK_GUIDE_LOOKS if not x.startswith("pose")]:
            g2 = SL.masked_only(fr2, m2, look=look)
            self.assertEqual(tuple(g2.shape), (3, 64, 64, 3), look)
            self.assertAlmostEqual(float(g2[0, 2, 2, 0]), 0.5, msg=look)                 # grey outside, every look
            if look != "colour":                                                       # no colour inside
                self.assertLess(float((g2[..., 0] - g2[..., 1]).abs().max()), 1e-5, look)
        t = SL.add_mask_video("x subject_definitions: <Subject 1> is ...", 2)
        self.assertIn("<Video 2> shows only the region", t)
        self.assertEqual(SL.add_mask_video("follow {mask_video}", 1), "follow <Video 1>")


class ConditionFlowTest(unittest.TestCase):
    """BFS Shot H3 Conditioning with the native H3 nodes replaced by recorders."""

    def setUp(self):
        sys.path.insert(0, str(ROOT))
        calls = self.calls = {"r2v": [], "guides": []}

        class Out:
            def __init__(self, *a):
                self.args = a

        class R2V:
            @staticmethod
            def execute(**kw):
                calls["r2v"].append(kw)
                N = sys.modules["comfy.nested_tensor"].NestedTensor
                return Out([[torch.zeros(1), {}]], {"samples": N((torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 37)))})

        class Guide:
            @staticmethod
            def execute(positive, latent, frame_idx, vae, image=None, **kw):
                calls["guides"].append(image)
                return Out(positive)

        mod = types.ModuleType("comfy_extras.nodes_minimax_h3")
        mod.MiniMaxH3ReferenceToVideo, mod.MiniMaxH3AddGuide = R2V, Guide
        sys.modules.setdefault("comfy_extras", types.ModuleType("comfy_extras"))
        self._old = sys.modules.get("comfy_extras.nodes_minimax_h3")
        sys.modules["comfy_extras.nodes_minimax_h3"] = mod
        if "comfy.nested_tensor" not in sys.modules:
            class _N:
                is_nested = True

                def __init__(self, t):
                    self.tensors = list(t)
            comfy = sys.modules.setdefault("comfy", types.ModuleType("comfy"))
            nt = types.ModuleType("comfy.nested_tensor"); nt.NestedTensor = _N
            comfy.nested_tensor = nt; sys.modules["comfy.nested_tensor"] = nt

    def tearDown(self):
        if self._old is not None:
            sys.modules["comfy_extras.nodes_minimax_h3"] = self._old
        else:
            sys.modules.pop("comfy_extras.nodes_minimax_h3", None)

    def _shot(self, **kw):
        m = torch.zeros(22, 64, 64); m[:, 16:48, 16:40] = 1
        shot = {"index": 0, "count": 1, "frames": torch.full((22, 64, 64, 3), 0.7), "ref": torch.zeros(1, 32, 32, 3),
                "ref2": None, "prompt": "subject_definitions: <Subject 1> is ...", "width": 64, "height": 64,
                "gen_length": 22, "audio": None, "crop": {"crop_mask": m, "expand": 0} if kw.get("crop") else None,
                "inpaint": kw.get("inpaint", True)}
        if not kw.get("crop"):
            shot["mask_src"] = {"path": __file__, "analysis": {"fps": 24.0}, "start": 0, "length": 22, "spec": dict(SL.DEFAULT_MASK, expand=0)}
        return shot, m

    def run_cond(self, shot, m, **kw):
        class VAE:
            def encode(self, px):
                return torch.zeros(1, 24, 2, px.shape[1] // 16, px.shape[2] // 16)
        orig = SL.shot_mask
        SL.shot_mask = lambda *a: {"masks": m.to(torch.uint8)}
        try:
            return SL.BFSShotH3Conditioning().condition(shot, None, VAE(), SL.BFSShotH3Conditioning.GUIDE_MODES[0],
                                                        "none", "match", **kw)
        finally:
            SL.shot_mask = orig

    def test_reference_video_of_the_masked_region(self):
        shot, m = self._shot()
        self.run_cond(shot, m, inpaint=SL.INPAINT_MODES[0], mask_guide=SL.MASK_GUIDES[3], mask_ref_size="full",
                      mask_guide_look="colour")
        kw = self.calls["r2v"][0]
        v = kw["ref_videos"]["ref_video_1"]
        self.assertAlmostEqual(float(v[0, 2, 2, 0]), 0.5)          # grey outside the mask
        self.assertAlmostEqual(float(v[0, 30, 30, 0]), 0.7)        # the subject
        self.assertIn("<Video 1> shows only the region", kw["prompt"])
        self.assertEqual(len(self.calls["guides"]), 1)             # the normal aligned guide is still there
        self.calls["r2v"].clear()
        self.run_cond(shot, m, inpaint=SL.INPAINT_MODES[0], mask_guide=SL.MASK_GUIDES[3], mask_ref_size="1/2")
        self.assertEqual(tuple(self.calls["r2v"][0]["ref_videos"]["ref_video_1"].shape[1:3]), (32, 32))

    def test_extra_and_instead_guides(self):
        shot, m = self._shot()
        self.run_cond(shot, m, inpaint=SL.INPAINT_MODES[0], mask_guide=SL.MASK_GUIDES[1])
        self.assertEqual(len(self.calls["guides"]), 2)
        self.calls["guides"].clear()
        self.run_cond(shot, m, inpaint=SL.INPAINT_MODES[0], mask_guide=SL.MASK_GUIDES[2])
        self.assertEqual(len(self.calls["guides"]), 1)
        self.assertAlmostEqual(float(self.calls["guides"][0][0, 2, 2, 0]), 0.5)   # it is the masked one

    def test_passes_every_reference_in_order(self):
        refs = [torch.full((1, 32, 32, 3), k / 12) for k in range(12)]
        shot, _ = self._shot(inpaint=False)
        shot.pop("mask_src")
        shot.update(refs=refs, ref=refs[0], ref2=refs[1], ref_names=[f"r{k}.png" for k in range(12)],
                    prompt="<Picture 12> is the {details@Picture 12}.")
        out = SL.BFSShotH3Conditioning().condition(shot, None, None, SL.BFSShotH3Conditioning.GUIDE_MODES[3],
                                                   "none", "match", inpaint="off")
        kw = self.calls["r2v"][0]
        self.assertEqual(list(kw["ref_images"]), [f"ref_image_{k}" for k in range(12)])
        self.assertIs(kw["ref_images"]["ref_image_11"], refs[11])
        self.assertEqual(out[3], "<Picture 12> is the character.")   # no VLM and no saved description
        self.assertNotIn("use_ref_2", SL.BFSShotH3Conditioning.INPUT_TYPES()["required"])
        self.assertNotIn("use_ref_2", SL.BFSShotH3ConditioningSimple.INPUT_TYPES()["required"])

    def test_every_marked_person_in_their_picture_color(self):
        refs = [torch.zeros(1, 32, 32, 3) for _ in range(3)]
        shot, m = self._shot(inpaint=False)
        marks = [{"picture": 1, "color": "#00ff00"}, {"picture": 3, "color": "#ff2020", "text": "the man", "points": [], "key": 0}]
        shot.update(refs=refs, color_guide=True, color_guide_marks=marks, color_guide_color="#00ff00", color_guide_opacity=0.5)
        second = torch.zeros(22, 64, 64); second[:, 0:10, 0:10] = 1
        orig = SL.shot_mask
        SL.shot_mask = lambda path, a, start, length, spec: {"masks": second if spec.get("text") == "the man" else m.to(torch.uint8)}
        try:
            SL.BFSShotH3Conditioning().condition(shot, None, None, SL.BFSShotH3Conditioning.GUIDE_MODES[0],
                                                 "none", "match", inpaint="off")
            guide = self.calls["guides"][0]
            self.assertGreater(float(guide[0, 5, 5, 0]), 0.75)    # the second person, in red
            self.assertGreater(float(guide[0, 30, 30, 1]), 0.75)  # the selection, in green
            with self.assertRaisesRegex(ValueError, "Picture 3"):
                SL.BFSShotH3Conditioning().condition(dict(shot, refs=refs[:2]), None, None,
                                                     SL.BFSShotH3Conditioning.GUIDE_MODES[0], "none", "match", inpaint="off")
        finally:
            SL.shot_mask = orig

    def test_ignored_outside_mask_only(self):
        shot, m = self._shot(inpaint=False)
        self.run_cond(shot, m, inpaint=SL.INPAINT_MODES[0], mask_guide=SL.MASK_GUIDES[3])
        self.assertNotIn("ref_videos", self.calls["r2v"][0])
        shot, m = self._shot(crop=True)
        self.calls["r2v"].clear()
        self.run_cond(shot, m, inpaint=SL.INPAINT_MODES[0], mask_guide=SL.MASK_GUIDES[3])
        self.assertNotIn("ref_videos", self.calls["r2v"][0])


class MultiRefTest(unittest.TestCase):
    """Up to 12 reference pictures per shot (plans before 1.113 had ref / ref2)."""

    def test_old_plans_become_a_list_and_keep_ref_and_ref2(self):
        p = SL._load_plan('{"global_ref": "a.png", "global_ref2": "b.png"}')
        self.assertEqual(p["global_refs"], ["a.png", "b.png"])
        p = SL._load_plan('{"global_refs": ["a.png", "", "c.png", ""], "global_ref": "stale.png"}')
        self.assertEqual(p["global_refs"], ["a.png", "", "c.png"])
        self.assertEqual((p["global_ref"], p["global_ref2"]), ("a.png", ""))
        self.assertEqual(len(SL.ref_list([f"{k}.png" for k in range(20)])), SL.MAX_REFS)

    def test_a_shot_slot_falls_back_to_the_global_one_and_gaps_close(self):
        self.assertEqual(SL.shot_ref_names(["", "x.png"], ["a.png", "b.png", "c.png"]), ["a.png", "x.png", "c.png"])
        self.assertEqual(SL.shot_ref_names(["", "", "z.png"], ["a.png"]), ["a.png", "z.png"])

    def test_signature_of_a_two_reference_shot_is_unchanged(self):
        seg = {"start": 0, "end": 10, "ref": "a.png", "ref2": "b.png", "prompt": ""}
        plan = dict(SL.DEFAULT_PLAN, global_ref="g.png")
        old = SL.shot_signature(plan, seg, 0, {})
        new = SL.shot_signature(dict(plan, global_refs=["g.png"]), dict(seg, refs=["a.png", "b.png"]), 0, {})
        self.assertEqual(old, new)
        three = SL.shot_signature(dict(plan, global_refs=["g.png"]), dict(seg, refs=["a.png", "b.png", "c.png"]), 0, {})
        self.assertNotEqual(old, three)

    def test_picture_details(self):
        names, imgs = ["a.png", "b.png"], [torch.zeros(1, 4, 4, 3)] * 2
        text = "<Subject 1> is the {details@Picture 2} shown in <Picture 2>, {details@Picture1}."
        out = SL.fill_picture_details(text, names, imgs, {"b.png": "the tall man"})
        self.assertEqual(out, "<Subject 1> is the tall man shown in <Picture 2>, character.")
        kept = SL.fill_picture_details(text, names, imgs, {"b.png": "tall man"}, keep_missing=True)
        self.assertIn("{details@Picture1}", kept)
        self.assertEqual(SL.fill_picture_details("{details@Picture 5}", names, imgs, {}), "character")
        self.assertEqual(SL.prompt_slot(text, "{details@Picture 2}"),
                         "<Subject 1> is the ___ shown in <Picture 2>, the subject of <Picture 1>.")
        self.assertEqual(SL.picture_marker(text, 1), "{details@Picture1}")
        self.assertEqual(SL.expand_two_ref_prompt(text), text)   # the prompt already says what <Picture 2> is

    def test_details_and_details2_are_pictures_1_and_2(self):
        self.assertEqual(SL.picture_markers("{details} and {details2}"), "{details@Picture 1} and {details@Picture 2}")
        old = {"ref_details": {"a.png|b.png": {"details": "goblin", "details2": "robot"}, "c.png|": "cat"},
               "picture_details": {"b.png": "tall robot"}}
        self.assertEqual(SL.saved_picture_details(old), {"a.png": "goblin", "b.png": "tall robot", "c.png": "cat"})

    def test_a_picture_can_have_its_own_preset(self):
        cfg = SL.vlm_cfg({"describe_preset": "short"})
        plan = {"picture_cfg": {"a.png": {"preset": "custom", "custom": "Describe the outfit only."}}}
        self.assertEqual(SL.picture_cfg(cfg, plan, "a.png")["describe_preset"], "custom")
        self.assertIn("outfit only", SL.describe_instruction(SL.picture_cfg(cfg, plan, "a.png")))
        self.assertIs(SL.picture_cfg(cfg, plan, "b.png"), cfg)

    def test_colored_mark_regions_follow_the_pictures(self):
        legacy = {"global_color_guide_ref1": "#168cff", "global_color_guide_ref2": "#9b30ff"}
        marks = SL.color_marks(legacy, {"color_guide_text2": "the man", "color_guide_key2": 4})
        self.assertEqual([(x["picture"], x["color"]) for x in marks], [(1, "#168cff"), (2, "#9b30ff")])
        self.assertEqual(marks[1]["text"], "the man")
        plan = {"picture_colors": ["#00ff00"]}
        marks = SL.color_marks(plan, {"color_guide_picture": 2, "color_guide_regions": [
            {"picture": 5, "text": "the dog"}, {"picture": 3}]})   # a region with neither text nor points is ignored
        self.assertEqual([(x["picture"], x["color"]) for x in marks], [(2, "#9b30ff"), (5, "#ffe600")])
        text = SL.color_guide_prompt("A scene.", marks)
        self.assertIn("marked in purple becomes the character from <Picture 2>", text)
        self.assertIn("marked in yellow becomes the character from <Picture 5>", text)
        one = SL.color_guide_prompt("", marks[:1])
        self.assertIn("Replace only the person marked in purple with the character from <Picture 2>", one)

    def test_auto_prompt_asks_with_the_official_format(self):
        seen = {}

        def fake(clip, prompt, images, max_tokens, *a, **k):
            seen.update(prompt=prompt, n=images.shape[0], tokens=max_tokens)
            return "```\n**subject_definitions:**\n<Subject 1> is the robot in <Picture 2>.\nsummary:\n[video editing] x```"
        old, SL.vlm_generate = SL.vlm_generate, fake
        try:
            seg = {"color_guide": True, "color_guide_marks": [{"picture": 2, "color": "#9b30ff"}], "mask": {"text": "the man"},
                   "inpaint": True}
            frame, pic = torch.zeros(1, 8, 8, 3), torch.zeros(1, 4, 4, 3)
            out = SL.write_h3_prompt(None, [frame] * 3, [0, 0.5, 1.0], [pic, pic], ["", "a silver robot"],
                                     "swap the man for the robot", seg, "video", 1.5)
        finally:
            SL.vlm_generate = old
        self.assertEqual(seen["n"], 5)                       # 3 frames + 2 pictures
        for want in ("swap the man for the robot", "<Picture 2>: image 5, described as: a silver robot",
                     "tinted purple becomes the subject from <Picture 2>", "Only the selected subject is generated again",
                     "edited version of <Video 1>", "retention_analysis:", "non_diegetic_music:", "00:00.500",
                     "Never write a timestamp inside a shot", "is replaced by <Subject N>"):
            self.assertIn(want, seen["prompt"])
        self.assertTrue(out.startswith("subject_definitions:\n<Subject 1>"))   # fences and bold removed
        self.assertIn("marked in purple", out)                # the mapping the writer left out is added
        seg2 = dict(seg, prompt="")
        self.assertEqual(SL.auto_prompt_for("x", ["a.png"], seg2), SL.auto_prompt_for("x", ["a.png"], dict(seg2)))
        self.assertNotEqual(SL.auto_prompt_for("x", ["a.png"], seg2), SL.auto_prompt_for("y", ["a.png"], seg2))
        self.assertEqual(SL.auto_frame_count(24, 24.0), 3)
        self.assertEqual(SL.auto_frame_count(240, 24.0), 8)

    def test_a_template_request_keeps_its_trigger_and_wording(self):
        seen = {}

        def fake(clip, prompt, images, max_tokens, *a, **k):
            seen["prompt"] = prompt
            return "subject_definitions: <Subject 1> is the woman in Picture 1. detailed_description: [Shot 1] She walks."
        old, SL.vlm_generate = SL.vlm_generate, fake
        template = ("body_swap: subject_definitions: <Subject 1> is the {details@Picture 1} shown in <Picture 1>. "
                    "summary: [video generation + reference] Replace the guide subject. detailed_description: [Shot 1] x")
        try:
            out = SL.write_h3_prompt(None, [torch.zeros(1, 8, 8, 3)], [0.0], [torch.zeros(1, 4, 4, 3)], [""], template,
                                     {}, "guide")
        finally:
            SL.vlm_generate = old
        self.assertIn("prompt TEMPLATE", seen["prompt"])
        self.assertIn("'body_swap'", seen["prompt"])
        self.assertIn("stands for the description of <Picture n>", seen["prompt"])
        self.assertTrue(out.startswith("body_swap: subject_definitions:"))        # the trigger the writer dropped
        self.assertIn("the woman in <Picture 1>", out)                            # brackets put back
        self.assertEqual(SL.request_trigger("make it rain: now"), "")
        self.assertEqual(SL.request_trigger("head_swap: subject_definitions: x"), "head_swap")
        self.assertFalse(SL.is_template("replace the bottle with the can"))

    def test_key_frames_cover_every_piece_and_the_cuts_reach_the_writer(self):
        self.assertEqual(SL.key_frames(0, 100, [], 4), [12, 38, 62, 88])
        frames = SL.key_frames(0, 300, [43, 86, 129, 172, 215, 258], 3)    # a montage: 7 pieces, one frame each at least
        self.assertEqual(len(frames), 7)
        self.assertTrue(all(any(a <= f < b for f in frames) for a, b in zip([0, 43, 86, 129, 172, 215, 258], [43, 86, 129, 172, 215, 258, 300])))
        seen = {}

        def fake(clip, prompt, images, max_tokens, *a, **k):
            seen["prompt"] = prompt
            return ("subject_definitions:\n<Subject 1> is x.\ndetailed_description:\n[Shot 1] At 00:00.000, a woman stands. "
                    "At 00:01.792, she walks.")
        old, SL.vlm_generate = SL.vlm_generate, fake
        try:
            one = SL.write_h3_prompt(None, [torch.zeros(1, 8, 8, 3)], [0.0], [], [], "Troque a mulher pela <Picture 1>", {})
            two = SL.write_h3_prompt(None, [torch.zeros(1, 8, 8, 3)], [0.0], [], [], "swap", {}, cuts=[1.792],
                                     template="body_swap: subject_definitions: a. summary: b. detailed_description: [Shot 1] c")
        finally:
            SL.vlm_generate = old
        self.assertIn("[Shot 1] A woman stands. She walks.", one)            # no timestamps for one continuous shot
        self.assertIn('"[Shot 2] At 00:01.792, the shot cuts to ..."', seen["prompt"])
        self.assertIn("The user's request: swap", seen["prompt"])
        self.assertIn("The prompt template:\nbody_swap:", seen["prompt"])
        self.assertTrue(two.startswith("body_swap: "))
        self.assertIn("[Shot 1] A woman stands. At 00:01.792", two)

    def test_a_looping_answer_is_cleaned_and_the_labels_are_tied(self):
        loop = " The goblin looks at the cat, and the cat looks at the goblin. The goblin then looks up." * 6
        raw = ("subject_definitions:\n<Subject 1> is the gaunt, emaciated goblin with pale skin in <Picture 1>.\n"
               "<Subject 2> is the stout goblin with red skin in <Picture 2>.\nsummary:\n[video editing] x\n"
               "retention_analysis:\n<Subject 1>: fully_preserved - x\ndetailed_description:\n[Shot 1] A goblin types."
               + loop + " The gaunt goblin")
        self.assertIn("sentences are repeated in a loop", SL.h3_prompt_problems(raw, 2))
        t = SL.tidy_h3_prompt(raw, {}, "", False, 2)
        self.assertEqual(t.count("The goblin then looks up."), 1)
        self.assertNotIn("The gaunt goblin\n", t)                     # the sentence cut in the middle
        self.assertIn("overall_soundscape:", t)
        self.assertIn("non_diegetic_music:\nN/A", t)
        named = SL.name_subjects(t)
        self.assertIn("[Shot 1] <Subject 1> is the gaunt. <Subject 2> is the stout goblin with red skin.", named)
        self.assertEqual(SL.h3_prompt_problems(named, 2), [])

    def test_caption_style_asks_for_the_training_captions_shape(self):
        seen = {}

        def fake(clip, prompt, images, max_tokens, *a, **k):
            seen.update(prompt=prompt, tokens=max_tokens)
            return ('"Replace the dog in the video with the gaunt goblin from Picture 1. Replace the cat in the video with '
                    'the stout goblin from <Picture 2>. Keep the same background and location as in the video, and maintain '
                    'the motion, camera angle, and timing."')
        old, SL.vlm_generate = SL.vlm_generate, fake
        try:
            out = SL.write_h3_prompt(None, [torch.zeros(1, 8, 8, 3)], [0.0], [torch.zeros(1, 4, 4, 3)] * 2, ["", ""],
                                     "Troque o cão pelo <Picture 1> e o gato pelo <Picture 2>", {}, style="captions")
        finally:
            SL.vlm_generate = old
        self.assertIn("short style of a video-edit request", seen["prompt"])
        self.assertNotIn("full-reference format", seen["prompt"])
        self.assertEqual(seen["tokens"], 512)
        self.assertTrue(out.startswith("Replace the dog in the video with the gaunt goblin from <Picture 1>."))
        self.assertEqual(SL.caption_problems(out, 2), [])
        self.assertTrue(SL.caption_problems("subject_definitions: <Subject 1> is x", 1))

    def test_a_written_prompt_is_part_of_the_shot_signature_only_with_auto_prompt_on(self):
        seg = {"start": 0, "end": 10, "ref": "", "ref2": "", "prompt": "make it rain"}
        plan = dict(SL.DEFAULT_PLAN, auto_prompts={"0-10": {"text": "A"}})
        base = SL.shot_signature(plan, seg, 0, {})
        self.assertEqual(base, SL.shot_signature(dict(SL.DEFAULT_PLAN), seg, 0, {}))
        on = dict(plan, auto_prompt={"enabled": True})
        self.assertNotEqual(SL.shot_signature(on, seg, 0, {}),
                            SL.shot_signature(dict(on, auto_prompts={"0-10": {"text": "B"}}), seg, 0, {}))

    def test_repack_replaces_the_first_two_and_keeps_the_rest(self):
        refs = [torch.zeros(1, 4, 4, 3) for _ in range(3)]
        new = torch.ones(1, 4, 4, 3)
        out = SL.BFSShotRepack().repack([dict(_shot(0, 0, 20), refs=list(refs), ref=refs[0], ref2=refs[1])],
                                        ref_image_2=[new])[0][0]
        self.assertTrue(torch.equal(out["refs"][1], new) and torch.equal(out["ref2"], new))
        self.assertIs(out["refs"][2], refs[2])

    def test_the_planner_has_no_reference_inputs(self):
        self.assertNotIn("ref_image", SL.BFSShotPlanner.INPUT_TYPES()["optional"])


class DescribeShortTest(unittest.TestCase):
    def test_short_details_fit_inside_the_prompt(self):
        self.assertEqual(SL.tidy_details('"The green goblin in a torn leather vest."\nextra', {"describe_preset": "short"}),
                         "green goblin in a torn leather vest")
        self.assertEqual(SL.tidy_details("A young woman.", {"describe_preset": "full body"}), "A young woman.")
        self.assertIn("one English noun phrase", SL.describe_instruction({}))

    def test_every_preset_removes_vlm_answer_labels(self):
        for preset in SL.DESCRIBE_PRESETS:
            with self.subTest(preset=preset):
                self.assertEqual(SL.tidy_details("**Answer:** elderly woman with large glasses", {"describe_preset": preset}),
                                 "elderly woman with large glasses")

    def test_short_mode_honours_its_twenty_word_contract(self):
        text = "Answer: The " + " ".join(f"word{i}" for i in range(30))
        out = SL.tidy_details(text, {"describe_preset": "short"})
        self.assertEqual(len(out.split()), 20)
        self.assertFalse(out.lower().startswith(("answer", "the ")))

    def test_long_modes_are_bounded_but_not_collapsed_to_short(self):
        text = " ".join(f"word{i}" for i in range(170))
        full = SL.tidy_details(text, {"describe_preset": "full body"})
        face = SL.tidy_details(text, {"describe_preset": "head / face"})
        self.assertEqual(len(full.split()), 150)
        self.assertEqual(len(face.split()), 110)
        self.assertTrue(full.endswith("."))


class FitPromptTest(unittest.TestCase):
    P = ("body_swap:  subject_definitions: <Subject 1> is the {details} shown in <Picture 1>, whose identity define the "
         "character. The guide video provides the scene; only {target} is replaced: their face become <Subject 1>.  "
         "summary: Replace {target} in the guide video with <Subject 1>.")

    def test_slot_is_the_sentence_with_a_blank(self):
        d = SL.prompt_slot(self.P, "{details}")
        self.assertIn("<Subject 1> is the ___ shown in <Picture 1>", d)
        self.assertNotIn("{target}", d)
        t = SL.prompt_slot(self.P, "{target}")
        self.assertIn("only ___ is replaced", t)
        self.assertEqual(SL.prompt_slot("no marker here", "{details}"), "")

    def test_answer_matches_the_blank(self):
        d = SL.prompt_slot(self.P, "{details}")
        self.assertEqual(SL.tidy_slot("A golden retriever with a thick coat.", d), "golden retriever with a thick coat")
        t = SL.prompt_slot(self.P, "{target}")
        self.assertEqual(SL.tidy_slot("the young woman in a black corset.", t), "the young woman in a black corset")
        self.assertIn("goes into the blank", SL.describe_instruction({"describe_preset": "short"}, d))
        self.assertNotIn("goes into the blank", SL.describe_instruction({"describe_preset": "short", "fit_prompt": False}, d))
        self.assertNotIn("goes into the blank", SL.describe_instruction({"describe_preset": "full body"}, d))
        # the VLM echoing the sentence (the bug the user hit)
        echo = ("humanoid robot is the ___ shown in <Picture 1>, whose identity define the character. The robot has a sleek "
                "body with a metallic silver torso.")
        self.assertEqual(SL.tidy_slot(echo, d), "humanoid robot")
        self.assertEqual(SL.tidy_slot("sleek silver humanoid robot with black joints shown in <Picture 1>", d),
                         "sleek silver humanoid robot with black joints")
        self.assertEqual(SL.tidy_slot("<Subject 1> is the green goblin in a vest", d), "green goblin in a vest")

    def test_target_label_is_removed_even_without_a_prompt_slot(self):
        old = SL.vlm_generate
        SL.vlm_generate = lambda *args, **kwargs: "Answer: elderly woman with large tortoiseshell glasses"
        try:
            crop = __import__("numpy").zeros((8, 8, 3), dtype="uint8")
            self.assertEqual(SL.describe_target(object(), crop),
                             "the elderly woman with large tortoiseshell glasses")
        finally:
            SL.vlm_generate = old


class RangeTest(unittest.TestCase):
    def test_start_end_and_max_total_from_the_start(self):
        self.assertEqual(SL.plan_range(1000, 24.0, 10, 20), (240, 480))
        self.assertEqual(SL.plan_range(1000, 24.0, 10, 0), (240, 1000))
        self.assertEqual(SL.plan_range(1000, 24.0, 10, 0, 5), (240, 360))       # max total counted from the start
        self.assertEqual(SL.plan_range(1000, 24.0, 0, 0, 0, start_f=250, end_f=500), (250, 500))   # frames win
        self.assertEqual(SL.plan_range(1000, 24.0, 0, 0, 0, start_f=250, end_f=0), (250, 1000))
        segs = SL.plan_segments(480, [100, 300], "shots", 107, 10, start=240)
        self.assertEqual(segs[0]["start"], 240)
        self.assertEqual(segs[-1]["end"], 480)
        self.assertTrue(all(240 <= s["start"] < s["end"] <= 480 for s in segs))
        self.assertTrue(any(s["start"] == 300 for s in segs))                    # the cut inside the range is kept
        fixed = SL.plan_segments(480, [], "fixed", 107, 10, start=240)
        self.assertEqual((fixed[0]["start"], fixed[-1]["end"]), (240, 480))
        manual = SL.plan_segments(480, [], "manual", 107, 10, manual=[100, 350], start=240)
        self.assertEqual([(s["start"], s["end"]) for s in manual], [(240, 350), (350, 480)])


class TrackModeTest(unittest.TestCase):
    def test_track_is_used_by_shots_without_their_own_selection_and_sliced_per_shot(self):
        p = {"track": {"points": [{"x": 0.5, "y": 0.5, "label": 1}], "frame": 120, "text": "woman"}, "fps": 24}
        self.assertIn("track", SL.plan_spec(p, {}, "v.mp4")["ext"])
        self.assertNotIn("ext", SL.plan_spec(p, {"text": "man"}, "v.mp4"))             # the shot's own text wins
        self.assertEqual(SL.plan_spec(p, {"video": "m.mp4"}, "v.mp4")["ext"]["scope"], "shot")
        whole = torch.zeros(200, 36, 64, dtype=torch.uint8)
        whole[100:150, 10:20, 10:20] = 1                                                # the subject only in frames 100-149
        orig = SL.track_video
        SL.track_video = lambda path, a, t: (whole, 50)                                 # the track covers frames 50-249
        try:
            spec = SL.plan_spec(p, {}, "v.mp4")
            an = {"n": 300, "fps": 24.0}
            m = SL.ext_masks(spec["ext"], an, 140, 20, 64, 36)                          # shot 140-159
            self.assertEqual(tuple(m.shape), (20, 36, 64))
            self.assertFalse(bool(m[0].any()))                                          # frame 140 = track index 90
            self.assertTrue(bool(m[-1].any()))                                          # frame 159 = index 109
        finally:
            SL.track_video = orig


class DuetPromptTest(unittest.TestCase):
    def test_fields_are_found_and_the_vlm_fills_them(self):
        tpl = ("subject_definitions:\n<Subject 1> is the woman from <Picture 1>: {skin, face}.\n<Subject 2> is the {room} "
               "from {setting}.\n\nsummary:\nx {layout}")
        self.assertEqual(SL.template_slots(tpl), ["skin, face", "room"])
        asked = []

        def gen(clip, q, images, max_tokens, temperature=0.0, seed=0):
            asked.append(q)
            if images is None:
                return tpl.replace("{skin, face}", "pale skin, oval face").replace("{room}", "kitchen")
            return '{"medium": "phone footage", "actions": "the person waves"}'

        old = SL.vlm_generate, SL.vlm_describe
        SL.vlm_generate, SL.vlm_describe = gen, lambda clip, ims, q, mt=320, temperature=0.0, seed=0: "the kitchen with white cabinets"
        try:
            fr = torch.rand(9, 32, 32, 3)
            out = SL.duet_prompt(object(), tpl, fr, [torch.rand(1, 32, 32, 3)], "character swap", "", 512,
                                 setting=torch.rand(1, 32, 32, 3))
            self.assertIn("pale skin, oval face", out); self.assertIn("{setting}", out); self.assertIn("{layout}", out)
            self.assertIn("the person waves", asked[-1])
            self.assertEqual(SL.duet_prompt(None, tpl, fr, [], "character swap", ""), tpl.strip())
            self.assertEqual(SL.duet_prompt(object(), "a full prompt", fr, [], "character swap", ""), "a full prompt")
            w = SL.duet_prompt(object(), "", fr, [torch.rand(1, 32, 32, 3)], "character swap", "", 512,
                               setting=torch.rand(1, 32, 32, 3))
            self.assertIn("<Subject 2> is the kitchen with white cabinets from {setting}.", w)
        finally:
            SL.vlm_generate, SL.vlm_describe = old


class ShotHistoryTest(unittest.TestCase):
    def test_history_is_taken_out_of_the_shot(self):
        h = torch.full((17, 8, 8, 3), 255, dtype=torch.uint8)
        shot = {"chain": "history", "index": 1, "count": 2, "history": h, "history_src": h}
        hist, src = SL.shot_history(shot)
        self.assertEqual((tuple(hist.shape), float(hist.max()), "history" in shot), ((17, 8, 8, 3), 1.0, False))
        self.assertEqual(SL.shot_history({"chain": "off", "index": 1}), (None, None))

    def test_auto_loop_takes_the_previous_render(self):
        SL._LAST_RESULT.clear()
        SL._LAST_RESULT.update(index=0, count=2, frames=torch.rand(40, 8, 8, 3))
        hist, _ = SL.shot_history({"chain": "history", "index": 1, "count": 2, "prev_length": 30, "chain_history": 34})
        self.assertEqual(hist.shape[0], 22)          # 30 real frames -> 22, the longest 17k+5 that fits
        SL._LAST_RESULT.clear()

    def test_history_lengths_are_whole_vae_clips(self):
        # 17 frames encode to 2 tokens (the first 5 frames): only 17k+5 reaches the boundary
        self.assertEqual([SL.history_frame_count(w, 200) for w in (5, 17, 22, 34, 39, 51, 56)], [5, 22, 22, 39, 39, 56, 56])
        self.assertEqual([SL.history_frame_count(56, a) for a in (4, 5, 21, 22, 38, 40)], [0, 5, 5, 22, 22, 39])


class ShotHeadTest(unittest.TestCase):
    """Continuity 'head': the previous shot's end starts the next shot, pinned in the latent, and the join cuts it."""

    @staticmethod
    def _segs(chain="head", n=22, cut=False, gap=0):
        return [{"start": 0, "end": 60, "chain": "off", "cut_before": False},
                {"start": 60 + gap, "end": 120, "chain": chain, "chain_history": n, "cut_before": cut}]

    def test_head_length(self):
        self.assertEqual(SL.head_length(self._segs(), 1, H3), 22)
        self.assertEqual(SL.head_length(self._segs(n=56), 1, H3), 39)        # capped at 39
        self.assertEqual(SL.head_length(self._segs(n=17), 1, H3), 22)        # old plans: next valid length up
        self.assertEqual(SL.head_length(self._segs(), 0, H3), 0)             # first shot
        self.assertEqual(SL.head_length(self._segs(cut=True), 1, H3), 0)     # a camera cut
        self.assertEqual(SL.head_length(self._segs(gap=5), 1, H3), 0)       # a skipped shot in between
        self.assertEqual(SL.head_length(self._segs(chain="history"), 1, H3), 0)
        self.assertEqual(SL.head_length(self._segs(), 1, "LTX / Wan (8n+1)"), 0)

    def test_join_cuts_the_head(self):
        img = torch.arange(30, dtype=torch.float32).view(30, 1, 1, 1).expand(30, 2, 2, 3)
        shot = {"head": 5, "gen_length": 30, "frames": img.clone(), "crop": {"mask": torch.zeros(30, 2, 2), "box": [0, 0, 1, 1]}}
        out, sh = SL.drop_head(img, shot)
        self.assertEqual((out.shape[0], float(out[0, 0, 0, 0]), sh["gen_length"], sh["frames"].shape[0],
                          sh["crop"]["mask"].shape[0], sh["head"], shot["frames"].shape[0]), (25, 5.0, 25, 25, 25, 0, 30))
        self.assertIs(SL.drop_head(img, {"head": 0})[0], img)

    def test_remember_result_skips_the_head(self):
        SL._LAST_RESULT.clear()
        SL.remember_result({"index": 0, "count": 2, "head": 5}, torch.zeros(30, 2, 2, 3))
        self.assertEqual(SL._LAST_RESULT["frames"].shape[0], 25)
        SL._LAST_RESULT.clear()

    def test_pin_head_keeps_the_first_latent_steps(self):
        class NT:
            def __init__(self, t):
                self.tensors, self.is_nested = list(t), True
        nt = types.ModuleType("comfy.nested_tensor")
        nt.NestedTensor = NT
        old = {k: sys.modules.get(k) for k in ("comfy", "comfy.nested_tensor")}
        comfy = types.ModuleType("comfy")
        comfy.nested_tensor = nt
        sys.modules["comfy"], sys.modules["comfy.nested_tensor"] = comfy, nt
        try:
            import bfs_h3_history as HH

            class VAE:
                def encode(self, px):
                    n = px.shape[0]
                    self.size = tuple(px.shape[1:3])
                    return torch.full((1, 24, 5 * -(-n // 17) - 3, px.shape[1] // 16, px.shape[2] // 16), 7.0)
            vae = VAE()
            latent = {"samples": NT((torch.zeros(1, 24, 12, 4, 4), torch.zeros(1, 8, 50)))}
            out, n = HH.pin_head(latent, vae, torch.rand(30, 32, 32, 3))   # 30 frames -> the last 22 -> 7 steps
            v, m = out["samples"].tensors[0], out["noise_mask"].tensors[0]
            self.assertEqual((n, vae.size), (22, (64, 64)))
            self.assertTrue(bool((v[:, :, :7] == 7).all()) and bool((v[:, :, 7:] == 0).all()))
            self.assertTrue(bool((m[:, :, :7] == 0).all()) and bool((m[:, :, 7:] == 1).all()))
            self.assertTrue(bool((out["noise_mask"].tensors[1] == 1).all()))
            # an existing inpaint mask stays as it is after the head
            inp = torch.full((1, 1, 12, 4, 4), 0.5)
            out2, _ = HH.pin_head(dict(latent, noise_mask=NT((inp, torch.ones(1, 8, 50)))), vae, torch.rand(22, 64, 64, 3))
            m2 = out2["noise_mask"].tensors[0]
            self.assertTrue(bool((m2[:, :, :7] == 0).all()) and bool((m2[:, :, 7:] == 0.5).all()) and bool((inp == 0.5).all()))
        finally:
            for k, v in old.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


class MemoryStatusTest(unittest.TestCase):
    def test_shot_tokens(self):
        # 107 H3 frames -> 7 clips -> 32 latent steps; 960x544 -> 60 x 34 cells
        self.assertEqual(SL.shot_tokens(107, 960, 544), 32 * 34 * 60)
        self.assertEqual(SL.shot_tokens(22, 32, 32), 7 * 2 * 2)
        self.assertGreater(SL.shot_tokens(90, 960, 544), SL.shot_tokens(73, 960, 544))   # a head makes the shot bigger
        self.assertEqual(SL.shot_tokens(81, 32, 32, "LTX / Wan (8n+1)"), 21 * 2 * 2)

    def test_record_and_status(self):
        import json as _json
        path = Path(_TMP) / SL._VRAM_FILE
        path.unlink(missing_ok=True)
        cuda = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, current_device=lambda: 0,
                                     get_device_name=lambda d: "Test GPU", max_memory_allocated=lambda: 12 * 2 ** 30,
                                     reset_peak_memory_stats=lambda: None, mem_get_info=lambda: (8 * 2 ** 30, 32 * 2 ** 30))
        with patch.object(SL.torch, "cuda", cuda):
            SL.vram_record(5000)
            SL.vram_record(3000)          # smaller: the biggest stays
            st = SL.vram_status()
        self.assertEqual((st["gpu"], st["max_tokens"], st["max_peak_gb"], st["runs"], st["total_gb"], st["free_gb"]),
                         ("Test GPU", 5000, 12.0, 2, 32.0, 8.0))
        self.assertIn("Test GPU", _json.loads(path.read_text()))
        SL.vram_record(4000)              # no CUDA in use: nothing recorded, nothing started
        self.assertEqual(len(_json.loads(path.read_text())["Test GPU"]["samples"]), 2)


class ShotStoreTest(unittest.TestCase):
    def test_storage_round_trip(self):
        d = tempfile.mkdtemp(dir=_TMP)
        y = torch.linspace(0, 1, 48).view(1, 48, 1, 1)
        frames = (((torch.sin(torch.arange(12).view(12, 1, 1, 1) + 9 * y) + 1) / 2).expand(12, 48, 64, 3) * 255).round().byte()
        for k, mode in enumerate(("lossless", "raw", "smaller")):
            SL.store_shot(d, k, frames, {"index": k, "fps": 24.0}, mode)
            back, meta = SL.load_shot(d, k)
            self.assertEqual((tuple(back.shape), meta["index"]), ((12, 48, 64, 3), k))
            diff = int((back.int() - frames.int()).abs().max())
            if mode == "smaller":
                self.assertLess(diff, 40)
            else:
                self.assertEqual(diff, 0, mode)        # lossless and raw are exact
        self.assertTrue(os.path.exists(os.path.join(d, "shot_0000.mkv")))
        self.assertFalse(os.path.exists(os.path.join(d, "shot_0001.mkv")))   # raw: the frames are in the .pt

    def test_status_cascades_through_continuity(self):
        d = tempfile.mkdtemp(dir=_TMP)
        for i in range(3):
            open(os.path.join(d, f"shot_{i:04d}.pt"), "wb").close()
        st = {"done": [0, 1, 2], "sig": {"0": "a", "1": "b", "2": "c"}, "gen": {"0": "g0", "1": "g1", "2": "g2"},
              "from": {"1": "g0", "2": "g1"}, "cur": {"sigs": ["a", "b", "c"], "prev": [False, True, False]}}
        self.assertEqual(SL.shot_status(st, d), [True, True, True])
        st["gen"]["0"] = "new"                                   # shot 1 was redone: shot 2 continues from it
        self.assertEqual(SL.shot_status(st, d), [True, False, True])   # shot 3 does not continue (cut)
        st["cur"]["sigs"][2] = "changed"                          # a setting of shot 3 changed
        self.assertEqual(SL.shot_status(st, d), [True, False, False])

    def test_join_can_write_an_mp4(self):
        import cv2
        d = tempfile.mkdtemp(dir=_TMP)
        imgs = [torch.full((22, 32, 32, 3), 0.2), torch.full((39, 32, 32, 3), 0.8)]
        shots = [_shot(0, 0, 20, gen=22), _shot(1, 20, 45, cut=True, gen=39)]
        audio = {"waveform": torch.zeros(1, 2, 48000 * 3), "sample_rate": 48000}
        with patch.object(SL, "_next_output_path", lambda prefix: os.path.join(d, prefix + "_00001.mp4")):
            out = SL.BFSShotJoin().join(imgs, shots, [0], audio=[audio], save_to=[SL.SAVE_TO[1]],
                                        filename_prefix=["test"])
        res = out
        self.assertEqual(res[0].shape[0], 45)                 # the output still carries the video (read back)
        self.assertAlmostEqual(float(res[0][30].mean()), 0.8, places=1)
        self.assertTrue(res[4].endswith("test_00001.mp4"))
        self.assertIsInstance(out, tuple)                       # no preview on the node: plain outputs
        cap = cv2.VideoCapture(res[4]); n = 0
        while cap.read()[0]:
            n += 1
        self.assertEqual(n, 45)


class RenderFingerprintTest(unittest.TestCase):
    """Changing what renders the shots (LoRA, seed...) must make the stored shots stale; the join's settings must not."""

    @staticmethod
    def graph():
        return {
            "1": {"class_type": "BFSShotPlanner", "inputs": {"plan": "{\"run\": \"auto\"}"}},
            "2": {"class_type": "BFSShotUnpack", "inputs": {"shot": ["1", 0]}},
            "3": {"class_type": "BFSShotH3ConditioningSimple", "inputs": {"shot": ["2", 10], "clip": ["4", 0], "guide_mode": "aligned guide"}},
            "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl.safetensors", "type": "minimax"}},
            "5": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3.safetensors", "weight_dtype": "default"}},
            "6": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["5", 0], "lora_name": "swap_v1.safetensors", "strength_model": 1.0}},
            "7": {"class_type": "KSampler", "inputs": {"model": ["6", 0], "positive": ["3", 0], "seed": 42, "steps": 8}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["9", 0]}},
            "9": {"class_type": "VAELoader", "inputs": {"vae_name": "vae.safetensors"}},
            "10": {"class_type": "BFSShotJoin", "inputs": {"images": ["8", 0], "shots": ["2", 10], "crossfade": 4}},
            "11": {"class_type": "BFSVideoCompare", "inputs": {"guide": ["10", 5], "target": ["10", 0], "gain": 2.0}},
        }

    def test_render_changes_count_and_join_settings_do_not(self):
        base = SL.render_fingerprint(self.graph(), "1")
        self.assertTrue(base)
        self.assertEqual(SL.render_fingerprint(self.graph(), "1"), base)
        for node, key, value, changes in (("6", "lora_name", "swap_v2.safetensors", True), ("6", "strength_model", 0.8, True),
                                          ("7", "seed", 43, True), ("3", "guide_mode", "reference video", True),
                                          ("10", "crossfade", 8, False), ("1", "plan", "{}", False), ("11", "gain", 4.0, False)):
            g = self.graph()
            g[node]["inputs"][key] = value
            self.assertEqual(SL.render_fingerprint(g, "1") != base, changes, f"{node}.{key}")
        g = self.graph()
        del g["10"]
        self.assertEqual(SL.render_fingerprint(g, "1"), "")

    def test_the_fingerprint_is_part_of_the_shot_signature(self):
        seg = {"start": 0, "end": 20, "prompt": "x"}
        a = SL.shot_signature({}, seg, 0, {"render": "aaa"})
        self.assertNotEqual(a, SL.shot_signature({}, seg, 0, {"render": "bbb"}))
        self.assertEqual(SL.BFSShotPlanner.INPUT_TYPES()["hidden"], {"graph": "PROMPT", "node_id": "UNIQUE_ID"})


class SecondPassSizeTest(unittest.TestCase):
    def _upscaler(self, w, h, mode, val, align=32):
        # Minimax H3 Latent Upscaler (3D), from a latent of w/16 x h/16 cells
        if mode == "scale": tw, th = w * val, h * val
        elif mode == "dims": tw, th = val
        else:
            th = (val * 1024 * 1024 / (w / h)) ** 0.5; tw = th * (w / h)
        tw, th = round(tw / align) * align, round(th / align) * align
        return int(round(tw / 16) * 16), int(round(th / 16) * 16)

    def test_matches_the_latent_upscaler(self):
        for w, h in ((832, 480), (480, 832), (1024, 576), (704, 704)):
            self.assertEqual(SL.second_pass_size(w, h, {"mode": "scale", "scale": 2.0}), self._upscaler(w, h, "scale", 2.0))
            self.assertEqual(SL.second_pass_size(w, h, {"mode": "scale", "scale": 1.5, "align": 64}),
                             self._upscaler(w, h, "scale", 1.5, 64))
            self.assertEqual(SL.second_pass_size(w, h, {"mode": "megapixels", "megapixels": 1.0}),
                             self._upscaler(w, h, "mega", 1.0))
        self.assertEqual(SL.second_pass_size(832, 480, {"mode": "dims", "width": 1280, "height": 704}), (1280, 704))
        self.assertEqual(SL.second_pass_size(832, 480, None), (1664, 960))     # defaults: x2, align 32


class JoinWithoutFfmpegTest(unittest.TestCase):
    def test_mp4_mode_works_without_the_ffmpeg_program(self):
        # issue #12: Windows portable, no ffmpeg on PATH, long video -> the RAM guard picks the mp4 mode
        d = tempfile.mkdtemp(dir=_TMP)
        imgs = [torch.full((22, 32, 32, 3), 0.2), torch.full((39, 32, 32, 3), 0.8)]
        shots = [_shot(0, 0, 20, gen=22), _shot(1, 20, 45, cut=True, gen=39)]
        audio = {"waveform": torch.zeros(1, 2, 48000 * 3), "sample_rate": 48000}
        with patch.dict(SL.VIO._BIN, {"ffmpeg": None, "ffprobe": None}), \
                patch.object(SL, "_ram_available", lambda: 1000), \
                patch.object(SL, "_next_output_path", lambda prefix: os.path.join(d, prefix + "_00001.mp4")):
            out = SL.BFSShotJoin().join(imgs, shots, [0], audio=[audio], comparison=[True])
        res = out
        self.assertTrue(os.path.getsize(res[4]) > 0 and os.path.exists(res[4][:-4] + "_comparison.mp4"))
        self.assertEqual((res[0].shape[0], res[3].shape[0], res[5].shape[0]), (45, 45, 45))
        self.assertAlmostEqual(float(res[0][30].mean()), 0.8, delta=0.05)


class JoinFileMetadataTest(unittest.TestCase):
    def test_mp4_carries_prompt_and_workflow(self):
        import json, subprocess
        d = tempfile.mkdtemp(dir=_TMP)
        wf = {"nodes": [{"id": 1, "widgets_values": ["a=b; c#d \\ e\nnew line"]}]}
        pr = {"1": {"class_type": "BFSShotPlanner", "inputs": {"plan": "{}"}}}
        with patch.object(SL, "_next_output_path", lambda prefix: os.path.join(d, prefix + "_00001.mp4")):
            out = SL.BFSShotJoin().join([torch.rand(22, 32, 32, 3)], [_shot(0, 0, 20, gen=22)], [0], save_to=[SL.SAVE_TO[1]],
                                        filename_prefix=["m"], prompt=[pr], extra_pnginfo=[{"workflow": wf}])
        tags = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json",
                                          out[4]], capture_output=True).stdout)["format"]["tags"]
        self.assertEqual((json.loads(tags["workflow"]), json.loads(tags["prompt"])), (wf, pr))


class JoinRamGuardTest(unittest.TestCase):
    def test_images_join_falls_back_to_mp4_when_ram_is_short(self):
        d = tempfile.mkdtemp(dir=_TMP)
        imgs = [torch.full((22, 32, 32, 3), 0.2), torch.full((39, 32, 32, 3), 0.8)]
        shots = [_shot(0, 0, 20, gen=22), _shot(1, 20, 45, cut=True, gen=39)]
        with patch.object(SL, "_ram_available", lambda: 1000), \
                patch.object(SL, "_next_output_path", lambda prefix: os.path.join(d, prefix + "_00001.mp4")):
            out = SL.BFSShotJoin().join(imgs, shots, [0], comparison=[True])
        res = out
        self.assertTrue(os.path.exists(res[4]) and os.path.exists(res[4][:-4] + "_comparison.mp4"))
        self.assertEqual(res[0].shape[0], 45)                                   # never an empty output node
        self.assertTrue(res[0].shape[1] < 32 and res[3].shape[0] == 45)          # scaled down to the (tiny) free RAM
        self.assertTrue(os.path.exists(res[4][:-4] + "_guide.mp4") and res[5].shape[0] == 45)
        normal = SL.BFSShotJoin().join(imgs, shots, [0])  # enough RAM: images as before
        self.assertEqual((normal[0].shape[0], normal[4]), (45, ""))

    def test_join_bytes(self):
        self.assertEqual(SL._join_bytes(10, 100, 200, False, 360), 10 * 100 * 200 * 12)
        self.assertGreater(SL._join_bytes(10, 100, 200, True, 1024), SL._join_bytes(10, 100, 200, True, 50))


class StoredComparisonTest(unittest.TestCase):
    def test_originals_go_to_a_video_not_the_pt(self):
        rid = "cmp_" + str(id(self))
        y = torch.linspace(0, 1, 64).view(1, 64, 1, 1)
        src = [((torch.sin(torch.arange(n).view(n, 1, 1, 1) + 7 * y) + 1) / 2).expand(n, 64, 64, 3).clone() for n in (22, 39)]
        shots = [dict(_shot(0, 0, 20, count=2, queue=True, run_id=rid, gen=22), frames=src[0]),
                 dict(_shot(1, 20, 45, cut=True, count=2, queue=True, run_id=rid, gen=39), frames=src[1])]
        SL.BFSShotJoin().join([src[0] * 0.5], [shots[0]], [0], comparison=[True], comparison_height=[64])
        out = SL.BFSShotJoin().join([src[1] * 0.5], [shots[1]], [0], comparison=[True], comparison_height=[64])
        d = SL.run_dir(rid)
        self.assertLess(os.path.getsize(os.path.join(d, "shot_0000.pt")), 50_000)          # info only
        self.assertTrue(os.path.exists(os.path.join(d, "shot_0000_orig.mkv")))
        video, comp = out[0], out[3]
        self.assertEqual(video.shape[0], 45)
        self.assertEqual(comp.shape[0], 45)
        # the original column (left) shows the source, not the dark filler of a missing original
        self.assertGreater(float(comp[:, comp.shape[1] // 2, 5:60].mean()), 0.2)
        guide = out[5]                                                           # the stored originals, aligned
        self.assertEqual(guide.shape[0], 45)
        self.assertAlmostEqual(float(guide[:20].mean()), float(src[0][:20].mean()), delta=0.03)
        self.assertAlmostEqual(float(guide[20:].mean()), float(src[1][:25].mean()), delta=0.03)


class ColorAnchorTest(unittest.TestCase):
    def _shot(self, sat):
        src = torch.full((12, 48, 48, 3), 0.5)
        res = src.clone()
        res[:, 12:36, 12:36] = torch.tensor([0.5 - sat, 0.5 + sat, 0.5 - sat])   # a green subject
        res[:, 12:36, 12:36] += torch.linspace(-0.05, 0.05, 24).view(1, 24, 1, 1)
        return res, src

    def test_subject_chroma_goes_back_to_the_anchor(self):
        r1, s1 = self._shot(0.15)
        anchor = SL.color_stats(r1, s1)
        r4, s4 = self._shot(0.30)                      # 4 shots later: twice as saturated
        out, before = SL.anchor_colors(r4, s4, anchor)
        after = SL.color_stats(out, s4)
        self.assertGreater(abs(before["mu"][0] - anchor["mu"][0]), 5)
        self.assertLess(abs(after["mu"][0] - anchor["mu"][0]), 1.5)
        self.assertLess(abs(after["mu"][1] - anchor["mu"][1]), 1.5)
        self.assertAlmostEqual(float(out[0, 2, 2, 0]), 0.5, places=2)   # the background is untouched
        self.assertIsNone(SL.color_stats(s1, s1))                        # no subject: nothing to anchor


class TidyDetailsTest(unittest.TestCase):
    def test_short_details_drop_the_echoed_prompt(self):
        cfg = {"describe_preset": "short"}
        for raw in ("is the elderly woman with glasses shown in", "elderly woman with glasses is the shown in <Picture 1>",
                    "<Subject 1> is the elderly woman with glasses shown in <Picture 1>.", "The elderly woman with glasses"):
            self.assertEqual(SL.tidy_details(raw, cfg), "elderly woman with glasses", raw)
        self.assertEqual(SL.tidy_details("man shown wearing a red cap", cfg), "man shown wearing a red cap")
        # descriptions already saved in a plan are cleaned when they go into the prompt
        self.assertEqual(SL.clean_slot("is the woman in red shown in <Picture 1>"), "woman in red")


class FillDetailsTest(unittest.TestCase):
    def test_render_node_fills_details_with_its_vlm(self):
        old = SL.vlm_describe
        SL.vlm_describe = lambda clip, ims, q, mt=320, temperature=0.0, seed=0: "The elderly man with white hair in a light blue t-shirt."
        try:
            out = SL.fill_details_now("<Subject 1> is the {details} shown in <Picture 1>.", object(), [torch.rand(1, 8, 8, 3)])
            self.assertEqual(out, "<Subject 1> is the elderly man with white hair in a light blue t-shirt shown in <Picture 1>.")
            self.assertIn("character", SL.fill_details_now("is the {details} shown", None, []))
            self.assertEqual(SL.fill_details_now("no slot", None, []), "no slot")
        finally:
            SL.vlm_describe = old


if __name__ == "__main__":
    unittest.main()


class AgentMaskKeyframeTest(unittest.TestCase):
    def test_later_keyframe_corrects_only_its_suffix(self):
        frames = torch.zeros(8, 2, 2, 3)
        seen = []
        original = SL.segment_frames

        def segment(images, spec):
            if spec.get('keyframes'):
                return original(images, spec)
            seen.append((len(images), spec['key'], spec['text']))
            return torch.full((len(images), 2, 2), 1.0 if spec['text']=='first' else 2.0)

        spec = SL.mask_spec({'keyframes':[
            {'key':2, 'text':'first', 'points':[]},
            {'key':5, 'text':'second', 'points':[]},
        ]})
        with patch.object(SL, 'segment_frames', side_effect=segment):
            out = original(frames, spec)
        self.assertEqual(seen, [(5,2,'first'),(3,0,'second')])
        self.assertEqual(out[:,0,0].tolist(), [1]*5+[2]*3)
        self.assertTrue(SL.spec_has_mask(spec))
