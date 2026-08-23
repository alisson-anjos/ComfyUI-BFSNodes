"""Reference-layout placement, on both LTX coordinate conventions.

The node runs against two shapes and two RoPE collapse rules, and the sidecar layout is the
one that is sensitive to all of them:

  [B, 3, N, 2] patch bounds  -- current builds, LTX-2.3 and LTX-2.5 alike
  [B, 3, N]    single corner -- legacy builds
  use_middle_indices_grid    -- True (2.5) collapses a token to (start+end)/2, False (2.3
                                checkpoints whose metadata omits the flag) reads the start

`_collapse` below mirrors comfy.ldm.lightricks.model.generate_freqs, so these tests compare
the position the model actually sees, not the raw tensor.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
_PKG = types.ModuleType("bfsnodes_under_test")
_PKG.__path__ = [str(ROOT)]
sys.modules.setdefault("bfsnodes_under_test", _PKG)
OVERLAP = importlib.import_module("bfsnodes_under_test.ltx_identity_overlap")


def _bounds(batch, tokens, t_span=576.0, height=384.0, width=704.0, seed=0):
    """Patch bounds [B, 3, N, 2] with a plausible per-patch extent."""
    generator = torch.Generator().manual_seed(seed)
    low = torch.stack([
        torch.rand(tokens, generator=generator) * t_span,
        torch.rand(tokens, generator=generator) * height,
        torch.rand(tokens, generator=generator) * width,
    ])
    high = low + torch.tensor([8.0, 32.0, 32.0]).unsqueeze(1)
    return torch.stack([low, high], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1).contiguous()


def _collapse(coords, use_middle):
    """The one number per token the RoPE actually uses -- see generate_freqs."""
    if use_middle and coords.dim() == 4:
        return (coords[..., 0] + coords[..., 1]) / 2.0
    return coords[..., 0] if coords.dim() == 4 else coords


def _axis_centre(coords, axis):
    return (coords[0, axis].min() + coords[0, axis].max()) / 2.0


class TassLayoutTest(unittest.TestCase):
    def setUp(self):
        # A batch-2 target (CFG) against a batch-1 reference: layouts run before the reference
        # is expanded, which is what crashed the first sidecar run.
        self.target = _bounds(2, 512, seed=1)
        self.reference = _bounds(1, 64, t_span=8.0, height=128.0, width=128.0, seed=2)

    def test_sidecar_places_reference_at_clip_centre_under_both_collapse_rules(self):
        for use_middle in (True, False):
            with self.subTest(use_middle=use_middle):
                shifted = OVERLAP._apply_tass_layout(
                    self.reference, self.target, "sidecar", use_middle_indices_grid=use_middle
                )
                reference_seen = _collapse(shifted, use_middle)
                target_seen = _collapse(self.target, use_middle)
                # T: centred on the target's clip, not pinned to frame zero
                self.assertAlmostEqual(
                    float(_axis_centre(reference_seen, 0)), float(_axis_centre(target_seen, 0)), delta=4.0
                )
                # H: centred on the target's height
                self.assertAlmostEqual(
                    float(_axis_centre(reference_seen, 1)), float(_axis_centre(target_seen, 1)), delta=1e-3
                )
                # W: entirely beside the target, never overlapping it
                self.assertGreaterEqual(
                    float(reference_seen[0, 2].min()), float(target_seen[0, 2].max()) - 1e-3
                )

    def test_sidecar_survives_legacy_single_corner_coordinates(self):
        shifted = OVERLAP._apply_tass_layout(
            self.reference[..., 0].contiguous(),
            self.target[..., 0].contiguous(),
            "sidecar",
            use_middle_indices_grid=False,
        )
        self.assertEqual(shifted.shape, self.reference[..., 0].shape)
        target = self.target[..., 0]
        self.assertAlmostEqual(float(_axis_centre(shifted, 0)), float(_axis_centre(target, 0)), delta=4.0)
        self.assertAlmostEqual(float(_axis_centre(shifted, 1)), float(_axis_centre(target, 1)), delta=1e-3)
        self.assertGreaterEqual(float(shifted[0, 2].min()), float(target[0, 2].max()) - 1e-3)

    def test_sidecar_margin_pushes_the_reference_further_right(self):
        flush = OVERLAP._apply_tass_layout(self.reference, self.target, "sidecar")
        spaced = OVERLAP._apply_tass_layout(
            self.reference, self.target, "sidecar", sidecar_margin_pixels=64.0
        )
        self.assertAlmostEqual(float(spaced[0, 2].min() - flush[0, 2].min()), 64.0, delta=1e-3)

    def test_patch_bounds_are_translated_not_squeezed(self):
        span = (self.reference[..., 1] - self.reference[..., 0])
        for layout, kwargs in (("st_drc", {}), ("strata", {"strata_start": 600.0}), ("sidecar", {})):
            with self.subTest(layout=layout):
                shifted = OVERLAP._apply_tass_layout(self.reference, self.target, layout, **kwargs)
                moved = shifted[..., 1] - shifted[..., 0]
                # sidecar deliberately rewrites the temporal span; H/W must keep their extent
                axes = slice(1, 3) if layout == "sidecar" else slice(0, 3)
                # atol covers float32 rounding on ~600px coordinates, not a real drift
                torch.testing.assert_close(moved[:, axes], span[:, axes], atol=1e-3, rtol=0)

    def test_integer_coordinates_keep_their_dtype(self):
        """ComfyUI hands out int64 pixel coords -- float shifts must not blow up on them."""
        target = self.target.round().to(torch.int64)
        reference = self.reference.round().to(torch.int64)
        for layout, kwargs in (
            ("overlap", {}), ("st_drc", {}), ("strata", {"strata_start": 600.0}), ("sidecar", {})
        ):
            with self.subTest(layout=layout):
                shifted = OVERLAP._apply_tass_layout(reference, target, layout, **kwargs)
                self.assertEqual(shifted.dtype, torch.int64)
                float_result = OVERLAP._apply_tass_layout(
                    reference.float(), target.float(), layout, **kwargs
                )
                # rounded, never truncated: at most half a pixel from the exact placement
                self.assertLessEqual(float((shifted.float() - float_result).abs().max()), 0.5)

    def test_overlap_is_an_exact_noop(self):
        self.assertIs(
            OVERLAP._apply_tass_layout(self.reference, self.target, "overlap"), self.reference
        )

    def test_unknown_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            OVERLAP._apply_tass_layout(self.reference, self.target, "nope")


if __name__ == "__main__":
    unittest.main()
