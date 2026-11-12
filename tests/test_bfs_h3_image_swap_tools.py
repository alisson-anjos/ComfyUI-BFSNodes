import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

if os.environ.get('COMFYUI_PATH'):
    sys.path.insert(0, os.environ['COMFYUI_PATH'])
path = Path(__file__).resolve().parents[1] / 'bfs_h3_image_swap_tools.py'
spec = importlib.util.spec_from_file_location('swap_tools', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReferenceToolsTest(unittest.TestCase):
    def test_none_slot_skips_loading_and_resizing(self):
        with patch.object(module.LoadImage, 'load_image', side_effect=AssertionError('An empty slot must not load')):
            self.assertEqual(module.BFSOptionalReferenceImage.execute('[none]', 2.0, 'area').args, (None,))

    def test_mp_and_interpolation_are_editable(self):
        image = torch.ones(1, 96, 128, 3)
        for mp, method in [(.1, 'area'), (.2, 'bilinear')]:
            with patch.object(module.LoadImage, 'load_image', return_value=(image, None)):
                output = module.BFSOptionalReferenceImage.execute('example.png', mp, method).args[0]
            height, width = output.shape[1:3]
            self.assertAlmostEqual(width / height, 128 / 96, delta=.005)
            self.assertAlmostEqual(width * height / 1024**2, mp, delta=.001)

    def test_comparison_requests_only_selected_colors(self):
        self.assertEqual(module.BFSH3ImageSwapComparison.check_lazy_status('{"active_subjects":[2]}'), ['blue_reference'])
        self.assertEqual(module.BFSH3ImageSwapComparison.check_lazy_status('{"active_subjects":[]}'), [])

    def test_reference_resize_changes_cache_fingerprint(self):
        with patch.object(module.LoadImage, 'IS_CHANGED', return_value='digest'):
            one = module.BFSOptionalReferenceImage.fingerprint_inputs('example.png', 1.0, 'area')
            two = module.BFSOptionalReferenceImage.fingerprint_inputs('example.png', 2.0, 'area')
            other = module.BFSOptionalReferenceImage.fingerprint_inputs('example.png', 1.0, 'lanczos')
            self.assertNotEqual(one, two)
            self.assertNotEqual(one, other)
