import importlib.util
import sys
import os
import unittest
from pathlib import Path
import torch
if os.environ.get('COMFYUI_PATH'):
    sys.path.insert(0, os.environ['COMFYUI_PATH'])
from comfy.ldm.minimax.model import PackedLayout
spec = importlib.util.spec_from_file_location('image_guide', Path(__file__).resolve().parents[1] / 'bfs_h3_image_guide.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ImageGuideTest(unittest.TestCase):
    def test_only_guide_positions_are_aligned_and_inputs_stay_unchanged(self):
        refs = [{'kind': 'image', 'latent_h': 8, 'latent_w': 8},
                {'kind': 'image', 'latent_h': 8, 'latent_w': 8, module.MARKER: True}]
        original = PackedLayout(4, 1, 8, 8, 0, refs=refs)
        positions = original.position_ids.clone()
        result = module.align_guide_layout(original, refs)
        ref_segments = [(a,b) for a,b,k in result.segments if k == 'ref_img']
        a,b = ref_segments[1]
        ta,tb = next((a,b) for a,b,k in result.segments if k == 'video')
        self.assertTrue(torch.equal(result.position_ids[a:b], result.position_ids[ta:tb]))
        a,b = ref_segments[0]
        self.assertTrue(torch.equal(result.position_ids[a:b], positions[a:b]))
        self.assertTrue(torch.equal(original.position_ids, positions))
        self.assertTrue(torch.equal(result.img_update, original.img_update))

    def test_incompatible_guide_grid_is_rejected(self):
        refs = [{'kind': 'image', 'latent_h': 4, 'latent_w': 4, module.MARKER: True}]
        layout = PackedLayout(4, 1, 8, 8, 0, refs=refs)
        with self.assertRaisesRegex(ValueError, 'match the target'):
            module.align_guide_layout(layout, refs)
