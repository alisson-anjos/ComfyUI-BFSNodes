import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch

if os.environ.get('COMFYUI_PATH'):
    sys.path.insert(0, os.environ['COMFYUI_PATH'])
from comfy.lora import calculate_weight


def load(name):
    path = Path(__file__).resolve().parents[1] / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


surgery = load('bfs_model_surgery')
pdd = load('bfs_h3_pdd_surgery')


class Model:
    def __init__(self, state, final=None):
        self.state, self.final, self.patches = state, final, {}

    def model_state_dict(self):
        return self.state

    def clone(self):
        result = Model(self.state, self.final)
        result.patches = {key: list(value) for key, value in self.patches.items()}
        return result

    def get_model_object(self, name):
        assert name == 'diffusion_model.final_layer'
        return self.final

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        for spec, patch in patches.items():
            key, offset = spec if isinstance(spec, tuple) else (spec, None)
            self.patches.setdefault(key, []).append((strength_patch, patch, strength_model, offset, None))
        return list(patches)


class SurgeryTest(unittest.TestCase):
    def test_overlapping_rules_scale_existing_patches_and_preserve_input(self):
        key = 'diffusion_model.block.weight'
        original = torch.ones(6, 3)
        model = Model({key: original})
        model.patches[key] = [(1.0, ('diff', (torch.ones_like(original),)), 1.0, None, None)]
        result = surgery.BFSModelSurgery.execute(model, '[{"pattern":"*.weight","scale":0.5},{"pattern":"*.weight","rows":[0,2],"scale":0.5}]').args[0]
        output = calculate_weight(result.patches[key], original.clone(), key)
        expected = torch.ones_like(original); expected[:2] *= .5
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(original, torch.ones_like(original))
        self.assertEqual(len(model.patches[key]), 1)

    def test_invalid_row_range_fails_before_patching(self):
        with self.assertRaises(ValueError):
            surgery.plan_rules({'a.weight': torch.zeros(3, 2)}, [{'pattern': '*', 'rows': [0, 4]}])

    def test_empty_rules_do_not_patch_model(self):
        result = surgery.BFSModelSurgery.execute(Model({'a.weight': torch.ones(3, 2)})).args[0]
        self.assertFalse(result.patches)

    def test_pdd_offsets_preserve_shared_base_and_scale_bias(self):
        weight, bias = torch.ones(6, 4), torch.ones(6)
        final = SimpleNamespace(video_dim=2, video_out=SimpleNamespace(weight=weight, bias=bias), audio_out=None)
        state = {'diffusion_model.final_layer.video_out.weight': weight, 'diffusion_model.final_layer.video_out.bias': bias}
        model = Model(state, final)
        result = pdd.BFSH3PDDSurgery.execute(model, offset_strength=.5, offset_1=.8).args[0]
        for key, value in state.items():
            actual = calculate_weight(result.patches[key], value.clone(), key)
            expected = value.clone(); expected[2:4] *= .4; expected[4:] *= .5
            torch.testing.assert_close(actual, expected)
        self.assertFalse(model.patches)
        self.assertFalse(pdd.BFSH3PDDSurgery.execute(model).args[0].patches)
