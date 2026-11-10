import importlib.util
import sys
import os
import unittest
from pathlib import Path
import torch

if os.environ.get('COMFYUI_PATH'):
    sys.path.insert(0, os.environ['COMFYUI_PATH'])
spec = importlib.util.spec_from_file_location('image_swap', Path(__file__).resolve().parents[1] / 'bfs_h3_image_swap_conditioning.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Clip:
    def tokenize(self, prompt, **kwargs):
        self.prompt, self.images = prompt, kwargs['images']
        return {'text': prompt}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 4, 8), {}]]


class Vae:
    def encode(self, pixels):
        return torch.zeros(1, 24, 1, pixels.shape[1] // 16, pixels.shape[2] // 16)


class ImageSwapTest(unittest.TestCase):
    def test_modes_use_still_image_language_and_shifted_picture_numbers(self):
        for mode in module.MODES:
            text = module.image_swap_prompt(3, mode)
            for word in ('video', 'motion', 'frame', 'timing'):
                self.assertNotIn(word, text.lower())
            for j, color in enumerate(module.COLORS, 1):
                self.assertIn(f'<Subject {j}> is the subject marked by the {color}', text)
                self.assertIn(f'appearance comes from <Picture {j+1}>', text)
            self.assertIn('foreground occluders', text)

    def test_native_conditioning_order_and_image_latent(self):
        clip = Clip()
        guide = torch.full((1, 32, 32, 3), .1)
        red = torch.full_like(guide, .2)
        blue = torch.full_like(guide, .3)
        result = module.BFSH3ImageCharacterSwapConditioning.execute(
            clip, Vae(), guide, {'reference_2': blue, 'reference_0': red}, width=128, height=96)
        positive, latent, prompt = result.args
        self.assertEqual(tuple(latent['samples'].shape), (1, 24, 1, 6, 8))
        self.assertEqual(len(positive[0][1]['minimax_refs']), 3)
        self.assertEqual([round(float(i.mean()), 1) for i in clip.images], [.1, .2, .3])
        self.assertIn('red subject replaced by <Picture 2>', prompt)
        self.assertIn('blue subject replaced by <Picture 3>', prompt)

    def test_override_preserved(self):
        clip = Clip()
        image = torch.zeros(1, 32, 32, 3)
        result = module.BFSH3ImageCharacterSwapConditioning.execute(
            clip, Vae(), image, {'reference_0': image}, width=64, height=64, prompt_override='Custom edit')
        self.assertEqual(result.args[2], 'Custom edit')
        self.assertEqual(clip.prompt, 'Custom edit')
