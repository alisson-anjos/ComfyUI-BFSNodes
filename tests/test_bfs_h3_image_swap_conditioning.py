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
                phrase = 'is identified by the' if mode == 'colored hatching + contour' else 'is the subject marked by the'
                self.assertIn(f'<Subject {j}> {phrase} {color}', text)
                self.assertIn(f'appearance comes from <Picture {j+1}>', text)
            self.assertIn('foreground occluders', text)

    def test_hatching_prompt_transfers_clothing_and_preserves_source_lighting(self):
        for count in (1, 2, 3):
            text = module.image_swap_prompt(count, 'colored hatching + contour')
            self.assertIn('complete outfit', text)
            self.assertIn('source lighting, shadows and exposure', text)
            self.assertIn('Remove all colored lines and contours', text)
            self.assertIn(f'<Picture {count+1}>', text)
            self.assertNotIn(f'<Subject {count+1}>', text)

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


class AutoImageSwapTest(unittest.TestCase):
    def test_blue_and_green_without_red_keep_color_reference_binding(self):
        clip = Clip()
        guide = torch.zeros(1, 32, 32, 3)
        blue, green = torch.full_like(guide, .3), torch.full_like(guide, .4)
        report = '{"active_subjects":[2,3],"mark_style":"colored hatching + contour"}'
        output = module.BFSH3ImageAutoCharacterSwapConditioning.execute(
            clip, Vae(), guide, report, blue_reference=blue, green_reference=green, width=128, height=128)
        text = output.args[2]
        self.assertIn('blue hatching', text)
        self.assertIn('green hatching', text)
        self.assertNotRegex(text, r'\bred hatching\b')
        self.assertIn('Colored hatching', text)
        self.assertIn('<Picture 3>', text)
        self.assertNotIn('<Picture 4>', text)
        self.assertEqual(len(clip.images), 3)
        self.assertAlmostEqual(float(clip.images[1].mean()), .3, delta=.004)
        self.assertAlmostEqual(float(clip.images[2].mean()), .4, delta=.004)

    def test_empty_picker_selection_blocks_sampling_without_reference_requests(self):
        report = '{"active_subjects":[],"mark_style":"colored hatching + contour"}'
        self.assertEqual(module.BFSH3ImageAutoCharacterSwapConditioning.check_lazy_status(report), [])
        output = module.BFSH3ImageAutoCharacterSwapConditioning.execute(Clip(), Vae(), torch.zeros(1, 32, 32, 3), report)
        self.assertIn('Select subjects', output.args[2])


    def test_prompt_override_is_used_verbatim_without_changing_selected_references(self):
        clip = Clip()
        guide = torch.zeros(1, 32, 32, 3)
        blue = torch.full_like(guide, .3)
        report = '{"active_subjects":[2],"mark_style":"colored hatching + contour"}'
        custom = '  Edit <Picture 1> using <Picture 2>; preserve red background objects.  '
        output = module.BFSH3ImageAutoCharacterSwapConditioning.execute(
            clip, Vae(), guide, report, blue_reference=blue, width=128, height=128, prompt_override=custom)
        self.assertEqual(output.args[2], custom)
        self.assertEqual(clip.prompt, custom)
        self.assertEqual(len(clip.images), 2)
        self.assertAlmostEqual(float(clip.images[1].mean()), .3, delta=.004)
        automatic = module.BFSH3ImageAutoCharacterSwapConditioning.execute(
            Clip(), Vae(), guide, report, blue_reference=blue, width=128, height=128, prompt_override='  ')
        self.assertIn('blue hatching', automatic.args[2])
