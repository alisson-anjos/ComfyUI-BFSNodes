"""Experimental cached-case embedding swap, preserving existing H3 latent conditioning."""
import os
import torch
import folder_paths
from safetensors import safe_open
from safetensors.torch import load_file

folder_paths.add_model_folder_path('h3_embedding_bridge', os.path.join(folder_paths.models_dir, 'h3_embedding_bridge'))


class BFSH3ExperimentalEmbeddingBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'positive': ('CONDITIONING',),
            'embedding_file': (folder_paths.get_filename_list('h3_embedding_bridge') or ['(no calibrated export)'],),
            'baseline_tolerance': ('FLOAT', {'default': .02, 'min': 0.0, 'max': .05, 'step': .001,
                'tooltip': 'Reject cached embeddings for a different input/encoder. Use the matching calibration case, prompt, media interval and encoder. Do not raise this to bypass mismatched inputs.'}),
        }}

    RETURN_TYPES = ('CONDITIONING', 'STRING')
    RETURN_NAMES = ('positive', 'experiment_info')
    FUNCTION = 'apply'
    CATEGORY = 'BFS/experimental'
    DESCRIPTION = ('Replace only text/vision embeddings with a calibrated 9B hidden-state bridge export for the SAME cached case. '
                   'Preserves native reference latents and aligned guides. Requires trained bridge exports; this is not a general encoder replacement.')

    def apply(self, positive, embedding_file, baseline_tolerance=.02):
        if len(positive) != 1:
            raise ValueError('Experimental bridge supports one native H3 conditioning entry')
        path = folder_paths.get_full_path_or_raise('h3_embedding_bridge', embedding_file)
        with safe_open(path, framework='pt') as f:
            md = f.metadata() or {}
        if md.get('experiment') != 'h3_9b_hidden_state_bridge_v1':
            raise ValueError('Select a held-out conditioning export, not bridge weights or another LoRA')
        data = load_file(path); base = positive[0][0]; source = base[0] if base.ndim == 3 else base
        expected = data['teacher_baseline'].float(); actual = source.detach().cpu().float()
        if actual.shape != expected.shape:
            raise ValueError('Cached case does not match current prompt/media token layout')
        relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-8))
        if relative > baseline_tolerance:
            raise ValueError(f'Cached case mismatch: baseline relative error {relative:.5f}; use exact matching inputs')
        replacement = data['conditioning']; tags = data['minimax_token_tags']
        if replacement.shape != expected.shape or len(tags) != len(replacement) or not torch.isfinite(replacement).all():
            raise ValueError('Invalid bridge conditioning export')
        current_tags = positive[0][1].get('minimax_token_tags')
        if current_tags is None or not torch.equal(current_tags.detach().cpu().reshape(-1).long(), tags.reshape(-1).long()):
            raise ValueError('Cached case has different modality tags')
        values = dict(positive[0][1])
        values['minimax_token_tags'] = tags.to(current_tags.device)
        replacement = replacement.to(device=base.device, dtype=base.dtype)
        if base.ndim == 3:
            replacement = replacement[None]
        info = (f"EXPERIMENTAL {md['source_id']}: thinking={md['thinking']}, "
                f"generated states zeroed={md['generated_states_zeroed']}; "
                f"baseline error={relative:.5f}; H3 quality not validated")
        print('[BFS H3 embedding bridge] ' + info, flush=True)
        return [[replacement, values]], info


NODE_CLASS_MAPPINGS = {'BFSH3ExperimentalEmbeddingBridge': BFSH3ExperimentalEmbeddingBridge}
NODE_DISPLAY_NAME_MAPPINGS = {'BFSH3ExperimentalEmbeddingBridge': 'BFS H3 Experimental 9B Embedding Bridge (Cached Case)'}
