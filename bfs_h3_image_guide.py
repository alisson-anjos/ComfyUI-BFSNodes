"""Aligned image guide for the native MiniMax H3 Image reference stream."""
import copy

import comfy.utils
from comfy_api.latest import io

MARKER = 'bfs_h3_image_guide'


def align_guide_layout(layout, refs):
    result = copy.copy(layout)
    result.position_ids = layout.position_ids.clone()
    target = next((a, b) for a, b, kind in layout.segments if kind == 'video')
    image_refs = iter(ref for ref in refs if ref['kind'] == 'image')
    for a, b, kind in layout.segments:
        if kind != 'ref_img':
            continue
        ref = next(image_refs)
        if ref.get(MARKER):
            ta, tb = target
            if b - a != tb - ta:
                raise ValueError('H3 Image guide must match the target canvas')
            result.position_ids[a:b] = layout.position_ids[ta:tb]
    return result


def patch_image_guide(model):
    patched = model.clone()
    if patched.model_options.get(MARKER):
        return patched
    previous = patched.model_options.get('model_function_wrapper')

    def wrapper(model_function, args):
        conds = args['c']
        payload = conds.get('minimax_payload', {})
        refs = payload.get('refs', [])
        if any(ref.get(MARKER) for ref in refs):
            layout = align_guide_layout(payload['layout'], refs)
            args = dict(args, c=dict(conds, minimax_payload=dict(payload, layout=layout)))
        if previous is not None:
            return previous(model_function, args)
        return model_function(args['input'], args['timestep'], **args['c'])

    patched.set_model_unet_function_wrapper(wrapper)
    patched.model_options[MARKER] = True
    return patched


class BFSH3ImageAddGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='BFSH3ImageAddGuide', display_name='Add Guide for MiniMax H3 Image (BFS)',
            category='BFS/MiniMax H3',
            description='Experimental aligned guide for H3 Image. Uses a frozen image reference with target-aligned positions. Connect the returned MODEL and positive to the sampler; use the native H3 Image latent.',
            inputs=[io.Model.Input('model'), io.Conditioning.Input('positive'), io.Vae.Input('vae'),
                    io.Image.Input('guide'), io.Latent.Input('latent')],
            outputs=[io.Model.Output(), io.Conditioning.Output(display_name='positive')])

    @classmethod
    def execute(cls, model, positive, vae, guide, latent):
        samples = latent['samples']
        if samples.is_nested or samples.ndim != 5 or samples.shape[1:3] != (24, 1):
            raise ValueError('Use the LATENT from MiniMax H3 Image Conditioning or Empty MiniMax H3 Image Latent')
        if model.model.diffusion_model.__class__.__name__ != 'MiniMaxH3ImageModel':
            raise ValueError('This guide requires the native MiniMax H3 Image model')
        h, w = samples.shape[-2:]
        pixels = comfy.utils.common_upscale(guide[:1, ..., :3].movedim(-1, 1), w * 16, h * 16, 'bicubic', 'center').movedim(1, -1)
        z = vae.encode(pixels)
        if z.ndim != 5 or tuple(z.shape[1:]) != (24, 1, h, w):
            raise ValueError('Guide VAE must produce native H3 Image latents matching the target')
        guide_ref = {'kind': 'image', 'latent_h': h, 'latent_w': w, 'latent': z, MARKER: True}
        out = []
        for cond, options in positive:
            refs = [r for r in options.get('minimax_refs', []) if not r.get(MARKER)]
            updated = dict(options, minimax_refs=refs + [guide_ref])
            out.append([cond, updated])
        return io.NodeOutput(patch_image_guide(model), out)


NODE_CLASS_MAPPINGS = {'BFSH3ImageAddGuide': BFSH3ImageAddGuide}
NODE_DISPLAY_NAME_MAPPINGS = {'BFSH3ImageAddGuide': 'Add Guide for MiniMax H3 Image (BFS)'}
