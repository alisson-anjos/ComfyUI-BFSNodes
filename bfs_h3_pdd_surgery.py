"""Reversible scaling of H3 PDD output-head row blocks using native patches."""
import math
from comfy_api.latest import io


class BFSH3PDDSurgery(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='BFSH3PDDSurgery', display_name='BFS H3 PDD Surgery',
            category='BFS/MiniMax H3',
            description='Experimental PDD head scaling. Base is the shared output head; offsets are learned differences, not standalone full heads. All strengths 1 preserve the original model. Zero offsets do not undo distillation. Connect after LoRA loaders.',
            inputs=[io.Model.Input('model'),
                    io.Float.Input('base_strength', default=1.0, min=0.0, max=2.0, step=0.05),
                    io.Float.Input('offset_strength', default=1.0, min=0.0, max=2.0, step=0.05)] + [
                io.Float.Input(f'offset_{i}', default=1.0, min=0.0, max=2.0, step=0.05,
                               tooltip='Multiplier for this offset block; unused blocks are ignored. Multiplied by offset_strength.')
                for i in range(1, 8)],
            outputs=[io.Model.Output(), io.String.Output(display_name='report')])

    @classmethod
    def execute(cls, model, base_strength=1.0, offset_strength=1.0, **kwargs):
        final = model.get_model_object('diffusion_model.final_layer')
        if not hasattr(final, 'video_dim') or not hasattr(final, 'video_out'):
            raise ValueError('Connect a native MiniMax H3 model with PDD output heads')
        heads = final.video_out.weight.shape[0] // final.video_dim
        if not 2 <= heads <= 8:
            raise ValueError(f'Expected 2–8 PDD heads; found {heads}')
        offsets = [offset_strength * kwargs.get(f'offset_{i}', 1.0) for i in range(1, heads)]
        factors = [base_strength] + offsets
        if not all(math.isfinite(v) for v in factors):
            raise ValueError('PDD strengths must be finite')
        result = model.clone()
        if any(v != 1.0 for v in factors):
            # Scale row blocks through ModelPatcher, including preceding LoRA patches.
            # No dense delta allocation or direct modification of shared model weights.
            for stream in ('video_out', 'audio_out'):
                head = getattr(final, stream, None)
                if head is None:
                    continue
                for param in ('weight', 'bias'):
                    value = getattr(head, param, None)
                    if value is None:
                        continue
                    if value.shape[0] % heads:
                        raise ValueError(f'{stream}.{param} rows are not divisible by the PDD head count')
                    rows = value.shape[0] // heads
                    for index, factor in enumerate(factors):
                        if factor == 1.0:
                            continue
                        key = (f'diffusion_model.final_layer.{stream}.{param}', (0, index * rows, rows))
                        accepted = result.add_patches({key: ('diff', (None,))}, strength_patch=0.0, strength_model=factor)
                        if key not in accepted:
                            raise ValueError('The model did not accept all PDD head patches')
        report = f'{heads} PDD heads: shared base={base_strength:g}; offset multipliers={offsets}. Applies to image/video and audio when present. Sigmas unchanged.'
        return io.NodeOutput(result, report)


NODE_CLASS_MAPPINGS = {'BFSH3PDDSurgery': BFSH3PDDSurgery}
NODE_DISPLAY_NAME_MAPPINGS = {'BFSH3PDDSurgery': 'BFS H3 PDD Surgery'}
