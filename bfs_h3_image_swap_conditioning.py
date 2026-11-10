"""Native MiniMax H3 Image conditioning with color-marked character references."""
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning

COLORS = ('red', 'blue', 'green')
MODES = ('normal map guide', 'DWPose guide', 'translucent overlay guide', 'outline guide')


def image_swap_prompt(count, mode):
    if mode not in MODES:
        raise ValueError(f'Unknown guide mode: {mode}')
    if not 1 <= count <= len(COLORS):
        raise ValueError('Connect one to three character reference images')
    marks = {
        'normal map guide': ('normal map', 'each subject to replace is marked with its surface-normal map colorized in one solid color (its 3D surface, without its real colors)'),
        'DWPose guide': ('pose skeleton', 'each subject to replace is shown as its pose skeleton (body, hands and face) drawn in one color'),
        'translucent overlay guide': ('translucent overlay', 'each subject to replace is identified by a translucent color overlay; the appearance beneath the tint is the source appearance to replace'),
        'outline guide': ('outline', 'each subject to replace is identified by a thin colored outline around that subject'),
    }
    mark, explanation = marks[mode]
    definitions = '\n'.join(
        f'<Subject {j}> is the subject marked by the {COLORS[j-1]} {mark} in <Picture 1>; its appearance comes from <Picture {j+1}>, its position, pose, head orientation, gaze and facial expression from the {COLORS[j-1]} mark.'
        for j in range(1, count+1))
    replacements = ', '.join(f'the {COLORS[j-1]} subject replaced by <Picture {j+1}>' for j in range(1, count+1))
    retention = '\n'.join(
        f'<Subject {j}>: attribute_transfer - keeps the position, pose, head orientation, gaze and facial expression of the {COLORS[j-1]} mark and takes the face, hair, body and whole outfit of <Picture {j+1}>.'
        for j in range(1, count+1))
    return (
        'character_swap:\nsubject_definitions:\n'
        f'<Picture 1> is the source of the edit. In it, {explanation}; the mark says who changes, not where the new look must end, and everything else in the guide is the original image.\n'
        f'{definitions}\n'
        f'summary: [image editing + reference generation] The target image is <Picture 1> with {replacements}, following each mark\'s pose and expression. '
        'Every other person, the layout, the background, lighting, camera perspective and framing stay exactly as in <Picture 1>. '
        'Keep held objects and foreground occluders in their original positions, shapes and colors; they are separate from the subjects being replaced. '
        'Remove the normal-map colors, skeletons, overlays and outlines from the final image.\n'
        f'retention_analysis:\n{retention}\n'
        'Everyone else, held objects, foreground occluders, the layout and the background: fully_preserved.')


class BFSH3ImageCharacterSwapConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='BFSH3ImageCharacterSwapConditioning',
            display_name='BFS H3 Image Character Swap Conditioning',
            category='BFS/MiniMax H3',
            description='Native H3 Image character swap: the marked guide is Picture 1; red, blue and green subjects take their appearance from Pictures 2, 3 and 4. Produces native image reference latents and a still-image prompt.',
            inputs=[
                io.Clip.Input('clip'), io.Vae.Input('vae'),
                io.Image.Input('guide_image', tooltip='Marked hybrid image from BFS Video Segment Picker; becomes Picture 1.'),
                io.Autogrow.Input('references', template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input('reference', tooltip='Character appearance in color order: red, blue, green; becomes Picture 2, 3, 4.'),
                    prefix='reference_', min=1, max=3)),
                io.Combo.Input('mode', options=MODES, default=MODES[0]),
                io.Int.Input('width', default=0, min=0, max=16384, step=32, tooltip='0 uses the native H3 Image sizing based on the guide.'),
                io.Int.Input('height', default=0, min=0, max=16384, step=32),
                io.String.Input('prompt_override', default='', multiline=True, optional=True,
                                tooltip='Optional complete replacement prompt. Guide remains Picture 1; character references start at Picture 2.'),
            ],
            outputs=[io.Conditioning.Output(display_name='positive'), io.Latent.Output(display_name='latent'),
                     io.String.Output(display_name='prompt')])

    @classmethod
    def execute(cls, clip, vae, guide_image, references, mode=MODES[0], width=0, height=0, prompt_override=''):
        names = sorted(references, key=lambda name: int(name.rsplit('_', 1)[-1]))
        pictures = [references[name] for name in names if references[name] is not None]
        prompt = prompt_override.strip() or image_swap_prompt(len(pictures), mode)
        images = {'image_1': guide_image}
        images.update({f'image_{j}': picture for j, picture in enumerate(pictures, 2)})
        positive, latent = MiniMaxH3ImageConditioning.execute(
            clip=clip, vae=vae, prompt=prompt, width=width, height=height, images=images).args[:2]
        return io.NodeOutput(positive, latent, prompt)


NODE_CLASS_MAPPINGS = {'BFSH3ImageCharacterSwapConditioning': BFSH3ImageCharacterSwapConditioning}
NODE_DISPLAY_NAME_MAPPINGS = {'BFSH3ImageCharacterSwapConditioning': 'BFS H3 Image Character Swap Conditioning'}
