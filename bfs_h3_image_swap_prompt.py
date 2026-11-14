"""Character-swap prompts for MiniMax H3 Image."""
from comfy_api.latest import io
from .bfs_h3_image_swap_conditioning import hatching_swap_prompt

COLORS = ('red', 'blue', 'green')
MARKS = {
    'normal map': ('surface-normal map', 'normal-map marking'),
    'DWPose': ('pose skeleton showing the body, hands and face', 'pose-skeleton marking'),
    'colored hatching + contour': ('hatching lines and matching contour', 'hatching'),
}


class BFSH3ImageCharacterSwapPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='BFSH3ImageCharacterSwapPrompt',
            display_name='BFS H3 Image Character Swap Prompt',
            category='BFS/MiniMax H3',
            description='Generate an H3 Image character-swap prompt. Picture 1 is the marked source image. Red, blue and green subjects use Pictures 2, 3 and 4.',
            inputs=[
                io.Combo.Input('mode', options=list(MARKS), default='normal map'),
                io.Int.Input('subjects', default=1, min=1, max=3, step=1,
                             tooltip='Number of marked subjects. References follow red, blue, green order.'),
            ],
            outputs=[io.String.Output(display_name='prompt')],
        )

    @classmethod
    def execute(cls, mode='normal map', subjects=1):
        if not 1 <= subjects <= len(COLORS):
            raise ValueError('Choose one to three marked subjects')
        if mode == 'colored hatching + contour':
            return io.NodeOutput(hatching_swap_prompt(subjects))
        mark, removal = MARKS[mode]
        definitions = []
        retention = []
        replacements = []
        for index, color in enumerate(COLORS[:subjects], 1):
            reference = index + 1
            definitions.append(
                f'<Subject {index}> is the subject marked by the {color} {mark} in <Picture 1>. '
                f'Its identity, face, hair or fur, skin, body shape, proportions and complete outfit come from <Picture {reference}>. '
                f'Its position, pose, head orientation, gaze and expression follow the {color}-marked subject in <Picture 1>.'
            )
            retention.append(
                f'<Subject {index}>: attribute_transfer - preserves the {color}-marked subject\'s position, pose, head orientation, gaze and expression '
                f'while taking its complete appearance from <Picture {reference}>.'
            )
            replacements.append(f'the {color}-marked subject with <Subject {index}> from <Picture {reference}>')
        prompt = (
            'character_swap:\nsubject_definitions:\n'
            f'<Picture 1> is the source image to edit. Each subject to replace is identified by a colored {mark}. '
            'The mark identifies the selected subject. The replacement follows the reference character\'s body shape and proportions.\n'
            + '\n'.join(definitions)
            + '\nsummary: [image editing + reference generation] Replace only '
            + ', '.join(replacements)
            + ' in <Picture 1>. Preserve every other person and animal, the background, lighting, layout, camera perspective and framing. '
            'Preserve held objects and foreground occluders in their original positions, shapes and colors. '
            f'Remove every {removal} from the final image.\nretention_analysis:\n'
            + '\n'.join(retention)
            + '\nAll other subjects, held objects, foreground occluders, the layout and the background: fully_preserved.'
        )
        return io.NodeOutput(prompt)


NODE_CLASS_MAPPINGS = {'BFSH3ImageCharacterSwapPrompt': BFSH3ImageCharacterSwapPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {'BFSH3ImageCharacterSwapPrompt': 'BFS H3 Image Character Swap Prompt'}
