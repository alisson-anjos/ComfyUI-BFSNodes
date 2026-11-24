"""Native MiniMax H3 Image conditioning with color-marked character references."""
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning

COLORS = ('red', 'blue', 'green')
MODES = ('normal map guide', 'DWPose guide', 'translucent overlay guide', 'outline guide', 'colored hatching + contour')


def hatching_swap_prompt(count):
    definitions = ' '.join(
        f'<Subject {i+1}> is identified by the {color} hatching lines and matching {color} contour in <Picture 1>; '
        f'its complete appearance comes from <Picture {i+2}>.'
        for i, color in enumerate(COLORS[:count]))
    return (
        'character_swap: subject_definitions: <Picture 1> is the source image. Colored hatching lines and matching contours identify the subjects to replace; '
        'the visible pixels between the lines provide pose, expression, lighting and scene context. ' + definitions
        + ' summary: [image editing + reference generation] Replace each marked subject with its assigned reference character. '
        'Remove all colored lines and contours from the result. retention_analysis: Transfer the face, hair, skin, body proportions and complete outfit '
        'from each assigned reference. Preserve the source pose, hand positions, head orientation, gaze and facial expression. '
        'Adapt each replacement to the source lighting, shadows and exposure. Do not copy the pose or lighting of the reference. '
        'Do not retain the original marked subject\'s identity or clothing. Preserve every unmarked person, the background, held objects and foreground occluders. '
        'The hatching identifies who changes and does not constrain the replacement silhouette.')


def image_swap_prompt(count, mode):
    if mode not in MODES:
        raise ValueError(f'Unknown guide mode: {mode}')
    if not 1 <= count <= len(COLORS):
        raise ValueError('Connect one to three character reference images')
    if mode == 'colored hatching + contour':
        return hatching_swap_prompt(count)
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


def picker_subjects(picker_report):
    import json
    report = json.loads(picker_report)
    subjects = sorted(set(int(s) for s in report.get('active_subjects', report.get('subjects', []))))
    if any(s not in (1, 2, 3) for s in subjects):
        raise ValueError('Select at least one subject in the Picker, then use Apply fix. Red, blue and green use their matching reference slots.')
    return subjects, report.get('mark_style', 'colored hatching + contour')


class BFSH3ImageAutoCharacterSwapConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='BFSH3ImageAutoCharacterSwapConditioning',
            display_name='BFS H3 Image Character Swap • Auto Subjects', category='BFS/MiniMax H3',
            description='Reads selected colors and mark style from the Picker report. Only selected colors enter the native image conditioning. No manual subject count. Color slots remain fixed even when red is not selected.',
            inputs=[io.Clip.Input('clip'), io.Vae.Input('vae'), io.Image.Input('guide_image'),
                    io.String.Input('picker_report', force_input=True),
                    io.Image.Input('red_reference', optional=True, lazy=True),
                    io.Image.Input('blue_reference', optional=True, lazy=True),
                    io.Image.Input('green_reference', optional=True, lazy=True),
                    io.Int.Input('width', default=0, min=0, max=16384, step=32),
                    io.Int.Input('height', default=0, min=0, max=16384, step=32),
                    io.String.Input('prompt_override', default='', multiline=True, optional=True,
                        tooltip='Optional full replacement prompt. Blank uses the automatic prompt. Picture 1 remains the guide; selected references follow in red/blue/green order starting at Picture 2.')],
            outputs=[io.Conditioning.Output(display_name='positive'), io.Latent.Output(display_name='latent'),
                     io.String.Output(display_name='prompt')])

    @classmethod
    def check_lazy_status(cls, picker_report, red_reference=None, blue_reference=None, green_reference=None, **kwargs):
        ids, _ = picker_subjects(picker_report)
        refs = (red_reference, blue_reference, green_reference)
        return [f'{COLORS[s-1]}_reference' for s in ids if refs[s-1] is None]

    @classmethod
    def execute(cls, clip, vae, guide_image, picker_report, red_reference=None, blue_reference=None, green_reference=None, width=0, height=0, prompt_override=''):
        ids, style = picker_subjects(picker_report)
        if not ids:
            from comfy_execution.graph import ExecutionBlocker
            return io.NodeOutput(ExecutionBlocker(None), ExecutionBlocker(None),
                'Select subjects in the Picker, then click Apply fix and queue again.')
        refs = (red_reference, blue_reference, green_reference)
        pictures = [refs[s-1] for s in ids]
        if any(p is None for p in pictures):
            raise ValueError('Load a character reference for each selected Picker color.')
        if style == 'colored hatching + contour':
            mode = style
        elif style.startswith('openpose'):
            mode = 'DWPose guide'
        elif style == 'translucent overlay':
            mode = 'translucent overlay guide'
        elif style == 'colored outline':
            mode = 'outline guide'
        else:
            mode = 'normal map guide'
        prompt = image_swap_prompt(len(ids), mode)
        # Picture numbering is compact, while color identity follows the Picker.
        for i, color in enumerate(COLORS):
            import re
            prompt = re.sub(r'\b' + color + r'\b', f'__BFS_COLOR_{i}__', prompt)
        for i, subject in enumerate(ids):
            prompt = prompt.replace(f'__BFS_COLOR_{i}__', COLORS[subject-1])
        if prompt_override and prompt_override.strip():
            prompt = prompt_override
        images = {'image_1': guide_image}
        images.update({f'image_{i}': image for i, image in enumerate(pictures, 2)})
        positive, latent = MiniMaxH3ImageConditioning.execute(
            clip=clip, vae=vae, prompt=prompt, width=width, height=height, images=images).args[:2]
        return io.NodeOutput(positive, latent, prompt)


NODE_CLASS_MAPPINGS['BFSH3ImageAutoCharacterSwapConditioning'] = BFSH3ImageAutoCharacterSwapConditioning
NODE_DISPLAY_NAME_MAPPINGS['BFSH3ImageAutoCharacterSwapConditioning'] = 'BFS H3 Image Character Swap • Auto Subjects'



# ---------- multi face swap (H3 Image face swap LoRA): guide = Picture 1, faces = Picture 2.. ----------
FACE_COLORS = ('red', 'blue', 'green', 'yellow')


def h3image_face_prompt(colors, caption="v1"):
    """The face swap image LoRA's training caption, word for word. The guide is the first image the text encoder sees
    ("Picture 1: " label) but the caption calls it "the guide image" and numbers only the faces: <Picture 1> is the
    red face, <Picture 2> the blue one, and so on."""
    if str(caption).startswith("v2"):
        from .bfs_h3_multi_swap_prompt import face_swap_prompt_v2
        return face_swap_prompt_v2(list(colors), image=True)
    k = len(colors)
    subs = "\n".join(f"<Subject {i + 1}> is the person whose face is outlined by the thin {c} line in the guide image."
                     for i, c in enumerate(colors))
    refs = [f"<Picture {i + 1}>" for i in range(k)]
    refs = refs[0] if k == 1 else ", ".join(refs[:-1]) + " and " + refs[-1]
    repl = "; ".join(f"replace the face of <Subject {i + 1}> with the face shown in <Picture {i + 1}>" for i in range(k))
    return ("face_swap:\n"
            "subject_definitions: The guide image is the source image. Thin colored lines on the inner border of a face mark the "
            "faces to replace; the pixels inside and around the lines provide head pose, expression, gaze, lighting and scene "
            f"context.\n{subs}\n{refs} {'is a reference' if k == 1 else 'are references'}.\n\n"
            "summary: [image editing + reference generation]\n\n"
            "detailed_description:\n"
            f"In the target image, {repl}.\n"
            "Transfer only the facial identity from each assigned reference: eyes, eyebrows, nose, lips, face shape and skin "
            "tone; do not copy the hair, head shape, pose, expression or lighting of the reference.\n"
            "For each <Subject> in the guide image, preserve the source head orientation, gaze and facial expression, the hair, "
            "ears, body and clothing, and adapt each new face to the source lighting, shadows and exposure; do not mix "
            "features between subjects.\n"
            "Preserve every unmarked person, the background, held objects, foreground occluders and the framing.\n"
            "Remove all colored lines from the result.")



class BFSH3ImageFaceSwapAutoSubjects(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='BFSH3ImageFaceSwapAutoSubjects',
            display_name='BFS H3 Image Face Swap • Auto Subjects', category='BFS/MiniMax H3',
            description='Multi face swap on H3 Image (experimental): the guide from the BFS Video Segment Picker in '
                        "'sparse face contour' mode is the first image; the face of each picked color (red, blue, green, "
                        'yellow) follows. Writes the face swap LoRA training prompt (faces numbered <Picture 1>, <Picture 2>, ...).',
            inputs=[io.Clip.Input('clip'), io.Vae.Input('vae'), io.Image.Input('guide_image'),
                    io.String.Input('picker_report', force_input=True),
                    io.Image.Input('red_face', optional=True, lazy=True),
                    io.Image.Input('blue_face', optional=True, lazy=True),
                    io.Image.Input('green_face', optional=True, lazy=True),
                    io.Image.Input('yellow_face', optional=True, lazy=True),
                    io.Int.Input('width', default=0, min=0, max=16384, step=32),
                    io.Int.Input('height', default=0, min=0, max=16384, step=32),
                    io.String.Input('prompt_override', default='', multiline=True, optional=True,
                        tooltip='Optional full replacement prompt. Use the training wording: the guide is "the guide image" and the picked faces are <Picture 1>, <Picture 2>, ... in color order.'),
                    io.Combo.Input('caption', options=['v1 (current LoRAs)', 'v2 (color-shuffled datasets)'], default='v1 (current LoRAs)', optional=True,
                        tooltip='Training caption of the LoRA: v1 for the current face swap LoRAs, v2 for LoRAs trained on the color-shuffled datasets.')],
            outputs=[io.Conditioning.Output(display_name='positive'), io.Latent.Output(display_name='latent'),
                     io.String.Output(display_name='prompt')])

    @staticmethod
    def _ids(picker_report):
        import json
        try:
            r = json.loads(picker_report or '{}')
        except Exception:
            r = {}
        return sorted({int(s) for s in (r.get('active_subjects') or r.get('subjects') or []) if 1 <= int(s) <= 4})

    @classmethod
    def check_lazy_status(cls, picker_report, red_face=None, blue_face=None, green_face=None, yellow_face=None, **kwargs):
        faces = (red_face, blue_face, green_face, yellow_face)
        return [f'{FACE_COLORS[s-1]}_face' for s in cls._ids(picker_report) if faces[s-1] is None]

    @classmethod
    def execute(cls, clip, vae, guide_image, picker_report, red_face=None, blue_face=None, green_face=None, yellow_face=None,
                width=0, height=0, prompt_override='', caption='v1 (current LoRAs)'):
        ids = cls._ids(picker_report)
        if not ids:
            raise ValueError("Pick at least one person in the Video Segment Picker ('sparse face contour' mode), then run again.")
        faces = (red_face, blue_face, green_face, yellow_face)
        pictures = [faces[s-1] for s in ids]
        if any(p is None for p in pictures):
            raise ValueError('Connect a face picture for every picked color: ' + ', '.join(FACE_COLORS[s-1] for s in ids))
        prompt = (prompt_override or '').strip() or h3image_face_prompt([FACE_COLORS[s-1] for s in ids], caption)
        images = {'image_1': guide_image[:1]}
        images.update({f'image_{i}': p for i, p in enumerate(pictures, 2)})
        positive, latent = MiniMaxH3ImageConditioning.execute(
            clip=clip, vae=vae, prompt=prompt, width=width, height=height, images=images).args[:2]
        return io.NodeOutput(positive, latent, prompt)


NODE_CLASS_MAPPINGS['BFSH3ImageFaceSwapAutoSubjects'] = BFSH3ImageFaceSwapAutoSubjects
NODE_DISPLAY_NAME_MAPPINGS['BFSH3ImageFaceSwapAutoSubjects'] = 'BFS H3 Image Face Swap • Auto Subjects'
