"""Optional image references and a compact character-swap comparison."""
import os
import json
import numpy as np
import torch
import folder_paths
from comfy_api.latest import io
from nodes import LoadImage
from PIL import Image, ImageDraw


class BFSOptionalReferenceImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = folder_paths.filter_files_content_types(os.listdir(folder_paths.get_input_directory()), ['image'])
        return io.Schema(node_id='BFSOptionalReferenceImage', display_name='BFS Optional Reference Image',
            category='BFS/image', description='Load a character reference and resize it proportionally using the native Resize Image/Mask implementation. An unused [none] slot returns no image and skips resizing.',
            inputs=[io.Combo.Input('image', options=['[none]'] + sorted(files), default='[none]', upload=io.UploadType.image),
                    io.Float.Input('megapixels', default=4.0, min=0.1, max=16.0, step=0.1,
                        tooltip='Editable target area in megapixels. 4 is only the default; width and height follow the original aspect ratio.'),
                    io.Combo.Input('scale_method', options=['nearest-exact', 'bilinear', 'area', 'bicubic', 'lanczos'], default='lanczos',
                        tooltip='Editable interpolation method used by native Resize Image/Mask.')],
            outputs=[io.Image.Output()])

    @classmethod
    def fingerprint_inputs(cls, image='[none]', megapixels=4.0, scale_method='lanczos'):
        if image == '[none]':
            return image
        return (LoadImage.IS_CHANGED(image), megapixels, scale_method)

    @classmethod
    def execute(cls, image='[none]', megapixels=4.0, scale_method='lanczos'):
        if image == '[none]':
            return io.NodeOutput(None)
        from comfy_extras.nodes_post_processing import ResizeImageMaskNode, ResizeType
        pixels = LoadImage().load_image(image)[0][:1]
        pixels = ResizeImageMaskNode.execute(pixels, scale_method=scale_method,
            resize_type={'resize_type': ResizeType.SCALE_TOTAL_PIXELS, 'megapixels': megapixels}).args[0]
        filename, _ = folder_paths.annotated_filepath(image)
        return io.NodeOutput(pixels, ui={'images': [{'filename': os.path.basename(filename),
            'subfolder': os.path.dirname(filename), 'type': 'input'}]})


class BFSH3ImageSwapComparison(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='BFSH3ImageSwapComparison', display_name='BFS H3 Image Swap Comparison', category='BFS/MiniMax H3',
            inputs=[io.Image.Input('original'), io.Image.Input('guide'), io.Image.Input('result'),
                    io.String.Input('picker_report', force_input=True),
                    io.Image.Input('red_reference', optional=True, lazy=True), io.Image.Input('blue_reference', optional=True, lazy=True),
                    io.Image.Input('green_reference', optional=True, lazy=True)], outputs=[io.Image.Output()])

    @classmethod
    def check_lazy_status(cls, picker_report, red_reference=None, blue_reference=None, green_reference=None, **kwargs):
        report = json.loads(picker_report)
        ids = report.get('active_subjects', report.get('subjects', []))
        names = ('red_reference', 'blue_reference', 'green_reference')
        refs = (red_reference, blue_reference, green_reference)
        return [names[int(s)-1] for s in ids if int(s) in (1, 2, 3) and refs[int(s)-1] is None]

    @classmethod
    def execute(cls, original, guide, result, picker_report, red_reference=None, blue_reference=None, green_reference=None):
        report = json.loads(picker_report)
        ids = report.get('active_subjects', report.get('subjects', []))
        refs = (red_reference, blue_reference, green_reference)
        panels = [('Original', original), ('Guide / Picture 1', guide)]
        for picture, sid in enumerate(sorted(set(map(int, ids))), 2):
            if sid not in (1, 2, 3):
                raise ValueError('Invalid Picker subject ID')
            ref = refs[sid-1]
            if ref is not None:
                panels.append((f'{("Red", "Blue", "Green")[sid-1]} / Picture {picture}', ref))
        panels.append(('Result', result))
        rendered = []
        for label, tensor in panels:
            array = (tensor[0, ..., :3].detach().cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            im = Image.fromarray(array)
            height = 720
            im = im.resize((max(1, round(im.width * height / im.height)), height), Image.Resampling.LANCZOS)
            canvas = Image.new('RGB', (im.width, height + 24), 'black')
            canvas.paste(im, (0, 24)); ImageDraw.Draw(canvas).text((6, 5), label, fill='white')
            rendered.append(canvas)
        canvas = Image.new('RGB', (sum(im.width for im in rendered), 744), 'black')
        x = 0
        for im in rendered:
            canvas.paste(im, (x, 0)); x += im.width
        return io.NodeOutput(torch.from_numpy(np.asarray(canvas).copy()).float()[None] / 255)


NODE_CLASS_MAPPINGS = {'BFSOptionalReferenceImage': BFSOptionalReferenceImage, 'BFSH3ImageSwapComparison': BFSH3ImageSwapComparison}
NODE_DISPLAY_NAME_MAPPINGS = {'BFSOptionalReferenceImage': 'BFS Optional Reference Image', 'BFSH3ImageSwapComparison': 'BFS H3 Image Swap Comparison'}
