import math
import torch
import numpy as np
from PIL import Image


def tensor_to_pil(img_tensor):
    """
    Converts a ComfyUI IMAGE tensor [H, W, C] float32 in range 0..1
    into a PIL image.
    """
    img = img_tensor.cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def pil_to_tensor(img_pil):
    """
    Converts a PIL image into a ComfyUI IMAGE tensor [H, W, C] float32 0..1.
    """
    arr = np.array(img_pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def fit_inside(src_w, src_h, max_w, max_h):
    """
    Resizes while preserving aspect ratio so the source fits entirely inside
    the target box, without cropping.
    """
    if src_w <= 0 or src_h <= 0:
        return 1, 1

    scale = min(max_w / src_w, max_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    return new_w, new_h


def aligned_offset(container_size, content_size, align):
    """
    Returns the position offset for start / center / end alignment.
    """
    if align == "start":
        return 0
    elif align == "end":
        return max(0, container_size - content_size)
    return max(0, (container_size - content_size) // 2)


def paste_with_alpha(dst, src_rgba, xy):
    """
    Pastes an image onto another, preserving alpha if present.
    """
    if src_rgba.mode == "RGBA":
        dst.paste(src_rgba, xy, src_rgba.split()[-1])
    else:
        dst.paste(src_rgba, xy)


def add_white_padding(img_rgba, pad_px=16):
    new_w = img_rgba.width + pad_px * 2
    new_h = img_rgba.height + pad_px * 2

    canvas = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 255))
    canvas.paste(img_rgba, (pad_px, pad_px), img_rgba if img_rgba.mode == "RGBA" else None)
    return canvas

class ReservedRegionFrameComposer:
    """
    ComfyUI node that composes a reserved region (left/right/top/bottom)
    into every frame, while preserving the original output frame size.

    Main features:
    - Keeps final output resolution equal to original frame resolution
    - Fits video content into remaining area without crop
    - Fills reserved region with chroma color
    - Places one or many face images in that region
    - Supports temporal distribution modes for face batches
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "face_images": ("IMAGE",),

                "region_position": (
                    ["left", "right", "top", "bottom"],
                    {"default": "left"}
                ),

                "region_size_px": (
                    "INT",
                    {"default": 320, "min": 8, "max": 8192, "step": 1}
                ),

                "face_distribution": (
                    [
                        "single_first",
                        "one_face_per_frame",
                        "one_face_per_interval",
                        "all_faces_every_frame"
                    ],
                    {"default": "one_face_per_interval"}
                ),

                "interval_frames": (
                    "INT",
                    {"default": 12, "min": 1, "max": 1000000, "step": 1}
                ),

                "overflow_mode": (
                    ["loop", "clamp", "error"],
                    {"default": "loop"}
                ),

                "stack_direction": (
                    ["auto", "vertical", "horizontal", "grid"],
                    {"default": "auto"}
                ),

                "face_scale_pct": (
                    "FLOAT",
                    {"default": 90.0, "min": 1.0, "max": 100.0, "step": 1.0}
                ),

                "face_padding_px": (
                    "INT",
                    {"default": 12, "min": 0, "max": 2048, "step": 1}
                ),

                "face_gap_px": (
                    "INT",
                    {"default": 12, "min": 0, "max": 2048, "step": 1}
                ),

                "face_align_main": (
                    ["start", "center", "end"],
                    {"default": "center"}
                ),

                "face_align_cross": (
                    ["start", "center", "end"],
                    {"default": "center"}
                ),

                "chroma_r": (
                    "INT",
                    {"default": 0, "min": 0, "max": 255, "step": 1}
                ),
                "chroma_g": (
                    "INT",
                    {"default": 255, "min": 0, "max": 255, "step": 1}
                ),
                "chroma_b": (
                    "INT",
                    {"default": 0, "min": 0, "max": 255, "step": 1}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames_out",)
    FUNCTION = "process"
    CATEGORY = "video/composition"

    def _prepare_face(self, face_tensor):
        """
        Converts input tensor to RGBA PIL so alpha transparency is preserved.
        """
        face = tensor_to_pil(face_tensor).convert("RGBA")
        face = add_white_padding(face, 16)
        return face

    def _resolve_faces_for_frame(
        self,
        faces_pil,
        frame_idx,
        face_distribution,
        interval_frames,
        overflow_mode,
    ):
        """
        Returns the list of face images that should be used for the current frame.
        """
        total = len(faces_pil)

        if total == 0:
            raise ValueError("No face images were provided.")

        if face_distribution == "single_first":
            return [faces_pil[0]]

        if face_distribution == "one_face_per_frame":
            idx = frame_idx
            if idx < total:
                return [faces_pil[idx]]

            if overflow_mode == "loop":
                return [faces_pil[idx % total]]
            elif overflow_mode == "clamp":
                return [faces_pil[-1]]
            else:
                raise ValueError(
                    f"Face batch is too small for one_face_per_frame. "
                    f"Frame index {frame_idx} requires face index {idx}, "
                    f"but only {total} face images were provided."
                )

        if face_distribution == "one_face_per_interval":
            idx = frame_idx // interval_frames
            if idx < total:
                return [faces_pil[idx]]

            if overflow_mode == "loop":
                return [faces_pil[idx % total]]
            elif overflow_mode == "clamp":
                return [faces_pil[-1]]
            else:
                raise ValueError(
                    f"Face batch is too small for one_face_per_interval. "
                    f"Frame index {frame_idx} requires interval face index {idx}, "
                    f"but only {total} face images were provided."
                )

        if face_distribution == "all_faces_every_frame":
            return faces_pil

        raise ValueError(f"Unsupported face_distribution mode: {face_distribution}")

    def _resize_single_face(
        self,
        face,
        region_w,
        region_h,
        face_scale_pct,
        face_padding_px,
    ):
        """
        Resizes one face to fit inside the usable region area.
        """
        usable_w = max(1, region_w - 2 * face_padding_px)
        usable_h = max(1, region_h - 2 * face_padding_px)

        target_w = max(1, int(round(usable_w * (face_scale_pct / 100.0))))
        target_h = max(1, int(round(usable_h * (face_scale_pct / 100.0))))

        fw, fh = face.size
        tw, th = fit_inside(fw, fh, target_w, target_h)
        return face.resize((tw, th), Image.LANCZOS)

    def _layout_faces_stack(
        self,
        faces_pil,
        region_w,
        region_h,
        face_scale_pct,
        face_padding_px,
        face_gap_px,
        stack_direction,
    ):
        """
        Prepares resized face images for stack mode.
        Returns a tuple describing the chosen layout and resized images.
        """
        usable_w = max(1, region_w - 2 * face_padding_px)
        usable_h = max(1, region_h - 2 * face_padding_px)

        count = len(faces_pil)
        if count == 0:
            return None

        if stack_direction == "auto":
            stack_direction = "vertical" if region_h >= region_w else "horizontal"

        items = []

        if stack_direction == "vertical":
            slot_h = max(1, (usable_h - face_gap_px * (count - 1)) // count)
            slot_w = usable_w

            for face in faces_pil:
                fw, fh = face.size
                tw, th = fit_inside(
                    fw,
                    fh,
                    max(1, int(round(slot_w * (face_scale_pct / 100.0)))),
                    max(1, int(round(slot_h * (face_scale_pct / 100.0))))
                )
                items.append(face.resize((tw, th), Image.LANCZOS))

            return ("vertical", items, usable_w, usable_h)

        if stack_direction == "horizontal":
            slot_w = max(1, (usable_w - face_gap_px * (count - 1)) // count)
            slot_h = usable_h

            for face in faces_pil:
                fw, fh = face.size
                tw, th = fit_inside(
                    fw,
                    fh,
                    max(1, int(round(slot_w * (face_scale_pct / 100.0)))),
                    max(1, int(round(slot_h * (face_scale_pct / 100.0))))
                )
                items.append(face.resize((tw, th), Image.LANCZOS))

            return ("horizontal", items, usable_w, usable_h)

        if stack_direction == "grid":
            cols = max(1, math.ceil(math.sqrt(count)))
            rows = max(1, math.ceil(count / cols))

            cell_w = max(1, (usable_w - face_gap_px * (cols - 1)) // cols)
            cell_h = max(1, (usable_h - face_gap_px * (rows - 1)) // rows)

            for face in faces_pil:
                fw, fh = face.size
                tw, th = fit_inside(
                    fw,
                    fh,
                    max(1, int(round(cell_w * (face_scale_pct / 100.0)))),
                    max(1, int(round(cell_h * (face_scale_pct / 100.0))))
                )
                items.append(face.resize((tw, th), Image.LANCZOS))

            return ("grid", items, usable_w, usable_h, cols, rows)

        raise ValueError(f"Unsupported stack_direction: {stack_direction}")

    def _paste_single_face(
        self,
        canvas,
        face,
        region_x,
        region_y,
        region_w,
        region_h,
        region_position,
        face_padding_px,
        face_align_main,
        face_align_cross,
    ):
        """
        Pastes one face inside the reserved region.
        """
        tw, th = face.size
        area_w = max(1, region_w - 2 * face_padding_px)
        area_h = max(1, region_h - 2 * face_padding_px)

        if region_position in ["left", "right"]:
            local_x = face_padding_px + aligned_offset(area_w, tw, face_align_cross)
            local_y = face_padding_px + aligned_offset(area_h, th, face_align_main)
        else:
            local_x = face_padding_px + aligned_offset(area_w, tw, face_align_main)
            local_y = face_padding_px + aligned_offset(area_h, th, face_align_cross)

        paste_with_alpha(canvas, face, (region_x + local_x, region_y + local_y))

    def _paste_stack_faces(
        self,
        canvas,
        faces_pil,
        region_x,
        region_y,
        region_w,
        region_h,
        face_scale_pct,
        face_padding_px,
        face_gap_px,
        face_align_main,
        face_align_cross,
        stack_direction,
    ):
        """
        Pastes multiple faces inside the reserved region.
        """
        layout = self._layout_faces_stack(
            faces_pil=faces_pil,
            region_w=region_w,
            region_h=region_h,
            face_scale_pct=face_scale_pct,
            face_padding_px=face_padding_px,
            face_gap_px=face_gap_px,
            stack_direction=stack_direction,
        )

        if layout is None:
            return

        usable_x = region_x + face_padding_px
        usable_y = region_y + face_padding_px

        kind = layout[0]

        if kind == "vertical":
            _, items, usable_w, usable_h = layout
            total_h = sum(img.size[1] for img in items) + face_gap_px * (len(items) - 1)
            start_y = usable_y + aligned_offset(usable_h, total_h, face_align_main)

            y = start_y
            for img in items:
                x = usable_x + aligned_offset(usable_w, img.size[0], face_align_cross)
                paste_with_alpha(canvas, img, (x, y))
                y += img.size[1] + face_gap_px
            return

        if kind == "horizontal":
            _, items, usable_w, usable_h = layout
            total_w = sum(img.size[0] for img in items) + face_gap_px * (len(items) - 1)
            start_x = usable_x + aligned_offset(usable_w, total_w, face_align_main)

            x = start_x
            for img in items:
                y = usable_y + aligned_offset(usable_h, img.size[1], face_align_cross)
                paste_with_alpha(canvas, img, (x, y))
                x += img.size[0] + face_gap_px
            return

        if kind == "grid":
            _, items, usable_w, usable_h, cols, rows = layout

            cell_w = max(1, (usable_w - face_gap_px * (cols - 1)) // cols)
            cell_h = max(1, (usable_h - face_gap_px * (rows - 1)) // rows)

            grid_w = cols * cell_w + (cols - 1) * face_gap_px
            grid_h = rows * cell_h + (rows - 1) * face_gap_px

            start_x = usable_x + aligned_offset(usable_w, grid_w, face_align_main)
            start_y = usable_y + aligned_offset(usable_h, grid_h, face_align_cross)

            for idx, img in enumerate(items):
                row = idx // cols
                col = idx % cols

                cell_x = start_x + col * (cell_w + face_gap_px)
                cell_y = start_y + row * (cell_h + face_gap_px)

                x = cell_x + (cell_w - img.size[0]) // 2
                y = cell_y + (cell_h - img.size[1]) // 2
                paste_with_alpha(canvas, img, (x, y))
            return

        raise ValueError(f"Unsupported stack layout kind: {kind}")

    def process(
        self,
        frames,
        face_images,
        region_position,
        region_size_px,
        face_distribution,
        interval_frames,
        overflow_mode,
        stack_direction,
        face_scale_pct,
        face_padding_px,
        face_gap_px,
        face_align_main,
        face_align_cross,
        chroma_r,
        chroma_g,
        chroma_b,
    ):
        """
        Main node execution.
        """
        if len(frames.shape) != 4:
            raise ValueError(
                "Input 'frames' must be a batch IMAGE tensor with shape [N, H, W, C]."
            )

        if len(face_images.shape) != 4 or face_images.shape[0] < 1:
            raise ValueError(
                "Input 'face_images' must be a valid IMAGE batch with at least one image."
            )

        n, orig_h, orig_w, _ = frames.shape
        face_count = face_images.shape[0]

        faces_pil = [self._prepare_face(face_images[i]) for i in range(face_count)]

        if region_position in ["left", "right"]:
            max_region_size = max(1, orig_w - 1)
        else:
            max_region_size = max(1, orig_h - 1)

        region_size_px = max(1, min(region_size_px, max_region_size))

        if region_position in ["left", "right"]:
            region_w = region_size_px
            region_h = orig_h
            video_max_w = orig_w - region_size_px
            video_max_h = orig_h
        else:
            region_w = orig_w
            region_h = region_size_px
            video_max_w = orig_w
            video_max_h = orig_h - region_size_px

        if video_max_w < 1 or video_max_h < 1:
            raise ValueError(
                "The reserved region size is too large for the current frame resolution."
            )

        fitted_video_w, fitted_video_h = fit_inside(orig_w, orig_h, video_max_w, video_max_h)

        chroma_rgba = (chroma_r, chroma_g, chroma_b, 255)

        out_frames = []

        for i in range(n):
            frame_pil_src = tensor_to_pil(frames[i]).convert("RGB")
            frame_pil = frame_pil_src.resize((fitted_video_w, fitted_video_h), Image.LANCZOS)

            canvas = Image.new("RGBA", (orig_w, orig_h), color=(0, 0, 0, 255))

            if region_position == "left":
                region_x, region_y = 0, 0
                video_x, video_y = region_size_px, (orig_h - fitted_video_h) // 2

            elif region_position == "right":
                region_x, region_y = orig_w - region_size_px, 0
                video_x, video_y = 0, (orig_h - fitted_video_h) // 2

            elif region_position == "top":
                region_x, region_y = 0, 0
                video_x, video_y = (orig_w - fitted_video_w) // 2, region_size_px

            else:  # bottom
                region_x, region_y = 0, orig_h - region_size_px
                video_x, video_y = (orig_w - fitted_video_w) // 2, 0

            region_img = Image.new("RGBA", (region_w, region_h), color=chroma_rgba)
            canvas.paste(region_img, (region_x, region_y))

            canvas.paste(frame_pil.convert("RGBA"), (video_x, video_y))

            faces_for_frame = self._resolve_faces_for_frame(
                faces_pil=faces_pil,
                frame_idx=i,
                face_distribution=face_distribution,
                interval_frames=interval_frames,
                overflow_mode=overflow_mode,
            )

            if len(faces_for_frame) == 1:
                single_face = self._resize_single_face(
                    face=faces_for_frame[0],
                    region_w=region_w,
                    region_h=region_h,
                    face_scale_pct=face_scale_pct,
                    face_padding_px=face_padding_px,
                )

                self._paste_single_face(
                    canvas=canvas,
                    face=single_face,
                    region_x=region_x,
                    region_y=region_y,
                    region_w=region_w,
                    region_h=region_h,
                    region_position=region_position,
                    face_padding_px=face_padding_px,
                    face_align_main=face_align_main,
                    face_align_cross=face_align_cross,
                )

            else:
                effective_stack_direction = stack_direction
                if stack_direction == "auto":
                    effective_stack_direction = (
                        "vertical" if region_position in ["left", "right"] else "horizontal"
                    )

                self._paste_stack_faces(
                    canvas=canvas,
                    faces_pil=faces_for_frame,
                    region_x=region_x,
                    region_y=region_y,
                    region_w=region_w,
                    region_h=region_h,
                    face_scale_pct=face_scale_pct,
                    face_padding_px=face_padding_px,
                    face_gap_px=face_gap_px,
                    face_align_main=face_align_main,
                    face_align_cross=face_align_cross,
                    stack_direction=effective_stack_direction,
                )

            out_frames.append(pil_to_tensor(canvas.convert("RGB")))

        out = torch.stack(out_frames, dim=0)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "ReservedRegionFrameComposer": ReservedRegionFrameComposer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReservedRegionFrameComposer": "Reserved Region Frame Composer"
}