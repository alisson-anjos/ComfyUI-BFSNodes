import math
import torch
from PIL import Image

from .util import (
    tensor_to_pil,
    pil_to_tensor,
    fit_inside,
    aligned_offset,
    paste_with_alpha,
    add_white_padding,
)


# ---------------------------------------------------------------------------
# FrameRangedFaceLoader
# ---------------------------------------------------------------------------

class FrameRangedFaceLoader:
    """
    Wraps a single face IMAGE with a frame range [frame_start, frame_end].
    Returns a FACE_SEQUENCE — a list of dicts used by ReservedRegionFrameComposer.

    frame_end = -1 means "until the last frame" (no upper limit).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "frame_start": (
                    "INT",
                    {"default": 0, "min": 0, "max": 999999, "step": 1},
                ),
                "frame_end": (
                    "INT",
                    {"default": -1, "min": -1, "max": 999999, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("FACE_SEQUENCE",)
    RETURN_NAMES = ("face_sequence",)
    FUNCTION = "load"
    CATEGORY = "video/composition"

    def load(self, image, frame_start, frame_end):
        """
        image: IMAGE tensor [N, H, W, C] — only the first image is used.
        Returns a FACE_SEQUENCE list with a single entry.
        """
        face_pil = tensor_to_pil(image[0]).convert("RGBA")
        face_pil = add_white_padding(face_pil, 16)

        entry = {
            "image": face_pil,
            "frame_start": frame_start,
            "frame_end": frame_end,  # -1 == no upper limit
        }
        return ([entry],)


# ---------------------------------------------------------------------------
# FaceSequenceBatch
# ---------------------------------------------------------------------------

class FaceSequenceBatch:
    """
    Joins two FACE_SEQUENCE lists into one.
    Chain multiple nodes to build batches with many ranges.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_sequence_a": ("FACE_SEQUENCE",),
                "face_sequence_b": ("FACE_SEQUENCE",),
            }
        }

    RETURN_TYPES = ("FACE_SEQUENCE",)
    RETURN_NAMES = ("face_sequence",)
    FUNCTION = "batch"
    CATEGORY = "video/composition"

    def batch(self, face_sequence_a, face_sequence_b):
        return (face_sequence_a + face_sequence_b,)


# ---------------------------------------------------------------------------
# ReservedRegionFrameComposer
# ---------------------------------------------------------------------------

class ReservedRegionFrameComposer:
    """
    ComfyUI node that composes a reserved region (left/right/top/bottom)
    into every frame, while preserving the original output frame size.

    Main features:
    - Keeps final output resolution equal to original frame resolution
    - Fits video content into remaining area without crop
    - Fills reserved region with chroma color
    - Places one or many face images in that region
    - Supports temporal distribution modes for face batches (IMAGE input)
    - Supports per-frame ranges via FACE_SEQUENCE input
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),

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
            },
            "optional": {
                # When connected, overrides face_images completely.
                # face_distribution / interval_frames are ignored.
                "face_sequence": ("FACE_SEQUENCE",),

                # Legacy flat batch input (IMAGE). Used only when
                # face_sequence is NOT connected.
                "face_images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames_out",)
    FUNCTION = "process"
    CATEGORY = "video/composition"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_face(self, face_tensor):
        """Converts input tensor to RGBA PIL so alpha transparency is preserved."""
        face = tensor_to_pil(face_tensor).convert("RGBA")
        face = add_white_padding(face, 16)
        return face

    # --- Resolution for plain IMAGE input ---

    def _resolve_faces_for_frame_image(
        self,
        faces_pil,
        frame_idx,
        face_distribution,
        interval_frames,
        overflow_mode,
    ):
        """
        Returns the list of face PIL images for the current frame
        when a plain IMAGE batch is used (legacy mode).
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

    # --- Resolution for FACE_SEQUENCE input ---

    def _resolve_faces_for_frame_sequence(self, face_sequence, frame_idx, overflow_mode):
        """
        Returns the list of face PIL images for the current frame
        when a FACE_SEQUENCE is used (range mode).

        Collects all entries whose [frame_start, frame_end] covers frame_idx.
        frame_end == -1 means no upper limit.

        If no entry matches:
          - loop: wraps around using sorted order by frame_start
          - clamp: uses the last entry (highest frame_start)
          - error: raises ValueError
        """
        matched = [
            e for e in face_sequence
            if e["frame_start"] <= frame_idx and (
                e["frame_end"] == -1 or frame_idx <= e["frame_end"]
            )
        ]

        if matched:
            return [e["image"] for e in matched]

        # Fallback
        sorted_seq = sorted(face_sequence, key=lambda e: e["frame_start"])
        if not sorted_seq:
            raise ValueError("FACE_SEQUENCE is empty.")

        if overflow_mode == "loop":
            total = len(sorted_seq)
            return [sorted_seq[frame_idx % total]["image"]]
        elif overflow_mode == "clamp":
            return [sorted_seq[-1]["image"]]
        else:
            raise ValueError(
                f"No face entry covers frame {frame_idx} and overflow_mode is 'error'."
            )

    # --- Layout helpers ---

    def _resize_single_face(self, face, region_w, region_h, face_scale_pct, face_padding_px):
        """Resizes one face to fit inside the usable region area."""
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
                    fw, fh,
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
                    fw, fh,
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
                    fw, fh,
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
        """Pastes one face inside the reserved region."""
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
        """Pastes multiple faces inside the reserved region."""
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
            items   = layout[1]
            usable_w = layout[2]
            usable_h = layout[3]
            total_h = sum(img.size[1] for img in items) + face_gap_px * (len(items) - 1)
            start_y = usable_y + aligned_offset(usable_h, total_h, face_align_main)

            y = start_y
            for img in items:
                x = usable_x + aligned_offset(usable_w, img.size[0], face_align_cross)
                paste_with_alpha(canvas, img, (x, y))
                y += img.size[1] + face_gap_px
            return

        if kind == "horizontal":
            items    = layout[1]
            usable_w = layout[2]
            usable_h = layout[3]
            total_w = sum(img.size[0] for img in items) + face_gap_px * (len(items) - 1)
            start_x = usable_x + aligned_offset(usable_w, total_w, face_align_main)

            x = start_x
            for img in items:
                y = usable_y + aligned_offset(usable_h, img.size[1], face_align_cross)
                paste_with_alpha(canvas, img, (x, y))
                x += img.size[0] + face_gap_px
            return

        if kind == "grid":
            items    = layout[1]
            usable_w = layout[2]
            usable_h = layout[3]
            cols     = layout[4]  # type: ignore[index]
            rows     = layout[5]  # type: ignore[index]

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

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def process(
        self,
        frames,
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
        face_sequence=None,
        face_images=None,
    ):
        """Main node execution."""
        if len(frames.shape) != 4:
            raise ValueError(
                "Input 'frames' must be a batch IMAGE tensor with shape [N, H, W, C]."
            )

        # Determine input mode
        use_sequence = face_sequence is not None
        use_image = face_images is not None

        if not use_sequence and not use_image:
            raise ValueError(
                "You must connect either 'face_sequence' (FACE_SEQUENCE) "
                "or 'face_images' (IMAGE) to the node."
            )

        # Pre-process legacy IMAGE input
        if use_image and not use_sequence:
            if len(face_images.shape) != 4 or face_images.shape[0] < 1:
                raise ValueError(
                    "Input 'face_images' must be a valid IMAGE batch with at least one image."
                )
            face_count = face_images.shape[0]
            faces_pil_legacy = [self._prepare_face(face_images[i]) for i in range(face_count)]
        else:
            faces_pil_legacy = []

        n, orig_h, orig_w, _ = frames.shape

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

            # Resolve faces for this frame
            if use_sequence:
                faces_for_frame = self._resolve_faces_for_frame_sequence(
                    face_sequence=face_sequence,
                    frame_idx=i,
                    overflow_mode=overflow_mode,
                )
            else:
                faces_for_frame = self._resolve_faces_for_frame_image(
                    faces_pil=faces_pil_legacy,
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


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "ReservedRegionFrameComposer": ReservedRegionFrameComposer,
    "FrameRangedFaceLoader": FrameRangedFaceLoader,
    "FaceSequenceBatch": FaceSequenceBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReservedRegionFrameComposer": "Reserved Region Frame Composer",
    "FrameRangedFaceLoader": "Frame Ranged Face Loader",
    "FaceSequenceBatch": "Face Sequence Batch",
}
