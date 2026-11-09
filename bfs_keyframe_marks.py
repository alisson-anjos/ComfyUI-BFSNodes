"""BFS Keyframe Marks: the tracked person's mask at every keyframe, and the keyframe frames marked for an image edit.

Pick the person once (BFS Segment Picker on a guide frame), track it over the whole guide (Run SAM3 Video Track ->
SAM3 Track to Mask), then this node takes the mask at each keyframe of a BFS Keyframe Planner keyframe_spec and tints
that frame (the Colored Mark convention: magenta at 30 % marks the person an image edit replaces). A keyframe whose mask is empty (the track lost the person, e.g. after a camera cut) is reported, so
it can be marked again by hand.
"""
import json

import torch
import torch.nn.functional as F


def parse_color(text):
    s = str(text or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except (ValueError, IndexError):
        return 255, 0, 255


def grow(mask, pixels):
    """[N,H,W] mask dilated (pixels > 0) or eroded (pixels < 0) by a square of that radius."""
    if not pixels:
        return mask
    r = abs(int(pixels)); k = 2 * r + 1
    m = mask[:, None].float()
    if pixels > 0:
        return F.max_pool2d(m, k, 1, r)[:, 0]
    return -F.max_pool2d(-m, k, 1, r)[:, 0]


class BFSKeyframeMarks:
    DESCRIPTION = __doc__.split("\n\n")[0]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "guide_video": ("IMAGE", {"tooltip": "The guide frames the track was made on (same frames as the Planner's guide)."}),
            "masks": ("MASK", {"tooltip": "The person to replace: one mask per guide frame, or one per keyframe (Segment Picker batch_masks on the Planner's source_frames)."}),
            "keyframe_spec": ("STRING", {"forceInput": True, "tooltip": "BFS Keyframe Planner keyframe_spec: which frames are keyframes."}),
            "mark_color": ("STRING", {"default": "#FF00FF", "tooltip": "Tint of the person in marked_frames (hex)."}),
            "mark_opacity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "How strongly the person is tinted (0.3 = the Colored Mark default)."}),
            "grow_pixels": ("INT", {"default": 0, "min": -64, "max": 64,
                                    "tooltip": "Grow (positive) or shrink (negative) each mask before tinting, e.g. to cover hair."}),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Mask values above this count as the person."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("marked_frames", "masks", "source_frames", "report")
    OUTPUT_TOOLTIPS = ("The keyframe frames with the person tinted, one per keyframe (in keyframe_spec order).",
                       "The person's mask at each keyframe.",
                       "The untouched keyframe frames.",
                       "JSON: per keyframe its frame, time, mask area and whether the mask is empty.")
    FUNCTION = "mark"
    CATEGORY = "BFS/MiniMax H3"

    def mark(self, guide_video, masks, keyframe_spec, mark_color, mark_opacity, grow_pixels, threshold):
        spec = json.loads(keyframe_spec or "{}")
        frames = [int(f) for f in spec.get("frames", [])]
        if not frames:
            raise ValueError("keyframe_spec lists no keyframes: add them in BFS Keyframe Planner.")
        n, h, w = guide_video.shape[:3]
        if masks.ndim == 2:
            masks = masks[None]
        per_keyframe = masks.shape[0] == len(frames) and masks.shape[0] != n   # masks made on the Planner's source_frames
        if not per_keyframe and masks.shape[0] != n:
            raise ValueError(f"{masks.shape[0]} masks: give one per guide frame ({n}) or one per keyframe ({len(frames)}).")
        if masks.shape[1:] != (h, w):
            masks = F.interpolate(masks[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        color = torch.tensor(parse_color(mark_color), dtype=guide_video.dtype) / 255.0
        fps = float(spec.get("fps", 24.0))
        src, out, kept, report = [], [], [], []
        for i, f in enumerate(frames):
            f = max(0, min(n - 1, f))
            j = i if per_keyframe else f
            m = grow((masks[j:j + 1] > threshold).float(), grow_pixels).clamp(0, 1)
            img = guide_video[f:f + 1, ..., :3]
            a = (m[0].to(img.dtype) * float(mark_opacity))[..., None]
            out.append(img * (1 - a) + color * a)
            src.append(img); kept.append(m)
            area = float(m.mean())
            report.append({"frame": f, "time": round(f / fps, 3), "area": round(area, 4), "empty": area < 1e-4})
        lost = [r["frame"] for r in report if r["empty"]]
        if len(lost) == len(report):
            raise ValueError("No person selected: click the person to replace in the Video Segment Picker, then run again.")
        if lost:
            print(f"[BFS Keyframe Marks] no mask at keyframe frame(s) {lost}: mark the person there by hand", flush=True)
        return torch.cat(out), torch.cat(kept), torch.cat(src), json.dumps({"keyframes": report, "lost": lost})


class BFSTrackFromFrame:
    """Track a person over the whole guide from the frame they were picked on (not only frame 0): SAM 3 tracks
    forward from that frame and backward (the frames before it, reversed), and the two halves are joined. With
    anchor_frame -1 the anchor is the first keyframe of keyframe_spec, so feed the Planner's source_frames to the
    Segment Picker and its mask lines up."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "guide_video": ("IMAGE", {"tooltip": "The guide frames (the same the Planner gets)."}),
            "sam3_model": ("MODEL", {"tooltip": "SAM 3.1 MODEL (Load Checkpoint with sam3.1_multiplex_fp16.safetensors)."}),
            "mask": ("MASK", {"tooltip": "The person on the anchor frame (BFS Segment Picker mask)."}),
            "anchor_frame": ("INT", {"default": -1, "min": -1, "max": 100000,
                                     "tooltip": "Guide frame the mask was made on; -1 = the first keyframe of keyframe_spec."}),
        }, "optional": {
            "keyframe_spec": ("STRING", {"forceInput": True, "tooltip": "BFS Keyframe Planner keyframe_spec (for anchor_frame -1)."}),
        }}

    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("masks", "anchor_frame")
    OUTPUT_TOOLTIPS = ("One mask per guide frame.", "The anchor frame used.")
    FUNCTION = "track"
    CATEGORY = "BFS/MiniMax H3"

    @staticmethod
    def _run(frames, model, mask):
        from comfy_extras.nodes_sam3 import SAM3_VideoTrack, SAM3_TrackToMask
        data = SAM3_VideoTrack.execute(images=frames, model=model, initial_mask=mask, max_objects=1).args[0]
        return SAM3_TrackToMask.execute(track_data=data, object_indices="").args[0]

    def track(self, guide_video, sam3_model, mask, anchor_frame, keyframe_spec=None):
        n, h, w = guide_video.shape[:3]
        k = int(anchor_frame)
        if k < 0:
            frames = json.loads(keyframe_spec or "{}").get("frames", [])
            if not frames:
                raise ValueError("anchor_frame -1 needs a keyframe_spec with at least one keyframe.")
            k = int(frames[0])
        k = max(0, min(n - 1, k))
        m = mask[:1] if mask.ndim == 3 else mask[None]
        if m.shape[1:] != (h, w):
            m = F.interpolate(m[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        m = (m > 0.5).float()
        fwd = self._run(guide_video[k:], sam3_model, m)                       # frames k .. n-1
        parts = [fwd]
        if k > 0:
            bwd = self._run(guide_video[:k + 1].flip(0), sam3_model, m)      # frames k .. 0
            parts = [bwd.flip(0)[:k], fwd]
        out = torch.cat([p.to(guide_video.device, torch.float32) for p in parts])
        if out.shape[1:] != (h, w):
            out = F.interpolate(out[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        print(f"[BFS Track From Frame] anchor frame {k}: tracked {k} frames back, {n - k} forward", flush=True)
        return out, k


NODE_CLASS_MAPPINGS = {"BFSKeyframeMarks": BFSKeyframeMarks, "BFSTrackFromFrame": BFSTrackFromFrame}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSKeyframeMarks": "BFS Keyframe Marks", "BFSTrackFromFrame": "BFS Track From Frame (SAM 3)"}
