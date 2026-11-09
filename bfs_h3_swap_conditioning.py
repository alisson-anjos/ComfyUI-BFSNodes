"""BFS H3 Character Swap Conditioning: MiniMax H3 Character Swap where the guide says who is replaced.

Three ways to mark the subjects (`mode`):
  - "keyframe pins": the source video as an aligned guide (Add Guide at frame 0) plus one Add Guide per edited keyframe
    at its own frame index (frames of the result with the new people in place, e.g. from BFS Keyframe Planner).
  - "normal map guide": the guide video with each subject to replace drawn as its normal map in gray detail, tinted with
    the subject's color (1 red, 2 blue, 3 green, 4 yellow = Picture 1..4).
  - "DWPose guide": the guide video with each subject erased to gray and drawn as its DWPose skeleton (body, hands and
    face, with the eyes and the mouth opening) in the subject's color.
The marked guides come from BFS Video Segment Picker (`hybrid` output). The Pictures are native references and the
prompt is written from the mode and the number of Pictures (or prompt_override).
"""
import json

import torch

GRAY = 128.0 / 255.0     # background of the keyframe_video preview
MAX_KEYFRAMES = 8
MAX_PICTURES = 4
PINS, NORMALS, POSE = "keyframe pins", "normal map guide", "DWPose guide"
MODES = [PINS, NORMALS, POSE]
SUBJECT_COLORS = ["red", "blue", "green", "yellow"]


def pose_caption(k):
    """The DWPose guide prompt (k Pictures)."""
    C = SUBJECT_COLORS
    defs = '\n'.join(f'<Subject {j}> is the subject marked by the {C[j - 1]} pose skeleton in the guide video; its appearance comes '
                      f'from <Picture {j}>, its pose, motion, facial expression and mouth movement from the {C[j - 1]} skeleton.'
                      for j in range(1, k + 1))
    ret = '\n'.join(f'<Subject {j}>: attribute_transfer - keeps the pose, motion, expression and lip movement of the {C[j - 1]} skeleton '
                     f'and takes the face, hair, body shape and whole outfit of <Picture {j}> in every frame.' for j in range(1, k + 1))
    who = ', '.join(f'the {C[j - 1]} subject replaced by <Picture {j}>' for j in range(1, k + 1))
    return ('character_swap:\nsubject_definitions:\nThe guide video is the source of the edit. In it, each subject to replace is erased to '
            'plain gray and shown only as its pose skeleton (body, hands and face, with the eyes and the mouth opening) drawn in one '
            'color; the mark gives the pose and the expression, not the body shape, and everything else in the guide is the real video.\n'
            + defs + f'\nsummary: [video editing + reference generation] The target video is the guide video with {who}, from the first '
            'frame to the last, following each skeleton\'s pose, expression and mouth. Every other person, the layout, the backgrounds, '
            'the camera and the motion stay exactly as in the guide video.\nretention_analysis:\n' + ret +
            '\nEveryone else, the layout and the backgrounds: fully_preserved.')


def normals_caption(k):
    """The normal map guide prompt (k Pictures)."""
    C = SUBJECT_COLORS
    defs = '\n'.join(f'<Subject {j}> is the subject marked by the {C[j - 1]} normal map in the guide video; its appearance comes '
                      f'from <Picture {j}>, its pose and motion from the {C[j - 1]} mark.' for j in range(1, k + 1))
    ret = '\n'.join(f'<Subject {j}>: attribute_transfer - keeps the pose and motion of the {C[j - 1]} mark and takes the face, '
                     f'hair, body and whole outfit of <Picture {j}> in every frame.' for j in range(1, k + 1))
    who = ', '.join(f'the {C[j - 1]} subject replaced by <Picture {j}>' for j in range(1, k + 1))
    return ('character_swap:\nsubject_definitions:\nThe guide video is the source of the edit. In it, each subject to replace is marked '
            'with its surface-normal map colorized in one solid color (its 3D surface, without its real colors); the mark says who '
            'changes, not where the new look must end, and everything else in the guide is the real video.\n' + defs +
            f'\nsummary: [video editing + reference generation] The target video is the guide video with {who}, from the first frame '
            'to the last, following each mark\'s pose and motion. Every other person, the layout, the backgrounds, the camera and the '
            'motion stay exactly as in the guide video.\nretention_analysis:\n' + ret + '\nEveryone else, the layout and the backgrounds: fully_preserved.')


def _tail(k):
    subs = '\n'.join(f'<Subject {j}> is the person of the guide video who appears with the look of <Picture {j}> in the '
                     f'keyframes; the appearance comes from <Picture {j}>, the motion from the guide video.' for j in range(1, k + 1))
    ret = '\n'.join(f'<Subject {j}>: attribute_transfer - keeps the motion of the guide video and takes the face, hair, body and '
                    f'whole outfit of <Picture {j}> in every frame.' for j in range(1, k + 1))
    summary = (f'\nsummary: [video editing + keyframe completion + reference generation] The target video is the guide video with '
               f'{k} {"people" if k > 1 else "person"} replaced: exactly the people that differ between the keyframes and the guide video, '
               f'replaced by {", ".join(f"<Picture {j}>" for j in range(1, k + 1))} from the first frame to the last. Every other '
               'person, the layout, the backgrounds, the camera and the motion stay exactly as in the guide video.\n'
               'retention_analysis:\n' + ret + '\nEveryone else, the layout and the backgrounds: fully_preserved.')
    return subs, summary


def pins_caption(k, times):
    """The keyframe pins prompt (k Pictures, keyframes at `times` seconds)."""
    when = ', '.join(f'{t:.1f}s' for t in times); n = len(times)
    subs, summary = _tail(k)
    return ('character_swap:\nsubject_definitions:\nThe guide video is the source of the edit. '
            f'{n} keyframe{"s" if n > 1 else ""} of the target video {"are" if n > 1 else "is"} pinned at {when}; at those times '
            'the target video matches the keyframes exactly.\n' + subs + summary)


def _fit(image, h, w):
    """[1,H,W,C] -> [h,w,3], center crop to the guide's aspect, then resize."""
    import comfy.utils
    x = image[:1, ..., :3].movedim(-1, 1)
    x = comfy.utils.common_upscale(x, w, h, "lanczos", "center")
    return x.movedim(1, -1)[0].clamp(0, 1)


class BFSH3KeyframeConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        opt = {f"picture_{j}": ("IMAGE", {"tooltip": f"Subject {j} ({SUBJECT_COLORS[j - 1]} mark in the guide modes)."})
               for j in range(2, MAX_PICTURES + 1)}
        opt.update({
            "model": ("MODEL", {"tooltip": "keyframe pins: connect to get it back with keyframe_until applied. Guide modes: passed through."}),
            "keyframes": ("IMAGE", {"tooltip": "keyframe pins: edited frames, one per keyframe (BFS Keyframe Planner keyframes)."}),
            "keyframe_spec": ("STRING", {"forceInput": True, "tooltip": "keyframe pins: BFS Keyframe Planner keyframe_spec (the frame of every keyframe)."}),
        })
        for i in range(1, MAX_KEYFRAMES + 1):
            opt[f"keyframe_{i}"] = ("IMAGE", {"tooltip": f"keyframe pins: edited frame {i}, a full frame of the RESULT (swapped person in place)."})
            opt[f"time_{i}"] = ("FLOAT", {"default": float(i - 1), "min": 0.0, "max": 600.0, "step": 0.05,
                                           "tooltip": f"keyframe pins: seconds into the guide where keyframe {i} sits."})
        return {"required": {
            "clip": ("CLIP", {"tooltip": "MiniMax H3's text encoder (Qwen3-VL)."}),
            "vae": ("VAE",),
            "guide_video": ("IMAGE", {"tooltip": "keyframe pins: the prepared 24-fps source video. Guide modes: the marked guide "
                                                 "(BFS Video Segment Picker `hybrid`)."}),
            "width": ("INT", {"forceInput": True}), "height": ("INT", {"forceInput": True}), "length": ("INT", {"forceInput": True}),
            "picture_1": ("IMAGE", {"tooltip": "Subject 1 (red mark in the guide modes)."}),
            "mode": (MODES, {"default": PINS, "tooltip": "How the subjects to replace are marked: edited keyframes pinned at their "
                                                         "frames, or a guide with each subject drawn as a colored normal map or "
                                                         "DWPose skeleton (BFS Video Segment Picker)."}),
            "keyframe_until": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "keyframe pins only, on the returned model: the guides condition every step whose "
                                                    "sigma is at least this; below it only the Pictures remain. 0.5 = more faithful to "
                                                    "the source, 0 = guides in every step. The guide modes keep the guide on every step."}),
            "keyframe_until_drops": (["keyframes only", "all guides"], {"default": "keyframes only",
                                     "tooltip": "keyframe pins only: below keyframe_until drop only the keyframes (the source video "
                                                "keeps guiding every step) or every guide."}),
            "prompt_override": ("STRING", {"multiline": True, "default": "", "tooltip": "Non-empty: used instead of the generated prompt."}),
        }, "optional": opt}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("model", "positive", "latent", "prompt", "keyframe_video", "pictures")
    OUTPUT_IS_LIST = (False, False, False, False, False, True)
    OUTPUT_TOOLTIPS = ("", "", "", "The prompt used.", "keyframe pins: the keyframes at their frames on gray. Guide modes: the guide.",
                       "The Pictures in order (Picture 1..4, any sizes), e.g. for BFS Video Compare's reference.")
    FUNCTION = "encode"
    CATEGORY = "BFS/MiniMax H3"
    DESCRIPTION = ("MiniMax H3 Character Swap conditioning. keyframe pins: the source video plus one Add Guide per edited "
                   "keyframe. normal map guide / DWPose guide: the marked guide from BFS Video Segment Picker, subjects 1-4 = "
                   "red, blue, green, yellow = Picture 1-4. The Pictures as references and the matching prompt.")

    @staticmethod
    def _collect(kw, fps=24.0):
        """[(frame, image[1,H,W,C])] from the Planner batch + spec and/or the keyframe_N / time_N inputs."""
        out = []
        batch, spec = kw.get("keyframes"), kw.get("keyframe_spec")
        if batch is not None:
            frames = json.loads(spec).get("frames", []) if spec else []
            if len(frames) != len(batch):
                raise ValueError(f"{len(batch)} keyframes but keyframe_spec lists {len(frames)} frames: connect the Planner's keyframe_spec.")
            out += [(int(f), batch[i:i + 1]) for i, f in enumerate(frames)]
        for i in range(1, MAX_KEYFRAMES + 1):
            if kw.get(f"keyframe_{i}") is not None:
                out.append((round(float(kw.get(f"time_{i}", 0.0)) * fps), kw[f"keyframe_{i}"]))
        return sorted(out, key=lambda x: x[0])

    def encode(self, clip, vae, guide_video, width, height, length, picture_1, mode=PINS, keyframe_until=0.7,
               keyframe_until_drops="keyframes only", prompt_override="", model=None, **kw):
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}: use one of {MODES}.")
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, MiniMaxH3ReferenceToVideo
        pictures = [picture_1] + [kw[f"picture_{j}"] for j in range(2, MAX_PICTURES + 1) if kw.get(f"picture_{j}") is not None]
        kfs = self._collect(kw)
        if mode != PINS:
            return self._guide(clip, vae, guide_video, width, height, length, pictures, prompt_override, model, kw, pose=mode == POSE)
        if not kfs:
            raise ValueError("No keyframes: connect the Planner's keyframes + keyframe_spec, or keyframe_1 / time_1.")
        if len(guide_video) != length:
            raise ValueError(f"guide has {len(guide_video)} frames, length is {length}: connect Prepare's length.")
        h, w = guide_video.shape[1:3]
        video = torch.full((length, h, w, 3), GRAY, dtype=guide_video.dtype)
        placed, frames = [], []
        for f, img in kfs:
            f = min(length - 1, max(0, int(f)))
            if f in placed:
                print(f"[BFS H3 keyframe] two keyframes at frame {f}", flush=True)
            frame = _fit(img, h, w).to(video.dtype)
            video[f] = frame
            placed.append(f); frames.append(frame)
        times = [f / 24.0 for f in placed]
        prompt = prompt_override.strip() or pins_caption(len(pictures), times)
        print("[BFS H3 keyframe] prompt:\n" + prompt, flush=True)
        positive, latent = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, vae=vae, prompt=prompt, width=width, height=height, length=length, ref_image_size="match",
            ref_images={f"ref_image_{i}": p for i, p in enumerate(pictures, 1)}).args[:2]
        positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae, image=guide_video).args[0]
        for f, frame in zip(placed, frames):
            positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=f, vae=vae, image=frame[None]).args[0]
        print(f"[BFS H3 keyframe] {mode}: {len(placed)} keyframes at frames {placed}; {len(pictures)} Pictures", flush=True)
        return (self._until(model, keyframe_until, keyframe_until_drops.startswith("keyframes")), positive, latent, prompt, video,
                pictures)

    def _guide(self, clip, vae, guide_video, width, height, length, pictures, prompt_override, model, kw, pose=False):
        """Normal map / DWPose guide: the marked guide (subjects drawn as color-coded marks) as the only aligned guide, every
        step; the Pictures in color order; no keyframes."""
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, MiniMaxH3ReferenceToVideo
        if len(guide_video) != length:
            raise ValueError(f"guide has {len(guide_video)} frames, length is {length}: connect Prepare's length.")
        prompt = prompt_override.strip() or (pose_caption if pose else normals_caption)(len(pictures))
        print("[BFS H3 keyframe] guide prompt:\n" + prompt, flush=True)
        positive, latent = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, vae=vae, prompt=prompt, width=width, height=height, length=length, ref_image_size="match",
            ref_images={f"ref_image_{i}": p for i, p in enumerate(pictures, 1)}).args[:2]
        positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae, image=guide_video).args[0]
        # the guide carries the background: it stays in every step (keyframe_until does not apply)
        return model, positive, latent, prompt, guide_video, pictures

    @staticmethod
    def _until(model, keyframe_until, keep_source=False):
        """The model with the guides dropped below sigma keyframe_until (as Viggle Remake's RemakeKeyframeUntil).
        keep_source: drop only the keyframes; the source video (the first Add Guide, see encode) stays."""
        if model is None:
            return None
        import comfy.patcher_extension
        until = float(keyframe_until)

        def release(executor, x, timestep, context, transformer_options, minimax_payload=None, **kwargs):
            p = minimax_payload or {}
            if p.get("keyframes") and float(timestep.flatten()[0]) / 1000 < until:
                refs = p.get("refs") or []
                kept = list(p["keyframes"][:1]) if keep_source else []
                minimax_payload = dict(p, keyframes=kept or None, layout=None,
                                       cond_video_latents=[k["latent"] for k in kept if k.get("latent") is not None] +
                                                          [r["latent"] for r in refs if "latent" in r],
                                       cond_audio_latents=[k["audio_latent"] for k in kept if k.get("audio_latent") is not None] +
                                                          [r["audio_latent"] for r in refs if r.get("audio_latent") is not None])
            return executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)

        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "bfs_h3_keyframe_until", release)
        return m


NODE_CLASS_MAPPINGS = {"BFSH3KeyframeConditioning": BFSH3KeyframeConditioning}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSH3KeyframeConditioning": "BFS H3 Character Swap Conditioning"}
