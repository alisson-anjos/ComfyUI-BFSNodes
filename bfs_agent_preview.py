"""Opt-in SAMPLER wrapper: observe denoised x0, never patch global sampler callbacks."""

from __future__ import annotations
import time
import math

from .bfs_agent_inference import cuda_context_fault, mark_cuda_fault


class PreviewRetry(RuntimeError):
    """An explicit VLM retry must still interrupt the render."""


def video_latent(x0, latent_shapes=None):
    """Read H3 video x0 from dense, nested or sampler-packed AV tensors."""
    if getattr(x0, "is_nested", False):
        x0 = x0.unbind()[0]
    if x0.ndim == 3 and x0.shape[1] == 1:
        if not latent_shapes:
            raise ValueError(
                f"Packed H3 preview {tuple(x0.shape)} has no latent_shapes metadata"
            )
        shapes = [tuple(shape) for shape in latent_shapes]
        if any(
            len(shape) < 2 or shape[0] != x0.shape[0] or any(d <= 0 for d in shape)
            for shape in shapes
        ):
            raise ValueError("Invalid H3 latent_shapes metadata")
        if sum(math.prod(shape[1:]) for shape in shapes) != x0.shape[-1]:
            raise ValueError("Packed H3 preview size does not match latent_shapes")
        shape = shapes[0]  # H3 packs video first, audio second
        if len(shape) != 5 or shape[1] != 24:
            raise ValueError(f"First H3 stream is not video: {shape}")
        x0 = x0[:, :, : math.prod(shape[1:])].reshape(shape)
    if x0.ndim == 4 and x0.shape[0] == 24:
        x0 = x0.unsqueeze(0)
    if x0.ndim != 5 or x0.shape[1] != 24:
        raise ValueError(f"Expected H3 video [B,24,T,H,W], received {tuple(x0.shape)}")
    return x0


class PreviewSampler:
    def __init__(
        self,
        sampler,
        session_id,
        every_steps,
        min_seconds,
        frames,
        review,
        vae=None,
        shot=None,
    ):
        self.sampler = sampler
        self.sid = session_id
        self.every = every_steps
        self.seconds = min_seconds
        self.frames = frames
        self.review = review
        self.vae = vae
        self.shot = shot

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        from .bfs_agent_runtime import (
            SESSIONS,
            GPU_LOCK,
            image_url,
            image_tensor,
            visual_batch,
        )
        from . import bfs_shot_loop as p
        import comfy.model_management as mm
        import torch

        s = SESSIONS.get(self.sid)
        if not s:
            return self.sampler.sample(
                model_wrap,
                sigmas,
                extra_args,
                callback,
                noise,
                latent_image,
                denoise_mask,
                disable_pbar,
            )
        # Reviews and errors from an earlier render must not describe this attempt.
        s.preview = None
        vae = self.vae
        if vae is None:
            from .bfs_agent_tae import load_preview_vae
            import comfy.utils
            import folder_paths

            name = next(
                (
                    x
                    for x in folder_paths.get_filename_list("vae_approx")
                    if x.startswith("taeh3")
                ),
                None,
            )
            if not name:
                raise ValueError(
                    "Agent preview needs taeh3 in models/vae_approx, or a compatible preview_vae input"
                )
            vae = load_preview_vae(
                comfy.utils.load_torch_file(
                    folder_paths.get_full_path("vae_approx", name)
                )
            )
        last = 0.0
        preview_failed = False
        patcher = model_wrap.model_patcher
        latent_format = patcher.get_model_object("latent_format")
        latent_shapes = getattr(
            getattr(model_wrap, "inner_model", None), "latent_shapes", None
        )
        full_vae = getattr(vae, "single_latent_preview", False) or any(
            cls.__name__ == "MiniMaxH3VideoVAE"
            for cls in type(vae.first_stage_model).__mro__
        )

        def observe(step, x0, x, total):
            nonlocal last, preview_failed
            if callback is not None:
                callback(step, x0, x, total)
            if s.status == "stopped":
                raise RuntimeError("BFS agent stopped by user")
            if preview_failed:
                return
            now = time.monotonic()
            if ((step + 1) % self.every and step + 1 != total) or (
                last and now - last < self.seconds and step + 1 != total
            ):
                return
            if s.status in ("paused", "error"):
                return
            with GPU_LOCK, torch.inference_mode():
                loaded = mm.loaded_models(only_currently_used=True)
                failed_cuda = False
                try:
                    latent = video_latent(x0, latent_shapes)
                    picks = sorted(
                        set(
                            round(i)
                            for i in torch.linspace(
                                0,
                                latent.shape[2] - 1,
                                min(self.frames, latent.shape[2]),
                            ).tolist()
                        )
                    )
                    imgs = []
                    for j in picks:
                        # Contiguous temporal neighborhood; never concatenate unrelated latent times into a fake clip.
                        # The full H3 VAE has a single-latent image path: avoid arbitrary
                        # short temporal chunks, which differ between ComfyUI versions.
                        lo = j if full_vae else max(0, j - 1)
                        hi = j + 1 if full_vae else min(latent.shape[2], j + 2)
                        z = latent_format.process_out(latent[:1, :, lo:hi].detach())
                        decoded = vae.decode(z)
                        if decoded.ndim == 5:
                            decoded = decoded.reshape(-1, *decoded.shape[-3:])
                        if decoded.ndim != 4 or not len(decoded):
                            raise ValueError("Unexpected preview decoder output")
                        imgs.append(
                            image_tensor(
                                decoded[len(decoded) // 2 : len(decoded) // 2 + 1]
                            )
                        )
                    preview = {
                        "step": step + 1,
                        "total_steps": total,
                        "fraction": (step + 1) / max(1, total),
                        "shot": (self.shot or {}).get("index"),
                        "approximate": True,
                        "decoder": type(vae.first_stage_model).__name__,
                        "decode_mode": "single-latent snapshots"
                        if full_vae
                        else "temporal neighborhoods",
                        "frames": [
                            {"latent_index": j, "src": image_url(im)}
                            for j, im in zip(picks, imgs)
                        ],
                    }
                    s.preview = preview
                    if s.record_preview:
                        s.record_preview(preview)
                    p._notify("bfs-agent-preview", {"id": s.id, "preview": preview})
                    # Review is synchronous at a sampler boundary; restore diffusion models before the next step.
                    if self.review and (step + 1) / max(1, total) >= 0.35:
                        from comfy_extras.nodes_textgen import TextGenerate

                        clip = p._VLM.get("clip")
                        if clip is None:
                            raise ValueError(
                                "No registered Planner VLM for preview review"
                            )
                        comparison = []
                        if self.shot is not None:
                            f = self.shot.get("frames")
                            if f is not None and len(f):
                                for j in (0, len(f) // 2, len(f) - 1):
                                    comparison.append(image_tensor(f[j]))
                            comparison.extend(self.shot.get("refs") or [])
                        prompt = (
                            "Evaluate this APPROXIMATE intermediate video preview against the task: "
                            + s.task
                            + f"\nProgress {(step + 1) / total:.0%}. Early previews cannot establish fine identity. "
                            'Images: intermediate generated previews first, then source frames and references when connected. Return JSON only: {"decision":"continue" or "retry", "reason":"brief observed evidence"}. '
                            "Choose retry only for an unmistakable major composition failure after 65% progress. "
                            "Do not request tools or change the task."
                        )
                        out = p._textgenerate_execute(
                            TextGenerate,
                            clip=clip,
                            prompt=prompt,
                            max_length=2048,
                            sampling_mode={"sampling_mode": "off"},
                            thinking=s.settings["thinking"],
                            image=visual_batch(imgs + comparison),
                        ).args[0]
                        decision = p._parse_json(str(out))
                        preview["review"] = decision
                        if s.record_preview:
                            s.record_preview(preview)
                        s.event(
                            "preview", step=step + 1, total_steps=total, review=decision
                        )
                        p._notify("bfs-agent-preview", {"id": s.id, "preview": preview})
                        if (
                            decision.get("decision") == "retry"
                            and (step + 1) / total >= 0.65
                        ):
                            s.observation = {
                                "preview_review": decision,
                                "message": "Generation aborted for a major observed failure. Correct the plan before rendering again.",
                            }
                            raise PreviewRetry(
                                "BFS agent preview requested retry: "
                                + str(decision.get("reason", ""))[:500]
                            )
                except PreviewRetry:
                    raise
                except Exception as exc:
                    failed_cuda = cuda_context_fault(exc)
                    if failed_cuda:
                        mark_cuda_fault()
                    if (
                        failed_cuda
                        or s.status == "stopped"
                        or isinstance(
                            exc, getattr(mm, "InterruptProcessingException", ())
                        )
                    ):
                        raise
                    preview_failed = True
                    message = f"{type(exc).__name__}: {exc}"
                    s.preview = {
                        **(s.preview or {}),
                        "error": message,
                        "step": step + 1,
                        "total_steps": total,
                        "approximate": True,
                    }
                    s.event(
                        "preview_error",
                        summary="Intermediate preview disabled for this render; generation continues.",
                        error=message,
                    )
                    p._notify("bfs-agent-preview", {"id": s.id, "preview": s.preview})
                finally:
                    if not failed_cuda:
                        mm.load_models_gpu(loaded)
                    last = time.monotonic()

        return self.sampler.sample(
            model_wrap,
            sigmas,
            extra_args,
            observe,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )


class BFSAgentPreviewSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "session_id": ("STRING", {"default": ""}),
                "every_steps": ("INT", {"default": 5, "min": 1, "max": 1000}),
                "min_seconds": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 3600.0}),
                "frames": ("INT", {"default": 4, "min": 1, "max": 8}),
                "review_with_vlm": ("BOOLEAN", {"default": True}),
            },
            "optional": {"preview_vae": ("VAE",), "shot": ("BFS_SHOT",)},
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "wrap"
    CATEGORY = "BFS/Shot Planner/Agent"
    DESCRIPTION = "Opt-in H3 x0 previews and VLM reviews. Connect between sampler selection and SamplerCustom. Agent fills session_id in its workflow snapshot."

    def wrap(
        self,
        sampler,
        session_id,
        every_steps,
        min_seconds,
        frames,
        review_with_vlm,
        preview_vae=None,
        shot=None,
    ):
        return (
            PreviewSampler(
                sampler,
                session_id,
                every_steps,
                min_seconds,
                frames,
                review_with_vlm,
                preview_vae,
                shot,
            ),
        )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


NODE_CLASS_MAPPINGS = {"BFSAgentPreviewSampler": BFSAgentPreviewSampler}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BFSAgentPreviewSampler": "BFS Agent Preview Sampler (H3)"
}


class BFSAgentContext:
    """Loads only the connected VLM before the first decision; never renders the video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"vlm": ("CLIP",), "session_id": ("STRING", {"default": ""})}
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "register"
    CATEGORY = "BFS/Shot Planner/Agent"

    def register(self, vlm, session_id):
        from . import bfs_shot_loop as p

        p._VLM["clip"] = vlm
        return ()

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


NODE_CLASS_MAPPINGS["BFSAgentContext"] = BFSAgentContext
NODE_DISPLAY_NAME_MAPPINGS["BFSAgentContext"] = "BFS Agent Context (internal)"
