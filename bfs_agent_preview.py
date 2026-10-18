"""Opt-in SAMPLER wrapper: observe denoised x0, never patch global sampler callbacks."""

from __future__ import annotations
import time


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
        vae = self.vae
        if vae is None:
            import comfy.sd
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
            vae = comfy.sd.VAE(
                comfy.utils.load_torch_file(
                    folder_paths.get_full_path("vae_approx", name)
                )
            )
        last = 0.0
        patcher = model_wrap.model_patcher
        latent_format = patcher.get_model_object("latent_format")

        def observe(step, x0, x, total):
            nonlocal last
            if callback is not None:
                callback(step, x0, x, total)
            if s.status == "stopped":
                raise RuntimeError("BFS agent stopped by user")
            now = time.monotonic()
            if ((step + 1) % self.every and step + 1 != total) or (
                last and now - last < self.seconds and step + 1 != total
            ):
                return
            if s.status in ("paused", "error"):
                return
            with GPU_LOCK, torch.inference_mode():
                loaded = mm.loaded_models(only_currently_used=True)
                try:
                    latent = x0.tensors[0] if getattr(x0, "is_nested", False) else x0
                    if latent.ndim != 5 or latent.shape[1] != 24:
                        raise ValueError(
                            "This preview adapter expects H3 [B,24,T,H,W] latents"
                        )
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
                        lo = max(0, j - 1)
                        hi = min(latent.shape[2], j + 2)
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
                        "frames": [
                            {"latent_index": j, "src": image_url(im)}
                            for j, im in zip(picks, imgs)
                        ],
                    }
                    s.preview = preview
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
                            raise RuntimeError(
                                "BFS agent preview requested retry: "
                                + str(decision.get("reason", ""))[:500]
                            )
                finally:
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
