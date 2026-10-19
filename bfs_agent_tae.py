"""Strict decoder-only adapter for Kijai's 2D H3 preview checkpoint."""

import torch
from torch import nn


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
        )
        self.skip = (
            nn.Identity()
            if inputs == outputs
            else nn.Conv2d(inputs, outputs, 1, bias=False)
        )

    def forward(self, x):
        return torch.relu(self.conv(x) + self.skip(x))


def build_h3_decoder(sd):
    """Known 24-channel, 96/64-wide, four-upsample layout; require every weight."""

    def conv(a, b, bias=False):
        return nn.Conv2d(a, b, 3, padding=1, bias=bias)

    def up():
        return nn.Upsample(scale_factor=2, mode="nearest")

    model = nn.Sequential(
        Clamp(),
        conv(24, 96, True),
        nn.ReLU(),
        Block(96, 96),
        Block(96, 96),
        Block(96, 96),
        up(),
        conv(96, 96),
        Block(96, 96),
        Block(96, 96),
        Block(96, 96),
        up(),
        conv(96, 96),
        Block(96, 64),
        Block(64, 64),
        Block(64, 64),
        up(),
        conv(64, 64),
        Block(64, 64),
        Block(64, 64),
        up(),
        conv(64, 64),
        Block(64, 64),
        conv(64, 3, True),
    )
    model.load_state_dict(sd, strict=True)
    return model.eval()


def is_2d_h3(sd):
    first, last = sd.get("1.weight"), sd.get("23.weight")
    return (
        first is not None
        and last is not None
        and tuple(first.shape) == (96, 24, 3, 3)
        and tuple(last.shape) == (3, 64, 3, 3)
    )


class H3TinyPreviewVAE:
    """Preview-only VAE interface; management can offload its small decoder normally."""

    single_latent_preview = True
    latent_channels = 24

    def __init__(self, sd):
        import comfy.model_management as mm
        from comfy.model_patcher import ModelPatcher

        self.device = mm.vae_device()
        self.dtype = mm.vae_dtype(
            self.device, allowed_dtypes=[torch.float16, torch.float32]
        )
        self.first_stage_model = build_h3_decoder(sd).to(dtype=self.dtype)
        self.patcher = ModelPatcher(
            self.first_stage_model,
            load_device=self.device,
            offload_device=mm.vae_offload_device(),
        )

    def decode(self, latent):
        import comfy.model_management as mm

        if latent.ndim != 5 or latent.shape[1] != 24:
            raise ValueError("H3 tiny preview needs [B,24,T,H,W]")
        mm.load_models_gpu([self.patcher])
        frames = []
        for batch in latent:
            for index in range(batch.shape[1]):
                image = self.first_stage_model(
                    batch[:, index]
                    .unsqueeze(0)
                    .to(device=self.device, dtype=self.dtype)
                )
                frames.append(image[0].clamp(0, 1).movedim(0, -1).float().cpu())
        return torch.stack(frames)


def load_preview_vae(sd):
    if is_2d_h3(sd):
        return H3TinyPreviewVAE(sd)
    import comfy.sd

    vae = comfy.sd.VAE(sd=sd)
    vae.throw_exception_if_invalid()
    return vae
