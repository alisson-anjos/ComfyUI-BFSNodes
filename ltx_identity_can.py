"""LTX Identity CAN (AdaLN) — apply the trained Conditioned Adaptive Normalization at inference.

The ArcFace projector (appended text tokens) has ~0 impact (cross-attn can ignore appended tokens).
CAN instead modulates the self-attention AdaLN (shift + gate) of the even blocks with the reference
ArcFace — a channel the model can't ignore. Trained on top of the reference LoRA it lifts identity
a lot (humo 0.52 -> 0.69) with no first-frame leak.

Key trick: the CAN delta depends only on the (constant) reference ArcFace, and the block AdaLN is
`scale_shift_table[row] + timestep`. So instead of patching the (version-specific, AV) block
forward, we just add the delta into `scale_shift_table` (row 0 = shift_msa, row 2 = gate_msa) via
ComfyUI's reversible add_object_patch — mathematically identical, works on any LTX/LTXAV version.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file


class CANModulation(nn.Module):
    """Exact replica of ltx-core CANModulation: id_global[512] -> (dshift, dgate)[dim]."""

    def __init__(self, id_dim: int, dim: int, hidden: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(id_dim)
        self.mlp = nn.Sequential(nn.Linear(id_dim, hidden), nn.SiLU(), nn.Linear(hidden, 3 * dim))
        self.dim = dim

    def forward(self, id_global):
        d = self.mlp(self.norm(id_global))
        dshift, _dscale, dgate = d.chunk(3, dim=-1)
        return dshift, dgate


def _find_diffusion_model(mp):
    m = getattr(mp, "model", mp)
    return getattr(m, "diffusion_model", m)


def apply_can_to_model(model_patcher, reference_image, can_weights_name, strength=1.0,
                       arcface_mode="auto_adjust"):
    """Add the trained CAN identity modulation to a ModelPatcher (in place, reversible).
    Returns the number of blocks modulated."""
    import folder_paths
    from .ltx_identity_overlap import _arcface_embed

    path = folder_paths.get_full_path("loras", can_weights_name)
    sd = load_file(path)
    idxs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("can.")})
    if not idxs:
        print("[BFS CAN] no can.* weights in file — nothing to apply.")
        return 0

    face = _arcface_embed(reference_image, arcface_mode) if _arcface_takes_mode() else _arcface_embed(reference_image)
    if face is None:
        print("[BFS CAN] no face detected in reference — CAN skipped.")
        return 0
    id_global = torch.as_tensor(np.asarray(face), dtype=torch.float32).view(1, -1)  # [1,512]

    dm = _find_diffusion_model(model_patcher)
    blocks = list(dm.transformer_blocks)
    even_blocks = blocks[::2]  # CAN was trained on even blocks (0,2,4,…) -> saved can.0,can.1,…
    n = min(len(idxs), len(even_blocks))
    applied = 0
    for j in range(n):
        # find this even block's index in the full list (for the dotted patch name)
        blk = even_blocks[j]
        block_index = j * 2
        sst = blk.scale_shift_table
        if sst.shape[0] < 3:
            continue
        dim = sst.shape[1]
        dev = sst.device
        can = CANModulation(id_dim=id_global.shape[1], dim=dim)
        can.load_state_dict({k[len(f"can.{idxs[j]}."):]: v for k, v in sd.items()
                             if k.startswith(f"can.{idxs[j]}.")})
        can = can.to(dev, torch.float32).eval()
        with torch.no_grad():
            dshift, dgate = can(id_global.to(dev, torch.float32))       # [1, dim], on the model device
        new_sst = sst.detach().clone().float()
        new_sst[0] = new_sst[0] + strength * dshift[0]                 # shift_msa
        new_sst[2] = new_sst[2] + strength * torch.tanh(dgate)[0]      # gate_msa
        new_param = nn.Parameter(new_sst.to(sst.dtype), requires_grad=False)
        name = f"diffusion_model.transformer_blocks.{block_index}.scale_shift_table"
        model_patcher.add_object_patch(name, new_param)
        applied += 1
    print(f"[BFS CAN] applied CAN (AdaLN) to {applied} even blocks (strength {strength}).")
    return applied


def _arcface_takes_mode():
    import inspect
    from .ltx_identity_overlap import _arcface_embed
    try:
        return "mode" in inspect.signature(_arcface_embed).parameters
    except Exception:
        return False


class LTXIdentityCAN:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        adapters = ["None"] + [f for f in folder_paths.get_filename_list("loras") if "identity_adapters" in f]
        return {
            "required": {
                "model": ("MODEL",),
                "reference_image": ("IMAGE",),
                "can_weights": (adapters,),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "BFS/LTX Identity"

    def apply(self, model, reference_image, can_weights, strength):
        if can_weights == "None":
            return (model,)
        m = model.clone()
        apply_can_to_model(m, reference_image, can_weights, strength)
        return (m,)


NODE_CLASS_MAPPINGS = {"LTXIdentityCAN": LTXIdentityCAN}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXIdentityCAN": "LTX Identity CAN / AdaLN"}
