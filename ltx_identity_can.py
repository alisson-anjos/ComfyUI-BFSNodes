"""LTX Identity CAN (AdaLN) — apply the trained Conditioned Adaptive Normalization at inference.

The ArcFace-projector (appended text tokens) has ~0 impact because cross-attention can ignore the
appended tokens. CAN instead injects identity into the AdaLN (shift+gate of the self-attention
normalization) of the even blocks — a channel the model CANNOT ignore, since it modulates every
activation. Trained on top of the reference/overlap LoRA it lifts identity a lot while keeping the
no-first-frame-leak behaviour.

This node reproduces the training CAN (ltx-core) on ComfyUI's NATIVE LTX blocks: it rebuilds a
CANModulation per even block, loads the trained weights (identity_adapters_step_*.safetensors,
keys can.<i>.*), extracts the reference face's ArcFace embedding, and patches each even block's
forward so shift_msa += dshift and gate_msa += tanh(dgate) — exactly the training formula.

Inputs: model, reference_image, can_weights (the identity_adapters file), strength.
Output: MODEL with CAN applied. Combine with the identity LoRA + the overlap reference node.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file


class CANModulation(nn.Module):
    """Exact replica of ltx-core CANModulation: id_global[512] -> (dshift,dscale,dgate)[dim].
    Only shift+gate are used (scale darkens). Zero-init output = no-op until trained."""

    def __init__(self, id_dim: int, dim: int, hidden: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(id_dim)
        self.mlp = nn.Sequential(nn.Linear(id_dim, hidden), nn.SiLU(), nn.Linear(hidden, 3 * dim))
        self.dim = dim

    def forward(self, id_global):
        d = self.mlp(self.norm(id_global))
        dshift, _dscale, dgate = d.chunk(3, dim=-1)
        return dshift, dgate


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
        import folder_paths
        import comfy.model_management as mm
        from .ltx_identity_overlap import _find_ltxv, _arcface_embed

        device = mm.get_torch_device()
        sd = load_file(folder_paths.get_full_path("loras", can_weights))
        # keys: can.<i>.norm.* / can.<i>.mlp.* — group by block index i
        idxs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("can.")})
        if not idxs:
            print("[BFS CAN] no can.* weights in file — nothing to apply.")
            return (model,)

        # ArcFace identity of the reference (same encoder as training).
        face = _arcface_embed(reference_image)
        id_global = torch.as_tensor(face, device=device, dtype=torch.float32).view(1, -1)  # [1,512]

        m = model.clone()
        ltxv = _find_ltxv(m.model)
        blocks = [b for b in ltxv.transformer_blocks]
        dtype = next(ltxv.parameters()).dtype

        # CAN was attached to EVEN blocks (0,2,4,…), saved as can.0, can.1, … in that order.
        even_blocks = blocks[::2]
        n = min(len(idxs), len(even_blocks))
        for j in range(n):
            blk = even_blocks[j]
            dim = blk.scale_shift_table.shape[1]
            can = CANModulation(id_dim=id_global.shape[1], dim=dim)
            can.load_state_dict({k[len(f"can.{idxs[j]}."):]: v for k, v in sd.items()
                                 if k.startswith(f"can.{idxs[j]}.")})
            can = can.to(device, dtype).eval()
            with torch.no_grad():
                dshift, dgate = can(id_global.to(dtype))          # [1, dim] each
            blk._can_dshift = (dshift * strength).to(dtype)
            blk._can_dgate = (torch.tanh(dgate) * strength).to(dtype)
            if not getattr(blk, "_can_patched", False):
                _patch_block(blk)
                blk._can_patched = True

        print(f"[BFS CAN] applied CAN to {n} even blocks (strength {strength}).")
        return (m,)


def _patch_block(blk):
    """Wrap the native block.forward so the CAN deltas are added to shift_msa / gate_msa —
    exactly the training formula (shift += dshift, gate += tanh(dgate))."""
    import comfy.ldm.common_dit
    orig = blk.forward

    def fwd(x, context=None, attention_mask=None, timestep=None, pe=None,
            transformer_options={}, self_attention_mask=None, prompt_timestep=None):
        sst = blk.scale_shift_table
        vals = (sst[None, None, :6].to(device=x.device, dtype=x.dtype)
                + timestep.reshape(x.shape[0], timestep.shape[1], sst.shape[0], -1)[:, :, :6, :]).unbind(dim=2)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = vals
        # CAN: identity modulates the self-attn AdaLN (shift + gate). Broadcast [1,dim]->[B,1,dim].
        shift_msa = shift_msa + blk._can_dshift.unsqueeze(1).to(shift_msa.dtype)
        gate_msa = gate_msa + blk._can_dgate.unsqueeze(1).to(gate_msa.dtype)

        x = x + blk.attn1(comfy.ldm.common_dit.rms_norm(x) * (1 + scale_msa) + shift_msa,
                          pe=pe, mask=self_attention_mask, transformer_options=transformer_options) * gate_msa
        if blk.cross_attention_adaln:
            from comfy.ldm.lightricks.model import apply_cross_attention_adaln
            sq, scq, gq = (sst[None, None, 6:9].to(device=x.device, dtype=x.dtype)
                           + timestep.reshape(x.shape[0], timestep.shape[1], sst.shape[0], -1)[:, :, 6:9, :]).unbind(dim=2)
            x = x + apply_cross_attention_adaln(x, context, blk.attn2, sq, scq, gq,
                                                blk.prompt_scale_shift_table, prompt_timestep,
                                                attention_mask, transformer_options)
        else:
            x = x + blk.attn2(x, context=context, mask=attention_mask, transformer_options=transformer_options)
        y = comfy.ldm.common_dit.rms_norm(x)
        y = torch.addcmul(y, y, scale_mlp).add_(shift_mlp)
        x = x + blk.ff(y) * gate_mlp
        return x

    blk.forward = fwd


NODE_CLASS_MAPPINGS = {"LTXIdentityCAN": LTXIdentityCAN}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXIdentityCAN": "LTX Identity CAN / AdaLN"}
