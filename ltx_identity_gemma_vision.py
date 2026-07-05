"""LTX Identity — Gemma-Vision Conditioning (MagicMirror-Caminho2).

The identity breakthrough for LTX: the model conditions in Gemma's embedding space, so foreign
encoders (CLIP/DINO/ArcFace adapters) fail. Instead we feed the REFERENCE IMAGE through LTX's OWN
multimodal Gemma 3 (vision_tower), so the reference becomes conditioning tokens IN THE NATIVE SPACE
— the model reads the person like text. No reference latent (so no copy-paste/mask), no foreign
embedding, no DiT surgery. Pairs with a LoRA trained the same way (+ ArcFace loss).

This node mirrors ComfyUI-LTXVideo's LTXVGemmaTextEncoderModel (Gemma -> feature_extractor ->
embeddings connector) but runs Gemma MULTIMODALLY (input_ids with image tokens + pixel_values),
producing standard CONDITIONING for the LTX sampler.

Inputs: clip (the LTX Gemma encoder), reference_image, prompt.
Output: CONDITIONING (identity of the reference is baked in, native space).
"""
from __future__ import annotations

import numpy as np
import torch


def _get_encoder(clip):
    """Return the LTXVGemmaTextEncoderModel (.model / .feature_extractor / .embeddings_processor)
    from a ComfyUI CLIP object, across the couple of wrappers ComfyUI may use."""
    for attr in ("cond_stage_model", "model"):
        m = getattr(clip, attr, None)
        if m is not None and hasattr(m, "feature_extractor") and hasattr(m, "model"):
            return m
    # cond_stage_model may itself wrap the encoder (e.g. .gemma / .transformer)
    csm = getattr(clip, "cond_stage_model", clip)
    for attr in ("gemma", "transformer", "text_encoder"):
        m = getattr(csm, attr, None)
        if m is not None and hasattr(m, "feature_extractor"):
            return m
    if hasattr(csm, "feature_extractor") and hasattr(csm, "model"):
        return csm
    raise RuntimeError("Could not locate the LTX Gemma encoder (feature_extractor/model) on the CLIP.")


class LTXIdentityGemmaVision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "reference_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "ref_t2v: a person in a room, medium shot."}),
            },
            "optional": {
                "gemma_path": ("STRING", {"default": "/data/models/gemma"}),
                "max_length": ("INT", {"default": 1024, "min": 128, "max": 2048}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "BFS/LTX Identity"

    def encode(self, clip, reference_image, prompt, gemma_path="/data/models/gemma", max_length=1024):
        import comfy.model_management as mm
        from transformers import AutoProcessor
        from PIL import Image

        enc = _get_encoder(clip)
        model = enc.model
        device = model.device
        dtype = next(model.parameters()).dtype

        # Reference image (ComfyUI IMAGE [1,H,W,C] float) -> PIL.
        img = Image.fromarray((reference_image[0].cpu().numpy() * 255).astype(np.uint8))
        proc = AutoProcessor.from_pretrained(gemma_path)

        # Multimodal input: image + prompt, left-padded to max_length (matches training precompute).
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        mi = proc(text=text, images=img, return_tensors="pt").to(device)
        ids, am, pv = mi.input_ids, mi.attention_mask, mi.pixel_values.to(dtype)
        L = ids.shape[1]
        if L > max_length:
            ids, am = ids[:, :max_length], am[:, :max_length]
        elif L < max_length:
            p = max_length - L
            ids = torch.cat([torch.zeros((1, p), device=device, dtype=ids.dtype), ids], dim=1)
            am = torch.cat([torch.zeros((1, p), device=device, dtype=am.dtype), am], dim=1)

        # Block 1 (multimodal) + Block 2 (feature extractor) + Block 3 (connector) — as in training.
        enc.to(device)
        with torch.inference_mode():
            out = model.model(input_ids=ids, pixel_values=pv, attention_mask=am, output_hidden_states=True)
            all_layer_hiddens = torch.stack(out.hidden_states, dim=-1)  # [B, T, D, L]
            features = enc.feature_extractor(all_layer_hiddens, am, "left")
            in_dtype = next(iter(features.values())).dtype
            conn_mask = (am - 1).to(in_dtype).reshape(am.shape[0], 1, -1, am.shape[-1]) * torch.finfo(in_dtype).max
            encoded, mask = enc.embeddings_processor.create_embeddings(features, conn_mask)

        cond = [[encoded, {"pooled_output": None, "attention_mask": mask}]]
        print(f"[BFS Gemma-Vision] identity conditioning ready: {tuple(encoded.shape)} (reference in native space)")
        mm.soft_empty_cache()
        return (cond,)


NODE_CLASS_MAPPINGS = {"LTXIdentityGemmaVision": LTXIdentityGemmaVision}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXIdentityGemmaVision": "LTX Identity Gemma-Vision (Caminho2)"}
