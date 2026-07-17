"""LTX-2.3 Identity OVERLAP — 100%-exact replica of ltx-trainer's overlap+source_phase reference.

Unlike the append_keyframe path (which makes the ref an I2V first-frame), this injects the
reference latent as SEPARATE tokens that share the target's frame-0 RoPE grid (overlap) and
are tagged with a per-source RoPE phase (source_phase), exactly as the ltx-trainer flexible
strategy did at train/validation time. The ref tokens are clean (timestep 0), attend to the
target in self-attention, and are sliced off the output (never rendered).

Model patch (installed idempotently on the LTX av_model), matching LTXBaseModel._forward:
  1) _process_input  : append ref video tokens (patchified ref latent) with overlap positions;
  2) _prepare_positional_embeddings : rotate the ref tokens' RoPE freqs by source_phase;
  3) _prepare_timestep : give the ref tokens clean (0) timestep;
  4) _process_output : trim the ref tokens before unpatchify.
Plus the ArcFace IdentityProjector tokens on the text context (as trained).

Load the matching LoRA on MODEL first. Requires insightface + buffalo_l (ArcFace, CPU).
Set LTX_IDOVERLAP_DEBUG=1 to log shapes at each patch point for iteration.
"""
import logging
import os
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

log = logging.getLogger("LTXIdentityOverlap")
_USE_GPU = os.environ.get("LTX_IDPROJ_ARCFACE_GPU", "0") == "1"
_FACE_APP = None


def _shape(x):
    """Readable shape/structure of tensors, lists, tuples, CompressedTimestep, etc."""
    try:
        if hasattr(x, "shape"):
            return f"T{tuple(x.shape)}"
        if isinstance(x, (list, tuple)):
            return f"{type(x).__name__}[{', '.join(_shape(i) for i in x)}]"
        return type(x).__name__
    except Exception:
        return "?"


# Per-step debug logging. Off by default; toggled by the node's `debug_log` input
# (env var LTX_IDOVERLAP_DEBUG=1 sets the initial value).
_DEBUG_ENABLED = os.environ.get("LTX_IDOVERLAP_DEBUG", "0") == "1"


def _dbg(*a):
    if _DEBUG_ENABLED:
        print("[LTXIdOverlap] " + " ".join(str(x) for x in a), flush=True)


# ---------------- ArcFace + projector (same as the other nodes) ----------------
def _get_face_app():
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if _USE_GPU else ["CPUExecutionProvider"])
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0 if _USE_GPU else -1, det_size=(640, 640))
        _FACE_APP = app
    return _FACE_APP


class IdentityProjector(nn.Module):
    def __init__(self, in_dim=512, context_dim=4096, num_tokens=4, proj=None):
        super().__init__()
        self.in_dim = in_dim; self.context_dim = context_dim; self.num_tokens = num_tokens
        # `proj` is built from the checkpoint (variable depth); fallback = original 2-linear MLP.
        self.proj = proj if proj is not None else nn.Sequential(
            nn.Linear(in_dim, 1024), nn.GELU(), nn.Linear(1024, num_tokens * context_dim))
        self.norm = nn.LayerNorm(context_dim)

    def forward(self, e):
        return self.norm(self.proj(e).reshape(-1, self.num_tokens, self.context_dim))


def _load_projector(path, device):
    """Load an IdentityProjector of ANY depth by rebuilding proj.N Linears from the state_dict
    (old = 2 linears proj.0/proj.2; new enhanced = 3 linears proj.0/proj.2/proj.4, 16 tokens)."""
    sd = load_file(path)
    context_dim = sd["norm.weight"].shape[0]
    # collect Linear layers in order (proj.<idx>.weight), with GELU between them
    idxs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("proj.") and k.endswith(".weight")})
    layers = []
    for j, i in enumerate(idxs):
        w = sd[f"proj.{i}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if j < len(idxs) - 1:
            layers.append(nn.GELU())
    proj = nn.Sequential(*layers)
    in_dim = sd["proj.0.weight"].shape[1]
    num_tokens = sd[f"proj.{idxs[-1]}.weight"].shape[0] // context_dim
    # Sequential indices must match the state_dict (Linear at even positions, GELU odd) — they do.
    p = IdentityProjector(in_dim=in_dim, context_dim=context_dim, num_tokens=num_tokens, proj=proj)
    p.load_state_dict(sd)
    log.info("IdentityProjector loaded: %d tokens, %d Linear layers", num_tokens, len(idxs))
    return p.to(device=device, dtype=torch.float32).eval()


def _arcface_embed(image_bhwc, mode="auto_adjust"):
    """Return the ArcFace embedding, or None if disabled / no face found.
    mode: 'as_is' (detect on the image only), 'auto_adjust' (retry with border-pad
    zoom-out + upscale when detection fails), 'disable' (skip ArcFace entirely).
    """
    if mode == "disable":
        return None
    import cv2
    app = _get_face_app()
    img = np.ascontiguousarray((np.clip(image_bhwc[0].detach().cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1])
    attempts = [img]
    if mode == "auto_adjust":
        h, w = img.shape[:2]
        pad = int(0.4 * max(h, w))                                   # zoom-out: face may fill the frame
        attempts.append(cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE))
        attempts.append(cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))  # upscale small faces
        attempts.append(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))   # downscale huge faces
    for a in attempts:
        faces = app.get(a)
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            return torch.from_numpy(f.normed_embedding.astype(np.float32))
    return None


def _append_ctx_tokens(conditioning, tokens):
    out = []
    for ce, cd in conditioning:
        t = tokens.to(device=ce.device, dtype=ce.dtype)
        if t.shape[0] != ce.shape[0]:
            t = t.expand(ce.shape[0], -1, -1)
        if t.shape[-1] < ce.shape[-1]:
            t = F.pad(t, (0, ce.shape[-1] - t.shape[-1]))
        elif t.shape[-1] > ce.shape[-1]:
            t = t[..., :ce.shape[-1]]
        nd = cd.copy()
        am = nd.get("attention_mask")
        if am is not None:
            nd["attention_mask"] = torch.cat([am, torch.ones((*am.shape[:-1], t.shape[1]), device=am.device, dtype=am.dtype)], dim=-1)
        out.append([torch.cat([ce, t], dim=1), nd])
    return out


# ---------------- source_phase RoPE (port of ltx_core rope.apply_segment_phase) ----------------
def _rotate_ref_freqs(pe, ref_len, seg_value, theta=10000.0):
    """Rotate the LAST ref_len tokens' RoPE freqs by phase = seg_value * theta^(-d/L).
    pe = (cos, sin, [split_flag]). cos/sin shape [..., T, L] or [B,H,T,L]. Returns new pe tuple.
    """
    if ref_len <= 0 or seg_value == 0.0:
        return pe
    cos, sin = pe[0], pe[1]
    rest = tuple(pe[2:])
    L = cos.shape[-1]
    d = torch.arange(L, device=cos.device, dtype=torch.float32)
    rate = theta ** (-d / float(L))                      # (0,1], high-freq carries the tag
    phase = (seg_value * rate)                           # [L]
    pc = phase.cos().to(cos.dtype); ps = phase.sin().to(sin.dtype)
    # index the token axis (=-2)
    idx = [slice(None)] * cos.dim()
    idx[-2] = slice(cos.shape[-2] - ref_len, cos.shape[-2])
    idx = tuple(idx)
    c0, s0 = cos[idx], sin[idx]
    cos = cos.clone(); sin = sin.clone()
    cos[idx] = c0 * pc - s0 * ps
    sin[idx] = s0 * pc + c0 * ps
    _dbg("rotate ref freqs: L", L, "ref_len", ref_len, "seg", seg_value)
    return (cos, sin, *rest)


# ---------------- model patches (idempotent, on the av_model instance) ----------------
def _find_ltxv(model):
    m = getattr(model, "model", model)
    m = getattr(m, "diffusion_model", m)
    return m


def _letterbox_resize(ref_img, tgt_w, tgt_h, pad_value=1.0):
    """Resize `ref_img` ([B,H,W,C]) to fit ENTIRELY inside tgt_w x tgt_h, preserving its own
    aspect ratio (no crop, no distortion), padding the leftover space with `pad_value`
    (default white, matching the composite sheet's own white background). Unlike
    common_upscale(..., crop="center") -- which center-crops to the target aspect ratio
    BEFORE resizing, silently discarding whatever isn't in the middle of the source image --
    this never discards any pixel of the reference."""
    import comfy.utils
    x = ref_img.movedim(-1, 1)  # [B,C,H,W]
    _, _, src_h, src_w = x.shape
    scale = min(tgt_w / src_w, tgt_h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = comfy.utils.common_upscale(x, new_w, new_h, "bilinear", "disabled")
    pad_w, pad_h = tgt_w - new_w, tgt_h - new_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top
    padded = F.pad(resized, (left, right, top, bottom), mode="constant", value=pad_value)
    return padded.movedim(1, -1)


def _anchored_crop_resize(ref_img, tgt_w, tgt_h, anchor="center"):
    """Like comfy.utils.common_upscale(..., crop="center") but with a configurable anchor
    for WHICH part of the source survives the crop when the aspect ratio doesn't match,
    instead of always the exact center (e.g. anchor="top" keeps the top of the sheet --
    useful when the face closeup panel isn't centered in your layout). Returns the
    cropped+resized image plus the crop box (x0, y0, crop_w, crop_h) in SOURCE pixel
    coords, for drawing a preview overlay."""
    import comfy.utils
    x_img = ref_img.movedim(-1, 1)  # [B,C,H,W]
    _, _, old_h, old_w = x_img.shape
    old_aspect = old_w / old_h
    new_aspect = tgt_w / tgt_h
    x0, y0, crop_w, crop_h = 0, 0, old_w, old_h
    if old_aspect > new_aspect:
        # source wider than target -- crop width
        crop_w = max(1, round(old_w * (new_aspect / old_aspect)))
        if anchor == "left":
            x0 = 0
        elif anchor == "right":
            x0 = old_w - crop_w
        else:
            x0 = (old_w - crop_w) // 2
    elif old_aspect < new_aspect:
        # source taller than target -- crop height
        crop_h = max(1, round(old_h * (old_aspect / new_aspect)))
        if anchor == "top":
            y0 = 0
        elif anchor == "bottom":
            y0 = old_h - crop_h
        else:
            y0 = (old_h - crop_h) // 2
    cropped = x_img[:, :, y0:y0 + crop_h, x0:x0 + crop_w]
    out = comfy.utils.common_upscale(cropped, tgt_w, tgt_h, "bilinear", "disabled")
    return out.movedim(1, -1), (x0, y0, crop_w, crop_h)


def _draw_crop_overlay(ref_img, box):
    """Original reference with a green rectangle around the region that SURVIVES the crop
    (everything outside the rectangle gets discarded). box = (x0, y0, w, h) in source pixel
    coords, e.g. from _anchored_crop_resize. If box covers the whole image (no crop, as in
    letterbox/native_resolution modes), the rectangle just outlines the full frame."""
    from PIL import Image, ImageDraw
    x0, y0, cw, ch = box
    arr = (ref_img[0, :, :, :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(pil)
    w, h = pil.size
    lw = max(2, min(w, h) // 150)
    draw.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], outline=(0, 255, 0), width=lw)
    out = torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).unsqueeze(0)
    return out


def _apply_tass_layout(reference_positions, target_positions, layout: str):
    """Place reference pixel-coords in a non-overlapping TASS region -- mirrors
    ltx_trainer.training_strategies.tass.apply_tass_layout's st_drc branch (kept in sync
    manually since this node can't import the trainer package), adapted to ComfyUI's own
    coordinate tensor shape [B, 3 (T/H/W), N] (one corner coordinate per token) instead of
    the trainer's [B, 3, N, 2] patch-bounds shape -- st_drc's "every axis" shift only needs
    min/max per axis either way. layout='overlap' returns the input unchanged."""
    if layout != "st_drc":
        return reference_positions
    target_extent = target_positions.amax(dim=2, keepdim=True)
    reference_origin = reference_positions.amin(dim=2, keepdim=True)
    return reference_positions + (target_extent - reference_origin)


def _install_patches(ltxv):
    if getattr(ltxv, "_id_overlap_patched", False):
        return
    orig_process_input = ltxv._process_input
    orig_prepare_ts = ltxv._prepare_timestep
    orig_prepare_pe = ltxv._prepare_positional_embeddings
    orig_process_output = ltxv._process_output

    def process_input(self, x, keyframe_idxs, denoise_mask, **kw):
        out = orig_process_input(x, keyframe_idxs, denoise_mask, **kw)
        ref_lat = kw.get("_id_ref_latent")
        if ref_lat is None:
            self._id_ref_len = 0
            return out
        try:
            from comfy.ldm.lightricks.model import latent_to_pixel_coords
            xx, pix, add = out
            is_av = isinstance(xx, (list, tuple))
            vx = xx[0] if is_av else xx
            vco = pix[0] if is_av else pix
            _dbg("process_input IN: is_av", is_av, "| vx", _shape(vx), "| vco", _shape(vco), "| ref_lat", _shape(ref_lat))
            rt, rlc = self.patchifier.patchify(ref_lat.to(dtype=vx.dtype, device=vx.device))
            rpc = latent_to_pixel_coords(latent_coords=rlc, scale_factors=self.vae_scale_factors,
                                         causal_fix=self.causal_temporal_positioning)
            rpc = _apply_tass_layout(rpc, vco, getattr(self, "_id_layout", "overlap"))
            rt = self.patchify_proj(rt)
            if rt.shape[0] != vx.shape[0]:
                rt = rt.expand(vx.shape[0], -1, -1)
            if rpc.shape[0] != vco.shape[0]:
                rpc = rpc.expand(vco.shape[0], *([-1] * (rpc.dim() - 1)))
            ref_len = rt.shape[1]
            self._id_target_len = vx.shape[1]                # video tokens BEFORE ref
            vx = torch.cat([vx, rt], dim=1)                  # APPEND ref after target
            vco = torch.cat([vco, rpc.to(vco)], dim=2)
            self._id_ref_len = ref_len
            add = dict(add); add["_id_ref_len"] = ref_len
            _dbg("process_input OUT: ref_len", ref_len, "| target_len", self._id_target_len,
                 "| vx", _shape(vx), "| vco", _shape(vco))
            if is_av:
                xx = [vx, xx[1]]; pix = [vco, pix[1]]
            else:
                xx, pix = vx, vco
            return xx, pix, add
        except Exception as e:
            _dbg("ERROR process_input:", repr(e), "| out", _shape(out), "| ref_lat", _shape(ref_lat))
            raise

    def prepare_timestep(self, timestep, batch_size, hidden_dtype, **kw):
        # Give the ref tokens clean (0) timestep by editing the timestep INPUT before the
        # model's per-frame compression/adaln (mirrors the audio-ref path in av_model).
        # Use ONLY the instance attribute, not kw["_id_ref_len"] — when multi-angle patches
        # are also installed, their process_input sets kw["_id_ref_len"] which we must NOT
        # consume here (that would double-extend the timestep and cause shape mismatches).
        ref_len = getattr(self, "_id_ref_len", 0)
        if ref_len:
            target_len = getattr(self, "_id_target_len", None)
            if timestep.dim() <= 1 and target_len is not None:
                timestep = timestep.view(-1, 1).expand(batch_size, target_len).contiguous()
            if timestep.dim() >= 2 and target_len is not None:
                cur = timestep.shape[1]
                _gm0 = kw.get("grid_mask")
                _full = _gm0.shape[-1] if (_gm0 is not None and hasattr(_gm0, "shape")) else None
                if _full is not None and cur == _full:
                    # Per-token timestep spans the grid-mask domain (video + IC-LoRA guide
                    # tokens, indexed later via timestep[:, grid_mask]). Do NOT trim — append
                    # ref zeros and let the grid_mask pad below keep everything in lockstep.
                    ref_ts = torch.zeros(batch_size, ref_len, *timestep.shape[2:], device=timestep.device, dtype=timestep.dtype)
                    timestep = torch.cat([timestep, ref_ts], dim=1)
                    _dbg("prepare_timestep: grid-domain cur", cur, "-> appended ref", ref_len)
                elif cur > target_len + ref_len:
                    _dbg("prepare_timestep: oversized", cur, "-> trimming to video-only", target_len)
                    timestep = timestep[:, :target_len]
                    cur = target_len
                if (_full is None or timestep.shape[1] != _full + ref_len) and cur == target_len:
                    ref_ts = torch.zeros(batch_size, ref_len, *timestep.shape[2:], device=timestep.device, dtype=timestep.dtype)
                    timestep = torch.cat([timestep, ref_ts], dim=1)
                    _dbg("prepare_timestep: ref_len", ref_len, "| timestep ->", _shape(timestep), "| target_len", target_len)
                else:
                    _dbg("prepare_timestep: skip (cur", cur, "!= target", target_len, ")")
                # Guide nodes (IC-LoRA Guide / Director Guide, etc.) inject a grid_mask that
                # FILTERS x tokens in _process_input (x = x[:, grid_mask]) and later indexes the
                # timestep (timestep[:, grid_mask]). Our ref tokens are appended AFTER that
                # filter, so extend the mask with True for them — keeps modulation and vx in
                # lockstep (False would drop our clean timesteps and desync again).
                gm = kw.get("grid_mask")
                if gm is not None and hasattr(gm, "shape"):
                    gap = timestep.shape[1] - gm.shape[-1]
                    if 0 < gap <= ref_len:
                        pad = torch.ones(*gm.shape[:-1], gap, dtype=gm.dtype, device=gm.device)
                        kw = dict(kw); kw["grid_mask"] = torch.cat([gm, pad], dim=-1)
                        _dbg("prepare_timestep: grid_mask extended by", gap)
        return orig_prepare_ts(timestep, batch_size, hidden_dtype, **kw)

    def prepare_pe(self, pixel_coords, frame_rate, x_dtype):
        pe = orig_prepare_pe(pixel_coords, frame_rate, x_dtype)
        ref_len = getattr(self, "_id_ref_len", 0)
        seg = getattr(self, "_id_seg_value", 2.0)
        theta = getattr(self, "_id_rope_theta", 10000.0)
        if not ref_len:
            return pe
        try:
            _dbg("prepare_pe IN: pe struct", _shape(pe), "| ref_len", ref_len, "| seg", seg)
            # av returns [(v_pe, av_cross_video), (a_pe, av_cross_audio)]; v_pe = (cos, sin, split).
            if isinstance(pe, list) and len(pe) and isinstance(pe[0], (list, tuple)) and isinstance(pe[0][0], (list, tuple)):
                v_pe, cross_v = pe[0][0], pe[0][1]
                _dbg("prepare_pe: v_pe", _shape(v_pe))
                v_pe = _rotate_ref_freqs(v_pe, ref_len, seg, theta)
                pe = [(v_pe, cross_v), pe[1]]
            else:
                pe = _rotate_ref_freqs(pe, ref_len, seg, theta)
            return pe
        except Exception as e:
            _dbg("ERROR prepare_pe:", repr(e), "| pe", _shape(pe))
            raise

    def process_output(self, x, embedded_timestep, keyframe_idxs, **kw):
        ref_len = getattr(self, "_id_ref_len", 0)
        if ref_len:
            try:
                from comfy.ldm.lightricks.av_model import CompressedTimestep
                _dbg("process_output IN: x", _shape(x), "| et", _shape(embedded_timestep), "| ref_len", ref_len)
                # trim ref tokens from the video stream
                if isinstance(x, (list, tuple)):
                    x = [x[0][:, :x[0].shape[1] - ref_len], *x[1:]]
                    import copy
                    et_list = list(embedded_timestep) if isinstance(embedded_timestep, (list, tuple)) else [embedded_timestep]
                    v_et = et_list[0]
                    if isinstance(v_et, CompressedTimestep):
                        # clone + edit slots directly (version-agnostic; some builds lack per_frame kwarg)
                        ppf = max(1, getattr(v_et, "patches_per_frame", 1) or 1)
                        n_ref_frames = max(1, ref_len // ppf)
                        v_et2 = copy.copy(v_et)
                        v_et2.data = v_et.data[:, : v_et.num_frames - n_ref_frames].contiguous()
                        v_et2.num_frames = v_et.num_frames - n_ref_frames
                        et_list[0] = v_et2
                    elif hasattr(v_et, "shape") and v_et.dim() >= 2 and v_et.shape[1] > 1:
                        et_list[0] = v_et[:, : v_et.shape[1] - ref_len]
                    embedded_timestep = et_list
                else:
                    x = x[:, :x.shape[1] - ref_len]
                    if hasattr(embedded_timestep, "shape") and embedded_timestep.dim() >= 2 and embedded_timestep.shape[1] > 1:
                        embedded_timestep = embedded_timestep[:, : embedded_timestep.shape[1] - ref_len]
                _dbg("process_output: trimmed -> x", _shape(x), "| et", _shape(embedded_timestep))
            except Exception as e:
                _dbg("ERROR process_output:", repr(e), "| x", _shape(x), "| et", _shape(embedded_timestep))
                raise
        return orig_process_output(x, embedded_timestep, keyframe_idxs, **kw)

    ltxv._process_input = types.MethodType(process_input, ltxv)
    ltxv._prepare_timestep = types.MethodType(prepare_timestep, ltxv)
    ltxv._prepare_positional_embeddings = types.MethodType(prepare_pe, ltxv)
    ltxv._process_output = types.MethodType(process_output, ltxv)
    ltxv._id_overlap_patched = True
    log.info("LTXIdentityOverlap patches installed on %s", type(ltxv).__name__)


class LTXIdentityOverlapConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        # Populate the projector dropdown from the loras folder (+ "None" = no projector).
        try:
            import folder_paths
            proj_choices = ["None"] + folder_paths.get_filename_list("loras")
        except Exception:
            proj_choices = ["None"]
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "latent": ("LATENT",),
            "reference_face": ("IMAGE",),
            "identity_projector": (proj_choices, {"default": "None",
                             "tooltip": "ArcFace projector .safetensors (from models/loras). 'None' = overlap only "
                                        "(the projector is a weak channel; overlap latent carries identity)."}),
            "source_id": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 1.0,
                                    "tooltip": "source_phase segment id (training used 2). 0 = no phase."}),
            "phase_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1}),
            "id_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50.0, "step": 0.5,
                             "tooltip": "Multiplies the ArcFace projector tokens (only when a projector is selected). "
                                        "Weak channel; push high (5-20) to test, very high may add artifacts."}),
            "arcface_mode": (["auto_adjust", "as_is", "disable"], {"default": "auto_adjust",
                             "tooltip": "auto_adjust: retry face detection with zoom-out/upscale, skip tokens if none. "
                                        "as_is: detect on the image only. disable: skip ArcFace, use only the overlap latent."}),
            "ref_resize_mode": (["match_target", "match_target_letterbox", "native_resolution"], {"default": "match_target",
                             "tooltip": "match_target: resize ref to the OUTPUT video's pixel size via a CENTER-CROP then "
                                        "resize (old single-face-crop recipes — ref resolution never mattered, but this "
                                        "silently discards whatever isn't in the middle of the ref for mismatched aspect "
                                        "ratios, e.g. a landscape character sheet used for a portrait output loses the "
                                        "side panels/face closeup). match_target_letterbox: same target pixel size, but "
                                        "fits the WHOLE ref inside it preserving aspect ratio (no crop, pads with white) "
                                        "-- use this for a landscape composite sheet + portrait (or any mismatched-aspect) "
                                        "output so no panel/face-detail gets cut off. native_resolution: encode the ref at "
                                        "ITS OWN size (rounded to the nearest 32px), independent of the video size — "
                                        "REQUIRED for checkpoints trained on a fixed ref resolution bucket that differs "
                                        "from the video's own bucket (e.g. the composite face+3views ref, trained at "
                                        "2048x1024 regardless of output video size) -- but for aspect-mismatched outputs "
                                        "(e.g. portrait video) this can bias the model toward the ref's own (landscape) "
                                        "composition, causing an off-center/cropped-looking result; try "
                                        "match_target_letterbox first if you hit that."}),
            "debug_log": ("BOOLEAN", {"default": False,
                          "tooltip": "Print per-step [LTXIdOverlap] shape logs to the console (for debugging)."}),
        }, "optional": {
            "crop_anchor": (["center", "top", "bottom", "left", "right"], {"default": "center",
                             "tooltip": "New in v1.10.13, optional -- old workflows without this input keep the "
                                        "previous always-center-crop behavior. Only used by match_target (the "
                                        "crop-then-resize mode). Which part of the reference survives the crop when its "
                                        "aspect ratio doesn't match the output -- e.g. if your sheet's face closeup is "
                                        "in the top row, set 'top' instead of the default center crop so it doesn't get "
                                        "cut off. No effect on match_target_letterbox or native_resolution (neither "
                                        "ever crops)."}),
            "layout": (["overlap", "st_drc"], {"default": "overlap",
                       "tooltip": "New, optional -- old workflows without this input keep the default 'overlap' "
                                  "behavior (reference shares the target's own RoPE coordinate range, distinguished "
                                  "only by source_phase -- what every checkpoint so far was trained with). 'st_drc' "
                                  "shifts the WHOLE reference block past the target's coordinate extent on every "
                                  "axis (non-overlapping region) -- only use this with a checkpoint specifically "
                                  "trained with layout='st_drc' (ltx_trainer's flexible strategy); using it with an "
                                  "'overlap'-trained checkpoint will not work, the model never learned that "
                                  "coordinate convention."}),
            "reference_guidance_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1,
                             "tooltip": "ST-DRC-style reference-CFG (arxiv 2606.02441). 1.0 = off (identical to "
                                        "before this input existed). >1.0 adds a THIRD forward pass per step with "
                                        "the reference tokens dropped, and amplifies the reference's own "
                                        "contribution: denoised += (scale-1)*(with_ref - without_ref) -- the same "
                                        "way CFG amplifies the text prompt's. Costs one extra (cheaper, "
                                        "ref-token-free) forward pass per step when enabled. Start around 2-4; "
                                        "same units/convention as CFG scale on the sampler."}),
        }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "debug", "ref_preview", "crop_overlay")
    FUNCTION = "apply"
    CATEGORY = "LTX/identity"
    DESCRIPTION = ("100%-exact overlap+source_phase reference (as trained) via a model patch, "
                   "plus ArcFace projector tokens. Ref is separate tokens (NOT I2V). Load LoRA on MODEL first. "
                   "ref_preview/crop_overlay outputs (v1.10.13+) show exactly what gets encoded and, for "
                   "match_target, what part of the reference survives the crop (green box) vs gets discarded.")

    def apply(self, model, positive, negative, vae, latent, reference_face,
              identity_projector="None", source_id=2.0, phase_scale=1.0, id_strength=1.0,
              arcface_mode="auto_adjust", ref_resize_mode="match_target", debug_log=False,
              crop_anchor="center", layout="overlap", reference_guidance_scale=1.0):
        import comfy.samplers
        import comfy.utils

        global _DEBUG_ENABLED
        _DEBUG_ENABLED = bool(debug_log)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = model.clone()
        ltxv = _find_ltxv(m)

        _, w_sf, h_sf = vae.downscale_index_formula
        if ref_resize_mode == "native_resolution":
            # Encode at the ref image's OWN resolution (rounded to nearest 32px), independent
            # of the output video size — matches training when the ref used a fixed/own bucket.
            _, src_h, src_w, _ = reference_face.shape
            tgt_w = max(w_sf, round(src_w / w_sf) * w_sf)
            tgt_h = max(h_sf, round(src_h / h_sf) * h_sf)
        else:
            # Legacy behavior: resize ref to match the target video's pixel size (correct for
            # recipes where the ref used the SAME resolution bucket as the video, e.g. a small
            # face crop — resolution never mattered there).
            _, _, _, lat_h, lat_w = latent["samples"].shape
            tgt_w, tgt_h = lat_w * w_sf, lat_h * h_sf
        _, src_h0, src_w0, _ = reference_face.shape
        crop_box = (0, 0, src_w0, src_h0)  # default: "nothing cropped" (letterbox/native modes)
        if ref_resize_mode == "match_target_letterbox":
            ref_px = _letterbox_resize(reference_face, tgt_w, tgt_h)[:1, :, :, :3]
        elif ref_resize_mode == "match_target" and crop_anchor != "center":
            # non-default anchor: use the configurable-anchor crop (new in v1.10.13)
            ref_px, crop_box = _anchored_crop_resize(reference_face, tgt_w, tgt_h, anchor=crop_anchor)
            ref_px = ref_px[:1, :, :, :3]
        else:
            # unchanged from before v1.10.13 -- exact original center-crop path, byte-identical
            ref_px = comfy.utils.common_upscale(reference_face.movedim(-1, 1), tgt_w, tgt_h, "bilinear", "center").movedim(1, -1)[:1, :, :, :3]
            if ref_resize_mode == "match_target":
                _, crop_box = _anchored_crop_resize(reference_face, tgt_w, tgt_h, anchor="center")  # for the preview overlay only
        ref_lat = vae.encode(ref_px)
        ref_preview = ref_px.clone()
        crop_overlay = _draw_crop_overlay(reference_face[:1], crop_box)

        _install_patches(ltxv)
        ltxv._id_seg_value = float(source_id) * float(phase_scale)
        ltxv._id_rope_theta = 10000.0
        ltxv._id_layout = layout
        m.model_options = dict(m.model_options)
        to = dict(m.model_options.get("transformer_options", {}))
        to["_id_ref_latent"] = ref_lat
        m.model_options["transformer_options"] = to

        if reference_guidance_scale != 1.0:
            # ST-DRC reference-CFG: a third forward pass per step with the reference tokens
            # dropped isolates the reference's own contribution, the same way CFG isolates
            # the text prompt's (arxiv 2606.02441). `args["input_cond"]` is the exact
            # conditioning list KSampler passed for the positive/"cond" branch (same prompt,
            # same batching) -- reuse it unchanged, only strip `_id_ref_latent` from the
            # model_options used for THIS extra call so process_input sees no reference.
            noref_to = dict(to)
            noref_to.pop("_id_ref_latent", None)
            ref_scale = float(reference_guidance_scale)

            def _ref_cfg_function(args, _noref_to=noref_to, _ref_scale=ref_scale):
                cond = args["cond"]
                uncond = args["uncond"]
                cond_scale = args["cond_scale"]
                denoised = uncond + (cond - uncond) * cond_scale
                noref_model_options = dict(args["model_options"])
                noref_model_options["transformer_options"] = _noref_to
                (noref_pred,) = comfy.samplers.calc_cond_batch(
                    args["model"], [args["input_cond"]], args["input"], args["timestep"], noref_model_options,
                )
                noref_denoised = args["input"] - noref_pred
                denoised = denoised + (_ref_scale - 1.0) * (cond - noref_denoised)
                return denoised

            m.set_model_sampler_cfg_function(_ref_cfg_function, disable_cfg1_optimization=True)

        # ArcFace projector tokens on the text context — fully OPTIONAL. Skipped when the
        # projector dropdown is 'None', arcface is disabled, or no face is detected. The
        # overlap latent carries the bulk of identity, so this is safe to skip.
        use_projector = identity_projector not in (None, "", "None")
        emb = _arcface_embed(reference_face, mode=arcface_mode) if use_projector else None
        if not use_projector:
            arc_status = "no projector (overlap only)"
        elif arcface_mode == "disable":
            arc_status = "disabled"
        else:
            arc_status = "OK" if emb is not None else "NO FACE -> skipped (overlap only)"
        if emb is not None:
            path = identity_projector
            if not os.path.isabs(path):
                try:
                    import folder_paths
                    resolved = folder_paths.get_full_path("loras", identity_projector)
                    if resolved:
                        path = resolved
                except Exception:
                    pass
            if not os.path.isabs(path) or not os.path.exists(path):
                for b in ("models/loras", "models", "."):
                    if os.path.exists(os.path.join(b, identity_projector)):
                        path = os.path.join(b, identity_projector); break
            projector = _load_projector(path, device)
            emb = emb.to(device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                id_tok = projector(emb) * float(id_strength)
                unc = projector(torch.zeros(1, projector.in_dim, device=device))
            positive = _append_ctx_tokens(positive, id_tok)
            negative = _append_ctx_tokens(negative, unc)

        dbg = (
            "=== LTX Identity OVERLAP (exact) ===\n"
            f"ref latent: {list(ref_lat.shape)} (encoded at {tgt_w}x{tgt_h}px, mode={ref_resize_mode}"
            f"{f', crop_anchor={crop_anchor}' if ref_resize_mode == 'match_target' else ''}) "
            f"-> {layout} tokens, source_phase seg={float(source_id)*float(phase_scale)}\n"
            f"arcface: {arc_status} (mode={arcface_mode}) | id_strength={id_strength}\n"
            f"patches on {type(ltxv).__name__}: process_input/prepare_timestep/prepare_pe/process_output\n"
            f"crop preview: kept region {crop_box[2]}x{crop_box[3]}px of the {src_w0}x{src_h0}px reference "
            "-- see the ref_preview/crop_overlay IMAGE outputs to inspect what gets kept vs discarded.\n"
            f"reference-CFG: {'off' if reference_guidance_scale == 1.0 else f'ON, scale={reference_guidance_scale} (+1 forward pass/step)'}\n"
            "Set LTX_IDOVERLAP_DEBUG=1 for per-step shape logs. Connect negative + CFG 3-5, no LightX2V."
        )
        log.info("\n" + dbg)
        # pass the latent through unchanged (the ref is injected inside the model, not here)
        # so the graph can chain Empty -> this node -> sampler without branching.
        return (m, positive, negative, latent, dbg, ref_preview, crop_overlay)


# Public node id + display name. Keep the old key as an alias so existing workflows load.
NODE_CLASS_MAPPINGS = {
    "LTXIdentityTransfer": LTXIdentityOverlapConditioning,
    "LTXIdentityOverlapConditioning": LTXIdentityOverlapConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXIdentityTransfer": "LTX Identity Transfer",
    "LTXIdentityOverlapConditioning": "LTX Identity Transfer",
}
