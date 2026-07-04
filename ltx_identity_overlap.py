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
        ref_len = kw.get("_id_ref_len", getattr(self, "_id_ref_len", 0))
        if ref_len:
            target_len = getattr(self, "_id_target_len", None)
            if timestep.dim() <= 1 and target_len is not None:
                timestep = timestep.view(-1, 1).expand(batch_size, target_len).contiguous()
            if timestep.dim() >= 2:
                ref_ts = torch.zeros(batch_size, ref_len, *timestep.shape[2:], device=timestep.device, dtype=timestep.dtype)
                timestep = torch.cat([timestep, ref_ts], dim=1)
            _dbg("prepare_timestep: ref_len", ref_len, "| timestep ->", _shape(timestep), "| target_len", getattr(self, "_id_target_len", None))
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
        ref_len = kw.get("_id_ref_len", getattr(self, "_id_ref_len", 0))
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
            "debug_log": ("BOOLEAN", {"default": False,
                          "tooltip": "Print per-step [LTXIdOverlap] shape logs to the console (for debugging)."}),
        }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "debug")
    FUNCTION = "apply"
    CATEGORY = "LTX/identity"
    DESCRIPTION = ("100%-exact overlap+source_phase reference (as trained) via a model patch, "
                   "plus ArcFace projector tokens. Ref is separate tokens (NOT I2V). Load LoRA on MODEL first.")

    def apply(self, model, positive, negative, vae, latent, reference_face,
              identity_projector="None", source_id=2.0, phase_scale=1.0, id_strength=1.0,
              arcface_mode="auto_adjust", debug_log=False):
        import comfy.utils

        global _DEBUG_ENABLED
        _DEBUG_ENABLED = bool(debug_log)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = model.clone()
        ltxv = _find_ltxv(m)

        # encode ref image -> single-frame latent at the target resolution (overlap grid)
        _, w_sf, h_sf = vae.downscale_index_formula
        _, _, _, lat_h, lat_w = latent["samples"].shape
        ref_px = comfy.utils.common_upscale(reference_face.movedim(-1, 1), lat_w * w_sf, lat_h * h_sf, "bilinear", "center").movedim(1, -1)[:1, :, :, :3]
        ref_lat = vae.encode(ref_px)

        _install_patches(ltxv)
        ltxv._id_seg_value = float(source_id) * float(phase_scale)
        ltxv._id_rope_theta = 10000.0
        m.model_options = dict(m.model_options)
        to = dict(m.model_options.get("transformer_options", {}))
        to["_id_ref_latent"] = ref_lat
        m.model_options["transformer_options"] = to

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
            f"ref latent: {list(ref_lat.shape)} -> overlap tokens (frame-0 grid), source_phase seg={float(source_id)*float(phase_scale)}\n"
            f"arcface: {arc_status} (mode={arcface_mode}) | id_strength={id_strength}\n"
            f"patches on {type(ltxv).__name__}: process_input/prepare_timestep/prepare_pe/process_output\n"
            "Set LTX_IDOVERLAP_DEBUG=1 for per-step shape logs. Connect negative + CFG 3-5, no LightX2V."
        )
        log.info("\n" + dbg)
        # pass the latent through unchanged (the ref is injected inside the model, not here)
        # so the graph can chain Empty -> this node -> sampler without branching.
        return (m, positive, negative, latent, dbg)


# Public node id + display name. Keep the old key as an alias so existing workflows load.
NODE_CLASS_MAPPINGS = {
    "LTXIdentityTransfer": LTXIdentityOverlapConditioning,
    "LTXIdentityOverlapConditioning": LTXIdentityOverlapConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXIdentityTransfer": "LTX Identity Transfer",
    "LTXIdentityOverlapConditioning": "LTX Identity Transfer",
}
