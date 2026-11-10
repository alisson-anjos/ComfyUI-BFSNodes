"""Aligned downscaled H3 guides matching ai-toolkit target_grid_stride_v1.

No global ComfyUI monkey patches: a cloned ModelPatcher replaces the packed
layout in its UNet wrapper only when our conditioning marker is present.
"""
from __future__ import annotations

import copy
import logging
import math
import json

try:
    from .minimax_h3_reference_rope import validate_options, PHASE_VERSION, add_source_phase
    from .minimax_h3_rope_layout import apply_rope_layout
except ImportError:
    from minimax_h3_reference_rope import validate_options, PHASE_VERSION, add_source_phase
    from minimax_h3_rope_layout import apply_rope_layout

import numpy as np
from PIL import Image, ImageOps
import torch

LOG = logging.getLogger(__name__)
MARKER = 'bfs_h3_aligned_guide'
WRAPPER_MARKER = 'bfs_h3_downscaled_guide_wrapper_v1'
SPATIAL_VERSION = 'target_crop_v2'
POSITION_VERSION = 'target_grid_stride_v1'


def prepare_guide_frames(images, width, height, factor):
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError('downscale_factor must be a positive integer')
    if width <= 0 or height <= 0 or width % (32 * factor) or height % (32 * factor):
        raise ValueError(f'Output dimensions must be positive multiples of {32 * factor}')
    if images.ndim != 4 or images.shape[0] < 1 or images.shape[-1] < 3:
        raise ValueError('image must be a nonempty IMAGE batch [frames, height, width, channels]')
    result = []
    for frame in images[..., :3].detach().float().cpu():
        pixels = (frame.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        source = Image.fromarray(pixels)
        target = ImageOps.fit(source, (width, height), Image.Resampling.BICUBIC)
        guide = target.resize((width // factor, height // factor), Image.Resampling.LANCZOS)
        result.append(torch.from_numpy(np.asarray(guide).copy()).float() / 255)
    return torch.stack(result)


def sample_posterior(moments, latent_mean, latent_std, seed=42, fp16_round=False):
    mean, logvar = moments.float().chunk(2, dim=1)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype).to(mean.device)
    z = mean + torch.exp(0.5 * logvar.clamp(-30, 20)) * noise
    if fp16_round:
        z = z.half().float()
    shape = (1, -1, 1, 1, 1)
    return (z - latent_mean.to(z).view(shape)) / latent_std.to(z).view(shape)


def encode_moments(stage, pixels, device, dtype):
    # The toolkit normalizes pixels in FP32 before casting for the encoder.
    # Native encode_temporal normalizes in the input dtype, so reproduce its
    # chunk plan here rather than changing the shared VAE instance.
    def encode_clip(clip):
        normalized = stage._normalize_pixels(clip.float().to(device)).to(dtype)
        return stage._adaptive_encode(normalized)

    if pixels.shape[2] == 1:
        return encode_clip(pixels)[:, :, -1:]
    chunks = []
    for start in range(0, pixels.shape[2], stage.clip_length):
        clip = pixels[:, :, start:start + stage.clip_length]
        missing = stage.clip_length - clip.shape[2]
        if missing:
            clip = torch.cat([clip, clip[:, :, -1:].repeat(1, 1, missing, 1, 1)], dim=2)
        chunks.append(encode_clip(clip))
    moments = torch.cat(chunks, dim=2)
    if stage.token_drop > 0:
        moments = moments[:, :, :-stage.token_drop]
    return moments


@torch.no_grad()
def encode_guide(vae, frames):
    import comfy.model_management
    stage = getattr(vae, 'first_stage_model', None)
    required = ('_adaptive_encode', '_normalize_pixels', 'clip_length', 'token_drop', 'latents_mean', 'latents_std')
    if stage is None or any(not hasattr(stage, attr) for attr in required):
        raise ValueError('Use a complete native MiniMax-H3 video VAE with its encoder')
    comfy.model_management.load_models_gpu([vae.patcher])
    x = frames.permute(3, 0, 1, 2).unsqueeze(0).float() * 2 - 1
    moments = encode_moments(stage, x, vae.device, vae.vae_dtype)
    # The fork's sampling path uses seed 42 and FP16 posterior rounding for
    # both image keyframes and aligned guide videos.
    latents = sample_posterior(moments, stage.latents_mean, stage.latents_std,
                               fp16_round=True)
    return latents.cpu()


def downscaled_layout(native, keyframes, refs=()):
    """Select coarse condition rows from the native target grid, reindex all streams.

    Native PackedLayout creates full-canvas keyframe rows even for small
    latents. Keeping only stride-factor target patch origins fixes both row
    counts and RoPE while preserving refs, audio, target rows and their clocks.
    """
    required = ('signature', 'segments', 'position_ids', 'seq_len', 'img_pos', 'img_update',
                'audio_pos', 'audio_update')
    if any(not hasattr(native, name) for name in required):
        raise RuntimeError('Unsupported ComfyUI H3 PackedLayout API; update ComfyUI')
    _, _, height, width, _ = native.signature
    if height % 2 or width % 2:
        raise ValueError('H3 requires complete 2x2 spatial latent patches')
    patch_h, patch_w = height // 2, width // 2
    full_rows = patch_h * patch_w
    video_guides = iter(kf for kf in keyframes if kf.get('latent') is not None)
    keep_segments, new_segments = [], []
    cursor = 0
    for start, stop, kind in native.segments:
        indices = torch.arange(start, stop)
        if kind == 'cond':
            kf = next(video_guides)
            latent = kf['latent']
            factor = kf.get(MARKER, {}).get('downscale_factor', 1)
            if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
                raise ValueError('Invalid guide factor in conditioning')
            if height % (2 * factor) or width % (2 * factor):
                raise ValueError('Guide factor does not divide the target patch grid')
            if latent.ndim != 5 or latent.shape[:2] != (1, 24):
                raise ValueError('Guide latent must have shape [1, 24, time, height, width]')
            if tuple(latent.shape[-2:]) != (height // factor, width // factor):
                raise ValueError('Guide latent dimensions do not match the target and downscale factor')
            vt = latent.shape[2]
            if stop - start != vt * full_rows:
                raise RuntimeError('Unsupported native H3 guide segment size')
            grid = torch.arange(full_rows).reshape(patch_h, patch_w)[::factor, ::factor].flatten()
            indices = (start + torch.arange(vt)[:, None] * full_rows + grid[None]).flatten()
        keep_segments.append(indices)
        new_segments.append((cursor, cursor + len(indices), kind))
        cursor += len(indices)
    keep = torch.cat(keep_segments)
    remap = torch.full((native.seq_len,), -1, dtype=torch.long)
    remap[keep] = torch.arange(cursor)
    result = copy.copy(native)
    result.seq_len = cursor
    result.position_ids = native.position_ids[keep]
    result.segments = new_segments
    for modality in ('img', 'audio'):
        positions = getattr(native, modality + '_pos')
        selected = remap[positions] >= 0
        setattr(result, modality + '_pos', remap[positions[selected]])
        setattr(result, modality + '_update', getattr(native, modality + '_update')[selected])
    # Metadata used only by this wrapper; native signatures remain intact.
    result.bfs_h3_guide_layout = True
    return apply_rope_layout(result, keyframes, refs)


def patch_guide_model(model, phase_enabled=False):
    if model.model_options.get(WRAPPER_MARKER):
        patched=model.clone()
        if phase_enabled: install_phase_patch(patched)
        return patched
    patched = model.clone()
    previous = patched.model_options.get('model_function_wrapper')
    cache = [None, None]

    def wrapper(model_function, args):
        conds = args.get('c', {})
        payload = conds.get('minimax_payload')
        keyframes = (payload or {}).get('keyframes', [])
        refs = (payload or {}).get('refs', [])
        if any(MARKER in kf for kf in keyframes + refs):
            native = payload.get('layout')
            if native is None:
                raise RuntimeError('H3 layout missing from payload; unsupported ComfyUI version')
            if cache[0] is not native:
                cache[:] = [native, downscaled_layout(native, keyframes, refs)]
            copied_payload = dict(payload, layout=cache[1])
            args = dict(args, c=dict(conds, minimax_payload=copied_payload))
        else:
            cache[:] = [None, None]
        if previous is not None:
            return previous(model_function, args)
        return model_function(args['input'], args['timestep'], **args['c'])

    patched.set_model_unet_function_wrapper(wrapper)
    patched.model_options[WRAPPER_MARKER] = True
    patched.model_options['bfs_h3_phase_getter'] = lambda: getattr(cache[1], 'bfs_source_phase_values', None)
    if phase_enabled: install_phase_patch(patched)
    return patched


def install_phase_patch(patched):
    if patched.model_options.get('bfs_h3_source_phase_patch'): return
    original=patched.get_model_object('diffusion_model.rope_freqs')
    getter=patched.model_options['bfs_h3_phase_getter']
    def rope(position_ids, device):
        return add_source_phase(original(position_ids,device), getter())
    patched.add_object_patch('diffusion_model.rope_freqs',rope)
    patched.model_options['bfs_h3_source_phase_patch']=True


def rope_inputs():
    return {'guide_rope_layout': (['overlap','sidecar'],),
            'reference_rope_layout': (['native','overlap','sidecar'],),
            'reference_source_phase': ('BOOLEAN',{'default':False}),
            'reference_phase_scale': ('FLOAT',{'default':1.0,'min':0.0}),
            'reference_sidecar_margin': ('FLOAT',{'default':0.0,'min':0.0}),
            'source_id': ('INT',{'default':1,'min':1,'max':99})}


def rope_marker(options, source_id, role):
    if isinstance(source_id,bool) or not isinstance(source_id,int) or source_id<1:
        raise ValueError('source_id must match a positive training control channel')
    return {'rope_layout': options['guide_rope_layout'] if role=='guide' else options['reference_rope_layout'],
            'source_phase': options['reference_source_phase'], 'source_id': source_id,
            'phase_scale': options['reference_phase_scale'], 'sidecar_margin': options['reference_sidecar_margin']}


def check_lora_metadata(lora_name, factor, options=None):
    if not lora_name or lora_name == '(manual)':
        return
    import folder_paths
    from safetensors import safe_open
    path = folder_paths.get_full_path_or_raise('loras', lora_name)
    with safe_open(path, framework='pt', device='cpu') as handle:
        metadata = handle.metadata() or {}
    for key, expected in (('minimax_h3_guide_spatial_version', SPATIAL_VERSION),
                          ('minimax_h3_guide_position_version', POSITION_VERSION)):
        if metadata.get(key) != expected:
            raise ValueError(f'LoRA {key} must be {expected!r}; got {metadata.get(key)!r}')
    if int(metadata.get('reference_downscale_factor', 0)) != factor:
        raise ValueError('downscale_factor must match the LoRA training metadata')
    stored=json.loads(metadata.get('minimax_h3_reference_rope','{}'))
    expected=validate_options() if options is None else options
    stored_options=validate_options(stored.get('guide_rope_layout','overlap'),stored.get('reference_rope_layout','native'),stored.get('reference_source_phase',False),stored.get('reference_phase_scale',1.0),stored.get('reference_sidecar_margin',0.0))
    if stored_options != expected or stored.get('phase_version',PHASE_VERSION)!=PHASE_VERSION:
        raise ValueError('Reference RoPE options must match LoRA training metadata')
    if not any(metadata.get(key, '').lower()=='true' for key in ('guide_latent_only','control_latent_only')):
        raise ValueError('This node expects a caption-only VLM with guide_latent_only=true')


class BFSMiniMaxH3DownscaledGuide:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {'required': {
            'model': ('MODEL',),
            'positive': ('CONDITIONING',),
            'vae': ('VAE',),
            'latent': ('LATENT',),
            'image': ('IMAGE', {'tooltip': 'Single image or video frames already resampled to 24 fps.'}),
            'downscale_factor': ('INT', {'default': 4, 'min': 1, 'max': 8, 'step': 1}),
            'frame_idx': ('INT', {'default': 0, 'min': -9999, 'max': 9999}),
        }, 'optional': {
            'audio': ('AUDIO', {'tooltip': 'Optional aligned guide soundtrack; never routed to the text encoder.'}),
            'audio_vae': ('VAE',),
            'lora_name': (['(manual)'] + folder_paths.get_filename_list('loras'),
                          {'tooltip': 'Validate metadata only. Load the same LoRA using your normal LoRA loader.'}),
            **rope_inputs(),
        }}

    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'IMAGE')
    RETURN_NAMES = ('model', 'positive', 'guide_preview')
    FUNCTION = 'apply'
    CATEGORY = 'MiniMax-H3'
    DESCRIPTION = 'Aligned lower-resolution VAE guides for H3 LoRAs trained with target_grid_stride_v1. Connect both returned MODEL and CONDITIONING to the sampler.'

    @torch.no_grad()
    def apply(self, model, positive, vae, latent, image, downscale_factor=4, frame_idx=0,
              lora_name='(manual)', audio=None, audio_vae=None, guide_rope_layout='overlap', reference_rope_layout='native',
              reference_source_phase=False, reference_phase_scale=1.0, reference_sidecar_margin=0.0, source_id=1):
        from comfy.ldm.minimax import model as h3
        samples = latent['samples']
        streams = getattr(samples, 'tensors', None)
        if not getattr(samples, 'is_nested', False) or streams is None or len(streams) != 2:
            raise ValueError('Use a native MiniMax-H3 joint video/audio target latent')
        video = streams[0]
        if video.ndim != 5 or video.shape[:2] != (1, 24):
            raise ValueError('Target video latent must have shape [1, 24, time, height, width]')
        diffusion = model.get_model_object('diffusion_model')
        if not isinstance(diffusion, h3.MiniMaxH3Model) or tuple(diffusion.patch_size) != (1, 2, 2):
            raise ValueError('This node requires the native MiniMax-H3 model with 1x2x2 patches')
        options=validate_options(guide_rope_layout,reference_rope_layout,reference_source_phase,reference_phase_scale,reference_sidecar_margin)
        check_lora_metadata(lora_name, downscale_factor, options)
        width, height = video.shape[-1] * 16, video.shape[-2] * 16
        frame_count = sum(h3.FRAME_PER_TOKEN[k % 5] for k in range(video.shape[2]))
        frames = image.shape[0]
        if frames != 1 and (frames < 5 or frames % 17 != 5):
            raise ValueError('Guide must be one image or exactly 17*n+5 video frames (5, 22, 39, 73, ...); trim/resample explicitly')
        index = frame_idx if frame_idx >= 0 else frame_count + frame_idx
        if index < 0 or index + frames > frame_count:
            raise ValueError('Guide frames do not fit in the target timeline at frame_idx')
        if not positive:
            raise ValueError('positive conditioning must not be empty')
        if audio is not None and audio_vae is None:
            raise ValueError('audio_vae is required for a guide soundtrack')
        preview = prepare_guide_frames(image, width, height, downscale_factor)
        z = encode_guide(vae, preview)
        expected_t = 1 if frames == 1 else (frames - 5) // 17 * 5 + 2
        if tuple(z.shape) != (1, 24, expected_t, height // (16 * downscale_factor), width // (16 * downscale_factor)):
            raise ValueError(f'Unexpected H3 VAE latent shape: {tuple(z.shape)}')
        guide = {'resolved_frame_index': index, 'latent': z, MARKER: {
            'downscale_factor': downscale_factor,
            'spatial_version': SPATIAL_VERSION, 'position_version': POSITION_VERSION,
            **rope_marker(options, source_id, 'guide'),
        }}
        if audio is not None:
            if audio_vae is None:
                raise ValueError('audio_vae is required for a guide soundtrack')
            from comfy_extras.nodes_minimax_h3 import _encode_ref_audio
            audio_z, _ = _encode_ref_audio(audio_vae, audio)
            available = math.floor(streams[1].shape[-1] - h3.FRAME_RESCALE * index)
            clip_audio_frames = round(frames / 24 * 40)
            if available < 1:
                raise ValueError('Guide soundtrack starts outside the target audio timeline')
            guide['audio_latent'] = audio_z[..., :min(available, clip_audio_frames)].clone()
        # Copy all conditioning entries, preserving existing guides/references.
        updated = []
        for embedding, values in positive:
            values = dict(values)
            values['minimax_keyframes'] = list(values.get('minimax_keyframes', [])) + [guide]
            updated.append([embedding, values])
        LOG.info('BFS H3 guide: %s frames, %sx%s pixels, factor %s, frame %s', frames,
                 preview.shape[2], preview.shape[1], downscale_factor, index)
        return (patch_guide_model(model, reference_source_phase), updated, preview)


class BFSMiniMaxH3IdentityReference:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        extra=rope_inputs()
        extra['source_id']=('INT',{'default':2,'min':1,'max':99})
        extra['downscale_factor']=('INT',{'default':4,'min':1,'max':8})
        extra['lora_name']=(['(manual)']+folder_paths.get_filename_list('loras'),)
        return {'required':{'model':('MODEL',),'positive':('CONDITIONING',),'vae':('VAE',),
                            'latent':('LATENT',),'image':('IMAGE',)},'optional':extra}

    RETURN_TYPES=('MODEL','CONDITIONING','IMAGE')
    RETURN_NAMES=('model','positive','reference_preview')
    FUNCTION='apply'
    CATEGORY='MiniMax-H3'
    DESCRIPTION='Caption-only image identity reference. Match source_id to the training control channel. Requires an upscale checkpoint trained with the same reference role and RoPE options.'

    @torch.no_grad()
    def apply(self,model,positive,vae,latent,image,guide_rope_layout='overlap',reference_rope_layout='native',
              reference_source_phase=False,reference_phase_scale=1.0,reference_sidecar_margin=0.0,
              source_id=2,downscale_factor=4,lora_name='(manual)'):
        from comfy.ldm.minimax import model as h3
        diffusion=model.get_model_object('diffusion_model')
        if not isinstance(diffusion,h3.MiniMaxH3Model) or tuple(diffusion.patch_size)!=(1,2,2):
            raise ValueError('Identity reference requires native MiniMax-H3')
        if image.ndim!=4 or image.shape[0]!=1:
            raise ValueError('Identity reference must contain exactly one image')
        if not positive:raise ValueError('positive conditioning must not be empty')
        streams=getattr(latent.get('samples'),'tensors',None)
        if streams is None or len(streams)!=2:raise ValueError('Use a joint H3 target latent')
        video=streams[0]
        if video.ndim!=5 or video.shape[:2]!=(1,24):raise ValueError('Target batch must be one H3 video')
        options=validate_options(guide_rope_layout,reference_rope_layout,reference_source_phase,reference_phase_scale,reference_sidecar_margin)
        check_lora_metadata(lora_name,downscale_factor,options)
        if lora_name!='(manual)':
            import folder_paths
            from safetensors import safe_open
            with safe_open(folder_paths.get_full_path_or_raise('loras',lora_name),framework='pt',device='cpu') as f:
                if (f.metadata() or {}).get('control_latent_only','').lower()!='true':
                    raise ValueError('Caption-only identity node requires control_latent_only=true in training')
        ih,iw=image.shape[1:3];target_h,target_w=video.shape[-2]*16,video.shape[-1]*16
        scale=min(1.0,math.sqrt(target_h*target_w/(ih*iw)))
        height=max(32,round(ih*scale/32)*32);width=max(32,round(iw*scale/32)*32)
        pixels=(image[0,...,:3].detach().float().cpu().clamp(0,1).numpy()*255).round().astype(np.uint8)
        ref=Image.fromarray(pixels).resize((width,height),Image.Resampling.LANCZOS)
        preview=torch.from_numpy(np.asarray(ref).copy()).float().unsqueeze(0)/255
        z=encode_guide(vae,preview)
        block={'kind':'image','latent_h':height//16,'latent_w':width//16,'latent':z,
               MARKER:rope_marker(options,source_id,'reference')}
        updated=[]
        for embedding,values in positive:
            copied=dict(values);copied['minimax_refs']=list(values.get('minimax_refs',[]))+[block]
            updated.append([embedding,copied])
        return patch_guide_model(model,reference_source_phase),updated,preview


class BFSMiniMaxH3GuideTarget:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'width': ('INT', {'default': 1024, 'min': 32, 'max': 16384, 'step': 32}),
            'height': ('INT', {'default': 768, 'min': 32, 'max': 16384, 'step': 32}),
            'frames': ('INT', {'default': 73, 'min': 1, 'max': 3600, 'step': 1,
                               'tooltip': '1 for a true still image, or exactly 17*n+5 video frames.'}),
        }}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'create'
    CATEGORY = 'MiniMax-H3'

    def create(self, width, height, frames):
        import comfy.model_management
        from comfy.nested_tensor import NestedTensor
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError('Target width/height must be positive multiples of 32')
        if frames != 1 and (frames < 5 or frames % 17 != 5):
            raise ValueError('Target frames must be 1 or 17*n+5')
        vt = 1 if frames == 1 else (frames - 5) // 17 * 5 + 2
        at = round(frames / 24 * 40)
        device = comfy.model_management.intermediate_device()
        video = torch.zeros(1, 24, vt, height // 16, width // 16, device=device)
        audio = torch.zeros(1, 32, 2, at, device=device)
        return ({'samples': NestedTensor((video, audio))},)


NODE_CLASS_MAPPINGS = {
    'BFSMiniMaxH3DownscaledGuide': BFSMiniMaxH3DownscaledGuide,
    'BFSMiniMaxH3GuideTarget': BFSMiniMaxH3GuideTarget,
    'BFSMiniMaxH3IdentityReference': BFSMiniMaxH3IdentityReference,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'BFSMiniMaxH3DownscaledGuide': 'MiniMax-H3 Downscaled Latent Guide (BFS)',
    'BFSMiniMaxH3GuideTarget': 'MiniMax-H3 Guide Target — Image / Video (BFS)',
    'BFSMiniMaxH3IdentityReference': 'MiniMax-H3 Identity Reference + RoPE (BFS)',
}

from .bfs_h3_image_guide import BFSH3ImageAddGuide
NODE_CLASS_MAPPINGS['BFSH3ImageAddGuide'] = BFSH3ImageAddGuide
NODE_DISPLAY_NAME_MAPPINGS['BFSH3ImageAddGuide'] = 'Add Guide for MiniMax H3 Image (BFS)'
