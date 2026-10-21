"""GPU stage 1. Reuses the existing H3 encoder; does not load the H3 DiT/VAE."""
import argparse, importlib.util, json, sys
from pathlib import Path
from common import load_manifest, require_idle_gpu, visual_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True); ap.add_argument('--cache', required=True)
    ap.add_argument('--comfy', default='/data/comfyui')
    ap.add_argument('--teacher', default='/data/models/minimax-h3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors')
    args = ap.parse_args(); require_idle_gpu(); sys.argv = [sys.argv[0]]
    import numpy as np
    import torch
    from safetensors.torch import save_file
    sys.path.insert(0, args.comfy)
    import comfy.options
    comfy.options.enable_args_parsing()
    import folder_paths, nodes
    teacher = Path(args.teacher); folder_paths.add_model_folder_path('text_encoders', str(teacher.parent))
    clip = nodes.CLIPLoader().load_clip(teacher.name, 'stable_diffusion')[0]
    path = Path(__file__).resolve().parents[2] / 'minimax_h3_clip_video.py'
    spec = importlib.util.spec_from_file_location('bridge_native_context', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    for row in load_manifest(args.manifest):
        dest = cache / (row['id'] + '.teacher.safetensors')
        if dest.exists():
            continue
        refs, frames, picks, info = visual_inputs(row)
        images = {f'ref_image_{i}': torch.from_numpy(np.asarray(im).copy()).float()[None] / 255 for i, im in enumerate(refs)}
        guide = torch.from_numpy(np.stack([np.asarray(im) for im in frames]).copy()).float() / 255
        positive = module.BFSMiniMaxH3ReferenceToVideoGuideContext.execute(
            clip=clip, prompt=row['prompt'], width=info['width'], height=info['height'], length=info['length'],
            ref_image_size='match', ref_images=images, guide_video_0=guide, clip_video_fps=2,
        ).args[0]
        assert len(positive) == 1
        embeds, extra = positive[0]; tags = extra['minimax_token_tags']
        embeds = embeds[0] if embeds.ndim == 3 else embeds
        tags = tags.reshape(-1)
        assert embeds.shape == (len(tags), 5120) and torch.isfinite(embeds).all()
        save_file({'teacher': embeds.cpu().to(torch.bfloat16).contiguous(), 'tags': tags.cpu().long().contiguous()},
                  str(dest), metadata={'input_key': row['input_key'], 'teacher': str(teacher), 'visual_inputs': json.dumps(info)})
        (cache / (row['id'] + '.input.json')).write_text(json.dumps(row, indent=2, ensure_ascii=False))
        print(json.dumps(dict(id=row['id'], teacher_shape=list(embeds.shape), split=row['split'])), flush=True)


if __name__ == '__main__':
    main()
