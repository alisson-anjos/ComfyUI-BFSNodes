"""GPU stage 2. Capture actual 9B decoder activations, not rewritten prompts.

A forward hook retains one selected layer only, avoiding generate's all-layer
hidden-state cache. No generated reasoning text is persisted or fed to H3.
"""
import argparse, json, sys
from pathlib import Path
from common import load_manifest, require_idle_gpu, visual_inputs


def decoder_layer(model):
    for path in ('model.language_model.layers', 'language_model.layers', 'model.layers'):
        obj = model
        try:
            for name in path.split('.'):
                obj = getattr(obj, name)
            return obj[-1]
        except AttributeError:
            pass
    raise RuntimeError('Unsupported Transformers decoder layout; no hidden states captured')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', required=True); ap.add_argument('--cache', required=True)
    ap.add_argument('--model', required=True, help='Local full Qwen3.5-9B Transformers directory; no automatic download')
    ap.add_argument('--max-new-tokens', type=int, default=512)
    args = ap.parse_args(); require_idle_gpu()
    if not Path(args.model).is_dir():
        raise RuntimeError('Full 9B weights are not present locally. Download in an isolated environment first.')
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map='auto', local_files_only=True)
    model.eval().requires_grad_(False)
    layer = decoder_layer(model); cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    for row in load_manifest(args.manifest):
        refs, frames, picks, info = visual_inputs(row)
        content = []
        for i, image in enumerate(refs, 1):
            content.extend([{'type': 'text', 'text': f'<Picture {i}>:'}, {'type': 'image', 'image': image}])
        content.append({'type': 'text', 'text': '<Video 1> chronological motion guide:'})
        for i in picks:
            content.extend([{'type': 'text', 'text': f'<{i / 24:.3f} seconds>:'}, {'type': 'image', 'image': frames[i]}])
        content.append({'type': 'text', 'text': 'Analyze the requested reference-to-performer assignments across these guide moments. '
                        'Keep reference identity separate from guide identity and motion. Do not invent unseen details. '
                        'Conclude with a short mapping summary. Request: ' + row['prompt']})
        messages = [{'role': 'user', 'content': content}]
        for thinking in (False, True):
            dest = cache / f'{row["id"]}.student_{int(thinking)}.safetensors'
            if dest.exists():
                continue
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  return_dict=True, return_tensors='pt', enable_thinking=thinking).to(model.device)
            captured = []
            def capture(_module, _inputs, output):
                x = output[0] if isinstance(output, tuple) else output
                assert x.ndim == 3 and x.shape[0] == 1 and x.shape[-1] == 4096
                captured.append(x[0].detach().cpu().to(torch.bfloat16).contiguous())
            handle = layer.register_forward_hook(capture)
            try:
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                               use_cache=True, return_dict_in_generate=False)
            finally:
                handle.remove()
            assert captured, 'No decoder activations captured'
            prefix = captured[0]; tail = torch.cat(captured[1:]) if len(captured) > 1 else prefix[:0]
            text = processor.tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=False)
            completed = not thinking or '</think>' in text
            meta = dict(input_key=row['input_key'], model=str(Path(args.model).resolve()), thinking=thinking,
                        thinking_completed=completed, prefix_tokens=len(prefix), generated_state_tokens=len(tail),
                        generated_tokens=generated.shape[1] - inputs['input_ids'].shape[1],
                        layer='last decoder block before final norm', visual_inputs=info,
                        extraction='Prefill activations plus generated-token activations; final sampled token may not be forwarded')
            if not completed:
                (cache / f'{row["id"]}.student_1.incomplete.json').write_text(json.dumps(meta, indent=2))
                raise RuntimeError('Thinking did not finish within token budget. Increase --max-new-tokens; no valid thinking cache saved.')
            save_file({'prefix': prefix, 'generated': tail}, str(dest), metadata={'experiment': json.dumps(meta)})
            print(json.dumps(dict(id=row['id'], thinking=thinking, completed=completed,
                                  prefix_tokens=len(prefix), generated_state_tokens=len(tail))), flush=True)


if __name__ == '__main__':
    main()
