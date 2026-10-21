"""GPU stage 3: one shared bridge, balanced thinking on/off, frozen cached encoders.
Embedding reconstruction is calibration only, NOT evidence that thinking helps H3.
"""
import argparse, json, random
from pathlib import Path
from common import load_manifest, require_idle_gpu
from bridge import BridgeConfig, HiddenStateBridge, alignment_loss


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', required=True); ap.add_argument('--cache', required=True)
    ap.add_argument('--output', required=True); ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--eval-every', type=int, default=50); ap.add_argument('--lr', type=float, default=0.0001)
    args = ap.parse_args(); require_idle_gpu()
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42); rng = random.Random(42)
    rows = load_manifest(args.manifest); cache = Path(args.cache); out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Refusing to overwrite an existing bridge run')
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for suffix in ['teacher', 'student_0', 'student_1']:
            path = cache / f'{row["id"]}.{suffix}.safetensors'
            with safe_open(str(path), framework='pt') as f:
                md = f.metadata() or {}
            key = md.get('input_key') if suffix == 'teacher' else json.loads(md['experiment'])['input_key']
            assert key == row['input_key'], 'Cache belongs to different inputs: ' + str(path)
    train = [r for r in rows if r['split'] == 'train']; val = [r for r in rows if r['split'] == 'validation']
    test = [r for r in rows if r['split'] == 'test']; assert train and val and test
    def case(row, thinking):
        teacher = load_file(str(cache / f'{row["id"]}.teacher.safetensors'))
        student = load_file(str(cache / f'{row["id"]}.student_{thinking}.safetensors'))
        return {k: v.cuda().float() for k, v in {**teacher, **student}.items()}
    model = HiddenStateBridge(BridgeConfig()).cuda()
    # Training-only per-channel scale; no held-out target features in model inputs.
    squares = torch.zeros(5120, dtype=torch.float64); tokens = 0
    for row in train:
        t = load_file(str(cache / f'{row["id"]}.teacher.safetensors'))['teacher'].float()
        squares += t.double().square().sum(0); tokens += len(t)
    model.teacher_channel_scale.copy_((squares / tokens).sqrt().clamp_min(1e-3).float().cuda())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    def predict(c, ablate=False):
        generated = torch.zeros_like(c['generated']) if ablate else c['generated']
        return model(c['prefix'], generated, c['tags'].long())
    @torch.no_grad()
    def measure(subset, export=False):
        model.eval(); results = []
        for row in subset:
            for thinking in (0, 1):
                c = case(row, thinking)
                for ablate in ([False, True] if thinking and export else [False]):
                    pred = predict(c, ablate); t = c['teacher']
                    assert torch.isfinite(pred).all()
                    relative = float((pred - t).norm() / t.norm().clamp_min(1e-8))
                    cosine = float(torch.nn.functional.cosine_similarity(pred.flatten(), t.flatten(), dim=0))
                    results.append(dict(id=row['id'], split=row['split'], thinking=bool(thinking),
                                        generated_states_zeroed=ablate, relative_error=relative, cosine=cosine,
                                        loss=float(alignment_loss(pred, t, c['tags']))))
                    if export:
                        label = 'thinking_ablated' if ablate else f'thinking_{thinking}'
                        name = row['id'] + '.' + label + '.safetensors'
                        metadata = dict(experiment='h3_9b_hidden_state_bridge_v1', input_key=row['input_key'],
                                        source_id=row['id'], thinking=str(bool(thinking)), generated_states_zeroed=str(ablate),
                                        prompt=row['prompt'], validated_h3_quality='false', teacher_layout_required='true')
                        save_file({'conditioning': pred.detach().cpu().to(torch.bfloat16).contiguous(),
                                   'teacher_baseline': t.cpu().to(torch.bfloat16).contiguous(),
                                   'minimax_token_tags': c['tags'].long().cpu().contiguous()},
                                  str(out / name), metadata=metadata)
        return results
    best = float('inf'); history = []
    for step in range(1, args.steps + 1):
        model.train(); row = rng.choice(train); thinking = step % 2
        c = case(row, thinking); optimizer.zero_grad(set_to_none=True)
        prediction = predict(c); loss = alignment_loss(prediction, c['teacher'], c['tags'])
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = measure(val); score = sum(r['loss'] for r in metrics) / len(metrics)
            entry = dict(step=step, train_loss=float(loss), validation_loss=score); history.append(entry)
            print(json.dumps(entry), flush=True)
            (out / 'history.json').write_text(json.dumps(history, indent=2))
            if score < best:
                best = score
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                          str(out / 'bridge.safetensors'), metadata={'config': json.dumps(model.config_dict()),
                                                                    'selected_by': 'validation reconstruction loss',
                                                                    'h3_quality_validated': 'false'})
    model.load_state_dict(load_file(str(out / 'bridge.safetensors')))
    metrics = measure(test, export=True)
    report = dict(test_metrics=metrics, h3_quality_validated=False, thinking_benefit_proven=False,
                  next_step='Compare native teacher, thinking off, thinking on and generated-state ablation in H3 with fixed seed/guide/references.',
                  limitation='Teacher-supervised reconstruction can suppress thinking-specific information; lower embedding error is not identity/motion quality.')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    (out / 'manifest.jsonl').write_text(Path(args.manifest).read_text())
    print(json.dumps(dict(phase='embedding_calibration_complete_gpu_video_validation_pending', output=str(out))), flush=True)


if __name__ == '__main__':
    main()
