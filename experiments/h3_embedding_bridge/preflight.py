"""Read-only CPU preflight: does not import torch, download weights or launch GPU jobs."""
import argparse, ast, json
from pathlib import Path
from common import load_manifest


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', required=True)
    ap.add_argument('--student', default='/data/models/Qwen3.5-9B')
    ap.add_argument('--output', required=True); args = ap.parse_args()
    rows = load_manifest(args.manifest)
    missing = []
    for row in rows:
        for path in [row['guide'], *row['references']]:
            if not Path(path).is_file():
                missing.append(path)
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    student = Path(args.student)
    ready = (student / 'config.json').exists() and any(student.glob('*.safetensors'))
    report = dict(phase='prepared_gpu_execution_pending', samples=len(rows),
                  splits={s: sum(r['split'] == s for r in rows) for s in ['train', 'validation', 'test']},
                  missing_input_files=missing, student_weights_found=ready, student_path=str(student),
                  syntax_checked=True, tensor_tests_run=False, gpu_tests_run=False, adapter_trained=False,
                  h3_quality_validated=False, active_dataset_review_untouched=True,
                  pending=['Provide full local Qwen3.5-9B Transformers weights in an isolated compatible environment.',
                           'After dataset GPU review exits, cache teacher then student activations sequentially.',
                           'Train shared bridge and evaluate held-out embedding reconstruction.',
                           'Render native baseline / thinking off / thinking on / generated-state ablation with fixed inputs.'])
    if missing:
        raise RuntimeError('Calibration input files missing: ' + str(missing))
    Path(args.output).write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
