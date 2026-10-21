"""CPU-only input selection. These calibration inputs are NOT approved H3 training samples."""
import argparse, hashlib, json, random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--train', type=int, default=32)
    parser.add_argument('--validation', type=int, default=8)
    parser.add_argument('--test', type=int, default=8)
    args = parser.parse_args()
    rows = []
    for family, root in [('genjutsu', Path('/data/datasets/openjutsu/train_mt_multi')),
                         ('animate', Path('/data/datasets/animate_motion_transfer_v1/train'))]:
        for caption in sorted((root / 'targets').glob('*.txt')):
            refs = []
            for i in range(1, 17):
                files = sorted((root / f'refs_{i}').glob(caption.stem + '.*'))
                if files:
                    refs.append(str(files[0]))
            if not 1 <= len(refs) <= 4:
                continue
            guide = root / ('guides_original' if family == 'animate' else 'guides') / (caption.stem + '.mp4')
            if not guide.is_file():
                continue
            rows.append(dict(id=family + '__' + caption.stem, guide=str(guide), references=refs,
                             prompt=caption.read_text().strip(), original_caption_sha256=hashlib.sha256(caption.read_bytes()).hexdigest(),
                             calibration_only=True, approved_for_h3_training=False))
    random.Random(42).shuffle(rows)
    selected = []; seen_assets = set()
    # No exact reference-asset reuse across selected cases/splits.
    for row in rows:
        hashes = {hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in row['references']}
        if hashes & seen_assets:
            continue
        seen_assets.update(hashes); row['reference_sha256'] = sorted(hashes); selected.append(row)
        if len(selected) >= args.train + args.validation + args.test:
            break
    assert len(selected) == args.train + args.validation + args.test, 'Insufficient disjoint reference assets'
    for i, row in enumerate(selected):
        row['split'] = 'train' if i < args.train else 'validation' if i < args.train + args.validation else 'test'
        row['input_key'] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    print(json.dumps(dict(output=str(out), total=len(selected), train=args.train, validation=args.validation, test=args.test,
                          purpose='Embedding reconstruction only; unfiltered input captions, not a curated identity-quality benchmark.')))


if __name__ == '__main__':
    main()
