import json, subprocess
from pathlib import Path


def require_idle_gpu():
    # No background polling/auto launch: fail before loading a model.
    active = subprocess.check_output(['ps', '-eo', 'args'], text=True)
    if any('audit_temporal_identity_binding.py' in line and not line.lstrip().startswith(('ps ', 'rg '))
           for line in active.splitlines()):
        raise RuntimeError('Dataset reviewer is active. No experiment GPU work was started.')
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if any(int(v) > 1024 for v in usage.split()):
        raise RuntimeError('GPU is occupied. No experiment GPU work was started.')


def load_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    assert rows and len({r['id'] for r in rows}) == len(rows)
    return rows


def visual_inputs(row, length=73, width=384, context_fps=2.0):
    """Identical 24-fps source interval and Picture order for teacher/student."""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    cap = cv2.VideoCapture(row['guide']); fps = cap.get(cv2.CAP_PROP_FPS); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count <= 0:
        raise ValueError('Unreadable guide ' + row['guide'])
    n = min(length, int(count / fps * 24)); n -= (n - 5) % 17
    if n < 5:
        raise ValueError('Guide shorter than H3 grid')
    frames = []
    # Sequential decoding avoids failed random seeks on some original files.
    wanted = [min(count - 1, round(i / 24 * fps)) for i in range(n)]; cursor = -1; bgr = None
    for index in wanted:
        while cursor < index:
            ok, bgr = cap.read(); cursor += 1
            if not ok:
                raise ValueError(f'Guide decoding failed at source frame {cursor}')
        frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        w = max(32, round(width / 32) * 32); h = max(32, round(frame.height / frame.width * w / 32) * 32)
        frames.append(frame.resize((w, h), Image.Resampling.LANCZOS))
    cap.release()
    picks = sorted({round(i * 24 / context_fps) for i in range(int((n - 1) / 24 * context_fps) + 1)})
    refs = []
    for path in row['references']:
        with Image.open(path) as raw:
            im = raw.convert('RGB'); scale = min(1.0, (w * h / (im.width * im.height)) ** .5)
            rw = max(32, round(im.width * scale / 32) * 32); rh = max(32, round(im.height * scale / 32) * 32)
            refs.append(im.resize((rw, rh), Image.Resampling.LANCZOS))
    return refs, frames, picks, dict(length=n, width=w, height=h, timestamps=[i / 24 for i in picks],
                                    context_fps=context_fps, input_key=row['input_key'])
