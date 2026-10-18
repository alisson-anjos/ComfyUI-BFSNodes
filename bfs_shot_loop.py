"""BFS Shot Loop: plan a long video into model-sized shots, run any workflow once per shot, join.

A video model only follows a guide reliably inside the clip length it was trained on (an H3 LoRA
trained on 107 frames loses the scene at 243). This splits the source into shots that fit, at
camera cuts when there are any, and hands them to the rest of the graph as a ComfyUI *list*.
ComfyUI then runs every node that receives a list once per item, pairing the lists by index, so
shot ``i`` always gets guide ``i``, reference ``i`` and prompt ``i``. ``BFS Shot Join`` takes the
decoded list back, trims each shot to its true length and concatenates them in order.

Every shot is generated at a length the model accepts (for example 17n+5 frames for MiniMax H3).
When a shot is shorter than that, the extra guide frames are taken from the video that follows it,
so the guide never repeats or stretches; the join then cuts the result back to the shot's own
frames, which keeps the timing identical to the source. Those extra frames also give the join a
real overlap to cross-fade across boundaries that are not camera cuts.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import math
import os
import subprocess
import time
from typing import Any

import numpy as np
import torch

import folder_paths

try:
    from . import bfs_video_io as VIO
except ImportError:   # loaded as a plain module (the tests): the file next to this one
    import importlib.util as _ilu
    _vio = _ilu.spec_from_file_location("bfs_video_io", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                     "bfs_video_io.py"))
    VIO = _ilu.module_from_spec(_vio)
    _vio.loader.exec_module(VIO)

# ---------------------------------------------------------------------------- frame grids

GRIDS: dict[str, tuple[int, int]] = {
    "H3 (17n+5)": (17, 5),
    "LTX / Wan (8n+1)": (8, 1),
    "Wan (4n+1)": (4, 1),
    "any": (1, 0),
}
DEFAULT_GRID = "H3 (17n+5)"
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def snap_up(n: int, grid: str) -> int:
    """Smallest valid clip length >= n."""
    step, off = GRIDS.get(grid, GRIDS[DEFAULT_GRID])
    n = max(1, int(n))
    if step == 1:
        return n
    if n <= off:
        return off
    return off + step * math.ceil((n - off) / step)


def snap_down(n: int, grid: str) -> int:
    """Largest valid clip length <= n (never below the grid's smallest length)."""
    step, off = GRIDS.get(grid, GRIDS[DEFAULT_GRID])
    n = max(1, int(n))
    if step == 1:
        return n
    if n <= off:
        return off
    return off + step * ((n - off) // step)


# ---------------------------------------------------------------------------- video io

def _input_path(name: str) -> str:
    path = folder_paths.get_annotated_filepath(name) if name else ""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"BFS Shot Loop: file not found in the input folder: {name!r}")
    return path


def _probe(path: str) -> dict:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"BFS Shot Loop: cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or n <= 0:
        raise ValueError(f"BFS Shot Loop: could not read fps/frame count from {path}")
    return {"fps_src": float(fps), "n_src": n, "width": w, "height": h, "duration": n / fps}


def _timeline(n_src: int, fps_src: float, fps: float) -> np.ndarray:
    """Source frame index for each frame of the resampled timeline (nearest frame, real time)."""
    n = max(1, int(math.floor(n_src / fps_src * fps + 1e-6)))
    idx = np.round(np.arange(n) * fps_src / fps).astype(np.int64)
    return np.clip(idx, 0, n_src - 1)


_STATUS_T = [0.0]


def _status(stage: str, done: int = 0, total: int = 0, force: bool = False) -> None:
    """Progress for the planner panel's loading bar (throttled)."""
    import time
    now = time.time()
    if not force and now - _STATUS_T[0] < 0.25 and done < total:
        return
    _STATUS_T[0] = now
    _notify("bfs-shotloop-status", {"stage": stage, "done": int(done), "total": int(total)})


def _read_frames(path: str, src_indices: np.ndarray, size: tuple[int, int] | None,
                 stage: str | None = None) -> list[np.ndarray]:
    """Decode the requested source frames (sorted, may repeat) as RGB uint8, sequentially."""
    import cv2
    want = sorted(set(int(i) for i in src_indices))
    top = want[-1] + 1 if want else 0
    got: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(path)
    i, k = 0, 0
    last = None
    while k < len(want):
        ok = cap.grab()
        if not ok:
            break
        if stage and i % 24 == 0:
            _status(stage, i, top)
        if i == want[k]:
            ok, fr = cap.retrieve()
            if ok:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                if size is not None:
                    fr = _fit(fr, size)
                last = fr
                got[i] = fr
            while k < len(want) and want[k] == i:
                k += 1
        i += 1
    cap.release()
    if not got:
        raise ValueError(f"BFS Shot Loop: no frames decoded from {path}")
    out = []
    for s in src_indices:
        s = int(s)
        out.append(got.get(s, last if s > max(got) else got[min(got, key=lambda x: abs(x - s))]))
    return out


def _fit(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Cover-resize then center-crop to (w, h)."""
    import cv2
    w, h = size
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (max(w, round(iw * s)), max(h, round(ih * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y:y + h, x:x + w]


def generation_size(src_w: int, src_h: int, megapixels: float, multiple: int) -> tuple[int, int]:
    """Aspect-preserving size with the given pixel area, both sides on the multiple grid."""
    multiple = max(8, int(multiple))
    area = max(0.05, float(megapixels)) * 1e6
    ar = src_w / max(1, src_h)
    h = math.sqrt(area / ar)
    w = h * ar
    w = max(multiple, int(round(w / multiple)) * multiple)
    h = max(multiple, int(round(h / multiple)) * multiple)
    return w, h


def plan_generation_size(src_w: int, src_h: int, plan: dict, multiple: int) -> tuple[int, int]:
    """Resolve the planner's megapixel, source-resolution, or custom generation size."""
    mode = str(plan.get("generation_size_mode") or "megapixels")
    if mode == "original":
        mp = max(0.05, int(src_w) * int(src_h) / 1e6)
        return generation_size(src_w, src_h, mp, multiple)
    if mode == "custom":
        multiple = max(8, int(multiple))
        w = max(multiple, int(round(max(1, int(plan.get("custom_width") or 1024)) / multiple)) * multiple)
        h = max(multiple, int(round(max(1, int(plan.get("custom_height") or 1024)) / multiple)) * multiple)
        return w, h
    return generation_size(src_w, src_h, float(plan.get("megapixels", 0.5)), multiple)


def audio_info(path: str) -> dict:
    """Whether the video has a usable audio track (PyAV, no ffmpeg binary needed)."""
    try:
        import av
        with av.open(path) as c:
            if not c.streams.audio:
                return {"has_audio": False, "reason": "no audio track"}
            st = c.streams.audio[0]
            cc = st.codec_context
            ch = getattr(cc, "channels", 0) or (len(cc.layout.channels) if getattr(cc, "layout", None) else 0)
            return {"has_audio": True, "sample_rate": int(cc.sample_rate or 0), "channels": int(ch or 0),
                    "codec": cc.name}
    except Exception as exc:  # noqa: BLE001
        return {"has_audio": False, "reason": f"audio unreadable ({type(exc).__name__})"}


def silence(seconds: float, sample_rate: int = 44100, channels: int = 2) -> dict:
    """A silent AUDIO of the given length: Create Video and friends always get a valid track."""
    n = max(1, int(round(max(0.0, seconds) * sample_rate)))
    return {"waveform": torch.zeros(1, channels, n), "sample_rate": sample_rate}


def _usable(audio: dict | None) -> bool:
    try:
        return audio is not None and audio["waveform"].ndim == 3 and audio["waveform"].shape[-1] > 0
    except Exception:  # noqa: BLE001
        return False


def _read_audio_av(path: str, sample_rate: int) -> dict | None:
    """PyAV fallback when the ffmpeg binary is missing."""
    import av
    with av.open(path) as c:
        if not c.streams.audio:
            return None
        st = c.streams.audio[0]
        layout = "stereo" if (getattr(st.codec_context, "channels", 2) or 2) >= 2 else "mono"
        rs = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
        chunks = []
        for fr in c.decode(st):
            for r in rs.resample(fr):
                chunks.append(r.to_ndarray())
        for r in rs.resample(None):
            chunks.append(r.to_ndarray())
    if not chunks:
        return None
    a = np.concatenate(chunks, axis=1).astype(np.float32)
    return {"waveform": torch.from_numpy(a)[None], "sample_rate": sample_rate}


def _read_audio(path: str, sample_rate: int = 44100) -> dict | None:
    """Decode the whole soundtrack as float32 [1, C, T] (ffmpeg, else PyAV); None when there is none."""
    a = _read_audio_ffmpeg(path, sample_rate)
    if not _usable(a):
        try:
            a = _read_audio_av(path, sample_rate)
        except Exception:  # noqa: BLE001 - audio is optional
            a = None
    return a if _usable(a) else None


def _read_audio_ffmpeg(path: str, sample_rate: int = 44100) -> dict | None:
    try:
        probe = subprocess.run([VIO.ffprobe_bin() or "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                "stream=channels", "-of", "csv=p=0", path],
                               capture_output=True, text=True, timeout=60)
        ch = int((probe.stdout.strip() or "0").split(",")[0] or 0)
        if ch <= 0:
            return None
        ch = min(ch, 2)
        raw = subprocess.run([VIO.ffmpeg_bin() or "ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", str(ch), "-ar", str(sample_rate),
                              "-f", "f32le", "-"], capture_output=True, timeout=600).stdout
        if not raw:
            return None
        a = np.frombuffer(raw, dtype=np.float32).reshape(-1, ch).T.copy()
        return {"waveform": torch.from_numpy(a)[None], "sample_rate": sample_rate}
    except Exception:  # noqa: BLE001 - audio is optional
        return None


def _slice_audio(audio: dict | None, start_s: float, dur_s: float) -> dict | None:
    if audio is None:
        return None
    sr = audio["sample_rate"]
    wf = audio["waveform"]
    a, n = int(round(start_s * sr)), int(round(dur_s * sr))
    seg = wf[..., a:a + n]
    if seg.shape[-1] < n:
        seg = torch.nn.functional.pad(seg, (0, n - seg.shape[-1]))
    return {"waveform": seg.contiguous(), "sample_rate": sr}


# ---------------------------------------------------------------------------- analysis

_ANALYSIS_CACHE: dict[tuple, dict] = {}


def analyze(path: str, fps: float, thumbs: int = 120, thumb_h: int = 72) -> dict:
    """Probe, per-frame cut score on the resampled timeline, and a strip of thumbnails."""
    import cv2
    key = (path, os.path.getmtime(path), round(float(fps), 4), thumbs, thumb_h)
    if key in _ANALYSIS_CACHE:
        return _ANALYSIS_CACHE[key]
    info = _probe(path)
    src = _timeline(info["n_src"], info["fps_src"], fps)
    n = len(src)
    _status("Reading the video", 0, info["n_src"], force=True)
    # small frames straight from the decoder: the full-size frames of a long video do not fit in memory
    sw = 64
    sh = max(8, int(round(sw * info["height"] / max(1, info["width"]))))   # keep the aspect: nothing cropped
    small = _read_frames(path, src, (sw, sh), stage="Reading the video")
    hsv = [cv2.calcHist([cv2.cvtColor(s, cv2.COLOR_RGB2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256]) for s in small]
    hsv = [cv2.normalize(h, h).flatten() for h in hsv]
    raw = np.zeros(n, np.float32)
    for i in range(1, n):
        pix = np.abs(small[i].astype(np.float32) - small[i - 1].astype(np.float32)).mean() / 255.0
        hist = cv2.compareHist(hsv[i - 1], hsv[i], cv2.HISTCMP_BHATTACHARYYA)
        raw[i] = 0.5 * pix + 0.5 * float(hist)
    # local normalisation: a cut stands out from the motion around it
    score = np.zeros(n, np.float32)
    w = max(4, int(round(fps)))
    for i in range(1, n):
        lo, hi = max(1, i - w), min(n, i + w + 1)
        neigh = np.concatenate([raw[lo:i], raw[i + 1:hi]])
        base = float(np.median(neigh)) if len(neigh) else 0.0
        score[i] = raw[i] / (base + 0.02)
    k = max(1, n // max(1, thumbs))
    th = []
    tw = max(16, int(round(thumb_h * info["width"] / max(1, info["height"]))))
    frames_for_thumbs = _read_frames(path, src[::k], (tw, thumb_h), stage="Making thumbnails")
    for j, fr in enumerate(frames_for_thumbs):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
        th.append({"f": int(j * k), "src": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()})
    out = dict(info, fps=float(fps), n=n, thumbs=th, thumb_w=tw, thumb_h=thumb_h, audio=audio_info(path),
               raw=[round(float(x), 4) for x in raw], score=[round(float(x), 3) for x in score])
    _ANALYSIS_CACHE[key] = out
    return out


_PSD_CACHE: dict[tuple, list[float]] = {}


def scenedetect_cut_times(path: str, detector: str, sensitivity: float) -> list[float] | None:
    """Cut times in seconds from PySceneDetect, or None when the package is missing."""
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import AdaptiveDetector, ContentDetector
    except ImportError:
        return None
    sensitivity = float(min(1.0, max(0.0, sensitivity)))
    key = (path, os.path.getmtime(path), detector, round(sensitivity, 3))
    if key in _PSD_CACHE:
        return _PSD_CACHE[key]
    _status("Detecting camera cuts (PySceneDetect)", force=True)
    video = open_video(path)
    min_len = max(3, int(round(video.frame_rate * 0.25)))
    sm = SceneManager()
    if detector == "content":
        sm.add_detector(ContentDetector(threshold=45.0 - 33.0 * sensitivity, min_scene_len=min_len))
    else:
        sm.add_detector(AdaptiveDetector(adaptive_threshold=5.0 - 3.5 * sensitivity, min_scene_len=min_len))
    sm.detect_scenes(video, show_progress=False)
    times = [float(a.get_seconds()) for a, _ in sm.get_scene_list()[1:]]
    _PSD_CACHE[key] = times
    return times


def find_cuts(path: str, analysis: dict, detector: str, sensitivity: float) -> tuple[list[int], str]:
    """Cut frames on the resampled timeline and which detector produced them."""
    fps = float(analysis["fps"])
    if detector in ("adaptive", "content"):
        times = scenedetect_cut_times(path, detector, sensitivity)
        if times is not None:
            n = analysis["n"]
            return sorted({int(round(t * fps)) for t in times if 0 < round(t * fps) < n}), f"PySceneDetect {detector}"
    return detect_cuts(analysis["score"], analysis["raw"], sensitivity, fps), "builtin"


def detect_cuts(score: list[float], raw: list[float], sensitivity: float, fps: float) -> list[int]:
    """Frame indices that start a new shot (built-in detector). Higher sensitivity finds more cuts."""
    sensitivity = float(min(1.0, max(0.0, sensitivity)))
    thr = 12.0 - 10.0 * sensitivity          # local ratio threshold: 12 (strict) .. 2 (loose)
    abs_min = 0.12 - 0.09 * sensitivity      # ignore tiny global changes
    gap = max(3, int(round(fps * 0.25)))      # no two cuts closer than a quarter second
    cand = [i for i in range(1, len(score)) if score[i] >= thr and raw[i] >= abs_min]
    cuts: list[int] = []
    for i in sorted(cand, key=lambda j: -score[j]):
        if all(abs(i - c) >= gap for c in cuts):
            cuts.append(i)
    return sorted(cuts)


# ---------------------------------------------------------------------------- planning

def plan_range(n: int, fps: float, start_s: float = 0.0, end_s: float = 0.0, max_total_s: float = 0.0,
               start_f: int | None = None, end_f: int | None = None) -> tuple[int, int]:
    """The part of the video the plan works on, in timeline frames [lo, hi): start / end frame (end 0 = the end of the
    video; older plans give seconds), then 'max total seconds' counted from the start."""
    a = int(start_f) if start_f is not None else int(round(float(start_s or 0) * fps))
    b = int(end_f) if end_f is not None else (int(round(float(end_s) * fps)) if end_s and float(end_s) > 0 else 0)
    lo = max(0, min(n - 1, a))
    hi = n if b <= 0 else max(lo + 1, min(n, b))
    if max_total_s and float(max_total_s) > 0:
        hi = min(hi, lo + max(1, int(round(float(max_total_s) * fps))))
    return lo, hi


def plan_segments(n: int, cuts: list[int], mode: str, max_len: int, min_len: int,
                  max_parts: int = 0, max_total: int = 0, manual: list[int] | None = None, start: int = 0) -> list[dict]:
    """Split [start, n) into shots (n = the end frame). Returns [{start, end, cut_before}]. Lengths are in frames."""
    lo = max(0, int(start or 0))
    if max_total and max_total > 0:
        n = min(n, lo + int(max_total))
    lo = min(lo, n - 1)
    max_len = max(1, int(max_len))
    min_len = max(1, min(int(min_len), max_len))
    cuts = sorted(c for c in set(cuts or []) if lo < c < n)
    cutset = set(cuts)
    if mode == "manual" and manual:
        bounds = sorted(set([lo, n] + [b for b in manual if lo < b < n]))
    elif mode == "single":       # the whole range as ONE shot, whatever its length (the maximum is not applied)
        bounds = [lo, n]
    elif mode == "fixed":
        k = max(1, math.ceil((n - lo) / max_len))
        bounds = sorted(set(lo + round((n - lo) * j / k) for j in range(k + 1)))
    else:  # shots
        bounds = [lo] + cuts + [n]
        # merge shots that are too short into a neighbour, as long as the merge still fits
        changed = True
        while changed:
            changed = False
            for j in range(len(bounds) - 1):
                if bounds[j + 1] - bounds[j] >= min_len or len(bounds) <= 2:
                    continue
                left = bounds[j + 1] - bounds[j - 1] if j > 0 else None
                right = bounds[j + 2] - bounds[j] if j + 2 < len(bounds) else None
                opts = [(v, side) for v, side in ((left, "l"), (right, "r")) if v is not None and v <= max_len]
                if not opts:
                    continue
                side = min(opts)[1]
                bounds.pop(j if side == "l" else j + 1)
                changed = True
                break
        # split shots that are too long into equal parts
        out = [lo]
        for a, b in zip(bounds[:-1], bounds[1:]):
            k = max(1, math.ceil((b - a) / max_len))
            out += [a + round((b - a) * j / k) for j in range(1, k + 1)]
        bounds = sorted(set(out))
    segs = [{"start": a, "end": b, "cut_before": a in cutset} for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    if max_parts and max_parts > 0:
        segs = segs[:int(max_parts)]
    return segs


DEFAULT_PLAN = {
    "video": "", "fps": 24.0, "grid": DEFAULT_GRID, "mode": "shots", "max_s": 4.5, "min_s": 1.0,
    "sensitivity": 0.5, "max_parts": 0, "max_total_s": 0.0, "start_s": 0.0, "end_s": 0.0, "bounds": [], "segs": [],
    "global_refs": [], "global_ref": "", "global_ref2": "", "global_prompt": "", "global_color_guide_ref1": "#00ff00",
    "global_color_guide_ref2": "#9b30ff", "picture_colors": [], "megapixels": 0.5, "multiple": 32,
    "generation_size_mode": "megapixels", "custom_width": 1024, "custom_height": 1024,
    "detector": "adaptive", "run": "recursive", "filters": {}, "skip_fill": "original",
    "cast": {}, "cast_split": False, "cast_only": False, "mask_cfg": {}, "vlm_cfg": {}, "audio_mode": "auto", "ref_details": {},
    "picture_details": {}, "picture_cfg": {}, "auto_prompt": {}, "auto_prompts": {}, "between_shots": "keep", "shot_cache": "output", "shot_storage": "lossless", "unload_before_join": True,
}


# runs that render one shot at a time and store it on disk: "queue" (the panel re-queues the workflow after each
# shot) and "recursive" (BFS Shot Join expands the graph for the next shot inside the same prompt)
STORED_RUNS = ("queue", "recursive")


# Second pass (upscale): the planner only works out the size; the graph does the upscale and the second sampling
# (e.g. Minimax H3 Latent Upscaler (3D) in "target dimensions" mode, then a sampler on the upscaled latent).
SECOND_PASS_DEFAULTS = {"enabled": False, "mode": "scale", "scale": 2.0, "width": 1280, "height": 704,
                        "megapixels": 1.0, "align": 32}
VAE_DOWNSAMPLE = 16


def second_pass_size(w: int, h: int, sp: dict | None) -> tuple[int, int]:
    """The upscaled size of a w x h generation, computed like Minimax H3 Latent Upscaler (3D): target pixels by
    multiplier, target dimensions or megapixels (keeping the aspect), rounded to `align`, then to whole latent cells."""
    c = {**SECOND_PASS_DEFAULTS, **(sp or {})}
    mode = c["mode"]
    if mode == "dims":
        tw, th = float(c["width"]), float(c["height"])
    elif mode == "megapixels":
        px = float(c["megapixels"]) * 1024 * 1024
        th = (px / (w / h)) ** 0.5
        tw = th * (w / h)
    else:
        tw, th = w * float(c["scale"]), h * float(c["scale"])
    al = max(1, int(c["align"]))
    tw, th = round(tw / al) * al, round(th / al) * al
    d = VAE_DOWNSAMPLE
    return max(d, int(round(tw / d) * d)), max(d, int(round(th / d) * d))


def _load_plan(plan_json: str) -> dict:
    try:
        p = json.loads(plan_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"BFS Shot Planner: the plan is not valid JSON ({exc})") from exc
    out = dict(DEFAULT_PLAN)
    out.update({k: v for k, v in p.items() if v is not None})
    out["global_refs"] = ref_list(p.get("global_refs"), out.get("global_ref"), out.get("global_ref2"))
    out["global_ref"], out["global_ref2"] = (out["global_refs"] + ["", ""])[:2]
    return out


MAX_REFS = 12   # reference pictures per shot, <Picture 1..12>; the continuity and setting pictures come after them


def ref_list(refs, ref="", ref2="") -> list[str]:
    """A reference list of the plan: `refs` (plans from 1.113 on), else the old ref / ref2 pair. '' = not set (a shot's
    empty slot uses the global reference at the same position); empty slots at the end are dropped."""
    out = [str(x or "") for x in refs] if isinstance(refs, list) else [str(ref or ""), str(ref2 or "")]
    out = out[:MAX_REFS]
    while out and not out[-1]:
        out.pop()
    return out


def shot_ref_names(own: list, global_refs: list) -> list[str]:
    """The pictures a shot uses, in <Picture n> order: its own slot, else the global one at the same position. A slot
    with neither is left out, so the pictures after it move up one number."""
    n = max(len(own), len(global_refs))
    names = [(own[k] if k < len(own) else "") or (global_refs[k] if k < len(global_refs) else "") for k in range(n)]
    return [x for x in names if x][:MAX_REFS]


def shot_refs(shot: dict) -> list:
    """The reference images of a shot handed out by the planner, in <Picture n> order."""
    refs = shot.get("refs")
    if refs is None:
        refs = [shot.get("ref"), shot.get("ref2")]
    return [r for r in refs if r is not None]


def resolve_plan(plan: dict, analysis: dict, path: str | None = None) -> list[dict]:
    """Shots for this plan: the boundaries edited in the panel, or an automatic split."""
    fps = float(analysis["fps"])
    grid = plan["grid"]
    max_len = snap_down(int(round(float(plan["max_s"]) * fps)), grid)
    min_len = max(1, int(round(float(plan["min_s"]) * fps)))
    lo, hi = plan_range(analysis["n"], fps, plan.get("start_s", 0), plan.get("end_s", 0), plan.get("max_total_s", 0),
                        plan.get("start_f"), plan.get("end_f"))
    if path:
        cuts, _ = find_cuts(path, analysis, plan.get("detector", "adaptive"), float(plan["sensitivity"]))
    else:
        cuts = detect_cuts(analysis["score"], analysis["raw"], float(plan["sensitivity"]), fps)
    manual = plan.get("bounds") or None
    mode = "manual" if manual else plan["mode"]
    cast = cached_cast(path, analysis) if path else None   # only once the people were analysed
    if cast is not None and plan.get("cast_split") and not manual:
        extra = []
        bounds = [0] + cuts + [analysis["n"]]
        for a, b in zip(bounds[:-1], bounds[1:]):
            extra += person_change_points(cast, a, b, max(min_len, int(round(fps))))
        cuts = sorted(set(cuts) | set(extra))
    segs = plan_segments(hi, cuts, mode, max_len, min_len, int(plan.get("max_parts") or 0), 0, manual, start=lo)
    meta = plan.get("segs") or []
    for i, s in enumerate(segs):
        m = meta[i] if i < len(meta) and isinstance(meta[i], dict) else {}
        s["enabled"] = bool(m.get("enabled", True))
        s["refs"] = ref_list(m.get("refs"), m.get("ref"), m.get("ref2"))
        s["ref"], s["ref2"] = (s["refs"] + ["", ""])[:2]
        s["prompt"] = m.get("prompt") or ""
        s["chain"] = m.get("chain") or "head"   # a continuing take starts on the previous end (cuts skip it)
        s["chain_frame"] = m.get("chain_frame") or "first"
        s["chain_history"] = int(m.get("chain_history") or 22)
        s["crop"] = bool(m.get("crop"))
        s["inpaint"] = bool(m.get("inpaint"))
        s["paste"] = bool(m.get("paste")) and not s["crop"] and not s["inpaint"]
        s["paste_text"] = str(m.get("paste_text") or "")
        s["color_guide"] = bool(m.get("color_guide"))
        s["color_guide_color"] = str(m.get("color_guide_color") or "")
        s["color_guide_text2"] = str(m.get("color_guide_text2") or "")
        s["color_guide_color2"] = str(m.get("color_guide_color2") or "")
        s["color_guide_points2"] = list(m.get("color_guide_points2") or [])
        s["color_guide_key2"] = int(m.get("color_guide_key2") or 0)
        s["color_guide_opacity"] = min(0.8, max(0.05, float(m.get("color_guide_opacity") or 0.35)))
        s["strength"] = min(1.0, max(0.05, float(m.get("strength") or 1.0)))
        s["mask"] = m.get("mask") or {}
        s["target"] = m.get("target") or ""
        s["cut_before"] = bool(s.get("cut_before")) or (s["start"] in cuts)
        s["gen_len"] = snap_up(s["end"] - s["start"], grid)
        s["people"], s["main"] = [], -1
        s["color_guide_color"] = s["color_guide_color"] or str(plan.get("global_color_guide_ref1") or "#00ff00")
        s["color_guide_color2"] = s["color_guide_color2"] or str(plan.get("global_color_guide_ref2") or "#9b30ff")
        s["color_guide_marks"] = color_marks(plan, m)
        s["color_guide_color"] = s["color_guide_marks"][0]["color"]
        if cast is not None:
            sp = shot_people(cast, s["start"], s["end"])
            s["people"], s["main"] = sp["people"], sp["main"]
            included = [p for p in sp["people"] if not (plan.get("cast") or {}).get(str(p), {}).get("ignore")]
            s["cast_skip"] = bool(plan.get("cast_only")) and not included
    return segs


# ---------------------------------------------------------------------------- content filters

DEFAULT_FILTERS = {
    "person": False, "min_person_area": 0.0, "max_persons": 0, "face": False,
    "skip_dark": False, "dark_level": 0.06, "skip_static": False, "static_level": 0.004,
    "min_frames": 0, "samples": 6,
}
_DET: dict[str, Any] = {}
_STATS_CACHE: dict[tuple, dict] = {}


def _yolo(kind: str):
    """YOLO model for 'person' or 'face' from models/ultralytics, or None (OpenCV fallback)."""
    if _DET.get(kind) is not None:
        return _DET[kind]
    model = None
    try:
        from ultralytics import YOLO
        root = os.path.join(folder_paths.models_dir, "ultralytics")
        cands = []
        for sub in ("bbox", "segm", "pose", ""):
            d = os.path.join(root, sub)
            if os.path.isdir(d):
                cands += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".pt") and kind in f.lower()]
        if cands:
            model = YOLO(cands[0])
    except Exception:  # noqa: BLE001 - fall back to OpenCV
        model = None
    _DET[kind] = model
    return model


def _detect(frames: list[np.ndarray]) -> list[dict]:
    """Per frame: person boxes (area fraction) and face count."""
    import cv2
    out = [{"persons": [], "faces": 0} for _ in frames]
    pm, fm = _yolo("person"), _yolo("face")
    if pm is not None:
        for i, r in enumerate(pm(frames, verbose=False, conf=0.35, classes=[0])):
            h, w = frames[i].shape[:2]
            for b in r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []:
                out[i]["persons"].append(float((b[2] - b[0]) * (b[3] - b[1]) / (w * h)))
    else:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        for i, fr in enumerate(frames):
            h, w = fr.shape[:2]
            rects, _ = hog.detectMultiScale(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), winStride=(8, 8))
            out[i]["persons"] = [float(rw * rh / (w * h)) for (_, _, rw, rh) in rects]
    if fm is not None:
        for i, r in enumerate(fm(frames, verbose=False, conf=0.4)):
            out[i]["faces"] = int(len(r.boxes)) if r.boxes is not None else 0
    else:
        casc = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        for i, fr in enumerate(frames):
            out[i]["faces"] = int(len(casc.detectMultiScale(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), 1.1, 5)))
    return out


def shot_stats(path: str, analysis: dict, seg: dict, samples: int = 6) -> dict:
    """Sampled content statistics for one shot (cached)."""
    key = (path, os.path.getmtime(path), float(analysis["fps"]), seg["start"], seg["end"], samples)
    if key in _STATS_CACHE:
        return _STATS_CACHE[key]
    src = _timeline(analysis["n_src"], analysis["fps_src"], float(analysis["fps"]))
    k = max(1, min(samples, seg["end"] - seg["start"]))
    pick = np.linspace(seg["start"], seg["end"] - 1, k).round().astype(int)
    w = 640
    h = max(32, int(round(w * analysis["height"] / max(1, analysis["width"]))))
    frames = _read_frames(path, src[pick], (w, h))
    det = _detect(frames)
    raw = analysis.get("raw") or []
    motion = float(np.mean(raw[seg["start"] + 1:seg["end"]])) if seg["end"] - seg["start"] > 1 and raw else 0.0
    st = {
        "persons": int(max(len(d["persons"]) for d in det)),
        "person_area": round(float(max([max(d["persons"]) for d in det if d["persons"]] or [0.0])), 4),
        "person_frames": int(sum(1 for d in det if d["persons"])),
        "faces": int(max(d["faces"] for d in det)),
        "brightness": round(float(np.mean([f.mean() / 255.0 for f in frames])), 4),
        "motion": round(motion, 4), "sampled": int(k),
    }
    _STATS_CACHE[key] = st
    return st


def skip_reason(stats: dict, length: int, f: dict) -> str:
    """Why a shot should be skipped under these filters ('' = keep)."""
    if f.get("min_frames") and length < int(f["min_frames"]):
        return f"shorter than {int(f['min_frames'])} frames"
    if f.get("skip_dark") and stats["brightness"] < float(f.get("dark_level", 0.06)):
        return "dark / fade"
    if f.get("skip_static") and stats["motion"] < float(f.get("static_level", 0.004)):
        return "static"
    if f.get("person") and stats["persons"] == 0:
        return "no person"
    if f.get("person") and float(f.get("min_person_area") or 0) > 0 and stats["person_area"] < float(f["min_person_area"]):
        return f"person smaller than {float(f['min_person_area']) * 100:.0f}% of the frame"
    if int(f.get("max_persons") or 0) > 0 and stats["persons"] > int(f["max_persons"]):
        return f"more than {int(f['max_persons'])} people"
    if f.get("face") and stats["faces"] == 0:
        return "no face"
    return ""


def filters_active(f: dict) -> bool:
    return any(f.get(k) for k in ("person", "face", "skip_dark", "skip_static")) or \
        int(f.get("max_persons") or 0) > 0 or int(f.get("min_frames") or 0) > 0


def apply_filters(plan: dict, analysis: dict, path: str, segs: list[dict]) -> list[dict]:
    """Mark each shot with stats and an automatic skip reason; per-shot 'force' overrides it."""
    f = dict(DEFAULT_FILTERS); f.update(plan.get("filters") or {})
    meta = plan.get("segs") or []
    need = filters_active(f)
    for i, s in enumerate(segs):
        force = (meta[i] or {}).get("force", "auto") if i < len(meta) and isinstance(meta[i], dict) else "auto"
        s["force"] = force
        s["stats"] = shot_stats(path, analysis, s, int(f.get("samples") or 6)) if need else None
        s["skip_reason"] = skip_reason(s["stats"], s["end"] - s["start"], f) if need else ""
        if not s["skip_reason"] and s.get("cast_skip"):
            s["skip_reason"] = "no detected person"
        s["run"] = s["enabled"] and (force == "run" or (force != "skip" and not s["skip_reason"]))
    return segs


# ---------------------------------------------------------------------------- cast (people by face)

_FACE_APP: Any = None
_CAST_CACHE: dict[tuple, dict] = {}


def _face_app():
    """InsightFace detector + ArcFace recogniser (buffalo_l), or None when unavailable."""
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP or None
    try:
        import insightface
        root = os.path.join(folder_paths.models_dir, "insightface")
        # CPU only: light, no VRAM, and no CUDA/cuDNN loading (a missing cuDNN aborts the whole process)
        app = insightface.app.FaceAnalysis(name="buffalo_l", root=root, allowed_modules=["detection", "recognition"],
                                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(320, 320))
        # small models: a few threads are faster than onnxruntime's default pool (one per core) and stay light
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = min(4, os.cpu_count() or 4)
        so.inter_op_num_threads = 1
        for model in app.models.values():
            model.session = ort.InferenceSession(model.model_file, so, providers=["CPUExecutionProvider"])
        _FACE_APP = app
    except Exception as exc:  # noqa: BLE001
        print(f"[BFS Shot Planner] face analysis unavailable: {exc!r}")
        _FACE_APP = False
    return _FACE_APP or None


def _cast_key(path: str, analysis: dict, step_s: float = 0.5, threshold: float = 0.42, min_share: float = 0.01) -> tuple:
    return (path, os.path.getmtime(path), float(analysis["fps"]), round(step_s, 3), round(threshold, 3), round(min_share, 4),
            "insightface" if _face_app() is not None else "opencv" if _cv_face() is not None else "sam3")


APPEARANCE_THRESHOLD = 0.86   # cosine on the colour signature (only when DINOv2 cannot load)
SFACE_THRESHOLD = 0.36        # cosine on SFace features (OpenCV's same-identity threshold is 0.363)
_CV_FACE: dict = {}
CV_FACE_MODELS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def _cv_face():
    """OpenCV Zoo's YuNet (detection) + SFace (recognition): Apache-2.0, run by OpenCV itself (no extra package),
    downloaded once into models/opencv_face (0.2 + 37 MB). None when it cannot load."""
    if "m" not in _CV_FACE:
        try:
            import cv2
            import urllib.request
            d = os.path.join(folder_paths.models_dir, "opencv_face")
            os.makedirs(d, exist_ok=True)
            for name, url in CV_FACE_MODELS.items():
                f = os.path.join(d, name)
                if not os.path.exists(f) or os.path.getsize(f) < 100000:
                    print(f"[BFS Shot Planner] downloading {name} (OpenCV Zoo, Apache-2.0)", flush=True)
                    urllib.request.urlretrieve(url, f + ".part")
                    os.replace(f + ".part", f)
            det = cv2.FaceDetectorYN.create(os.path.join(d, "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.6, 0.3, 50)
            rec = cv2.FaceRecognizerSF.create(os.path.join(d, "face_recognition_sface_2021dec.onnx"), "")
            _CV_FACE["m"] = (det, rec)
        except Exception as exc:  # noqa: BLE001
            print(f"[BFS Shot Planner] OpenCV face models unavailable ({exc!r})", flush=True)
            _CV_FACE["m"] = None
    return _CV_FACE["m"]


def appearance_signature(frame: np.ndarray, box) -> np.ndarray:
    """A license-free identity signature for grouping people inside ONE video: colour histograms of the head (hair,
    skin) and of the torso under it (clothes), plus a tiny grey thumbnail of the face. Weaker than ArcFace across
    videos, enough to tell the people of a clip apart."""
    import cv2
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    bw, bh = max(2, x1 - x0), max(2, y1 - y0)
    head = frame[max(0, y0 - bh // 4):min(H, y1), max(0, x0 - bw // 6):min(W, x1 + bw // 6)]
    torso = frame[min(H - 1, y1):min(H, y1 + 2 * bh), max(0, x0 - bw // 3):min(W, x1 + bw // 3)]

    def hist(img):
        if img.size == 0:
            return np.zeros(12 * 4 * 4, np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel().astype(np.float32)
        return np.sqrt(h / max(1.0, h.sum()))      # Hellinger: cosine of sqrt-histograms
    face = frame[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    thumb = cv2.resize(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY), (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() \
        if face.size else np.zeros(64, np.float32)
    thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6) * 0.05
    v = np.concatenate([hist(head), 1.3 * hist(torso), thumb])
    return v / (np.linalg.norm(v) + 1e-8)


def _sam3_faces(frames: list, threshold: float = 0.4) -> list:
    """Faces per frame with SAM 3.1 (ComfyUI core): [[(x0, y0, x1, y1, score), ...], ...] in pixels."""
    from comfy_extras.nodes_sam3 import SAM3_Detect
    model, clip = _sam3()
    cond = clip.encode_from_tokens_scheduled(clip.tokenize("face"))
    out = []
    for k in range(0, len(frames), 8):
        _status("Finding faces (SAM 3.1)", k, len(frames))
        batch = torch.from_numpy(np.stack(frames[k:k + 8]).astype(np.float32) / 255.0)
        boxes = SAM3_Detect.execute(model=model, image=batch, conditioning=cond, threshold=threshold,
                                    refine_iterations=0).args[1]
        for fb in boxes:
            out.append([(b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"], b.get("score", 1.0)) for b in (fb or [])])
    return out


def cached_cast(path: str, analysis: dict) -> dict | None:
    """The cast of a video when it was already analysed with the default settings, else None (never computes)."""
    try:
        return _CAST_CACHE.get(_cast_key(path, analysis))
    except OSError:
        return None


def analyze_cast(path: str, analysis: dict, step_s: float = 0.5, threshold: float = 0.42,
                 min_share: float = 0.01, max_faces: int = 3) -> dict:
    """Detect faces every `step_s` seconds (CPU), keep the `max_faces` most probable ones per frame, group them
    into people by ArcFace similarity.

    Returns {"people": [{id, thumb, count, share}], "samples": [{f, faces: [{pid, area}]}]} where
    `f` is a timeline frame. Person ids are stable for the same video and settings.
    """
    import cv2
    key = _cast_key(path, analysis, step_s, threshold, min_share)
    if key in _CAST_CACHE:
        return _CAST_CACHE[key]
    # InsightFace (ArcFace) when installed; else OpenCV's YuNet + SFace (Apache-2.0, no extra package); else SAM 3.1
    # faces grouped by colours (hair, skin, clothes)
    app = _face_app()
    cvf = _cv_face() if app is None else None
    if app is None:
        threshold = SFACE_THRESHOLD if cvf is not None else APPEARANCE_THRESHOLD
        print("[BFS Shot Planner] Find people: insightface is not installed, using "
              + ("OpenCV YuNet + SFace faces" if cvf is not None else "SAM 3.1 faces + colours (hair, skin, clothes)"),
              flush=True)
    fps = float(analysis["fps"])
    step = max(1, int(round(step_s * fps)))
    frames_idx = np.arange(0, analysis["n"], step)
    src = _timeline(analysis["n_src"], analysis["fps_src"], fps)
    w = 960
    h = max(32, int(round(w * analysis["height"] / max(1, analysis["width"]))))
    frames = _read_frames(path, src[frames_idx], (w, h), stage="Reading frames for faces")
    dets = []   # (sample index, embedding, area fraction, crop)
    if app is None and cvf is not None:
        det, rec = cvf
        det.setInputSize((w, h))
        for si, fr in enumerate(frames):
            _status("Finding faces", si, len(frames))
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            _, faces = det.detect(bgr)
            cand = []
            for row in (faces if faces is not None else []):
                x, y, bw, bh, score = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[-1])
                area = max(0.0, bw) * max(0.0, bh) / float(w * h)
                if area >= 0.0015 and score >= 0.6:
                    cand.append((score, row, area))
            for score, row, area in sorted(cand, key=lambda c: -c[0])[:max_faces]:
                feat = rec.feature(rec.alignCrop(bgr, row)).ravel().astype(np.float32)
                x0, y0, x1, y1 = int(row[0]), int(row[1]), int(row[0] + row[2]), int(row[1] + row[3])
                pad = int(0.25 * max(x1 - x0, y1 - y0))
                crop = fr[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
                dets.append((si, feat / (np.linalg.norm(feat) + 1e-8), area, crop))
    elif app is None:
        for si, (fr, faces) in enumerate(zip(frames, _sam3_faces(frames, threshold=0.55))):
            cand = []
            for x0, y0, x1, y1, score in faces:
                bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
                area = bw * bh / float(w * h)
                if area >= 0.0015 and 0.55 <= bw / bh <= 1.4:     # a face-shaped box (logos and text are skipped)
                    cand.append((float(score), (x0, y0, x1, y1), area))
            for score, box, area in sorted(cand, reverse=True)[:max_faces]:
                x0, y0, x1, y1 = [int(v) for v in box]
                pad = int(0.25 * max(x1 - x0, y1 - y0))
                crop = fr[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
                dets.append((si, appearance_signature(fr, box), area, crop))
    frames_for_faces = frames if app is not None else []
    from_insight = app is not None
    if from_insight:
        from insightface.app.common import Face
        rec = app.models["recognition"]
    for si, fr in enumerate(frames_for_faces):
        _status("Finding faces", si, len(frames))
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        boxes, kpss = app.det_model.detect(bgr, max_num=0, metric="default")
        # only the most probable faces of the frame get recognised, so a crowd stays cheap and out of the cast
        cand = []
        for k in range(boxes.shape[0]):
            x0, y0, x1, y1 = [int(v) for v in boxes[k, :4]]
            area = max(0, x1 - x0) * max(0, y1 - y0) / float(w * h)
            if area >= 0.0015 and boxes[k, 4] >= 0.5:
                cand.append((float(boxes[k, 4]), k, area))
        for score, k, area in sorted(cand, reverse=True)[:max_faces]:
            f = Face(bbox=boxes[k, :4], kps=kpss[k] if kpss is not None else None, det_score=score)
            rec.get(bgr, f)
            x0, y0, x1, y1 = [int(v) for v in f.bbox]
            pad = int(0.25 * max(x1 - x0, y1 - y0))
            crop = fr[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
            dets.append((si, f.normed_embedding.astype(np.float32), area, crop))
    # greedy online clustering on cosine similarity, then merge close clusters
    cents, members = [], []
    for k, (_, e, _, _) in enumerate(dets):
        if cents:
            sims = np.array([c @ e / (np.linalg.norm(c) + 1e-8) for c in cents])
            j = int(sims.argmax())
            if sims[j] >= threshold:
                members[j].append(k); cents[j] = cents[j] + e
                continue
        cents.append(e.copy()); members.append([k])
    merged = True
    while merged:
        merged = False
        for a in range(len(cents)):
            for b in range(a + 1, len(cents)):
                ca, cb = cents[a] / np.linalg.norm(cents[a]), cents[b] / np.linalg.norm(cents[b])
                if ca @ cb >= threshold:
                    cents[a] = cents[a] + cents[b]; members[a] += members[b]
                    del cents[b]; del members[b]; merged = True
                    break
            if merged:
                break
    total = max(1, len(frames))
    groups = [m for m in members if len({dets[k][0] for k in m}) / total >= min_share]
    groups.sort(key=lambda m: -len({dets[k][0] for k in m}))
    people, owner = [], {}
    for pid, m in enumerate(groups):
        best = max(m, key=lambda k: dets[k][2])
        crop = cv2.resize(dets[best][3], (96, 96), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        seen = sorted({dets[k][0] for k in m})
        people.append({"id": pid, "thumb": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
                       "count": len(seen), "share": round(len(seen) / total, 4),
                       "first": int(frames_idx[seen[0]]), "last": int(frames_idx[seen[-1]])})
        for k in m:
            owner[k] = pid
    samples = [{"f": int(f), "faces": []} for f in frames_idx]
    for k, (si, _, area, _) in enumerate(dets):
        if k in owner:
            samples[si]["faces"].append({"pid": owner[k], "area": round(float(area), 5)})
    out = {"people": people, "samples": samples, "step": step}
    _CAST_CACHE[key] = out
    return out


def shot_people(cast: dict, start: int, end: int) -> dict:
    """Who appears in [start, end): per person, sampled frames seen and summed face area; plus the main one."""
    score: dict[int, list[float]] = {}
    for smp in cast["samples"]:
        if start <= smp["f"] < end:
            for fc in smp["faces"]:
                v = score.setdefault(fc["pid"], [0, 0.0]); v[0] += 1; v[1] += fc["area"]
    main = max(score, key=lambda p: (score[p][0], score[p][1])) if score else -1
    return {"people": sorted(score), "main": main}


def person_change_points(cast: dict, start: int, end: int, min_run: int) -> list[int]:
    """Frames inside [start, end) where the main (largest) face switches to another person for at least min_run."""
    runs = []
    for smp in cast["samples"]:
        if start <= smp["f"] < end:
            main = max(smp["faces"], key=lambda fc: fc["area"])["pid"] if smp["faces"] else -1
            if main >= 0 and (not runs or runs[-1][1] != main):
                runs.append([smp["f"], main])
    pts = []
    for (f0, p0), (f1, p1) in zip(runs, runs[1:]):
        if f1 - (pts[-1] if pts else start) >= min_run and end - f1 >= min_run:
            pts.append(f1)
    return pts


# ---------------------------------------------------------------------------- shot masks (SAM 3) and crop / uncrop

SAM3_FILE = "sam3.1_multiplex_fp16.safetensors"
SAM3_URL = "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"
_SAM3: dict[str, Any] = {}
_MASK_CACHE: dict[tuple, dict] = {}
# global mask settings (the planner's Mask card); a shot only says what to segment (text / points / key frame)
DEFAULT_MASK = {"text": "", "points": [], "key": 0, "keys": [], "threshold": 0.5, "max_objects": 4, "invert": False,
                "fill_holes": True, "temporal_expand": 2, "blockify": 0, "padding": 0.15, "expand": 16,
                "feather": 12, "paste": "box"}
SHOT_MASK_KEYS = ("text", "points", "key", "keys")


def point_keys(spec: dict) -> list[tuple[int, list]]:
    """The shot's key frames with their points, in frame order: spec['keys'] ([{key, points}], 1.116), else the single
    key frame of older plans (points + key)."""
    keys = {int(k.get("key") or 0): list(k.get("points") or []) for k in spec.get("keys") or [] if k.get("points")}
    if not keys and spec.get("points"):
        keys = {int(spec.get("key") or 0): list(spec["points"])}
    return sorted(keys.items())


def _sam3_path() -> str:
    """The official SAM 3.1 checkpoint in models/checkpoints, downloaded once when missing."""
    p = folder_paths.get_full_path("checkpoints", SAM3_FILE)
    if p:
        return p
    import urllib.request
    dst = os.path.join(folder_paths.get_folder_paths("checkpoints")[0], SAM3_FILE)
    tmp = dst + ".part"
    print(f"[BFS Shot Planner] downloading {SAM3_FILE} (1.7 GB) from Comfy-Org/sam3.1 ...")
    with urllib.request.urlopen(SAM3_URL) as r, open(tmp, "wb") as f:
        total, done, step = int(r.headers.get("Content-Length") or 0), 0, 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk); done += len(chunk)
            if total and done * 10 // total > step:
                step = done * 10 // total
                print(f"[BFS Shot Planner] {SAM3_FILE}: {step * 10}%")
    os.replace(tmp, dst)
    return dst


def _sam3():
    if "model" not in _SAM3:
        import comfy.sd
        out = comfy.sd.load_checkpoint_guess_config(_sam3_path(), output_vae=False, output_clip=True,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))
        _SAM3["model"], _SAM3["clip"] = out[0], out[1]
    return _SAM3["model"], _SAM3["clip"]


def mask_spec(m: dict | None, cfg: dict | None = None) -> dict:
    """Global settings (cfg) with the shot's own target (text / points / key frame) on top."""
    spec = dict(DEFAULT_MASK)
    spec.update({k: v for k, v in (cfg or {}).items() if v is not None and k not in SHOT_MASK_KEYS})
    spec.update({k: v for k, v in (m or {}).items() if v is not None and k in SHOT_MASK_KEYS})
    return spec


# external masks (e.g. rotoscoped in another tool) replace SAM 3. Most specific first: the shot's own mask video
# (covers only that shot), the shot's own SAM 3 text / points, the planner's `mask` input (the whole video), the
# panel's mask video (the whole video).
_EXT_MASK: dict[str, torch.Tensor] = {}     # video path -> the planner's `mask` input from its last run


def plan_spec(p: dict, m: dict | None, path: str = "") -> dict:
    """mask_spec plus the external mask this shot uses, if any (spec['ext'])."""
    spec = mask_spec(m, p.get("mask_cfg"))
    if (m or {}).get("video"):
        spec["ext"] = {"file": m["video"], "scope": "shot"}
    elif (m or {}).get("points") or str((m or {}).get("text") or "").strip():
        pass                                                    # the shot's own SAM 3 selection
    elif path and (p.get("track") or {}).get("points"):         # one selection tracked through the whole video
        fps = float(p.get("fps") or 24)
        spec["ext"] = {"track": {**p["track"], "start_s": p.get("start_s", 0), "end_s": p.get("end_s", 0),
                                 "start_f": p.get("start_f"), "end_f": p.get("end_f"), "max_total_s": p.get("max_total_s", 0),
                                 "fps": fps}, "path": path}
    elif path and path in _EXT_MASK:
        spec["ext"] = {"tensor": path}
    elif p.get("mask_video"):
        spec["ext"] = {"file": p["mask_video"], "scope": "video"}
    return spec


def spec_has_mask(spec: dict) -> bool:
    return bool(spec.get("ext") or point_keys(spec) or str(spec.get("text") or "").strip())


def ext_masks(ext: dict, analysis: dict, start: int, length: int, w: int, h: int) -> torch.Tensor:
    """[length,h,w] uint8 from an external mask (white = the subject) for timeline frames start..start+length."""
    idx = np.clip(np.arange(start, start + length), 0, analysis["n"] - 1)
    if "track" in ext:
        tm, lo = track_video(ext["path"], analysis, ext["track"])
        sel = torch.from_numpy(np.clip(idx - lo, 0, tm.shape[0] - 1).astype(np.int64))
        m = torch.nn.functional.interpolate(tm[sel][:, None].float(), size=(h, w), mode="nearest")[:, 0]
        return (m > 0.5).to(torch.uint8)
    if "tensor" in ext:
        t = _EXT_MASK[ext["tensor"]].float()
        if t.ndim == 4:                                    # an IMAGE batch (e.g. Load Video of a black/white video)
            t = t[..., :3].mean(-1)
        if t.ndim == 2:
            t = t[None]
        src = _timeline(analysis["n_src"], analysis["fps_src"], float(analysis["fps"]))[idx]
        n = t.shape[0]                                     # one mask per source frame; otherwise spread over the video
        ii = src if n == analysis["n_src"] else np.round(src * (n - 1) / max(1, analysis["n_src"] - 1))
        m = t[torch.from_numpy(np.clip(ii, 0, n - 1).astype(np.int64))]
        m = torch.nn.functional.interpolate(m[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    else:
        fpath = _input_path(ext["file"])
        pr = _probe(fpath)
        t = (idx - start) / float(analysis["fps"]) if ext.get("scope") == "shot" else idx / float(analysis["fps"])
        mi = np.clip(np.round(t * pr["fps_src"]), 0, pr["n_src"] - 1).astype(np.int64)
        fr = _read_frames(fpath, mi, (w, h), stage="Reading the mask video")
        m = torch.from_numpy(np.stack(fr).astype(np.float32).mean(-1) / 255.0)
    return (m > 0.5).to(torch.uint8)


def _ext_name(ext: dict) -> str:
    if "track" in ext:
        t = ext["track"]
        return f"the whole-video track ({len(t.get('points') or [])} points at frame {t.get('frame')})"
    if "tensor" in ext:
        t = _EXT_MASK.get(ext["tensor"])
        return f"the planner's mask input ({tuple(t.shape) if t is not None else 'gone'})"
    return f"mask video '{ext['file']}'" + (" (this shot)" if ext.get("scope") == "shot" else " (whole video)")


def shape_mask(masks: torch.Tensor, spec: dict) -> torch.Tensor:
    """invert -> fill holes -> temporal expand -> blockify, on [N,H,W] 0/1 masks."""
    x = masks.float()
    if spec.get("invert"):
        x = 1 - x
    if spec.get("fill_holes"):
        try:
            from scipy.ndimage import binary_fill_holes
            x = torch.from_numpy(np.stack([binary_fill_holes(f > 0.5) for f in x.numpy()]).astype(np.float32))
        except ImportError:
            pass
    t = int(spec.get("temporal_expand") or 0)
    if t > 0 and x.shape[0] > 1:
        N, H, W = x.shape
        x = torch.nn.functional.max_pool1d(x.permute(1, 2, 0).reshape(-1, 1, N), 2 * t + 1, 1, t) \
            .reshape(H, W, N).permute(2, 0, 1)
    b = int(spec.get("blockify") or 0)
    if b > 1:
        H, W = x.shape[1:]
        cov = torch.nn.functional.avg_pool2d(x[:, None], b, b, ceil_mode=True)[:, 0]
        x = (cov >= 0.5).float().repeat_interleave(b, 1).repeat_interleave(b, 2)[:, :H, :W]
    return (x > 0.5).to(torch.uint8)


_TRACK_CACHE: dict[tuple, tuple] = {}
TRACK_CHUNK = 150   # frames tracked at once; each chunk starts from the previous one's last mask


def track_video(path: str, analysis: dict, t: dict) -> tuple[torch.Tensor, int]:
    """One selection (points on one frame, optional text) tracked by SAM 3 through the plan's whole range, forward and
    backward from that frame, in chunks that hand their last mask to the next (the text, when given, finds the subject
    again after a camera cut). Returns (masks [hi-lo, h, w] uint8 at a 640 px working size, lo)."""
    fps = float(analysis["fps"])
    lo, hi = plan_range(analysis["n"], fps, t.get("start_s", 0), t.get("end_s", 0), t.get("max_total_s", 0),
                        t.get("start_f"), t.get("end_f"))
    key = (path, os.path.getmtime(path), fps, lo, hi, json.dumps({k: t.get(k) for k in ("points", "frame", "text", "threshold")},
                                                                 sort_keys=True))
    if key in _TRACK_CACHE:
        return _TRACK_CACHE[key]
    from comfy_extras.nodes_sam3 import SAM3_Detect
    model, clip = _sam3()
    sw, sh = analysis["width"], analysis["height"]
    sc = 640 / max(sw, sh)
    w, h = max(32, int(sw * sc) // 2 * 2), max(32, int(sh * sc) // 2 * 2)
    src = _timeline(analysis["n_src"], analysis["fps_src"], fps)
    k = max(lo, min(hi - 1, int(t.get("frame") or lo)))
    text = str(t.get("text") or "").strip()
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(text)) if text else None
    spec = dict(DEFAULT_MASK, threshold=float(t.get("threshold") or DEFAULT_MASK["threshold"]), max_objects=1)
    out = np.zeros((hi - lo, h, w), np.uint8)
    t0 = time.time()
    print(f"[BFS Shot] SAM 3: tracking {len(t.get('points') or [])} points from frame {k} through frames {lo}-{hi - 1}"
          + (f" (re-found by '{text}')" if text else "") + "…", flush=True)
    allf = _read_frames(path, src[np.arange(lo, hi)], (w, h), stage="Reading the video for the track")   # read once
    key_img = torch.from_numpy(allf[k - lo][None].astype(np.float32) / 255.0)
    pts = t.get("points") or []
    pos = [{"x": q["x"] * w, "y": q["y"] * h} for q in pts if q.get("label", 1)]
    neg = [{"x": q["x"] * w, "y": q["y"] * h} for q in pts if not q.get("label", 1)]
    first = SAM3_Detect.execute(model=model, image=key_img, positive_coords=json.dumps(pos), negative_coords=json.dumps(neg),
                                threshold=float(spec["threshold"]), refine_iterations=2).args[0][:1].float()
    for direction in (1, -1):                       # forward from the key frame, then backward
        frames = list(range(k, hi)) if direction == 1 else list(range(k, lo - 1, -1))
        init, done = first, 0
        while done < len(frames):
            part = frames[max(0, done - 1): done + TRACK_CHUNK]       # one frame of overlap carries the mask on
            seed = init if init is not None and bool(init.any()) else None
            if seed is None and cond is None:
                break                                   # the subject is lost and there is no text to find it again
            imgs = torch.from_numpy(np.stack([allf[f - lo] for f in part]).astype(np.float32) / 255.0)
            mk = (_track(model, imgs, seed, cond, spec) > 0.5).numpy().astype(np.uint8)
            for j, f in enumerate(part):
                out[f - lo] |= mk[j]
            init = torch.from_numpy(mk[-1:]).float()
            done += TRACK_CHUNK
            _status(f"Tracking with SAM 3: {min(done, len(frames))}/{len(frames)} frames", min(done, len(frames)), len(frames))
            _node_boundary()
    res = (torch.from_numpy(out), lo)
    _TRACK_CACHE[key] = res
    print(f"[BFS Shot] SAM 3: track done in {time.time() - t0:.0f}s, {mask_report(res[0])}", flush=True)
    return res


def _track(model, imgs: torch.Tensor, init: torch.Tensor | None, cond, spec: dict) -> torch.Tensor:
    from comfy_extras.nodes_sam3 import SAM3_TrackToMask, SAM3_VideoTrack
    data = SAM3_VideoTrack.execute(images=imgs, model=model, initial_mask=init, conditioning=cond,
                                   detection_threshold=float(spec["threshold"]),
                                   max_objects=int(spec["max_objects"]), detect_interval=1).args[0]
    return SAM3_TrackToMask.execute(track_data=data, object_indices="").args[0].float().cpu()


def segment_frames(imgs: torch.Tensor, spec: dict) -> torch.Tensor:
    """[N,H,W,3] -> [N,H,W] masks with SAM 3, from the text or from the points of the key frames: each key frame's
    points select the subject there and it is tracked from that frame up to the next key frame (before the first one,
    backward from it), so a later key frame corrects the mask from its frame on."""
    from comfy_extras.nodes_sam3 import SAM3_Detect
    model, clip = _sam3()
    text = str(spec.get("text") or "").strip()
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(text)) if text else None
    N, H, W = imgs.shape[:3]
    keys = [(max(0, min(N - 1, k)), pts) for k, pts in point_keys(spec)]
    if not keys:
        if cond is None:
            raise ValueError("the shot's mask needs a text prompt or points")
        return _track(model, imgs, None, cond, spec)

    def detect(k, pts):
        pos = [{"x": p["x"] * W, "y": p["y"] * H} for p in pts if p.get("label", 1)]
        neg = [{"x": p["x"] * W, "y": p["y"] * H} for p in pts if not p.get("label", 1)]
        return SAM3_Detect.execute(model=model, image=imgs[k:k + 1], positive_coords=json.dumps(pos),
                                   negative_coords=json.dumps(neg), threshold=float(spec["threshold"]),
                                   refine_iterations=2).args[0][:1].float()
    out = torch.zeros(N, H, W)
    for i, (k, pts) in enumerate(keys):
        seed = detect(k, pts)
        if i == 0 and k > 0:                                                  # first key frame -> start
            out[:k + 1] = _track(model, imgs[:k + 1].flip(0), seed, None, spec).flip(0)
        end = keys[i + 1][0] if i + 1 < len(keys) else N                      # this key frame -> the next one
        out[k:end] = _track(model, imgs[k:end], seed, None, spec)
    return out


def _mask_key(path: str, analysis: dict, start: int, length: int, spec: dict) -> tuple:
    ext = spec.get("ext")
    if ext and "track" in ext:
        e = ("track", json.dumps(ext["track"], sort_keys=True))
    elif ext and "tensor" in ext:
        t = _EXT_MASK.get(ext["tensor"])
        e = ("tensor", id(t), tuple(t.shape) if t is not None else None)
    elif ext:
        f = folder_paths.get_annotated_filepath(ext["file"])
        e = ("file", ext["file"], ext.get("scope"), os.path.getmtime(f) if f and os.path.isfile(f) else 0)
    else:
        e = None
    return (path, os.path.getmtime(path), float(analysis["fps"]), int(start), int(length),
            json.dumps({"keys": point_keys(spec), **{k: spec[k] for k in ("text", "threshold", "max_objects")}}, sort_keys=True), e)


def shot_mask(path: str, analysis: dict, start: int, length: int, spec: dict) -> dict:
    """Masks of one shot (timeline frames start..start+length) at a working size, cached; plus the crop box."""
    key = _mask_key(path, analysis, start, length, spec)
    if key not in _MASK_CACHE:
        src = _timeline(analysis["n_src"], analysis["fps_src"], float(analysis["fps"]))
        idx = np.clip(np.arange(start, start + length), 0, analysis["n"] - 1)
        sw, sh = analysis["width"], analysis["height"]
        s = 640 / max(sw, sh)
        w, h = max(32, int(sw * s) // 2 * 2), max(32, int(sh * s) // 2 * 2)
        if spec.get("ext"):
            t0 = time.time()
            _MASK_CACHE[key] = {"masks": ext_masks(spec["ext"], analysis, start, length, w, h), "size": (w, h)}
            print(f"[BFS Shot] {_ext_name(spec['ext'])} instead of SAM 3, timeline {start}-{start + length - 1}: "
                  f"{mask_report(_MASK_CACHE[key]['masks'])} ({time.time() - t0:.1f}s)", flush=True)
            out = dict(_MASK_CACHE[key])
            out["masks"] = shape_mask(out["masks"], spec)
            out["box"] = crop_box(out["masks"], float(spec["padding"]))
            return out
        frames = _read_frames(path, src[idx], (w, h), stage="Reading the shot")
        imgs = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
        _status(f"Segmenting {len(frames)} frames with SAM 3", force=True)
        keys = point_keys(spec)
        what = (", ".join(f"{len(p)} points on frame {k}" for k, p in keys)) if keys else f"'{spec.get('text', '')}'"
        print(f"[BFS Shot] SAM 3: segmenting {what} on {len(frames)} frames ({w}x{h}, timeline {start}-{start + length - 1})…",
              flush=True)
        t0 = time.time()
        _MASK_CACHE[key] = {"masks": (segment_frames(imgs, spec) > 0.5).to(torch.uint8), "size": (w, h)}
        print(f"[BFS Shot] SAM 3: done in {time.time() - t0:.1f}s, {mask_report(_MASK_CACHE[key]['masks'])}", flush=True)
        _node_boundary()
    out = dict(_MASK_CACHE[key])
    out["masks"] = shape_mask(out["masks"], spec)
    out["box"] = crop_box(out["masks"], float(spec["padding"]))
    return out


def crop_box(masks: torch.Tensor, padding: float) -> list[float] | None:
    """One box for the whole shot (union of every frame's mask) plus padding, normalised [x0, y0, x1, y1]."""
    if masks.numel() == 0 or not bool(masks.any()):
        return None
    union = masks.amax(0)
    ys, xs = torch.nonzero(union, as_tuple=True)
    H, W = union.shape
    x0, x1, y0, y1 = xs.min().item() / W, (xs.max().item() + 1) / W, ys.min().item() / H, (ys.max().item() + 1) / H
    px, py = (x1 - x0) * padding, (y1 - y0) * padding
    return [max(0.0, x0 - px), max(0.0, y0 - py), min(1.0, x1 + px), min(1.0, y1 + py)]


def grow_blur(m: torch.Tensor, grow: int, blur: int) -> torch.Tensor:
    """[N,H,W] in 0-1: dilate by `grow` px, then soften the edge by `blur` px."""
    x = m.float()[:, None]
    if grow > 0:   # separable: a square max = max over rows then over columns (much faster for big kernels)
        k = 2 * grow + 1
        x = torch.nn.functional.max_pool2d(x, (1, k), 1, (0, grow))
        x = torch.nn.functional.max_pool2d(x, (k, 1), 1, (grow, 0))
    for _ in range(2 if blur > 0 else 0):   # box blur, separable too
        k = 2 * blur + 1
        x = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(x, (blur, blur, 0, 0), mode="replicate"), (1, k), 1)
        x = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(x, (0, 0, blur, blur), mode="replicate"), (k, 1), 1)
    return x[:, 0].clamp(0, 1)


def box_px(box: list[float], W: int, H: int, multiple: int = 2) -> tuple[int, int, int, int]:
    x0, y0 = int(box[0] * W), int(box[1] * H)
    x1, y1 = max(x0 + multiple, int(round(box[2] * W))), max(y0 + multiple, int(round(box[3] * H)))
    return x0, y0, min(W, x1), min(H, y1)


def crop_panel(img: torch.Tensor, shot: dict) -> torch.Tensor:
    """Cut a duet panel off a decoded canvas (the shot was conditioned with a duet panel)."""
    info = shot.get("panel")
    if not info or img.shape[1:3] == (shot.get("height"), shot.get("width")):
        return img
    h, w = info["h"] * 16, info["w"] * 16
    y0 = info["strip_h"] * 16 if info["position"] == "top" else 0
    x0 = info["strip_w"] * 16 if info["position"] == "left" else 0
    if img.shape[1] < y0 + h or img.shape[2] < x0 + w:
        return img
    return img[:, y0:y0 + h, x0:x0 + w]


# --- stitch finishing: edge hardness, Reinhard colour match (LAB) and Poisson seamless clone. The three operations are
# adapted from Neko (Nekodificador)'s ComfyUI-NKD-Basic-Tools (NKD Inpaint Stitch, MIT licence), built on AbleJones's
# workflow and nodes. Differences: the colour
# statistics are taken over the whole shot (no flicker), and by default from a ring AROUND the subject, so a swap's new
# person keeps its own colours while the model's light / colour drift is corrected.

def _alpha_hardness(alpha: torch.Tensor, hardness: float) -> torch.Tensor:
    """Black / white point remap of the paste alpha: 0 = unchanged, 1 = hard cut at 0.5 (removes the halo of the
    original subject in the soft edge)."""
    if hardness <= 0.0:
        return alpha
    bp = min(hardness * 0.5, 0.499)
    return ((alpha - bp) / (1.0 - 2 * bp)).clamp(0.0, 1.0)


def _rgb_to_lab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]], np.float32)
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    d = 6.0 / 29.0
    f = lambda t: np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)
    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], -1).astype(np.float32)


def _lab_to_rgb(lab):
    fy = (lab[..., 0] + 16.0) / 116.0
    fx, fz = lab[..., 1] / 500.0 + fy, fy - lab[..., 2] / 200.0
    d = 6.0 / 29.0
    fi = lambda t: np.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))
    xyz = np.stack([fi(fx) * 0.95047, fi(fy), fi(fz) * 1.08883], -1)
    Mi = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]], np.float32)
    lin = np.clip(xyz @ Mi.T, 0.0, None)
    return np.clip(np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055), 0, 1).astype(np.float32)


def _shot_color_transfer(orig: torch.Tensor, gen: torch.Tensor, region: torch.Tensor, strength: float) -> torch.Tensor:
    """Reinhard transfer in LAB with statistics over the whole shot inside `region` [N,H,W] (original vs generated),
    applied to every generated frame. Returns gen unchanged when the region is too small."""
    if strength <= 0.0:
        return gen
    sel = region > 0.5
    if int(sel.sum()) < 400:
        return gen
    step = max(1, int(sel.sum()) // 200000)
    o = _rgb_to_lab(orig[sel].numpy()[::step].astype(np.float32))
    g = _rgb_to_lab(gen[sel].numpy()[::step].astype(np.float32))
    om, os_, gm, gs = o.mean(0), o.std(0) + 1e-5, g.mean(0), g.std(0) + 1e-5
    ratio = np.clip(os_ / gs, 0.5, 2.0)          # a flat region has ~0 spread: never blow the contrast up
    lab = _rgb_to_lab(gen.numpy().astype(np.float32))
    out = _lab_to_rgb((lab - gm) * ratio + om)
    return torch.from_numpy(out) * strength + gen * (1.0 - strength)


def _seamless(orig: np.ndarray, gen: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Poisson clone of gen onto orig inside alpha (one frame, float RGB 0-1)."""
    import cv2
    binary = (alpha > 0.1).astype(np.uint8) * 255
    binary[0, :] = binary[-1, :] = 0
    binary[:, 0] = binary[:, -1] = 0
    x, y, w, h = cv2.boundingRect(binary)
    a3 = alpha[..., None]
    if w == 0 or h == 0:
        return orig * (1 - a3) + gen * a3
    try:
        c = cv2.seamlessClone((np.clip(gen, 0, 1) * 255).astype(np.uint8), (np.clip(orig, 0, 1) * 255).astype(np.uint8),
                              binary, (x + w // 2, y + h // 2), cv2.NORMAL_CLONE).astype(np.float32) / 255.0
        return orig * (1 - a3) + c * a3
    except Exception:  # noqa: BLE001
        return orig * (1 - a3) + gen * a3


MATCH_REGIONS = ["around the subject (swap)", "inside the subject (same content)"]


def paste_back(result: torch.Tensor, shot: dict, new_outline: bool = True, **finish) -> torch.Tensor:
    """'Frame + paste' shots: the whole frame was generated (so the model followed the guide freely); paste only the
    subject onto the original frame. The pasted area is the shot's mask, plus (new_outline) the new subject found by
    SAM 3 on the result near the original one (the shot's 'new subject' text, default 'person'), so a bigger
    silhouette is not cut."""
    base = shot["frames"]
    H, W = base.shape[1:3]
    ms = shot["mask_src"]
    tag = f"[BFS Shot Join] shot {shot.get('index', 0) + 1}"
    sm = shot_mask(ms["path"], ms["analysis"], ms["start"], ms["length"], ms["spec"])["masks"]
    n = min(result.shape[0], base.shape[0], sm.shape[0])
    m = torch.nn.functional.interpolate(sm[:n, None].float(), size=(H, W), mode="nearest")[:, 0]
    for sm2, _ in extra_mark_masks(shot, tag) if shot.get("color_guide") else []:   # every marked person is pasted
        sm2 = torch.nn.functional.interpolate(sm2[:n, None].float(), size=(H, W), mode="nearest")[:, 0]
        m = torch.maximum(m, sm2)
    if new_outline and bool(m.any()):
        x0, y0, x1, y1 = box_px(crop_box(m, 0.5), W, H)              # where the new person may be
        s = 640 / max(H, W)
        small = torch.nn.functional.interpolate(result[:n, ..., :3].movedim(-1, 1).float(), size=(max(32, int(H * s) // 2 * 2),
                    max(32, int(W * s) // 2 * 2)), mode="bilinear", align_corners=False).movedim(1, -1).clamp(0, 1)
        t0 = time.time()
        what = str(shot.get("paste_text") or "").strip() or "person"
        new = segment_frames(small, dict(DEFAULT_MASK, text=what, max_objects=8))
        _node_boundary()
        new = torch.nn.functional.interpolate(new[:, None].float(), size=(H, W), mode="nearest")[:, 0]
        near = torch.zeros(H, W); near[y0:y1, x0:x1] = 1
        m = torch.maximum(m, new * near)
        print(f"{tag}: Frame + paste, new outline ('{what}') by SAM 3 in {time.time() - t0:.1f}s", flush=True)
    sp = ms["spec"]
    crop = {"box": [0.0, 0.0, 1.0, 1.0], "mask": m, "paste": "mask",
            "expand": int(sp["expand"] if sp.get("expand") is not None else 16),
            "feather": int(sp["feather"] if sp.get("feather") is not None else 12)}
    print(f"{tag}: pasting the person onto the original frame: {mask_report(m)}", flush=True)
    return uncrop(result, {"crop": crop, "full_frames": base[:n]}, **finish)


def uncrop(result: torch.Tensor, shot: dict, edge_hardness: float = 0.0, match_colors: float = 0.0,
           match_region: str = MATCH_REGIONS[0], seamless: bool = False) -> torch.Tensor:
    """Paste a cropped shot's result back into its full frames (feathered by the mask or the box), with optional
    edge hardness, colour match and seamless clone."""
    c = shot["crop"]
    full = shot["full_frames"]
    W, H = full.shape[2], full.shape[1]
    x0, y0, x1, y1 = box_px(c["box"], W, H)
    n = min(result.shape[0], full.shape[0])
    res = torch.nn.functional.interpolate(result[:n, ..., :3].movedim(-1, 1).float(), size=(y1 - y0, x1 - x0),
                                          mode="bilinear", align_corners=False).movedim(1, -1).clamp(0, 1)
    out = full[:n].clone()
    if c.get("paste") == "box":
        a = torch.zeros(n, H, W)
        a[:, y0:y1, x0:x1] = 1
        alpha = grow_blur(a, 0, int(c.get("feather", 12)))
    else:
        m = c["mask"][:n].float()
        alpha = grow_blur(m, int(c.get("expand", 8)), int(c.get("feather", 12)))
    alpha = _alpha_hardness(alpha, float(edge_hardness))
    if match_colors > 0 and c.get("mask") is not None:
        m = c["mask"][:n].float()[:, y0:y1, x0:x1]
        if match_region == MATCH_REGIONS[0]:     # ring around the subject: background the model also regenerated
            k = max(3, int(0.06 * max(y1 - y0, x1 - x0)))
            region = grow_blur(m, k, 0) - grow_blur(m, max(2, k // 3), 0)
        else:
            region = m
        res = _shot_color_transfer(full[:n, y0:y1, x0:x1].float(), res, region, float(match_colors))
    a = alpha[:, y0:y1, x0:x1, None]
    if seamless:
        for i in range(n):
            out[i, y0:y1, x0:x1] = torch.from_numpy(_seamless(out[i, y0:y1, x0:x1].numpy().astype(np.float32),
                                                              res[i].numpy().astype(np.float32), a[i, ..., 0].numpy()))
    else:
        out[:, y0:y1, x0:x1] = res * a + out[:, y0:y1, x0:x1] * (1 - a)
    if result.shape[0] > n:   # frames past the planned length stay as generated, pasted the same way
        out = torch.cat([out, out[-1:].expand(result.shape[0] - n, -1, -1, -1)], 0)
    return out


def mask_preview(path: str, analysis: dict, start: int, length: int, spec: dict, count: int = 6,
                 color: str = "#ff283c", opacity: float = 0.55) -> dict:
    """`count` frames of the shot (0 = every frame, smaller, for the points window's player) with the selected region
    tinted and the crop box, as data URLs."""
    import cv2
    r = shot_mask(path, analysis, start, length, spec)
    masks, (w, h), box = r["masks"], r["size"], r["box"]
    src = _timeline(analysis["n_src"], analysis["fps_src"], float(analysis["fps"]))
    picks = list(range(length)) if count <= 0 else sorted(set(int(round(i)) for i in np.linspace(0, length - 1, min(count, length))))
    frames = _read_frames(path, src[np.clip(np.array(picks) + start, 0, analysis["n"] - 1)], (w, h))
    small = 384 / max(w, h) if count <= 0 and max(w, h) > 384 else 1.0   # every frame: ~10 KB each
    try:
        tint = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
        alpha = min(0.8, max(0.05, float(opacity)))
    except (TypeError, ValueError, IndexError):
        tint, alpha = np.array([255, 40, 60], dtype=np.float32), 0.55
    out = []
    for i, fr in zip(picks, frames):
        img = fr.copy()
        m = masks[min(i, masks.shape[0] - 1)].numpy().astype(bool)
        img[m] = (img[m].astype(np.float32) * (1 - alpha) + tint * alpha).astype(np.uint8)
        if box:
            x0, y0, x1, y1 = box_px(box, w, h)
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (255, 210, 90), 2)
        if small < 1.0:
            img = cv2.resize(img, (int(w * small) // 2 * 2, int(h * small) // 2 * 2), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80 if small == 1.0 else 70])
        out.append({"f": start + i, "src": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()})
    cover = float(masks.float().mean()) if masks.numel() else 0.0
    return {"box": box, "frames": out, "coverage": cover, "empty": box is None}


# ---------------------------------------------------------------------------- {target}: who is replaced

TARGET_Q = ("Describe the main subject in this picture (a person, an animal, a creature, a robot or a cartoon character) as "
            "one short English noun phrase for a video prompt, at most 14 words: what it is first, then for a person the "
            "apparent gender and age group, hair and main clothing with colours, for anything else its kind and most "
            "visible colours or features. Examples: 'the young woman with long black hair in a white T-shirt and denim "
            "shorts', 'the brown dog with a red collar', 'the green goblin in a torn leather vest'. Start with 'the'. "
            "Reply with the phrase only.")


def target_crop(path: str, analysis: dict, start: int, length: int, spec: dict, pad: float = 0.12):
    """The selected subject (the shot's SAM 3 points / text, on its key frame) cropped from that frame, RGB uint8."""
    src = _timeline(analysis["n_src"], analysis["fps_src"], float(analysis["fps"]))
    keys = point_keys(spec)
    key = max(0, min(length - 1, keys[0][0] if keys else int(spec.get("key") or length // 2)))
    fr = _read_frames(path, src[np.clip(np.array([start + key]), 0, analysis["n"] - 1)], None)[0]
    if spec.get("ext"):
        sm = shot_mask(path, analysis, start, length, spec)["masks"]
        m = torch.nn.functional.interpolate(sm[min(key, sm.shape[0] - 1)][None, None].float(), size=fr.shape[:2],
                                            mode="nearest")[0, 0] > 0.5
    else:
        img = torch.from_numpy(fr.astype(np.float32) / 255.0)[None]
        m = segment_frames(img, dict(spec, key=0))[0] > 0.5
        _node_boundary()
    if not bool(m.any()):
        return None
    ys, xs = np.where(m.numpy())
    H, W = fr.shape[:2]
    py, px = int((ys.max() - ys.min()) * pad) + 4, int((xs.max() - xs.min()) * pad) + 4
    return fr[max(0, ys.min() - py):min(H, ys.max() + py), max(0, xs.min() - px):min(W, xs.max() + px)]


def prompt_slot(prompt: str, marker: str, limit: int = 360) -> str:
    """The sentence of `prompt` that holds `marker`, with the marker as a blank ('___') and the other placeholders
    made readable: what the VLM fills in, so the description fits the prompt (and not the other way round)."""
    text = str(prompt or "")
    if marker not in text:
        return ""
    i = text.index(marker)
    starts = [text.rfind(sep, 0, i) for sep in (". ", "; ", ":  ", "\n")]
    a = max(starts)
    a = 0 if a < 0 else a + 2
    ends = [x for x in (text.find(sep, i + len(marker)) for sep in (". ", "; ", "\n")) if x >= 0]
    b = min(ends) + 1 if ends else len(text)
    snippet = text[a:b].strip()
    snippet = snippet.replace(marker, "___", 1)
    for other, plain in (("{target}", "the guide subject"), ("{details2}", "the second character"),
                         ("{details}", "the character"), ("{shot}", ""), ("{setting}", "")):
        snippet = snippet.replace(other, plain)
    snippet = PICTURE_DETAILS.sub(lambda m: f"the subject of <Picture {m.group(1)}>", snippet)
    return snippet[:limit]


# {details@Picture 3} (or {details@Picture3}): the description of the shot's third picture alone
PICTURE_DETAILS = re.compile(r"\{details@Picture\s*(\d+)\}")


def picture_marker(prompt: str, k: int | None = None) -> str:
    """The prompt's {details@Picture k} marker as written (the first one of any picture with k=None), or ''."""
    for m in PICTURE_DETAILS.finditer(str(prompt or "")):
        if k is None or int(m.group(1)) == k:
            return m.group(0)
    return ""


def expand_two_ref_prompt(prompt: str) -> str:
    """Add a second subject slot when a shot has a second reference and its template has only one (not when the
    prompt already says what <Picture 2> is, or names its pictures with {details@Picture n})."""
    text = str(prompt or "")
    if "{details2}" in text or "<Subject 2>" in text or "<Picture 2>" in text or "{details@" in text:
        return text
    second = "<Subject 2> is the {details2} shown in <Picture 2>."
    if re.search(r"(?im)^\s*subject_definitions\s*:", text):
        if re.search(r"(?im)^\s*summary\s*:", text):
            return re.sub(r"(?im)(^\s*summary\s*:)", lambda m: second + "\n" + m.group(1), text, count=1)
        return text.rstrip() + "\n" + second
    return "subject_definitions: " + second + "\n\n" + text


def fit_note(slot: str) -> str:
    return (f" Your answer goes into the blank (___) of this part of the video prompt: \"{slot}\". Reply with ONLY the "
            "words that go in the blank, so that the sentence reads naturally with them in place: do not repeat the "
            "sentence, do not write '___', no full sentence.") if slot else ""


_ANSWER_LABEL = re.compile(r"(?:\*\*|__|#+\s*)?\s*(?:final\s+|short\s+)?(?:answer|description|phrase|noun\s+phrase|output|result|response|caption)"
                           r"\s*(?:\*\*|__)?\s*[:\-\u2013]\s*(?:\*\*|__)?\s*", re.I)


def strip_answer_label(text: str) -> str:
    """'**Answer:** elderly woman with ...' -> 'elderly woman with ...'. The VLM sometimes labels its reply (or reasons
    first and then writes 'Answer: ...'): keep what follows the LAST label, without markdown marks around it."""
    t = str(text or "").strip()
    last = None
    for m in _ANSWER_LABEL.finditer(t):
        if m.start() == 0 or t[m.start() - 1] in " \n\t.*_#>":
            last = m
    if last is not None and t[last.end():].strip():
        t = t[last.end():]
    return t.strip().strip("*_`#").strip().strip('"\'').strip()


def tidy_slot(text: str, slot: str, max_words: int = 25) -> str:
    """Only the words for the blank: cut an echo of the sentence around it (the VLM sometimes rewrites the whole
    sentence), drop a leading article when the prompt already has one before the blank, keep it short."""
    t = strip_answer_label(" ".join(str(text or "").strip().split("\n")[0].split()).strip().strip('"\''))
    if "___" in slot:
        head, tail = slot.split("___", 1)
        tail_key = " ".join(tail.split()[:2])                 # e.g. "shown in"
        if "___" in t:                                         # "humanoid robot is the ___ shown in ..." -> before the echo
            t = t.split("___", 1)[0]
        if tail_key and tail_key.lower() in t.lower():
            t = t[:t.lower().index(tail_key.lower())]
        hw = head.strip().split()
        for n in range(min(4, len(hw)), 0, -1):                # trailing "is the" copied from before the blank
            end = " " + " ".join(hw[-n:])
            if t.lower().rstrip().endswith(end.lower()):
                t = t.rstrip()[: -len(end)]
                break
        lead = " ".join(hw[-2:]).lower()                       # a leading "<Subject 1> is the" copied from the prompt
        k = t.lower().find(lead + " ") if lead else -1
        if 0 <= k <= 20:
            t = t[k + len(lead) + 1:]
    t = " ".join(t.split()[:max_words]).strip(" ,;:.")
    before = slot.split("___")[0].rstrip().lower().split()[-1:] if "___" in slot else []
    if before and before[0] in ("the", "a", "an"):
        for art in ("the ", "a ", "an "):
            if t.lower().startswith(art):
                t = t[len(art):]
                break
    return t


def describe_target(clip, crop: np.ndarray, max_tokens: int = 64, slot: str = "", temperature: float = 0.0,
                   seed: int = 0, image_max_side: int = 448) -> str:
    img = torch.from_numpy(crop.astype(np.float32) / 255.0)[None]
    side = max(224, min(4096, int(image_max_side or 1024)))
    scale = min(1.0, side / max(crop.shape[:2]))
    if scale < 1.0:
        h, w = max(2, round(crop.shape[0] * scale)), max(2, round(crop.shape[1] * scale))
        img = torch.nn.functional.interpolate(img.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1)
    t = vlm_generate(clip, TARGET_Q + fit_note(slot), img, max_tokens, temperature, seed)
    t = strip_answer_label(t).split("\n")[0].strip().strip('"\'').rstrip(".")
    if slot:
        t = tidy_slot(t, slot)
    elif t and not t.lower().startswith("the "):
        t = "the " + t[0].lower() + t[1:]
    return " ".join(t.split()[:16]) or "the person"


def fill_target(text: str, desc: str) -> str:
    """{target} = who is replaced in this shot (picked + described in the panel), else 'the person'."""
    return text.replace("{target}", desc.strip() or "the person") if "{target}" in text else text


# ---------------------------------------------------------------------------- VLM suggestions (optional)

DEFAULT_VLM = {"enabled": False, "frames": 3, "max_tokens": 1024, "auto_segment": True, "auto_shot": True,
               "instruction": "", "describe_preset": "short", "describe_custom": "", "fit_prompt": True,
               "temperature": 0.0, "seed": 0, "image_max_side": 1024}
_VLM: dict[str, Any] = {}            # the VLM connected to a planner (kept for the panel's Analyse button)
_VLM_CACHE: dict[tuple, dict] = {}

VLM_QUESTION = """You see {n} frames (in time order) of ONE camera shot from a video that will be edited with an AI video model.
Answer with one JSON object only, no code fence:
{{"segment": "the main person or object a user would edit, as a short English noun phrase a segmentation model understands, describing what it looks like (e.g. 'woman in a black top', 'man in a grey suit', 'red car'); never on-screen text",
"shot": "one or two sentences for a video prompt: camera distance and angle, camera movement, where the subject is in the frame and what they do, in time order; describe actions and framing, not the person's face or clothes",
"people": <number of people visible>,
"recommend": "run" or "skip" (skip when the shot has nobody to edit, is a title card, a black frame or text only),
"reason": "a few words"}}{extra}"""


async def _off_loop(fn):
    """Run slow work in a thread so the server keeps answering (and the panel gets progress events)."""
    import asyncio
    _outside_prompt()
    return await asyncio.get_running_loop().run_in_executor(None, fn)


def _outside_prompt() -> None:
    """Models run from a panel button (no prompt running): ComfyUI's progress hook reads the server's last prompt
    and node ids, which do not exist until a first prompt has run."""
    try:
        from server import PromptServer
        srv = PromptServer.instance
        for attr in ("last_prompt_id", "last_node_id"):
            if not hasattr(srv, attr):
                setattr(srv, attr, None)
    except Exception:  # noqa: BLE001
        pass


def _node_boundary() -> None:
    """What ComfyUI does between two nodes: drop the per-thread CUDA malloc graph and prefetch queues. Needed when
    one node runs a model several times (a second text generation in the same node otherwise hits a device assert)."""
    try:
        import comfy.model_prefetch
        comfy.model_prefetch.cleanup_prefetch_queues()
    except Exception:  # noqa: BLE001 - older ComfyUI without it
        pass


def _textgenerate_execute(TextGenerate, **kwargs):
    """Keep MTP disabled when TextGenerate supports it, and tolerate older signatures without it."""
    import inspect

    try:
        params = inspect.signature(TextGenerate.execute).parameters
        supports_mtp = "mtp" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        supports_mtp = False

    call_kwargs = dict(kwargs)
    if supports_mtp:
        # MTP is speculative decoding. It is optional and off preserves the planner's prior behavior.
        call_kwargs["mtp"] = "off"
    try:
        return TextGenerate.execute(**call_kwargs)
    except TypeError as e:
        # Some wrappers advertise **kwargs but forward to an older execute() implementation.
        if "mtp" not in call_kwargs or "unexpected keyword argument 'mtp'" not in str(e):
            raise
        call_kwargs.pop("mtp")
        return TextGenerate.execute(**call_kwargs)


def vlm_generate(clip, prompt: str, images: torch.Tensor, max_tokens: int, temperature: float = 0.0, seed: int = 0,
                 sampling: dict | None = None) -> str:
    """Text from a vision-language CLIP. Some models end the answer before writing anything for some wordings:
    retry with a reworded request, then with sampling, and clean leftovers of the chat template. temperature > 0
    samples (with `seed`) from the first try, so another seed gives another answer."""
    from comfy_extras.nodes_textgen import TextGenerate
    first = ({"sampling_mode": "on", "temperature": float(temperature), "top_k": 40, "top_p": 0.9, "min_p": 0.0,
              "repetition_penalty": 1.05, "presence_penalty": 0.0, "seed": int(seed)}
             if temperature and temperature > 0 else {"sampling_mode": "off"})
    if sampling:   # e.g. long answers: greedy decoding falls into loops there
        first = {"sampling_mode": "on", "top_k": 40, "top_p": 0.9, "min_p": 0.0, "seed": int(seed), **sampling}
    tries = [(prompt, first),
             ("Look at the picture(s) carefully. " + prompt + " Write the answer now.", {"sampling_mode": "off"}),
             (prompt, {"sampling_mode": "on", "temperature": 0.7, "top_k": 40, "top_p": 0.9, "min_p": 0.0,
                       "repetition_penalty": 1.05, "presence_penalty": 0.0, "seed": 1})]
    text = ""
    for q, mode in tries:
        text = _textgenerate_execute(TextGenerate, clip=clip, prompt=q, max_length=int(max_tokens),
                                     sampling_mode=mode, image=images).args[0]
        _node_boundary()
        text = str(text or "").strip()
        for junk in ("assistant\n", "assistant:", "assistant"):
            if text.lower().startswith(junk):
                text = text[len(junk):].strip()
        if text:
            break
    return text


def vlm_cfg(cfg: dict | None) -> dict:
    out = dict(DEFAULT_VLM)
    out.update({k: v for k, v in (cfg or {}).items() if v is not None})
    return out


def _parse_json(text: str) -> dict:
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1]
        t = t[4:] if t.lower().startswith("json") else t
    a, b = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[a:b + 1]) if a >= 0 and b > a else {}
    except json.JSONDecodeError:
        return {}


def vlm_shot(clip, path: str, analysis: dict, start: int, end: int, cfg: dict) -> dict:
    """Ask the VLM about one shot (a few frames); cached per shot and settings."""
    key = (path, os.path.getmtime(path), int(start), int(end), id(clip),
           json.dumps({k: cfg[k] for k in ("frames", "max_tokens", "instruction", "image_max_side")}, sort_keys=True))
    if key in _VLM_CACHE:
        return _VLM_CACHE[key]
    from comfy_extras.nodes_textgen import TextGenerate
    _status("Asking the VLM about the shot", force=True)
    n = max(1, min(8, int(cfg["frames"])))
    src = _timeline(analysis["n_src"], analysis["fps_src"], float(analysis["fps"]))
    picks = np.clip(np.linspace(start, end - 1, n).round().astype(int), 0, analysis["n"] - 1)
    side = max(224, min(4096, int(cfg.get("image_max_side") or 1024)))
    s = side / max(analysis["width"], analysis["height"])
    size = (max(32, int(analysis["width"] * s) // 2 * 2), max(32, int(analysis["height"] * s) // 2 * 2))
    imgs = torch.from_numpy(np.stack(_read_frames(path, src[picks], size)).astype(np.float32) / 255.0)
    extra = ("\n" + cfg["instruction"].strip()) if str(cfg.get("instruction") or "").strip() else ""
    q = VLM_QUESTION.format(n=n, extra=extra)
    text = vlm_generate(clip, q, imgs, int(cfg["max_tokens"]))

    out = _parse_json(text)
    out = {"segment": str(out.get("segment") or "").strip(), "shot": str(out.get("shot") or "").strip(),
           "people": out.get("people"), "recommend": str(out.get("recommend") or "run").lower(),
           "reason": str(out.get("reason") or "").strip(), "raw": "" if out else text}
    _VLM_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------- reference descriptions (VLM)

# the subject can be a person, a creature, a robot, an animal or a cartoon character: say what it is first, then the
# traits that apply to it (a goblin is not described as "a person with greenish skin")
_WHAT = ("First say what the subject is in a few words (e.g. 'A young woman', 'A goblin', 'A humanoid robot', 'A golden "
         "retriever', 'An anime girl'), then describe it with the traits that apply to that kind of subject: ")
_DESCRIBE_ONLY = ("Treat all pictures as views of the same subject and use only clearly visible details. Return only "
                  "the requested English description, with no heading, label, preamble, bullet list or Markdown.")
DESCRIBE_PRESETS = {
    # default: a noun phrase that fits inside the prompt, "<Subject 1> is the {details} shown in <Picture 1>"
    "short": ("Describe the subject of these reference pictures for a video-generation prompt. Write one English noun "
              "phrase of at most 20 words: first identify what the subject is (such as a young woman, goblin, humanoid "
              "robot, dog or anime character), then give only its most distinctive visible features and clothing with "
              "colours. The phrase must fit after '<Subject 1> is the' and before 'shown in <Picture 1>', so do not start "
              "with an article. Never describe pose, action, expression, camera or background. " + _DESCRIBE_ONLY),
    "full body": ("Describe the subject of these reference pictures for a video-generation prompt: one paragraph of 3 to 5 "
                  "sentences in English. " + _WHAT + "for people and creatures the apparent gender and age, skin or fur "
                  "colour and texture, face, head shape, ears, horns or other distinctive features, hair, body build and "
                  "proportions, then the clothing piece by piece with colours and materials, shoes and accessories; for "
                  "robots the design, materials, colours, head or face plate, panels, joints and lights; for animals the "
                  "breed or species, fur colour, length and markings, ears, tail, build and any collar or accessory. Only "
                  "what is visible. Describe only appearance: never the pose, gesture, action, expression, camera, framing "
                  "or background. Concrete words only: no negations, no names, no opinions. " + _DESCRIBE_ONLY),
    "head / face": ("Describe the head and face of the subject of these reference pictures for a video-generation prompt: one "
                    "paragraph of 3 to 4 sentences in English. " + _WHAT + "skin, fur or surface (colour and texture), "
                    "head and face shape, eyes, eyebrows, nose or snout, mouth, jawline, ears, horns, facial hair, "
                    "wrinkles, marks, glasses or face plate, and the hair or fur: colour, length, texture, style. Only "
                    "what is visible. Describe only appearance: never the head angle, gaze, expression, action, camera or "
                    "background. Concrete words only: no negations, no names, no opinions. " + _DESCRIBE_ONLY),
    "face attributes": ("Describe the face and head in these reference pictures for a video-generation prompt. Write one "
                        "short comma-separated English attribute list. First identify the apparent gender and age group "
                        "for a person, or identify the kind of creature, robot, animal or cartoon character. Then cover, "
                        "in order: head and face shape and proportions (such as oval, round, square, heart-shaped, "
                        "narrow/wide, jaw structure, whether the head reads as small/average/large relative to the "
                        "shoulders), skin, fur or surface colour, eye color, hair or fur color and style, and any distinctive "
                        "features (ears, horns, facial hair, jewelry, makeup, glasses, etc.). Physical traits only, never the "
                        "pose, expression or background. " + _DESCRIBE_ONLY),
    # an object (prop, product, vehicle, piece of furniture...): a noun phrase that fits inside the prompt like "short"
    "object": ("Describe the object in these reference pictures for a video-generation prompt. Write one English noun "
               "phrase of at most 25 words: first say what the object is (such as a leather handbag, red sports car, "
               "wooden chair, ceramic mug or sword), then its most distinctive visible features: shape, materials, colors, "
               "finish, patterns, printed text or logos, and notable parts. The phrase must fit after 'the' and before "
               "'shown in <Picture 1>', so do not start with an article. Never describe the background, a person holding "
               "it, the camera or the lighting. Treat all pictures as views of the same object and use only clearly "
               "visible details. Return only the requested English description, with no heading, label, preamble, "
               "bullet list or Markdown."),
    "outfit": ("Describe only what the subject of these reference pictures wears, piece by piece, for a video-generation "
               "prompt: garments, armour or accessories, colours, materials, fit and shoes, in 2 or 3 sentences in "
               "English. Start with what the subject is in two or three words (e.g. 'The goblin wears...'). Never the "
               "pose, action or background. No negations, no names. " + _DESCRIBE_ONLY),
}
_DESCRIBE_WORD_LIMITS = {"short": 20, "object": 25, "full body": 150, "head / face": 110, "face attributes": 65,
                         "outfit": 90}
# presets that write a noun phrase for a blank inside a prompt sentence (the others write paragraphs)
PHRASE_PRESETS = ("short", "object")
_DESCRIBE_CACHE: dict[tuple, str] = {}


def _image_key(img: torch.Tensor | None) -> str:
    if img is None:
        return ""
    x = torch.nn.functional.interpolate(img[:1, ..., :3].movedim(-1, 1).float(), size=(32, 32), mode="area")
    return hashlib.sha1((x * 255).round().to(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def clean_slot(d: str) -> str:
    """A description going into '<Subject n> is the {details} shown in ...': without the prompt words a VLM (or an old
    saved description) repeats around it, which would come out twice."""
    t = str(d or "").strip()
    for _ in range(3):
        t = re.sub(r"^(<Subject \d>\s*)?(is\s+)?(the|a|an)\s+", "", t, flags=re.I).strip()
        t = re.sub(r"\s*\bshown in(\s+<Picture \d>)?[\s.]*$", "", t, flags=re.I).strip()
        t = re.sub(r"\s+is(\s+(the|a|an))?[\s.]*$", "", t, flags=re.I).strip()
    return t


def tidy_details(text: str, cfg: dict) -> str:
    """Normalise every description preset. In particular, remove labels the VLM adds despite the prompt and keep
    automatic descriptions inside the size promised by the selected preset. The short phrase also has to fit inside
    '<Subject 1> is the {details} shown in ...'."""
    preset = cfg.get("describe_preset") or "short"
    t = str(text or "").replace("```", " ").strip()
    t = strip_answer_label(t)
    if preset in PHRASE_PRESETS:
        t = t.splitlines()[0] if t.splitlines() else ""
    t = " ".join(t.split()).strip().strip('"\'').strip()
    if preset in PHRASE_PRESETS:
        # the VLM sometimes repeats the prompt sentence around its answer ("is the ... shown in <Picture 1>", "... is
        # the shown in"): those words are already in the prompt, so they would come out twice
        for _ in range(3):
            t = re.sub(r"^(<Subject \d>\s*)?(is\s+)?(the|a|an)\s+", "", t, flags=re.I).strip()
            t = re.sub(r"\s*\bshown in(\s+<Picture \d>)?[\s.]*$", "", t, flags=re.I).strip()
            t = re.sub(r"\s+is(\s+(the|a|an))?[\s.]*$", "", t, flags=re.I).strip()
        t = t.rstrip(" .").strip('"\'').rstrip(" .")
    limit = _DESCRIBE_WORD_LIMITS.get(preset, 180)
    words = t.split()
    if len(words) > limit:
        t = " ".join(words[:limit]).rstrip(" ,;:")
        if preset not in PHRASE_PRESETS:
            t = t.rstrip(".") + "."
    return t


def describe_instruction(cfg: dict, slot: str = "") -> str:
    """The describe instruction; with fit_prompt and the prompt sentence that holds {details}, the VLM writes for it."""
    preset = cfg.get("describe_preset", "short")
    if preset == "custom":
        base = str(cfg.get("describe_custom") or DESCRIBE_PRESETS["short"]).strip()
    else:
        base = DESCRIBE_PRESETS.get(preset, DESCRIBE_PRESETS["short"])
    # only a short phrase fits a blank inside a sentence; the paragraph presets are written on their own
    fits = cfg.get("fit_prompt", True) and preset in PHRASE_PRESETS
    return base + (fit_note(slot) if fits else "")


def vlm_describe(clip, images: list, instruction: str, max_tokens: int = 320, temperature: float = 0.0,
                 seed: int = 0, image_max_side: int = 448) -> str:
    """One description for a set of reference pictures (e.g. a close-up and a full-body photo of one person)."""
    images = [im for im in images if im is not None]
    if not images:
        return ""
    side = max(224, min(4096, int(image_max_side or 1024)))
    key = (tuple(_image_key(im) for im in images), instruction, id(clip), int(max_tokens), float(temperature), int(seed), side)
    if key in _DESCRIBE_CACHE:
        return _DESCRIBE_CACHE[key]
    from comfy_extras.nodes_textgen import TextGenerate
    _status("Describing the references with the VLM", force=True)
    # Keep each image's aspect ratio and letterbox to a shared canvas. The setting controls
    # the longest side, before any additional resizing done by the model's own processor.
    batch = []
    for im in images:
        x = im[:1, ..., :3].movedim(-1, 1).float()
        s = min(1.0, side / max(x.shape[-2:]))
        nh, nw = max(2, int(x.shape[-2] * s)), max(2, int(x.shape[-1] * s))
        x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        canvas = torch.full((1, 3, side, side), 0.5)
        canvas[:, :, (side - nh) // 2:(side - nh) // 2 + nh, (side - nw) // 2:(side - nw) // 2 + nw] = x
        batch.append(canvas)
    imgs = torch.cat(batch, 0).movedim(1, -1).clamp(0, 1)
    text = vlm_generate(clip, instruction, imgs, int(max_tokens), temperature, seed)
    text = " ".join(str(text).replace("```", " ").split())
    _DESCRIBE_CACHE[key] = text
    return text


def describe_picture(clip, image, prompt: str, marker: str, cfg: dict) -> str:
    """One picture described alone ({details@Picture k}), written for the prompt sentence that holds `marker`."""
    slot = prompt_slot(prompt, marker) if marker and cfg.get("fit_prompt", True) else ""
    d = tidy_details(vlm_describe(clip, [image], describe_instruction(cfg, slot), int(cfg["max_tokens"]),
                                  float(cfg.get("temperature") or 0), int(cfg.get("seed") or 0),
                                  int(cfg.get("image_max_side") or 1024)), cfg)
    return tidy_slot(d, slot) if slot and (cfg.get("describe_preset") or "short") in PHRASE_PRESETS else d


def picture_markers(text: str) -> str:
    """{details} and {details2} are the descriptions of pictures 1 and 2: {details@Picture 1} / {details@Picture 2}."""
    return str(text or "").replace("{details2}", "{details@Picture 2}").replace("{details}", "{details@Picture 1}")


def saved_picture_details(plan: dict) -> dict:
    """One description per picture: picture_details, plus the ones plans before 1.113 saved per reference set in
    ref_details ('a|b' -> {details, details2}, or one text for a)."""
    out = {}
    for key, value in (plan.get("ref_details") or {}).items():
        a, _, b = str(key).partition("|")
        pairs = ((a, value.get("details") or value.get("legacy")), (b, value.get("details2"))) \
            if isinstance(value, dict) else ((a, value),)
        for name, text in pairs:
            if name and text and name not in out:
                out[name] = str(text)
    out.update({k: v for k, v in (plan.get("picture_details") or {}).items() if v})
    return out


def picture_cfg(cfg: dict, plan: dict, name: str) -> dict:
    """The describe settings of one picture: the VLM card's, with the picture's own preset (and custom instruction)
    when it has one (plan['picture_cfg'][name] = {preset, custom})."""
    own = (plan.get("picture_cfg") or {}).get(name) or {}
    if not own.get("preset"):
        return cfg
    return dict(cfg, describe_preset=own["preset"], describe_custom=own.get("custom") or cfg.get("describe_custom", ""))


def fill_picture_details(text: str, names: list, images: list, saved: dict, clip=None, cfg: dict | None = None,
                         tag: str = "", keep_missing: bool = False) -> str:
    """{details@Picture k} = the description of the shot's k-th picture: the one saved for that image (📝 Describe each
    picture, or typed in the panel), else the VLM describes it now, else 'character' (keep_missing: the marker stays,
    for the conditioning node's VLM)."""
    if "{details@" not in (text or ""):
        return text

    def sub(m):
        k = int(m.group(1)) - 1
        name = names[k] if 0 <= k < len(names) else ""
        image = images[k] if 0 <= k < len(images) else None
        d = (saved or {}).get(name, "") if name else ""
        if not d and clip is not None and image is not None:
            d = describe_picture(clip, image, text, m.group(0), cfg or vlm_cfg({}))
            if d:
                print(f"{tag}: {m.group(0)} = {d!r} (described automatically)", flush=True)
        if d:
            return clean_slot(d)
        if image is None:
            print(f"{tag}: {m.group(0)}: the shot has no picture {k + 1}", flush=True)
        elif keep_missing:
            return m.group(0)
        return "character"
    return PICTURE_DETAILS.sub(sub, text)


# ---------------------------------------------------------------------------- prompt writer (VLM)

WRITER_TASKS = ["character swap", "style", "setting", "appearance", "lighting / weather", "custom"]

WRITER_INSTRUCTION = """You write the prompt for MiniMax H3, a video model, for a split-screen "duet": the KEPT FOOTAGE (the first {nf} images, one camera shot in time order) stays exactly as it is in one half of the frame, and the other half is generated moving in sync with it. {refs_txt}

Task: {task}. {change}

Rules (render-tested, follow them exactly):
1. Never describe the kept footage's performer, face, hair, clothes, props or room, not even to contrast them: whatever you name gets drawn, whatever you leave out is copied from the kept footage.
2. The kept footage has no tag: call it "the kept footage". Never write <Video 1>.
3. In the shot paragraph, restate the new look with two or three concrete words from the reference pictures (hair, outfit, face){face_rule}.
4. Describe what the camera records in the kept footage: distance, angle and movement, where the subject is, and what they do, one sentence per real action, in time order.
5. Write only what is seen. No negations (no "not", "no", "never", "without"), no quality words ("realistic", "high quality", "cinematic").
6. Keep the literal text {{layout}} exactly where it is in the template.

Output exactly these six sections in this order, nothing before or after:

subject_definitions:
{subject_line}

summary:
[reference generation] The target video is a split screen: the kept footage beside <the generated half>, which moves in sync with it.

retention_analysis:
<one line per subject: "... fully_preserved - ... are retained.">
The kept footage: fully_preserved - the panel is kept exactly.

detailed_description:
The target video is in <style and medium>. {{layout}}

[Shot 1] <camera, framing, the subject restated, the actions>

overall_soundscape:
<the sounds of the scene>

non_diegetic_music:
None."""

TASK_SUBJECT = {
    "character swap": '"<Subject 1> is the person whose appearance comes from <Picture 1>[ and <Picture 2>]: <face, hair, skin, outfit piece by piece, from the pictures>."',
    "style": '"The kept footage is the motion reference of the split screen." (and, when there are reference pictures, "<Picture 1> is the style reference.")',
    "setting": '"<Subject 1> is the place: <the requested place with two or three concrete details>." (from <Picture 1> when given)',
    "appearance": '"<Subject 1> is the performer of the kept footage, now <the requested look in concrete seen words>."',
    "lighting / weather": '"The kept footage is the motion reference of the split screen."',
    "custom": '"<Subject 1> is ..." for whatever the change brings in, from the pictures when given.',
}

TASK_HINTS = {
    "character swap": "Replace the performer with the person from the reference pictures; the place stays the same room.",
    "style": "Keep the same performance and camera, redrawn in the requested style.",
    "setting": "Keep the same performer and performance, moved to the requested place.",
    "appearance": "Keep the same performance, with the performer's look changed as requested.",
    "lighting / weather": "Keep the same performance and place, under the requested light or weather.",
    "custom": "Do what the change below asks, keeping the kept footage's performance and camera in sync.",
}


def _letterbox_batch(images: list, side: int = 448) -> torch.Tensor:
    batch = []
    for im in images:
        x = im[:1, ..., :3].movedim(-1, 1).float()
        s = side / max(x.shape[-2:])
        nh, nw = max(2, int(x.shape[-2] * s)), max(2, int(x.shape[-1] * s))
        x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        canvas = torch.full((1, 3, side, side), 0.5)
        canvas[:, :, (side - nh) // 2:(side - nh) // 2 + nh, (side - nw) // 2:(side - nw) // 2 + nw] = x
        batch.append(canvas)
    return torch.cat(batch, 0).movedim(1, -1).clamp(0, 1)


def _clean_written(text: str) -> str:
    t = text.replace("```", "").strip()
    i = t.find("subject_definitions:")
    if i > 0:
        t = t[i:]
    t = scrub_meta(t)
    if "{layout}" not in t and "split screen divided" not in t and "{panel_region}" not in t and "detailed_description:" in t:
        a = t.index("detailed_description:") + len("detailed_description:")
        nl = t.find("\n[Shot", a)
        cut = nl if nl > 0 else len(t)
        t = t[:cut].rstrip() + " {layout}\n" + t[cut:]
    return t


# ---------------------------------------------------------------------------- auto prompt (MiniMax H3)
# The VLM writes each shot's prompt in H3's full-reference format (MiniMaxAI/MiniMax-H3 docs/
# VIDEO_PROMPT_WRITING_GUIDE_ref_en.md) from key frames of the shot, the reference pictures and their descriptions, the
# user's request (the prompt box) and the shot's marks and mode.

# frames 0 = from the shot's length; style: "h3" (the official six sections) or "captions" (the short OpenJutsu captions)
AUTO_PROMPT_DEFAULTS = {"enabled": False, "frames": 0, "source": "guide", "style": "h3"}
H3_SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape",
               "non_diegetic_music")

AUTO_PROMPT_RULES = """Write exactly these six sections, in English, in this order, with nothing before or after them:

subject_definitions:
One line per referenced item. A <Subject N> is visible content taken from the reference pictures (a person, creature, object, place, outfit or style), written as "<Subject 1> is the <what it is> in <Picture 1>, with <its main visible features>." Number subjects from 1 in the order of their pictures. Cite each picture inside its subject's line; never give a picture a line of its own.{source_line}

summary:
One short paragraph that starts with the task type in square brackets, e.g. [video editing + reference generation], then {summary_start} Use only the labels defined above.

retention_analysis:
One line per <Subject N>: "<Subject 1> (appears in [Shot 1]): fully_preserved - <what of it is kept>." The marker is one of fully_preserved, partially_preserved, attribute_transfer, weak_reference.

detailed_description:
Open with one or two sentences on the visual style and medium (e.g. live-action handheld smartphone footage, cinematic, 3D CG, 2D-animated) and the light. Then "[Shot 1]" with no timestamp. {shots_rule} Never write a timestamp inside a shot: describe what happens within a shot in order with words like then, as, while. Describe the target video in playback order and in detail, 250 to 450 words: the composition, where each subject is and what it does, beat by beat, with the change the request asks for, the place and the light, the camera movement as natural English with these terms (static shot, push in, pull out, zoom in, pan left, pan right, tilt up, tilt down, truck left, truck right, tracking shot, arc shot, shake slightly; "with small amplitude" / "with large amplitude", "at slow speed" / "at fast speed" only when they matter), and each sound when it happens. Insert <Subject N> at the first appearance of each subject with its key visible features, and keep using the same label later. Say plainly what changes: when a subject replaces someone of the source, write that the person of the {source_name} is replaced by <Subject N>, who takes their exact place, timing, pose, gestures, head motion, gaze and expressions. In detailed_description always call each subject by its label (<Subject 1>, <Subject 2>), with a few of its features at its first appearance, never only by a description. A subject acts with its own body: a person or creature replacing an animal uses hands and arms, an animal replacing a person uses paws. Describe each action once, in order; never repeat a sentence. Whatever the request does not change stays as it is in the source frames: describe it concretely (the same room, the same camera move, the same actions and timing). Speech: give a speaking person a speaker ID such as (S1); write words inside <d>[Language] ...</d> only when the request gives them, otherwise describe the mouth moving as they talk. Visible on-screen text goes in double quotes as it reads.

overall_soundscape:
One to four sentences of ambience and physical sounds across the video.

non_diegetic_music:
N/A, or one to three sentences (instruments, tempo, dynamics) when the request asks for music only the audience hears.

Always write the labels with angle brackets: <Subject 1>, <Picture 1>{video_label}. Write only what is seen or heard. No negations (not, no, never, without), no quality words (realistic, high quality, masterpiece, 4k), no names of real people, no explanations of the rules."""

# a request that is already a prompt template (e.g. the one a LoRA was trained with): kept as it is, made specific
AUTO_TEMPLATE_RULES = """The user's request is a prompt TEMPLATE (for example the one their LoRA was trained with). Return the same template, made specific to this shot:
- Keep every sentence of the template, in the same order, with the same section names; add no section and remove none.
- Keep its first word{trigger} exactly as the very first text of your answer.
- Where the template describes a subject in general words, add the visible features from the reference pictures and their descriptions.
- Apply the user's request to it (who is replaced by whom, what changes), in the template's own words.
- In detailed_description, right after "[Shot 1]" (add "[Shot 1]" if the template has none), add three to six sentences on this shot: the camera distance, angle and movement (static shot, push in, pull out, pan left, pan right, tilt up, tilt down, tracking shot...), the place and the light, and what the person does in order. {shots_rule} Never write a timestamp inside a shot.
- Keep every label as written, with angle brackets: <Subject 1>, <Picture 1>.
Return only the finished prompt, with nothing before or after it."""

# the short style of the OpenJutsu training captions (openjutsu/scripts/gen_empty_captions.py), written from the request
AUTO_CAPTION_RULES = """Write the prompt in the short style of a video-edit request: two to four sentences in English, one paragraph, no section names, no labels other than <Picture n>, no lists, no markdown.
- First, for each change the request asks for, one sentence like "Replace the man in the video with the young woman with long wavy red hair, wearing a black leather jacket and blue jeans, from <Picture 1>." Say who or what in the video is replaced, then a short description of what the picture shows, taken from the picture itself: for a person or creature its apparent gender and age, hair and clothes; for an animal its kind and coat; for an outfit or an object what it is, its colour and material; for a place what kind of place it is.
- When there are several people or animals, say which one in the video becomes which picture (left, right, in front, behind, the one who enters later). Mention every picture the request uses, as <Picture 1>, <Picture 2> and so on, and call the source "the video".
- A subject acts with its own body: a person or creature replacing an animal uses hands, an animal replacing a person walks on four legs.
- Then one sentence on what stays: whether the background and location are kept or replaced (by a picture only if that picture shows a place), and that the motion, camera and timing follow the video.
Do not invent anything. Reply with the prompt only."""


def auto_frame_count(length: int, fps: float, frames: int = 0) -> int:
    """Key frames the writer looks at: the plan's number, else about 1.5 per second of the shot (3 to 8)."""
    if frames:
        return max(2, min(12, int(frames)))
    return max(3, min(8, round(length / max(1.0, fps) * 1.5)))


def shot_mode(seg: dict) -> str:
    """frame, paste, mask, crop or cropmask (the panel's names for a shot's mode)."""
    if seg.get("crop"):
        return "cropmask" if seg.get("inpaint") else "crop"
    return "mask" if seg.get("inpaint") else "paste" if seg.get("paste") else "frame"


def auto_prompt_for(instruction: str, names: list, seg: dict, template: str = "", style: str = "h3") -> str:
    """What a written prompt was made from (the panel computes the same string to show it is out of date)."""
    marks = [[x["picture"], x["color"]] for x in seg.get("color_guide_marks") or []] if seg.get("color_guide") else []
    return json.dumps([instruction or "", names, marks, shot_mode(seg), (seg.get("mask") or {}).get("text") or "",
                       seg.get("target") or "", template or "", style or "h3"], separators=(",", ":"), ensure_ascii=False)


def write_h3_prompt(clip, frames: list, times: list, pictures: list, picture_texts: list, instruction: str,
                    seg: dict, source: str = "guide", duration: float = 0.0, max_tokens: int = 2048,
                    cuts: list | None = None, template: str = "", style: str = "h3") -> str:
    """One shot's prompt in H3's full-reference format, written by the VLM from key frames of the shot ([1,H,W,3] each,
    at `times` seconds into it), the shot's reference pictures (<Picture 1> first) with their descriptions, the user's
    request and the shot's Colored Marks, selection and mode."""
    nf, npics = len(frames), len(pictures)
    tt = ", ".join(_mmss(t) for t in times) if times else ""
    pics = ""
    if npics:
        lines = [f"<Picture {k + 1}>: image {nf + k + 1}" + (f", described as: {d}" if d else "")
                 for k, d in enumerate(picture_texts[:npics] + [""] * max(0, npics - len(picture_texts)))]
        pics = (f" The next {npics} images are the reference pictures, in order:\n" + "\n".join(lines))
    sel = (seg.get("mask") or {}).get("text") or ""
    notes = []
    if seg.get("color_guide"):
        marks = seg.get("color_guide_marks") or []
        who = "; ".join(f"the person tinted {_COLOR_GUIDE_NAMES.get(x['color'], x['color'])} becomes the subject from "
                        f"<Picture {x['picture']}>" for x in marks)
        notes.append("In the source frames, translucent colored tints are temporary editing marks, not anyone's look: "
                     f"{who}. In subject_definitions and detailed_description, say which tinted person each subject "
                     "replaces, and that the marks are removed completely in the target video.")
    if seg.get("target"):
        notes.append(f"The subject being replaced in the source: {seg['target']}.")
    elif sel:
        notes.append(f"The selected subject in the source: {sel}.")
    mode = shot_mode(seg)
    if mode in ("mask", "cropmask"):
        notes.append("Only the selected subject is generated again; everything else in the frame stays exactly as in "
                     "the source.")
    if mode in ("crop", "cropmask"):
        notes.append("The frames are a crop around the selected subject; describe what this crop shows.")
    if source == "video":
        source_line = " Also add the line \"<Video 1> is the source video for the target video edit.\""
        summary_start = "\"The target video is an edited version of <Video 1>.\""
        rel = "The source shot is <Video 1>, the video being edited."
        source_name, video_label = "<Video 1>", ", <Video 1>"
    else:
        source_line = ""
        summary_start = ("\"The target video is an edited version of the guide video.\" and what it shows with which "
                         "subjects.")
        rel = ("The source shot is given to the model as the guide video, aligned frame by frame (it has no tag; call it "
               "\"the guide video\"): the target keeps its timing, camera and motion unless the request changes them.")
        source_name, video_label = "guide video", ""
    # the template: the conditioning prompt input (the one a LoRA was trained with), or a request that is a whole prompt
    if not template and is_template(instruction):
        template, instruction = instruction, ""
    trigger = request_trigger(template)
    if cuts:
        shots_rule = ("The source has camera cuts at " + ", ".join(_mmss(t) for t in cuts) + ": start a new shot at each "
                      "of them and nowhere else, " + ", ".join(f'"[Shot {k + 2}] At {_mmss(t)}, the shot cuts to ..."'
                                                                for k, t in enumerate(cuts)) + ".")
    else:
        shots_rule = "The source is one continuous shot: write only [Shot 1] and no timestamp at all."
    captions = style == "captions" and not template
    if captions:
        rules = AUTO_CAPTION_RULES
    elif template:
        rules = AUTO_TEMPLATE_RULES.format(trigger=f" '{trigger}'" if trigger else "", shots_rule=shots_rule)
    else:
        rules = AUTO_PROMPT_RULES.format(source_line=source_line, summary_start=summary_start, source_name=source_name,
                                         video_label=video_label, shots_rule=shots_rule)
    ask = (f"You write the prompt for MiniMax H3, a video model, in its full-reference format. The target video is an "
           f"edited version of a source shot{f' of {duration:.2f} s' if duration else ''}: the first {nf} images are "
           f"frames of that shot in time order{f' (at {tt} on its clock)' if tt else ''}.{pics}\n\n"
           f"{rel}\n\nThe user's request: {instruction.strip() or ('(none: follow the template)' if template else '(none: use the reference pictures as they are)')}\n"
           + (f"\nThe prompt template:\n{template.strip()}\n" if template else "")
           + ("A {details@Picture n} stands for the description of <Picture n>: write the visible features of that "
              "picture in its place.\n" if "{details@" in instruction + template else "")
           + ("".join(f"\n{n}" for n in notes)) + "\n\n"
           + rules)
    if captions:   # the caption style has no H3 sections, no shot structure and no <Subject n>
        ask = ask.replace("You write the prompt for MiniMax H3, a video model, in its full-reference format.",
                          "You write the request a user types to edit a video with a video model.")
        ask = ask.replace(rel, "The source is the video; the target keeps its timing, camera and motion unless the "
                               "request changes them.")
    images = _letterbox_batch(list(frames) + list(pictures))
    tokens = 512 if captions else max(1536, int(max_tokens))
    sampling = {"temperature": 0.6, "repetition_penalty": 1.1, "presence_penalty": 0.8}
    tidy = (lambda t: tidy_caption(t, seg)) if captions else \
        (lambda t: tidy_h3_prompt(t, seg, trigger, bool(cuts), npics, bool(template)))
    check = (lambda t: caption_problems(t, npics)) if captions else (lambda t: h3_prompt_problems(t, npics, bool(template)))
    text = tidy(vlm_generate(clip, ask, images, tokens, sampling=sampling))
    problems = check(text)
    if problems:   # one more try, told what was wrong
        print(f"[BFS Auto prompt] writing again: {'; '.join(problems)}", flush=True)
        again = (ask + "\n\nA previous answer had these problems, avoid them: " + "; ".join(problems) + ".")
        text2 = tidy(vlm_generate(clip, again, images, tokens, seed=1, sampling=sampling))
        if len(check(text2)) < len(problems):
            text = text2
    return text if captions else name_subjects(text)


def tidy_caption(text: str, seg: dict | None = None) -> str:
    """A caption-style answer: one paragraph, no fences or quotes, no repeated sentences, labels with brackets, and
    the Colored Mark mapping added when a mark color is missing."""
    t = str(text or "").replace("```text", "").replace("```", "").strip().strip('"').strip()
    t = " ".join(drop_repeats(t).split())
    t = re.sub(r"(?<![<\w])(Picture) (\d+)(?![\w>])", r"<\1 \2>", t)
    if seg and seg.get("color_guide"):
        names = [_COLOR_GUIDE_NAMES.get(x["color"], x["color"]) for x in seg.get("color_guide_marks") or []]
        if any(n not in t for n in names):
            t = color_guide_prompt(t, seg["color_guide_marks"])
    return t


def caption_problems(text: str, npics: int) -> list[str]:
    """What is wrong with a caption-style answer."""
    problems = []
    if re.search(r"(?i)\b(subject_definitions|summary|detailed_description)\s*:", text):
        problems.append("it has H3 section names: write a short paragraph instead")
    if "<Subject" in text:
        problems.append("it uses <Subject n>: name the pictures as <Picture n>")
    if npics and "<Picture" not in text:
        problems.append("it never names a picture as <Picture n>")
    if len(text.split()) > 170:
        problems.append("it is too long: two to four sentences")
    return problems


def name_subjects(text: str) -> str:
    """Last resort when detailed_description still never says <Subject N>: right after [Shot 1], one sentence that
    ties each missing label to its short description from subject_definitions, so H3 knows who is who."""
    i = text.find("detailed_description:")
    if i < 0 or "[Shot 1]" not in text[i:]:
        return text
    j = text.find("overall_soundscape:", i)
    body = text[i:j if j > 0 else len(text)]
    defs = dict(re.findall(r"<Subject (\d+)> is (?:the |a |an )?([^.\n]+?)(?: in| shown in| from) <Picture", text.split("summary:")[0]))
    missing = [k for k in sorted(defs, key=int) if f"<Subject {k}>" not in body]
    if not missing:
        return text
    short = lambda d: " ".join(d.split(",")[0].split()[:8])
    line = " ".join(f"<Subject {k}> is the {short(defs[k])}." for k in missing)
    k0 = text.index("[Shot 1]", i) + len("[Shot 1]")
    return text[:k0] + " " + line + text[k0:]


def _sentences(text: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]


def drop_repeats(text: str) -> str:
    """A VLM stuck in a loop repeats its sentences: keep the first of each, per section."""
    out = []
    for line in text.split("\n"):
        seen, kept = set(), []
        for sent in _sentences(line):
            key = re.sub(r"\W+", " ", sent.lower()).strip()
            if key in seen and len(key.split()) > 3:
                continue
            seen.add(key)
            kept.append(sent)
        out.append(" ".join(kept) if kept else line)
    return "\n".join(out)


def h3_prompt_problems(text: str, npics: int, template: bool = False) -> list[str]:
    """What is wrong with a written prompt: a missing section, a subject never named by its label in
    detailed_description, a sentence repeated (a loop), an answer cut short."""
    problems = []
    if not template:
        problems += [f"the section {n}: is missing" for n in H3_SECTIONS if f"{n}:" not in text]
    i = text.find("detailed_description:")
    body = text[i:text.find("overall_soundscape:", i) if "overall_soundscape:" in text[i:] else len(text)] if i >= 0 else ""
    defined = sorted({int(k) for k in re.findall(r"<Subject (\d+)>", text.split("summary:")[0])})
    missing = [k for k in defined if f"<Subject {k}>" not in body]
    if body and missing:
        problems.append("detailed_description never names " + ", ".join(f"<Subject {k}>" for k in missing)
                        + " by its label")
    keys = [re.sub(r"\W+", " ", x.lower()).strip() for x in _sentences(body)]
    if keys and max(keys.count(k) for k in keys) >= 3:
        problems.append("sentences are repeated in a loop")
    return problems


def request_trigger(text: str) -> str:
    """'body_swap' for a request that starts 'body_swap: ...' (a LoRA's trigger word), '' otherwise."""
    m = re.match(r"\s*([A-Za-z][\w-]*)\s*:", str(text or ""))
    return m.group(1) if m and m.group(1).lower() not in H3_SECTIONS + ("retention", "subject") else ""


def is_template(text: str) -> bool:
    """A request that is a whole prompt (it has H3's sections), not a short instruction."""
    return len(re.findall(r"(?i)\b(subject_definitions|summary|detailed_description)\s*:", str(text or ""))) >= 2


def tidy_h3_prompt(text: str, seg: dict | None = None, trigger: str = "", cuts: bool = True, npics: int = 0,
                   template: bool = False) -> str:
    """The VLM's answer as a prompt: from subject_definitions on (after the request's trigger word, kept first), no
    code fences or Markdown bold around the section names, labels with their angle brackets, and the Colored Mark
    mapping added when the writer left a mark color out."""
    t = str(text or "").replace("```text", "").replace("```", "").strip()
    for name in H3_SECTIONS:
        t = re.sub(rf"(?im)^\s*[#*_]*\s*{name}\s*[*_]*\s*:[ \t]*[*_]*", f"{name}:", t)
    i = t.find("subject_definitions:")
    if i > 0:
        t = t[i:]
    if trigger and not t.lower().startswith(trigger.lower() + ":"):
        t = f"{trigger}: {t}"
    # [Shot 1] has no timestamp; a one-shot source has none at all (the VLM still writes them from the frame times)
    t = re.sub(r"(\[Shot 1\])\s*At \d+:\d+(?:\.\d+)?\s*,\s*(\w)", lambda m: f"{m.group(1)} {m.group(2).upper()}", t)
    if not cuts:
        t = re.sub(r"\bAt \d+:\d+(?:\.\d+)?\s*,\s*(\w)", lambda m: m.group(1).upper(), t)
    t = drop_repeats(t)
    if not template:   # an answer cut short (a loop until max tokens) still ends with the sound sections
        if "overall_soundscape:" not in t and not re.search(r"[.!?\"']\s*$", t):
            t = re.sub(r"(?<=[.!?])\s+[^.!?\n]*$", "", t)   # the sentence cut in the middle
        if "overall_soundscape:" not in t:
            t = t.rstrip() + "\noverall_soundscape:\nThe ambient sound of the guide video continues throughout."
        if "non_diegetic_music:" not in t:
            t = t.rstrip() + "\nnon_diegetic_music:\nN/A"
    # 'Picture 1' / 'Subject 2' / 'Video 1' written without their brackets
    t = re.sub(r"(?<![<\w])(Picture|Subject|Video|Audio) (\d+)(?![\w>])", r"<\1 \2>", t)
    if seg and seg.get("color_guide"):
        names = [_COLOR_GUIDE_NAMES.get(x["color"], x["color"]) for x in seg.get("color_guide_marks") or []]
        if any(n not in t for n in names):
            t = color_guide_prompt(t, seg["color_guide_marks"])
    return t.strip()


def key_frames(start: int, end: int, cuts: list, n: int) -> list[int]:
    """n frames of [start, end) for the writer: at least one in the middle of each piece between camera cuts, the
    rest shared by length (at most 12)."""
    edges = [start] + [c for c in cuts if start < c < end] + [end]
    pieces = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a][:12]
    n = max(n, len(pieces))
    share = [1 + int((n - len(pieces)) * (b - a) / max(1, end - start)) for a, b in pieces]
    out = []
    for (a, b), k in zip(pieces, share):
        out += [int(round(a + (b - a) * (j + 0.5) / k)) for j in range(k)]
    return sorted(set(min(end - 1, f) for f in out))


def auto_prompt_shot(clip, plan: dict, analysis: dict, path: str, seg: dict, names: list, pictures: list,
                     pic_details: dict, instruction: str, vcfg: dict, template: str = "") -> dict:
    """Write one shot's prompt: {"text", "for"} (what it was made from, see auto_prompt_for)."""
    ap = {**AUTO_PROMPT_DEFAULTS, **(plan.get("auto_prompt") or {})}
    fps = float(analysis["fps"])
    L = seg["end"] - seg["start"]
    n = auto_frame_count(L, fps, int(ap.get("frames") or 0))
    src = _timeline(analysis["n_src"], analysis["fps_src"], fps)
    cuts = [c for c in find_cuts(path, analysis, plan.get("detector", "adaptive"), float(plan.get("sensitivity", 0.5)))[0]
            if seg["start"] < c < seg["end"]]
    picks = np.clip(np.array(key_frames(seg["start"], seg["end"], cuts, n)), 0, analysis["n"] - 1)
    side = max(224, min(4096, int(vcfg.get("image_max_side") or 1024)))
    sc = side / max(analysis["width"], analysis["height"])
    size = (max(32, int(analysis["width"] * sc) // 2 * 2), max(32, int(analysis["height"] * sc) // 2 * 2))
    frames = [torch.from_numpy(f.astype(np.float32) / 255.0)[None] for f in _read_frames(path, src[picks], size)]
    # the request may name pictures with {details...}: their saved descriptions; the writer fills the others
    request = fill_picture_details(picture_markers(instruction), names, pictures, pic_details, keep_missing=True)
    tmpl = fill_picture_details(picture_markers(template), names, pictures, pic_details, keep_missing=True)
    _status("Writing the shot's prompt with the VLM", force=True)
    text = write_h3_prompt(clip, frames, [(int(x) - seg["start"]) / fps for x in picks], pictures,
                           [pic_details.get(nm, "") for nm in names], request, seg, ap.get("source") or "guide", L / fps,
                           int(vcfg.get("max_tokens") or 1024), [(c - seg["start"]) / fps for c in cuts], tmpl,
                           ap.get("style") or "h3")
    return {"text": text, "for": auto_prompt_for(instruction, names, seg, template, ap.get("style") or "h3")}


PLACE_Q = ("This picture shows a place; the patch of black-and-white noise only hides a person, ignore it. Describe the "
           "place in ONE short phrase of concrete seen words, at most 25 words: what kind of place, then two or three "
           "details (furniture, walls, light). Start with 'the'. The phrase only.")


def phrase(text: str) -> str:
    """A VLM answer used inside a sentence: one line, no quotes, no final full stop, lower-case start
    ('An elderly woman ...' -> 'an elderly woman ...')."""
    t = " ".join(str(text or "").strip().split("\n")[0].split()).strip().strip('"\'').rstrip(".").strip()
    if len(t) > 1 and t[0].isupper() and not t[1].isupper() and not t.startswith("<"):
        t = t[0].lower() + t[1:]
    return t


def describe_place(clip, setting: torch.Tensor | None, max_tokens: int = 256) -> str:
    if setting is None:
        return ""
    t = phrase(vlm_describe(clip, [setting], PLACE_Q, max_tokens))
    for verb in (" features ", " has ", " with ", " contains ", " is "):   # a sentence back into a noun phrase
        if verb in t and verb != " with ":
            t = t.replace(verb, " with ", 1)
            break
    return t if not t or t.lower().startswith(("the ", "a ", "an ")) else "the " + t


# words that describe nothing on screen (TSC: H3 reads one description of one video and does not know what is pinned)
META_FIXES = (("<Video 1>", "the original performance"), (" in sync with the kept footage", " at the same moments"),
              (" in sync with it", " at the same moments"), ("moves in sync", "moves at the same moments"),
              ("the kept footage", "the original performance"), ("The kept footage", "The original performance"),
              (" seamlessly", ""), (" frame by frame", ""))


def scrub_meta(text: str) -> str:
    out = []
    for line in text.split("\n"):
        if line.strip().lower().startswith(("the kept footage: fully_preserved", "the panel: fully_preserved")):
            continue   # no retention line for the pinned clip: it is not a reference
        for a, b in META_FIXES:
            line = line.replace(a, b)
        out.append(line)
    return "\n".join(out)


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:06.3f}"


def video_facts(clip, frames: list, times: list | None, max_tokens: int) -> dict:
    """What the clip shows, from a few frames with their times on the video's clock: one clause for its region, the
    medium, place, light, camera, the actions beat by beat (times for the big moments) and the sounds."""
    tt = ""
    if times:
        tt = " The frames are at " + ", ".join(_mmss(t) for t in times) + " (MM:SS.mmm) on the video's clock."
    q = (f"These {len(frames)} images are frames, in time order, of one video.{tt} Answer with one JSON object only, "
         "no code fence:\n{"
         '"seen": "one short clause of what is seen: who, doing what, where, e.g. a man dancing in a small bedroom", '
         '"medium": "what the video is and how it was captured, e.g. handheld vertical smartphone footage under warm '
         'lamp light", "style": "realistic, 3D CG or anime", "place": "the place in concrete words with two or three '
         'details", "light": "where the light comes from and what it falls on", "camera": "camera distance, angle and '
         'movement", "actions": "what the person does, in time order, one short sentence per real action; start a '
         'sentence with At MM:SS.mmm only for big moments (cuts, strikes, jumps, a spoken line), using the frame times; '
         'call them the person and never describe their looks", "sounds": "the sounds these actions make in this '
         'place, and when"}')
    d = _parse_json(vlm_generate(clip, q, _letterbox_batch(frames), max_tokens)) if frames else {}
    return {k: str(v).strip().rstrip(".") for k, v in d.items() if v and isinstance(v, (str, int, float))}


def write_duet_prompt(clip, frames: list, refs: list, task: str, change: str, max_tokens: int = 1024,
                      mode: str = "canvas", setting: torch.Tensor | None = None, times: list | None = None) -> str:
    """A duet prompt from frames of the pinned clip ([1,H,W,3] each) and the reference pictures, in TSC's newer
    wording: the finished split screen described as plain footage ({panel_region} shows <one clause of the clip>;
    {video_region} shows <Subject 1> doing the same movements at the same moments, in <place>), the motion spelled out
    beat by beat with times on the video's clock, the sounds of the new scene. The VLM only gives facts; the six
    sections are assembled here, so a small model cannot break the format."""
    refs = [r for r in refs if r is not None]
    fr = [f for f in frames if f is not None]
    key = ("write3", tuple(_image_key(f) for f in fr), tuple(_image_key(r) for r in refs), task, change, id(clip),
           int(max_tokens), mode, _image_key(setting), tuple(times or ()))
    if key in _DESCRIBE_CACHE:
        return _DESCRIBE_CACHE[key]
    _status("Writing the prompt with the VLM", force=True)
    swap = task == "character swap" and bool(refs)
    # separate questions: mixing the clip's frames and the reference photos in one batch makes the VLM describe the
    # reference photo (studio, standing still) as if it were the video
    look = ""
    if swap:
        look = vlm_describe(clip, refs, "Describe the subject of these reference pictures (a person, creature, robot or "
                            "animal) in ONE sentence of concrete seen words: what it is, apparent gender and age when it "
                            "has them, face, hair colour, length and style, skin, and the clothing piece by piece with "
                            "colours. Looks only: never the pose, expression, camera or background.",
                            max_tokens)
        look = phrase(look)
    d = video_facts(clip, fr, times, max_tokens)
    seen = d.get("seen") or "the original performance"
    medium = d.get("medium") or "real camera footage"
    style = (d.get("style") or "realistic").lower()
    style = style if style in ("realistic", "3d cg", "anime") else "realistic"
    style = "3D CG" if style == "3d cg" else style
    camera = d.get("camera") or "The same framing and camera movement"
    actions = d.get("actions") or ""
    light = d.get("light") or ""
    place = describe_place(clip, setting) if setting is not None else (d.get("place") or "the same place")
    place_at = f"{place} from {{setting}}" if setting is not None else place
    pics = " and ".join(f"<Picture {k + 1}>" for k in range(len(refs)))
    chg = change.strip().rstrip(".")
    split = mode != "shifted"
    if swap:
        actions = actions.replace("The person", "<Subject 1>").replace("the person", "<Subject 1>")
        defs = f"<Subject 1> is the subject whose appearance comes from {pics}" + (f": {look}." if look else ".")
        if chg:
            defs += f" {chg}."
        if setting is not None:
            defs += f"\n<Subject 2> is {place} from {{setting}}."
        who = "<Subject 1>"
        doing = f"<Subject 1> doing the same movements at the same moments, in {place_at}"
        keep = f"<Subject 1> (appears in [Shot 1]): fully_preserved - the face, hair and clothing from {pics} are retained."
        if setting is not None:
            keep += f"\n<Subject 2> (appears in [Shot 1]): fully_preserved - {place} is retained."
        first = actions[len("<Subject 1> "):] if actions.startswith("<Subject 1> ") else actions
        shot = (f"[Shot 1] {camera}. <Subject 1>, the face from <Picture 1>, "
                + (first or "performs every movement of the original performance at the same moments") + ".")
    else:
        what = {"style": f"redrawn as {chg or 'the requested style'}", "setting": f"moved to {chg or 'the requested place'}",
                "appearance": f"with the performer now {chg or 'changed'}", "lighting / weather": f"under {chg or 'the new light'}",
                }.get(task, chg or "changed as requested")
        actions = actions.replace("The person", "The performer").replace("the person", "the performer")
        defs = (f"<Subject 1> is the place: {chg}." if task == "setting" and chg else
                "<Subject 1> is the performer of the original performance.") + (f" {pics} show the look to follow." if refs else "")
        if setting is not None:
            defs += f" {{setting}} shows the place: {place}."
        who = "the same performance"
        doing = f"the same performance, {what}, the same movements at the same moments"
        keep = "<Subject 1> (appears in [Shot 1]): fully_preserved - every movement and expression, at the same moments."
        shot = f"[Shot 1] {camera}, {what}. " + (actions + "." if actions else "Every movement at the same moments.")
        if task == "style" and chg:
            style = chg
    if split:
        summary = (f"[reference generation] The target video is a split screen: {{panel_region}} shows {seen}, and "
                   f"{{video_region}} shows {doing}.")
        split_line = (f" A split screen divided by a thin straight {{split_line}} line: {{panel_region}} shows {seen}; "
                      f"{{video_region}} shows {who} performing the same movements at the same moments, in {place_at}"
                      f"{', ' + light if light and setting is None else ''}{{wider}}.")
    else:
        summary = f"[reference generation] The target video shows {doing}."
        split_line = ""
    style_line = (f"The target video is in {'an' if style[:1].lower() in 'aeiou' else 'a'} {style}"
                  + ("" if "style" in style.lower() else " style")
                  + f", as {medium}." if task != "style" or not chg else f"The target video is in {style}.")
    sounds = d.get("sounds") or "the sounds of the scene"
    out = "\n\n".join([
        "subject_definitions:\n" + defs,
        "summary:\n" + summary,
        "retention_analysis:\n" + keep,
        "detailed_description:\n" + style_line + split_line + "\n\n" + shot,
        "overall_soundscape:\n" + sounds[0].upper() + sounds[1:] + ".",
        "non_diegetic_music:\nNone.",
    ])
    out = scrub_meta(out)
    _DESCRIBE_CACHE[key] = out
    return out


RESERVED_SLOTS = ("layout", "setting", "target", "details", "details2", "panel_region", "video_region", "split_line", "wider")
_SLOT = re.compile(r"\{([^{}\n]{2,200})\}")


def template_slots(text: str) -> list[str]:
    """The {...} fields of a hand-written prompt template that a VLM should fill ({layout}, {setting}, {target} and
    {details} are filled by the nodes)."""
    return [m for m in _SLOT.findall(text or "") if m.strip().lower() not in RESERVED_SLOTS]


def fill_duet_template(clip, template: str, frames: list, refs: list, task: str, change: str, max_tokens: int = 1024,
                       setting: torch.Tensor | None = None, times: list | None = None) -> str:
    """A hand-written duet prompt with {...} fields (TSC's template) filled by the VLM. The facts are gathered with
    separate questions (references, place, clip), then a text-only pass fills the fields and keeps every other word."""
    refs = [r for r in refs if r is not None]
    fr = [f for f in frames if f is not None]
    key = ("fill2", tuple(times or ()), template, tuple(_image_key(f) for f in fr), tuple(_image_key(r) for r in refs), task, change,
           id(clip), int(max_tokens), _image_key(setting))
    if key in _DESCRIBE_CACHE:
        return _DESCRIBE_CACHE[key]
    slots = template_slots(template)
    if not slots:
        return template
    _status("Filling the prompt template with the VLM", force=True)
    facts = []
    if refs:
        pics = " and ".join(f"<Picture {k + 1}>" for k in range(len(refs)))
        look = vlm_describe(clip, refs, "Describe the subject of these reference pictures (a person, creature, robot or "
                            "animal) in concrete seen words: what it is, apparent gender and age when it has them, skin, "
                            "face shape, eye shape and colour, brows, nose, lips, hair colour, length and cut, and the "
                            "clothing piece by piece with colours. Looks only: never the pose, camera or background.",
                            max_tokens).strip()
        facts.append(f"The new character ({pics}): {look}")
    place = describe_place(clip, setting)
    if place:
        facts.append(f"The place ({{setting}}): {place}")
    for k, v in video_facts(clip, fr, times, max_tokens).items():
        facts.append(f"The source clip, {k}: {v}")
    if change.strip():
        facts.append(f"The requested change: {change.strip()}")
    fields = "\n".join(f"- {{{x}}}" for x in dict.fromkeys(slots))
    ask = ("Fill a video prompt template. Replace every field written in curly braces with concrete seen words taken "
           "from the facts below, matching what the field asks for, and keep every other character of the template "
           "exactly as it is (section names, tags like <Subject 1>, <Picture 1>, {setting}, {layout}, {panel_region}, "
           "{video_region}, line breaks). The prompt describes one finished split-screen video as plain footage. Rules: "
           "a field about what the source shows gets one short clause of what is seen there (who, doing what, where); "
           "the new character's looks come only from the character facts; spell out the motion beat by beat from the "
           "actions, keeping their times; the soundscape is the sounds of the new scene; never write kept footage, "
           "generated, preserved, kept exactly, in sync with, seamlessly; no negations, no quality words; write 'they' "
           "for a character with no clear gender. "
           "If a field is an optional extra shot that the facts do not support, delete that [Shot n] paragraph. "
           f"Task: {task}.\n\nFacts:\n" + "\n".join(facts) + f"\n\nFields to fill:\n{fields}\n\nTemplate:\n"
           + template + "\n\nReply with the filled template only.")
    out = scrub_meta(_clean_written(vlm_generate(clip, ask, None, max(int(max_tokens), 1536))))
    if "subject_definitions:" not in out or len(out) < len(template) * 0.5:
        print("[BFS] the VLM did not return the template; the fields stay as written", flush=True)
        out = template
    _DESCRIBE_CACHE[key] = out
    return out


def duet_prompt(vlm, prompt: str, frames: torch.Tensor | None, refs: list, task: str, instruction: str,
                max_tokens: int = 1024, mode: str = "canvas", setting: torch.Tensor | None = None,
                fps: float = 24.0, start: int = 0) -> str:
    """The prompt a duet node uses. Without a VLM: the prompt as written ('' = the node's template). With one:
    a prompt with {...} fields gets them filled from the clip and the pictures; an empty prompt is written from the
    task and the instruction; a prompt without fields is used as written."""
    text = (prompt or "").strip()
    if vlm is None or frames is None:
        if text and template_slots(text):
            print(f"[BFS H3 Duet] the prompt still has fields to fill ({', '.join(template_slots(text)[:3])}...): connect "
                  "a VLM to fill them, or write them in", flush=True)
        return text
    idx = sorted(set(int(round(x)) for x in torch.linspace(0, frames.shape[0] - 1, min(6, frames.shape[0])).tolist()))
    fr = [frames[i:i + 1, ..., :3].float() for i in idx]
    times = [(start + i) / float(fps) for i in idx]   # the video's clock: the pinned clip starts at frame `start`
    if text and template_slots(text):
        return fill_duet_template(vlm, text, fr, refs, task, instruction, max_tokens, setting, times)
    if not text:
        return write_duet_prompt(vlm, fr, refs, task, instruction, max_tokens, mode, setting, times)
    return text


# ---------------------------------------------------------------------------- continuity between shots

CHAIN_MODES = ("off", "reference", "first frame", "history", "head")
try:
    from .bfs_source_blur import EFFECTS as BLUR_EFFECTS, SEGMENTERS as BLUR_SEGMENTERS, SOURCE_BLUR, blur_source
except ImportError:
    from bfs_source_blur import EFFECTS as BLUR_EFFECTS, SEGMENTERS as BLUR_SEGMENTERS, SOURCE_BLUR, blur_source
_LAST_RESULT: dict[str, Any] = {}   # auto loop: the last shot a render node produced (index, count, frames)


def pick_frame(frames: torch.Tensor, length: int, which: str) -> torch.Tensor:
    """One frame [1,H,W,3] of a shot's result: its first, middle or last real frame (not the overlap)."""
    n = max(1, min(int(length), frames.shape[0]))
    i = {"first": 0, "middle": n // 2}.get(which, n - 1)
    return frames[i:i + 1, ..., :3].float()


def remember_result(shot: dict, frames: torch.Tensor) -> None:
    _LAST_RESULT.clear()
    frames = frames[int(shot.get("head") or 0):]   # the head is the previous shot's end, not this shot
    _LAST_RESULT.update(index=int(shot.get("index", 0)), count=int(shot.get("count", 0)), frames=frames.detach().cpu())


def chain_image(shot: dict) -> torch.Tensor | None:
    """The previous shot's result frame this shot continues from, if it asked for one and it exists."""
    if shot.get("chain", "off") == "off" or int(shot.get("index", 0)) == 0:
        return None
    if shot.get("chain_image") is not None:
        return shot["chain_image"]
    if _LAST_RESULT.get("index") == int(shot["index"]) - 1 and _LAST_RESULT.get("count") == int(shot.get("count", 0)):
        prev_len = shot.get("prev_length") or _LAST_RESULT["frames"].shape[0]
        return pick_frame(_LAST_RESULT["frames"], prev_len, shot.get("chain_frame", "first"))
    return None


def history_frame_count(wanted: int, available: int) -> int:
    """Return a valid H3 history window (17k+5 frames: 5, 22, 39, 56); see bfs_h3_history.history_length."""
    try:
        from .bfs_h3_history import history_length
    except ImportError:
        from bfs_h3_history import history_length
    return history_length(wanted, available)


def history_for(path, analysis, src, seg, i, prev_len, queue, rid, W, H, crop, ft):
    """Continuity 'history': the end of the previous shot's result (5/22/39/56 frames) at this shot's size, plus the
    source frames under it (the duet panel shows them). Same take only: a camera cut has nothing to carry over."""
    tag = f"[BFS Shot Planner] shot {i + 1}: history"
    if seg["cut_before"]:
        print(f"{tag} skipped: a camera cut starts this shot", flush=True)
        return None, None
    want = int(seg.get("chain_history") or 22)
    prev = None
    if queue:
        if os.path.exists(os.path.join(run_dir(rid), f"shot_{i - 1:04d}.pt")):
            prev = load_shot(run_dir(rid), i - 1)[0]     # uint8, sliced before any float copy
    if prev is None:
        if queue:
            print(f"{tag}: the previous shot's result is not there yet (shots run in order)", flush=True)
        return None, None
    avail = min(int(prev_len), prev.shape[0])
    n = history_frame_count(want, avail)
    if n == 0:
        need = 5
        print(f"{tag} skipped: the previous shot has only {avail} frames (needs {need})", flush=True)
        return None, None
    hf = prev[avail - n:avail, ..., :3].float() / 255.0
    del prev
    lo = max(0, int(seg["start"]) - n)
    sf = _read_frames(path, src[np.arange(lo, int(seg["start"]))], (W, H))
    hs = torch.from_numpy(np.stack(sf)).float() / 255.0
    del sf
    if hs.shape[0] < n:
        hs = torch.cat([hs[:1].expand(n - hs.shape[0], -1, -1, -1), hs], 0)
    if hf.shape[1:3] != (H, W):
        hf = _fit_to(hf, W, H)
    if crop is not None:   # the same box as this shot's crop, at the crop's generation size
        x0, y0, x1, y1 = box_px(crop["box"], W, H)
        hf = _fit_to(hf[:, y0:y1, x0:x1], ft.shape[2], ft.shape[1])
        hs = _fit_to(hs[:, y0:y1, x0:x1], ft.shape[2], ft.shape[1])
    print(f"{tag}: the last {n} frames of shot {i} go before this one on H3's timeline", flush=True)
    # kept as uint8 on the CPU until the conditioning encodes them (4x less RAM per shot)
    return _to_u8(hf), _to_u8(hs)


def shot_history(shot: dict):
    """(history, source under it) of a shot with continuity 'history', as float frames, taken OUT of the shot so nothing
    stays allocated afterwards; auto loop: from the previous render kept in memory. (None, None) when there is none."""
    if shot.get("chain") != "history" or int(shot.get("index", 0)) == 0:
        return None, None
    hist, hsrc = _from_u8(shot.pop("history", None)), _from_u8(shot.pop("history_src", None))
    if hist is None and not shot.get("crop") and _LAST_RESULT.get("index") == int(shot["index"]) - 1 \
            and _LAST_RESULT.get("count") == int(shot.get("count", 0)):
        prev_len = int(shot.get("prev_length") or _LAST_RESULT["frames"].shape[0])
        f = _LAST_RESULT["frames"][:prev_len]
        n = history_frame_count(int(shot.get("chain_history") or 22), f.shape[0])
        if n:
            hist = f[f.shape[0] - n:].float()
            hist = hist / 255.0 if hist.max() > 1.5 else hist
    return hist, hsrc


def _release_between_shots(free: bool = False, when: str = "between shots") -> None:
    """Between two stored shots: drop what the stored shot no longer needs. free (plan 'Between shots = free memory',
    for low VRAM): also unload every model, so the next shot starts from an empty GPU (they load again, slower).
    Also called before the join (plan 'Unload models before the join'): every shot is rendered, the models are no
    longer needed and the join gets their VRAM and RAM."""
    import gc
    try:
        import comfy.model_management as mm
        if free:
            mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        if free:
            print(f"[BFS Shot Join] {when}: models unloaded and memory freed", flush=True)
    except Exception:
        gc.collect()


# ---------------------------------------------------------------------------- memory status (learned on this GPU)
# There is no honest fixed "GB per frame" for H3 (ComfyUI offloads weights as needed), so the panel compares each shot
# with the biggest shot that already ran on this GPU: a shot's size is its latent tokens (generated frames x
# resolution); every shot the join receives was generated, so it is recorded with the peak VRAM of its run.
_VRAM_FILE = "bfs_shotloop_vram.json"


def shot_tokens(gen_frames: int, width: int, height: int, grid: str = DEFAULT_GRID) -> int:
    """Latent tokens of one generated shot: H3 latent steps (17 frames -> 5, minus the 3 the VAE drops) x 16 px cells;
    other grids: ~4 frames per step. A size to compare shots, not an exact memory figure."""
    g = max(1, int(gen_frames))
    steps = 5 * math.ceil(g / 17) - 3 if GRIDS.get(grid) == GRIDS[DEFAULT_GRID] else math.ceil(g / 4)
    return max(1, steps) * math.ceil(int(height) / 16) * math.ceil(int(width) / 16)


def _vram_path() -> str:
    user = getattr(folder_paths, "get_user_directory", None)
    return os.path.join(user() if user else folder_paths.get_temp_directory(), _VRAM_FILE)


def _cuda_on() -> bool:
    """CUDA in use by this process (always, inside ComfyUI); never starts it just for the status."""
    return torch.cuda.is_available() and torch.cuda.is_initialized()


def _gpu_name() -> str:
    return torch.cuda.get_device_name(torch.cuda.current_device()) if _cuda_on() else ""


def vram_record(tokens: int) -> None:
    """The biggest shot of a join that just ran, with the peak VRAM since the last record (then reset)."""
    if not _cuda_on() or tokens <= 0:
        return
    try:
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        torch.cuda.reset_peak_memory_stats()
        path = _vram_path()
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        g = data.setdefault(_gpu_name(), {"max_tokens": 0, "max_peak_gb": 0.0, "samples": []})
        if tokens >= g["max_tokens"]:
            g["max_tokens"], g["max_peak_gb"] = int(tokens), round(peak, 2)
        g["samples"] = (g["samples"] + [{"tokens": int(tokens), "peak_gb": round(peak, 2), "t": int(time.time())}])[-40:]
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:   # the status is optional
        print(f"[BFS Shot Join] memory status not recorded: {e}", flush=True)


def vram_status() -> dict:
    out = {"gpu": _gpu_name(), "total_gb": 0.0, "free_gb": 0.0, "max_tokens": 0, "max_peak_gb": 0.0, "runs": 0,
           "ram_free_gb": round(_ram_available() / 2 ** 30, 1)}
    if _cuda_on():
        free, total = torch.cuda.mem_get_info()
        out.update(total_gb=round(total / 2 ** 30, 1), free_gb=round(free / 2 ** 30, 1))
    try:
        with open(_vram_path()) as f:
            g = json.load(f).get(out["gpu"]) or {}
        out.update(max_tokens=int(g.get("max_tokens", 0)), max_peak_gb=float(g.get("max_peak_gb", 0)),
                   runs=len(g.get("samples", [])))
    except (OSError, ValueError):
        pass
    return out


def head_length(segs: list, i: int, grid: str) -> int:
    """Continuity 'head': how many frames of the previous shot's end start this shot (5/22/39), 0 when it cannot:
    first shot, a camera cut, a skipped shot in between, or a grid other than H3's."""
    s = segs[i]
    if s.get("chain") != "head" or i == 0 or s.get("cut_before") or GRIDS.get(grid) != GRIDS[DEFAULT_GRID]:
        return 0
    prev = segs[i - 1]
    if prev["end"] != s["start"]:
        return 0
    return history_frame_count(min(int(s.get("chain_history") or 22), 39), prev["end"] - prev["start"])


def head_for(i, n, prev_len, queue, rid, W, H, crop, ft):
    """Continuity 'head': the last `n` delivered frames of the previous shot's result, cut like this shot (its crop box,
    its generation size), as uint8. None when the result is not there (the head is then generated from the guide)."""
    tag = f"[BFS Shot Planner] shot {i + 1}: head"
    fp = os.path.join(run_dir(rid), f"shot_{i - 1:04d}.pt") if queue else ""
    if not fp or not os.path.exists(fp):
        if queue:
            print(f"{tag}: the previous shot's result is not there yet (shots run in order)", flush=True)
        return None
    prev = load_shot(run_dir(rid), i - 1)[0]
    avail = min(int(prev_len), prev.shape[0])
    if avail < n:
        print(f"{tag} skipped: the previous shot has only {avail} frames", flush=True)
        return None
    hf = prev[avail - n:avail, ..., :3].float() / 255.0
    del prev
    if crop is not None:
        x0, y0, x1, y1 = box_px(crop["box"], W, H)
        hf = _fit_to(_fit_to(hf, W, H)[:, y0:y1, x0:x1], ft.shape[2], ft.shape[1])
    elif tuple(hf.shape[1:3]) != tuple(ft.shape[1:3]):
        hf = _fit_to(hf, ft.shape[2], ft.shape[1])
    print(f"{tag}: the last {n} frames of shot {i} start this shot, kept as they are", flush=True)
    return _to_u8(hf)


def shot_head(shot: dict):
    """The previous shot's end that starts this shot (continuity 'head'), as float frames, taken OUT of the shot; auto
    loop: from the previous render kept in memory (not with a crop). None when there is none."""
    n = int(shot.get("head") or 0)
    if shot.get("chain") != "head" or not n:
        return None
    hf = _from_u8(shot.pop("head_frames", None))
    if hf is None and not shot.get("crop") and _LAST_RESULT.get("index") == int(shot["index"]) - 1 \
            and _LAST_RESULT.get("count") == int(shot.get("count", 0)):
        f = _LAST_RESULT["frames"][:int(shot.get("prev_length") or _LAST_RESULT["frames"].shape[0])]
        if f.shape[0] >= n:
            hf = f[f.shape[0] - n:].float()
            hf = hf / 255.0 if hf.max() > 1.5 else hf
    return hf


def drop_head(img: torch.Tensor, shot: dict):
    """A generated shot without its continuity head (the frames that are the previous shot's end), and the shot's
    source frames cut the same way, so the rest of the join sees the shot as if it started at its own first frame."""
    n = int(shot.get("head") or 0)
    if not n:
        return img, shot
    sh = dict(shot, head=0, gen_length=int(shot.get("gen_length") or img.shape[0]) - n)
    for k in ("frames", "full_frames"):
        if isinstance(sh.get(k), torch.Tensor) and sh[k].shape[0] > n:
            sh[k] = sh[k][n:]
    if sh.get("crop"):   # the comparison draws the crop's masks frame by frame
        sh["crop"] = dict(sh["crop"], **{k: v[n:] for k, v in sh["crop"].items()
                                         if k in ("mask", "crop_mask") and isinstance(v, torch.Tensor) and v.shape[0] > n})
    sh["mask_skip"] = n   # masks segmented on the whole window (mask_src) still start with the head
    return img[n:], sh


def _skip_head(masks: torch.Tensor, shot: dict) -> torch.Tensor:
    n = int(shot.get("mask_skip") or 0)
    return masks[n:] if n and masks.ndim == 3 and masks.shape[0] > n else masks


def fill_details_now(text: str, vlm, refs: list, tag: str = "", names: list | None = None) -> str:
    """{details}, {details2} and {details@Picture k} left by the planner, filled at render time (this node's VLM, else
    'character')."""
    refs = [r for r in refs if r is not None]
    return fill_picture_details(picture_markers(text), names or [], refs, {}, vlm, vlm_cfg({}), tag)


def _to_u8(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


def _from_u8(x: torch.Tensor | None) -> torch.Tensor | None:
    return None if x is None else (x.float() / 255.0 if x.dtype == torch.uint8 else x.float())


def _fit_to(img: torch.Tensor, w: int, h: int) -> torch.Tensor:
    x = img[..., :3].movedim(-1, 1).float()
    return torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1)


# ---------------------------------------------------------------------------- queue loop state

def run_id_for(plan_json: str, path: str) -> str:
    """Old run id (the whole plan): kept for older callers; runs now use run_key."""
    h = hashlib.sha1((plan_json + str(os.path.getmtime(path))).encode()).hexdigest()[:16]
    return h


def run_key(plan: dict, path: str) -> str:
    """The stored shots of a video: one run folder per source video (its path and date), whatever else changes in the
    plan; each shot carries the signature of its own settings, so an edit reruns only the shots it changes (and the
    ones that continue from them). 'tmp_': kept in ComfyUI's temp folder (Shot cache = temp)."""
    h = hashlib.sha1((os.path.abspath(path) + str(os.path.getmtime(path))).encode()).hexdigest()[:16]
    return ("tmp_" if plan.get("shot_cache") == "temp" else "") + h


def run_dir(run_id: str, create: bool = True) -> str:
    """The stored shots of a run: in ComfyUI's output folder (output/bfs_shotloop), because ComfyUI empties its temp
    folder at every start; 'tmp_' runs (Shot cache = temp) stay in temp and are lost on restart."""
    out = getattr(folder_paths, "get_output_directory", None)
    base = folder_paths.get_temp_directory() if run_id.startswith("tmp_") or out is None else out()
    d = os.path.join(base, "bfs_shotloop", run_id)
    if create:
        os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------- shot cache (signatures, redo)
_SIG_SKIP = ("stats", "people", "main", "cast_skip", "skip_reason", "run", "enabled", "force", "refs",
             "color_guide_marks")
_SIG_PLAN = ("grid", "fps", "megapixels", "generation_size_mode", "custom_width", "custom_height", "multiple",
             "global_prompt", "global_ref", "global_ref2", "global_color_guide_ref1", "global_color_guide_ref2",
             "mask_cfg", "vlm_cfg", "ref_details", "audio_mode", "mask_video")


def _tensor_sig(t) -> str:
    if t is None:
        return ""
    x = t.detach().cpu()
    return hashlib.sha1(str(tuple(x.shape)).encode() + x[:, ::8, ::8].float().contiguous().numpy().tobytes()).hexdigest()[:12]


def shot_signature(plan: dict, seg: dict, head: int, extra: dict) -> str:
    """What a stored shot was made from: its own settings (range, prompt, references, mask, crop, continuity...), the
    plan's shared ones (size, grid, global prompt / references), the node inputs (references and prompt connected to
    the planner) and, in extra["render"], what renders it (render_fingerprint: model, LoRAs, seed, sampler,
    conditioning settings)."""
    doc = {"s": {k: v for k, v in seg.items() if k not in _SIG_SKIP}, "head": int(head),
           "g": {k: plan.get(k) for k in _SIG_PLAN}, "x": extra}
    # pictures 3+ and per-picture descriptions only count when used: shots stored before them keep their signature
    if len(seg.get("refs") or []) > 2:
        doc["s"]["refs_more"] = seg["refs"][2:]
    if seg.get("color_guide"):
        doc["s"]["color_guide_marks"] = seg.get("color_guide_marks")
    if len(plan.get("global_refs") or []) > 2:
        doc["g"]["global_refs_more"] = plan["global_refs"][2:]
    for k in ("picture_details", "picture_cfg"):
        if plan.get(k):
            doc["g"][k] = plan[k]
    if (plan.get("auto_prompt") or {}).get("enabled"):   # the shot's written prompt is its prompt
        doc["g"]["auto_prompt"] = plan["auto_prompt"]
        doc["s"]["written"] = ((plan.get("auto_prompts") or {}).get(f"{seg.get('start')}-{seg.get('end')}") or {}).get("text", "")
    return hashlib.sha1(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:16]


def render_fingerprint(graph: dict | None, planner_id=None) -> str:
    """What renders the shots, read from the queued workflow: every node upstream of BFS Shot Join's `images` (model
    and LoRA loaders, sampler, seed, conditioning settings), with links reduced to the class of the node they come
    from. The planner and its unpack are left out (the shot signature covers them), and so is the join itself, so its
    crossfade or comparison never count as a change, while a new LoRA, strength or seed does ("" without a join)."""
    if not isinstance(graph, dict):
        return ""
    stack = []
    for n in graph.values():
        if isinstance(n, dict) and n.get("class_type") == "BFSShotJoin":
            src = (n.get("inputs") or {}).get("images")
            if isinstance(src, list) and src:
                stack.append(str(src[0]))
    seen, nodes = set(), {}
    while stack:
        k = stack.pop()
        if k in seen or k not in graph or k == str(planner_id):
            continue
        seen.add(k)
        n = graph[k]
        if n.get("class_type") in ("BFSShotPlanner", "BFSShotUnpack"):
            continue
        ins = {}
        for name, v in (n.get("inputs") or {}).items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int)) and isinstance(v[1], int):
                stack.append(str(v[0]))
                ins[name] = ["@", (graph.get(str(v[0])) or {}).get("class_type"), v[1]]
            else:
                ins[name] = v
        nodes[k] = [n.get("class_type"), ins]
    if not nodes:
        return ""
    return hashlib.sha1(json.dumps(nodes, sort_keys=True, default=str).encode()).hexdigest()[:16]


def uses_previous(seg: dict) -> bool:
    """Whether a shot is made from the previous shot's result (so it is redone when that result changes)."""
    c = seg.get("chain") or "off"
    return c in ("reference", "first frame") or (c in ("head", "history") and not seg.get("cut_before"))


def shot_status(st: dict, d: str) -> list[bool]:
    """Per shot of the last plan the planner saw (st['cur']): stored and still valid. A shot is valid when it was
    stored with the same signature and, if it continues from the previous shot, from that shot's current take."""
    cur = st.get("cur") or {}
    sigs, prev = cur.get("sigs") or [], cur.get("prev") or []
    done, sig, gen, frm = set(st.get("done", [])), st.get("sig", {}), st.get("gen", {}), st.get("from", {})
    ok: list[bool] = []
    for i, sg in enumerate(sigs):
        good = i in done and sig.get(str(i)) == sg and os.path.exists(os.path.join(d, f"shot_{i:04d}.pt"))
        if good and i > 0 and i < len(prev) and prev[i]:
            good = ok[i - 1] and frm.get(str(i)) == gen.get(str(i - 1))
        ok.append(good)
    return ok


# a stored shot: shot_NNNN.pt (its info, small) + shot_NNNN.mkv (its frames), in the format of the plan's "Shot
# storage" (measured on 145 frames at 1280x720: raw 401 MB). The final video of a stored run is assembled from these
# files, so only the lossy formats can change it; raw frames go inside the .pt (also when ffmpeg cannot write a video).
SHOT_STORAGE = {
    "lossless": ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb24"],                         # exact, ~34 MB
    "smaller": ["-c:v", "libx264rgb", "-preset", "fast", "-crf", "8", "-pix_fmt", "rgb24"],     # PSNR ~52 dB, ~21 MB
    "smallest": ["-c:v", "libx264rgb", "-preset", "fast", "-crf", "14", "-pix_fmt", "rgb24"],   # PSNR ~48 dB, ~9 MB
    "raw": None,                                                                                # exact, no ffmpeg
}


def _write_rgb_video(path: str, frames: torch.Tensor, fps: float = 24.0, storage: str = "lossless") -> bool:
    """uint8 [N,H,W,3] -> a video file in the storage's codec (lossless FFV1 when that codec is missing); False when
    no video can be written (raw storage, ffmpeg missing...)."""
    codec = SHOT_STORAGE.get(storage, SHOT_STORAGE["lossless"])
    if codec is None:
        return False
    n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    data = frames.contiguous().numpy()
    for c in (codec, SHOT_STORAGE["lossless"]) if codec is not SHOT_STORAGE["lossless"] else (codec,):
        try:
            r = subprocess.run([VIO.ffmpeg_bin() or "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                "-r", f"{fps:g}", "-i", "-", *c, path], input=memoryview(data), capture_output=True)
            if r.returncode == 0 and os.path.getsize(path) > 0:
                return True
        except (OSError, ValueError):
            pass
    if os.path.exists(path):
        os.remove(path)
    return False


def _read_rgb_video(path: str, n: int, h: int, w: int) -> torch.Tensor:
    r = subprocess.run([VIO.ffmpeg_bin() or "ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True)
    k = len(r.stdout) // (h * w * 3)
    if k < n:
        raise ValueError(f"BFS Shot Loop: {path} holds {k} frames, expected {n}")
    return torch.from_numpy(np.frombuffer(r.stdout, np.uint8, count=n * h * w * 3).reshape(n, h, w, 3).copy())


def store_shot(d: str, index: int, frames: torch.Tensor, meta: dict, storage: str = "lossless") -> None:
    """Store one generated shot (uint8 [N,H,W,3]) of a run. The source frames kept for the comparison's original
    column (meta['frames']) go to shot_NNNN_orig.mkv (visually lossless: they are only shown), not into the .pt."""
    pt, vid = os.path.join(d, f"shot_{index:04d}.pt"), os.path.join(d, f"shot_{index:04d}.mkv")
    orig = meta.get("frames")
    if isinstance(orig, torch.Tensor) and orig.dtype == torch.uint8 and orig.shape[0]:
        ov = os.path.join(d, f"shot_{index:04d}_orig.mkv")
        if _write_rgb_video(ov, orig, float(meta.get("fps") or 24.0), "smaller"):
            meta = {k: v for k, v in meta.items() if k != "frames"}
            meta["frames_video"] = [os.path.basename(ov), [int(x) for x in orig.shape]]
    if _write_rgb_video(vid, frames, float(meta.get("fps") or 24.0), storage):
        torch.save({"shot": meta, "video": os.path.basename(vid), "shape": tuple(int(x) for x in frames.shape)}, pt)
    else:
        torch.save({"frames": frames, "shot": meta}, pt)


def load_shot(d: str, index: int, frames: bool = True):
    """(uint8 frames or None, info) of a stored shot."""
    x = torch.load(os.path.join(d, f"shot_{index:04d}.pt"), map_location="cpu", weights_only=False)
    if not frames:
        return None, x["shot"]
    if "frames" in x:
        return x["frames"], x["shot"]
    n, h, w, _ = x["shape"]
    return _read_rgb_video(os.path.join(d, x["video"]), n, h, w), x["shot"]


# ---------------------------------------------------------------------------- color anchor (continuity drift)
# A shot that starts on the previous shot's result (head / history) comes out a little more saturated right after it,
# and the next shot inherits that: over a long take the new subject's colours keep getting stronger (measured +66%
# saturation by the 4th shot). The join corrects the subject's chroma of every continuing shot to the first shot of the
# take, and stores the corrected shot, so the next one starts from the anchored colours and nothing accumulates.

def _subject_weight(res: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Where the generated subject is (0..1, soft edge): the pixels that differ clearly from the source video."""
    import cv2
    d = np.abs(res - src).mean(-1)
    m = (d > 0.12).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3), 0, 1)


def _to_np(frames: torch.Tensor) -> np.ndarray:
    x = frames[..., :3]
    return (x.float() / 255.0 if x.dtype == torch.uint8 else x.float()).clamp(0, 1).cpu().numpy()


def color_stats(img: torch.Tensor, src: torch.Tensor, length: int | None = None) -> dict | None:
    """Mean / std of the subject's a, b (CIELAB) over a shot's frames, or None when there is hardly any subject."""
    import cv2
    n = min(int(length or img.shape[0]), img.shape[0], src.shape[0])
    ab, tot = [], 0
    for i in range(0, n, 3):
        r, s_ = _to_np(img[i]), _to_np(src[i])
        if s_.shape != r.shape:
            s_ = cv2.resize(s_, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)
        w = _subject_weight(r, s_) > 0.5
        tot += w.size
        if w.any():
            ab.append(cv2.cvtColor(r, cv2.COLOR_RGB2LAB)[..., 1:][w])
    if not ab or sum(len(x) for x in ab) < 0.005 * tot:
        return None
    ab = np.concatenate(ab, 0)
    return {"mu": [float(v) for v in ab.mean(0)], "sd": [float(v) for v in ab.std(0) + 1e-3], "n": int(len(ab))}


def anchor_colors(img: torch.Tensor, src: torch.Tensor, anchor: dict, length: int | None = None,
                  strength: float = 1.0) -> tuple[torch.Tensor, dict | None]:
    """The shot with its subject's a, b moved to the anchor's statistics (one transform for the whole shot, so no
    flicker; the background and the lightness are left alone). Returns (frames, the shot's own stats before)."""
    import cv2
    st = color_stats(img, src, length)
    if st is None or not anchor:
        return img, st
    mu, sd = np.array(st["mu"]), np.array(st["sd"])
    k = np.clip(np.array(anchor["sd"]) / sd, 0.5, 2.0)
    out = img.clone() if img.dtype != torch.uint8 else img.float() / 255.0
    for i in range(img.shape[0]):
        r = _to_np(img[i])
        s_ = _to_np(src[min(i, src.shape[0] - 1)])
        if s_.shape != r.shape:
            s_ = cv2.resize(s_, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)
        w = _subject_weight(r, s_)[..., None] * float(strength)
        lab = cv2.cvtColor(r, cv2.COLOR_RGB2LAB)
        ab = lab[..., 1:]
        lab[..., 1:] = ab + w * (((ab - mu) * k + np.array(anchor["mu"])) - ab)
        out[i, ..., :3] = torch.from_numpy(np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1))
    return out, st


def shot_originals(d: str, meta: dict):
    """The comparison's original frames of a stored shot (uint8), or None."""
    if isinstance(meta.get("frames"), torch.Tensor):
        return meta["frames"]
    fv = meta.get("frames_video")
    if not fv or not os.path.exists(os.path.join(d, fv[0])):
        return None
    n, h, w, _ = fv[1]
    return _read_rgb_video(os.path.join(d, fv[0]), n, h, w)


def shot_source_frames(tl: dict, start: int, length: int, H: int, W: int, comp_h: int):
    """The original frames under a shot (uint8, at the comparison height), read from the timeline's source video."""
    try:
        src = np.asarray(tl["src"])
        idx = np.clip(np.arange(start, start + max(1, length)), 0, len(src) - 1)
        h = min(H, comp_h or H)
        w = max(16, int(round(W * h / H)))
        return torch.from_numpy(np.stack(_read_frames(tl["path"], src[idx], (w, h))))
    except Exception as e:   # the comparison is optional: the video still comes out
        print(f"[BFS Shot Join] no original column for the shot at frame {start}: {e}", flush=True)
        return None


def run_state(run_id: str) -> dict:
    f = os.path.join(run_dir(run_id), "state.json")
    if os.path.isfile(f):
        with open(f) as fh:
            return json.load(fh)
    return {"done": [], "count": 0}


def save_state(run_id: str, st: dict) -> None:
    with open(os.path.join(run_dir(run_id), "state.json"), "w") as fh:
        json.dump(st, fh)


def _notify(event: str, data: dict) -> None:
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(event, data)
    except Exception:  # noqa: BLE001 - headless runs have no UI to notify
        pass


# ---------------------------------------------------------------------------- images

def _load_image(name: str) -> torch.Tensor | None:
    if not name:
        return None
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(_input_path(name))).convert("RGB")
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)[None]


def _grey(h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.full((1, h, w, 3), 0.5)


# ---------------------------------------------------------------------------- nodes

class BFSShotPlanner:
    """Split a video into shots that fit the model, with a reference and prompt per shot."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("STRING", {"default": json.dumps(DEFAULT_PLAN), "multiline": True,
                                    "tooltip": "The panel writes this. JSON with the video, split settings, "
                                               "boundaries and the per-shot reference and prompt."}),
            },
            "optional": {
                "prompt": ("STRING", {"forceInput": True,
                                      "tooltip": "Default prompt for shots without their own "
                                                 "(overrides the panel's global prompt)."}),
                "mask": ("MASK,IMAGE", {"tooltip": "Optional mask of the WHOLE video (white = the subject), e.g. "
                                    "rotoscoped in another tool and loaded with Load Video (+ Convert Image to Mask). "
                                    "Replaces SAM 3 for every shot without its own mask video / text / points. One mask per source "
                                    "frame; a different count is spread over the video."}),
                "vlm": ("CLIP", {"tooltip": "Optional vision-language model (CLIPLoader with qwen3vl_4b / qwen3vl_8b). "
                                            "It looks at every shot and suggests what to segment, a description of the "
                                            "shot (fills {shot} in the prompt) and whether to run it. Settings in the "
                                            "panel's VLM card."}),
            },
            # the queued workflow: what renders the shots (LoRA, seed, sampler...) is part of a stored shot's signature
            "hidden": {"graph": "PROMPT", "node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("BFS_SHOT", "INT", "FLOAT", "INT", "INT", "AUDIO", "STRING", "BFS_SHOT_TIMELINE", "IMAGE", "INT",
                    "BOOLEAN", "INT", "INT")
    # refs and ref_count sit where ref_image and ref_image_2 were (1.113), so the outputs after them keep their slots
    RETURN_NAMES = ("shots", "count", "fps", "width", "height", "audio", "summary", "timeline", "refs", "ref_count",
                    "second_pass", "upscale_width", "upscale_height")
    OUTPUT_IS_LIST = (True, False, False, False, False, False, False, False, True, False, False, False, False)
    OUTPUT_TOOLTIPS = (
        "One item per shot. Every node that receives this list runs once per shot; connect it to "
        "BFS Shot Unpack or BFS Shot H3 Conditioning, sample, decode, then BFS Shot Join.",
        "Number of shots that will run.", "Timeline frame rate.", "Generation width.",
        "Generation height.", "The whole soundtrack, trimmed to the planned duration.",
        "Human-readable plan.",
        "Every shot in order, including the ones that do not run (disabled or filtered out). Connect it "
        "to BFS Shot Join so skipped shots are filled with the original video (or dropped).",
        "Every reference picture the shots use, without repeats, <Picture 1> first (a grey image when there is none).",
        "How many different reference pictures the shots use.",
        "Run tab > Second pass: on or off. Use it to switch the upscale branch (latent upscale + second sampler).",
        "Second pass width (the generation width upscaled as set in the Run tab, on the latent grid).",
        "Second pass height.")
    FUNCTION = "plan_shots"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Split a long video into model-sized shots (at camera cuts, fixed length, or by hand), "
                   "give every shot its own reference image and prompt, and run the rest of the graph once "
                   "per shot. Join the decoded shots with BFS Shot Join.")

    @classmethod
    def IS_CHANGED(cls, plan, **kwargs):
        try:
            if _load_plan(plan).get("run") in STORED_RUNS:
                return float("nan")   # the next shot depends on the loop state on disk
        except ValueError:
            pass
        return plan

    def plan_shots(self, plan, prompt=None, vlm=None, mask=None, graph=None, node_id=None, ref_image=None,
                   ref_image_2=None):
        p = _load_plan(plan)
        if ref_image is not None or ref_image_2 is not None:
            # the ref_image / ref_image_2 inputs were removed in 1.113 (the references live in the panel, up to
            # 12); a workflow saved with them connected still links them, and they are ignored
            print("[BFS Shot Planner] ref_image / ref_image_2 are no longer planner inputs: the connected images "
                  "are ignored. Set the references in the panel (Prompts & refs tab, up to 12).", flush=True)
        if vlm is not None:
            _VLM["clip"] = vlm       # the panel's Analyse button uses it too
        vcfg = vlm_cfg(p.get("vlm_cfg"))
        use_vlm = vlm is not None and vcfg["enabled"]
        suggestions = []
        path = _input_path(p["video"])
        fps = float(p["fps"])
        a = analyze(path, fps)
        if mask is not None:
            _EXT_MASK[path] = mask.detach().cpu()
        else:
            _EXT_MASK.pop(path, None)
        if p.get("cast") or p.get("cast_split") or p.get("cast_only"):
            analyze_cast(path, a)   # warms the cache so the plan uses the same people as the panel
        all_segs = apply_filters(p, a, path, resolve_plan(p, a, path))
        segs = [s for s in all_segs if s["run"]]
        if not segs:
            raise ValueError("BFS Shot Planner: no shot is left to run (all disabled or filtered out).")
        W, H = plan_generation_size(a["width"], a["height"], p, int(p["multiple"]))
        src = _timeline(a["n_src"], a["fps_src"], fps)
        audio = _read_audio(path) if p.get("audio_mode", "auto") != "silent" else None
        g_prompt = prompt if prompt is not None else p.get("global_prompt", "")
        queue = p.get("run") in STORED_RUNS
        rid = run_key(p, path) if queue else ""
        todo = list(range(len(segs)))
        sigs: list[str] = []
        st: dict = {}
        assemble_only = False   # every shot stored and up to date: nothing is generated, the join only assembles
        if queue:
            # "ref" / "ref2": the removed reference inputs, always empty now (kept so stored shots stay valid)
            extra = {"ref": "", "ref2": "", "prompt": prompt or "",
                     "mask": _tensor_sig(mask) if mask is not None else "", "size": [W, H],
                     "render": render_fingerprint(graph, node_id)}
            sigs = [shot_signature(p, s, head_length(segs, i, p["grid"]), extra) for i, s in enumerate(segs)]
            st = run_state(rid)
            st["count"] = len(segs)
            st["cur"] = {"sigs": sigs, "prev": [bool(i > 0 and uses_previous(s)) for i, s in enumerate(segs)]}
            ok = shot_status(st, run_dir(rid))
            pending = [i for i, g in enumerate(ok) if not g]
            st["pending"] = pending
            save_state(rid, st)
            if pending:
                print(f"[BFS Shot Planner] shot cache: {len(segs) - len(pending)}/{len(segs)} shots stored and up to "
                      f"date, {len(pending)} to run (next: shot {pending[0] + 1})", flush=True)
            else:
                assemble_only = True
                print(f"[BFS Shot Planner] shot cache: every shot is stored and up to date: nothing to generate, "
                      f"BFS Shot Join assembles the stored shots", flush=True)
            todo = pending[:1] if pending else [len(segs) - 1]
        # assemble only: the one shot handed out is a placeholder the join does not render (no VLM, SAM 3 or history)
        work = [] if assemble_only else todo
        cache_imgs: dict[str, torch.Tensor] = {}

        def ref_for(name):
            if not name:
                return None
            if name not in cache_imgs:
                cache_imgs[name] = _load_image(name)
            return cache_imgs[name]

        def names_of(s):
            return shot_ref_names(s["refs"], p["global_refs"])

        def refs_of(s):
            return [ref_for(n) for n in names_of(s)]

        used_names = list(dict.fromkeys(n for s in segs for n in names_of(s)))
        used_refs = [img for img in (ref_for(n) for n in used_names) if img is not None]
        pic_details = saved_picture_details(p)

        def details_template(text, names):
            """The prompt with every description marker as {details@Picture k} ({details} = picture 1, {details2} =
            picture 2); a two-picture shot first gets the <Subject 2> line when its template has one subject only."""
            return picture_markers(expand_two_ref_prompt(text) if len(names) > 1 else text)

        # the VLM answers for every shot first: switching between it and SAM 3 mid-generation breaks the VLM
        vlm_out = {i: vlm_shot(vlm, path, a, segs[i]["start"], segs[i]["end"], vcfg) for i in work} if use_vlm else {}
        # 🤖 Auto prompt: the prompt box is the request, the shot's prompt is the one written for it (in the panel, or
        # now by the VLM for a shot without one, before any SAM 3 work; the panel stores the new ones)
        # the prompt input is then the template (e.g. a LoRA's); the panel's text is the request
        ap = {**AUTO_PROMPT_DEFAULTS, **(p.get("auto_prompt") or {})}
        written, new_written = {}, {}
        if ap["enabled"]:
            stored = p.get("auto_prompts") or {}
            template = prompt or ""
            for i in work:
                key, request = f"{segs[i]['start']}-{segs[i]['end']}", segs[i]["prompt"] or p.get("global_prompt", "")
                got = stored.get(key) or {}
                if got.get("text"):
                    written[i] = got["text"]
                    if got.get("for") != auto_prompt_for(request, names_of(segs[i]), segs[i], template, ap["style"]):
                        print(f"[BFS Shot Planner] shot {i + 1}: its written prompt is older than its request, pictures or "
                              "marks (🤖 Write in the panel to write it again)", flush=True)
                elif vlm is not None:
                    new_written[key] = auto_prompt_shot(vlm, p, a, path, segs[i], names_of(segs[i]), refs_of(segs[i]),
                                                        pic_details, request, vcfg, template)
                    written[i] = new_written[key]["text"]
                    print(f"[BFS Shot Planner] shot {i + 1}: prompt written by the VLM ({len(written[i].split())} words)",
                          flush=True)
                else:
                    written[i] = template or request
                    print(f"[BFS Shot Planner] shot {i + 1}: Auto prompt is on, but the shot has no written prompt and the "
                          f"planner has no VLM: its {'template (prompt input)' if template else 'request'} is used as the "
                          "prompt", flush=True)
            # the panel's 🤖 Write uses the same template (it cannot see the prompt input)
            _notify("bfs-shotloop-autoprompt", {"video": p["video"], "prompts": new_written, "template": template})
        # descriptions nobody wrote: each picture described now, by itself, before any SAM 3 work (switching between
        # SAM 3 and the VLM mid-generation breaks the VLM: that is why the button worked and the run did not)
        if vlm is not None:
            for i in work:
                names = names_of(segs[i])
                text = details_template(written.get(i) or segs[i]["prompt"] or g_prompt or "", names)
                for m in PICTURE_DETAILS.finditer(text):
                    k = int(m.group(1)) - 1
                    if k < len(names) and not pic_details.get(names[k]):
                        pic_details[names[k]] = describe_picture(vlm, ref_for(names[k]), text, m.group(0),
                                                                 picture_cfg(vcfg, p, names[k]))
                        print(f"[BFS Shot Planner] {m.group(0)} = {pic_details[names[k]]!r} (described automatically)",
                              flush=True)
        # {target} with no description of its own: the shot's selection is cut out (SAM 3 for every shot first) and the
        # VLM describes it (written for the prompt's {target} sentence with fit_prompt)
        auto_target = {}
        if vlm is not None:
            need = [i for i in work if not segs[i].get("target")
                    and "{target}" in (written.get(i) or segs[i]["prompt"] or g_prompt or "")]
            crops = {}
            for i in need:
                tspec = plan_spec(p, segs[i].get("mask"), path)
                if spec_has_mask(tspec):
                    crops[i] = target_crop(path, a, segs[i]["start"], segs[i]["end"] - segs[i]["start"], tspec)
            for i, crop in crops.items():
                if crop is None:
                    continue
                text = written.get(i) or segs[i]["prompt"] or g_prompt or ""
                slot = prompt_slot(text, "{target}") if vcfg.get("fit_prompt", True) else ""
                auto_target[i] = describe_target(vlm, crop, slot=slot, temperature=float(vcfg.get("temperature") or 0),
                                                 seed=int(vcfg.get("seed") or 0),
                                                 image_max_side=int(vcfg.get("image_max_side") or 1024))
                print(f"[BFS Shot Planner] shot {i + 1}: {{target}} = {auto_target[i]!r} (described automatically)", flush=True)
        shots = []
        for i, s in enumerate(segs):
            if i not in todo:
                continue
            # continuity 'head': the window starts `head` frames early, on the previous shot's end (cut again by the join)
            head = head_length(segs, i, p["grid"]) if queue else 0
            if not queue and i > 0 and s["chain"] == "head" and not s["cut_before"]:
                print(f"[BFS Shot Planner] shot {i + 1}: head needs the queue loop (the previous result is not there "
                      "yet in the auto loop): generated without it", flush=True)
            g0 = s["start"] - head
            glen = snap_up(s["end"] - g0, p["grid"]) if head else s["gen_len"]
            idx = np.arange(g0, g0 + glen)
            idx = np.clip(idx, 0, a["n"] - 1)          # past the end: hold the last frame
            frames = _read_frames(path, src[idx], (W, H))
            ft = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
            names, refs = names_of(s), refs_of(s)
            ref, ref2 = (refs + [None, None])[:2]   # pictures 1 and 2: Colored Mark, Unpack's ref_image outputs
            prev_len = (segs[i - 1]["end"] - segs[i - 1]["start"]) if i > 0 else 0
            chain_img = None
            if queue and i > 0 and s["chain"] != "off":
                fp = os.path.join(run_dir(rid), f"shot_{i - 1:04d}.pt")
                if os.path.exists(fp):
                    prev = load_shot(run_dir(rid), i - 1)[0]
                    chain_img = _fit_to(pick_frame(prev, prev_len, s["chain_frame"]).float() / 255.0, W, H)
                    del prev
            sug = vlm_out.get(i)
            if sug is not None:
                suggestions.append(dict(sug, start=s["start"], end=s["end"]))
                if vcfg["auto_segment"] and sug["segment"] and not spec_has_mask(plan_spec(p, s["mask"], path)):
                    s["mask"] = dict(s["mask"], text=sug["segment"])
            crop, full_frames, mask_src = None, None, None
            sspec = plan_spec(p, s.get("mask"), path)
            if assemble_only:
                pass
            elif not s.get("crop") and spec_has_mask(sspec):
                # mask without crop: segmented only if the conditioning asks for it (inpaint "only the mask")
                mask_src = {"path": path, "analysis": a, "start": g0, "length": glen, "spec": sspec}
            if s.get("crop"):
                spec = sspec
                r = shot_mask(path, a, g0, glen, spec)
                # the box holds the frames the shot delivers (as the panel's preview shows it): the head before them and
                # the frames generated past the end (often a different scene after a cut) are cut by the join anyway
                L = s["end"] - s["start"]
                own = r["masks"][head:head + L]
                if own.shape[0] and bool(own.any()):
                    r = dict(r, box=crop_box(own, float(spec["padding"])))
                if r["box"] is not None:
                    full_frames = ft
                    sx0, sy0, sx1, sy1 = box_px(r["box"], a["width"], a["height"])
                    cw, chh = plan_generation_size(sx1 - sx0, sy1 - sy0, p, int(p["multiple"]))
                    hi = _read_frames(path, src[idx], None)               # source resolution, then crop
                    ft = torch.from_numpy(np.stack([_fit(f[sy0:sy1, sx0:sx1], (cw, chh)) for f in hi])
                                          .astype(np.float32) / 255.0)
                    m = torch.nn.functional.interpolate(r["masks"][:, None].float(), size=(H, W), mode="nearest")[:, 0]
                    crop = {"box": r["box"], "mask": m, "paste": spec["paste"], "expand": int(spec["expand"]),
                            "feather": int(spec["feather"])}
                    bx0, by0, bx1, by1 = box_px(r["box"], r["size"][0], r["size"][1])
                    crop["crop_mask"] = torch.nn.functional.interpolate(
                        r["masks"][:, None, by0:by1, bx0:bx1].float(), size=(chh, cw), mode="nearest")[:, 0]
            hist, hist_src = (history_for(path, a, src, s, i, prev_len, queue, rid, W, H, crop, ft)
                              if s["chain"] == "history" and i > 0 and not assemble_only else (None, None))
            head_px = head_for(i, head, prev_len, queue, rid, W, H, crop, ft) if head and not assemble_only else None
            template = (written.get(i) or s["prompt"] or g_prompt or "").replace(
                "{shot}", sug["shot"] if (sug and vcfg["auto_shot"]) else "")
            # a description nobody wrote is left for the conditioning node's VLM when the planner has none
            template = fill_picture_details(details_template(template, names), names, refs, pic_details,
                                            tag=f"[BFS Shot Planner] shot {i + 1}", keep_missing=vlm is None)
            shot_prompt = fill_target(template, s.get("target", "") or auto_target.get(i, ""))
            if s.get("color_guide") and i not in written:   # a written prompt already has the mapping
                shot_prompt = color_guide_prompt(shot_prompt, s["color_guide_marks"])
            shots.append({
                "index": i, "count": len(segs), "start": s["start"], "end": s["end"],
                "length": s["end"] - s["start"], "gen_length": glen, "head": head, "fps": fps,
                "cut_before": s["cut_before"], "width": ft.shape[2], "height": ft.shape[1], "frames": ft,
                "crop": crop, "full_frames": full_frames, "mask_src": mask_src,
                "inpaint": bool(s.get("inpaint")), "inpaint_strength": float(s.get("strength") or 1.0),
                "paste": bool(s.get("paste")), "paste_text": s.get("paste_text") or "",
                "color_guide": bool(s.get("color_guide")),
                "color_guide_color": str(s.get("color_guide_color") or "#00ff00"),
                "color_guide_opacity": float(s.get("color_guide_opacity") or 0.35),
                "color_guide_text2": str(s.get("color_guide_text2") or ""),
                "color_guide_color2": str(s.get("color_guide_color2") or "#9b30ff"),
                "color_guide_points2": list(s.get("color_guide_points2") or []),
                "color_guide_key2": int(s.get("color_guide_key2") or 0),
                "color_guide_marks": s["color_guide_marks"],
                "ref": ref, "ref2": ref2, "refs": refs, "ref_names": names,
                "prompt": shot_prompt,
                "audio": _slice_audio(audio, g0 / fps, glen / fps),
                "run_id": rid, "queue": queue, "run_mode": p.get("run"),
                "between_shots": p.get("between_shots") or "keep", "grid": p["grid"],
                "unload_before_join": bool(p.get("unload_before_join", True)), "assemble_only": assemble_only,
                "sig": sigs[i] if sigs else "", "prev_gen": (st.get("gen") or {}).get(str(i - 1), "") if sigs else "",
                "shot_storage": p.get("shot_storage") or "lossless",
                "chain": s["chain"], "chain_frame": s["chain_frame"], "chain_image": chain_img,
                "history": hist, "history_src": hist_src, "head_frames": head_px,
                "prev_length": prev_len,
                "source": {"path": path, "frames": [int(x) for x in src[idx]]},
            })
        if suggestions:
            _notify("bfs-shotloop-vlm", {"video": p["video"], "segs": suggestions})
        total = sum(s["end"] - s["start"] for s in segs)
        # no (usable) soundtrack: a silent track of the right length, so Create Video never gets None
        full_audio = _slice_audio(audio, segs[0]["start"] / fps, total / fps) if audio else silence(total / fps)
        # it covers the shots that run only: with skipped shots kept as original video, BFS Shot Join takes the
        # timeline's soundtrack instead (else a skipped shot is silent, or the sound after it is shifted)
        full_audio = dict(full_audio, bfs_runs_only=True)
        full_total = sum(s["end"] - s["start"] for s in all_segs)
        timeline = {"path": path, "fps": fps, "width": W, "height": H, "fill": p.get("skip_fill", "original"),
                    "audio": _slice_audio(audio, all_segs[0]["start"] / fps, full_total / fps) if audio else None,
                    "segs": [{"start": s["start"], "end": s["end"], "run": s["run"], "cut_before": s["cut_before"],
                              "run_index": segs.index(s) if s["run"] else -1} for s in all_segs],
                    "src": src.tolist(), "n": a["n"]}
        if queue:   # the panel's Combine assembles the stored shots on this timeline (original video where none ran)
            torch.save(timeline, os.path.join(run_dir(rid), "timeline.pt"))
        skipped = [(i, s.get("skip_reason") or ("disabled" if not s["enabled"] else "skip")) for i, s in enumerate(all_segs) if not s["run"]]
        lines = [f"{len(segs)} shots, {total} frames ({total / fps:.2f}s) at {fps:g} fps, {W}x{H}"
                 + (" | every shot stored: assembling only" if assemble_only else
                    f" | queue loop: running shot {shots[0]['index'] + 1}/{len(segs)}" if queue else "")]
        for i, why in skipped:
            lines.append(f"skip shot {i + 1} (frames {all_segs[i]['start']}-{all_segs[i]['end'] - 1}): {why}")
        for i, g in sorted(vlm_out.items()):
            lines.append(f"VLM #{i + 1}: segment '{g['segment']}', {g['recommend']}"
                         f"{' (' + g['reason'] + ')' if g['reason'] else ''} | {g['shot']}")
        for s in shots:
            lines.append(f"#{s['index'] + 1}: frames {s['start']}-{s['end'] - 1} ({s['length']} -> generate "
                         f"{s['gen_length']}{', the first ' + str(s['head']) + ' = previous end' if s.get('head') else ''})"
                         f"{' cut' if s['cut_before'] else ''}"
                         f"{' ' + str(len(s['refs'])) + ' ref' + ('s' if len(s['refs']) > 1 else '') if s['refs'] else ''}")
        sp = {**SECOND_PASS_DEFAULTS, **(p.get("second_pass") or {})}
        UW, UH = second_pass_size(W, H, sp)
        if sp["enabled"]:
            lines.insert(1, f"second pass: {W}x{H} -> {UW}x{UH}")
        return (shots, len(segs), fps, W, H, full_audio, "\n".join(lines), timeline, used_refs or [_grey()], len(used_refs),
                bool(sp["enabled"]), UW, UH)


class BFSShotUnpack:
    """Open one shot into plain values for any workflow (runs once per shot)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"shot": ("BFS_SHOT",)}}

    # refs and ref_count sit where ref_image and ref_image_2 were (1.113), so the other outputs keep their slots
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING", "INT", "IMAGE", "AUDIO", "INT", "INT", "INT", "BFS_SHOT",
                    "IMAGE", "MASK")
    RETURN_NAMES = ("guide_frames", "refs", "ref_count", "prompt", "length", "first_frame",
                    "audio", "width", "height", "index", "shot", "previous_result", "mask")
    OUTPUT_IS_LIST = (False, True, False, False, False, False, False, False, False, False, False, False, False)
    OUTPUT_TOOLTIPS = (
        "The shot's guide frames, already at a length the model accepts.",
        "This shot's reference pictures, <Picture 1> first, as a list: the next node runs once per picture (e.g. a "
        "background remover), and BFS Shot Repack's refs input puts them back in this shot. A grey placeholder when "
        "the shot has none.",
        "How many reference pictures the shot has.",
        "This shot's prompt.", "Frames to generate (the grid-valid length).",
        "First guide frame of the shot.", "Soundtrack of the shot (silence if the video has none).",
        "Generation width.", "Generation height.", "Shot index (0-based).",
        "The same shot, unchanged: connect it to BFS Shot Repack after editing the pieces.",
        "The previous shot's result frame this shot continues from (Continuity in the planner; queue loop), "
        "or a grey image.",
        "The shot's SAM 3 mask over the guide frames (in the crop when the shot is cropped), or all ones.")
    FUNCTION = "unpack"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = "Split one shot into its guide frames, reference pictures, prompt and length."

    def unpack(self, shot):
        audio = shot["audio"] or {"waveform": torch.zeros(1, 1, int(44100 * shot["gen_length"] / shot["fps"])),
                                  "sample_rate": 44100}
        refs = shot_refs(shot)
        return (shot["frames"], refs or [_grey()], len(refs), shot["prompt"], shot["gen_length"],
                shot["frames"][:1], audio, shot["width"], shot["height"], shot["index"], shot,
                chain_image(shot) if chain_image(shot) is not None else _grey(),
                shot["crop"]["crop_mask"] if shot.get("crop") else torch.ones(shot["frames"].shape[:3]))


class BFSShotRepack:
    """Put edited pieces back into a shot, e.g. its reference pictures with the background removed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"shot": ("BFS_SHOT",)},
            "optional": {
                "guide_frames": ("IMAGE", {"tooltip": "Replacement guide frames. Resized to the shot's generation "
                                                      "size; padded or trimmed to its length."}),
                "refs": ("IMAGE", {"tooltip": "Replacement reference pictures: BFS Shot Unpack's refs list after any "
                                              "processing (e.g. background removed), one image per picture, in order."}),
                "prompt": ("STRING", {"forceInput": True, "tooltip": "Replacement prompt."}),
            },
        }

    # lists in and out: Unpack's refs come as one flat list (every picture of every shot in order), split back here
    INPUT_IS_LIST = True
    RETURN_TYPES = ("BFS_SHOT",)
    RETURN_NAMES = ("shot",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "repack"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Rebuild a shot after editing its pieces (Unpack -> any processing -> Repack). Inputs left "
                   "unconnected keep the shot's original values; timing, cuts and audio are unchanged.")

    def repack(self, shot, guide_frames=None, refs=None, prompt=None, ref_image=None, ref_image_2=None):
        shots = shot if isinstance(shot, list) else [shot]
        listed = lambda v: None if v is None else v if isinstance(v, list) else [v]
        guide_frames, prompt, ref_image_2 = listed(guide_frames), listed(prompt), listed(ref_image_2)
        refs = listed(refs) if refs is not None else listed(ref_image)   # workflows before 1.113 wired ref_image
        # Unpack hands out max(1, pictures) images per shot (a grey one when it has none)
        counts = [len(shot_refs(s)) for s in shots]
        split = refs is not None and len(refs) == sum(max(1, n) for n in counts)
        if refs is not None and not split:
            print(f"[BFS Shot Repack] {len(refs)} reference picture(s) for {sum(counts)} in {len(shots)} shot(s): "
                  "every shot gets the pictures as they come", flush=True)

        def one(values, i):   # one value per shot, or the same one for every shot
            return None if not values else values[i] if len(values) == len(shots) else values[min(i, len(values) - 1)]
        out, pos = [], 0
        for i, s in enumerate(shots):
            new = refs
            if split:
                new = refs[pos:pos + counts[i]]
                pos += max(1, counts[i])
            out.append(self.repack_one(s, one(guide_frames, i), new, one(prompt, i), one(ref_image_2, i)))
        return (out,)

    @staticmethod
    def repack_one(shot, guide_frames=None, refs=None, prompt=None, ref_image_2=None):
        out = dict(shot)
        if guide_frames is not None:
            fr = guide_frames
            H, W = shot["height"], shot["width"]
            if tuple(fr.shape[1:3]) != (H, W):
                arr = (fr.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                fr = torch.from_numpy(np.stack([_fit(a, (W, H)) for a in arr]).astype(np.float32) / 255.0)
            n = shot["gen_length"]
            if fr.shape[0] != n:
                print(f"[BFS Shot Repack] shot {shot['index'] + 1}: guide has {fr.shape[0]} frames, "
                      f"expected {n}; {'trimming' if fr.shape[0] > n else 'holding the last frame'}")
                fr = fr[:n] if fr.shape[0] > n else torch.cat([fr, fr[-1:].expand(n - fr.shape[0], -1, -1, -1)], 0)
            out["frames"] = fr
        new = [r[:1] for r in refs] if refs is not None else shot_refs(shot)
        if ref_image_2 is not None:   # an old workflow's second reference
            new = new[:1] + [ref_image_2[:1]] + new[2:] if new else [ref_image_2[:1]]
        out["refs"] = new
        out["ref"], out["ref2"] = (new + [None, None])[:2]
        if prompt is not None:
            out["prompt"] = prompt
        return out


SETTING_MODES = ["off", "on (generation size)", "on (source size, slower)"]


def tv_static(img: torch.Tensor, mask: torch.Tensor, grow: float = 0.03, seed: int = 0) -> torch.Tensor:
    """[1,H,W,3] with the masked region (dilated by `grow` of the short side) covered in black-and-white TV static,
    TSC's 'person noised out' setting picture: the place is seen in full detail, the person is not."""
    H, W = img.shape[1:3]
    m = torch.nn.functional.interpolate(mask.float().reshape(1, 1, *mask.shape[-2:]), size=(H, W), mode="nearest")[0, 0]
    g = max(1, int(min(H, W) * grow))
    m = torch.nn.functional.max_pool2d(m[None, None], 2 * g + 1, stride=1, padding=g)[0, 0] > 0.5
    gen = torch.Generator().manual_seed(seed)
    grain = 2
    n = (torch.rand((H + grain - 1) // grain, (W + grain - 1) // grain, generator=gen) > 0.5).float()
    n = n.repeat_interleave(grain, 0).repeat_interleave(grain, 1)[:H, :W]
    out = img.clone()
    out[0][m] = n[m][:, None].expand(-1, 3).to(out)
    return out


def setting_picture(shot: dict, mode: str, mask: torch.Tensor | None = None) -> torch.Tensor:
    """The middle frame of the shot with the person covered in TV static. Mask: the given one, else the shot's SAM 3
    crop mask, else SAM 3 'person' on that frame."""
    src = shot.get("source") or {}
    k = len(shot["frames"]) // 2
    if mode == SETTING_MODES[2] and src.get("path"):
        f = _read_frames(src["path"], np.array([src["frames"][min(k, len(src["frames"]) - 1)]]), None)[0]
        img = torch.from_numpy(f.astype(np.float32) / 255.0)[None]
        short = min(img.shape[1:3])
        if short > 2048:   # what H3's 'max' reference size keeps anyway
            s = 2048 / short
            img = torch.nn.functional.interpolate(img.movedim(-1, 1), scale_factor=s, mode="bilinear",
                                                  align_corners=False).movedim(1, -1)
    else:
        full = shot.get("full_frames")
        img = (full if full is not None else shot["frames"])[k:k + 1]
    if mask is not None:
        m = mask[min(k, mask.shape[0] - 1)] if mask.ndim == 3 else mask
    elif shot.get("crop") is not None:
        cm = shot["crop"]["mask"]
        m = cm[min(k, cm.shape[0] - 1)]
    elif shot.get("mask_src"):                      # the person marked on the shot (mask without crop)
        ms = shot["mask_src"]
        sm = shot_mask(ms["path"], ms["analysis"], ms["start"], ms["length"], ms["spec"])["masks"]
        m = sm[min(k, sm.shape[0] - 1)]
    else:
        spec = dict(DEFAULT_MASK, text="person", max_objects=8)
        small = torch.nn.functional.interpolate(img.movedim(-1, 1), size=_fit_size(img.shape[1:3], 640),
                                                mode="bilinear", align_corners=False).movedim(1, -1)
        m = segment_frames(small, spec)[0]
        _node_boundary()
    return tv_static(img, m)


def _fit_size(hw, side: int) -> tuple[int, int]:
    s = side / max(hw)
    return max(32, int(hw[0] * s) // 2 * 2), max(32, int(hw[1] * s) // 2 * 2)


def setting_line(k: int, swap: bool) -> str:
    who = "<Subject 1>" if swap else "the performer"
    return (f"<Picture {k}> shows the setting, the same place as the original performance in full detail; the noise "
            f"patch in it is where {who} stands.")


def add_setting(text: str, k: int, swap: bool) -> str:
    """`{setting}` becomes <Picture k>; without it, a sentence goes at the end of subject_definitions."""
    if "{setting}" in text:
        return text.replace("{setting}", f"<Picture {k}>")
    line = setting_line(k, swap)
    if "subject_definitions:" in text:
        a = text.index("subject_definitions:")
        b = text.find("\n\n", a)
        b = len(text) if b < 0 else b
        return text[:b].rstrip() + " " + line + text[b:]
    return (text.rstrip() + "\n\n" + line).strip()


def add_mask_video(text: str, k: int) -> str:
    """`{mask_video}` becomes <Video k>; without it, a sentence goes at the end of subject_definitions."""
    if "{mask_video}" in text:
        return text.replace("{mask_video}", f"<Video {k}>")
    line = (f"<Video {k}> shows only the region being replaced, the rest grey: follow its pose, outline and motion "
            f"frame by frame.")
    if "subject_definitions:" in text:
        a = text.index("subject_definitions:")
        b = text.find("\n\n", a)
        b = len(text) if b < 0 else b
        return text[:b].rstrip() + " " + line + text[b:]
    return (text.rstrip() + "\n\n" + line).strip()


INPAINT_MODES = ["per shot (planner)", "off", "only the mask"]
MASK_REF_SIZES = {"full": 1.0, "1/2": 0.5, "1/3": 1 / 3, "1/4": 0.25}
MASK_GUIDES = ["off", "+ extra guide (masked region only)", "instead of the full guide (masked region only)",
               "+ reference video (masked region only)"]


MASK_GUIDE_LOOKS = ["grey blurred", "colour", "silhouette", "edges", "pose (people)"]
# COCO-17 skeleton, coloured per limb like OpenPose drawings
_LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
          (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)]
_LIMB_RGB = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
             (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
             (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def pose_frames(frames: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """The skeleton (COCO-17, OpenPose colours) of the people inside the mask on black, [F,H,W,3]. Uses a YOLO pose
    model from models/ultralytics (e.g. pose/yolov8m-pose.pt), the same place and package as the planner's filters."""
    import cv2
    model = _yolo("pose")
    if model is None:
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            raise ValueError("mask_guide_look 'pose' needs the ultralytics package (the same one the Impact Pack and "
                             "the planner's filters use): pip install ultralytics") from None
        raise ValueError("mask_guide_look 'pose' needs a YOLO pose model in ComfyUI/models/ultralytics/pose/, e.g. "
                         "yolov8m-pose.pt from https://github.com/ultralytics/assets/releases/download/v8.2.0/"
                         "yolov8m-pose.pt (no restart needed after adding it)")
    F, H, W = frames.shape[:3]
    imgs = [cv2.cvtColor((f[..., :3].numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for f in frames]
    m = None
    if mask is not None:
        m = torch.nn.functional.interpolate(mask[:, None].float(), size=(H, W), mode="nearest")[:, 0].numpy() > 0.5
    th = max(2, int(min(H, W) / 90))
    out = np.zeros((F, H, W, 3), np.uint8)
    for i in range(0, F, 16):
        res = model(imgs[i:i + 16], verbose=False)
        for j, r in enumerate(res):
            k = i + j
            if r.keypoints is None or r.keypoints.xy is None:
                continue
            xy, cf = r.keypoints.xy.cpu().numpy(), r.keypoints.conf
            cf = cf.cpu().numpy() if cf is not None else np.ones(xy.shape[:2])
            for p in range(xy.shape[0]):
                pts, ok = xy[p], cf[p] > 0.3
                if m is not None:                       # only the people in the shot's mask
                    vis = [(int(x), int(y)) for (x, y), o in zip(pts, ok) if o and 0 <= int(x) < W and 0 <= int(y) < H]
                    if not vis or sum(m[min(k, m.shape[0] - 1), y, x] for x, y in vis) < max(1, len(vis) // 3):
                        continue
                for (a, b), col in zip(_LIMBS, _LIMB_RGB):
                    if ok[a] and ok[b]:
                        cv2.line(out[k], tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), col, th, cv2.LINE_AA)
                for q in range(17):
                    if ok[q]:
                        cv2.circle(out[k], tuple(int(v) for v in pts[q]), th + 1, (255, 255, 255), -1, cv2.LINE_AA)
    return torch.from_numpy(out.astype(np.float32) / 255.0)


def guide_look(frames: torch.Tensor, look: str, mask: torch.Tensor | None = None) -> torch.Tensor:
    """What the mask guide shows of the subject: its colours, or only its shape and motion (no clothes / face to
    copy): luminance blurred, a flat silhouette, its edges, or its pose skeleton (people)."""
    fr = frames[..., :3].float()
    if look == "colour":
        return fr
    if look.startswith("pose"):
        return pose_frames(fr, mask)
    H, W = fr.shape[1:3]
    lum = (fr * torch.tensor([0.299, 0.587, 0.114])).sum(-1)                 # [F,H,W]
    if look == "silhouette":
        return torch.full_like(fr, 0.15)
    if look == "edges":
        import cv2
        out = []
        for f in lum.numpy():
            g = cv2.GaussianBlur((f * 255).astype(np.uint8), (5, 5), 0)
            e = cv2.dilate(cv2.Canny(g, 40, 120), np.ones((2, 2), np.uint8))
            out.append(np.where(e > 0, 0.95, 0.1).astype(np.float32))
        return torch.from_numpy(np.stack(out))[..., None].expand(-1, -1, -1, 3).contiguous()
    r = max(2, int(min(H, W) * 0.02))                                        # grey blurred: volume and light only
    b = grow_blur(lum, 0, r)
    return b[..., None].expand(-1, -1, -1, 3).contiguous()


def masked_only(frames: torch.Tensor, mask: torch.Tensor, grey: float = 0.5, look: str = "colour") -> torch.Tensor:
    """The shot's frames with everything outside the mask grey: what the mask guide shows (`look`: see guide_look)."""
    m = torch.nn.functional.interpolate(mask[:, None].float(), size=frames.shape[1:3], mode="nearest")[:, 0]
    idx = torch.clamp(torch.arange(frames.shape[0]), max=m.shape[0] - 1)
    m = m[idx][..., None]
    if look.startswith("pose"):                       # the skeleton on black inside the (grown) mask
        return guide_look(frames, look, mask) * m + grey * (1 - m)
    return guide_look(frames, look) * m + grey * (1 - m)


def color_guide(frames: torch.Tensor, mask: torch.Tensor, color: str = "#00ff00",
                opacity: float = 0.35) -> torch.Tensor:
    """Tint a selected subject in the guide while keeping its underlying motion and expression visible."""
    try:
        rgb = torch.tensor([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)], dtype=torch.float32)
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"Colored Mark: invalid color {color!r}; expected a hex color such as #00ff00") from None
    if mask.ndim == 2:
        mask = mask[None]
    m = torch.nn.functional.interpolate(mask[:, None].float(), size=frames.shape[1:3], mode="nearest")[:, 0]
    idx = torch.clamp(torch.arange(frames.shape[0]), max=m.shape[0] - 1)
    alpha = m[idx][..., None] * min(0.8, max(0.05, float(opacity)))
    tint = rgb.to(device=frames.device, dtype=frames.dtype)
    return (frames[..., :3].float() * (1 - alpha) + tint * alpha).clamp(0, 1)


# one mark color per picture, <Picture 1> first: the first seven are the colors the first Colored Mark sets used
MARK_PALETTE = ["#00ff00", "#9b30ff", "#ff2020", "#168cff", "#ffe600", "#ff00d4", "#00e5c0", "#ff8000", "#ffffff",
                "#8b4513", "#ff9ec4", "#808000"]
_COLOR_GUIDE_NAMES = {"#00ff00": "green", "#9b30ff": "purple", "#ff2020": "red", "#168cff": "blue", "#ffe600": "yellow",
                      "#ff00d4": "magenta", "#00e5c0": "cyan", "#ff8000": "orange", "#ffffff": "white",
                      "#8b4513": "brown", "#ff9ec4": "pink", "#808000": "olive"}


def picture_color(plan: dict, k: int) -> str:
    """The mark color of <Picture k>: the plan's picture_colors, else (plans before 1.113) the marker colors of the
    two global references, else the palette."""
    colors = plan.get("picture_colors")
    if isinstance(colors, list):
        c = colors[k - 1] if 0 < k <= len(colors) else ""
    else:
        c = {1: plan.get("global_color_guide_ref1"), 2: plan.get("global_color_guide_ref2")}.get(k, "")
    if isinstance(c, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", c):
        return c.lower()
    return MARK_PALETTE[(k - 1) % len(MARK_PALETTE)]


def color_marks(plan: dict, m: dict) -> list[dict]:
    """A shot's Colored Mark regions: its own selection (the shot's mask) first, then the extra regions (SAM 3 text or
    points), each with the picture the person becomes and that picture's color. Plans before 1.113 had one extra
    region (color_guide_text2 / points2), which becomes <Picture 2>."""
    extra = m.get("color_guide_regions")
    if not isinstance(extra, list):
        extra = ([{"picture": 2, "text": m.get("color_guide_text2"), "points": m.get("color_guide_points2"),
                   "key": m.get("color_guide_key2")}] if m.get("color_guide_text2") or m.get("color_guide_points2") else [])
    marks = [{"picture": max(1, int(m.get("color_guide_picture") or 1))}]
    marks += [{"picture": max(1, int(r.get("picture") or 2)), "text": str(r.get("text") or "").strip(),
               "points": list(r.get("points") or []), "key": int(r.get("key") or 0), "keys": list(r.get("keys") or [])}
              for r in extra if isinstance(r, dict) and (str(r.get("text") or "").strip() or r.get("points") or r.get("keys"))]
    for x in marks:
        x["color"] = picture_color(plan, x["picture"])
    return marks


def extra_mark_masks(shot: dict, log: str = "") -> list[tuple]:
    """The extra Colored Mark regions of a shot (the second one on): (masks over the guide frames, mark), each one
    person (SAM 3, one object). They need the full frame: a cropped shot only has its own selection."""
    extra = (shot.get("color_guide_marks") or [])[1:]
    if not extra:
        return []
    ms = shot.get("mask_src")
    if not ms:
        raise ValueError(f"{log or 'Colored Mark'}: more than one marked person needs Crop turned off, so every person "
                         "can be marked on the full frame")
    out = []
    for x in extra:
        spec = dict(DEFAULT_MASK, text=x.get("text") or "", points=x.get("points") or [], key=x.get("key") or 0,
                    keys=x.get("keys") or [],
                    max_objects=1)
        out.append((shot_mask(ms["path"], ms["analysis"], ms["start"], ms["length"], spec)["masks"], x))
    return out


def color_guide_prompt(prompt: str, marks: list[dict]) -> str:
    """Say which picture each mark color becomes, inside the prompt's subject_definitions / detailed_description when
    it has them (else at the end). Written for any model and LoRA: the person marked in a color becomes the character
    of that color's picture."""
    seen, pairs = set(), []
    for x in marks:
        name = _COLOR_GUIDE_NAMES.get(str(x["color"]).lower(), str(x["color"]))
        if (name, x["picture"]) not in seen:
            seen.add((name, x["picture"]))
            pairs.append((name, x["picture"]))
    keep = ("the movement, pose, gestures, head motion, gaze and expression seen through the mark")
    if len(pairs) == 1:
        name, k = pairs[0]
        mapping = (f"The {name} mark shows the only person to replace. Replace only the person marked in {name} with "
                   f"the character from <Picture {k}>, keeping {keep}. Keep every other visible person and all unmarked "
                   "content unchanged, and remove the mark completely from the result.")
    else:
        who = "; ".join(f"the person marked in {name} becomes the character from <Picture {k}>" for name, k in pairs)
        mapping = (f"The colored marks show the people to replace: {who}. Each one keeps {keep.replace('the mark', 'their mark')}. "
                   "Keep every unmarked person and all unmarked content unchanged, and remove every mark completely "
                   "from the result.")
    # Keep the color contract inside the matching sections of the user's existing structured prompt.
    text, n = re.subn(r"(?is)(\bsubject_definitions\s*:.*?)(\s+summary\s*:)",
                      lambda m: m.group(1) + " " + mapping + m.group(2), str(prompt or ""), count=1)
    if not n:
        text = (str(prompt or "").rstrip() + "\n\n" + mapping).strip()
    detailed, n = re.subn(r"(?is)(\bdetailed_description\s*:.*?)(\s+overall_soundscape\s*:)",
                          lambda m: m.group(1) + " " + mapping + m.group(2), text, count=1)
    return detailed if n else text


def mask_report(m: torch.Tensor) -> str:
    """'37 frames 512x288, covers 12.3% (min 8.1%, max 15.0%)' for console logs."""
    cov = m.float().flatten(1).mean(1) if m.ndim == 3 and m.shape[0] else m.float().mean()[None]
    empty = int((cov == 0).sum())
    return (f"{m.shape[0]} frames {m.shape[-1]}x{m.shape[-2]}, covers {float(cov.mean()) * 100:.1f}% "
            f"(min {float(cov.min()) * 100:.1f}%, max {float(cov.max()) * 100:.1f}%)"
            + (f", EMPTY on {empty} frame(s)" if empty else ""))


def generation_mask(shot: dict, log: str = "") -> torch.Tensor:
    """The shot's person mask [F,H,W] at the generation size, grown by the mask's `expand`: the crop's own mask for a
    cropped shot, otherwise the whole frame's (segmented here, cached). `log`: prefix for console messages.
    Kept on the shot, so the mask guide, the inpaint and the join reuse it."""
    if shot.get("_gen_mask") is not None:
        return shot["_gen_mask"]
    crop = shot.get("crop") or {}
    if crop.get("crop_mask") is not None:
        m, grow = crop["crop_mask"], int(crop.get("expand") or 0)
        if log:
            print(f"{log}: using the crop's SAM 3 mask (segmented by the planner), expand {grow}px", flush=True)
    elif shot.get("mask_src"):
        ms = shot["mask_src"]
        if log:
            sp = ms["spec"]
            what = f"{len(sp.get('points') or [])} points" if sp.get("points") else f"'{sp.get('text', '')}'"
            cached = _mask_key(ms["path"], ms["analysis"], ms["start"], ms["length"], sp) in _MASK_CACHE
            print(f"{log}: mask {what}, expand {int(sp.get('expand') or 0)}px, "
                  + ("already segmented (cache)" if cached else "not segmented yet: running SAM 3"), flush=True)
        r = shot_mask(ms["path"], ms["analysis"], ms["start"], ms["length"], ms["spec"])
        H, W = shot["frames"].shape[1:3]
        m = torch.nn.functional.interpolate(r["masks"][:, None].float(), size=(H, W), mode="nearest")[:, 0]
        grow = int(ms["spec"].get("expand") or 0)
    else:
        raise ValueError(f"shot {shot.get('index', 0) + 1}: inpaint 'only the mask' needs a SAM 3 mask on the shot "
                         "(mask text or points in the Planner)")
    shot["_gen_mask"] = grow_blur(m, grow, 0) if grow > 0 else m.float()
    return shot["_gen_mask"]


def overlay_of(shot: dict) -> dict | None:
    """What the comparison draws over the original column: a cropped shot's mask and box, or the generation mask of a
    mask-only shot (segmented already when it ran; never runs SAM 3 just for the picture unless the shot uses it)."""
    if shot.get("color_guide"):
        if shot.get("comparison_mark_regions"):
            return {"regions": shot["comparison_mark_regions"]}
        opacity = float(shot.get("color_guide_opacity") or 0.35)
        regions = []
        crop = shot.get("crop")
        if crop and crop.get("mask") is not None:
            regions.append({"mask": crop["mask"], "color": shot.get("color_guide_color") or "#00ff00",
                            "opacity": opacity})
        else:
            ms = shot.get("mask_src")
            if ms:
                try:
                    r1 = shot_mask(ms["path"], ms["analysis"], ms["start"], ms["length"], ms["spec"])
                    regions.append({"mask": _skip_head(r1["masks"], shot),
                                    "color": shot.get("color_guide_color") or "#00ff00", "opacity": opacity})
                    regions += [{"mask": _skip_head(mk, shot), "color": x["color"], "opacity": opacity}
                                for mk, x in extra_mark_masks(shot)]
                except Exception as e:  # the comparison picture is optional
                    print(f"[BFS Shot Join] no Colored Mark overlay for shot {shot.get('index', 0) + 1}: {e}")
        return {"regions": regions, "box": crop.get("box") if crop else None}
    if shot.get("crop"):
        return shot["crop"]
    ms = shot.get("mask_src")
    if not ms:
        return None
    try:
        if not shot.get("inpaint") and _mask_key(ms["path"], ms["analysis"], ms["start"], ms["length"], ms["spec"]) not in _MASK_CACHE:
            return None
        return {"mask": _skip_head(generation_mask(shot), shot)}
    except Exception as e:   # the picture is optional
        print(f"[BFS Shot Join] no mask overlay for shot {shot.get('index', 0) + 1}: {e}")
        return None


def _comparison_mark_regions(shot: dict, source: torch.Tensor, height: int) -> list[dict]:
    """Pack low-resolution Colored Mark masks into queue files for the later comparison join."""
    H, W = source.shape[1:3]
    scale = min(1.0, (height or H) / max(1, H))
    size = (max(16, int(H * scale)), max(16, int(W * scale)))
    opacity = float(shot.get("color_guide_opacity") or 0.35)
    masks = []
    crop = shot.get("crop")
    ms = shot.get("mask_src")
    if crop and crop.get("mask") is not None:
        masks.append((crop["mask"], shot.get("color_guide_color") or "#00ff00"))
    elif ms:
        r1 = shot_mask(ms["path"], ms["analysis"], ms["start"], ms["length"], ms["spec"])
        masks.append((_skip_head(r1["masks"], shot), shot.get("color_guide_color") or "#00ff00"))
    if not crop and ms:
        masks += [(_skip_head(mk, shot), x["color"]) for mk, x in extra_mark_masks(shot)]
    packed = []
    for mask, color in masks:
        if mask.ndim == 2:
            mask = mask[None]
        small = torch.nn.functional.interpolate(mask[:, None].float(), size=size, mode="nearest")[:, 0]
        bits = torch.from_numpy(np.packbits((small > 0.5).cpu().numpy().astype(np.uint8), axis=-1).copy())
        packed.append({"packed": bits, "shape": tuple(small.shape), "color": color, "opacity": opacity})
    return packed


def inpaint_latent(latent: dict, positive, vae, frames: torch.Tensor, mask: torch.Tensor, strength: float = 1.0):
    """H3 AV latent whose video starts from `frames` (VAE-encoded) and whose generation mask is `mask` (1 = regenerate)
    on H3's latent grid (grown by one cell); the audio stays fully generated. strength < 1 puts the masked rows at
    strength x the noise level, so they keep some of the original (H3 reads fractional mask rows natively).
    Per-row mixed-timestep masking for H3: credit to Neko (Nekodificador) and AbleJones, whose workflow and nodes this
    follows."""
    import comfy.nested_tensor
    try:
        from .bfs_h3_side_panel import _encode, latent_mask
    except ImportError:
        from bfs_h3_side_panel import _encode, latent_mask
    samples = latent["samples"]
    video, audio = samples.tensors[0], samples.tensors[1]
    T, h, w = video.shape[2], video.shape[3], video.shape[4]
    if frames.shape[1] != h * 16 or frames.shape[2] != w * 16:
        frames = torch.nn.functional.interpolate(frames[..., :3].movedim(-1, 1).float(), size=(h * 16, w * 16),
                                                 mode="bilinear", align_corners=False).movedim(1, -1)
    enc = _encode(vae, frames[..., :3]).to(video)
    if enc.shape[2] != T:
        raise ValueError(f"inpaint: the shot encodes to {enc.shape[2]} latent frames, the video has {T}")
    vmask = latent_mask(mask, T, h, w, grow=1).to(video.device) * float(min(1.0, max(0.0, strength)))
    out = dict(latent)
    out["samples"] = comfy.nested_tensor.NestedTensor((enc, audio))
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((vmask, torch.ones_like(audio)))
    return positive, out


class BFSShotH3Conditioning:
    """Native MiniMax H3 conditioning for one shot: references, prompt and the shot as a guide."""

    GUIDE_MODES = ["aligned guide (Add Guide)", "native reference video (Video 1)", "both", "none"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot": ("BFS_SHOT",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "guide_mode": (cls.GUIDE_MODES, {"default": cls.GUIDE_MODES[0], "tooltip":
                    "aligned guide: the shot's frames sit on the generated frames (MiniMax H3 Add Guide), "
                    "frame by frame. native reference video: the shot enters as <Video 1> like the reference "
                    "video input. both: the two together."}),
                "first_frame": (["none", "shot's first frame"], {"default": "none", "tooltip":
                    "Anchor the shot's own first frame as an extra aligned image guide."}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "with_audio": ("BOOLEAN", {"default": False, "tooltip":
                    "Attach the shot's soundtrack to the guide / reference video (needs audio_vae)."}),
                "model": ("MODEL", {"tooltip": "Optional: passed through to the model output unchanged (workflows "
                                               "that routed the model through this node keep working)."}),
                "vlm": ("CLIP", {"tooltip": "Optional VLM (CLIPLoader with a Qwen3-VL text encoder). Fills {details} "
                                            "when the planner left it (no 📝 Describe refs, no VLM on the planner)."}),
                "setting_ref": (SETTING_MODES, {"default": "off", "tooltip":
                    "TSC's trick: one more reference picture, the shot's middle frame with the person covered in TV "
                    "static, so the model sees the place in full detail (the panel / guide is often small). It is "
                    "the last <Picture n>; a sentence about it is added to subject_definitions (or write {setting} "
                    "where you want its tag). Mask: setting_mask, else the shot's SAM 3 mask (crop or not), "
                    "else SAM 3 'person' on that frame. 'source size' uses the video's own resolution (up to 2048 short edge): "
                    "sharper, slower."}),
                "setting_mask": ("MASK", {"tooltip": "Optional mask of the person to cover in the setting picture."}),
                "inpaint": (INPAINT_MODES, {"default": INPAINT_MODES[0], "tooltip":
                    "per shot (planner): each shot's mode in the Shot Planner decides (Full frame / Mask only / Crop / Crop + mask). only the mask: every shot. off: never. "
                    "Regenerate ONLY the person. The latent starts from the shot's own frames and the shot's SAM 3 "
                    "mask (grown by the mask's expand + one latent cell) becomes H3's generation mask; everything else "
                    "stays exactly as it was. With 'Crop to mask' on the shot it works inside the crop (more pixels "
                    "for the person, BFS Shot Join pastes it back); without crop it works on the whole frame (no "
                    "crop / uncrop at all). Needs a mask (text or points) on the shot. Crop + H3 generation mask: credit to Neko (Nekodificador) and AbleJones, "
                    "whose workflow and nodes this follows. On ComfyUI builds without native H3 generation masks, add "
                    "a per-row mask patch to the model (e.g. ComfyUI-MiniMaxH3-PerRowMasking)."}),
                "mask_guide": (MASK_GUIDES, {"default": MASK_GUIDES[0], "tooltip":
                    "Experimental, 'Mask only' shots (mask, no crop): the shot with ONLY the masked region visible "
                    "(the rest grey), so the model looks at the subject's pose and outline on its own while the mask "
                    "limits where it generates. '+ extra guide': a second aligned guide latent next to the normal one "
                    "(the model sees both). 'instead of the full guide': replaces the normal guide. '+ reference "
                    "video': enters as a native reference video (<Video n>, after the shot's own if guide_mode uses "
                    "it). The LoRAs were trained with one full guide, so test against off; it can also pull the old "
                    "subject's look."}),
                "source_blur": (SOURCE_BLUR, {"default": "off", "tooltip":
                    "Blur the source person in what the model SEES of the source (aligned guide, native reference "
                    "video): '<region> (<what stays sharp>)'. Region: face / head + hands / person (every "
                    "person the segmenter finds) or masked area (the shot's own mask, the person being replaced, with "
                    "its expand). keep mouth: lips, teeth and tongue stay sharp; keep lips + tongue: the teeth are "
                    "blurred too; mouth too: nothing stays sharp. The model still reads the pose and motion but has "
                    "no face to copy, so the identity comes from the reference. The kept area of 'Mask only' and the "
                    "output are never blurred. JalenBrunson's swap trick."}),
                "source_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "How coarse the source blur is (1 = no detail left)."}),
                "source_blur_segmenter": (BLUR_SEGMENTERS, {"default": BLUR_SEGMENTERS[0], "tooltip":
                    "What finds the face / head / hands / person and the mouth for source_blur. SAM 3.1: text prompts "
                    "(already used by the planner). Sapiens2: body-part classes in one pass, finer lips and teeth "
                    "(downloads facebook/sapiens2-seg-0.4b, 1.6 GB, to models/sapiens2 the first time). The 'keep lips + "
                    "tongue' modes always use Sapiens2 for the lips."}),
                "source_blur_effect": (BLUR_EFFECTS, {"default": BLUR_EFFECTS[0], "tooltip":
                    "blur: no detail left. invert colours: keeps the shapes and the expression but not the skin tone "
                    "or the colours."}),
                "mask_guide_look": (MASK_GUIDE_LOOKS, {"default": MASK_GUIDE_LOOKS[0], "tooltip":
                    "What the mask guide shows of the subject. grey blurred: its volume, light and head direction "
                    "without colours or a face (default). colour: as it is (can make the model copy the old subject). "
                    "silhouette: only its flat shape. edges: only its outlines. pose (people): the skeleton of the "
                    "people in the mask (OpenPose colours), from a YOLO pose model in models/ultralytics (e.g. "
                    "pose/yolov8m-pose.pt; uses ultralytics like the planner's filters). The less it shows, the less "
                    "the model can copy the old subject; the more, the better it follows the pose."}),
                "mask_ref_size": (list(MASK_REF_SIZES), {"default": "1/2", "tooltip":
                    "mask_guide '+ reference video' only: size of that reference video (fraction of the generation "
                    "size). The native reference channel keeps a smaller video as it is, so 1/2 costs ~1/4 of the "
                    "tokens; it only has to show the pose and outline."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "MODEL", "STRING")
    RETURN_NAMES = ("positive", "latent", "model", "prompt")
    FUNCTION = "condition"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Build MiniMax H3 conditioning for one shot with the native nodes: Reference to Video "
                   "(prompt, references, length) plus the shot as an aligned guide and/or reference video.")

    def condition(self, shot, clip, vae, guide_mode, first_frame, ref_image_size,
                  audio_vae=None, with_audio=False, model=None, vlm=None, setting_ref="off", setting_mask=None, inpaint="off", mask_guide="off", mask_ref_size="1/2", mask_guide_look="grey blurred",
                  source_blur="off", source_blur_strength=1.0, source_blur_segmenter="SAM 3.1", source_blur_effect="blur"):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, MiniMaxH3ReferenceToVideo
        # every picture of the shot, <Picture 1> first (chosen in the planner; use_ref_2 left in 1.113)
        refs = {f"ref_image_{k}": r for k, r in enumerate(shot_refs(shot)[:MAX_REFS])}
        if shot.get("color_guide"):
            marks = shot.get("color_guide_marks") or [{"picture": 1, "color": shot.get("color_guide_color", "#00ff00")}]
            for x in marks:
                if x["picture"] > len(refs):
                    raise ValueError(f"Colored Mark: a person is marked as <Picture {x['picture']}>, but the shot passes "
                                     f"{len(refs)} picture(s). Add the picture to the shot or the global list.")
            owner = {}
            for x in marks:
                if owner.setdefault(x["color"], x["picture"]) != x["picture"]:
                    raise ValueError(f"Colored Mark: <Picture {owner[x['color']]}> and <Picture {x['picture']}> have the "
                                     f"same mark color ({x['color']}). Give one of them another color (Prompts & refs tab).")
        prev = chain_image(shot)
        if prev is not None and shot.get("chain") == "reference":
            refs[f"ref_image_{len(refs)}"] = prev   # one more <Picture n>, after the shot's own
        audio = shot["audio"] if (with_audio and audio_vae is not None) else None
        native = guide_mode in (self.GUIDE_MODES[1], self.GUIDE_MODES[2])
        aligned = guide_mode in (self.GUIDE_MODES[0], self.GUIDE_MODES[2])
        text = shot["prompt"] or ""
        setting = setting_picture(shot, setting_ref, setting_mask) if setting_ref and setting_ref != "off" else None
        if setting is not None:
            if ref_image_size == "match":   # keep 'match' for the people, the setting keeps its own size
                for k, r in list(refs.items()):
                    sc = min(1.0, math.sqrt(shot["width"] * shot["height"] / (r.shape[1] * r.shape[2])))
                    if sc < 1.0:
                        refs[k] = torch.nn.functional.interpolate(
                            r[:1].movedim(-1, 1), size=(max(16, int(r.shape[1] * sc)), max(16, int(r.shape[2] * sc))),
                            mode="bilinear", align_corners=False).movedim(1, -1)
                if setting_ref == SETTING_MODES[1]:
                    sc = min(1.0, math.sqrt(shot["width"] * shot["height"] / (setting.shape[1] * setting.shape[2])))
                    if sc < 1.0:
                        setting = torch.nn.functional.interpolate(
                            setting.movedim(-1, 1), size=(int(setting.shape[1] * sc), int(setting.shape[2] * sc)),
                            mode="bilinear", align_corners=False).movedim(1, -1)
                ref_image_size = "max"
            refs[f"ref_image_{len(refs)}"] = setting
            text = add_setting(text, len(refs), "<Subject 1>" in text)
        text = fill_details_now(text, vlm, shot_refs(shot), f"[BFS Shot H3 Conditioning] shot {shot.get('index', 0) + 1}",
                                shot.get("ref_names"))
        if len(refs) > 9:   # H3's own node stops at 9; OpenJutsu-style LoRAs were trained with up to 9 pictures
            print(f"[BFS Shot H3 Conditioning] shot {shot.get('index', 0) + 1}: {len(refs)} reference pictures "
                  "(more than the 9 H3 and its LoRAs usually see)", flush=True)
        kwargs = dict(clip=clip, prompt=text, width=shot["width"], height=shot["height"],
                      length=shot["gen_length"], ref_image_size=ref_image_size, vae=vae,
                      audio_vae=audio_vae, ref_images=refs or None)
        use_mask = inpaint == "only the mask" or (inpaint == INPAINT_MODES[0] and shot.get("inpaint"))
        gtag = f"[BFS Shot H3 Conditioning] shot {shot.get('index', 0) + 1}/{shot.get('count', 1)}"
        # what the model sees of the source; the kept area (keep_video) stays the original
        area = None
        if source_blur and source_blur.startswith("masked area") and (shot.get("mask_src") or (shot.get("crop") or {}).get("crop_mask") is not None):
            area = generation_mask(shot, log=gtag)
        seen = blur_source(shot["frames"], source_blur, float(source_blur_strength), gtag, source_blur_segmenter,
                           source_blur_effect, area) if source_blur and source_blur != "off" else shot["frames"]
        if shot.get("color_guide"):
            if not shot.get("mask_src") and not shot.get("crop"):
                raise ValueError(f"{gtag}: Colored Mark needs a person selection (SAM 3 text/points or a mask video)")
            cmask = generation_mask(shot, log=gtag)
            seen = color_guide(seen, cmask, shot.get("color_guide_color", "#00ff00"),
                               shot.get("color_guide_opacity", 0.35))
            for mk, x in extra_mark_masks(shot, gtag):
                cmask2 = torch.nn.functional.interpolate(mk[:, None].float(), size=shot["frames"].shape[1:3],
                                                         mode="nearest")[:, 0]
                seen = color_guide(seen, cmask2, x["color"], shot.get("color_guide_opacity", 0.35))
                print(f"{gtag}: Colored Mark <Picture {x['picture']}> ({len(x.get('points') or [])} points"
                      f"{', ' + repr(x['text']) if x.get('text') else ''}), {x['color']} at "
                      f"{float(shot.get('color_guide_opacity', 0.35)):.2f}, {mask_report(cmask2)}", flush=True)
            print(f"{gtag}: Colored Mark {shot.get('color_guide_color', '#00ff00')} at "
                  f"{float(shot.get('color_guide_opacity', 0.35)):.2f} opacity, {mask_report(cmask)}", flush=True)
        pg = mask_guide if mask_guide in MASK_GUIDES[1:] else None
        if pg and (not use_mask or shot.get("crop")):
            print(f"{gtag}: mask_guide ignored: it is for 'Mask only' shots (mask, no crop)", flush=True)
            pg = None
        guide_px = None
        if pg:
            pmask = generation_mask(shot, log=gtag)
            guide_px = masked_only(shot["frames"], pmask, look=mask_guide_look)
            print(f"{gtag}: mask guide ({pg}, look: {mask_guide_look}), {mask_report(pmask)}", flush=True)
        if native or pg == MASK_GUIDES[3]:
            vids = {}
            if native:
                vids["ref_video_1"] = seen
                if audio is not None:
                    kwargs["ref_video_audios"] = {"ref_video_audio_1": audio}
            if pg == MASK_GUIDES[3]:
                sc = MASK_REF_SIZES.get(mask_ref_size, 0.5)
                if sc < 1.0:
                    H0, W0 = guide_px.shape[1:3]
                    size = (max(32, int(H0 * sc) // 32 * 32), max(32, int(W0 * sc) // 32 * 32))
                    guide_px = torch.nn.functional.interpolate(guide_px.movedim(-1, 1), size=size, mode="bilinear",
                                                               align_corners=False).movedim(1, -1).clamp(0, 1)
                print(f"{gtag}: masked-region reference video at {guide_px.shape[2]}x{guide_px.shape[1]} ({mask_ref_size})",
                      flush=True)
                vids[f"ref_video_{len(vids) + 1}"] = guide_px
                kwargs["prompt"] = add_mask_video(kwargs["prompt"], len(vids))
                print(f"{gtag}: the masked region is <Video {len(vids)}> (write {{mask_video}} in the prompt to place "
                      f"its tag, otherwise a sentence is added)", flush=True)
            kwargs["ref_videos"] = vids
        positive, latent = MiniMaxH3ReferenceToVideo.execute(**kwargs).args[:2]
        if pg in (MASK_GUIDES[1], MASK_GUIDES[2]):
            positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae,
                                                 image=guide_px).args[0]
            if pg == MASK_GUIDES[2]:
                aligned = False                        # the mask guide replaces the full one
        if aligned:
            positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae,
                                                 audio_vae=audio_vae if audio is not None else None,
                                                 image=seen, audio=audio).args[0]
        if first_frame != "none":
            positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae,
                                                 image=seen[:1]).args[0]
        prev = chain_image(shot)
        if prev is not None and shot.get("chain") == "first frame":
            positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae,
                                                 image=prev).args[0]
        shot.pop("panel", None)
        keep_mask = keep_video = None
        tag = gtag
        mode = ("Crop + mask" if shot.get("crop") else "Mask only") if use_mask else \
            ("Crop" if shot.get("crop") else "Frame + paste (BFS Shot Join pastes the person back)" if shot.get("paste") else "Full frame")
        print(f"{tag}: {mode}" + (f" (inpaint = {inpaint})" if use_mask else ""), flush=True)
        if use_mask:
            t0 = time.time()
            keep_mask, keep_video = generation_mask(shot, log=tag), shot["frames"]
            print(f"{tag}: generation mask {mask_report(keep_mask)}, ready in {time.time() - t0:.1f}s", flush=True)
        if keep_mask is not None:
            strength = float(shot.get("inpaint_strength") or 1.0)
            positive, latent = inpaint_latent(latent, positive, vae, keep_video, keep_mask, strength)
            nm = latent["noise_mask"].tensors[0]
            print(f"{tag}: {float((nm > 0).float().mean()) * 100:.1f}% of the latent is regenerated"
                  + (f" at mask opacity {strength:.2f} (keeps ~{100 - strength * 100:.0f}% of the original there)" if strength < 1 else "")
                  + f", the rest is kept ({int(nm.shape[2])} latent frames)", flush=True)
        positive = self._history(shot, positive, vae, gtag)
        latent = self._head(shot, latent, vae, gtag)
        return (positive, latent, model, text)

    @staticmethod
    def _head(shot, latent, vae, tag):
        """Continuity 'head': the shot's first frames are the previous result's end, written into the latent and kept
        by the sampler (noise mask 0), so they are not swapped again and the new frames continue from them."""
        n = int(shot.get("head") or 0)
        if not n:
            return latent
        hf = shot_head(shot)
        if hf is None:
            print(f"{tag}: head: no previous result, its {n} frames are generated from the guide (and cut by the join)",
                  flush=True)
            return latent
        W, H = int(shot["width"]), int(shot["height"])
        hf = _fit_to(hf, W, H) if tuple(hf.shape[1:3]) != (H, W) else hf
        try:
            from .bfs_h3_history import pin_head
        except ImportError:
            from bfs_h3_history import pin_head
        latent, used = pin_head(latent, vae, hf)
        if used:
            print(f"{tag}: head = the last {used} frames of the previous shot start this one, kept as they are "
                  f"(not generated again); BFS Shot Join cuts them", flush=True)
        return latent

    @staticmethod
    def _history(shot, positive, vae, tag):
        """Continuity 'history': the end of the previous result as clean frames just before this shot on H3's
        timeline (added last, so the other guides leave it alone)."""
        hist, _ = shot_history(shot)
        if hist is None:
            return positive
        W, H = int(shot["width"]), int(shot["height"])
        hist = _fit_to(hist, W, H) if tuple(hist.shape[1:3]) != (H, W) else hist
        try:
            from .bfs_h3_history import add_history
        except ImportError:
            from bfs_h3_history import add_history
        positive, n = add_history(positive, vae, hist)
        del hist
        if n:
            print(f"{tag}: history = the last {n} frames of the previous shot, before frame 0 on H3's timeline", flush=True)
        return positive


class BFSShotH3ConditioningSimple:
    """The common H3 shot-conditioning path, without the mask-guide and setting-picture controls."""

    GUIDE_MODES = BFSShotH3Conditioning.GUIDE_MODES

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot": ("BFS_SHOT",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "guide_mode": (cls.GUIDE_MODES, {"default": cls.GUIDE_MODES[0], "tooltip":
                    "aligned guide: preserve the shot frame by frame with MiniMax H3 Add Guide. "
                    "native reference video: provide it as <Video 1> context. both: use both paths."}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "with_audio": ("BOOLEAN", {"default": False, "tooltip":
                    "Condition on the shot soundtrack too. This can improve lip sync when audio_vae is connected."}),
                "vlm": ("CLIP", {"tooltip":
                    "Optional Qwen3-VL encoder. Fills missing reference details in the planner prompt."}),
                "inpaint": (INPAINT_MODES, {"default": INPAINT_MODES[0], "tooltip":
                    "Use each shot's Planner mask mode, force mask-only generation, or disable it."}),
                "source_blur": (SOURCE_BLUR, {"default": "off", "tooltip":
                    "Blur the source subject in the guide so the model is less likely to copy its identity: "
                    "'<region> (<what stays sharp>)'; masked area = the shot's own mask."}),
                "source_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "source_blur_segmenter": (BLUR_SEGMENTERS, {"default": BLUR_SEGMENTERS[0], "tooltip":
                    "What finds the face / head / hands / person and the mouth for source_blur. SAM 3.1: text prompts "
                    "(already used by the planner). Sapiens2: body-part classes in one pass, finer lips and teeth "
                    "(downloads facebook/sapiens2-seg-0.4b, 1.6 GB, to models/sapiens2 the first time). The 'keep lips + "
                    "tongue' modes always use Sapiens2 for the lips."}),
                "source_blur_effect": (BLUR_EFFECTS, {"default": BLUR_EFFECTS[0], "tooltip":
                    "blur: no detail left. invert colours: keeps the shapes and the expression but not the skin tone "
                    "or the colours."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "latent", "prompt")
    FUNCTION = "condition"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Build the usual MiniMax H3 shot conditioning with references, prompt and an aligned guide "
                   "and/or native reference video. Use the Advanced node for the mask guide, setting picture and first frame.")

    def condition(self, shot, clip, vae, guide_mode, ref_image_size,
                  audio_vae=None, with_audio=False, vlm=None, inpaint=INPAINT_MODES[0],
                  source_blur="off", source_blur_strength=1.0, source_blur_segmenter="SAM 3.1", source_blur_effect="blur"):
        positive, latent, _, prompt = BFSShotH3Conditioning().condition(
            shot, clip, vae, guide_mode, "none", ref_image_size,
            audio_vae=audio_vae, with_audio=with_audio, vlm=vlm, inpaint=inpaint,
            source_blur=source_blur, source_blur_strength=source_blur_strength,
            source_blur_segmenter=source_blur_segmenter, source_blur_effect=source_blur_effect)
        return positive, latent, prompt


class BFSShotJoin:
    """Put the decoded shots back together in order, trimmed to their true lengths."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"lazy": True, "tooltip": "Decoded frames of every shot (the list from VAE Decode). "
                                     "Not requested when every shot is already stored: nothing is generated again."}),
                "shots": ("BFS_SHOT", {"tooltip": "The shots list from BFS Shot Planner."}),
                "crossfade": ("INT", {"default": 4, "min": 0, "max": 24, "tooltip":
                    "Frames to blend across boundaries that are not camera cuts, using the overlap each "
                    "shot generated past its end. 0 = hard joins everywhere."}),
            },
            "optional": {"audio": ("AUDIO", {"tooltip": "Soundtrack to return with the video "
                                                       "(e.g. the planner's audio output)."}),
                         "timeline": ("BFS_SHOT_TIMELINE", {"tooltip": "The planner's timeline. With it, shots that "
                             "did not run are filled with the original video (or dropped, per the planner's setting) "
                             "and the soundtrack follows."}),
                         "comparison": ("BOOLEAN", {"default": False, "tooltip": "Also output a side-by-side video: "
                             "original shot | references | result, with the shot's info on top and its prompt below."}),
                         "label": ("STRING", {"default": "", "multiline": False, "tooltip": "Extra text for the "
                             "comparison's top bar, e.g. the model, LoRA, steps and seed."}),
                         "comparison_height": ("INT", {"default": 360, "min": 0, "max": 4096, "step": 16, "tooltip":
                             "Height of each column in the comparison (0 = full size). Smaller is much lighter "
                             "for long videos."}),
                         "edge_hardness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                             "Cropped / Frame + paste shots: harden the soft edge of the paste (0 = as feathered, 1 = hard cut). Raise it "
                             "when a faint ghost of the original person shows around the new one. From NKD Inpaint Stitch."}),
                         "match_colors": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                             "Cropped / Frame + paste shots: correct the colour / brightness drift of the generated patch (Reinhard in "
                             "LAB, statistics over the whole shot, so no flicker). 0.5-1 typical. From NKD Inpaint Stitch."}),
                         "match_region": (MATCH_REGIONS, {"default": MATCH_REGIONS[0], "tooltip":
                             "Where the colour statistics are measured. 'around the subject (swap)': a ring of background "
                             "around the mask, so a new person keeps their own colours and only the scene's light drift "
                             "is fixed. 'inside the subject': inside the mask (retouching the same person/content)."}),
                         "seamless_edges": ("BOOLEAN", {"default": True, "tooltip": "Cropped / Frame + paste shots: Poisson blend "
                             "(OpenCV seamlessClone) for stubborn seams. Slower; can shift colours near the edge. From "
                             "NKD Inpaint Stitch."}),
                         "paste_new_outline": ("BOOLEAN", {"default": True, "tooltip": "'Frame + paste' shots: also paste "
                             "where the NEW subject is (SAM 3 on the result near the original one, with the shot's "
                             "'new subject' text from the planner, default 'person'), so a bigger silhouette (hair, "
                             "body) is not cut. Off = only the original mask (+ expand)."}),
                         "comparison_mask": ("BOOLEAN", {"default": True, "tooltip": "In the comparison, shots with "
                             "a crop or a generation mask (Mask only / Crop + mask) show the selected mask over the "
                             "original column; Colored Mark uses its selected colors, other masks use red. Crop boxes "
                             "are yellow."}),
                         "keep_colors": ("BOOLEAN", {"default": False, "tooltip": "Shot by shot / queue loop with "
                             "continuity: shots that continue from the previous one can drift in colour over a long take. "
                             "On: the new subject's colours (chroma) of every continuing shot are matched to the first "
                             "shot of the take before it is stored (the background and the lightness are not touched; a "
                             "camera cut or a shot without continuity starts a new take). Off by default: the match is one "
                             "shift for the whole shot, so with continuity 'head' (whose first frames already continue the "
                             "previous shot) it shows as a colour jump at every seam. Turn it on only for a long take "
                             "whose colours visibly drift."}),
                         "keep_colors_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                             "tooltip": "How much of that correction is applied (1 = fully to the first shot)."}),
                         "save_to": (SAVE_TO, {"default": SAVE_TO[0], "tooltip": "images: output the video as IMAGE "
                             "(connect Create Video / Save Video); the whole video is held in memory. mp4 file: the join "
                             "writes the video (and the comparison) straight to MP4 files in the output folder, one shot "
                             "at a time, so memory stays at about one shot: use it for long videos. The images and comparison "
                             "outputs still carry the video, read back from the files at a size that fits the free RAM "
                             "(the files keep the full size), the node shows the files, and video_path gives the path."}),
                         "filename_prefix": ("STRING", {"default": "bfs_shot_loop", "tooltip":
                             "mp4 file: the file name in the output folder (a counter is added)."}),
                         "mp4_crf": ("INT", {"default": 17, "min": 0, "max": 51, "tooltip":
                             "mp4 file: H.264 quality (lower = better and bigger; 0 = lossless, 17 = visually "
                             "lossless, 23 = ffmpeg's default)."})},
            # recursive run: the join expands the graph for the next shot
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("images", "audio", "fps", "comparison", "video_path", "guide")
    OUTPUT_TOOLTIPS = (
        "The joined video.", "Its soundtrack.", "Its frame rate.",
        "original | references | result, side by side (needs 'comparison').",
        "The MP4 written in 'mp4 file' mode (empty otherwise).",
        "The original frames under each output frame, same length and timing as 'images', at comparison_height "
        "(needs 'comparison'): connect it with 'images' to BFS Video Compare.")
    FUNCTION = "join"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = "Concatenate the generated shots in order, trim each to its length and cross-fade soft joins."

    def check_lazy_status(self, images=None, shots=None, **kwargs):
        """Shot by shot / queue loop with every shot stored and up to date: the join assembles them from disk, so the
        decoded images (and the sampler behind them) are not requested."""
        if shots and shots[0].get("assemble_only"):
            return []
        return ["images"]

    def join(self, images=None, shots=None, crossfade=None, audio=None, timeline=None, comparison=None, label=None,
             comparison_height=None, comparison_mask=None, edge_hardness=None, match_colors=None, match_region=None,
             seamless_edges=None, paste_new_outline=None, keep_colors=None, keep_colors_strength=None, save_to=None,
             filename_prefix=None, mp4_crf=None, dynprompt=None, unique_id=None, prompt=None, extra_pnginfo=None):
        tl = timeline[0] if timeline else None
        self._keep_colors = (bool(keep_colors[0]) if keep_colors else False,
                             float(keep_colors_strength[0]) if keep_colors_strength else 1.0)
        # the MP4s the join writes carry the workflow like Save Video's (drag one back into ComfyUI)
        self._file_meta = dict((extra_pnginfo[0] if extra_pnginfo and extra_pnginfo[0] else None) or {})
        if prompt and prompt[0] is not None:
            self._file_meta["prompt"] = prompt[0]
        self._to_file = None
        self._prefix = (filename_prefix[0] if filename_prefix else "") or "bfs_shot_loop"
        if save_to and save_to[0] == SAVE_TO[1]:
            self._to_file = {"path": _next_output_path((filename_prefix[0] if filename_prefix else "") or "bfs_shot_loop"),
                             "crf": int(mp4_crf[0]) if mp4_crf else 17}
        self._label = None
        self._dynprompt = dynprompt[0] if dynprompt else None
        self._unique_id = unique_id[0] if unique_id else None
        fin = dict(edge_hardness=float(edge_hardness[0]) if edge_hardness else 0.0,
                   match_colors=float(match_colors[0]) if match_colors else 0.0,
                   match_region=match_region[0] if match_region else MATCH_REGIONS[0],
                   seamless=bool(seamless_edges[0]) if seamless_edges else True)
        self._comp_mask = bool(comparison_mask[0]) if comparison_mask else True
        want = bool(comparison[0]) if comparison else False
        self._comp_h = int(comparison_height[0]) if comparison_height else 360
        self._want = want
        lab = (label[0] if label else "") or ""
        self._parts = []   # (shot or None, frames in the output, original frames) per piece, for the comparison
        if shots and shots[0].get("assemble_only"):   # nothing was generated: the video from the stored shots
            shot = shots[0]
            d = run_dir(shot["run_id"])
            print(f"[BFS Shot Join] every shot is stored and up to date: assembling {shot['count']} shot(s) "
                  "without generating", flush=True)
            metas = [load_shot(d, i, frames=False)[1] for i in range(shot["count"])]
            self._run_dir, self._label = d, lab
            return self._outputs(self._assemble(metas, lambda j: load_shot(d, int(metas[j]["index"]))[0], crossfade,
                                                audio, tl, self._to_file, lab))
        images = [crop_panel(img, sh) for img, sh in zip(images, shots)]
        outline = bool(paste_new_outline[0]) if paste_new_outline else True
        images = [uncrop(img, sh, **fin) if sh.get("crop") and sh.get("full_frames") is not None
                  else paste_back(img, sh, outline, **fin) if sh.get("paste") and sh.get("mask_src") else img
                  for img, sh in zip(images, shots)]
        vram_record(max((shot_tokens(int(sh.get("gen_length") or img.shape[0]), int(sh.get("width") or img.shape[2]),
                                     int(sh.get("height") or img.shape[1]), sh.get("grid") or DEFAULT_GRID)
                         for img, sh in zip(images, shots)), default=0))
        cut = [drop_head(img, sh) for img, sh in zip(images, shots)]
        images, shots = [c[0] for c in cut], [c[1] for c in cut]
        self._label = lab
        if shots and shots[0].get("queue"):
            out = self._join_queue(images, shots, crossfade, audio, tl)
            if isinstance(out, dict):   # recursive run, not the last shot: the graph for the next one
                return out
            if len(out) == 3 and not isinstance(out[0], torch.Tensor):   # queue loop, not the last shot
                return out + (out[0],) * (len(self.RETURN_TYPES) - 3)
        else:
            if len(images) != len(shots):
                raise ValueError(f"BFS Shot Join: got {len(images)} image batches for {len(shots)} shots. "
                                 "Connect the decoded images of the same shot list.")
            if shots and shots[0].get("unload_before_join", True):
                _release_between_shots(True, "before the join")
            out = self._assemble(list(shots), lambda j: images[j], crossfade, audio, tl, self._to_file, lab)
        return self._outputs(out)

    def _guide_blocker(self):
        from comfy_execution.graph_utils import ExecutionBlocker
        return ExecutionBlocker("BFS Shot Join: the 'guide' output needs 'comparison' turned on (for a shot-by-shot "
                                "run, before the shots are generated: the originals are kept with each shot)")

    def _outputs(self, out):
        video, a, fps, comp = out
        if self._to_file:
            path, comp_path, guide_path = self._files
            print(f"[BFS Shot Join] video written to {path}" + (f", comparison to {comp_path}" if comp_path else ""),
                  flush=True)
            # the outputs still carry the video (and the comparison), read back from the files at a size that fits the
            # free RAM, so the nodes after the join always show the result; the files keep the full quality
            budget = 0.35 * _ram_available()
            sh_v, sh_c, sh_g = (0.5, 0.3, 0.2) if guide_path else (0.6 if comp_path else 1.0, 0.4, 0.0)
            try:
                video, sv = _read_video(path, budget * sh_v)
                comp, sc = _read_video(comp_path, budget * sh_c) if comp_path else (video[:1], 1.0)
                guide = _read_video(guide_path, budget * sh_g)[0] if guide_path else self._guide_blocker()
            except Exception as e:  # noqa: BLE001 - the files are written; only the preview failed
                from comfy_execution.graph_utils import ExecutionBlocker
                print(f"[BFS Shot Join] outputs not filled ({e}): see the files", flush=True)
                blocker = ExecutionBlocker(None)
                return (blocker, a, fps, blocker, path, blocker)
            if sv < 1 or (comp_path and sc < 1):
                print(f"[BFS Shot Join] outputs at {sv * 100:.0f}% size" + (f" (comparison {sc * 100:.0f}%)" if comp_path else "")
                      + " to fit the free RAM; the MP4 files keep the full size", flush=True)
            # no preview on the node: the video goes out through `images` (and the files) to whatever shows it
            return (video, a, fps, comp, path, guide)
        guide = getattr(self, "_guide", None)
        return (video, a, fps, comp if comp is not None else video[:1], "",
                guide if guide is not None else self._guide_blocker())

    def _join_queue(self, images, shots, crossfade, audio, tl=None):
        from comfy_execution.graph_utils import ExecutionBlocker
        shot = shots[0]
        rid = shot["run_id"]
        d = run_dir(rid)
        img = images[0]
        meta = {k: v for k, v in shot.items() if k in ("index", "count", "start", "end", "length", "gen_length",
                                                         "fps", "cut_before", "prompt", "ref", "ref2")}
        if getattr(self, "_want", False):   # originals only for the comparison: small and uint8
            src = shot.get("full_frames") if shot.get("full_frames") is not None else shot.get("frames")
            if src is not None:
                h = getattr(self, "_comp_h", 360) or src.shape[1]
                sc = min(1.0, h / src.shape[1])
                small = _scale_batch(src, max(16, int(src.shape[1] * sc)), max(16, int(src.shape[2] * sc)))
                meta["frames"] = (small * 255).round().to(torch.uint8)
                if shot.get("color_guide"):
                    try:
                        meta["color_guide"] = True
                        meta["color_guide_color"] = shot.get("color_guide_color") or "#00ff00"
                        meta["color_guide_opacity"] = shot.get("color_guide_opacity") or 0.35
                        meta["comparison_mark_regions"] = _comparison_mark_regions(shot, src, h)
                    except Exception as e:  # keep the video join available if its optional comparison overlay fails
                        print(f"[BFS Shot Join] no Colored Mark overlay for shot {shot.get('index', 0) + 1}: {e}")
        img = self._anchor_colors(img, shot, rid)
        store_shot(d, int(shot["index"]), (img.clamp(0, 1) * 255).round().to(torch.uint8).cpu(), meta,
                   shot.get("shot_storage") or "lossless")
        st = run_state(rid)
        i = int(shot["index"])
        if i not in st["done"]:
            st["done"].append(i)
        st["count"] = shot["count"]
        take = hashlib.sha1(f"{rid}:{i}:{time.time()}".encode()).hexdigest()[:10]   # this take of the shot
        st.setdefault("sig", {})[str(i)] = shot.get("sig", "")
        st.setdefault("gen", {})[str(i)] = take
        st.setdefault("from", {})[str(i)] = shot.get("prev_gen", "")
        if st.get("keep_next", {}).pop(str(i), None) and str(i + 1) in st.get("from", {}):
            st["from"][str(i + 1)] = take   # Redo "only this shot": the next shot keeps its take
        save_state(rid, st)
        if st.get("cur"):   # stored and valid shots of the current plan (a new take invalidates the shots after it)
            ok = shot_status(st, d)
            done = sum(ok)
            if done < len(ok):
                print(f"[BFS Shot Join] shot {i + 1} stored; {len(ok) - done} shot(s) still to run "
                      f"(next: shot {ok.index(False) + 1})", flush=True)
            else:
                print(f"[BFS Shot Join] shot {i + 1} stored; every shot is stored: putting the video together", flush=True)
        else:
            done = len(set(st["done"]))
        _notify("bfs-shotloop-progress", {"run_id": rid, "done": done, "count": shot["count"]})
        if done < shot["count"]:
            if shot.get("run_mode") == "recursive" and getattr(self, "_dynprompt", None) is not None:
                del images, shots
                _release_between_shots(shot.get("between_shots") == "free")
                return self._recurse(self._dynprompt, str(self._unique_id))
            if shot.get("between_shots") == "free":
                _release_between_shots(True)
            _notify("bfs-shotloop-next", {"run_id": rid, "done": done, "count": shot["count"]})
            blocker = ExecutionBlocker(None)
            return (blocker, blocker, blocker)
        # the whole video from the stored shots, one shot in memory at a time
        del images, shots, img
        _release_between_shots(bool(shot.get("unload_before_join", True)), "before the join")
        metas = [load_shot(d, i, frames=False)[1] for i in range(shot["count"])]
        self._run_dir = d
        return self._assemble(metas, lambda j: load_shot(d, int(metas[j]["index"]))[0], crossfade, audio, tl,
                              self._to_file, getattr(self, "_label", "") or "")

    def _anchor_colors(self, img: torch.Tensor, shot: dict, rid: str) -> torch.Tensor:
        """Keep colours: the first shot of a take records its subject's colours; every shot that continues from the one
        before is matched to them before it is stored (see anchor_colors)."""
        on, strength = getattr(self, "_keep_colors", (False, 1.0))
        src = shot.get("full_frames") if shot.get("full_frames") is not None else shot.get("frames")
        if not on or src is None:
            return img
        i, L = int(shot["index"]), int(shot.get("length") or img.shape[0])
        st = run_state(rid)
        anchors = st.setdefault("color_anchors", {})
        starts = i == 0 or not uses_previous(shot)
        if not starts and anchors.pop(str(i), None) is not None:   # no longer the start of a take
            save_state(rid, st)
        prev = [int(k) for k in anchors if int(k) < i]
        if starts or not prev:
            stats = color_stats(img, src, L)
            if stats:
                anchors[str(i)] = stats
                save_state(rid, st)
            return img
        a = anchors[str(max(prev))]
        try:
            out, own = anchor_colors(img, src, a, L, strength)
        except Exception as e:   # never lose a shot over its colour
            print(f"[BFS Shot Join] shot {i + 1}: colours not anchored ({e})", flush=True)
            return img
        if own:
            sat = lambda s_: float(np.hypot(*s_["sd"]))
            print(f"[BFS Shot Join] shot {i + 1}: subject colours anchored to shot {max(prev) + 1} "
                  f"(a {own['mu'][0] - a['mu'][0]:+.1f}, b {own['mu'][1] - a['mu'][1]:+.1f}, "
                  f"spread x{sat(own) / sat(a):.2f} -> x1.00 at strength {strength:g})", flush=True)
        return out

    def _recurse(self, dynprompt, unique_id: str) -> dict:
        """Recursive run: the nodes between BFS Shot Planner and this join (the shot's body: conditioning, sampler,
        decode...), cloned with ComfyUI's node expansion, so the same prompt renders the next shot after this one has
        been stored. The planner's clone reads the stored shots and hands out the next one, so each shot can read the
        result of the one before (continuity head / history). Nodes outside the body (models, loaders) are shared.
        Graph walk adapted from Ethanfel's MiniMax H3 Context Loop / SxCP loop nodes (GPL-3.0)."""
        from comfy_execution.graph_utils import GraphBuilder, is_link
        feeds: dict[str, list[str]] = {}   # node -> the nodes it feeds, among this join's ancestors

        def explore(nid):
            for v in dynprompt.get_node(nid).get("inputs", {}).values():
                if is_link(v):
                    pid = str(v[0])
                    if pid not in feeds:
                        feeds[pid] = []
                        explore(pid)
                    feeds[pid].append(nid)

        explore(unique_id)
        planners = [n for n in feeds if dynprompt.get_node(n).get("class_type") == "BFSShotPlanner"]
        if len(planners) != 1:
            raise ValueError(f"BFS Shot Join: the recursive run needs exactly one BFS Shot Planner before the join "
                             f"(found {len(planners)}). Use the queue loop for this graph.")
        body, stack = {unique_id, planners[0]}, [planners[0]]
        while stack:
            for child in feeds.get(stack.pop(), []):
                if child not in body:
                    body.add(child)
                    stack.append(child)
        # output nodes fed by the body (e.g. a preview of each decoded shot) run for every shot too
        shown = {dynprompt.get_display_node_id(n): n for n in body if n != unique_id}
        outputs = {}
        try:
            import nodes as comfy_nodes
            for oid, node in dynprompt.get_original_prompt().items():
                cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(node.get("class_type"))
                if cls is not None and getattr(cls, "OUTPUT_NODE", False) and oid not in shown and any(
                        is_link(v) and str(v[0]) in shown for v in node.get("inputs", {}).values()):
                    outputs[oid] = node
        except Exception as e:   # previews are optional
            print(f"[BFS Shot Join] recursive run: shot previews not repeated ({e})", flush=True)
        graph = GraphBuilder()
        clone = {n: ("Recurse" if n == unique_id else n) for n in body}
        for n in body:
            graph.node(dynprompt.get_node(n)["class_type"], clone[n]).set_override_display_id(
                dynprompt.get_display_node_id(n))
        for n in body:
            node = graph.lookup_node(clone[n])
            for key, v in dynprompt.get_node(n).get("inputs", {}).items():
                node.set_input(key, graph.lookup_node(clone[str(v[0])]).out(v[1]) if is_link(v) and str(v[0]) in body
                               else v)
        for oid, onode in outputs.items():
            node = graph.node(onode["class_type"], f"shot_out_{oid}")
            node.set_override_display_id(oid)
            for key, v in onode.get("inputs", {}).items():
                node.set_input(key, graph.lookup_node(clone[shown[str(v[0])]]).out(v[1])
                               if is_link(v) and str(v[0]) in shown else v)
        rec = graph.lookup_node("Recurse")
        print(f"[BFS Shot Join] recursive run: next shot ({len(body)} nodes cloned"
              f"{', ' + str(len(outputs)) + ' preview(s)' if outputs else ''})", flush=True)
        return {"result": tuple(rec.out(i) for i in range(len(self.RETURN_TYPES))), "expand": graph.finalize()}

    def _join_all(self, images, shots, crossfade, audio=None, tl=None):
        """The shots in memory, joined (video, audio, fps)."""
        if len(images) != len(shots):
            raise ValueError(f"BFS Shot Join: got {len(images)} image batches for {len(shots)} shots. "
                             "Connect the decoded images of the same shot list.")
        return self._assemble(list(shots), lambda j: images[j], crossfade, audio, tl)[:3]

    def _assemble(self, metas: list, frames_of, crossfade, audio=None, tl=None, to_file: dict | None = None,
                  label: str = ""):
        """Put the shots together one at a time: each is read (frames_of(j): uint8 or float frames of metas[j]),
        trimmed to its length, cross-faded with the one before when no cut separates them, and written into the
        final video, so memory holds the output plus one shot (not every shot twice over). With a timeline, shots
        that did not run (or are missing) are the original video, or dropped, and the soundtrack follows.
        to_file ({"path", "crf"}): the video (and the comparison, when asked) are streamed to MP4 files instead, so
        memory stays at about one shot whatever the length. Returns (video or None, audio, fps, comparison or None)."""
        xf = int(crossfade[0]) if isinstance(crossfade, (list, tuple)) else int(crossfade or 0)
        order = sorted(range(len(metas)), key=lambda j: int(metas[j]["index"]))
        by_idx = {int(metas[j]["index"]): j for j in order}
        on_tl = tl is not None and any(not g["run"] or g["run_index"] not in by_idx for g in tl["segs"])
        fill = (tl or {}).get("fill", "original")
        if on_tl:
            pieces = [(by_idx[g["run_index"]] if g["run"] and g["run_index"] in by_idx else None, g) for g in tl["segs"]]
            pieces = [(j, g) for j, g in pieces if j is not None or fill != "drop"]
        else:
            pieces = [(j, None) for j in order]
        if not pieces:
            raise ValueError("BFS Shot Join: nothing to join")
        fps = float(tl["fps"]) if on_tl else float(metas[order[0]].get("fps") or 24.0)
        length = lambda j, g: int(metas[j]["length"]) if j is not None else int(g["end"]) - int(g["start"])
        total = sum(length(j, g) for j, g in pieces)

        # the soundtrack first (an MP4 is written with it in one pass)
        a = audio[0] if audio else None
        if on_tl:
            if a is None or (isinstance(a, dict) and a.get("bfs_runs_only") and tl.get("audio") is not None):
                a = tl.get("audio") or a   # the planner's own soundtrack covers only the shots that ran
            if _usable(a) and fill == "drop":
                sr, base = a["sample_rate"], tl["segs"][0]["start"]
                a = {"waveform": torch.cat([a["waveform"][..., int(round((int(g["start"]) - base) / fps * sr)):
                                                          int(round((int(g["start"]) - base + length(j, g)) / fps * sr))]
                                            for j, g in pieces], -1), "sample_rate": sr}
        if _usable(a):
            n = int(round(total / fps * a["sample_rate"]))
            wf = a["waveform"][..., :n]
            if wf.shape[-1] < n:
                wf = torch.nn.functional.pad(wf, (0, n - wf.shape[-1]))
            a = {"waveform": wf, "sample_rate": a["sample_rate"]}
        else:
            a = silence(total / fps)

        want = bool(getattr(self, "_want", False))
        comp_h, comp_mask = getattr(self, "_comp_h", 360), getattr(self, "_comp_mask", True)
        count = sum(1 for j, _ in pieces if j is not None)
        first = frames_of(order[0])
        H, W = int(first.shape[1]), int(first.shape[2])
        if not to_file:   # an IMAGE output holds the whole video (and the comparison): never more than the RAM left
            need = _join_bytes(total, H, W, want, comp_h)
            _make_room(need)
            free = _ram_available()
            if need > 0.85 * free:
                to_file = {"path": _next_output_path(getattr(self, "_prefix", "") or "bfs_shot_loop"), "crf": 17}
                self._to_file = to_file
                print(f"[BFS Shot Join] the video{' and its comparison' if want else ''} would need ~{need / 2 ** 30:.1f} GB "
                      f"of RAM as images, {free / 2 ** 30:.1f} GB is free: writing MP4 files instead ({to_file['path']}). "
                      f"For long videos set save_to = mp4 file{', or a lower comparison_height' if want else ''}.",
                      flush=True)
        video = None if to_file else torch.empty((total, H, W, 3), dtype=torch.float32)
        vw = cw = None
        comp_size = None
        comp_parts, parts = [], []
        gw = gw_size = None
        self._guide = None
        if to_file:
            vw = _Mp4Writer(to_file["path"], W, H, fps, a, int(to_file.get("crf", 17)), getattr(self, "_file_meta", None))
            if want:
                cw_path = to_file["path"][:-4] + "_comparison.mp4"
                gw_path = to_file["path"][:-4] + "_guide.mp4"
        pos, k, prev_tail = 0, 0, None
        try:
            for j, g in pieces:
                L = length(j, g)
                if j is not None:
                    fr = first if j == order[0] and first is not None else frames_of(j)
                    if j == order[0]:
                        first = None
                    meta = metas[j]
                    body = fr[:L]
                    if body.shape[0] < L:   # shorter than planned: hold the last frame
                        body = torch.cat([body, body[-1:].expand(L - body.shape[0], -1, -1, -1)], 0)
                    body = body[..., :3].float() / 255.0 if body.dtype == torch.uint8 else body[..., :3].float()
                    if tuple(body.shape[1:3]) != (H, W):
                        body = _scale_batch(body, H, W)
                    cut = (g or meta).get("cut_before")
                    if xf > 0 and prev_tail is not None and not cut:
                        m = min(xf, prev_tail.shape[0], body.shape[0])
                        if m > 0:
                            w = torch.linspace(0, 1, m + 2)[1:-1].view(-1, 1, 1, 1)
                            body = body.clone()
                            body[:m] = prev_tail[:m] * (1 - w) + body[:m] * w
                    tail = fr[L:L + max(xf, 0)]
                    prev_tail = None if not tail.shape[0] else (
                        tail[..., :3].float() / 255.0 if tail.dtype == torch.uint8 else tail[..., :3].float())
                    if prev_tail is not None and tuple(prev_tail.shape[1:3]) != (H, W):
                        prev_tail = _scale_batch(prev_tail, H, W)
                    del fr
                    k += 1
                    orig = meta.get("full_frames") if meta.get("full_frames") is not None else meta.get("frames")
                    if orig is None and want and meta.get("frames_video") and getattr(self, "_run_dir", None):
                        orig = shot_originals(self._run_dir, meta)   # stored shot: its original column, read now
                    if orig is None and want and tl is not None and meta.get("start") is not None:
                        # stored while the comparison was off (e.g. turned on afterwards, the join assembling the
                        # stored shots): the original column read from the source video on the timeline
                        orig = shot_source_frames(tl, int(meta["start"]), L, H, W, comp_h)
                    part = (meta, L, orig)
                else:
                    prev_tail = None
                    src = np.asarray(tl["src"])
                    body = torch.from_numpy(np.stack(_read_frames(tl["path"], src[int(g["start"]):int(g["end"])], (W, H)))
                                            .astype(np.float32) / 255.0)
                    small = body if not want else _scale_batch(body, min(H, comp_h or H),
                                                                 max(16, int(W * min(H, comp_h or H) / H)))
                    part = ({"skipped": True, "start": g["start"], "end": g["end"], "fps": fps,
                             "cut_before": g["cut_before"]}, L, small if want else None)
                if to_file:
                    vw.write(body)
                    if want:   # streamed 16 frames at a time, at the first shot's canvas size
                        def sink(chunk):
                            nonlocal cw
                            if cw is None:
                                cw = _Mp4Writer(cw_path, chunk.shape[2], chunk.shape[1], fps, a, int(to_file.get("crf", 17)),
                                                getattr(self, "_file_meta", None))
                            cw.write(chunk)
                        comp_size = comparison_video(body, [part], label, comp_h, comp_mask, first_k=k - (j is not None),
                                                     count=count, sink=sink, canvas=comp_size)

                        def gsink(chunk):   # the guide: the same originals, alone, at the comparison's column size
                            nonlocal gw
                            if gw is None:
                                gw = _Mp4Writer(gw_path, chunk.shape[2], chunk.shape[1], fps, None,
                                                int(to_file.get("crf", 17)))
                            gw.write(chunk)
                        gw_size = guide_video(body, [part], comp_h, sink=gsink, size=gw_size)
                else:
                    video[pos:pos + L] = body
                    parts.append(part)
                pos += L
                del body
            path = vw.close() if vw else None
            comp_path = cw.close() if cw else None
            guide_path = gw.close() if gw else None
        except BaseException:
            for w_ in (vw, cw, gw):
                if w_:
                    w_.abort()
            raise
        if to_file:
            self._files = (path, comp_path, guide_path)
            return None, a, fps, None
        comp = None
        if want:
            comp = comparison_video(video, parts, label, comp_h, comp_mask)
            self._guide = guide_video(video, parts, comp_h)
        return video, a, fps, comp


def cache_status(plan: dict, reset: bool = False) -> dict:
    """The panel's view of the shot cache of a plan's video (as of the last time the planner ran)."""
    path = _input_path(plan["video"])
    rid = run_key(plan, path)
    d = run_dir(rid, create=False)
    if reset:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    if not os.path.isdir(d):   # nothing stored for this video (no folder is made just to look)
        return {"run_id": rid, "folder": d, "count": 0, "done": 0, "ok": [], "stored": [], "mb": 0.0,
                "temp": rid.startswith("tmp_")}
    st = run_state(rid)
    ok = shot_status(st, d) if st.get("cur") else []
    files = os.listdir(d)
    stored = sorted(int(f[5:9]) for f in files if f.startswith("shot_") and f.endswith(".pt"))
    size = sum(os.path.getsize(os.path.join(d, f)) for f in files if f.startswith("shot_"))
    return {"run_id": rid, "folder": d, "count": len(ok) or st.get("count", 0), "done": sum(ok),
            "ok": ok, "stored": stored, "mb": round(size / 1e6, 1), "temp": rid.startswith("tmp_")}


def redo_shots(plan: dict, shots: list[int], cascade: bool = True) -> dict:
    """Mark stored shots to run again on the next Queue. cascade: the shots that continue from them are redone too
    (automatically, once the new take exists); off: they keep their take (the join may show a seam there)."""
    rid = run_key(plan, _input_path(plan["video"]))
    st = run_state(rid)
    for i in shots:
        if int(i) in st.get("done", []):
            st["done"].remove(int(i))
        if cascade:
            st.get("keep_next", {}).pop(str(int(i)), None)
        else:
            st.setdefault("keep_next", {})[str(int(i))] = True
    save_state(rid, st)
    return cache_status(plan)


def combine_stored(plan: dict, crossfade: int = 4, crf: int = 17, prefix: str = "bfs_shot_loop") -> dict:
    """Assemble the stored shots of a plan's video into an MP4 (output folder), one shot in memory at a time, on the
    timeline of the last planner run: the soundtrack, skipped shots as original video, and shots not stored (or out of
    date) as original video too, so it also previews a run in progress."""
    rid = run_key(plan, _input_path(plan["video"]))
    d = run_dir(rid)
    tf = os.path.join(d, "timeline.pt")
    if not os.path.exists(tf):
        raise ValueError("no stored run for this video yet: Queue the workflow first")
    tl = torch.load(tf, map_location="cpu", weights_only=False)
    st = run_state(rid)
    ok = shot_status(st, d) if st.get("cur") else []
    runs = {g["run_index"]: g for g in tl["segs"] if g["run"]}
    metas = []
    for i, good in enumerate(ok):
        if good and i in runs:
            m = load_shot(d, i, frames=False)[1]
            if int(m.get("start", -1)) == int(runs[i]["start"]) and int(m.get("end", -1)) == int(runs[i]["end"]):
                metas.append(m)
    if not metas:
        raise ValueError("no stored shot is up to date with this plan yet")
    tl = dict(tl, segs=[dict(g, run=g["run"] and any(int(m["index"]) == g["run_index"] for m in metas))
                        for g in tl["segs"]])
    join = BFSShotJoin()
    join._want, join._parts = False, []
    out = {"path": _next_output_path(prefix), "crf": int(crf)}
    done = [0]

    def frames_of(j):
        done[0] += 1
        _status("Combining stored shots", done[0], len(metas), force=True)
        return load_shot(d, int(metas[j]["index"]))[0]

    join._assemble(metas, frames_of, int(crossfade), None, tl, out)
    path = join._files[0]
    root = getattr(folder_paths, "get_output_directory", folder_paths.get_temp_directory)()
    rel = os.path.relpath(path, root)
    return {"path": path, "filename": os.path.basename(rel), "subfolder": os.path.dirname(rel),
            "shots": len(metas), "of": len(ok)}


SAVE_TO = ["images (output)", "mp4 file (long videos, low memory)"]


def _output_root() -> str:
    return getattr(folder_paths, "get_output_directory", folder_paths.get_temp_directory)()


def _read_video(path: str, budget: float) -> tuple[torch.Tensor, float]:
    """A video file as IMAGE, scaled down (never up) so it takes at most `budget` bytes; streamed into one tensor."""
    return VIO.read_video_scaled(path, budget)


def _join_bytes(frames: int, h: int, w: int, comparison: bool, comp_h: int) -> int:
    """RAM an images-output join needs: the video as float32, plus the comparison (three columns at its height, which
    is never above the video's, with the info / prompt bars)."""
    need = frames * h * w * 3 * 4
    if comparison:
        ch = h if not comp_h or comp_h <= 0 else min(int(comp_h), h)
        cw = max(16, w * ch // h)
        need += frames * int(ch * 1.45) * (2 * cw + cw // 2) * 3 * 4 + frames * ch * cw * 3   # + its original column
    return int(need)


def _ram_available() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return 1 << 62


def _make_room(nbytes: int) -> None:
    """Ask ComfyUI's RAM-pressure cache to drop cached outputs (the earlier shots' frames, decoded videos...) until
    this much RAM is free, the same release ComfyUI does between nodes; nothing happens with other cache modes."""
    import gc
    try:
        import comfy.memory_management as cm
        release = getattr(cm, "extra_ram_release", None)
        if callable(release):
            freed = int(release(int(nbytes) + (2 << 30), free_active=True) or 0)
            if freed:
                print(f"[BFS Shot Join] freed {freed / 2 ** 30:.1f} GB of cached outputs for the join", flush=True)
    except Exception:  # noqa: BLE001
        pass
    gc.collect()


def _next_output_path(prefix: str) -> str:
    """<output>/<prefix>_00001.mp4, the first free number (the prefix may hold sub folders)."""
    out = getattr(folder_paths, "get_output_directory", folder_paths.get_temp_directory)()
    prefix = prefix.strip().strip("/\\") or "bfs_shot_loop"
    base = os.path.normpath(os.path.join(out, prefix))
    if not base.startswith(os.path.normpath(out)):
        raise ValueError("BFS Shot Join: filename_prefix must stay inside the output folder")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    i = 1
    while os.path.exists(f"{base}_{i:05d}.mp4"):
        i += 1
    return f"{base}_{i:05d}.mp4"


def _fit_canvas(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Crop / pad (dark) a batch to h x w: every chunk of a streamed comparison has the same size."""
    out = torch.full((x.shape[0], h, w, 3), 18 / 255)
    hh, ww = min(h, x.shape[1]), min(w, x.shape[2])
    out[:, :hh, :ww] = x[:, :hh, :ww, :3]
    return out


class _Mp4Writer(VIO.Mp4Writer):
    """Frames streamed to an MP4 (H.264, yuv420p, even size), with the soundtrack muxed in the same pass: ffmpeg when
    there is one, else PyAV (Windows portable installs often have no ffmpeg program)."""

    def __init__(self, path: str, w: int, h: int, fps: float, audio: dict | None, crf: int = 17,
                 metadata: dict | None = None):
        super().__init__(path, w, h, fps, audio if _usable(audio) else None, crf, metadata, who="BFS Shot Join")


def _font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _text_bar(lines: list[str], width: int, size: int, max_lines: int, fg=(235, 235, 235), bg=(18, 18, 22)):
    """A dark bar with word-wrapped lines, as a float tensor [h, width, 3]."""
    from PIL import Image, ImageDraw
    font = _font(size)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    wrapped = []
    for line in lines:
        words, cur = line.split(), ""
        for w in words:
            t = (cur + " " + w).strip()
            if probe.textlength(t, font=font) <= width - 2 * size and cur:
                cur = t
            elif not cur:
                cur = w
            else:
                wrapped.append(cur); cur = w
        wrapped.append(cur)
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = wrapped[-1][: max(0, len(wrapped[-1]) - 3)] + "..."
    lh = int(size * 1.3)
    img = Image.new("RGB", (width, lh * len(wrapped) + size), bg)
    d = ImageDraw.Draw(img)
    for i, t in enumerate(wrapped):
        d.text((size, size // 2 + i * lh), t, fill=fg, font=font)
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)


def _column_labels(columns: list[tuple[str, int]], size: int) -> torch.Tensor:
    """One row with each label centred over its column."""
    from PIL import Image, ImageDraw
    font = _font(size)
    total = sum(w for _, w in columns)
    img = Image.new("RGB", (total, int(size * 1.6)), (30, 30, 36))
    d = ImageDraw.Draw(img)
    x = 0
    for name, w in columns:
        tw = d.textlength(name, font=font)
        d.text((x + (w - tw) / 2, size * 0.25), name, fill=(255, 210, 90), font=font)
        x += w
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)


def _fit_box(img: torch.Tensor | None, w: int, h: int) -> torch.Tensor:
    """[1,H,W,3] or None -> [h,w,3], letterboxed on dark grey."""
    out = torch.full((h, w, 3), 0.1)
    if img is None:
        return out
    x = img[:1, ..., :3].movedim(-1, 1).float()
    s = min(w / x.shape[-1], h / x.shape[-2])
    nw, nh = max(1, int(x.shape[-1] * s)), max(1, int(x.shape[-2] * s))
    x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)[0].movedim(0, -1)
    out[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw] = x.clamp(0, 1)
    return out


def _scale_batch(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """[N,H,W,C] -> [N,h,w,3] float on the CPU (area filter when shrinking)."""
    x = x[..., :3].float().cpu()
    if tuple(x.shape[1:3]) == (h, w):
        return x
    mode = "area" if x.shape[1] >= h else "bilinear"
    y = torch.nn.functional.interpolate(x.movedim(-1, 1), size=(h, w), mode=mode,
                                        **({} if mode == "area" else {"align_corners": False}))
    return y.movedim(1, -1).clamp(0, 1)


def _mask_overlay(o: torch.Tensor, crop: dict, L: int, start: int = 0) -> torch.Tensor:
    """Overlay selection masks and an optional crop box on the original comparison column [L,H,W,3] (frames start ..
    start + L of the shot)."""
    H, W = o.shape[1:3]
    regions = crop.get("regions") or [{"mask": crop.get("mask"), "color": "#ff283c", "opacity": 0.45}]
    for region in regions:
        m = region.get("mask")
        if m is None and region.get("packed") is not None:
            shape = tuple(region.get("shape") or ())
            if len(shape) == 3:
                packed = region["packed"]
                packed = packed.cpu().numpy() if isinstance(packed, torch.Tensor) else np.asarray(packed)
                m = torch.from_numpy(np.unpackbits(packed, axis=-1, count=shape[-1])
                                     .reshape(shape).copy())
        if m is None or not m.numel():
            continue
        if m.ndim == 2:
            m = m[None]
        idx = torch.clamp(torch.arange(start, start + L), max=m.shape[0] - 1)
        mm = torch.nn.functional.interpolate(m[idx].float()[:, None], size=(H, W), mode="nearest")[:, 0].clamp(0, 1)
        color = str(region.get("color") or "#ff283c")
        try:
            rgb = torch.tensor([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)], dtype=torch.float32)
        except (ValueError, IndexError):
            rgb = torch.tensor([1.0, 0.16, 0.24])
        alpha = mm[..., None] * min(0.8, max(0.05, float(region.get("opacity", 0.45))))
        o = o.float() * (1 - alpha) + rgb * alpha
    box = crop.get("box")
    if box:
        x0, y0, x1, y1 = [int(round(v)) for v in (box[0] * W, box[1] * H, box[2] * W, box[3] * H)]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W - 1, x1), min(H - 1, y1)
        yel, t = torch.tensor([1.0, 0.82, 0.35]), max(1, H // 180)
        o = o.clone()
        o[:, y0:y0 + t, x0:x1] = yel; o[:, max(0, y1 - t):y1, x0:x1] = yel
        o[:, y0:y1, x0:x0 + t] = yel; o[:, y0:y1, max(0, x1 - t):x1] = yel
    return o


def comparison_video(video: torch.Tensor, parts: list, label: str = "", height: int = 480,
                     show_mask: bool = True, first_k: int = 0, count: int | None = None, sink=None,
                     canvas: tuple[int, int] | None = None):
    """original | references | result for every output frame, with shot info on top and the prompt below.
    Built at `height` px per column (0 = full size) so long videos stay light. With show_mask, selected regions appear
    over the original column in their Colored Mark colors, or red for other masks; crop boxes are yellow."""
    vH, vW = video.shape[1:3]
    s = min(1.0, height / vH) if height and height > 0 else 1.0
    H, W = max(16, int(vH * s) // 2 * 2), max(16, int(vW * s) // 2 * 2)
    rw = max(32, W // 2 // 2 * 2)
    total_w = W * 2 + rw
    size = max(12, total_w // 70)
    count = count if count is not None else sum(1 for sh, _, _ in parts if sh and not sh.get("skipped"))
    cols = _column_labels([("original", W), ("references", rw), ("result", W)], size)
    pieces, pos, k = [], 0, int(first_k)
    for shot, L, orig in parts:
        shot = shot or {}
        fps = float(shot.get("fps") or 24.0)
        skipped = bool(shot.get("skipped"))
        if not skipped:
            k += 1
        t0, t1 = shot.get("start", 0) / fps, shot.get("end", 0) / fps
        head = [f"{'skipped (original video)' if skipped else f'shot {k}/{count}'}   "
                f"{int(t0 // 60)}:{t0 % 60:05.2f} -> {int(t1 // 60)}:{t1 % 60:05.2f}   {L} frames"
                f"{' -> ' + str(shot.get('gen_length')) + ' generated' if shot.get('gen_length') else ''}"
                f"{'   cut' if shot.get('cut_before') else ''}"]
        if label:
            head.append(label)
        top = torch.cat([_text_bar(head, total_w, size, 3), cols], 0)
        prompt = " ".join(str(shot.get("prompt") or "").split())
        bottom = _text_bar([("prompt: " + prompt) if prompt else ("" if skipped else "prompt: (none)")], total_w,
                           max(10, int(size * 0.8)), 6)
        refs = shot_refs(shot)[:4]
        if refs:
            rh = H // len(refs)
            col = torch.cat([_fit_box(r, rw, rh) for r in refs] + ([torch.full((H - rh * len(refs), rw, 3), 0.1)]
                                                                    if H - rh * len(refs) else []), 0)
        else:
            col = _fit_box(None, rw, H)
        pieces.append((top, bottom, shot, L, orig, pos, col))
        pos += L
    if not pieces:
        return video[:1]
    # same height for every frame, and both sides on a multiple of 16: video encoders (x264 / yuv420p) need even sizes
    # the frame's bars are known first, then the columns are built 16 frames at a time and written straight into the
    # output (or handed to `sink`, which streams them to a file, at the fixed `canvas` size): never every piece at once
    hmax = max(t.shape[0] + H + bt.shape[0] for t, bt, *_ in pieces)
    H16, W16 = canvas or (-(-hmax // 16) * 16, -(-total_w // 16) * 16)
    out = None if sink else torch.full((sum(p[3] for p in pieces), H16, W16, 3), 18 / 255)

    def put(buf, y, x0, x1, t):   # clipped to the canvas
        hh, ww = min(t.shape[-3], H16 - y), min(x1, W16) - x0
        if hh > 0 and ww > 0:
            buf[:, y:y + hh, x0:x0 + ww] = t[..., :hh, :ww, :]

    i = 0
    for top, bottom, shot, L, orig, pos, col in pieces:
        ov = overlay_of(shot) if show_mask and orig is not None and orig.shape[0] else None
        for a in range(0, L, 16):
            n = min(16, L - a)
            buf = out[i + a:i + a + n] if out is not None else torch.full((n, H16, W16, 3), 18 / 255)
            res = _scale_batch(video[pos + a:pos + a + n], H, W)
            if orig is not None and orig.shape[0]:
                o = orig[torch.clamp(torch.arange(a, a + n), max=orig.shape[0] - 1)]
                o = _scale_batch(o.float() / 255.0 if o.dtype == torch.uint8 else o, H, W)
                if ov is not None:
                    o = _mask_overlay(o, ov, n, a)
            else:
                o = torch.full((n, H, W, 3), 0.1)
            y = top.shape[0]
            put(buf, 0, 0, total_w, top)
            put(buf, y, 0, W, o)
            put(buf, y, W, W + col.shape[1], col[None].expand(n, -1, -1, -1))
            put(buf, y, W + col.shape[1], total_w, res)
            put(buf, y + H, 0, total_w, bottom)
            if sink:
                sink(buf)
            del o, res, buf
        i += L
    return (H16, W16) if sink else out


def guide_video(video, parts: list, height: int = 480, sink=None, size: tuple[int, int] | None = None):
    """The original frames each output frame replaced (the comparison's original column, without its marks): the guide
    to compare with the result frame by frame. `height` px like the comparison (0 = full size, never above the result).
    With sink, streamed 16 frames at a time at `size` and returns the size; otherwise returns [N, H, W, 3]."""
    vH, vW = video.shape[1:3]
    s = min(1.0, height / vH) if height and height > 0 else 1.0
    H, W = size or (max(16, int(vH * s) // 2 * 2), max(16, int(vW * s) // 2 * 2))
    out = None if sink else torch.empty((sum(L for _, L, _ in parts), H, W, 3))
    pos = 0
    for _, L, orig in parts:
        for a in range(0, L, 16):
            n = min(16, L - a)
            if orig is not None and orig.shape[0]:
                o = orig[torch.clamp(torch.arange(a, a + n), max=orig.shape[0] - 1)][..., :3]
                o = _scale_batch(o.float() / 255.0 if o.dtype == torch.uint8 else o.float(), H, W)
            else:
                o = torch.full((n, H, W, 3), 0.1)
            if sink:
                sink(o)
            else:
                out[pos + a:pos + a + n] = o
        pos += L
    return (H, W) if sink else out


class BFSShotSeed:
    """The seed of each shot (runs once per shot): the same for all, counting up, or unrelated per shot."""

    MODES = ["same for every shot", "increment per shot", "random per shot"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot": ("BFS_SHOT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "The base seed (shot 1 always uses it)."}),
                "mode": (cls.MODES, {"default": cls.MODES[0], "tooltip":
                    "same for every shot: every shot starts from the same noise (as when one seed widget feeds the "
                    "sampler). increment per shot: seed, seed+1, seed+2 ... random per shot: an unrelated seed for "
                    "each shot, derived from the base seed, so the run is still repeatable."}),
            },
            "optional": {
                "overrides": ("STRING", {"default": "", "tooltip":
                    "Seeds for single shots, 'shot:seed' pairs separated by commas (shots count from 1), e.g. "
                    "'3:12345, 5:777'. To keep a shot that came out well and re-roll the others."}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("seed", "label")
    OUTPUT_TOOLTIPS = ("This shot's seed: connect it to the sampler / noise node.", "'shot N: seed' for notes and the comparison bar.")
    FUNCTION = "pick"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("One seed per shot from a single base seed: the same for every shot, counting up, or random per "
                   "shot, with optional per-shot overrides. Connect the shot from BFS Shot Planner / Unpack and feed "
                   "the seed to the sampler.")

    def pick(self, shot, seed, mode, overrides=""):
        i = int(shot.get("index", 0))
        value = shot_seed(int(seed), i, mode, overrides)
        return (value, f"shot {i + 1}: {value}")


def shot_seed(seed: int, index: int, mode: str, overrides: str = "") -> int:
    for part in re.split(r"[,;\n]+", overrides or ""):
        m = re.match(r"\s*(\d+)\s*[:=]\s*(\d+)\s*$", part)
        if m and int(m.group(1)) == index + 1:
            return int(m.group(2)) & 0xffffffffffffffff
    if mode.startswith("increment"):
        return (seed + index) & 0xffffffffffffffff
    if mode.startswith("random") and index > 0:
        return int.from_bytes(hashlib.sha256(f"{seed}/{index}".encode()).digest()[:8], "big")
    return seed


NODE_CLASS_MAPPINGS = {
    "BFSShotPlanner": BFSShotPlanner,
    "BFSShotUnpack": BFSShotUnpack,
    "BFSShotRepack": BFSShotRepack,
    "BFSShotH3Conditioning": BFSShotH3Conditioning,
    "BFSShotH3ConditioningSimple": BFSShotH3ConditioningSimple,
    "BFSShotJoin": BFSShotJoin,
    "BFSShotSeed": BFSShotSeed,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BFSShotPlanner": "BFS Shot Planner",
    "BFSShotUnpack": "BFS Shot Unpack",
    "BFSShotRepack": "BFS Shot Repack",
    "BFSShotH3Conditioning": "BFS Shot H3 Conditioning (Advanced)",
    "BFSShotH3ConditioningSimple": "BFS Shot H3 Conditioning",
    "BFSShotJoin": "BFS Shot Join",
    "BFSShotSeed": "BFS Shot Seed",
}

# ---------------------------------------------------------------------------- chunked uploads
# ComfyUI refuses any request over --max-upload-size (100 MB by default), so a big video sent in one POST to
# /upload/image fails. The panel sends it in pieces well under that limit; they are written straight to disk (never
# held in memory) and the finished file is moved into the input folder.
UPLOAD_CHUNK = 8 * 1024 * 1024
_UPLOAD_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _upload_dir() -> str:
    d = os.path.join(folder_paths.get_input_directory(), ".bfs_upload")
    os.makedirs(d, exist_ok=True)
    return d


def _clean_uploads(max_age_s: float = 86400.0) -> None:
    """Pieces of uploads that never finished (a closed tab), older than a day."""
    d = _upload_dir()
    for f in os.listdir(d):
        p = os.path.join(d, f)
        try:
            if time.time() - os.path.getmtime(p) > max_age_s:
                os.remove(p)
        except OSError:
            pass


def write_chunk(uid: str, offset: int, data: bytes) -> int:
    """Writes one piece at its offset (pieces may be retried); returns the bytes written."""
    if not _UPLOAD_ID.match(uid or ""):
        raise ValueError("bad upload id")
    if offset < 0:
        raise ValueError("bad offset")
    part = os.path.join(_upload_dir(), uid + ".part")
    with open(part, "r+b" if os.path.exists(part) else "wb") as f:
        f.seek(offset)
        f.write(data)
    return len(data)


def finish_upload(uid: str, name: str, size: int, subfolder: str = "") -> str:
    """Checks the size and moves the assembled file into input/ (overwriting a file of the same name, like ComfyUI's
    own upload). Returns the name to use in the panel."""
    if not _UPLOAD_ID.match(uid or ""):
        raise ValueError("bad upload id")
    base = os.path.basename(str(name or "").replace("\\", "/")).strip()
    if not base or base.startswith("."):
        raise ValueError("bad file name")
    sub = "/".join(x for x in str(subfolder or "").replace("\\", "/").split("/") if x and x not in (".", ".."))
    part = os.path.join(_upload_dir(), uid + ".part")
    if not os.path.exists(part):
        raise ValueError("nothing was uploaded")
    got = os.path.getsize(part)
    if got != int(size):
        raise ValueError(f"upload incomplete: {got} of {int(size)} bytes")
    root = folder_paths.get_input_directory()
    dest_dir = os.path.join(root, sub) if sub else root
    if not os.path.abspath(dest_dir).startswith(os.path.abspath(root)):
        raise ValueError("bad subfolder")
    os.makedirs(dest_dir, exist_ok=True)
    os.replace(part, os.path.join(dest_dir, base))
    _clean_uploads()
    return f"{sub}/{base}" if sub else base


# ---------------------------------------------------------------------------- http api

try:
    from aiohttp import web
    from server import PromptServer

    def _list_input(exts):
        root = folder_paths.get_input_directory()
        out = []
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(exts):
                    rel = os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/")
                    out.append(rel)
        return sorted(out, key=str.lower)

    @PromptServer.instance.routes.post("/bfs/upload/chunk")
    async def _bfs_upload_chunk(request):
        try:
            data = await request.read()
            n = await _off_loop(lambda: write_chunk(request.query.get("id", ""), int(request.query.get("offset", "-1")), data))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=400)
        return web.json_response({"written": n})

    @PromptServer.instance.routes.post("/bfs/upload/finish")
    async def _bfs_upload_finish(request):
        try:
            b = await request.json()
            name = await _off_loop(lambda: finish_upload(b.get("id", ""), b.get("name", ""), int(b.get("size", -1)),
                                                         b.get("subfolder", "")))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=400)
        return web.json_response({"name": name})

    @PromptServer.instance.routes.get("/bfs/shotloop/files")
    async def _bfs_shot_files(request):
        # images are only listed on request: an input folder can hold thousands of them
        images = _list_input(IMAGE_EXTS) if request.query.get("images") in ("1", "true") else []
        return web.json_response({"videos": _list_input(VIDEO_EXTS), "images": images})

    @PromptServer.instance.routes.get("/bfs/shotloop/analyze")
    async def _bfs_shot_analyze(request):
        name = request.query.get("video", "")
        fps = float(request.query.get("fps", "24") or 24)
        try:
            a = await _off_loop(lambda: analyze(_input_path(name), fps))
        except Exception as exc:  # noqa: BLE001 - the panel shows the reason
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response(a)

    @PromptServer.instance.routes.post("/bfs/shotloop/plan")
    async def _bfs_shot_plan(request):
        body = await request.json()
        try:
            p = _load_plan(json.dumps(body.get("plan", {})))

            def work():
                a = analyze(_input_path(p["video"]), float(p["fps"]))
                path = _input_path(p["video"])
                if p.get("cast") or p.get("cast_split") or p.get("cast_only"):
                    try:
                        analyze_cast(path, a)
                    except Exception:  # noqa: BLE001 - plan without people
                        pass
                segs = resolve_plan(p, a, path)
                cuts, used = find_cuts(path, a, p.get("detector", "adaptive"), float(p["sensitivity"]))
                return a, segs, cuts, used
            a, segs, cuts, used = await _off_loop(work)
            W, H = plan_generation_size(a["width"], a["height"], p, int(p["multiple"]))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"segs": segs, "cuts": cuts, "width": W, "height": H, "detector": used,
                                  "max_len": snap_down(int(round(float(p["max_s"]) * float(p["fps"]))), p["grid"])})
    @PromptServer.instance.routes.post("/bfs/shotloop/filters")
    async def _bfs_shot_filters(request):
        body = await request.json()
        try:
            p = _load_plan(json.dumps(body.get("plan", {})))
            f = dict(DEFAULT_FILTERS); f.update(p.get("filters") or {})
            path = _input_path(p["video"])

            def work():
                a = analyze(path, float(p["fps"]))
                segs = resolve_plan(p, a, path)
                for i, s in enumerate(segs):   # stats always, so the panel can show them before any filter is on
                    _status("Detecting people and faces", i, len(segs))
                    s["stats"] = shot_stats(path, a, s, int(f.get("samples") or 6))
                    s["skip_reason"] = skip_reason(s["stats"], s["end"] - s["start"], f)
                return segs
            segs = await _off_loop(work)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"segs": [{"start": s["start"], "end": s["end"], "stats": s["stats"],
                                            "skip_reason": s["skip_reason"]} for s in segs]})

    @PromptServer.instance.routes.get("/bfs/shotloop/frame")
    async def _bfs_shot_frame(request):
        import cv2
        q = request.rel_url.query
        try:
            path = _input_path(q["video"])
            a = analyze(path, float(q.get("fps", 24)))
            f = max(0, min(a["n"] - 1, int(q.get("f", 0))))
            src = _timeline(a["n_src"], a["fps_src"], float(a["fps"]))
            w = int(q.get("w", 960))
            h = max(16, int(round(w * a["height"] / max(1, a["width"]))))
            fr = _read_frames(path, src[[f]], (w, h))[0]
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.Response(body=buf.tobytes(), content_type="image/jpeg")

    @PromptServer.instance.routes.post("/bfs/shotloop/mask")
    async def _bfs_shot_mask(request):
        body = await request.json()
        try:
            p = _load_plan(json.dumps(body.get("plan", {})))
            path = _input_path(p["video"])
            a = analyze(path, float(p["fps"]))
            segs = resolve_plan(p, a, path)
            seg = segs[int(body.get("index", 0))]
            mask_override = body.get("mask")
            spec = plan_spec(p, mask_override or seg.get("mask"), path)
            if isinstance(mask_override, dict):
                for key in ("threshold", "max_objects", "invert", "fill_holes", "temporal_expand", "blockify",
                            "padding", "expand", "feather", "paste"):
                    if key in mask_override:
                        spec[key] = mask_override[key]
            import asyncio   # SAM 3 takes seconds: keep the server responsive
            _outside_prompt()
            preview_color = (mask_override or {}).get("preview_color") or (
                seg.get("color_guide_color") if seg.get("color_guide") else "#ff283c")
            out = await asyncio.get_running_loop().run_in_executor(
                None, lambda: mask_preview(path, a, seg["start"], seg["end"] - seg["start"], spec,
                                           int(body.get("count", 6)),
                                           preview_color,
                                           seg.get("color_guide_opacity") if seg.get("color_guide") else 0.55))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response(out)

    @PromptServer.instance.routes.post("/bfs/shotloop/vlm")
    async def _bfs_shot_vlm(request):
        body = await request.json()
        try:
            clip = _VLM.get("clip")
            if clip is None:
                raise ValueError("The panel has no VLM yet. Connect the VLM (a CLIPLoader with a Qwen3-VL text encoder, "
                                 "the same one Generate Text uses) to the planner's vlm input and run the workflow once "
                                 "(Queue): ComfyUI only hands models to nodes when they run. After that the panel's "
                                 "VLM buttons use it.")
            p = _load_plan(json.dumps(body.get("plan", {})))
            path = _input_path(p["video"])
            a = analyze(path, float(p["fps"]))
            segs = resolve_plan(p, a, path)
            want = body.get("indices")
            pick = [segs[i] for i in want] if want else segs
            cfg = vlm_cfg(p.get("vlm_cfg"))
            import asyncio
            _outside_prompt()
            out = await asyncio.get_running_loop().run_in_executor(
                None, lambda: [dict(vlm_shot(clip, path, a, s["start"], s["end"], cfg), start=s["start"], end=s["end"])
                               for s in pick])
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"segs": out})

    @PromptServer.instance.routes.post("/bfs/shotloop/target")
    async def _bfs_shot_target(request):
        """Pick + describe who is replaced: body {plan, index, mask?} -> {desc, crop}. The subject is the shot's SAM 3
        selection (points on its key frame, else its text); the VLM, when connected, writes the description."""
        body = await request.json()
        try:
            p = _load_plan(json.dumps(body.get("plan", {})))
            path = _input_path(p["video"])
            a = analyze(path, float(p["fps"]))
            seg = resolve_plan(p, a, path)[int(body.get("index", 0))]
            spec = plan_spec(p, body.get("mask") or seg.get("mask"), path)
            if not spec_has_mask(spec):
                raise ValueError("Select the person first: 🎯 Points… (click on them), type what to segment, or give the "
                                 "shot a mask video.")
            clip = _VLM.get("clip")

            def work():
                import cv2
                crop = target_crop(path, a, seg["start"], seg["end"] - seg["start"], spec)
                if crop is None:
                    raise ValueError("SAM 3 found nothing at that selection.")
                tcfg = vlm_cfg(p.get("vlm_cfg"))
                tslot = prompt_slot(seg.get("prompt") or p.get("global_prompt", ""), "{target}") if tcfg.get("fit_prompt", True) else ""
                desc = describe_target(clip, crop, slot=tslot, temperature=float(tcfg.get("temperature") or 0),
                                       seed=int(tcfg.get("seed") or 0),
                                       image_max_side=int(tcfg.get("image_max_side") or 1024)) if clip is not None else ""
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
                return {"desc": desc, "crop": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
                        "no_vlm": clip is None}
            out = await _off_loop(work)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response(out)

    @PromptServer.instance.routes.post("/bfs/shotloop/describe")
    async def _bfs_shot_describe(request):
        """Each picture described alone: body {plan, pictures: [name, ...]} -> {pictures: {name: text}}. With fit_prompt
        the description is written for the global prompt's sentence about that picture ({details} = picture 1)."""
        body = await request.json()
        try:
            clip = _VLM.get("clip")
            if clip is None:
                raise ValueError("The panel has no VLM yet. Connect the VLM (a CLIPLoader with a Qwen3-VL text encoder, "
                                 "the same one Generate Text uses) to the planner's vlm input and run the workflow once "
                                 "(Queue): ComfyUI only hands models to nodes when they run. After that the panel's "
                                 "VLM buttons use it.")
            p = _load_plan(json.dumps(body.get("plan", {})))
            cfg = vlm_cfg(p.get("vlm_cfg"))
            prompt = picture_markers(body.get("prompt") or p.get("global_prompt", ""))
            pictures = [str(x) for x in body.get("pictures") or [] if x]

            def work():
                out = {}
                for name in pictures:
                    k = p["global_refs"].index(name) + 1 if name in p["global_refs"] else None
                    marker = (k and picture_marker(prompt, k)) or picture_marker(prompt)
                    out[name] = describe_picture(clip, _load_image(name), prompt, marker, picture_cfg(cfg, p, name))
                return out
            texts = await _off_loop(work)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"pictures": texts})

    @PromptServer.instance.routes.post("/bfs/shotloop/autoprompt")
    async def _bfs_shot_autoprompt(request):
        """🤖 Auto prompt: the VLM writes the H3 prompt of some shots: body {plan, indices} -> {prompts: {"start-end":
        {text, for}}}."""
        body = await request.json()
        try:
            clip = _VLM.get("clip")
            if clip is None:
                raise ValueError("The panel has no VLM yet. Connect the VLM to the planner's vlm input and run the "
                                 "workflow once (Queue): ComfyUI only hands models to nodes when they run.")
            p = _load_plan(json.dumps(body.get("plan", {})))
            path = _input_path(p["video"])
            cfg = vlm_cfg(p.get("vlm_cfg"))
            details = saved_picture_details(p)

            def work():
                a = analyze(path, float(p["fps"]))
                segs = resolve_plan(p, a, path)
                out = {}
                for i in body.get("indices") or range(len(segs)):
                    seg = segs[int(i)]
                    names = shot_ref_names(seg["refs"], p["global_refs"])
                    out[f"{seg['start']}-{seg['end']}"] = auto_prompt_shot(
                        clip, p, a, path, seg, names, [_load_image(n) for n in names], details,
                        seg["prompt"] or p.get("global_prompt", ""), cfg, (p.get("auto_prompt") or {}).get("template", ""))
                return out
            prompts = await _off_loop(work)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"prompts": prompts})

    @PromptServer.instance.routes.post("/bfs/shotloop/cast")
    async def _bfs_shot_cast(request):
        body = await request.json()
        try:
            p = _load_plan(json.dumps(body.get("plan", {})))
            path = _input_path(p["video"])
            c = await _off_loop(lambda: analyze_cast(path, analyze(path, float(p["fps"]))))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"people": c["people"], "step": c["step"], "samples": len(c["samples"])})

    @PromptServer.instance.routes.get("/bfs/shotloop/vram")
    async def _bfs_shot_vram(request):
        try:
            return web.json_response(vram_status())
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    @PromptServer.instance.routes.post("/bfs/shotloop/progress")
    async def _bfs_shot_progress(request):
        body = await request.json()
        try:
            return web.json_response(cache_status(_load_plan(json.dumps(body.get("plan", {}))), bool(body.get("reset"))))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    @PromptServer.instance.routes.post("/bfs/shotloop/redo")
    async def _bfs_shot_redo(request):
        body = await request.json()
        try:
            return web.json_response(redo_shots(_load_plan(json.dumps(body.get("plan", {}))),
                                                [int(x) for x in body.get("shots", [])], bool(body.get("cascade", True))))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    @PromptServer.instance.routes.post("/bfs/shotloop/combine")
    async def _bfs_shot_combine(request):
        body = await request.json()
        try:
            import asyncio
            p = _load_plan(json.dumps(body.get("plan", {})))
            res = await asyncio.get_running_loop().run_in_executor(None, lambda: combine_stored(
                p, int(body.get("crossfade", 4)), int(body.get("crf", 17)), str(body.get("prefix") or "bfs_shot_loop")))
            return web.json_response(res)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
except Exception as _exc:  # noqa: BLE001 - nodes still work without the panel
    print(f"[BFSNodes] Shot loop HTTP routes not registered: {_exc!r}")
