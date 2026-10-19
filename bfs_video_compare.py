"""BFS Video Compare: two videos in one player (wipe, overlay or side by side, either one on top) plus a per-pixel
difference map. Made for BFS Shot Join's `guide` and `images` outputs, but any two IMAGE batches work.

The difference is measured here on the frames themselves (not on the browser's compressed preview): per pixel, the mean
absolute RGB difference, drawn black (equal) -> red -> yellow -> white (very different), scaled by `gain`. The player
gets three temporary MP4s (guide, target, difference) and the per-frame difference, so it can jump to the worst frames.
"""
import math
import os
import re
import subprocess
import uuid

import numpy as np
import torch
import torch.nn.functional as F

import folder_paths

try:
    from . import bfs_video_io as VIO
except ImportError:   # loaded as a plain module (the tests): the file next to this one
    import importlib.util as _ilu
    _vio = _ilu.spec_from_file_location("bfs_video_io", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                     "bfs_video_io.py"))
    VIO = _ilu.module_from_spec(_vio)
    _vio.loader.exec_module(VIO)

CHUNK = 16


def _scale(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if tuple(x.shape[1:3]) == (h, w):
        return x
    return F.interpolate(x.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False,
                         antialias=True).permute(0, 2, 3, 1).clamp(0, 1)


def heat(v: torch.Tensor) -> torch.Tensor:
    """[..., H, W] in 0..1 -> RGB: black (0), red, yellow, white (1)."""
    return torch.stack([(3 * v).clamp(0, 1), (3 * v - 1).clamp(0, 1), (3 * v - 2).clamp(0, 1)], -1)


def difference(guide: torch.Tensor, target: torch.Tensor, gain: float = 4.0):
    """Both [N, H, W, 3] in 0..1 (same size). Returns (map [N, H, W, 3], mean abs difference per frame in 0..255,
    mean squared error per frame in 0..1)."""
    d = (guide[..., :3].float() - target[..., :3].float()).abs()
    mae = d.mean(dim=(1, 2, 3)) * 255
    mse = (d * d).mean(dim=(1, 2, 3))
    return heat((d.mean(-1) * float(gain)).clamp(0, 1)), mae, mse


def psnr(mse: float) -> float:
    return float("inf") if mse <= 0 else 10 * math.log10(1.0 / mse)


MAX_REFS = 12


def ref_pictures(reference) -> list:
    """The reference pictures, one [H,W,3] each: from a list of IMAGEs of any sizes (e.g. the planner's or BFS Shot
    Unpack's refs) and / or IMAGE batches."""
    out = []
    for item in reference if isinstance(reference, list) else [reference]:
        if item is not None:
            out += [item[i] for i in range(int(item.shape[0]))]
    return out[:MAX_REFS]


def save_refs(reference, out_dir: str, height: int) -> list:
    """The reference pictures as PNGs no taller than the player (no upscale) -> file names."""
    from PIL import Image
    names = []
    for i, pic in enumerate(ref_pictures(reference)):
        im = Image.fromarray((pic[..., :3].float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
        if im.height > height:
            im = im.resize((max(1, round(im.width * height / im.height)), height), Image.LANCZOS)
        names.append(f"ref_{i}.png")
        im.save(os.path.join(out_dir, names[-1]))
    return names


def video_batches(value, name):
    parts = value if isinstance(value, list) else [value]
    if not parts or any(not isinstance(x, torch.Tensor) or x.ndim != 4 or
                        x.shape[0] < 1 or x.shape[-1] < 3 or min(x.shape[1:3]) < 1 for x in parts):
        raise ValueError(f"{name} must contain nonempty IMAGE batches")
    return parts


def video_chunk(parts, start, end, h, w):
    """Read a bounded frame interval across shot batches without concatenating full videos."""
    result, pos = [], 0
    for part in parts:
        count = int(part.shape[0])
        a, b = max(0, start - pos), min(count, end - pos)
        if a < b:
            result.append(_scale(part[a:b, ..., :3].float(), h, w))
        pos += count
        if pos >= end:
            break
    return result[0] if len(result) == 1 else torch.cat(result, 0)


class BFSVideoCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide": ("IMAGE", {"tooltip": "The original video (e.g. BFS Shot Join's 'guide' output)."}),
                "target": ("IMAGE", {"tooltip": "The result to compare with it (e.g. BFS Shot Join's 'images')."}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001, "tooltip":
                                  "Frame rate of both videos (connect BFS Shot Join's 'fps')."}),
                "audio": ("AUDIO", {"tooltip": "Soundtrack for the player (e.g. BFS Shot Join's 'audio')."}),
                "gain": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 64.0, "step": 0.5, "tooltip":
                                   "How strongly differences light up in the map: a mean RGB difference of 1/gain "
                                   "(128/255 at 2) is already white. Raise it to see small changes."}),
                "size": (["guide", "target"], {"default": "guide", "tooltip":
                                               "Which video's size the other is scaled to before comparing. "
                                               "The Join's guide is at comparison_height, so 'guide' is lighter."}),
                "preview_height": ("INT", {"default": 720, "min": 0, "max": 4096, "step": 16, "tooltip":
                                           "Height of the player's videos (0 = the compared size). Only the preview: "
                                           "the difference is measured before it."}),
                "reference": ("IMAGE", {"tooltip": "Optional reference pictures (e.g. the people swapped in), shown in a "
                                        "strip below the videos, numbered as <Picture n>, and in what ⬇ save / ● rec "
                                        "download. Takes the refs list of BFS Shot Planner / Unpack (pictures of any "
                                        "size) or a batch; up to 12."}),
                "label_guide": ("STRING", {"default": "guide"}),
                "label_target": ("STRING", {"default": "target"}),
            },
        }

    # IMAGE lists contain every shot, in order. Only scalar settings take their first item.
    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("difference", "report", "mean_difference")
    OUTPUT_TOOLTIPS = ("The difference map, one frame per compared frame (black = equal, white = very different).",
                       "Frame count, sizes, mean difference, PSNR and the most / least different frames.",
                       "Mean absolute RGB difference over all frames, 0-255.")
    OUTPUT_NODE = True
    FUNCTION = "compare"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Play two videos in one player (wipe, overlay or side by side; choose which is on top) and see "
                   "where they differ, pixel by pixel.")

    def compare(self, guide, target, fps=24.0, audio=None, gain=2.0, size="guide", preview_height=720,
                label_guide="guide", label_target="target", reference=None):
        one = lambda v: v[0] if isinstance(v, list) and v else v
        fps, audio, gain, size, preview_height, label_guide, label_target = map(
            one, (fps, audio, gain, size, preview_height, label_guide, label_target))
        guide, target = video_batches(guide, "guide"), video_batches(target, "target")
        ng, nt = sum(x.shape[0] for x in guide), sum(x.shape[0] for x in target)
        if len(guide) > 1 or len(target) > 1:
            if ng != nt or (len(guide) == len(target) and any(g.shape[0] != t.shape[0] for g, t in zip(guide, target))):
                raise ValueError("Guide/target shot lists have different timing. Connect BFS Shot Join's guide and images outputs from the SAME Join; Unpack guide frames can include continuity heads and generation padding.")
        notes = []
        n = min(ng, nt)
        if ng != nt:
            notes.append(f"lengths differ (guide {ng}, target {nt} frames): "
                         f"compared the first {n}")
        ref = guide if size == "guide" else target
        H, W = int(ref[0].shape[1]), int(ref[0].shape[2])
        for name, parts in (("guide", guide), ("target", target)):
            if len(parts) > 1:
                notes.append(f"{name}: compared all {len(parts)} shot batches in input order")
            for dims in sorted(set(tuple(v.shape[1:3]) for v in parts)):
                if dims != (H, W):
                    notes.append(f"{name} scaled from {dims[1]}x{dims[0]} to {W}x{H}")
        if ng != nt:
            notes.append("Timing may be misaligned. For Planner results, connect guide and images from the SAME BFS Shot Join.")
        ph = min(H, int(preview_height)) if preview_height and preview_height > 0 else H
        PH, PW = max(16, ph // 2 * 2), max(16, int(round(W * ph / H)) // 2 * 2)

        out_dir = os.path.join(folder_paths.get_temp_directory(), "bfs_compare", uuid.uuid4().hex[:12])
        os.makedirs(out_dir, exist_ok=True)
        names = {"guide": "guide.mp4", "target": "target.mp4", "difference": "difference.mp4"}
        writers = {k: VIO.Mp4Writer(os.path.join(out_dir, f), PW, PH, float(fps), audio if k == "target" else None,
                                    crf=14, who="BFS Video Compare") for k, f in names.items()}
        maps, mae, mse = [], [], []
        try:
            for a in range(0, n, CHUNK):
                e = min(a + CHUNK, n)   # the longer video stops where the shorter one ends
                g = video_chunk(guide, a, e, H, W)
                t = video_chunk(target, a, e, H, W)
                m, e, s = difference(g, t, gain)
                maps.append(m)
                mae.append(e)
                mse.append(s)
                for k, v in (("guide", g), ("target", t), ("difference", m)):
                    writers[k].write(_scale(v, PH, PW))
        finally:
            for w in writers.values():
                w.close()
        diff_map = torch.cat(maps, 0)
        mae, mse = torch.cat(mae), torch.cat(mse)
        mean = float(mae.mean())
        worst, best = int(mae.argmax()), int(mae.argmin())
        report = "\n".join([
            f"{label_guide} vs {label_target}: {n} frames at {fps:g} fps, compared at {W}x{H}",
            f"mean difference {mean:.2f}/255, PSNR {psnr(float(mse.mean())):.2f} dB",
            f"most different: frame {worst} ({worst / fps:.2f} s) {float(mae[worst]):.2f}/255",
            f"least different: frame {best} ({best / fps:.2f} s) {float(mae[best]):.2f}/255",
            *notes])
        print(f"[BFS Video Compare] {report}", flush=True)
        sub = os.path.relpath(out_dir, folder_paths.get_temp_directory())
        files = {k: {"filename": f, "subfolder": sub, "type": "temp"} for k, f in names.items()}
        refs = [{"filename": f, "subfolder": sub, "type": "temp"} for f in save_refs(reference, out_dir, PH)]
        ui = {"bfs_compare": [{**files, "fps": float(fps), "frames": n, "width": PW, "height": PH,
                               "labels": [label_guide, label_target], "mae": [round(float(x), 2) for x in mae],
                               "refs": refs,
                               "report": report, "notes": notes}]}
        return {"ui": ui, "result": (diff_map, report, mean)}


# ---------------------------------------------------------------------------- save / record: the browser's file -> a shareable one

EXPORT_FORMATS = {   # what the player's format list offers; "webm" never reaches the server (the browser's own file)
    "mp4": "MP4 (H.264)",
    "small": "MP4 small (under 10 MB)",
    "gif": "GIF (under 10 MB, no sound)",
}
LIMIT_BYTES = int(9.5 * 1024 * 1024)   # under the 10 MB many chat apps accept for free
_ID = re.compile(r"^[0-9a-f]{8,32}$")


def _export_dir() -> str:
    d = os.path.join(folder_paths.get_temp_directory(), "bfs_compare", "exports")
    os.makedirs(d, exist_ok=True)
    return d


def write_part(rid: str, offset: int, data: bytes) -> int:
    """One piece of a recording uploaded by the player (pieces keep big recordings under the upload limit)."""
    if not _ID.match(rid or "") or offset < 0:
        raise ValueError("bad recording id / offset")
    path = os.path.join(_export_dir(), f"{rid}.rec")
    with open(path, "r+b" if os.path.exists(path) else "wb") as f:
        f.seek(offset)
        f.write(data)
    return len(data)


def _run(cmd: list) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[-600:]}")


def encode(src: str, fmt: str, seconds: float, out_base: str) -> str:
    """The recording `src` (WebM / fragmented MP4 from MediaRecorder, which many apps won't play or preview) as
    `fmt` -> the written path. Both MP4s are H.264 + AAC, yuv420p, faststart. mp4: CRF 18. small: the bitrate set so
    the file stays under 9.5 MB (smaller frame when it gets low). gif: palette per video, 15 fps, made smaller until
    it fits 9.5 MB."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"unknown format {fmt!r}")
    exe = VIO.ffmpeg_bin()
    if exe is None:
        return _encode_av(src, fmt, out_base)
    seconds = max(0.5, float(seconds or 0))
    even = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    if fmt == "gif":
        out = out_base + ".gif"
        for width, fps in ((720, 15), (540, 15), (480, 12), (360, 12), (270, 10)):
            vf = f"fps={fps},scale='min({width},iw)':-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
            _run([exe, "-y", "-v", "error", "-i", src, "-vf", vf, "-loop", "0", out])
            if os.path.getsize(out) <= LIMIT_BYTES:
                break
        return out
    out = out_base + ".mp4"
    if fmt == "mp4":
        _run([exe, "-y", "-v", "error", "-i", src, "-vf", even, "-c:v", "libx264", "-preset", "medium", "-crf", "18",
              "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out])
        return out
    audio_k = 128
    for shrink in (1.0, 0.92, 0.8):   # a VBR encode can overshoot a little: a second, lower try
        kbps = int(min(8000, max(250, LIMIT_BYTES * 8 / 1000 / seconds * shrink - audio_k)))
        vf = even if kbps >= 1500 else "scale=-2:'min(720,ih)':flags=lanczos," + even if kbps >= 700 else \
            "scale=-2:'min(480,ih)':flags=lanczos," + even
        _run([exe, "-y", "-v", "error", "-i", src, "-vf", vf, "-c:v", "libx264", "-preset", "medium",
              "-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.4)}k", "-bufsize", f"{kbps * 2}k", "-pix_fmt", "yuv420p",
              "-r", "30", "-c:a", "aac", "-b:a", f"{audio_k}k", "-ac", "2", "-movflags", "+faststart", out])
        if os.path.getsize(out) <= LIMIT_BYTES:
            break
    return out


def _encode_av(src: str, fmt: str, out_base: str) -> str:
    """No ffmpeg program: decode with PyAV and write an MP4 (no sound) or a GIF (PIL); 'small' at most 720p."""
    import av
    from PIL import Image
    frames, fps = [], 30.0
    with av.open(src) as c:
        st = c.streams.video[0]
        fps = float(st.average_rate or st.guessed_rate or 30)
        for f in c.decode(st):
            frames.append(f.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError("the recording has no frames")
    h, w = frames[0].shape[:2]
    if fmt == "gif":
        out = out_base + ".gif"
        step = max(1, round(fps / 12))
        for side in (540, 400, 300, 220):
            k = min(1.0, side / max(h, w))
            ims = [Image.fromarray(x).resize((max(2, int(w * k)), max(2, int(h * k))), Image.LANCZOS) for x in frames[::step]]
            ims[0].save(out, save_all=True, append_images=ims[1:], duration=int(1000 * step / fps), loop=0, optimize=True)
            if os.path.getsize(out) <= LIMIT_BYTES:
                break
        return out
    out = out_base + ".mp4"
    k = 1.0 if fmt == "mp4" else min(1.0, 720 / h)
    W2, H2 = int(w * k) // 2 * 2, int(h * k) // 2 * 2
    wr = VIO.Mp4Writer(out, W2, H2, fps, None, crf=18 if fmt == "mp4" else 27, who="BFS Video Compare")
    try:
        for a in range(0, len(frames), 16):
            x = torch.from_numpy(np.stack(frames[a:a + 16])).float() / 255
            wr.write(_scale(x, H2, W2))
    finally:
        wr.close()
    print("[BFS Video Compare] no ffmpeg program: saved without sound", flush=True)
    return out


def finish_export(rid: str, fmt: str, seconds: float, name: str) -> dict:
    src = os.path.join(_export_dir(), f"{rid}.rec")
    if not _ID.match(rid or "") or not os.path.exists(src):
        raise ValueError("recording not found")
    stem = re.sub(r"[^\w.-]+", "_", os.path.splitext(name or "bfs_compare")[0])[:80] or "bfs_compare"
    try:
        out = encode(src, fmt, seconds, os.path.join(_export_dir(), f"{stem}_{rid[:6]}"))
    finally:
        os.remove(src)
    sub = os.path.relpath(os.path.dirname(out), folder_paths.get_temp_directory()).replace(os.sep, "/")
    mb = os.path.getsize(out) / 1024 / 1024
    print(f"[BFS Video Compare] saved {os.path.basename(out)} ({EXPORT_FORMATS[fmt]}, {mb:.1f} MB)", flush=True)
    return {"filename": os.path.basename(out), "subfolder": sub, "type": "temp", "mb": round(mb, 2)}


try:
    import asyncio

    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/bfs/compare/part")
    async def _bfs_compare_part(request):
        try:
            data = await request.read()
            n = await asyncio.get_running_loop().run_in_executor(
                None, lambda: write_part(request.query.get("id", ""), int(request.query.get("offset", "-1")), data))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=400)
        return web.json_response({"written": n})

    @PromptServer.instance.routes.post("/bfs/compare/export")
    async def _bfs_compare_export(request):
        try:
            b = await request.json()
            r = await asyncio.get_running_loop().run_in_executor(
                None, lambda: finish_export(b.get("id", ""), b.get("format", ""), float(b.get("seconds") or 0), b.get("name", "")))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=400)
        return web.json_response(r)
except Exception as _exc:  # noqa: BLE001 - outside ComfyUI (tests)
    if "PromptServer" not in repr(_exc) and "server" not in repr(_exc):
        print(f"[BFS Video Compare] export routes not registered: {_exc!r}", flush=True)


NODE_CLASS_MAPPINGS = {"BFSVideoCompare": BFSVideoCompare}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSVideoCompare": "BFS Video Compare"}
