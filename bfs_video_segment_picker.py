"""BFS Video Segment Picker (SAM 3): pick who to replace in a video or in a set of keyframes.

The image Segment Picker shows one image; this one shows every frame of the batch (the Keyframe Planner's
source_frames, or a short clip) as a strip: click a frame, then click the person on it, exactly like the image picker.
A long video shows an evenly spread strip of up to 32 frames. Select on one frame or on several (e.g. one per camera shot); frames without a selection take the mask of the nearest
selected frame, tracked with SAM 3 from there (forward and backward). One run segments every frame, so you can pick on
any of them without running again.

All state lives in the hidden `selection` STRING widget (JSON): one image-picker selection per frame,
    {"frames": {"0": {"selected": ["a2"], "added": [...], "next": 3}, "4": {...}}}
The image Segment Picker node and its workflows are untouched; its helpers are reused.
"""
import json
import time
from collections import OrderedDict

import torch

from . import bfs_segment_picker as sp
from .bfs_keyframe_marks import BFSTrackFromFrame

MAX_FRAMES = 32           # frames segmented for the panel; a longer video shows an evenly spread strip
# the hybrid guide: subject k drawn without its texture (e.g. its normal map in gray detail), tinted with its color
HYBRID_COLORS = [(235, 40, 40), (35, 110, 235), (60, 205, 60)]   # RGB: red, blue, green
HYBRID_TINT = 0.75


def hybrid_tint(normal_rgb, color, alpha=HYBRID_TINT):
    """normal_rgb [H,W,3] in 0..1 (MoGe OpenCV normals * 0.5 + 0.5) -> gray detail tinted with the subject color, 0..1."""
    gray = normal_rgb[..., 0] * 0.299 + normal_rgb[..., 1] * 0.587 + normal_rgb[..., 2] * 0.114
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)
    detail = (0.25 + 0.75 * gray)[..., None]
    c = torch.tensor(color, dtype=normal_rgb.dtype, device=normal_rgb.device) / 255.0
    return (detail * c * alpha + detail * (1 - alpha)).clamp(0, 1)


FREE_MODES = ["unload all models from the GPU", "unload all models + release Molmo2 RAM", "unload SAM 3 from the GPU", "unload all models + clear the picker caches",
              "keep everything loaded"]
LEGACY_FREE = {"unload SAM 3 + clear the picker caches": "unload all models + clear the picker caches"}
PROTECTED_NORMAL_STYLE = "normal map inside mask (needs normals)"
HATCHING_STYLE = "colored hatching + contour"
MARK_STYLES = ["normal map (needs normals)", PROTECTED_NORMAL_STYLE, "openpose + face (needs pose_keypoint)", "flat color", "translucent overlay", "colored outline", "shading (needs normals)", "blur",
               "pixelate", "gray detail", "edges", HATCHING_STYLE]


def hatching_alpha(mask, scale, width=1.0, spacing=1.0):
    import cv2
    line_width = max(2.5, scale * 0.01) * width
    period = max(max(2.5, scale * 0.01) * 7, scale * 0.09) * spacing
    h, w = mask.shape
    y = torch.arange(h, device=mask.device, dtype=torch.float32)[:, None]
    x = torch.arange(w, device=mask.device, dtype=torch.float32)[None, :]
    distance = ((y + 0.29 * x + period) % period - period / 2).abs() / (1 + 0.29**2)**0.5
    lines = (line_width / 2 + 0.5 - distance).clamp(0, 1)
    inside = cv2.distanceTransform(mask.detach().cpu().numpy().astype('uint8'), cv2.DIST_L2, 5)
    border = (line_width + 0.5 - torch.from_numpy(inside).to(mask.device)).clamp(0, 1)
    return torch.maximum(lines, border) * mask


def mark_fill(style, frame, normal_rgb, mask, color, alpha=HYBRID_TINT):
    """The subject's region drawn without its texture, tinted with the subject's color. frame/normal_rgb [H,W,3] 0..1.
    'normal map' and 'openpose + face' are the guide modes of BFS H3 Character Swap Conditioning; the others are
    texture-free alternatives (no MoGe needed for most)."""
    import torch.nn.functional as F
    c = torch.tensor(color, dtype=frame.dtype, device=frame.device) / 255.0
    if style.startswith("normal"):
        return hybrid_tint(normal_rgb, color, alpha)
    if style.startswith("shading"):
        n = normal_rgb * 2 - 1
        light = n.new_tensor([-0.35, -0.5, 0.8]); light = light / light.norm()
        lum = (0.30 + 0.70 * (n * light).sum(-1).clamp(0, 1))[..., None]
        return (lum * c).clamp(0, 1)
    if style in ("flat color", "translucent overlay", "colored outline", HATCHING_STYLE):
        return c.expand_as(frame).clone()
    gray = (frame[..., 0] * 0.299 + frame[..., 1] * 0.587 + frame[..., 2] * 0.114)
    ys, xs = torch.nonzero(mask, as_tuple=True)
    size = float(((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) ** 0.5) if len(ys) else 64.0
    if style == "blur":
        k = max(3, int(size * 0.12) | 1)
        g = F.avg_pool2d(F.avg_pool2d(gray[None, None], k, 1, k // 2), k, 1, k // 2)[0, 0]
        detail = g
    elif style == "pixelate":
        b = max(2, int(size / 12))
        h, w = gray.shape
        small = F.interpolate(gray[None, None], size=(max(1, h // b), max(1, w // b)), mode="area")
        detail = F.interpolate(small, size=(h, w), mode="nearest")[0, 0]
    elif style == "edges":
        gx = gray[:, 2:] - gray[:, :-2]; gy = gray[2:, :] - gray[:-2, :]
        e = torch.zeros_like(gray); e[1:-1, 1:-1] = (gx[1:-1] ** 2 + gy[:, 1:-1] ** 2).sqrt()
        detail = 1 - (e / (e.max() + 1e-6)).clamp(0, 1) * 0.9
    else:   # gray detail: the original's own luminance, posterized to 4 levels
        detail = (gray * 4).floor().clamp(0, 3) / 3
    m = mask.bool()
    if m.any():
        lo, hi = detail[m].min(), detail[m].max()
        detail = (detail - lo) / (hi - lo + 1e-6)
    detail = (0.25 + 0.75 * detail.clamp(0, 1))[..., None]
    return (detail * c * alpha + detail * (1 - alpha)).clamp(0, 1)


DESCRIBE_SOURCE = ('Describe the foreground person or costumed character in this image as ONE short English noun '
                   'phrase of 6 to 18 words that uniquely identifies their visual appearance in a video: outfit or '
                   'costume colors and pieces, mask/helmet/glasses if any, hair, and build. '
                   'No leading article, no background, no pose, no full sentence.')
VERIFY_SAME_PERSON = ('Picture 1 shows the reference person selected by the user. '
                      'Picture 2 shows a candidate person from another frame of the video. '
                      'Do Picture 1 and Picture 2 show the exact same individual wearing the exact same outfit or '
                      'costume? Answer with a single word: yes or no.')


def detect_cuts(image, thresh=0.22):
    """Camera cuts: frame indices where a new shot starts (mean abs diff of small gray frames, plus a histogram check)."""
    import torch.nn.functional as F
    g = image[..., :3].float().mean(-1)[:, None]
    small = F.interpolate(g, size=(36, 64), mode="area")[:, 0]
    cuts = []
    for i in range(1, small.shape[0]):
        d = (small[i] - small[i - 1]).abs().mean().item()
        h0 = torch.histc(small[i - 1], 16, 0, 1); h1 = torch.histc(small[i], 16, 0, 1)
        hd = (h0 / h0.sum() - h1 / h1.sum()).abs().sum().item() / 2
        if d > thresh * 0.6 and hd > 0.25 or d > thresh:
            cuts.append(i)
    return cuts


def shots_of(n, cuts):
    edges = [0] + [c for c in cuts if 0 < c < n] + [n]
    return [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]


def person_at(sam, frame, x, y, threshold):
    """The whole person under Molmo's point: SAM 3 'person' instances, the one containing the point (else the nearest);
    falls back to a point prompt."""
    h, w = frame.shape[1:3]
    px, py = min(w - 1, int(x * w)), min(h - 1, int(y * h))
    inst = [m for m, _ in sp.detect_text(sam, frame, "person", threshold, 12)]
    best, bd = None, (0.06 * max(h, w)) ** 2
    for m in inst:
        if m[py, px]:
            return m
        ys, xs = torch.nonzero(m, as_tuple=True)
        if len(xs):
            d = float(((xs - px) ** 2 + (ys - py) ** 2).min())
            if d < bd: best, bd = m, d
    if best is not None:
        return best
    return sp.detect_points(sam, frame, [{"x": x, "y": y, "label": 1}], threshold)


_MOLMO_CACHE = OrderedDict()   # (video, subject, anchor frame, anchor mask) -> (description, points): Molmo2 runs once per pick
_SHOT_CACHE = OrderedDict()    # (video, subject, shot, its anchors + masks, threshold) -> (masks, report) of that shot
_PICKER_MASK_CACHE_MAX_BYTES = 2 * 1024 ** 3  # keep picker mask caches bounded by RAM as well as entry count


def mask_key(m):
    import hashlib
    import numpy as np
    return hashlib.sha1(np.packbits((m > 0.5).cpu().numpy()).tobytes()).hexdigest()[:16]


def _cache_tensor_bytes(value, seen=None):
    """Count unique tensor storages in a cache value; item-count caps alone are unsafe for full videos."""
    if seen is None:
        seen = set()
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        storage_key = (str(value.device), storage.data_ptr(), storage.nbytes())
        if storage_key in seen:
            return 0
        seen.add(storage_key)
        return storage.nbytes()
    if isinstance(value, dict):
        return sum(_cache_tensor_bytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_cache_tensor_bytes(v, seen) for v in value)
    return 0


def _remember(cache, key, value, cap, max_bytes=None):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > cap or (max_bytes is not None and sum(_cache_tensor_bytes(v) for v in cache.values()) > max_bytes):
        cache.popitem(last=False)


def person_crop(frame, mask):
    """Crop [H,W,3] around mask > 0.5 with a small context margin and non-mask pixels dimmed toward neutral gray."""
    m = mask > 0.5
    ys, xs = torch.nonzero(m, as_tuple=True)
    if not len(ys):
        return None
    h, w = frame.shape[:2]
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    py = max(4, int((y1 - y0) * 0.08)); px = max(4, int((x1 - x0) * 0.08))
    y0, y1 = max(0, y0 - py), min(h, y1 + py)
    x0, x1 = max(0, x0 - px), min(w, x1 + px)
    crop = frame[y0:y1, x0:x1, :3].float().clone()
    cm = m[y0:y1, x0:x1]
    crop[~cm] = crop[~cm] * 0.25 + 0.375
    return crop


def highlight_person_frame(frame, mask):
    """Full [H,W,3] frame with the picked person in full color and everything else dimmed toward neutral gray."""
    m = mask > 0.5
    out = frame[..., :3].float() * 0.20 + 0.40
    out[m] = frame[..., :3].float()[m]
    return out


def person_color_sim(frame_a, mask_a, frame_b, mask_b):
    """Histogram intersection similarity (0..1) of RGB + chroma inside two person masks."""
    ma, mb = mask_a > 0.5, mask_b > 0.5
    if int(ma.sum()) < 16 or int(mb.sum()) < 16:
        return 0.0
    pa = frame_a[..., :3].float()[ma].clamp(0, 1)
    pb = frame_b[..., :3].float()[mb].clamp(0, 1)
    sims = []
    for c in range(3):
        ha = torch.histc(pa[:, c], 16, 0.0, 1.0); hb = torch.histc(pb[:, c], 16, 0.0, 1.0)
        sims.append(float(torch.minimum(ha / (ha.sum() + 1e-6), hb / (hb.sum() + 1e-6)).sum()))
    ca = pa.max(-1).values - pa.min(-1).values
    cb = pb.max(-1).values - pb.min(-1).values
    ha = torch.histc(ca, 12, 0.0, 1.0); hb = torch.histc(cb, 12, 0.0, 1.0)
    sims.append(float(torch.minimum(ha / (ha.sum() + 1e-6), hb / (hb.sum() + 1e-6)).sum()))
    return sum(sims) / len(sims)


def mask_overlap(mask_a, mask_b):
    """(IoU, containment_in_smaller) between two 2D masks."""
    a, b = mask_a > 0.5, mask_b > 0.5
    inter = float((a & b).sum())
    if inter <= 0:
        return 0.0, 0.0
    union = float((a | b).sum())
    small = float(min(a.sum(), b.sum()))
    return inter / max(1.0, union), inter / max(1.0, small)


def molmo_verify_candidate(molmo, ref_crop, ref_frame, ref_mask, cand_frame, cand_mask, claimed_masks=None, memo=None):
    """Verify a candidate re-anchor mask against the user's clicked reference person using:
    1) non-overlap with masks already claimed by other subjects on that frame,
    2) RGB + chroma appearance similarity, and
    3) Molmo2 visual comparison of [Picture 1: clicked person crop, Picture 2: candidate person crop]."""
    if not bool((cand_mask > 0.5).any()):
        return False
    for cm in (claimed_masks or []):
        iou, contain = mask_overlap(cand_mask, cm)
        if iou > 0.40 or contain > 0.55:
            return False
    sim = person_color_sim(ref_frame, ref_mask, cand_frame, cand_mask)
    if sim < 0.38:
        return False
    if molmo is None or not hasattr(molmo, "runtime") or ref_crop is None:
        return sim >= 0.55
    cand_crop = person_crop(cand_frame, cand_mask)
    if cand_crop is None:
        return False
    vkey = memo and ("verify_v2",) + tuple(memo) + (mask_key(ref_mask), mask_key(cand_mask))
    if vkey and vkey in _MOLMO_CACHE:
        ok = _MOLMO_CACHE[vkey]
        _MOLMO_CACHE.move_to_end(vkey)
        return bool(ok)
    text = molmo.runtime.generate(
        VERIFY_SAME_PERSON,
        [{"type": "image", "data": ref_crop[None]}, {"type": "image", "data": cand_crop[None]}],
        16, molmo.phase_b_max_frames, do_sample=False, repetition_penalty=1.0
    )["text"]
    ans = str(text or "").strip().lower()
    ok = ans.startswith("yes") or ("yes" in ans.split()[:2] and "no" not in ans.split()[:2])
    if vkey:
        _remember(_MOLMO_CACHE, vkey, bool(ok), 64)
    return bool(ok)


def molmo_relocate(molmo, sam, image, anchor_mask, anchor, shots, has_anchor, threshold, fps=24.0, memo=None, claimed_by_frame=None, text_override=None):
    """Find the subject in later shots that have no anchor yet: Molmo2 describes the picked person (masked crop) or uses text_override,
    tracks forward from the picked frame (with the picked person visually highlighted in the first frame of the clip), and in
    each later shot without an anchor verifies the candidate person crop against the clicked reference crop with Molmo2.
    Returns {frame: mask}, report, pts."""
    import re
    m = anchor_mask > 0.5
    ys, xs = torch.nonzero(m, as_tuple=True)
    if not len(ys):
        return {}, "no anchor mask", []
    n, h, w = image.shape[:3]
    crop = person_crop(image[anchor, ..., :3], m)
    cx = float(xs.float().mean()) / max(1, w) * 100.0
    cy = float(ys.float().mean()) / max(1, h) * 100.0
    key = memo and (("relocate_v2",) + tuple(memo) + (anchor, mask_key(anchor_mask), text_override or ""))
    reused = key in _MOLMO_CACHE if key else False
    if reused:   # same first pick of this subject: the description and Molmo2's forward track are the same
        desc, pts = _MOLMO_CACHE[key]
        _MOLMO_CACHE.move_to_end(key)
    else:
        if text_override:
            desc = text_override.strip()
        else:
            text = molmo.runtime.generate(DESCRIBE_SOURCE, [{"type": "image", "data": crop[None]}], 96, molmo.phase_b_max_frames,
                                          do_sample=False, repetition_penalty=1.0)["text"]
            desc = re.sub(r"^(the|a|an)\s+", "", " ".join(str(text).strip().strip('"').split()).rstrip(". "), flags=re.I)
        if anchor < n - 1:
            sub = image[anchor:, ..., :3].clone()
            sub[0] = highlight_person_frame(image[anchor, ..., :3], m).to(sub.dtype)
            query = f"Track the {desc} (the person highlighted in color at ({cx:.0f}%, {cy:.0f}%) in the first frame)."
            ans = molmo.runtime.track_subjects([query], sub, fps, max_fps=4.0)[0]
            t_off = float(anchor) / float(fps)
            pts = sorted((float(t) + t_off, int(o), float(x), float(y))
                         for t, o, x, y in ans.get("points", [])
                         if int(round((float(t) + t_off) * fps)) > anchor)
        else:
            pts = []
        if key:
            _remember(_MOLMO_CACHE, key, (desc, pts), 16)
    found = {}
    for si, (s0, s1) in enumerate(shots):
        if has_anchor[si] or s1 <= anchor:
            continue
        lo = max(s0, anchor + 1)
        inside = [p for p in pts if lo <= int(round(float(p[0]) * fps)) < s1]
        if not inside:
            continue
        mid = (lo + s1 - 1) / 2.0                       # try points nearest the shot's middle first: steadiest framing
        seen_f = set()
        for t, _o, x, y in sorted(inside, key=lambda p: abs(float(p[0]) * fps - mid)):
            f = min(s1 - 1, max(lo, int(round(float(t) * fps))))
            if f in seen_f:
                continue
            seen_f.add(f)
            mk = person_at(sam, image[f:f + 1, ..., :3].float().cpu(), x, y, threshold)
            if bool(mk.any()) and molmo_verify_candidate(
                molmo, crop, image[anchor, ..., :3], m, image[f, ..., :3], mk,
                claimed_masks=(claimed_by_frame or {}).get(f), memo=(memo or ()) + (anchor, f)
            ):
                found[f] = mk.float()
                break
            if len(seen_f) >= 3:
                break
    return found, (f"Molmo2{' (reused)' if reused else ''}: '{desc}', {len(pts)} points from f{anchor}, {len(shots)} shots, "
                   f"{len(found)} shots re-anchored"), pts


def molmo_relocate_legacy(molmo, sam, image, anchor_mask, anchor, shots, has_anchor, threshold, fps=24.0, memo=None, text_override=None):
    """Classic BFS path: describe the selected mask, track that text through the full video, seed unanchored shots."""
    import re
    m = anchor_mask > 0.5
    ys, xs = torch.nonzero(m, as_tuple=True)
    if not len(ys):
        return {}, "no anchor mask", []
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    crop = image[anchor, y0:y1, x0:x1, :3].clone()
    crop[~m[y0:y1, x0:x1]] = 0.5
    key = memo and (("legacy",) + tuple(memo) + (anchor, mask_key(anchor_mask), text_override or ""))
    reused = key in _MOLMO_CACHE if key else False
    if reused:
        desc, pts = _MOLMO_CACHE[key]
        _MOLMO_CACHE.move_to_end(key)
    else:
        if text_override:
            desc = text_override.strip()
        else:
            text = molmo.runtime.generate(DESCRIBE_SOURCE, [{"type": "image", "data": crop[None]}], 96, molmo.phase_b_max_frames,
                                          do_sample=False, repetition_penalty=1.0)["text"]
            desc = re.sub(r"^(the|a|an)\s+", "", " ".join(str(text).strip().strip('"').split()).rstrip(". "), flags=re.I)
        ans = molmo.runtime.track_subjects([f"Track the {desc}."], image[..., :3], fps, max_fps=4.0)[0]
        pts = sorted(ans.get("points", []))
        if key:
            _remember(_MOLMO_CACHE, key, (desc, pts), 16)
    found = {}
    for si, (s0, s1) in enumerate(shots):
        if has_anchor[si]:
            continue
        inside = [p for p in pts if s0 <= int(round(float(p[0]) * fps)) < s1]
        if not inside:
            continue
        mid = (s0 + s1 - 1) / 2.0
        t, _o, x, y = min(inside, key=lambda p: abs(float(p[0]) * fps - mid))
        f = min(s1 - 1, max(s0, int(round(float(t) * fps))))
        mk = person_at(sam, image[f:f + 1, ..., :3].float().cpu(), x, y, threshold)
        if bool(mk.any()):
            found[f] = mk.float()
    return found, (f"Molmo2 legacy{' (reused)' if reused else ''}: '{desc}', {len(pts)} points, {len(shots)} shots, "
                   f"{len(found)} shots re-anchored"), pts


# OpenPose 18-point body (COCO order of controlnet_aux DWPose) and the 70-point face (68 + 2 pupils)
BODY_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
              (0, 14), (14, 16), (0, 15), (15, 17)]
HAND_LIMBS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def _pts(flat, W, H, thr=0.3):
    """[x, y, c, ...] -> list of (x, y) pixels or None; DWPose writes 0..1 coordinates (pixels are handled too). Points
    outside the frame (guessed limbs of a close-up) are dropped."""
    if not flat:
        return []
    arr = [(flat[i], flat[i + 1], flat[i + 2]) for i in range(0, len(flat) - 2, 3)]
    norm = max((max(abs(x), abs(y)) for x, y, c in arr if c > 0), default=0) <= 1.5
    out = []
    for x, y, c in arr:
        if c <= 0 or (x <= 0 and y <= 0):
            out.append(None); continue
        px, py = (x * W, y * H) if norm else (x, y)
        out.append((px, py) if (-0.01 * W <= px <= 1.01 * W and -0.01 * H <= py <= 1.01 * H) else None)
    return out


def _limb_ok(a, b, cap):
    return a is not None and b is not None and ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 <= cap


def pick_person(people, mask, W, H, min_inside=0.5, min_diag=0.3, exclude=()):
    """The DWPose person of this subject: most of its body/face points inside the subject's OWN mask (not the grown gray
    region, which can swallow someone in the background) and a size that matches it. None when DWPose missed the subject
    in this frame (e.g. an upside-down dancer): then no skeleton is drawn rather than somebody else's."""
    import numpy as np
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    mdiag = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min())) + 1
    best, score = None, 0.0
    excl_ids = {id(p) for p in (exclude or ())}
    for p in people or []:
        if id(p) in excl_ids:
            continue
        pts = [q for q in _pts(p.get("pose_keypoints_2d"), W, H) + _pts(p.get("face_keypoints_2d"), W, H) if q]
        if len(pts) < 4:
            continue
        inside = sum(1 for x, y in pts if 0 <= int(y) < H and 0 <= int(x) < W and mask[int(y), int(x)]) / len(pts)
        px, py = [q[0] for q in pts], [q[1] for q in pts]
        pdiag = float(np.hypot(max(px) - min(px), max(py) - min(py)))
        if inside < min_inside or pdiag < min_diag * mdiag:
            continue
        sc = inside + min(pdiag / mdiag, 1.0)
        if sc > score: best, score = p, sc
    return best


def draw_pose(person, color, W, H):
    """The skeleton (body + hands) and a face whose mouth opening and eyes are FILLED areas, so speech and blinks survive
    the video VAE; line widths scale with the person's size. Returns an [H, W, 3] uint8 RGB canvas and its drawn mask."""
    import cv2
    import numpy as np
    canvas = np.zeros((H, W, 3), np.uint8); col = tuple(int(c) for c in color)
    body = _pts(person.get("pose_keypoints_2d"), W, H); face = _pts(person.get("face_keypoints_2d"), W, H)
    allp = [q for q in body + face if q]
    if not allp:
        return canvas
    xs, ys = [q[0] for q in allp], [q[1] for q in allp]
    size = max(max(xs) - min(xs), max(ys) - min(ys), 16.0)
    lw = max(2, int(size * 0.025))
    has_face = len(face) >= 68 and sum(1 for q in face[:68] if q) > 40
    head = {0, 14, 15, 16, 17}                         # with a face, the face draws the head (body head points would cover it)
    cap = 0.6 * (W * W + H * H) ** 0.5                 # a limb longer than this is a wrong guess (close-ups)
    fbox, closeup = None, False
    if has_face:
        fq = [q for q in face[:68] if q]
        fx0, fx1 = min(q[0] for q in fq), max(q[0] for q in fq); fy0, fy1 = min(q[1] for q in fq), max(q[1] for q in fq)
        mx, my = (fx1 - fx0) * 0.1, (fy1 - fy0) * 0.1
        fbox = (fx0 - mx, fy0 - my, fx1 + mx, fy1 + my)
        closeup = (fy1 - fy0) > 0.3 * H                # the face fills the frame: arms and legs are guesses
    shoulders = {(1, 2), (1, 5)}
    for a, b in BODY_LIMBS:
        if has_face and (a in head or b in head):
            continue
        if closeup and (a, b) not in shoulders:
            continue
        if a < len(body) and b < len(body) and _limb_ok(body[a], body[b], cap):
            if fbox is not None:                       # never across the face
                cx, cy = (body[a][0] + body[b][0]) / 2, (body[a][1] + body[b][1]) / 2
                if fbox[0] <= cx <= fbox[2] and fbox[1] <= cy <= fbox[3]:
                    continue
            cv2.line(canvas, tuple(map(int, body[a])), tuple(map(int, body[b])), col, lw, cv2.LINE_AA)
    for k, q in enumerate(body):
        if not q or (has_face and k in head) or (closeup and k not in (1, 2, 5)):
            continue
        if fbox is not None and fbox[0] <= q[0] <= fbox[2] and fbox[1] <= q[1] <= fbox[3]:
            continue
        cv2.circle(canvas, tuple(map(int, q)), lw, col, -1, cv2.LINE_AA)
    if has_face and len(body) > 1 and body[1]:          # neck to the chin
        chin = face[8] if face[8] else None
        if chin: cv2.line(canvas, tuple(map(int, body[1])), tuple(map(int, chin)), col, max(1, lw // 2), cv2.LINE_AA)
    for key in ("hand_left_keypoints_2d", "hand_right_keypoints_2d"):
        hand = _pts(person.get(key), W, H)
        hp = [q for q in hand if q]
        if len(hp) < 12:                                   # a hand is drawn only when most of it was found
            continue
        hs = max(max(q[0] for q in hp) - min(q[0] for q in hp), max(q[1] for q in hp) - min(q[1] for q in hp))
        if hs > 0.5 * size:                                # a "hand" as big as the person is noise
            continue
        for a, b in HAND_LIMBS:
            if a < len(hand) and b < len(hand) and _limb_ok(hand[a], hand[b], 0.35 * size):
                cv2.line(canvas, tuple(map(int, hand[a])), tuple(map(int, hand[b])), col, max(1, lw // 2), cv2.LINE_AA)
    if has_face:
        fx = [q[0] for q in face[:68] if q]; fy = [q[1] for q in face[:68] if q]
        fs = max(max(fx) - min(fx), max(fy) - min(fy), 8.0)
        fw = max(2, int(fs * 0.04))
        light = tuple(min(255, int(c * 0.55 + 255 * 0.45)) for c in col)
        def poly(idx, closed):
            q = [face[i] for i in idx if face[i]]
            return np.array([[int(x), int(y)] for x, y in q], np.int32) if len(q) >= 3 else None
        for idx, closed in ((range(0, 17), False), (range(17, 22), False), (range(22, 27), False), (range(27, 31), False), (range(31, 36), False)):
            pl = poly(idx, closed)
            if pl is not None: cv2.polylines(canvas, [pl], closed, col, fw, cv2.LINE_AA)
        for idx in (range(36, 42), range(42, 48)):          # eyes: filled, so a blink is an area closing
            pl = poly(idx, True)
            if pl is not None: cv2.fillPoly(canvas, [pl], light, cv2.LINE_AA); cv2.polylines(canvas, [pl], True, col, max(1, fw // 2), cv2.LINE_AA)
        outer, inner = poly(range(48, 60), True), poly(range(60, 68), True)
        if outer is not None:
            cv2.fillPoly(canvas, [outer], col, cv2.LINE_AA)       # lips: solid color
        if inner is not None:
            cv2.fillPoly(canvas, [inner], (20, 20, 20), cv2.LINE_AA)   # mouth opening: dark area, its size = how open
        for q in face[68:70]:
            if q: cv2.circle(canvas, tuple(map(int, q)), max(1, fw // 2), (20, 20, 20), -1, cv2.LINE_AA)
    return canvas


def to_opencv_normals(img, convention):
    """Render MoGe Geometry draws normals flipped (Z, and Y for OpenGL); undo it to get the raw MoGe normals as RGB."""
    n = img[..., :3] * 2 - 1
    n = n * n.new_tensor([1.0, -1.0 if convention == "normal_opengl" else 1.0, -1.0])
    return (n * 0.5 + 0.5).clamp(0, 1)


def grow_masks(masks, rel):
    """Dilate each mask by rel * sqrt(its mean area) pixels (the margin around the subject)."""
    if rel <= 0:
        return masks
    import torch.nn.functional as F
    area = float(masks.sum(dim=(1, 2)).mean())
    r = max(1, int(round((area ** 0.5) * rel)))
    return F.max_pool2d(masks[:, None].float(), 2 * r + 1, 1, r)[:, 0]


def feather_inside_mask(mask, radius):
    """Soften a rendered mask edge without painting outside the selected pixels."""
    if radius <= 0:
        return mask.float()
    import cv2
    import numpy as np
    binary = mask.numpy().astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    alpha = np.clip(distance / radius, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.8)
    return torch.from_numpy(alpha * binary)


def anchor_windows(anchors, start, end):
    """Split a shot at nearest-anchor midpoints; ties stay with the earlier anchor."""
    anchors = sorted(a for a in set(anchors) if start <= a < end)
    for pos, anchor in enumerate(anchors):
        lo = start if pos == 0 else (anchors[pos - 1] + anchor) // 2 + 1
        hi = end if pos == len(anchors) - 1 else (anchor + anchors[pos + 1]) // 2 + 1
        yield anchor, lo, hi


def consecutive_runs(indices):
    run = []
    for i in indices:
        if run and i != run[-1] + 1:
            yield run
            run = []
        run.append(i)
    if run:
        yield run


def fill_empty_mask_runs(image, model, masks, report, shots, active_frame=-1):
    """Track only empty stretches from the nearest existing mask in the same shot."""
    filled = 0
    for s0, s1 in shots:
        anchors = [i for i in range(s0, s1) if bool(masks[i].any())]
        if not anchors:
            continue
        empty = [i for i in range(s0, s1) if not bool(masks[i].any())]
        for run in consecutive_runs(empty):
            lo, hi = run[0], run[-1] + 1
            if all(report[i].get("source") == "before first pick" for i in run):
                continue
            anchor = min(anchors, key=lambda a: (min(abs(a - lo), abs(a - (hi - 1))), a != active_frame, a))
            begin, end = min(anchor, lo), max(anchor + 1, hi)
            tracked = BFSTrackFromFrame().track(image[begin:end, ..., :3], model,
                                                masks[anchor][None].float(), anchor - begin)[0]
            for i in run:
                new_mask = tracked[i - begin] > 0.5
                if bool(new_mask.any()):
                    masks[i] = new_mask
                    report[i] = {"frame": i, "source": f"filled from {anchor}"}
                    filled += 1
    return filled


from collections import OrderedDict
_TRACK_CACHE = OrderedDict()   # (video, subject, its picks, cuts, options) -> (masks, report), most recent last
_LAST_TRACK_CACHE = OrderedDict()  # (video, subject) -> last completed propagated masks/report
_LOCK_CACHE = OrderedDict()    # (video, locked objects and cuts) -> tracked protection masks


def _cache_track_result(ckey, vkey, sj, masks, report, picks, keep_last=True):
    # Masks are binary after propagation. Keep one bool storage shared by both caches instead of
    # two full-resolution float32 clones (4x larger per frame).
    compact = masks > 0.5
    report_copy = [dict(r) for r in report]
    _remember(_TRACK_CACHE, ckey, (compact, report_copy), 24, _PICKER_MASK_CACHE_MAX_BYTES)
    if keep_last:
        _remember(_LAST_TRACK_CACHE, (vkey, sj), (compact, report_copy, picks), 24,
                  _PICKER_MASK_CACHE_MAX_BYTES)


def video_key(image):
    """A cheap fingerprint of the frames (subsampled pixels + shape) for the track cache."""
    import hashlib
    smp = image[:: max(1, image.shape[0] // 12), ::24, ::24, :3].float().cpu().numpy()
    return hashlib.sha1(smp.tobytes() + str(tuple(image.shape)).encode()).hexdigest()[:16]


def timeline_strip(image, per_subject, protected=None, width=256, max_frames=400):
    """Small JPEGs of every frame with each subject's mask tinted in its color, for the panel's whole-video scrubber
    (so a wrong mask can be found on any frame, not only on the strip)."""
    import os, uuid
    import numpy as np
    from PIL import Image
    n, h, w = image.shape[:3]
    idx = sorted(set(int(round(x)) for x in torch.linspace(0, n - 1, min(n, max_frames)).tolist()))
    sw = min(width, w); sh = max(1, round(h * sw / w))
    out_dir = os.path.join(sp._temp_dir(), "tl_" + uuid.uuid4().hex[:8]); os.makedirs(out_dir, exist_ok=True)
    files = []
    for i in idx:
        fr = image[i, ..., :3].float().clone()
        for sj, m in per_subject.items():
            c = torch.tensor(HYBRID_COLORS[(int(sj) - 1) % len(HYBRID_COLORS)], dtype=fr.dtype) / 255.0
            mk = (m[i] > 0.5)[..., None].to(fr.dtype) * 0.55
            fr = fr * (1 - mk) + c * mk
        if protected is not None and bool(protected[i].any()):
            mk = protected[i][..., None].to(fr.dtype) * 0.55
            fr = fr * (1 - mk) + fr.new_tensor([1.0, 0.78, 0.18]) * mk
        arr = (fr.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        name = f"f{i:05d}.jpg"
        Image.fromarray(arr).resize((sw, sh), Image.BILINEAR).save(os.path.join(out_dir, name), quality=80)
        files.append({"filename": name, "subfolder": "bfs_segpick/" + os.path.basename(out_dir), "type": "temp", "frame": i})
    return files


class Status:
    """Live status for the panel (message 'bfs_vsegpick_status' to this node) + ComfyUI's own progress bar + console log."""
    def __init__(self, uid):
        import time
        self.uid, self.t0, self.last = (str(uid) if uid is not None else None), time.time(), 0.0
        try:
            import comfy.utils
            self.bar = comfy.utils.ProgressBar(1000)
        except Exception:
            self.bar = None

    def __call__(self, text, progress=None, log=False):
        import time
        now = time.time()
        if progress is not None and self.bar is not None:
            self.bar.update_absolute(int(max(0.0, min(1.0, progress)) * 1000))
        if log:
            print(f"[BFS Video Segment Picker] {text}", flush=True)
        if not log and now - self.last < 0.25:      # throttle the frequent updates
            return
        self.last = now
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("bfs_vsegpick_status", {"node": self.uid, "text": text,
                                            "progress": progress, "log": bool(log), "elapsed": round(now - self.t0, 1)})
        except Exception:
            pass


def parse_frames(text):
    """{"frames": {"<frame>": {"<subject>": selection}}}; a frame holding a bare selection is subject 1."""
    try:
        s = json.loads(text or "{}") or {}
    except Exception:
        s = {}
    frames = s.get("frames") if isinstance(s.get("frames"), dict) else {}
    out = {}
    for k, v in frames.items():
        if not isinstance(v, dict):
            continue
        out[str(k)] = {"1": v} if ("selected" in v or "added" in v) else {
            str(sk): sv for sk, sv in v.items()
            if isinstance(sv, dict) and (str(sk) == "lock" or (str(sk).isdigit() and 1 <= int(sk) <= len(HYBRID_COLORS)))
        }
    return out


def find_connected_picture_count(prompt, unique_id):
    """Inspects the prompt workflow graph to find downstream nodes consuming this node's output,
    and returns the maximum index of connected picture inputs (e.g. 2 if only picture_1 and picture_2 are linked)."""
    if not prompt or not unique_id:
        return None
    uid = str(unique_id)
    direct_consumers = []
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for in_k, in_v in inputs.items():
            if isinstance(in_v, list) and len(in_v) >= 1 and str(in_v[0]) == uid:
                direct_consumers.append((nid, node))
                break
    max_pics = None
    for nid, node in direct_consumers:
        inputs = node.get("inputs", {})
        pics = []
        for k in range(1, 10):
            p = inputs.get(f"picture_{k}") or inputs.get(f"ref_image_{k}")
            if isinstance(p, list) and len(p) >= 1:
                pics.append(k)
        if pics:
            cnt = max(pics)
            max_pics = cnt if max_pics is None else max(max_pics, cnt)
    return max_pics


class BFSVideoSegmentPicker:
    DESCRIPTION = __doc__.split("\n\n")[0]

    @classmethod
    def INPUT_TYPES(cls):
        t = sp.BFSSegmentPicker.INPUT_TYPES()
        req = dict(t["required"])
        req["image"] = ("IMAGE", {"tooltip": f"The frames: the Keyframe Planner's source_frames or a short clip (at most {MAX_FRAMES})."})
        req["mark_opacity"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                             "tooltip": "Opacity of the subject mark in both outputs: the marked preview and the H3 hybrid guide. Use 1 for a clearly visible normal map; lower values blend the original frame into both outputs."})
        req["selection"] = ("STRING", {"default": "{}", "multiline": True, "tooltip": "Set by the panel: the selection on each frame."})
        req["track"] = ("BOOLEAN", {"default": True, "tooltip": "Frames without a selection take the nearest selected frame's mask, tracked with SAM 3."})
        opt = {
            "normals": ("IMAGE", {"lazy": True, "tooltip": "Optional: MoGe normals of the same frames (Run MoGe Inference -> Render MoGe Geometry, a normal mode). "
                                             "With it the node also returns the normal-map hybrid guide."}),
            "normals_convention": (["normal_directx", "normal_opengl"], {"default": "normal_directx",
                                   "tooltip": "The Render MoGe Geometry output mode the normals come from."}),
            "hybrid_base": ("IMAGE", {"tooltip": "Optional: the hybrid output of another Video Segment Picker, to add this subject on top (one picker per subject)."}),
            "pose_keypoint": ("POSE_KEYPOINT", {"lazy": True, "tooltip": "Optional: DWPose Estimator keypoints of the same frames (body + hands + face), "
                                                           "for mark_style 'openpose + face'."}),
            "molmo_clip": ("CLIP", {"lazy": True, "tooltip": "Optional: Molmo2 as H3 encoder. It sees the whole video, finds the picked person again after "
                                               "camera cuts and helps recover missed masks. In the default Molmo2 (classic) mode it describes the selected person, tracks that description through the video, and uses its points to guide SAM 3.1."}),
            "point_model": ("MOLMO_POINT_MODEL", {"lazy": True, "tooltip": "MolmoPoint-8B point tracker. Use with propagation_method 'MolmoPoint-8B'; its native point tokens guide SAM 3 without changing the H3 text encoder."}),
            "mark_style": (MARK_STYLES, {"default": MARK_STYLES[0],
                           "tooltip": "How each subject is drawn in the hybrid output, always in its color: the normal map, the DWPose "
                                      "skeleton with the face, or texture-free alternatives that need no MoGe (flat, blur, pixelate, gray "
                                      "detail, edges). 'normal map inside mask' hides source texture only inside the person: "
                                      "everything outside is protected automatically. It feathers the boundary and also "
                                      "keeps objects selected with the padlock or protect_mask unchanged inside the person mask. "
                                      "'translucent overlay' colors the exact mask with mark_opacity. "
                                      "'colored outline' draws only a thin border inside the mask, preserving frame detail "
                                      "and held objects. 'colored hatching + contour' draws lines and a matching inner contour, "
                                      "preserving the original texture and lighting between lines. These modes need no normals."}),
            "hybrid_grow": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01,
                                      "tooltip": "Margin around the subject in the hybrid guide, relative to its size (normal map: 0.04-0.12; "
                                                 "openpose: 0.08-0.18 around the person's convex hull)."}),
            "strip_frames": ("INT", {"default": 16, "min": 2, "max": MAX_FRAMES,
                                     "tooltip": "How many frames the panel shows to pick on (SAM segments each one): fewer is faster."}),
            "free_after": (FREE_MODES, {"default": FREE_MODES[0],
                           "tooltip": "After the node: free the GPU for the next nodes (H3). 'all models' also unloads H3 / the text "
                                      "encoder left from the last generation. 'release Molmo2 RAM' drops its CPU weights too, so the next Molmo2 run reloads them. Clearing the caches also "
                                      "frees RAM, but the next run segments and tracks again. Before MoGe / DWPose / Molmo2 run "
                                      "(only on Propagate) every model is unloaded anyway, so a second run does not run out of memory."}),
            "pose_region": (["person outline", "convex hull"], {"default": "person outline",
                            "tooltip": "openpose mark: the gray region around each subject. The outline (hybrid_grow ~0.02-0.05) hides "
                                       "the least background and usually looks best; the convex hull (with hybrid_grow margin) "
                                       "hides the original body shape (the new one stays free); the outline follows the person "
                                       "(less background hidden, but the original silhouette shows)."}),
            "pose_gray_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "openpose mark: how opaque the gray region is (1.0 = fully hidden). Lower lets the "
                                             "background show through, but also the original person (clothes, body), which the "
                                             "result may then copy."}),
            "max_subjects": (["auto", "1", "2", "3"], {"default": "auto",
                             "tooltip": "Limit active subjects: 'auto' automatically detects connected picture inputs in downstream H3 conditioning (ignoring extra subjects like green if only 2 pictures are connected), or force 1, 2, or 3."}),
            "propagate_scope": (["all frames", "from first selected frame onward", "only this frame", "from active frame onward", "fill only empty masks"], {"default": "fill only empty masks",
                             "tooltip": "Fill only empty masks keeps every existing non-empty frame unchanged and fills only frames whose mask is empty. Only this frame replaces one mask; from active frame onward recalculates from the open frame and preserves earlier results; all frames recomputes the clip."}),
            "propagation_method": (["Molmo2 (classic)", "MolmoPoint-8B", "SAM 3 (current)"], {"default": "Molmo2 (classic)",
                             "tooltip": "MolmoPoint-8B uses native point tokens to track and re-anchor the selected person; Molmo2 (classic) uses text coordinates. Both use SAM 3 for per-frame masks."}),
            "molmo_descriptions": ("STRING", {"multiline": False, "default": "",
                                   "tooltip": "Optional tracking description for subject 1. Leave blank to let Molmo2 describe the person you selected automatically. This only overrides the text query; the selected mask remains the visual anchor."}),
            "protect_mask": ("MASK", {"tooltip": "Optional external mask of objects to keep unchanged in 'normal map inside mask'. You can instead select protected objects with the padlock in this node. One mask or one per guide frame."}),
            "edge_feather": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.1, "step": 0.005,
                              "tooltip": "'normal map inside mask': soften the mark inward from the person and protected-object edges. Fraction of the shorter frame side."}),
            "hatch_width": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                              "tooltip": "Hatching line and contour thickness multiplier. 1 matches the training format."}),
            "hatch_spacing": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                                "tooltip": "Hatching line spacing multiplier. 1 matches the training format."}),
        }
        return {"required": req, "optional": opt, "hidden": {**t.get("hidden", {}), "prompt": "PROMPT"}}

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("masks", "marked", "overlay", "report", "hybrid")
    OUTPUT_TOOLTIPS = ("One mask per frame: the person to replace.", "Every frame with the person tinted (mark_color, mark_opacity).",
                       "The strip frames with all objects found, numbered.", "JSON per frame: picked by hand, tracked from frame N, or empty.",
                       "The H3 guide with each subject drawn in its color using mark_style and mark_opacity; 'normal map inside mask' keeps pixels outside the person and protect_mask unchanged.")
    FUNCTION = "pick"
    CATEGORY = "BFS/Mask"
    OUTPUT_NODE = True

    def pick(self, image, sam3_model, sam3_clip, concepts, threshold, max_per_concept, min_area, mark_color, mark_opacity,
             selection, track=True, strip_frames=16, normals=None, normals_convention="normal_directx", hybrid_base=None, subject=None,
             mark_style=MARK_STYLES[0], molmo_clip=None, point_model=None, pose_keypoint=None,
             hybrid_grow=0.08, free_after=FREE_MODES[0], pose_region="person outline",
             pose_gray_opacity=1.0, max_subjects="auto", molmo_descriptions="", propagate_scope="fill only empty masks",
             propagation_method="Molmo2 (classic)", unique_id=None, prompt=None, protect_mask=None, edge_feather=0.01,
             hatch_width=1.0, hatch_spacing=1.0, **kw):
        try:
            return self._pick(image, sam3_model, sam3_clip, concepts, threshold, max_per_concept, min_area, mark_color, mark_opacity,
                              selection, track, strip_frames, normals, normals_convention, hybrid_base, subject, mark_style,
                              molmo_clip, pose_keypoint, hybrid_grow, unique_id, pose_region, pose_gray_opacity,
                              max_subjects, molmo_descriptions, propagate_scope, propagation_method, prompt,
                              protect_mask=protect_mask, edge_feather=edge_feather, point_model=point_model,
                              hatch_width=hatch_width, hatch_spacing=hatch_spacing)
        finally:
            self._free(sam3_model, free_after, point_model or molmo_clip)

    def check_lazy_status(self, selection="{}", mark_style=MARK_STYLES[0], track=True, normals=None, pose_keypoint=None,
                          molmo_clip=None, point_model=None, propagation_method="Molmo2 (classic)", propagate_scope="fill only empty masks",
                          image=None, unique_id=None, prompt=None, **kw):
        """The heavy optional inputs (MoGe normals, DWPose keypoints, Molmo2) are only needed when propagating, not while
        picking; before they run, every model leaves the GPU (e.g. H3 from the last generation), or MoGe runs out of memory."""
        try:
            segment_only = bool(json.loads(selection or "{}").get("segment_only"))
        except Exception:
            segment_only = False
        if segment_only:
            return []
        need = []
        if normals is None and ("needs normals" in mark_style or mark_style == PROTECTED_NORMAL_STYLE):
            need.append("normals")
        if pose_keypoint is None and mark_style.startswith("openpose"):
            need.append("pose_keypoint")
        reuse_existing = False
        if propagate_scope == "fill only empty masks" and isinstance(image, torch.Tensor):
            states = parse_frames(selection)
            subjects = {int(sk) for frame_state in states.values() for sk, state in frame_state.items()
                        if str(sk).isdigit() and state.get("selected")} or {1}
            vkey = video_key(image)
            reuse_existing = all((vkey, sj) in _LAST_TRACK_CACHE for sj in subjects)
        if molmo_clip is None and track and propagation_method == "Molmo2 (classic)" and not reuse_existing:
            need.append("molmo_clip")
        if point_model is None and track and propagation_method == "MolmoPoint-8B" and not reuse_existing:
            need.append("point_model")
        links = ((prompt or {}).get(str(unique_id)) or {}).get("inputs", {})
        need = [x for x in need if isinstance(links.get(x), list)]    # only connected inputs can be requested
        if need:
            try:
                import comfy.model_management as mm
                mm.unload_all_models(); mm.soft_empty_cache()
                print(f"[BFS Video Segment Picker] all models unloaded before {', '.join(need)}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[BFS Video Segment Picker] could not unload models: {e!r}", flush=True)
        return need

    @staticmethod
    def _free(sam3_model, mode, molmo_clip=None):
        mode = LEGACY_FREE.get(mode, mode)
        if mode.startswith("keep"):
            return
        try:
            import gc
            import comfy.model_management as mm
            if mode.startswith("unload all"):
                mm.unload_all_models()
            else:
                mm.unload_model_and_clones(sam3_model)
            if "caches" in mode:
                sp._SESSIONS.clear(); _TRACK_CACHE.clear(); _LAST_TRACK_CACHE.clear(); _SHOT_CACHE.clear(); _MOLMO_CACHE.clear()
            if "release Molmo2 RAM" in mode and molmo_clip is not None:
                runtime = getattr(molmo_clip, "runtime", None)
                if runtime is not None and hasattr(runtime, "release_ram"):
                    runtime.release_ram()
                if hasattr(molmo_clip, "cache"):
                    molmo_clip.cache.clear()
            gc.collect()
            mm.soft_empty_cache(True)
            print(f"[BFS Video Segment Picker] memory freed ({mode})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[BFS Video Segment Picker] could not free memory: {e!r}", flush=True)

    def _pick(self, image, sam3_model, sam3_clip, concepts, threshold, max_per_concept, min_area, mark_color, mark_opacity,
              selection, track, strip_frames, normals, normals_convention, hybrid_base, subject, mark_style, molmo_clip,
              pose_keypoint, hybrid_grow, unique_id, pose_region="person outline", pose_gray_opacity=1.0,
              max_subjects="auto", molmo_descriptions="", propagate_scope="fill only empty masks",
              propagation_method="Molmo2 (classic)", prompt=None, protect_mask=None, edge_feather=0.01,
              point_model=None, hatch_width=1.0, hatch_spacing=1.0):
        st = Status(unique_id)
        n = int(image.shape[0])
        st(f"start: {n} frames", 0.0, log=True)
        strip = sorted(set(int(round(x)) for x in torch.linspace(0, n - 1, min(n, int(strip_frames), MAX_FRAMES)).tolist()))
        for sel_frame in parse_frames(selection):          # frames picked on earlier stay in the strip
            if sel_frame.isdigit() and int(sel_frame) < n and int(sel_frame) not in strip:
                strip = sorted(strip + [int(sel_frame)])
        sp.MAX_SESSIONS = max(sp.MAX_SESSIONS, MAX_FRAMES + 8)   # every frame keeps its session for the panel's "add"
        h, w = image.shape[1:3]
        spec = {"concepts": ", ".join(sp.split_concepts(concepts)), "threshold": round(float(threshold), 4),
                "max_per_concept": int(max_per_concept), "min_area": round(float(min_area), 5)}
        states = parse_frames(selection)
        try:
            selection_meta = json.loads(selection or "{}")
            segment_only = bool(selection_meta.get("segment_only"))
            active_frame = int(selection_meta.get("active_frame", -1))
        except Exception:
            segment_only = False
            active_frame = -1
        if segment_only:
            st("segment only (picking): no tracking or Molmo2; press Apply fix to track", None, log=True)
        if not segment_only and normals is None and (mark_style == PROTECTED_NORMAL_STYLE or "needs normals" in mark_style):
            raise ValueError("Connect MoGe normals to 'normals' for this mark style, or choose 'translucent overlay' to render without normals.")
        user_descs = [l.strip().strip('. ') for l in str(molmo_descriptions or '').splitlines()]
        while user_descs and not user_descs[-1]: user_descs.pop()

        subjects = sorted({int(sk) for frame_state in states.values() for sk, sv in frame_state.items()
                           if str(sk).isdigit() and (sv.get("selected") or [])} or {1})
        max_allowed = None
        if str(max_subjects) in ("1", "2", "3"):
            max_allowed = int(max_subjects)
        else:
            max_allowed = find_connected_picture_count(prompt, unique_id)
        if max_allowed is not None:
            active = [s for s in subjects if s <= max_allowed]
            if not active:
                active = [1]
            if set(active) != set(subjects):
                dropped = sorted(set(subjects) - set(active))
                st(f"auto-desmarcação: sujeito(s) {dropped} desativado(s) automaticamente (limite de {max_allowed} referências no H3)", None, log=True)
                subjects = active
        panels, overlays = [], []
        own = {sj: {} for sj in subjects}
        own_locked = {}
        for k_strip, i in enumerate(strip):
            st(f"SAM 3: segmenting frame {i} ({k_strip + 1}/{len(strip)})", 0.02 + 0.28 * k_strip / max(1, len(strip)))
            frame = image[i:i + 1, ..., :3].float().cpu()
            sid, sess = sp.session_for((sam3_model, sam3_clip), frame, spec)
            chosen_all = set()
            for sj in subjects:
                state = sp.parse_state(json.dumps(states.get(str(i), {}).get(str(sj), {})))
                objects = sp.all_objects(sess, state)
                by_id = {o["id"]: o for o in objects}
                chosen = [by_id[k] for k in state["selected"] if k in by_id]
                chosen_all |= {o["id"] for o in chosen}
                cut = [o["mask"] for o in chosen if o.get("remove")]
                cut = torch.stack(cut).any(0) if cut else torch.zeros(h, w, dtype=torch.bool)
                mask, _ = sp.compose([o["mask"] & ~cut for o in chosen if not o.get("remove")], h, w)
                own[sj][i] = mask.reshape(h, w).float()
            lock_state = sp.parse_state(json.dumps(states.get(str(i), {}).get("lock", {})))
            lock_objects = sp.all_objects(sess, lock_state)
            lock_by_id = {o["id"]: o for o in lock_objects}
            lock_chosen = [lock_by_id[k] for k in lock_state["selected"] if k in lock_by_id]
            chosen_all |= {o["id"] for o in lock_chosen}
            lock_cut = [o["mask"] for o in lock_chosen if o.get("remove")]
            lock_cut = torch.stack(lock_cut).any(0) if lock_cut else torch.zeros(h, w, dtype=torch.bool)
            lock_mask, _ = sp.compose([o["mask"] & ~lock_cut for o in lock_chosen if not o.get("remove")], h, w)
            lock_mask = lock_mask.reshape(h, w) > 0.5
            selected_people = torch.stack([own[sj][i] > 0.5 for sj in subjects]).any(0)
            if bool(lock_mask.any()) and mask_overlap(lock_mask, selected_people)[0] >= 0.7:
                st(f"frame {i}: lock overlaps the selected subject; pick the object by text or add negative points", None, log=True)
                lock_mask = torch.zeros_like(lock_mask)
            own_locked[i] = lock_mask
            objects = sp.all_objects(sess, sp.parse_state(json.dumps({"added": [a for st in states.get(str(i), {}).values() for a in st.get("added", [])]})))
            panels.append(sp.panel_data(sid, sess, objects))
            overlays.append(sp.overlay(frame, objects, chosen_all))
        cuts = detect_cuts(image) if n > 1 else []
        if cuts:
            st(f"camera cuts at frames {cuts}", 0.3, log=True)
        per_subject, report_s = {}, {}
        vkey = video_key(image)
        lock_picks = {k: frame_state["lock"] for k, frame_state in sorted(states.items()) if frame_state.get("lock")}
        lock_anchors = sorted(i for i in strip if bool(own_locked[i].any()))
        locked = None
        if lock_anchors:
            lock_key = (vkey, json.dumps(spec, sort_keys=True), json.dumps(lock_picks, sort_keys=True),
                        tuple(cuts), bool(track), bool(segment_only))
            cached_lock = _LOCK_CACHE.get(lock_key)
            if cached_lock is not None:
                locked = cached_lock
                _LOCK_CACHE.move_to_end(lock_key)
            else:
                locked = torch.zeros((n, h, w), dtype=torch.bool)
                if track and not segment_only and n > 1:
                    for s0, s1 in shots_of(n, cuts):
                        shot_anchors = [a for a in lock_anchors if s0 <= a < s1]
                        for a in shot_anchors:
                            st(f"SAM 3: tracking protected object from frame {a} through shot {s0}-{s1 - 1}", 0.3)
                            tracked = BFSTrackFromFrame().track(
                                image[s0:s1, ..., :3], sam3_model, own_locked[a][None].float(), a - s0
                            )[0].detach().cpu() > 0.5
                            locked[s0:s1] |= tracked
                for a in lock_anchors:
                    locked[a] = own_locked[a]
                _remember(_LOCK_CACHE, lock_key, locked, 16, _PICKER_MASK_CACHE_MAX_BYTES)
        for sj in subjects:
            mine = {k: v[str(sj)] for k, v in sorted(states.items()) if v.get(str(sj))}   # only this subject's picks
            previous = _LAST_TRACK_CACHE.get((vkey, sj))
            ckey = ("v4", vkey, sj, json.dumps(mine, sort_keys=True), tuple(cuts), bool(track),
                    (point_model if propagation_method == "MolmoPoint-8B" else molmo_clip) is not None,
                    float(threshold),
                    propagate_scope, propagation_method, active_frame)
            if (not segment_only and propagate_scope not in ("all frames", "fill only empty masks") and previous is not None
                    and len(previous) > 2 and previous[2] == mine):
                per_subject[sj], report_s[sj] = previous[0].clone(), [dict(r) for r in previous[1]]
                continue
            hit = None if propagate_scope == "fill only empty masks" else _TRACK_CACHE.get(ckey)
            if hit is not None:   # same video, same picks for this subject: reuse its masks (style / margin changes are instant)
                st(f"subject {sj}: masks reused from the last run (same video and picks)", 0.3 + 0.4 * subjects.index(sj) / len(subjects), log=True)
                per_subject[sj], report_s[sj] = hit[0].clone(), [dict(r) for r in hit[1]]
                _TRACK_CACHE.move_to_end(ckey)
                continue
            anchors = [i for i in strip if bool(own[sj][i].any())]
            user_anchors = list(anchors)
            fix_frame = active_frame if 0 <= active_frame < n else (user_anchors[-1] if user_anchors else 0)
            first_pick = (fix_frame if propagate_scope == "from active frame onward" else
                          (user_anchors[0] if user_anchors else 0))
            if propagate_scope == "from active frame onward":
                anchors = [a for a in anchors if a >= fix_frame]
                if not anchors and previous is not None:
                    per_subject[sj], report_s[sj] = previous[0].clone(), [dict(r) for r in previous[1]]
                    continue
            prev_strip = max([s for s in strip if s < first_pick], default=0)
            min_valid_frame = (first_pick if propagate_scope in ("from first selected frame onward", "from active frame onward")
                               else (0 if first_pick == 0 else prev_strip))
            if segment_only:   # picking only: the picked frames' masks (the last propagated masks are kept by the cache)
                m = previous[0].clone() if previous is not None else torch.zeros(n, h, w, dtype=torch.bool)
                rp_prev = [dict(r) for r in previous[1]] if previous is not None else [
                    {"frame": i, "source": "empty"} for i in range(n)]
                for i in user_anchors:
                    m[i] = own[sj][i] > 0.5
                    rp_prev[i] = {"frame": i, "source": "picked (pending propagate)"}
                per_subject[sj] = m
                report_s[sj] = rp_prev
                continue
            if propagate_scope == "fill only empty masks" and previous is not None:
                # A local correction must not trigger Molmo2 or retrack good frames.
                # Apply changed picks first, then use nearby existing masks only for gaps.
                m = previous[0].clone()
                rep = [dict(r) for r in previous[1]]
                old_picks = previous[2] if len(previous) > 2 else {}
                edited = {a for a in user_anchors if mine.get(str(a)) != old_picks.get(str(a))}
                if fix_frame in user_anchors:
                    edited.add(fix_frame)
                for a in sorted(edited):
                    m[a] = own[sj][a] > 0.5
                    rep[a] = {"frame": a, "source": "picked (frame fix)"}
                filled = fill_empty_mask_runs(image, sam3_model, m, rep, shots_of(n, cuts), fix_frame) if track else 0
                st(f"subject {sj}: {len(edited)} picked frame(s) applied, {filled} empty frame(s) filled; "
                   "existing masks kept", None, log=True)
                per_subject[sj] = m > 0.5; report_s[sj] = rep
                _cache_track_result(ckey, vkey, sj, m, rep, mine)
                continue
            m = torch.zeros(n, h, w)
            for i in anchors:
                m[i] = own[sj][i]
            rep = [{"frame": i, "source": "picked" if i in anchors else "empty"} for i in range(n)]
            if propagate_scope == "only this frame":
                m = previous[0].clone() if previous is not None else torch.zeros(n, h, w, dtype=torch.bool)
                rep = [dict(r) for r in previous[1]] if previous is not None else [
                    {"frame": i, "source": "empty"} for i in range(n)]
                if 0 <= fix_frame < n and bool(own[sj].get(fix_frame, torch.zeros(h, w)).any()):
                    m[fix_frame] = own[sj][fix_frame] > 0.5
                    rep[fix_frame] = {"frame": fix_frame, "source": "picked (frame fix)"}
                per_subject[sj] = m; report_s[sj] = rep
                _cache_track_result(ckey, vkey, sj, m, rep, mine)
                continue
            if anchors and track and n > 1:
                def claimed_at(f):
                    out_c = [per_subject[ok][f] for ok in per_subject if ok != sj and bool((per_subject[ok][f] > 0.5).any())]
                    for ok in subjects:
                        if ok != sj and f in own.get(ok, {}) and bool((own[ok][f] > 0.5).any()):
                            out_c.append(own[ok][f])
                    return out_c

                shots = shots_of(n, cuts)
                shot_of = lambda f: next(k for k, (s0, s1) in enumerate(shots) if s0 <= f < s1)
                has = [any(s0 <= a < s1 for a in anchors) for s0, s1 in shots]
                tracker = point_model if propagation_method == "MolmoPoint-8B" else molmo_clip
                use_molmo = (propagation_method in ("Molmo2 (classic)", "MolmoPoint-8B")
                             and tracker is not None and hasattr(tracker, "runtime"))
                if propagation_method == "MolmoPoint-8B" and not use_molmo:
                    raise ValueError("Connect Auto Download & Load MolmoPoint-8B to point_model.")
                pts = []
                if use_molmo:
                    st(f"Molmo2: describing and tracking subject {sj} from frame {first_pick} forward", 0.3 + 0.4 * subjects.index(sj) / len(subjects))
                    t0 = time.time()
                    claimed_map = {f: claimed_at(f) for f in range(first_pick + 1, n)}
                    desc_override = user_descs[sj - 1] if 0 <= sj - 1 < len(user_descs) and user_descs[sj - 1] else None
                    if desc_override:
                        st(f"Molmo2 subject {sj}: using explicit text description '{desc_override}'", None, log=True)
                    relocate = molmo_relocate_legacy if propagation_method in ("Molmo2 (classic)", "MolmoPoint-8B") else molmo_relocate
                    if propagation_method in ("Molmo2 (classic)", "MolmoPoint-8B"):
                        extra, note, pts = relocate(
                            tracker, (sam3_model, sam3_clip), image, own[sj][first_pick], first_pick,
                            shots, has, float(threshold), memo=(vkey, sj, propagation_method, id(tracker.runtime),
                                                                 float(threshold), desc_override or ""),
                            text_override=desc_override
                        )
                    else:
                        extra, note, pts = relocate(
                            molmo_clip, (sam3_model, sam3_clip), image, own[sj][first_pick], first_pick,
                            shots, has, float(threshold), memo=(vkey, sj, float(threshold), desc_override or ""),
                            claimed_by_frame=claimed_map, text_override=desc_override
                        )
                    note += f" ({time.time() - t0:.0f}s)"
                    st(f"subject {sj}: {note}", None, log=True)
                    for f, mk in extra.items():
                        own[sj][f] = mk; anchors = sorted(set(anchors) | {f})

                def track_shots(anchors):
                    out, rp = {}, {}
                    kept = 0
                    for si_, (s0, s1) in enumerate(shots):   # each shot is tracked only from anchors inside it: never across a cut
                        if s1 <= min_valid_frame:
                            for i in range(s0, s1):
                                out[i] = torch.zeros(h, w); rp[i] = "before first pick"
                            continue
                        sa = [a for a in anchors if s0 <= a < s1]
                        skey = ("v3", vkey, sj, s0, s1, min_valid_frame, tuple((a, mask_key(own[sj][a])) for a in sa), float(threshold))
                        if skey in _SHOT_CACHE and propagate_scope != "fill only empty masks":   # fill mode retries shots with empty masks instead of reusing a failed track
                            cm, cr = _SHOT_CACHE[skey]
                            _SHOT_CACHE.move_to_end(skey)
                            for i in range(s0, s1):
                                out[i] = cm[i - s0].float(); rp[i] = cr[i - s0]
                            kept += 1
                            continue
                        st(f"SAM 3: tracking subject {sj}, shot {si_ + 1}/{len(shots)} (frames {s0}-{s1 - 1})",
                           0.3 + 0.4 * (subjects.index(sj) + si_ / max(1, len(shots))) / len(subjects))
                        seg_s0 = max(s0, min_valid_frame)
                        windows = list(anchor_windows(sa, seg_s0, s1))
                        if not windows:
                            for i in range(s0, s1):
                                out[i] = torch.zeros(h, w)
                                rp[i] = "before first pick" if i < first_pick else "lost (no anchor in this shot)"
                            continue
                        for i in range(s0, seg_s0):
                            out[i] = torch.zeros(h, w); rp[i] = "before first pick"
                        for a, lo, hi in windows:
                            seg = image[lo:hi, ..., :3]
                            tr = (BFSTrackFromFrame().track(seg, sam3_model, own[sj][a][None], a - lo)[0]
                                  if hi - lo > 1 else own[sj][a][None])
                            for i in range(lo, hi):
                                if i in sa:
                                    out[i] = own[sj][i]; rp[i] = "picked" if i in user_anchors else "Molmo2 + SAM"; continue
                                out[i] = (tr[i - lo] > 0.5).float()
                                rp[i] = f"tracked from {a}" if bool(out[i].any()) else (
                                    "before first pick" if i < first_pick else f"lost (tracked from {a})")
                        if first_pick > seg_s0 and seg_s0 <= first_pick < s1:
                            for i in range(first_pick - 1, seg_s0 - 1, -1):
                                ok_bwd = (
                                    bool(out[i].any())
                                    and mask_overlap(out[i], out[i + 1])[0] >= 0.15
                                    and not any(mask_overlap(out[i], cm)[1] > 0.55 for cm in claimed_at(i))
                                    and person_color_sim(image[first_pick, ..., :3], own[sj][first_pick], image[i, ..., :3], out[i]) >= 0.48
                                )
                                if not ok_bwd:
                                    for j in range(seg_s0, i + 1):
                                        out[j] = torch.zeros(h, w); rp[j] = "before first pick"
                                    break
                        for i in range(seg_s0, s1):
                            if bool(out[i].any()) and i not in user_anchors:
                                cl = claimed_at(i)
                                if cl:
                                    cu = torch.stack([cm > 0.5 for cm in cl]).any(0)
                                    out[i] = out[i] * (~cu).float()
                                    if not bool(out[i].any()):
                                        rp[i] = "before first pick" if i < first_pick else rp[i].replace("tracked from", "lost (tracked from") + (")" if not rp[i].endswith(")") else "")
                        if sa:
                            _remember(_SHOT_CACHE, skey, (torch.stack([out[i] > 0.5 for i in range(s0, s1)]),
                            [rp[i] for i in range(s0, s1)]), 64, _PICKER_MASK_CACHE_MAX_BYTES)
                    if kept:
                        st(f"subject {sj}: {kept}/{len(shots)} shots unchanged, reused", None, log=True)
                    return out, rp

                out, rp = track_shots(anchors)
                if pts and propagation_method in ("Molmo2 (classic)", "MolmoPoint-8B"):
                    # Classic behavior: Molmo points audit the full SAM track; two nearby misses indicate a lost stretch.
                    off = set()
                    for t_, _o, x_, y_ in pts:
                        f_ = min(n - 1, max(0, int(round(float(t_) * 24))))
                        if f_ < min_valid_frame:
                            continue
                        px_, py_ = min(w - 1, int(x_ * w)), min(h - 1, int(y_ * h))
                        mk_ = out[f_] > 0.5
                        if bool(mk_.any()) and not bool(mk_[max(0, py_ - 6):py_ + 7, max(0, px_ - 6):px_ + 7].any()):
                            off.add(f_)
                    offs = sorted(off); wrong = set()
                    for a_, b_ in zip(offs, offs[1:]):
                        if b_ - a_ <= 8: wrong.update(range(a_, b_ + 1))
                    lost = sorted({i for i in range(min_valid_frame, n) if not bool(out[i].any())} | wrong)
                    runs, cur = [], []
                    for i in lost:
                        if cur and i != cur[-1] + 1: runs.append(cur); cur = []
                        cur.append(i)
                    if cur: runs.append(cur)
                    added = 0
                    for run in [r for r in runs if len(r) >= 6]:
                        inside = [q for q in pts if run[0] <= int(round(float(q[0]) * 24)) <= run[-1]]
                        if not inside: continue
                        mid = (run[0] + run[-1]) / 2.0
                        t, _o, x, y = min(inside, key=lambda q: abs(float(q[0]) * 24 - mid))
                        f = min(run[-1], max(run[0], int(round(float(t) * 24))))
                        mk = person_at((sam3_model, sam3_clip), image[f:f + 1, ..., :3].float().cpu(), x, y, float(threshold))
                        if bool(mk.any()):
                            own[sj][f] = mk.float(); anchors = sorted(set(anchors) | {f}); added += 1
                    if added:
                        st(f"subject {sj}: {added} lost stretches re-anchored by Molmo2 (classic mode), tracking again", None, log=True)
                        out, rp = track_shots(anchors)
                if pts and propagation_method not in ("Molmo2 (classic)", "MolmoPoint-8B"):   # current mode recovery
                    ref_crop = person_crop(image[first_pick, ..., :3], own[sj][first_pick])
                    off = set()
                    for t_, _o, x_, y_ in pts:
                        f_ = min(n - 1, max(0, int(round(float(t_) * 24))))
                        if f_ <= first_pick:
                            continue
                        px_, py_ = min(w - 1, int(x_ * w)), min(h - 1, int(y_ * h))
                        mk_ = out[f_] > 0.5
                        if bool(mk_.any()) and not bool(mk_[max(0, py_ - 6):py_ + 7, max(0, px_ - 6):px_ + 7].any()):
                            if person_color_sim(image[first_pick, ..., :3], own[sj][first_pick], image[f_, ..., :3], mk_) < 0.42:
                                off.add(f_)
                    # Molmo points about every 6 frames: two in a row off the mask -> the frames between them count as lost
                    offs = sorted(off); wrong = set()
                    for a_, b_ in zip(offs, offs[1:]):
                        if b_ - a_ <= 8: wrong.update(range(a_, b_ + 1))
                    lost = sorted({i for i in range(first_pick + 1, n) if not bool(out[i].any())} | wrong)
                    runs, cur = [], []
                    for i in lost:
                        if cur and i != cur[-1] + 1: runs.append(cur); cur = []
                        cur.append(i)
                    if cur: runs.append(cur)
                    added = 0
                    for run in [r for r in runs if len(r) >= 6]:
                        inside = [q for q in pts if run[0] <= int(round(float(q[0]) * 24)) <= run[-1]]
                        if not inside: continue
                        mid = (run[0] + run[-1]) / 2.0
                        seen_f = set()
                        for t, _o, x, y in sorted(inside, key=lambda q: abs(float(q[0]) * 24 - mid)):
                            f = min(run[-1], max(run[0], int(round(float(t) * 24))))
                            if f in seen_f:
                                continue
                            seen_f.add(f)
                            mk = person_at((sam3_model, sam3_clip), image[f:f + 1, ..., :3].float().cpu(), x, y, float(threshold))
                            if bool(mk.any()) and molmo_verify_candidate(
                                molmo_clip, ref_crop, image[first_pick, ..., :3], own[sj][first_pick],
                                image[f, ..., :3], mk, claimed_masks=claimed_at(f), memo=(vkey, sj, float(threshold), first_pick, f)
                            ):
                                own[sj][f] = mk.float(); anchors = sorted(set(anchors) | {f}); added += 1
                                break
                            if len(seen_f) >= 3:
                                break
                    if added:
                        st(f"subject {sj}: {added} lost stretches re-anchored by Molmo2 (visually verified), tracking again", None, log=True)
                        prev_out, prev_rp = dict(out), dict(rp)
                        out, rp = track_shots(anchors)
                        for i in range(n):
                            if bool(prev_out[i].any()) and i not in wrong:
                                out[i], rp[i] = prev_out[i], prev_rp[i]
                for i in range(n):
                    m[i] = out[i]; rep[i] = {"frame": i, "source": rp[i]}
            if propagate_scope == "from active frame onward" and previous is not None:
                m[:fix_frame] = previous[0][:fix_frame].to(m.dtype)
                rep[:fix_frame] = [dict(r) for r in previous[1][:fix_frame]]
            per_subject[sj] = m > 0.5; report_s[sj] = rep
            _cache_track_result(ckey, vkey, sj, m, rep, mine, keep_last=not segment_only)
        masks = torch.stack(list(per_subject.values())).amax(0).float()
        anchors = sorted({i for sj in subjects for i in strip if bool(own[sj][i].any())})
        report = [{"frame": i, "subjects": {str(sj): report_s[sj][i]["source"] for sj in subjects}} for i in range(n)]
        if not anchors:
            st("nothing selected yet: click a frame, then the person, then run again", None, log=True)
        color = torch.tensor(sp.parse_color(mark_color), dtype=image.dtype) / 255.0
        a = (masks.to(image.dtype) * float(mark_opacity))[..., None]
        marked = image[..., :3] * (1 - a) + color * a
        if mark_style == PROTECTED_NORMAL_STYLE and edge_feather > 0:
            # The marked output should show the same soft boundary as the H3 guide.
            feather_px = max(1, round(min(h, w) * float(edge_feather)))
            for i in range(n):
                alpha = feather_inside_mask(masks[i] > 0.5, feather_px).to(image.dtype)
                alpha = (alpha * float(mark_opacity))[..., None]
                marked[i] = image[i, ..., :3] * (1 - alpha) + color * alpha
        base_source = (hybrid_base if hybrid_base is not None else image)[..., :3]
        base = base_source.clone()
        if mark_style.startswith("openpose") and pose_keypoint is None and segment_only:
            st("picking: DWPose runs on Apply fix, then the hybrid shows the marks", None, log=True)
        elif mark_style.startswith("openpose"):
            if pose_keypoint is None:
                raise ValueError("mark_style 'openpose + face' needs the pose_keypoint input (DWPose Estimator on the same frames).")
            frames_kp = pose_keypoint if isinstance(pose_keypoint, list) else [pose_keypoint]
            if len(frames_kp) != n:
                raise ValueError(f"{len(frames_kp)} pose frames for {n} frames: run DWPose on the same frames.")
            import numpy as np
            import cv2
            st("drawing the DWPose marks", 0.75)
            ga = float(pose_gray_opacity)
            subj_data = {}
            # Pass 1: compute hulls and erase all subjects to neutral gray FIRST so no subject's gray region erases another subject's skeleton
            for sj, m in per_subject.items():
                mk = (m > 0.5).numpy().astype(np.uint8)
                area = float(mk.sum(axis=(1, 2)).mean()) + 1
                hull = pose_region.startswith("convex")
                r = max(3, int((area ** 0.5) * (max(float(hybrid_grow), 0.08) if hull else float(hybrid_grow))))
                se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                hulls = []
                for i in range(n):
                    hm_i = np.zeros_like(mk[i])
                    cs, _ = cv2.findContours(mk[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cs:
                        cv2.fillPoly(hm_i, [cv2.convexHull(np.concatenate(cs))] if hull else cs, 1)
                    region = torch.from_numpy(cv2.dilate(hm_i, se) > 0)
                    hulls.append(region)
                    if bool(region.any()):
                        base[i][region] = base[i][region] * (1 - ga) + 0.5 * ga
                subj_data[sj] = (mk, hulls)

            # Pass 2: match DWPose skeletons per frame (with relaxed fallback + neighbor frame fallback) and draw all skeletons on top
            used_per_frame = {i: [] for i in range(n)}
            for sj, (mk, hulls) in subj_data.items():
                col = HYBRID_COLORS[(int(sj) - 1) % len(HYBRID_COLORS)]
                matched = [None] * n
                for i in range(n):
                    if not bool(hulls[i].any()):
                        continue
                    fk = frames_kp[i] or {}
                    ppl = fk.get("people")
                    tight = cv2.dilate(mk[i], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
                    person = pick_person(ppl, tight, w, h, exclude=used_per_frame[i])
                    if person is None:
                        grow_t = cv2.dilate(mk[i], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) > 0
                        person = pick_person(ppl, grow_t, w, h, min_inside=0.25, min_diag=0.2, exclude=used_per_frame[i])
                    if person is not None:
                        matched[i] = person
                        used_per_frame[i].append(person)
                for i in range(n):
                    if not bool(hulls[i].any()):
                        continue
                    person = matched[i]
                    if person is None:
                        for delta in (-1, 1, -2, 2):
                            j = i + delta
                            if 0 <= j < n and matched[j] is not None:
                                person = matched[j]
                                break
                    if person is None:
                        continue
                    cv = torch.from_numpy(draw_pose(person, col, w, h)).to(base.dtype) / 255.0
                    drawn = cv.amax(-1) > 0
                    base[i][drawn] = cv[drawn]                             # the skeleton may leave the region: the body is free
        needs_normals = "needs normals" in mark_style and not mark_style.startswith("openpose")
        if mark_style == PROTECTED_NORMAL_STYLE and normals is None and not segment_only:
            raise ValueError("'normal map inside mask' needs the normals input from MoGe.")
        if not mark_style.startswith("openpose") and (normals is not None or not needs_normals):
            import torch.nn.functional as F
            protected = None
            if mark_style == PROTECTED_NORMAL_STYLE and protect_mask is not None:
                protected = protect_mask.detach().float().cpu()
                if protected.ndim == 2:
                    protected = protected[None]
                if protected.ndim != 3 or protected.shape[0] not in (1, n):
                    raise ValueError("protect_mask must be one mask or one mask per guide frame.")
                if protected.shape[1:] != (h, w):
                    protected = F.interpolate(protected[:, None], size=(h, w), mode="nearest")[:, 0]
                if protected.shape[0] == 1:
                    protected = protected.expand(n, -1, -1)
                protected = protected > 0.5
            if mark_style in (PROTECTED_NORMAL_STYLE, HATCHING_STYLE) and locked is not None:
                protected = locked if protected is None else (protected | locked)
            if needs_normals and normals is not None:
                if normals.shape[0] != n:
                    raise ValueError(f"{normals.shape[0]} normal maps for {n} frames: run MoGe on the same frames.")
                nm = normals[..., :3].float()
                if nm.shape[1:3] != (h, w):
                    nm = F.interpolate(nm.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1)
                nm = to_opencv_normals(nm, normals_convention)
            else:
                nm = None
            for sj, m in per_subject.items():   # subject j: the j-th color (= Picture j)
                selected = m > 0.5
                if mark_style in ("translucent overlay", PROTECTED_NORMAL_STYLE, HATCHING_STYLE):
                    hm = selected
                elif mark_style == "colored outline":
                    # Draw inside the selected person only. Keep the scene, held props and
                    # occluding objects visible instead of painting over the whole mask.
                    radius = max(1, round(min(h, w) * 0.005))
                    eroded = 1 - F.max_pool2d((~selected)[:, None].float(), 2 * radius + 1, 1, radius)[:, 0]
                    hm = selected & (eroded < 0.5)
                else:
                    hm = grow_masks(selected.float(), float(hybrid_grow)) > 0.5
                col = HYBRID_COLORS[(int(sj) - 1) % len(HYBRID_COLORS)]
                if mark_style == HATCHING_STYLE:
                    first = next((mask for mask in selected if bool(mask.any())), None)
                    ys, xs = torch.where(first) if first is not None else ([], [])
                    hatch_scale = max(64, min(int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))) if len(xs) else 64
                for i in range(n):
                    if bool(hm[i].any()):
                        fill = mark_fill(mark_style, image[i, ..., :3].float(), nm[i] if nm is not None else None, hm[i], col)
                        if mark_style == HATCHING_STYLE:
                            target = hm[i] if protected is None else hm[i] & ~protected[i]
                            alpha = hatching_alpha(target, hatch_scale, hatch_width, hatch_spacing)[..., None].to(base.dtype)
                            base[i] = base[i] * (1 - alpha) + fill.to(base.dtype) * alpha
                        elif mark_style == PROTECTED_NORMAL_STYLE:
                            import cv2
                            import numpy as np
                            target = hm[i]
                            if protected is not None:
                                keep = cv2.dilate(protected[i].numpy().astype(np.uint8),
                                                  np.ones((5, 5), np.uint8)) > 0
                                target = target & ~torch.from_numpy(keep)
                            if bool(target.any()):
                                feather_px = round(min(h, w) * max(0.0, float(edge_feather)))
                                alpha = feather_inside_mask(target, feather_px).to(base.dtype)
                                alpha = alpha[..., None]
                                base[i] = base[i] * (1 - alpha) + fill.to(base.dtype) * alpha
                        else:
                            base[i][hm[i]] = fill.to(base.dtype)[hm[i]]
        # Apply the same opacity control to the guide sent to H3. Blending against the
        # original hybrid base preserves stacked pickers and fades both fills and DWPose marks.
        opacity = min(1.0, max(0.0, float(mark_opacity)))
        if opacity < 1.0:
            base = base_source + (base - base_source) * opacity
        lost_runs = []   # stretches where a picked subject has no mask after its first appearance: the panel offers them to add to the strip
        for sj in subjects:
            if not any(r["source"] == "picked" for r in report_s[sj]):
                continue
            first_active = min([i for i in range(n) if bool(per_subject[sj][i].any())] or [0])
            run = None
            for i in range(first_active, n):
                bad = not bool(per_subject[sj][i].any())
                if bad and run is None: run = [i, i]
                elif bad: run[1] = i
                elif run is not None: lost_runs.append({"subject": sj, "start": run[0], "end": run[1]}); run = None
            if run is not None: lost_runs.append({"subject": sj, "start": run[0], "end": run[1]})
        ui = {"bfs_vsegpick": [{"frames": panels, "indices": strip, "report": [report[i] for i in strip], "subjects": subjects,
                                "protect_anchors": lock_anchors,
                                "protected_frames": [bool(locked[i].any()) if locked is not None else False for i in strip],
                                "count": n, "lost": lost_runs,
                                "timeline": (st("building the timeline previews", 0.9) or timeline_strip(image, per_subject, locked))}]}
        lost_frames = sum(r["end"] - r["start"] + 1 for r in lost_runs)
        st(f"done: {len(subjects)} subject(s), {lost_frames} frames without a mask", 1.0, log=True)
        return {"ui": ui, "result": (masks, marked, torch.cat(overlays), json.dumps({"frames": report, "anchors": anchors, "subjects": subjects, "active_subjects": [int(sj) for sj in subjects if bool(per_subject[sj].any())], "mark_style": mark_style}), base)}


NODE_CLASS_MAPPINGS = {"BFSVideoSegmentPicker": BFSVideoSegmentPicker}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSVideoSegmentPicker": "BFS Video Segment Picker (SAM 3)"}
