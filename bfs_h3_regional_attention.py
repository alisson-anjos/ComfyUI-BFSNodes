"""BFS H3 Regional Reference Attention: routes each reference picture to its own person inside the DiT attention.

Recommended mode (`guide` connected, routing 'soft', no masks): the guide's thin colored face contours (BFS Video Segment
Picker, 'sparse face contour' mode) seed each person; at `route_layer` the model's own features follow that person through
every latent frame (label propagation). The result is a soft attention bias, not a block:
    token of person j        -> +strength toward Picture j, -strength toward the other pictures and people
    token of nobody          -> -background toward every picture
    Picture j's own rows     -> -strength toward the other pictures (pictures do not mix)
Picture j's keys are its ref_img rows plus its vision tokens inside the text sequence (minimax_token_tags). The bias has
rank 2P and rides in extra head dimensions (q' = [q | qe], k' = [k | ke]): one fused attention call, no S x S mask.
H3 Image (one target frame) is supported: the marked guide is reference image 1 and the pictures follow from image 2.

Hard routing and `subject_map` (mask mode) are kept for comparison: queries are grouped by their allowed pictures and each
group runs the normal attention over its allowed keys.

Pictures are matched to subjects in color order (red, blue, green, yellow, ...), like BFS H3 Face Swap • Auto Subjects:
Picture 1 = red, Picture 2 = blue; a color the guide reading never finds leaves its picture neutral."""
import logging
import math

import torch
import torch.nn.functional as F

from .bfs_video_segment_picker import HYBRID_COLORS

PLAN_KEY = "bfs_h3_regional_plan"


def _latent_spans(latent_t):
    # pixel frames per latent frame: the H3 VAE packs 17 frames into 5 latents (1, 4, 4, 4, 4)
    return [(1, 4, 4, 4, 4)[i % 5] for i in range(latent_t)]


def subject_regions(subject_map):
    """[frames, H, W, 3] color map -> [P, frames, H, W] float masks, one per color present, in color order."""
    cols = torch.tensor(HYBRID_COLORS, dtype=torch.float32) / 255.0
    m = subject_map[..., :3].float()
    dist = (m[..., None, :] - cols).abs().amax(-1)            # [f, H, W, C]
    near = dist.argmin(-1); hit = dist.amin(-1) < 0.1
    regions = [((near == c) & hit).float() for c in range(len(cols))]
    return torch.stack([r for r in regions if bool(r.any())]) if any(bool(r.any()) for r in regions) else None


def token_regions(regions, latent_t, gh, gw, dilate):
    """[P, frames, H, W] -> [P, latent_t * gh * gw] bool: the token is (near) inside the subject in that latent frame."""
    p, n = regions.shape[:2]
    grid = F.interpolate(regions, size=(gh, gw), mode="area")             # [P, frames, gh, gw]
    out, f = [], 0
    for span in _latent_spans(latent_t):
        idx = [min(i, n - 1) for i in range(f, f + span)]; f += span
        out.append(grid[:, idx].amax(1))
    tok = torch.stack(out, 1) > 0.02                                       # [P, latent_t, gh, gw]
    if dilate > 0:
        k = 2 * dilate + 1
        tok = F.max_pool2d(tok.flatten(0, 1).float()[:, None], k, stride=1, padding=dilate)[:, 0].reshape(tok.shape) > 0
    return tok.reshape(p, -1)


def text_picture_spans(tags):
    """Runs of vision tokens (tag 0) in the text sequence, in order: one per picture."""
    t = tags.reshape(-1).tolist(); spans, start = [], None
    for i, v in enumerate(t + [1]):
        if v == 0 and start is None:
            start = i
        elif v != 0 and start is not None:
            spans.append((start, i)); start = None
    return spans, len(t)


TEXT, OTHER, GUIDE, BG, PIC, TV, SUBJ, GSUBJ = range(8)   # row owners; PIC / TV / SUBJ / GSUBJ carry a picture set
# GSUBJ = guide tokens of a person: they read that person's picture (the LoRA binds color -> picture there), never another
LEVELS = ["pictures only", "people", "people + text"]


def _allowed(q, k, level, outside_all):
    """May a query row owned by q = (owner, picture set) read a key row owned by k?"""
    (qo, qs), (ko, ks) = q, k
    pic_key = ko in (PIC, TV)
    if ko == GSUBJ:
        ko, ks = GUIDE, 0                               # as a key a person's guide token is plain guide (the clean source)
    if qo == GSUBJ:
        if pic_key:
            return bool(ks & qs)
        return level == 0 or ko in (TEXT, OTHER, GUIDE)
    if qo == BG and outside_all:
        return True
    if level == 0:
        if not pic_key or qo in (TEXT, OTHER, TV):
            return True
        if qo in (PIC, SUBJ):
            return bool(ks & qs)
        return False                                    # guide and background read no picture
    if qo in (TEXT, OTHER) or (qo == TV and level == 1):
        return level == 1 or ko in (TEXT, OTHER, GUIDE, BG)
    if qo == TV:                                        # people + text: a picture's text tokens see only that picture
        return ko in (TEXT, OTHER, GUIDE, BG) or (pic_key and bool(ks & qs))
    if qo == PIC:
        return ko in (TEXT, OTHER) or (pic_key and bool(ks & qs))
    if qo == SUBJ:
        return ko in (TEXT, OTHER, GUIDE, BG) or ((pic_key or ko == SUBJ) and bool(ks & qs))
    if qo == BG:
        return ko in (TEXT, OTHER, GUIDE, BG)
    return ko in (TEXT, OTHER, GUIDE)                   # guide: the clean source, never reads the generated people


def build_plan(layout, cfg, device, video_bits=None, guide_bits=None):
    """Per query group: (query indices, key indices or None = every key). video_bits: picture set per target row
    (attention-tracked); otherwise the subjects come from cfg["regions"] (subject_map)."""
    segs = layout.segments; S = layout.seq_len
    _, latent_t, lat_h, lat_w, _ = layout.signature
    gh, gw = lat_h // 2, lat_w // 2; frame_rows = gh * gw
    refs = [(a, b) for a, b, kind in segs if kind == "ref_img"]
    P = min(len(refs), cfg["subjects"])
    if P == 0:
        return None
    owner = torch.full((S,), OTHER, dtype=torch.long); pset = torch.zeros(S, dtype=torch.long)
    text_len = 0
    for a, b, kind in segs:
        if kind == "text":
            owner[a:b] = TEXT; text_len = b
    for j in range(P):
        owner[refs[j][0]:refs[j][1]] = PIC; pset[refs[j][0]:refs[j][1]] = 1 << j
    spans = cfg["text_spans"].get(text_len)
    if spans is not None and len(spans) >= P:
        for j in range(P):
            owner[spans[j][0]:spans[j][1]] = TV; pset[spans[j][0]:spans[j][1]] = 1 << j
    elif cfg["text_spans"]:
        logging.warning("[BFS H3 Regional Attention] text vision spans not found for this conditioning; masking the DiT picture rows only")
    for a, b, kind in segs:
        if kind not in ("video", "cond") or (b - a) % frame_rows:
            continue
        if kind == "cond":
            if guide_bits is not None:
                gb = guide_bits.cpu()[:b - a] & ((1 << P) - 1)
            elif video_bits is None:
                tok = token_regions(cfg["regions"][:P], (b - a) // frame_rows, gh, gw, cfg["dilate"])
                gb = sum(tok[j].long() << j for j in range(P))
            else:
                gb = torch.zeros(b - a, dtype=torch.long)
            owner[a:b] = torch.where(gb > 0, GSUBJ, GUIDE); pset[a:b] = gb
            continue
        if video_bits is not None:
            rb = video_bits.cpu() & ((1 << P) - 1)
        else:
            tok = token_regions(cfg["regions"][:P], (b - a) // frame_rows, gh, gw, cfg["dilate"])
            rb = torch.zeros(b - a, dtype=torch.long)
            for j in range(P):
                rb |= tok[j].long() << j
        owner[a:b] = torch.where(rb > 0, SUBJ, BG); pset[a:b] = rb
    code = owner * 64 + pset
    kinds = [(c // 64, c % 64) for c in torch.unique(code).tolist()]
    level = LEVELS.index(cfg["isolation"]); outside_all = cfg["outside"] == "every picture"
    plan = []
    for q in kinds:
        q_idx = (code == q[0] * 64 + q[1]).nonzero().flatten()
        ok = [k[0] * 64 + k[1] for k in kinds if _allowed(q, k, level, outside_all)]
        keep = torch.isin(code, torch.tensor(ok))
        plan.append((q_idx.to(device), None if bool(keep.all()) else keep.nonzero().flatten().to(device)))
    return plan


def guide_seeds(guide, n_lat, gh, gw):
    """Contour colors in the guide frames -> (colors present in color order, [P, n_lat * gh * gw] bool seed tokens inside
    each filled contour). A contour is a thin closed line of one subject color around an area: filled blobs of a similar
    color (lights, a green shirt) are not contours (pixels / hull area must stay low)."""
    import cv2
    import numpy as np
    cols = np.array(HYBRID_COLORS, np.float32) / 255.0
    F_, H, W = guide.shape[:3]
    fills = np.zeros((len(cols), F_, gh, gw), np.float32)
    for f in range(F_):
        g = guide[f, ..., :3].float().numpy()
        for c in range(len(cols)):
            line = (np.abs(g - cols[c]).max(-1) < 0.12).astype(np.uint8)
            if line.sum() < 40:
                continue
            n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(line, np.ones((3, 3), np.uint8)), connectivity=8)
            best = None
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < 40:
                    continue
                ys, xs = np.nonzero(lab == i)
                hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
                area = cv2.contourArea(hull)
                if area < 400 or stats[i, cv2.CC_STAT_AREA] / area > 0.55:
                    continue
                if best is None or area > best[0]:
                    best = (area, hull)
            if best is not None:
                m = np.zeros((H, W), np.float32); cv2.fillPoly(m, [best[1]], 1.0)
                fills[c, f] = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA)
    present = [c for c in range(len(cols)) if fills[c].max() > 0.3]
    grid = torch.from_numpy(fills[present])
    out, f = [], 0
    for span in _latent_spans(n_lat):
        idx = [min(i, F_ - 1) for i in range(f, f + span)]; f += span
        out.append(grid[:, idx].amax(1))
    seeds = torch.stack(out, 1) > 0.3                                     # [P, n_lat, gh, gw]
    return present, seeds.reshape(len(present), -1)


def track_owners(v, guide_rows, seeds, dilate, margin, n_lat, gh, gw, sharpness=0.05, radius=3, topk=8, min_sim=0.3, match=False,
                 match_topk=16, match_min=0.4):
    """Owner of every guide token by label propagation on the model's own features (value vectors: content, no rope),
    frame to frame from the marked frames, forward and backward: a token takes the labels of its most similar tokens
    within `radius` in the previous (next) latent frame, weighted by softmax(cosine / sharpness); a token with no
    neighbor above `min_sim` (a camera cut, a new person) takes nothing. In marked frames the contours are certain
    (seeds) and the tokens 2+ away from every contour are certain background. Labels: P people + background.
    Returns hard bits [n_lat * gh * gw] (a person's label above 0.5 + margin) and soft affinities [P, n_lat * gh * gw]."""
    H, N, D = v.shape[1], len(guide_rows), v.shape[-1]
    f = F.normalize(v[0, :, guide_rows].float(), dim=-1).permute(1, 0, 2).reshape(N, H * D) / H ** 0.5   # cos = mean over heads
    f = f.reshape(n_lat, gh, gw, H * D)
    P = seeds.shape[0]; dev = v.device
    sd = seeds.reshape(P, n_lat, gh, gw).to(dev).float()
    anyseed = sd.amax(0)
    near = F.max_pool2d(anyseed[:, None], 5, stride=1, padding=2)[:, 0]
    marked = anyseed.flatten(1).amax(1) > 0
    fixed = torch.zeros(P + 1, n_lat, gh, gw, device=dev)                 # certain labels in the marked frames
    fixed[:P] = sd; fixed[P] = (near == 0).float() * marked[:, None, None].float()
    certain = fixed.sum(0) > 0
    k = 2 * radius + 1
    def step(lab_prev, f_prev, f_cur):
        nb_f = F.unfold(f_prev.permute(2, 0, 1)[None], k, padding=radius)[0].reshape(f.shape[-1], k * k, gh * gw)   # [C, K, HW]
        nb_l = F.unfold(lab_prev[None], k, padding=radius)[0].reshape(P + 1, k * k, gh * gw)                          # [L, K, HW]
        sim = torch.einsum("ckn,nc->kn", nb_f, f_cur.reshape(gh * gw, -1))                          # [K, HW]
        valid = nb_l.sum(0) > 0
        sim = sim.masked_fill(~valid, -1.0)
        top, idx = sim.topk(min(topk, k * k), dim=0)
        w = torch.softmax(top / sharpness, 0) * (top > min_sim)
        lab = (nb_l.gather(1, idx[None].expand(P + 1, -1, -1)) * w[None]).sum(1)                                    # [L, HW]
        return lab.reshape(P + 1, gh, gw)
    passes = []
    for order in (range(n_lat), range(n_lat - 1, -1, -1)):
        lab = torch.zeros(P + 1, n_lat, gh, gw, device=dev); prev = None
        for t in order:
            cur = fixed[:, t].clone() if prev is None else step(lab[:, prev], f[prev], f[t])
            cur = torch.where(certain[t][None], fixed[:, t], cur)
            lab[:, t] = cur; prev = t
        passes.append(lab)
    lab = torch.maximum(passes[0], passes[1])                              # labels reach a frame from either side
    lab = lab / lab.sum(0, keepdim=True).clamp(min=1.0)
    if match:   # fast motion / camera cuts: what the propagation could not reach is matched to every certain token
        lab = lab + (1 - lab.sum(0, keepdim=True)).clamp(min=0) * global_match(f, fixed, certain, sharpness, match_topk, match_min)
        lab = lab / lab.sum(0, keepdim=True).clamp(min=1.0)
    aff = lab[:P].reshape(P, -1)
    bits = torch.zeros(n_lat * gh * gw, dtype=torch.long, device=dev)
    for j in range(P):
        bits |= (aff[j] > 0.5 + margin).long() << j
    if dilate > 0:
        planes = torch.stack([(bits >> j & 1).float() for j in range(P)]).reshape(-1, 1, gh, gw)
        planes = F.max_pool2d(planes, 2 * dilate + 1, stride=1, padding=dilate).reshape(P, -1) > 0
        bits = sum(planes[j].long() << j for j in range(P))
    return bits, aff


def global_match(f, fixed, certain, sharpness, topk, min_sim, chunk=4096):
    """Label of every token from its most similar certain tokens anywhere in the clip (any frame, any position):
    the people AND the background of the marked frames (an unmarked person there is background, so its face is
    matched to the background, not to a picture). [L, n_lat, gh, gw]."""
    L = fixed.shape[0]; C = f.shape[-1]
    ff = f.reshape(-1, C); ci = certain.reshape(-1).nonzero()[:, 0]
    out = torch.zeros(L, ff.shape[0], device=f.device)
    if ci.numel() == 0:
        return out.reshape(fixed.shape)
    fc = ff[ci].half(); lc = fixed.reshape(L, -1)[:, ci]                  # [M, C], [L, M]
    k = min(topk, ci.numel())
    for a in range(0, ff.shape[0], chunk):
        sim = ff[a:a + chunk].half() @ fc.T                                # [n, M]
        top, idx = sim.float().topk(k, dim=1)
        w = torch.softmax(top / sharpness, 1) * (top > min_sim)
        out[:, a:a + chunk] = (lc[:, idx] * w[None]).sum(-1)
    return out.reshape(fixed.shape)


def save_owner_debug(bits, P, n_lat, gh, gw, name):
    import folder_paths, numpy as np, cv2
    cols = np.array(HYBRID_COLORS[:P], np.uint8)
    b = bits.reshape(n_lat, gh, gw).cpu().numpy()
    tiles = []
    for t in range(n_lat):
        img = np.zeros((gh, gw, 3), np.uint8)
        for j in range(P):
            img[(b[t] >> j & 1) > 0] = cols[j]
        tiles.append(cv2.resize(img, (gw * 8, gh * 8), interpolation=cv2.INTER_NEAREST))
    rows = [np.hstack(tiles[i:i + 8] + [np.zeros_like(tiles[0])] * (8 - len(tiles[i:i + 8]))) for i in range(0, len(tiles), 8)]
    path = f"{folder_paths.get_output_directory()}/{name}"
    cv2.imwrite(path, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    return path


def make_override(cfg, previous):
    state = {}   # attention-tracked plans of this patched model, per layout

    def override(func, *args, **kwargs):
        attn = func if previous is None else (lambda *a, **k: previous(func, *a, **k))
        q, k, v, heads = args[:4]
        topts = kwargs.get("transformer_options") or {}
        layout = topts.get("minimax_h3_layout")
        if layout is None or q.ndim != 4 or q.shape[2] != layout.seq_len or not kwargs.get("skip_reshape"):
            return attn(*args, **kwargs)
        key = (layout.signature, layout.seq_len)
        if cfg.get("guide") is not None:
            if key not in state:
                if topts.get("block_index", 0) < cfg["route_layer"]:
                    return attn(*args, **kwargs)                          # first forward, before the tracking layer
                state[key] = route_from_attention(q, k, v, layout, cfg)
            if state[key] is None:
                return attn(*args, **kwargs)
            if cfg.get("from_route") and topts.get("block_index", 0) < cfg["route_layer"]:
                return attn(*args, **kwargs)                              # bias only from the tracking layer on (as trained)
            if "soft" in state[key]:
                return biased_attention(q, k, v, state[key]["soft"], kwargs.get("skip_output_reshape", False))
            plan = state[key]["plan"]
        else:
            plans = topts.setdefault(PLAN_KEY, {})
            if key not in plans:
                plans[key] = build_plan(layout, cfg, q.device)
            plan = plans[key]
        if plan is None:
            return attn(*args, **kwargs)
        out = None
        for q_idx, k_idx in plan:
            qg = q.index_select(2, q_idx)
            kg = k if k_idx is None else k.index_select(2, k_idx)
            vg = v if k_idx is None else v.index_select(2, k_idx)
            og = attn(qg, kg, vg, heads, *args[4:], **kwargs)   # [B, rows, heads * dim], or [B, heads, rows, dim] with skip_output_reshape
            dim = 2 if og.ndim == 4 else 1
            if out is None:
                shape = list(og.shape); shape[dim] = q.shape[2]
                out = og.new_empty(shape)
            out.index_copy_(dim, q_idx, og)
        return out
    return override


def route_from_attention(q, k, v, layout, cfg):
    segs = layout.segments
    _, latent_t, lat_h, lat_w, _ = layout.signature
    gh, gw = lat_h // 2, lat_w // 2; fr = gh * gw
    conds = [(a, b) for a, b, kind in segs if kind == "cond" and (b - a) % fr == 0]
    video = next((a, b) for a, b, kind in segs if kind == "video")
    ref_segs = [(a, b) for a, b, kind in segs if kind == "ref_img"]
    cfg["image_mode"] = False
    if not conds and latent_t == 1 and len(ref_segs) >= 2 and cfg["guide"] is not None:
        return route_image(q, v, layout, cfg, ref_segs, video, gh, gw)   # H3 Image: the guide is the first reference image
    if not conds:
        logging.warning("[BFS H3 Regional Attention] no guide latent in the sequence: attention tracking is off")
        return None
    ga, gb = max(conds, key=lambda s: s[1] - s[0]); n_g = (gb - ga) // fr
    if cfg["regions"] is not None:      # subject_map given too: its masks seed every frame (still a soft bias)
        seeds = token_regions(cfg["regions"], n_g, gh, gw, 0); present = list(range(seeds.shape[0]))
    else:
        present, seeds = guide_seeds(cfg["guide"], n_g, gh, gw)
    if not present:
        logging.warning("[BFS H3 Regional Attention] no contour colors found in the guide: attention tracking is off")
        return None
    n_pics = len([1 for a, b, kind in segs if kind == "ref_img"])
    cfg["missing"] = []
    if cfg["regions"] is None and n_pics and max(present) < n_pics:   # color j = Picture j; a color never found: neutral
        full = torch.zeros(n_pics, seeds.shape[1], dtype=seeds.dtype); full[present] = seeds
        cfg["missing"] = [j for j in range(n_pics) if j not in present]
        if cfg["missing"]:
            logging.warning(f"[BFS H3 Regional Attention] no contour found for picture(s) {[j + 1 for j in cfg['missing']]}: not routed")
        seeds = full; present = list(range(n_pics))
    cfg["subjects"] = len(present)
    bits_g, aff_g = track_owners(v, torch.arange(ga, gb, device=q.device), seeds, cfg["dilate"], cfg["margin"], n_g, gh, gw,
                                 cfg["sharpness"], radius=cfg["radius"], match=cfg["tracking"] == "propagate + match")
    bits_v = torch.zeros(video[1] - video[0], dtype=torch.long, device=q.device)
    n = min(bits_v.numel(), bits_g.numel()); bits_v[:n] = bits_g[:n]
    P = len(present)
    share = [round(float((bits_g >> j & 1).float().mean()) * 100, 1) for j in range(P)]
    path = save_owner_debug(bits_g, P, n_g, gh, gw, "bfs_regional_owners.png")
    logging.info(f"[BFS H3 Regional Attention] tracked {P} subjects at layer {cfg['route_layer']}: % of guide tokens {share}; map: {path}")
    if cfg["routing"] == "soft":
        return {"soft": soft_bias(layout, cfg, aff_g, (ga, gb), video, q.shape[-1], q.device, q.dtype)}
    return {"plan": build_plan(layout, cfg, q.device, video_bits=bits_v, guide_bits=bits_g)}


def _image_grid(n, H, W):
    """(rows, cols) of a reference image's 2x2-patch token grid: n tokens, aspect closest to H:W."""
    best = None
    for r in range(1, n + 1):
        if n % r == 0:
            err = abs(math.log((n // r) / r) - math.log(W / H))
            if best is None or err < best[0]:
                best = (err, r, n // r)
    return best[1], best[2]


def route_image(q, v, layout, cfg, ref_segs, video, gh, gw):
    """H3 Image (one target frame): the marked guide is reference image 1, the pictures are references 2.. in color order
    of the picked people. Seeds from the guide contours are the owners (one frame, nothing to propagate); they are
    resized from the guide's token grid to the target's."""
    if cfg["routing"] != "soft":
        logging.warning("[BFS H3 Regional Attention] H3 Image supports soft routing only: routing is off")
        return None
    (ga, gb), pics = ref_segs[0], ref_segs[1:]
    H, W = cfg["guide"].shape[1:3]
    ggh, ggw = _image_grid(gb - ga, H, W)
    present, seeds = guide_seeds(cfg["guide"][:1], 1, ggh, ggw)
    if not present:
        logging.warning("[BFS H3 Regional Attention] no contour colors found in the guide: attention tracking is off")
        return None
    if len(present) != len(pics):
        logging.warning(f"[BFS H3 Regional Attention] {len(present)} contour colors but {len(pics)} pictures: "
                        "routing the first colors to the first pictures in order")
    P = min(len(present), len(pics))
    seeds = seeds[:P]
    cfg.update(image_mode=True, missing=[], subjects=P)
    _, aff_g = track_owners(v, torch.arange(ga, gb, device=q.device), seeds, 0, cfg["margin"], 1, ggh, ggw,
                            cfg["sharpness"], radius=cfg["radius"])
    aff_v = F.interpolate(aff_g.reshape(P, 1, ggh, ggw).float(), size=(gh, gw), mode="bilinear", align_corners=False)
    aff_v = aff_v.reshape(P, -1).clamp(0, 1)
    share = [round(float((aff_v[j] > 0.5).float().mean()) * 100, 1) for j in range(P)]
    logging.info(f"[BFS H3 Regional Attention] H3 Image: {P} subjects from the guide contours, % of target tokens {share}")
    return {"soft": soft_bias(layout, cfg, aff_g, (ga, gb), video, q.shape[-1], q.device, q.dtype, aff_v=aff_v, pics=pics)}


def soft_bias(layout, cfg, aff_g, guide, video, head_dim, device, dtype, aff_v=None, pics=None):
    """Extra query / key columns whose dot product is the attention bias (logits): exact, rank 2P, no S x S mask.
        picture columns: target token -> Picture j gets strength * (2 a_j - sum a) - background * (1 - sum a): + for
                         its person, - for the other people, a mild - where the token is no tracked person; the guide
                         follows the same rule (the LoRA binds color -> picture on the marked guide tokens); Picture i rows get
                         -strength towards Picture j (pictures do not mix)
        people columns (isolation 'people'): target token -> target tokens of person i gets -strength * (1 - a_i) * a'_i:
                         a person (and its guide tokens) reads little of another generated person"""
    S = layout.seq_len; P = aff_g.shape[0]; s = cfg["strength"] * head_dim ** 0.5     # undo the 1/sqrt(d) of the logits
    segs = layout.segments
    refs = (pics if pics is not None else [(a, b) for a, b, kind in segs if kind == "ref_img"])[:P]
    text_len = next(b for a, b, kind in segs if kind == "text")
    spans = cfg["text_spans"].get(text_len) or []
    people = cfg["isolation"] != "pictures only"
    E = 2 * P if people else P
    qe = torch.zeros(S, E, device=device); ke = torch.zeros(S, E, device=device)
    for j, (a, b) in enumerate(refs):
        ke[a:b, j] = 1.0; qe[a:b, :P] = -s; qe[a:b, j] = 0.0
    if pics is not None:
        spans = spans[1:]                              # H3 Image: the first vision-token run is the guide image
    for j, (a, b) in enumerate(spans[:P]):
        ke[a:b, j] = 1.0
    ga, gb = guide; va, vb = video
    av_src = aff_g if aff_v is None else aff_v
    n = min(vb - va, av_src.shape[1]); a_v = av_src[:, :n].T                       # [n, P]
    tot = a_v.sum(1, keepdim=True).clamp(max=1.0); bgs = cfg["background"] * head_dim ** 0.5
    qe[va:va + n, :P] = s * (2 * a_v - tot) - bgs * (1 - tot)        # uncertain / background tokens: only a mild push away
    qe[va + n:vb, :P] = -bgs
    a_g = aff_g[:, :gb - ga].T; tg = a_g.sum(1, keepdim=True).clamp(max=1.0)
    qe[ga:ga + a_g.shape[0], :P] = s * (2 * a_g - tg) - bgs * (1 - tg)   # guide: same rule (the LoRA binds color -> picture there)
    if people:
        qe[va:va + n, P:] = -s * (1 - a_v); ke[va:va + n, P:] = a_v
        qe[ga:ga + a_g.shape[0], P:] = -s * (1 - a_g)
    for j in cfg.get("missing", []):          # a picture whose color the guide reading never found: no push either way
        if j < P:
            qe[va:vb, j] = 0.0; qe[ga:gb, j] = 0.0
            if people:
                qe[:, P + j] = 0.0; ke[:, P + j] = 0.0
    pad = (-(head_dim + E)) % 8
    return (F.pad(qe, (0, pad)).to(dtype), F.pad(ke, (0, pad)).to(dtype))


def biased_attention(q, k, v, bias, skip_output_reshape):
    """Attention with the rank-E logit bias appended as extra head dims (q . k + qe . ke), scale of the real head dim."""
    qe, ke = bias; B, H, S, D = q.shape
    q2 = torch.cat([q, qe.expand(B, H, S, -1)], -1); k2 = torch.cat([k, ke.expand(B, H, S, -1)], -1)
    v2 = F.pad(v, (0, qe.shape[-1]))
    out = F.scaled_dot_product_attention(q2, k2, v2, scale=D ** -0.5)[..., :D]
    return out if skip_output_reshape else out.transpose(1, 2).reshape(B, S, H * D)


class BFSH3RegionalReferenceAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "model": ("MODEL",),
                    "positive": ("CONDITIONING", {"tooltip": "The conditioning with the pictures (to find each picture's vision tokens in the text)."}),
                    "isolation": (LEVELS, {"default": "people", "tooltip": (
                        "pictures only: each person reads only its picture; people still see each other. "
                        "people: each person is an island: reads its picture, the guide, the background and the text, never another "
                        "person; the background and unpicked people read only the guide, the background and the text; a person's guide "
                        "tokens read only that person's picture. people + text: the text rows inside the DiT also stop reading pictures and people.")}),
                    "outside_subjects": (["no picture", "every picture"], {"default": "no picture",
                                         "tooltip": "What the background and unpicked people may read. 'no picture' = no identity can land outside the picked subjects."}),
                    "dilate": ("INT", {"default": 1, "min": 0, "max": 8, "tooltip": "Grow every subject by N tokens (32 px each)."})},
                "optional": {
                    "guide": ("IMAGE", {"tooltip": "Attention tracking (no masks): the marked guide (Video Segment Picker `hybrid` output in "
                                                   "'Multi Face Swap — Sparse Face Contour' mode). "
                                                   "The contours seed each person; the model's own attention finds that person in every frame."}),
                    "subject_map": ("IMAGE", {"tooltip": "Mask mode instead: BFS Video Segment Picker `subject_map` (each subject's mask in its color)."}),
                    "route_layer": ("INT", {"default": 12, "min": 1, "max": 49, "tooltip": "Attention tracking: DiT layer whose queries/keys find the people (mid layers are the most semantic)."}),
                    "margin": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01, "tooltip": "Attention tracking, hard routing: how clearly a token must match a person over the background and the other people."}),
                    "routing": (["soft", "hard"], {"default": "soft", "tooltip": (
                        "Attention tracking. soft: nothing is blocked, the attention is only biased: a token pays more attention to the "
                        "picture of the person it looks like (the model's own affinity) and less to the others. hard: blocked.")}),
                    "strength": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.5, "tooltip": "Soft routing: logit bias (4 = about 55x more attention to the right picture than to a wrong one)."}),
                    "sharpness": ("FLOAT", {"default": 0.05, "min": 0.005, "max": 1.0, "step": 0.005, "tooltip": "Soft routing: temperature of the person affinity (lower = closer to all-or-nothing)."}),
                    "background": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.5, "tooltip": "Soft routing: logit push away from every picture for tokens that are no tracked person (0 = neutral)."}),
                    "tracking": (["propagate", "propagate + match"], {"default": "propagate", "tooltip": (
                        "propagate: each person is followed frame to frame from the marked frames. propagate + match: what the "
                        "frame-to-frame step cannot reach (fast motion, camera cuts) is matched to the most similar marked tokens of the whole clip.")}),
                    "bias_layers": (["all", "from route layer"], {"default": "from route layer", "tooltip": "Which DiT layers get the bias: all, or only the tracking layer and the ones after it (what an attention-trained LoRA saw)."}),
                    "radius": ("INT", {"default": 3, "min": 1, "max": 12, "tooltip": "Attention tracking: how far (in 32 px tokens) a person may move between two latent frames (4 video frames)."})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "BFS/MiniMax H3"
    DESCRIPTION = ("Routes each reference picture to its own person inside the attention, without masks: the guide's colored "
                   "face contours seed each person, the model's own features follow them through the video, and a soft bias "
                   "makes person j read Picture j (color order) and the background read no picture. Works with H3 video "
                   "and H3 Image face swap. Recommended: routing soft, strength 4, background 4, bias_layers 'from route layer'.")

    def apply(self, model, positive, isolation="people", outside_subjects="no picture", dilate=1, guide=None, subject_map=None,
              route_layer=12, margin=0.05, routing="soft", strength=4.0, sharpness=0.05, background=4.0, tracking="propagate", radius=3, bias_layers="from route layer"):
        if model is None:
            raise ValueError("No model reached BFS H3 Regional Reference Attention. If its model comes from "
                             "BFS H3 Face Swap • Auto Subjects, connect that node's optional `model` input "
                             "(or wire the model here straight from the LoRA loader).")
        if guide is None and subject_map is None:
            raise ValueError("Connect `guide` (attention tracking, no masks) or `subject_map` (masks)")
        text_spans = {}
        for _, d in positive:
            tags = d.get("minimax_token_tags")
            if tags is not None:
                spans, n = text_picture_spans(tags)
                text_spans[n] = spans
        cfg = {"dilate": int(dilate), "outside": outside_subjects, "isolation": isolation, "text_spans": text_spans,
               "route_layer": int(route_layer), "margin": float(margin), "guide": None, "regions": None, "subjects": 0,
               "routing": routing, "strength": float(strength), "sharpness": float(sharpness), "background": float(background),
               "tracking": tracking, "radius": int(radius),
               "from_route": bias_layers == "from route layer"}
        if subject_map is not None:
            regions = subject_regions(subject_map.cpu())
            if regions is None:
                raise ValueError("subject_map has no subject colors: connect the Video Segment Picker `subject_map` output")
            cfg["regions"] = regions; cfg["subjects"] = regions.shape[0]
        if guide is not None:
            cfg["guide"] = guide.cpu(); cfg["subjects"] = cfg["subjects"] or len(HYBRID_COLORS)
        m = model.clone()
        topts = dict(m.model_options.get("transformer_options", {}))
        topts["optimized_attention_override"] = make_override(cfg, topts.get("optimized_attention_override"))
        m.model_options["transformer_options"] = topts
        logging.info(f"[BFS H3 Regional Attention] {('attention tracking from ' + ('subject_map seeds' if subject_map is not None else 'guide contours')) if guide is not None else 'subject masks (hard)'}, routing: {routing}, isolation: {isolation}")
        return (m,)


NODE_CLASS_MAPPINGS = {"BFSH3RegionalReferenceAttention": BFSH3RegionalReferenceAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSH3RegionalReferenceAttention": "BFS H3 Regional Reference Attention"}
