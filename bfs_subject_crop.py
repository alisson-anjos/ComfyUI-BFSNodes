"""Stable subject-crop planning, vendored from drozbay/MaskVidExperiments.

Source: https://github.com/drozbay/MaskVidExperiments (commit d98cc89, v0.2.0)
Copyright (c) drozbay. Licensed GPL-3.0, the same licence this pack carries,
which is what makes vendoring it legal here.

Vendored rather than imported so the head-swap sampler has no runtime
dependency on that pack being installed. Behaviour is unchanged; only the
ComfyUI node classes were dropped (they stay in the original pack, where they
belong) and the planner import was repointed.

Why this code and not a crop of my own: naive per-frame crops around a mask
jitter in position and size, and a video model reads that jitter as camera
motion. These boxes are planned over the whole clip, so they hold still through
mask noise and occlusion and follow only sustained movement.
"""

"""Subject-tracked batch crop/uncrop for video mask workflows. The crop
boxes come from planner.py, which plans position, size, and shape jointly
over the whole clip."""

import logging
import math

import numpy as np
import torch
from scipy import ndimage

import comfy.utils

from . import bfs_crop_planner as planner

CATEGORY = "MaskVidExperiments"


def _clean_mask_stack(masks, value_thresh, min_pixels, min_frames):
    """Binarize a mask batch and drop noise blobs.

    Connected components are labelled in 3D (time + space). A blob survives if
    its peak single-frame area clears min_pixels and it is either subject-sized
    (peak >= 50% of the largest blob, so fast motion that splits the subject
    into per-frame components doesn't delete it) or persists min_frames frames.
    """
    binary = masks.detach().cpu().numpy() > value_thresh
    if not binary.any():
        return binary

    labels, count = ndimage.label(binary, structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        return binary

    span = np.zeros(count + 1, dtype=np.int64)
    for label, location in enumerate(ndimage.find_objects(labels), start=1):
        if location is not None:
            span[label] = location[0].stop - location[0].start

    peak = np.zeros(count + 1, dtype=np.int64)
    for frame in labels:
        peak = np.maximum(peak, np.bincount(frame.ravel(), minlength=count + 1))

    span[0] = 0
    peak[0] = 0
    subject = peak >= 0.5 * peak.max() if peak.max() > 0 else np.zeros_like(peak, dtype=bool)
    keep = (peak >= min_pixels) & (subject | (span >= min_frames))
    keep[0] = False
    return keep[labels]


def _open_reconstruct(masks, value_thresh, radius, min_frames):
    """Morphological opening by reconstruction on a mask batch.

    Erodes each frame spatially so blobs thinner than ~2*radius vanish, then
    instead of dilating back, keeps the original shape of every blob with a
    surviving core, so the subject's silhouette (thin protrusions included) is
    untouched. Blobs are 3D components (time + space): a core surviving in any
    frame keeps the whole track. border_value=1 so subjects cut off by the
    frame edge are not eroded from that side.
    """
    binary = masks.detach().cpu().numpy() > value_thresh
    if not binary.any():
        return binary

    spatial = np.zeros((3, 3, 3), dtype=bool)
    spatial[1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    eroded = ndimage.binary_erosion(binary, structure=spatial, iterations=radius, border_value=1)

    labels, count = ndimage.label(binary, structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        return binary

    keep = np.zeros(count + 1, dtype=bool)
    keep[np.unique(labels[eroded])] = True
    if min_frames > 1:
        for label, location in enumerate(ndimage.find_objects(labels), start=1):
            if location is not None and location[0].stop - location[0].start < min_frames:
                keep[label] = False
    keep[0] = False
    return keep[labels]


def _resize_image(img_hwc, w, h, method="lanczos"):
    if img_hwc.shape[0] == h and img_hwc.shape[1] == w:
        return img_hwc
    return comfy.utils.common_upscale(
        img_hwc.movedim(-1, 0).unsqueeze(0), w, h, method, "disabled"
    ).squeeze(0).movedim(0, -1)


def _resize_mask(mask_hw, w, h, method="bilinear"):
    if mask_hw.shape[0] == h and mask_hw.shape[1] == w:
        return mask_hw
    return comfy.utils.common_upscale(
        mask_hw[None, None], w, h, method, "disabled"
    )[0, 0]


def _upscale_size(w, h, megapixels, divisible_by):
    """Size scaling a w x h crop to about abs(megapixels) pixels on the
    divisible_by grid. A positive value is a floor: None when already at or
    above it. A negative value is a target, reached by downscaling too."""
    scale = math.sqrt(abs(megapixels) * 1024 * 1024 / (w * h))
    if megapixels > 0 and scale <= 1.0:
        return None
    return (max(1, round(w * scale / divisible_by)) * divisible_by,
            max(1, round(h * scale / divisible_by)) * divisible_by)


def _cell_params(mode):
    """Raw planner dials for the standard node's (padding, prefer) cell.
    The advanced node feeds the same dials directly. guaranteed floors the
    full promise against every raw frame (floor window 1): a true
    guarantee, so mask noise must be cleaned upstream. firm floors 70%
    against the sustained tracks; flexible has no floor. In tracked mode
    the crop never rescales, so prefer differentiates through the oversize
    rent instead: stillness holds a bigger box to move less. In zoomed
    mode the rent is never scaled by prefer (cheapening it tips whole
    clips into one constant maximum size, killing zoom-follow); prefer
    differentiates through the resize cost."""
    sel = mode["mode"]
    pad_level = mode.get("padding", "firm")
    prefer = mode.get("prefer", "stillness")
    floor = {"guaranteed": 1.0, "firm": 0.7, "flexible": 0.0}[pad_level]
    still = prefer == "stillness"
    if sel == "tracked":
        oversize = 32.0 if still else 8.0
    else:
        oversize = float(mode.get("pad_surplus_tol", 16))
    return {
        "crop_scale": mode["crop_scale"],
        "min_padding_allowed": floor,
        "min_padding_allowed_window": 1 if pad_level == "guaranteed" else 16,
        "pad_deficit_tol": 16.0,
        "pad_surplus_tol": oversize,
        "resize_cost": 2.0 if still else 1.0,
        "movement_cost": 1.0,
        "center_pull": 1e-4,
        "end_tightening": 0.0,
        "end_tightening_window": 80 if floor > 0 else 0,
        "zoom_step": mode.get("zoom_step", 1.0),
        "max_zoom_rate": 0.0,
        "aspect_ratio": mode["aspect_ratio"],
        "seamless_loop": mode.get("seamless_loop", False),
    }


def _debug_summary(binary, boxes, scale, sel, info, output_size):
    """Brief per-run report: what shape the planner chose and how well the
    plan delivered the padding promise."""
    n = len(boxes)
    worst = []
    img_h, img_w = binary.shape[1:]
    for i in range(n):
        ys, xs = binary[i].nonzero()
        if len(xs) == 0:
            worst.append(np.nan)
            continue
        b = boxes[i]
        bx1 = b["x"] + b["width"] - 1
        by1 = b["y"] + b["height"] - 1
        wx = (scale - 1) / 2 * (xs.max() - xs.min() + 1)
        wy = (scale - 1) / 2 * (ys.max() - ys.min() + 1)
        fr = []
        if b["x"] > 0: fr.append((xs.min() - b["x"]) / max(wx, 1e-6))
        if bx1 < img_w - 1: fr.append((bx1 - xs.max()) / max(wx, 1e-6))
        if b["y"] > 0: fr.append((ys.min() - b["y"]) / max(wy, 1e-6))
        if by1 < img_h - 1: fr.append((by1 - ys.max()) / max(wy, 1e-6))
        worst.append(min(fr) if fr else np.nan)
    worst = np.array(worst)
    v = worst[~np.isnan(worst)]
    cx = np.array([b["x"] + b["width"] / 2 for b in boxes])
    cy = np.array([b["y"] + b["height"] / 2 for b in boxes])
    travel = np.hypot(np.diff(cx), np.diff(cy)).sum()

    ar = info["aspect"]
    if info.get("swept"):
        shape = f"{ar:.2f} (chosen from {info['candidates']} candidates)"
    elif sel == "zoomed":
        shape = f"{ar:.2f} (manual)"
    else:
        shape = f"{ar:.2f}"
    ws = [b["width"] for b in boxes]
    hs = [b["height"] for b in boxes]
    lines = [
        f"mode: {sel}",
        f"frames: {n} ({img_w}x{img_h} source)",
        f"aspect_ratio: {shape}",
        f"smallest_box: {min(ws)}x{min(hs)}",
        f"largest_box: {max(ws)}x{max(hs)}",
    ]
    if sel == "zoomed":
        lines.append(f"zoom_ratio: {max(hs) / max(min(hs), 1):.2f} "
                     "(largest box / smallest box)")
    lines.append(f"crop region resizing: {max(ws)} x {max(hs)} -> "
                 f"{output_size[0]} x {output_size[1]}")
    lines.append(f"movement: {travel:.0f}px "
                 "(total distance the box center travels)")
    if len(v) and scale > 1.0:
        lines.append(f"padding_promised: {(scale - 1) / 2:.0%} of the "
                     f"subject's size on each side (crop_scale {scale:.2f})")
        lines.append(f"padding_worst: {max(v.min(), 0):.0%} of promised")
        lines.append(f"padding_typical: {max(np.median(v), 0):.0%} of promised")
    return "\n".join(lines)


def _plan_and_crop(original_images, masks, sel, p, divisible_by,
                   mask_threshold, upscale_megapixels=0.0):
    if original_images.shape[0] != masks.shape[0]:
        raise ValueError(f"original_images ({original_images.shape[0]}) and masks ({masks.shape[0]}) must have the same frame count")
    if original_images.shape[1:3] != masks.shape[1:3]:
        raise ValueError(f"original_images ({original_images.shape[2]}x{original_images.shape[1]}) and masks ({masks.shape[2]}x{masks.shape[1]}) must have the same dimensions")

    if 0.0 < p["crop_scale"] < 1.0:
        raise ValueError("crop_scale must be 0 (full frame) or at least 1.0")
    full_frame = p["crop_scale"] <= 0.0

    binary = masks.detach().cpu().numpy() > mask_threshold
    if full_frame:
        img_h, img_w = masks.shape[1], masks.shape[2]
        boxes = [{"x": 0, "y": 0, "width": img_w, "height": img_h}
                 for _ in range(masks.shape[0])]
        info = {"aspect": img_w / img_h}
    else:
        if not binary.any():
            raise ValueError("all masks are empty, nothing to crop")
        boxes, info = planner.plan(binary, sel, p, divisible_by)

    bw = max(b["width"] for b in boxes)
    bh = max(b["height"] for b in boxes)
    size = None
    img_method, mask_method = "lanczos", "bilinear"
    if full_frame:
        gw = math.ceil(bw / divisible_by) * divisible_by
        gh = math.ceil(bh / divisible_by) * divisible_by
        if (gw, gh) != (bw, bh):
            size = (gw, gh)
    elif sel == "zoomed":
        # Target sized to the largest planned box: that stretch is ~1:1
        # and everything else upscales rather than losing detail.
        th = math.ceil(bh / divisible_by) * divisible_by
        size = (math.ceil(th * info["aspect"] / divisible_by) * divisible_by, th)
    if upscale_megapixels != 0:
        up = _upscale_size(*(size or (bw, bh)), upscale_megapixels, divisible_by)
        if up is not None:
            size = up
            img_method, mask_method = "bicubic", "nearest-exact"

    debug = _debug_summary(binary, boxes, p["crop_scale"],
                           f"{sel} (full frame)" if full_frame else sel,
                           info, size or (bw, bh))

    if size is not None:
        tw, th = size
        cropped_images = torch.stack([
            _resize_image(original_images[i, b["y"]:b["y"] + b["height"], b["x"]:b["x"] + b["width"], :], tw, th, img_method)
            for i, b in enumerate(boxes)
        ])
        cropped_masks = torch.stack([
            _resize_mask(masks[i, b["y"]:b["y"] + b["height"], b["x"]:b["x"] + b["width"]], tw, th, mask_method)
            for i, b in enumerate(boxes)
        ])
    else:
        cropped_images = torch.stack([
            original_images[i, b["y"]:b["y"] + b["height"], b["x"]:b["x"] + b["width"], :]
            for i, b in enumerate(boxes)
        ])
        cropped_masks = torch.stack([
            masks[i, b["y"]:b["y"] + b["height"], b["x"]:b["x"] + b["width"]]
            for i, b in enumerate(boxes)
        ])
    return (cropped_images, cropped_masks,
                         [[b] for b in boxes], debug)
