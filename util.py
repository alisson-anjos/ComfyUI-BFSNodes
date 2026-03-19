import math
import torch
import numpy as np
from PIL import Image


def tensor_to_pil(img_tensor):
    """
    Converts a ComfyUI IMAGE tensor [H, W, C] float32 in range 0..1
    into a PIL image.
    """
    img = img_tensor.cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def pil_to_tensor(img_pil):
    """
    Converts a PIL image into a ComfyUI IMAGE tensor [H, W, C] float32 0..1.
    """
    arr = np.array(img_pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def fit_inside(src_w, src_h, max_w, max_h):
    """
    Resizes while preserving aspect ratio so the source fits entirely inside
    the target box, without cropping.
    """
    if src_w <= 0 or src_h <= 0:
        return 1, 1

    scale = min(max_w / src_w, max_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    return new_w, new_h


def aligned_offset(container_size, content_size, align):
    """
    Returns the position offset for start / center / end alignment.
    """
    if align == "start":
        return 0
    elif align == "end":
        return max(0, container_size - content_size)
    return max(0, (container_size - content_size) // 2)


def paste_with_alpha(dst, src_rgba, xy):
    """
    Pastes an image onto another, preserving alpha if present.
    """
    if src_rgba.mode == "RGBA":
        dst.paste(src_rgba, xy, src_rgba.split()[-1])
    else:
        dst.paste(src_rgba, xy)


def add_white_padding(img_rgba, pad_px=16):
    new_w = img_rgba.width + pad_px * 2
    new_h = img_rgba.height + pad_px * 2

    canvas = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 255))
    canvas.paste(img_rgba, (pad_px, pad_px), img_rgba if img_rgba.mode == "RGBA" else None)
    return canvas
