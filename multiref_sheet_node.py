"""Multi-Ref Sheet Builder — combine 1-5 reference images into the single
composite reference sheet used by the multi-ref LTX-2 LoRA (source_id=2,
layout=overlap). Same grid convention as the training pipeline's
build_multiref_sheet.py: fixed 512x512 panels, deterministic grid by count
(1x1, 2x1, 3x1, 2x2, 3-top+2-bottom), centered with padding onto a fixed
1536x1024 canvas so every sheet is the same resolution regardless of how
many refs are plugged in.

Panel order = image index order (ref_image_1 -> image0, ref_image_2 ->
image1, ...), matching the training data's image0/image1/... convention.
"""
import torch
from PIL import Image

from .util import tensor_to_pil, pil_to_tensor

CATEGORY = "BFS/multiref"

PANEL_SIZE = 512
CANVAS_W, CANVAS_H = 1536, 1024
BG_COLOR = (255, 255, 255)

# row layout per ref count: list of ints = panels per row, top to bottom.
LAYOUTS = {
    1: [1],
    2: [2],
    3: [3],
    4: [2, 2],
    5: [3, 2],
}


def _cover_resize_crop(img: Image.Image, size: int) -> Image.Image:
    """Resize+center-crop to exactly fill a size x size square (no stretch, crops excess)."""
    img = img.convert("RGB")
    w, h = img.size
    scale = max(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - size) // 2, (nh - size) // 2
    return img.crop((x0, y0, x0 + size, y0 + size))


def _justified_compose(imgs, rows_counts, canvas_w, canvas_h, bg):
    """Row-justified layout (no cropping, no distortion) -- identical algorithm to
    build_multiref_sheet.py's _justified_compose, kept in sync so inference-time
    sheets match training-time sheets exactly. Each row is split into
    `rows_counts[i]` images; every image in a row shares that row's height, and
    its width = row_height * (image's own aspect ratio) -- one uniform scale
    factor per image, so nothing is stretched.

    Each row's natural height fills canvas_w at scale=1. If all rows fit within
    canvas_h, they're drawn at natural size (leftover vertical space -> a single
    top+bottom bar, whole block centered) -- never blown up past canvas_w. Only
    if rows would collectively overflow canvas_h do all rows shrink by one
    shared factor < 1, giving a single shared pair of side bars instead of
    per-panel scattered padding.
    """
    it = iter(imgs)
    rows = []
    for count in rows_counts:
        row_imgs = [next(it) for _ in range(count)]
        aspects = [im.width / im.height for im in row_imgs]
        rows.append((row_imgs, aspects))

    natural_heights = [canvas_w / sum(aspects) for _, aspects in rows]
    scale = min(1.0, canvas_h / sum(natural_heights))

    row_heights = [round(nat_h * scale) for nat_h in natural_heights]
    drift_h = round(sum(natural_heights) * scale) - sum(row_heights)
    if row_heights:
        row_heights[-1] += drift_h

    sheet = Image.new("RGB", (canvas_w, canvas_h), bg)
    y = (canvas_h - sum(row_heights)) // 2
    for (row_imgs, aspects), row_h in zip(rows, row_heights):
        widths = [max(1, round(row_h * a)) for a in aspects]
        row_w = sum(widths)
        x = (canvas_w - row_w) // 2
        for im, w_i in zip(row_imgs, widths):
            resized = im.convert("RGB").resize((max(1, w_i), max(1, row_h)), Image.LANCZOS)
            sheet.paste(resized, (x, y))
            x += w_i
        y += row_h
    return sheet


def compose_sheet(imgs, panel_size=PANEL_SIZE, canvas_w=CANVAS_W, canvas_h=CANVAS_H, bg=BG_COLOR, fit_mode="crop"):
    """fit_mode: 'crop' fills each fixed-size panel completely (crops excess, current
    default, matches the training data); 'fit' uses a row-justified layout that keeps
    every pixel of every reference (no cropping, no distortion) while maximizing
    canvas coverage."""
    n = len(imgs)
    if not 1 <= n <= 5:
        raise ValueError(f"expected 1-5 reference images, got {n}")
    rows = LAYOUTS[n]

    if fit_mode == "fit":
        return _justified_compose(imgs, rows, canvas_w, canvas_h, bg)

    native_w = max(rows) * panel_size
    native_h = len(rows) * panel_size
    native = Image.new("RGB", (native_w, native_h), bg)

    it = iter(imgs)
    for row_idx, count in enumerate(rows):
        row_w = count * panel_size
        x_offset = (native_w - row_w) // 2  # center short rows (e.g. bottom row of a 5-ref sheet)
        y = row_idx * panel_size
        for col in range(count):
            panel = _cover_resize_crop(next(it), panel_size)
            x = x_offset + col * panel_size
            native.paste(panel, (x, y))

    sheet = Image.new("RGB", (canvas_w, canvas_h), bg)
    px = (canvas_w - native_w) // 2
    py = (canvas_h - native_h) // 2
    sheet.paste(native, (px, py))
    return sheet


class MultiRefSheetBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fit_mode": (["crop", "fit"], {
                    "default": "crop",
                    "tooltip": "crop: zoom+center-crop to fill each panel completely (matches training data, "
                               "may cut off edges). fit: scale each reference down to fit entirely inside its "
                               "panel with no cropping (preserves every pixel, aspect ratio never distorted -- "
                               "one uniform scale factor for both axes -- leftover space padded with background).",
                }),
            },
            "optional": {
                "ref_image_1": ("IMAGE", {"tooltip": "image0 in the compositional prompt (anchor)."}),
                "ref_image_2": ("IMAGE", {"tooltip": "image1."}),
                "ref_image_3": ("IMAGE", {"tooltip": "image2."}),
                "ref_image_4": ("IMAGE", {"tooltip": "image3."}),
                "ref_image_5": ("IMAGE", {"tooltip": "image4."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("sheet", "n_refs", "debug")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Combines 1-5 plugged-in reference images into the fixed 1536x1024 composite "
                   "sheet the multi-ref LoRA was trained on. Leave slots empty for fewer refs; "
                   "an empty slot is simply skipped, not padded with blank content.")

    def build(self, fit_mode="crop", ref_image_1=None, ref_image_2=None, ref_image_3=None,
              ref_image_4=None, ref_image_5=None):
        slots = [ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5]
        provided = [s for s in slots if s is not None]
        if not provided:
            raise ValueError("MultiRefSheetBuilder needs at least one ref_image_N input.")

        pil_imgs = [tensor_to_pil(t[0] if t.dim() == 4 else t) for t in provided]
        sheet = compose_sheet(pil_imgs, fit_mode=fit_mode)
        sheet_t = pil_to_tensor(sheet).unsqueeze(0)  # [1,H,W,C]

        dbg = (f"MultiRefSheet | {len(provided)} refs -> {CANVAS_W}x{CANVAS_H} "
               f"({'+'.join(str(r) for r in LAYOUTS[len(provided)])} grid, fit_mode={fit_mode})")
        return (sheet_t, len(provided), dbg)


NODE_CLASS_MAPPINGS = {"BFSMultiRefSheetBuilder": MultiRefSheetBuilder}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSMultiRefSheetBuilder": "Multi-Ref Sheet Builder"}
