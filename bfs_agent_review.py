"""CPU-only frame selection and text context for paged visual review."""

import math


def review_rate(value):
    if isinstance(value, bool):
        raise ValueError("Review fps must be 0..24 (0 reviews every frame)")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Review fps must be 0..24") from exc
    if not math.isfinite(value) or not 0 <= value <= 24:
        raise ValueError("Review fps must be 0..24 (0 reviews every frame)")
    return value


def review_grid(length, fps, rate):
    if length < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid shot length/fps")
    rate = review_rate(rate)
    count = (
        length
        if rate == 0
        else min(length, max(2, math.ceil((length - 1) * rate / fps) + 1))
    )
    return sorted({round(i * (length - 1) / max(1, count - 1)) for i in range(count)})


def select_frames(length, required, seen, args):
    indices = args.get("frame_indices")
    if indices is not None:
        if not isinstance(indices, list) or not 1 <= len(indices) <= 6:
            raise ValueError("Request 1..6 shot-relative frame_indices per review page")
        if any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < length
            for i in indices
        ):
            raise ValueError(f"frame_indices must be integers within 0..{length - 1}")
        return sorted(set(indices))
    if "start_frame" in args or "end_frame" in args:
        start, end = args.get("start_frame", 0), args.get("end_frame", length - 1)
        count = args.get("count", 6)
        if (
            any(
                isinstance(x, bool) or not isinstance(x, int)
                for x in (start, end, count)
            )
            or not 0 <= start <= end < length
            or not 1 <= count <= 6
        ):
            raise ValueError(
                "Use an inclusive shot-relative frame range and count 1..6"
            )
        return sorted(
            {round(start + i * (end - start) / max(1, count - 1)) for i in range(count)}
        )
    return ([i for i in required if i not in seen] or required)[:6]


def review_region(value):
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            for x in value
        )
    ):
        raise ValueError("region must be normalized [left, top, right, bottom]")
    if not 0 <= value[0] < value[2] <= 1 or not 0 <= value[1] < value[3] <= 1:
        raise ValueError("region must be inside the frame with positive area")
    return value


def crop_review(image, region):
    if region is None:
        return image
    h, w = image.shape[-3:-1]
    x0, y0 = int(region[0] * w), int(region[1] * h)
    x1, y1 = (
        max(x0 + 1, math.ceil(region[2] * w)),
        max(y0 + 1, math.ceil(region[3] * h)),
    )
    return image[..., y0:y1, x0:x1, :]


def text_context(value):
    """Keep frame labels/coverage in the prompt, not base64 image payloads."""
    if isinstance(value, dict):
        return {key: text_context(item) for key, item in value.items()}
    if isinstance(value, list):
        return [text_context(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return "[image supplied separately]"
    return value
