"""Screenshot persistence and cheap resource snapshots for the Planner chat."""

import base64
import io
from pathlib import Path
import time
import uuid
import urllib.parse


def save_attachments(directory, attachments):
    if not isinstance(attachments, list) or len(attachments) > 4:
        raise ValueError("Attach up to four screenshots per message")
    from PIL import Image, ImageOps

    prepared = []
    for item in attachments:
        if not isinstance(item, dict):
            raise ValueError("Invalid screenshot attachment")
        src = item.get("src", "")
        if (
            not isinstance(src, str)
            or not src.startswith(
                (
                    "data:image/png;base64,",
                    "data:image/jpeg;base64,",
                    "data:image/webp;base64,",
                )
            )
            or len(src) > 3_000_000
        ):
            raise ValueError("Screenshots must be PNG/JPEG/WebP, at most 2 MB each")
        try:
            raw = base64.b64decode(src.split(",", 1)[1], validate=True)
            if len(raw) > 2_000_000:
                raise ValueError("Screenshot exceeds 2 MB")
            with Image.open(io.BytesIO(raw)) as im:
                if im.width * im.height > 24_000_000:
                    raise ValueError("Screenshot dimensions are too large")
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((1280, 1280))
                encoded = io.BytesIO()
                im.save(encoded, format="JPEG", quality=86)
                prepared.append(
                    (
                        str(item.get("name", "screenshot"))[:120],
                        encoded.getvalue(),
                        im.size,
                    )
                )
        except (OSError, SyntaxError) as exc:
            raise ValueError("Screenshot could not be decoded") from exc
    folder = Path(directory)
    result = []
    for name, raw, (w, h) in prepared:
        folder.mkdir(parents=True, exist_ok=True)
        filename = uuid.uuid4().hex + ".jpg"
        (folder / filename).write_bytes(raw)
        result.append(
            {
                "name": name,
                "filename": filename,
                "width": w,
                "height": h,
                "bytes": len(raw),
            }
        )
    return result


def attachment_url(session_id, filename):
    return "/view?" + urllib.parse.urlencode(
        {
            "filename": filename,
            "subfolder": f"bfs_agent/{session_id}/feedback",
            "type": "output",
        }
    )


def resource_snapshot(output_directory):
    report = {"time": time.time(), "gpu": None, "ram": None, "disk": None}
    try:
        import psutil

        memory = psutil.virtual_memory()
        report["ram"] = {
            "used_gib": round(memory.used / 1024**3, 2),
            "total_gib": round(memory.total / 1024**3, 2),
            "percent": memory.percent,
            "process_gib": round(psutil.Process().memory_info().rss / 1024**3, 2),
        }
        disk = psutil.disk_usage(str(output_directory))
        report["disk"] = {
            "free_gib": round(disk.free / 1024**3, 1),
            "percent": disk.percent,
        }
    except (ImportError, OSError):
        pass
    try:
        import torch

        if torch.cuda.is_initialized():
            device = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(device)
            report["gpu"] = {
                "name": torch.cuda.get_device_name(device),
                "free_gib": round(free / 1024**3, 2),
                "total_gib": round(total / 1024**3, 2),
                "used_gib": round((total - free) / 1024**3, 2),
            }
    except (ImportError, RuntimeError):
        pass
    return report
