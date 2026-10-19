"""Resolve explicitly labelled task media on the ComfyUI server before context loading."""

import copy
import hashlib
import os
from pathlib import Path
import re
import shutil
import uuid

VIDEO = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LABEL = re.compile(
    r"^\s*(guide|guia|ref(?:erence)?|refer[eê]ncia|picture)\s*(\d*)\s*:\s*(.+?)\s*$",
    re.I,
)


def prepare_media(task, plan, input_directory):
    """Only explicit Guide:/Ref N: lines import files; prose and web links are not paths."""
    root = Path(input_directory).resolve()
    selected, refs = {}, {}
    for line in str(task).splitlines():
        match = LABEL.match(line)
        if not match:
            continue
        label, number, raw = match.groups()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'", "`"):
            raw = raw[1:-1]
        path = Path(raw).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        guide = label.lower() in ("guide", "guia")
        if path.suffix.lower() not in (VIDEO if guide else IMAGE):
            raise ValueError(
                "Guide/Ref paths must be local video/image files on the ComfyUI server"
            )
        if not path.is_file():
            raise ValueError(f"Media path does not exist on the ComfyUI server: {path}")
        if guide:
            if number or "guide" in selected:
                raise ValueError("Provide exactly one Guide: path")
            selected["guide"] = path
        else:
            index = int(number or 1)
            if not 1 <= index <= 12 or index in refs:
                raise ValueError("Reference numbers must be unique within 1..12")
            refs[index] = path
    if not selected and not refs:
        return plan, []
    # Validate all references before creating imports or changing a plan.
    from PIL import Image

    for path in refs.values():
        with Image.open(path) as image:
            image.verify()

    def imported(path):
        if path.is_relative_to(root):
            return path.relative_to(root).as_posix()
        stat = path.stat()
        digest = hashlib.sha256(
            f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode()
        ).hexdigest()[:20]
        name = digest + path.suffix.lower()
        folder = root / "bfs_agent_imports"
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / name
        if not dest.exists():
            temp = folder / (name + "." + uuid.uuid4().hex + ".tmp")
            try:
                try:
                    temp.symlink_to(path)
                except OSError:
                    shutil.copy2(path, temp)
                os.replace(temp, dest)
            finally:
                if temp.is_symlink() or temp.exists():
                    temp.unlink()
        return dest.relative_to(root).as_posix()

    out, events = copy.deepcopy(plan), []
    if "guide" in selected:
        name = imported(selected["guide"])
        if name != out.get("video"):
            # New source, new timing/identity metadata; keep generation configuration.
            out.update(
                video=name,
                bounds=[],
                segs=[],
                cast={},
                cast_split=False,
                cast_only=False,
                auto_prompts={},
                start_f=0,
                end_f=0,
                start_s=0,
                end_s=0,
                max_total_s=0,
                mode="fixed",
                max_s=min(15.0, max(0.5, float(out.get("max_s") or 15))),
                min_s=min(0.5, float(out.get("min_s") or 0.5)),
            )
        events.append(
            {"role": "guide", "path": str(selected["guide"]), "input_name": name}
        )
    names = list(out.get("global_refs") or [])
    for index, path in sorted(refs.items()):
        while len(names) < index:
            names.append("")
        names[index - 1] = imported(path)
        events.append(
            {
                "role": f"Picture {index}",
                "path": str(path),
                "input_name": names[index - 1],
            }
        )
    if refs:
        out["global_refs"] = names
        out["global_ref"], out["global_ref2"] = (names + ["", ""])[:2]
        # Explicit task references supersede previous per-shot reference overrides.
        for seg in out.get("segs", []):
            for key in ("refs", "ref", "ref2"):
                seg.pop(key, None)
    return out, events
