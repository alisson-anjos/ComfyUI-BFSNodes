"""Planner tool adapters, local VLM, public web context and HTTP session lifecycle."""

from __future__ import annotations
import base64
import copy
import io
import ipaddress
import json
from pathlib import Path
import shutil
import socket
import threading
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from .bfs_planner_agent import Session, SYSTEM, TOOLS, bounded_path

SESSIONS = {}
GPU_LOCK = threading.RLock()


class PageText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.links = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.hidden += 1
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def check_public_url(url):
    u = urllib.parse.urlsplit(url)
    if (
        u.scheme != "https"
        or not u.hostname
        or u.username
        or u.password
        or u.port not in (None, 443)
    ):
        raise ValueError("Only public HTTPS pages on port 443 are supported")
    addresses = socket.getaddrinfo(u.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(
        not ipaddress.ip_address(x[4][0]).is_global for x in addresses
    ):
        raise ValueError("Private and local network URLs are not web context")
    return url


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_page(url):
    check_public_url(url)
    opener = urllib.request.build_opener(PublicRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": "BFSPlannerAgent/0.1"})
    with opener.open(req, timeout=15) as r:
        kind = r.headers.get_content_type()
        if kind not in ("text/html", "text/plain", "application/json", "text/markdown"):
            raise ValueError("This tool reads text pages, not binary downloads")
        data = r.read(524289)
        if len(data) > 524288:
            data = data[:524288]
        raw = data.decode(r.headers.get_content_charset() or "utf-8", errors="replace")
    p = PageText()
    p.feed(raw)
    return {
        "url": url,
        "untrusted": True,
        "text": ("\n".join(p.parts) if kind == "text/html" else raw)[:18000],
        "links": p.links[:40],
    }


def image_tensor(data):
    import numpy as np
    import torch
    from PIL import Image

    if isinstance(data, str):
        data = Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1]))).convert(
            "RGB"
        )
    if isinstance(data, Image.Image):
        data = np.array(data.convert("RGB"))
    x = torch.as_tensor(data).detach().cpu()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x.float() / 255 if x.dtype == torch.uint8 else x.float()


def image_url(data):
    import numpy as np
    from PIL import Image

    if isinstance(data, str):
        return data
    x = image_tensor(data)[0].clamp(0, 1).numpy()
    im = Image.fromarray((x * 255).astype(np.uint8))
    im.thumbnail((640, 640))
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def visual_batch(images):
    import torch

    out = []
    for im in images[:24]:
        x = image_tensor(im)[:1].movedim(-1, 1)
        scale = 512 / max(x.shape[-2:])
        h, w = [max(1, round(v * scale)) for v in x.shape[-2:]]
        x = torch.nn.functional.interpolate(
            x, size=(h, w), mode="bilinear", align_corners=False
        )
        out.append(
            torch.nn.functional.pad(
                x,
                (
                    (512 - w) // 2,
                    512 - w - (512 - w) // 2,
                    (512 - h) // 2,
                    512 - h - (512 - h) // 2,
                ),
                value=0.5,
            )
        )
    return torch.cat(out).movedim(1, -1) if out else None


class Runtime:
    def __init__(self, planner):
        self.p = planner

    def folder(self):
        return Path(self.p.folder_paths.get_output_directory()) / "bfs_agent"

    def get(self, sid):
        if sid not in SESSIONS:
            if (
                not isinstance(sid, str)
                or len(sid) != 32
                or any(c not in "0123456789abcdef" for c in sid)
            ):
                raise ValueError("Invalid session id")
            f = self.folder() / (sid + ".json")
            d = json.loads(f.read_text())
            s = Session(d["task"], d["plan"], d["settings"], sid)
            s.events = d.get("events", [])
            s.renders = d.get("renders", 0)
            s.workflow = d.get("workflow")
            s.planner_id = d.get("planner_id")
            s.client_id = d.get("client_id")
            s.context_ready = False
            s.render_snapshot = d.get("render_snapshot")
            s.checklist = d.get("checklist", [])
            self.attach(s)
            s.revision = d.get("revision", 0)
            s.rendered_revision = d.get("rendered_revision", -1)
            s.reviewed = set(d.get("reviewed", []))
            s.combined = d.get("combined")
            s.preview = d.get("preview")
            s.vlm_info = d.get("vlm_info")
            s.observation = {
                "message": "Session restored. Inspect current plan and results before continuing."
            }
            s.pending_render = d.get("pending_render")
            s.status = (
                d.get("status")
                if d.get("status") in ("complete", "stopped", "budget_exhausted")
                else "paused"
            )
            SESSIONS[sid] = s
        return SESSIONS[sid]

    def attach(self, s):
        s.notify = lambda event: self.p._notify(
            "bfs-agent-event", {"id": s.id, "event": event}
        )

    def save(self, s):
        s.persist(self.folder())

    def context(self, s):
        p = self.p
        path = p._input_path(s.plan["video"])
        a = p.analyze(path, float(s.plan["fps"]))
        return (
            path,
            a,
            p.apply_filters(s.plan, a, path, p.resolve_plan(s.plan, a, path)),
        )

    def shot(self, s, args):
        path, a, segs = self.context(s)
        i = args.get("shot")
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(segs):
            raise ValueError(f"shot must be an integer from 0 to {len(segs) - 1}")
        return path, a, segs, i, segs[i]

    def invalidate(self, s):
        s.reviewed.clear()
        s.combined = None
        s.revision += 1

    def update_meta(self, s, i, changes):
        meta = s.plan.setdefault("segs", [])
        while len(meta) <= i:
            meta.append({})
        meta[i] = {**meta[i], **changes}
        self.invalidate(s)

    def source_images(self, s, args, include_refs=True):
        import numpy as np

        path, a, segs, i, seg = self.shot(s, args)
        picks = np.linspace(seg["start"], seg["end"] - 1, 4).astype(int)
        src = self.p._timeline(a["n_src"], a["fps_src"], float(a["fps"]))
        frames = self.p._read_frames(
            path,
            src[np.clip(picks, 0, a["n"] - 1)],
            (
                max(2, round(a["width"] * min(1, 512 / max(a["width"], a["height"])))),
                max(2, round(a["height"] * min(1, 512 / max(a["width"], a["height"])))),
            ),
        )
        images = [image_tensor(f) for f in frames]
        labels = [f"source frame {n}, {n / a['fps']:.3f}s" for n in picks]
        if include_refs:
            names = self.p.shot_ref_names(seg["refs"], s.plan["global_refs"])
            for n in names[: self.p.MAX_REFS]:
                images.append(self.p._load_image(n))
                labels.append("reference " + n)
        return labels, images

    def diagnose_vlm(self, s):
        from .bfs_agent_diagnostics import inspect_vlm

        try:
            link = (s.workflow or {}).get(s.planner_id, {}).get("inputs", {}).get("vlm")
            loader = (
                (s.workflow or {}).get(str(link[0]), {})
                if isinstance(link, list) and link
                else {}
            )
            checkpoint = loader.get("inputs", {}).get("clip_name")
            s.vlm_info = inspect_vlm(self.p._VLM.get("clip"), s.settings, checkpoint)
            s.event(
                "vlm_info",
                summary="Connected VLM diagnostics available",
                report=s.vlm_info,
            )
        except Exception as exc:
            s.vlm_info = {
                "warnings": ["VLM diagnostics unavailable: " + type(exc).__name__]
            }

    def decide(self, s):
        p = self.p
        clip = p._VLM.get("clip")
        if clip is None:
            raise ValueError(
                "Planner VLM context is missing; restart the Agent context initialization."
            )
        # Reasoning is handled internally by TextGenerate. Only final structured actions are persisted.
        from comfy_extras.nodes_textgen import TextGenerate

        history = [
            {k: v for k, v in e.items() if k not in ("time",)} for e in s.events[-12:]
        ]
        prompt = (
            SYSTEM
            + "\nTOOLS:\n"
            + json.dumps(TOOLS)
            + "\nTASK:\n"
            + s.task
            + "\nCURRENT PLAN:\n"
            + json.dumps(s.plan)[:24000]
        )
        prompt += "\nHISTORY:\n" + json.dumps(history, ensure_ascii=False)[-20000:]
        prompt += (
            "\nLATEST OBSERVATION:\n"
            + json.dumps(s.observation, ensure_ascii=False)[:24000]
        )
        prompt += "\nGENERATION STATE: " + json.dumps(
            {
                "current_revision": s.revision,
                "rendered_revision": s.rendered_revision,
                "current_plan_rendered": s.rendered_revision == s.revision,
            }
        )
        prompt += f"\nBUDGET: renders {s.renders}/{s.settings['max_renders']}. Images are in the order specified by the observation. Return one tool action JSON."
        budget = s.settings["max_tokens"]
        thinking = s.settings["thinking"]
        reserve = min(2048, budget // 4) if thinking else 0
        images = visual_batch(s.images)

        def generate(text, limit, use_thinking):
            return p._textgenerate_execute(
                TextGenerate,
                clip=clip,
                prompt=text,
                max_length=limit,
                sampling_mode={
                    "sampling_mode": "on",
                    "temperature": 0.6 if use_thinking else 0.2,
                    "top_k": 20,
                    "top_p": 0.95,
                    "min_p": 0.0,
                    "seed": len(s.events),
                },
                thinking=use_thinking,
                image=images,
            )

        with GPU_LOCK:
            output = generate(prompt, budget - reserve, thinking)
            text = str(output.args[0] or "").strip()
            if not text and thinking:
                if s.cancel.is_set() or s.status != "running":
                    return ""  # tick checks cancellation before parsing
                s.event(
                    "decision",
                    summary=(
                        "Thinking returned no final action. Requesting action JSON with Thinking off "
                        f"using the reserved {reserve} tokens; Thinking remains enabled for future decisions."
                    ),
                    phase="finalize_action",
                    token_budget=reserve,
                )
                # Do not persist, display or feed truncated private reasoning back as instructions.
                # Reuse the original task, observed state and images, with an explicit final-action request.
                output = generate(
                    prompt
                    + "\nFINAL ACTION REQUIRED: The previous attempt returned no final action. "
                    "Return exactly one registered tool action JSON now, for example "
                    '{"tool":"inspect_source","arguments":{"shot":0},"summary":"Inspect source and reference"}. '
                    "Choose the next useful action from the actual state above. Do not repeat the example blindly. "
                    "No prose, reasoning or Markdown. Do not claim a render happened unless the state confirms it.",
                    reserve,
                    False,
                )
                text = str(output.args[0] or "").strip()
                if not text:
                    raise ValueError(
                        "VLM returned no final action even with Thinking off during finalization. "
                        "Check the checkpoint/tokenizer compatibility or disable Thinking and Resume."
                    )
        if not text:
            raise ValueError(
                "VLM returned no final action with Thinking off. Check the checkpoint/tokenizer compatibility."
            )
        return text

    def execute(self, s, tool, args):
        # Serialize SAM, VLM and preview decoding: these models may offload one another.
        with GPU_LOCK:
            return self._execute(s, tool, args)

    def _execute(self, s, tool, args):
        p = self.p
        if tool == "set_checklist":
            steps = args.get("steps")
            if not isinstance(steps, list) or not 1 <= len(steps) <= 24:
                raise ValueError("Checklist needs 1..24 steps")
            for step in steps:
                if (
                    not isinstance(step, dict)
                    or not isinstance(step.get("title"), str)
                    or not step["title"].strip()
                    or step.get("status") not in ("pending", "in_progress", "done")
                ):
                    raise ValueError("Invalid checklist step")
            s.checklist = [
                {"title": step["title"][:300], "status": step["status"]}
                for step in steps
            ]
            return {"checklist": s.checklist}, []
        if tool == "inspect_plan":
            _, a, segs = self.context(s)
            return {
                "analysis": {
                    k: v for k, v in a.items() if k not in ("score", "raw", "thumbs")
                },
                "shots": segs,
                "revision": s.revision,
                "rendered_revision": s.rendered_revision,
                "reviewed_shots": sorted(s.reviewed),
                "cache": p.cache_status(s.plan),
                "generation_nodes": {
                    k: v
                    for k, v in (s.workflow or {}).items()
                    if v.get("class_type")
                    in (
                        "KSampler",
                        "KSamplerAdvanced",
                        "RandomNoise",
                        "BasicScheduler",
                        "CFGGuider",
                        "BFSAgentPreviewSampler",
                    )
                },
            }, []
        if tool == "inspect_source":
            labels, imgs = self.source_images(s, args)
            return {"image_order": labels}, imgs
        if tool == "update_plan":
            changes = args.get("changes", {})
            allowed = {
                "global_prompt",
                "mode",
                "max_s",
                "min_s",
                "max_parts",
                "megapixels",
                "generation_size_mode",
                "custom_width",
                "custom_height",
            }
            if not changes or set(changes) - allowed:
                raise ValueError("Unsupported global settings")
            if "mode" in changes and changes["mode"] not in (
                "shots",
                "fixed",
                "single",
            ):
                raise ValueError("Invalid split mode")
            for k in ("max_s", "min_s", "megapixels", "custom_width", "custom_height"):
                if k in changes and (
                    isinstance(changes[k], bool)
                    or not isinstance(changes[k], (int, float))
                    or not 0.01 <= changes[k] <= 8192
                ):
                    raise ValueError("Invalid " + k)
            candidate = copy.deepcopy(s.plan)
            candidate.update(changes)
            path = p._input_path(candidate["video"])
            a = p.analyze(path, float(candidate["fps"]))
            p.resolve_plan(candidate, a, path)
            s.plan = candidate
            self.invalidate(s)
            return {"changed": changes}, []
        if tool == "update_shot":
            _, _, _, i, seg = self.shot(s, args)
            changes = args.get("changes", {})
            allowed = {
                "prompt",
                "target",
                "refs",
                "crop",
                "inpaint",
                "paste",
                "strength",
                "chain",
                "chain_history",
                "enabled",
            }
            if not changes or set(changes) - allowed:
                raise ValueError("Unsupported shot changes")
            for k in ("crop", "inpaint", "paste", "enabled"):
                if k in changes and not isinstance(changes[k], bool):
                    raise ValueError(k + " must be boolean")
            for k in ("prompt", "target"):
                if k in changes and (
                    not isinstance(changes[k], str) or len(changes[k]) > 16000
                ):
                    raise ValueError("Invalid " + k)
            if "strength" in changes and not 0.05 <= float(changes["strength"]) <= 1:
                raise ValueError("Strength must be .05..1")
            if "chain" in changes and changes["chain"] not in (
                "off",
                "head",
                "history",
            ):
                raise ValueError("Invalid continuity")
            if "refs" in changes:
                if (
                    not isinstance(changes["refs"], list)
                    or len(changes["refs"]) > p.MAX_REFS
                ):
                    raise ValueError("Invalid references")
                for n in changes["refs"]:
                    if n:
                        p._input_path(n)
            self.update_meta(s, i, changes)
            return {"shot": i, "changed": changes}, []
        if tool == "split_shot":
            _, _, segs, i, seg = self.shot(s, args)
            f = args.get("frame")
            if not isinstance(f, int) or not seg["start"] < f < seg["end"]:
                raise ValueError("Split frame must be inside shot")
            # Convert to explicit boundaries and materialize metadata before inserting.
            meta = [
                {
                    **(
                        s.plan.get("segs", [{}] * len(segs))[j]
                        if j < len(s.plan.get("segs", []))
                        else {}
                    )
                }
                for j in range(len(segs))
            ]
            meta.insert(i + 1, copy.deepcopy(meta[i]))
            s.plan["segs"] = meta
            s.plan["bounds"] = [x["end"] for x in segs[:-1]] + [f]
            s.plan["bounds"] = sorted(set(s.plan["bounds"]))
            self.invalidate(s)
            return {"split_frame": f}, []
        if tool in ("segment", "preview_mask"):
            path, a, segs, i, seg = self.shot(s, args)
            if tool == "segment":
                text = args.get("text", "")
                points = args.get("points", [])
                key = int(args.get("key", 0))
                if (
                    not isinstance(text, str)
                    or not isinstance(points, list)
                    or len(points) > 64
                    or not 0 <= key < seg["end"] - seg["start"]
                ):
                    raise ValueError("Invalid SAM selection")
                for q in points:
                    if (
                        not isinstance(q, dict)
                        or not 0 <= float(q.get("x", -1)) <= 1
                        or not 0 <= float(q.get("y", -1)) <= 1
                        or q.get("label", 1) not in (0, 1)
                    ):
                        raise ValueError("Invalid normalized point")
                keyframes = args.get("keyframes", [])
                if not isinstance(keyframes, list) or len(keyframes) > 8:
                    raise ValueError("Use at most eight SAM correction keyframes")
                seen = set()
                for kf in keyframes:
                    if not isinstance(kf, dict) or set(kf) - {"key", "points", "text"}:
                        raise ValueError("Invalid keyframe fields")
                    frame = kf.get("key")
                    if (
                        isinstance(frame, bool)
                        or not isinstance(frame, int)
                        or not 0 <= frame < seg["end"] - seg["start"]
                        or frame in seen
                    ):
                        raise ValueError(
                            "Keyframes must have unique relative frame indices inside the shot"
                        )
                    seen.add(frame)
                    if not isinstance(kf.get("text", text), str) or not isinstance(
                        kf.get("points", []), list
                    ):
                        raise ValueError("Invalid keyframe selection")
                    for point in kf.get("points", []):
                        if (
                            not isinstance(point, dict)
                            or not 0 <= float(point.get("x", -1)) <= 1
                            or not 0 <= float(point.get("y", -1)) <= 1
                            or point.get("label", 1) not in (0, 1)
                        ):
                            raise ValueError("Invalid normalized keyframe point")
                cfg = args.get("settings", {})
                if set(cfg) - set(p.DEFAULT_MASK) or set(cfg) & {
                    "text",
                    "points",
                    "key",
                    "keyframes",
                }:
                    raise ValueError("Unsupported mask settings")
                for k, v in cfg.items():
                    default = p.DEFAULT_MASK[k]
                    if isinstance(default, bool):
                        if not isinstance(v, bool):
                            raise ValueError("Invalid " + k)
                    elif isinstance(default, (int, float)):
                        if not isinstance(v, (int, float)) or not 0 <= v <= 256:
                            raise ValueError("Invalid " + k)
                s.plan["mask_cfg"] = {**s.plan.get("mask_cfg", {}), **cfg}
                self.update_meta(
                    s,
                    i,
                    {
                        "mask": {
                            "text": text,
                            "points": points,
                            "key": key,
                            "keyframes": keyframes,
                        }
                    },
                )
                path, a, segs, i, seg = self.shot(s, args)
            spec = p.plan_spec(s.plan, seg.get("mask"), path)
            out = p.mask_preview(
                path, a, seg["start"], seg["end"] - seg["start"], spec, count=4
            )
            imgs = [image_tensor(x["src"]) for x in out["frames"]]
            return out, imgs
        if tool == "describe_refs":
            _, _, _, i, seg = self.shot(s, args)
            clip = p._VLM.get("clip")
            if clip is None:
                raise ValueError("VLM is not registered")
            names = p.shot_ref_names(seg["refs"], s.plan["global_refs"])
            out = {}
            for n in names:
                out[n] = p.describe_picture(
                    clip,
                    p._load_image(n),
                    s.plan.get("global_prompt", ""),
                    "",
                    p.vlm_cfg(s.plan.get("vlm_cfg")),
                )
            s.plan.setdefault("picture_details", {}).update(out)
            return {"pictures": out}, []
        if tool == "write_prompt":
            path, a, segs, i, seg = self.shot(s, args)
            clip = p._VLM.get("clip")
            if clip is None:
                raise ValueError("VLM is not registered")
            names = p.shot_ref_names(seg["refs"], s.plan["global_refs"])
            out = p.auto_prompt_shot(
                clip,
                s.plan,
                a,
                path,
                seg,
                names,
                [p._load_image(n) for n in names],
                p.saved_picture_details(s.plan),
                str(args.get("instruction") or s.task),
                p.vlm_cfg(s.plan.get("vlm_cfg")),
                "",
            )
            self.update_meta(s, i, {"prompt": out["text"]})
            return out, []
        if tool == "inspect_result":
            import numpy as np

            _, a, segs, i, seg = self.shot(s, args)
            if s.rendered_revision != s.revision:
                raise ValueError(
                    "Render the changed plan before reviewing cached results"
                )
            if not seg.get("run"):
                raise ValueError("Shot is disabled or filtered out")
            ri = sum(bool(x.get("run")) for x in segs[:i])
            status = p.cache_status(s.plan)
            if ri >= len(status["ok"]) or not status["ok"][ri]:
                raise ValueError("This shot has no current rendered take")
            cached_video = Path(status["folder"]) / f"shot_{ri:04d}.mkv"
            if cached_video.exists():
                probe = p._probe(str(cached_video))
                picks = np.linspace(
                    0, probe["n_src"] - 1, min(6, probe["n_src"])
                ).astype(int)
                scale = min(1, 512 / max(probe["width"], probe["height"]))
                frames = p._read_frames(
                    str(cached_video),
                    picks,
                    (
                        max(2, round(probe["width"] * scale)),
                        max(2, round(probe["height"] * scale)),
                    ),
                )
                imgs = [image_tensor(f) for f in frames]
            else:
                frames, meta = p.load_shot(status["folder"], ri)
                picks = np.linspace(0, len(frames) - 1, min(6, len(frames))).astype(int)
                imgs = [image_tensor(frames[j]) for j in picks]
            labels = [f"generated frame {j}" for j in picks]
            lab, src = self.source_images(s, args)
            imgs += src
            labels += lab
            s.reviewed.add(i)
            return {
                "shot": i,
                "image_order": labels,
                "frames": [
                    {"src": image_url(x), "label": label}
                    for label, x in zip(labels, imgs)
                ],
            }, imgs
        if tool == "inspect_preview":
            if not s.preview:
                raise ValueError(
                    "No intermediate preview captured. Connect BFS Agent Preview Sampler."
                )
            return s.preview, [
                image_tensor(x["src"]) for x in s.preview.get("frames", [])
            ]
        if tool in ("retry_shot", "restore_take"):
            _, _, segs, i, seg = self.shot(s, args)
            status = p.cache_status(s.plan)
            if not seg.get("run"):
                raise ValueError("Shot is disabled or filtered out")
            ri = sum(bool(x.get("run")) for x in segs[:i])
            d = Path(status["folder"])
            backup_root = self.folder() / s.id / "takes" / f"{i:04d}"
            if tool == "retry_shot":
                import uuid

                if not s.render_snapshot:
                    raise ValueError("No successful render snapshot to archive")
                backup = backup_root / uuid.uuid4().hex
                backup.mkdir(parents=True, exist_ok=True)
                copied = 0
                for f in d.glob(f"shot_{ri:04d}*"):
                    if f.is_file():
                        shutil.copy2(f, backup / f.name)
                        copied += 1
                if not copied:
                    raise ValueError("This shot has no stored take to archive")
                (backup / "render.json").write_text(
                    json.dumps(s.render_snapshot), encoding="utf-8"
                )
                out = p.redo_shots(s.plan, [ri], True)
                out["archived_take"] = backup.name
            else:
                archives = (
                    sorted(backup_root.iterdir(), key=lambda x: x.stat().st_mtime)
                    if backup_root.exists()
                    else []
                )
                if not archives:
                    raise ValueError("No archived take")
                backup = archives[-1]
                snapshot = json.loads((backup / "render.json").read_text())
                for f in backup.iterdir():
                    if f.name != "render.json":
                        shutil.copy2(f, d / f.name)
                s.plan = copy.deepcopy(snapshot["plan"])
                s.workflow = copy.deepcopy(snapshot["workflow"])
                out = {
                    "restored": i,
                    "take_id": backup.name,
                    "message": "Full archived render settings restored. Render to reconcile cached shots, then review.",
                }
            self.invalidate(s)
            return out, []
        if tool == "combine":
            _, _, segs = self.context(s)
            status = p.cache_status(s.plan)
            active = {i for i, x in enumerate(segs) if x.get("run")}
            if not active or not active <= s.reviewed:
                raise ValueError("Inspect every active rendered shot before combining")
            if (
                s.rendered_revision != s.revision
                or any(not x for x in status["ok"])
                or len(status["ok"]) != len(active)
            ):
                raise ValueError("Render every active shot first")
            out = p.combine_stored(s.plan, prefix="bfs_agent_" + s.id[:8])
            s.combined = out
            return out, []
        if tool == "update_sampler":
            node_id = str(args.get("node_id", ""))
            changes = args.get("changes", {})
            if not s.workflow or node_id not in s.workflow:
                raise ValueError("Unknown workflow node")
            node = s.workflow[node_id]
            kind = node.get("class_type")
            allowed = {
                "KSampler": {"seed", "steps", "cfg"},
                "KSamplerAdvanced": {"noise_seed", "steps", "cfg"},
                "RandomNoise": {"noise_seed"},
                "BasicScheduler": {"steps", "denoise"},
                "CFGGuider": {"cfg"},
            }
            if kind not in allowed or not changes or set(changes) - allowed[kind]:
                raise ValueError("Unsupported sampler change")
            for k, v in changes.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError("Numeric settings required")
                if k in ("seed", "noise_seed") and (
                    not isinstance(v, int) or not 0 <= v < 2**64
                ):
                    raise ValueError("Invalid seed")
                if k == "steps" and (not isinstance(v, int) or not 1 <= v <= 200):
                    raise ValueError("Invalid steps")
                if k == "cfg" and not 0 <= v <= 30:
                    raise ValueError("Invalid cfg")
                if k == "denoise" and not 0 < v <= 1:
                    raise ValueError("Invalid denoise")
                if isinstance(node.get("inputs", {}).get(k), list):
                    raise ValueError(
                        "Setting is connected; change its source node instead"
                    )
            node["inputs"].update(changes)
            self.invalidate(s)
            return {"node_id": node_id, "changed": changes}, []
        if tool in ("list_directory", "read_file", "inspect_media"):
            roots = {
                "input": p.folder_paths.get_input_directory(),
                "output": p.folder_paths.get_output_directory(),
                "project": s.settings["project_root"],
            }
            path = bounded_path(
                roots, args.get("root", "input"), str(args.get("path", ""))
            )
            if tool == "inspect_media":
                if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                    from PIL import Image

                    with Image.open(path) as im:
                        imgs = [image_tensor(im.convert("RGB"))]
                elif path.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".avi"):
                    import numpy as np

                    a = p._probe(str(path))
                    picks = np.linspace(0, a["n_src"] - 1, min(6, a["n_src"])).astype(
                        int
                    )
                    scale = min(1, 512 / max(a["width"], a["height"]))
                    frames = p._read_frames(
                        str(path),
                        picks,
                        (
                            max(2, round(a["width"] * scale)),
                            max(2, round(a["height"] * scale)),
                        ),
                    )
                    imgs = [image_tensor(x) for x in frames]
                else:
                    raise ValueError("Unsupported media file")
                return {
                    "path": str(path),
                    "frames": [
                        {"src": image_url(x), "label": str(i)}
                        for i, x in enumerate(imgs)
                    ],
                }, imgs
            if tool == "list_directory":
                entries = []
                for x in sorted(path.iterdir(), key=lambda x: x.name)[:200]:
                    if not x.name.startswith("."):
                        entries.append(
                            {
                                "name": x.name,
                                "directory": x.is_dir(),
                                "bytes": x.stat().st_size if x.is_file() else None,
                            }
                        )
                return {
                    "root": args.get("root", "input"),
                    "path": args.get("path", ""),
                    "entries": entries,
                }, []
            if path.suffix.lower() not in (
                ".txt",
                ".md",
                ".json",
                ".yaml",
                ".yml",
                ".csv",
                ".log",
                ".py",
                ".js",
            ):
                raise ValueError("Read text context files only")
            with path.open("r", encoding="utf-8", errors="replace") as f:
                text = f.read(24000)
            return {"untrusted": True, "path": str(path), "text": text}, []
        if tool in ("search_web", "fetch_url"):
            if not s.settings["web"]:
                raise ValueError("Web tools disabled for this session")
            if tool == "fetch_url":
                return fetch_page(str(args.get("url", ""))), []
            query = str(args.get("query", "")).strip()[:500]
            if not query:
                raise ValueError("Search query is empty")
            return fetch_page(
                "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
            ), []
        raise ValueError("Tool not implemented: " + tool)


def register(planner):
    from aiohttp import web
    from server import PromptServer
    import asyncio

    rt = Runtime(planner)
    routes = PromptServer.instance.routes

    tasks = {}

    async def off(fn):
        return await asyncio.to_thread(fn)

    async def queue(s):
        import execution
        import inspect
        import uuid

        if not s.workflow or s.planner_id not in s.workflow:
            raise ValueError(
                "Start Agent from a Planner with a valid workflow snapshot"
            )
        graph = copy.deepcopy(s.workflow)
        graph[s.planner_id]["inputs"]["plan"] = json.dumps(s.plan)
        # Recursive mode completes all shots within this one queue entry.
        render_plan = copy.deepcopy(s.plan)
        render_plan["run"] = "recursive"
        render_plan["auto_continue"] = False
        graph[s.planner_id]["inputs"]["plan"] = json.dumps(render_plan)
        # Select the chosen Planner's downstream outputs and their input dependencies.
        downstream = {s.planner_id}
        changed = True
        while changed:
            changed = False
            for key, node in graph.items():
                if key not in downstream and any(
                    isinstance(v, list) and len(v) == 2 and v[0] in downstream
                    for v in node.get("inputs", {}).values()
                ):
                    downstream.add(key)
                    changed = True
        import nodes

        outputs = {
            k
            for k in downstream
            if getattr(
                nodes.NODE_CLASS_MAPPINGS.get(graph[k].get("class_type")),
                "OUTPUT_NODE",
                False,
            )
        }
        if not outputs:
            raise ValueError(
                "Connect Planner shots through conditioning/sampling to BFS Shot Join and an output"
            )
        needed = set()

        def collect(key):
            if key in needed:
                return
            needed.add(key)
            for v in graph[key].get("inputs", {}).values():
                if (
                    isinstance(v, list)
                    and len(v) == 2
                    and isinstance(v[0], str)
                    and v[0] in graph
                ):
                    collect(v[0])

        for key in outputs:
            collect(key)
        # Keep the Planner/VLM ancestor chain for the initialization-only queue as well.
        collect(s.planner_id)
        graph = {k: v for k, v in graph.items() if k in needed}
        if s.pending_render.get("kind") == "context":
            link = graph[s.planner_id]["inputs"].get("vlm")
            if not isinstance(link, list):
                raise ValueError(
                    "Connect a VLM to the Planner vlm input before starting Agent"
                )
            root = "bfs_agent_context_" + s.id
            graph[root] = {
                "class_type": "BFSAgentContext",
                "inputs": {"vlm": link, "session_id": s.id},
            }
            keep = set()

            def visit(n):
                if n in keep:
                    return
                if n not in graph:
                    raise ValueError("Missing VLM dependency " + n)
                keep.add(n)
                for value in graph[n].get("inputs", {}).values():
                    if (
                        isinstance(value, list)
                        and len(value) == 2
                        and isinstance(value[0], str)
                    ):
                        visit(value[0])

            visit(root)
            graph = {k: v for k, v in graph.items() if k in keep}
        for node in graph.values():
            if node.get("class_type") == "BFSAgentPreviewSampler":
                node["inputs"]["session_id"] = s.id
                node["inputs"]["shot"] = [s.planner_id, 0]
        prompt_id = uuid.uuid4().hex
        fn = execution.validate_prompt
        count = len(inspect.signature(fn).parameters)
        valid = fn(prompt_id, graph, None) if count >= 3 else fn(prompt_id, graph)
        if inspect.isawaitable(valid):
            valid = await valid
        if not valid[0]:
            raise ValueError(
                "Workflow validation failed: " + json.dumps(valid[1:])[:4000]
            )
        server = PromptServer.instance
        extra = {
            "client_id": s.client_id,
            "create_time": int(time.time() * 1000),
            "bfs_agent_session": s.id,
        }
        number = server.number
        server.number += 1
        # Queue tuple format follows the installed ComfyUI version.
        import execution as ex

        item = (number, prompt_id, graph, extra, valid[2])
        if hasattr(ex, "SENSITIVE_EXTRA_DATA_KEYS"):
            item += ({},)
        if s.cancel.is_set() or s.status != "waiting_render":
            return
        s.pending_render["prompt_id"] = prompt_id
        if s.pending_render.get("kind") != "context":
            s.pending_render["snapshot"] = {
                "plan": copy.deepcopy(s.plan),
                "workflow": copy.deepcopy(s.workflow),
            }
        rt.save(s)  # persist prompt id before submission for reconnect/recovery
        server.prompt_queue.put(item)
        s.event(
            "render",
            prompt_id=prompt_id,
            summary="Loading Agent VLM context"
            if s.pending_render.get("kind") == "context"
            else "Rendering Planner workflow",
        )

    async def drive(s):
        try:
            while s.status in ("running", "waiting_render"):
                if s.status == "running" and not s.context_ready:
                    import uuid

                    s.pending_render = {
                        "nonce": uuid.uuid4().hex,
                        "prompt_id": None,
                        "kind": "context",
                    }
                    s.status = "waiting_render"
                if s.status == "waiting_render":
                    if not s.pending_render.get("prompt_id"):
                        await queue(s)
                    if s.status != "waiting_render" or s.cancel.is_set():
                        break
                    pid = s.pending_render["prompt_id"]
                    entry = PromptServer.instance.prompt_queue.get_history(
                        prompt_id=pid
                    ).get(pid)
                    if not entry:
                        running, queued = (
                            PromptServer.instance.prompt_queue.get_current_queue()
                        )
                        ids = {str(x[1]) for x in [*running, *queued]}
                        if pid not in ids:
                            s.status = "error"
                            s.event(
                                "error",
                                error="Render is absent from queue/history. Check ComfyUI restart or history eviction.",
                            )
                            break
                        await asyncio.sleep(2)
                        continue
                    st = entry.get("status", {})
                    preview_review = (s.preview or {}).get("review")
                    kind = s.pending_render.get("kind", "render")
                    revision = s.pending_render.get("revision", -1)
                    snapshot = s.pending_render.get("snapshot")
                    s.pending_render = None
                    s.status = "running"
                    if st.get("status_str") == "success":
                        if kind == "context":
                            s.context_ready = True
                            await off(lambda: rt.diagnose_vlm(s))
                        else:
                            s.rendered_revision = revision
                            s.render_snapshot = snapshot
                    elif kind == "context":
                        s.status = "error"
                        s.event(
                            "error",
                            error="VLM context initialization failed: " + str(st),
                        )
                        break
                    s.observation = {
                        "render_status": st,
                        "preview_review": preview_review,
                        "message": (
                            "VLM context loaded. No video was generated by this initialization. "
                            "Inspect the plan and source, prepare the shots, then call render before inspect_result."
                            if kind == "context"
                            else "Inspect generated shots before combine."
                        )
                        if st.get("status_str") == "success"
                        else "Render failed; correct the observed failure before retrying.",
                        "kind": kind,
                        "current_plan_rendered": s.rendered_revision == s.revision,
                    }
                    s.event("observation", result=s.observation)
                    rt.save(s)
                    continue
                # Do not load SAM/VLM alongside another queued or running generation.
                running, _ = PromptServer.instance.prompt_queue.get_current_queue()
                if running:
                    await asyncio.sleep(2)
                    continue
                await off(lambda: s.tick(rt.decide, rt.execute))
                rt.save(s)
                await asyncio.sleep(0.1)
        except Exception as exc:
            if s.status not in ("paused", "stopped"):
                s.status = "error"
                s.event("error", error=f"{type(exc).__name__}: {exc}")
        finally:
            rt.save(s)

    def launch(s):
        if s.id not in tasks or tasks[s.id].done():
            tasks[s.id] = asyncio.create_task(drive(s))

    def response(s):
        rt.save(s)
        return web.json_response(s.public())

    @routes.post("/bfs/shotloop/agent/start")
    async def start(request):
        try:
            b = await request.json()
            if any(
                x.status in ("running", "waiting_render") for x in SESSIONS.values()
            ):
                raise ValueError(
                    "Another agent session is active; pause or stop it first"
                )
            plan = planner._load_plan(json.dumps(b.get("plan", {})))
            s = Session(b.get("task", ""), plan, b.get("settings"))
            workflow = b.get("workflow")
            pid = str(b.get("planner_id", ""))
            if (
                not isinstance(workflow, dict)
                or pid not in workflow
                or workflow[pid].get("class_type") != "BFSShotPlanner"
            ):
                raise ValueError("Workflow snapshot must contain this BFSShotPlanner")
            s.workflow = workflow
            s.planner_id = pid
            s.client_id = b.get("client_id")
            if not isinstance(workflow[pid].get("inputs", {}).get("vlm"), list):
                raise ValueError(
                    "Connect a VLM to the Planner vlm input before starting Agent"
                )
            # Agent controls the Planner prompt; linked external prompt text must not silently override it.
            if workflow[pid].get("inputs", {}).get("prompt"):
                raise ValueError(
                    "Disconnect or clear the optional Planner prompt input; the agent writes prompts in the plan."
                )
            rt.attach(s)
            SESSIONS[s.id] = s
            s.event("status", summary="Agent session started")
            rt.save(s)
            launch(s)
            return response(s)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @routes.get("/bfs/shotloop/agent/session/{sid}")
    async def status(request):
        try:
            return web.json_response(rt.get(request.match_info["sid"]).public())
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @routes.post("/bfs/shotloop/agent/control")
    async def control(request):
        try:
            b = await request.json()
            s = rt.get(b["id"])
            action = b["action"]
            if action in ("pause", "stop"):
                s.cancel.set()
                s.status = "paused" if action == "pause" else "stopped"
                if (
                    action == "stop"
                    and s.pending_render
                    and s.pending_render.get("prompt_id")
                ):
                    pid = s.pending_render["prompt_id"]
                    pq = PromptServer.instance.prompt_queue
                    pq.delete_queue_item(lambda item: str(item[1]) == pid)
                    running, _ = pq.get_current_queue()
                    if any(str(item[1]) == pid for item in running):
                        import nodes

                        nodes.interrupt_processing(True)
                    s.pending_render = None
            elif action == "resume":
                if s.status not in ("paused", "error", "budget_exhausted"):
                    raise ValueError(
                        "Only paused, errored or budget-limited sessions can resume"
                    )
                settings = b.get("settings") or {}
                for key in ("thinking", "web"):
                    if key in settings:
                        s.settings[key] = bool(settings[key])
                for key, maximum in (
                    ("max_actions", 200),
                    ("max_renders", 20),
                    ("max_tokens", 16384),
                ):
                    if key in settings:
                        value = int(settings[key])
                        if not 1 <= value <= maximum:
                            raise ValueError("Invalid " + key)
                        s.settings[key] = value
                if s.lock.locked():
                    raise ValueError("Wait for current action to finish")
                if any(
                    x.id != s.id and x.status in ("running", "waiting_render")
                    for x in SESSIONS.values()
                ):
                    raise ValueError("Another Agent session is active")
                if isinstance(b.get("plan"), dict):
                    candidate = planner._load_plan(json.dumps(b["plan"]))
                    if candidate.get("video") != s.plan.get("video"):
                        raise ValueError("Start a new task to change the source video")
                    old = {k: v for k, v in s.plan.items() if k != "agent_session_id"}
                    new = {
                        k: v for k, v in candidate.items() if k != "agent_session_id"
                    }
                    if old != new:
                        s.plan = candidate
                        rt.invalidate(s)
                if s.context_ready:
                    rt.diagnose_vlm(s)
                s.cancel.clear()
                s.error_count = 0
                s.status = "waiting_render" if s.pending_render else "running"
                launch(s)
            else:
                raise ValueError("Unknown control action")
            s.event("status", summary=action)
            return response(s)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    return rt
