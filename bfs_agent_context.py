"""Persistent, grounded context compaction. No extra GPU generation is needed."""

import json
import math
import time
from .bfs_agent_review import text_context


class ContextBudgetError(ValueError):
    pass


def tokens(text):
    # This is a working estimate, not the model tokenizer or measured KV usage.
    return math.ceil(len(text.encode("utf-8")) / 3)


def compact(s, through):
    previous = s.context_summary or {}
    start = min(previous.get("through_event", 0), through)
    facts = list(previous.get("facts", []))
    for event in s.events[start:through]:
        kind = event.get("kind")
        if kind == "action":
            facts.append(
                {
                    "type": "requested_tool",
                    "tool": event.get("tool"),
                    "arguments": text_context(event.get("arguments", {})),
                    "unverified_assistant_claim": event.get("summary", "")[:300],
                }
            )
        elif kind == "error":
            facts.append({"type": "failure", "error": event.get("error", "")[:600]})
        elif kind in ("feedback", "reply", "media"):
            facts.append({"type": kind, "note": event.get("summary", "")[:600]})
        elif kind == "observation":
            result = event.get("result", {})
            facts.append(
                {
                    "type": "tool_observation",
                    "tool": event.get("tool"),
                    "result": json.dumps(text_context(result), ensure_ascii=False)[
                        :800
                    ],
                }
            )
    # Older raw events remain available on disk and through inspect_history.
    facts = facts[-16:]
    while len(json.dumps(facts, ensure_ascii=False)) > 5000 and len(facts) > 1:
        facts.pop(0)
    s.context_summary = {
        "facts": facts,
        "through_event": through,
        "compactions": previous.get("compactions", 0) + 1,
        "updated": time.time(),
        "note": "Grounded activity digest; requests are not proof of success. Full events remain saved.",
    }
    s.event(
        "context",
        summary="Condensed older activity; task, feedback, live plan and saved evidence remain available.",
        through_event=through,
        compactions=s.context_summary["compactions"],
    )


def prepare_context(s, system, tools, image_count):
    info = s.vlm_info or {}
    requested = int(s.settings.get("context_window", 32768))
    backend = info.get("context_tokens")
    capacity = (
        min(requested, backend)
        if isinstance(backend, int) and backend > 0
        else requested
    )
    source = (
        "backend configuration + working limit"
        if backend
        else "working limit; backend maximum unknown"
    )
    vision = (
        min(24, image_count) * 512
    )  # heuristic for 512px letterboxed images; backend-dependent
    reserve = max(1024, int(capacity * 0.08))
    output = s.settings["max_tokens"]
    text_room = capacity - output - vision - reserve
    state = {
        "revision": s.revision,
        "rendered_revision": s.rendered_revision,
        "current_plan_rendered": s.revision == s.rendered_revision,
        "reviewed_shots": sorted(s.reviewed),
        "active_shots": getattr(s, "active_shots", None),
        "combined": bool(s.combined),
        "coverage_is_not_quality_approval": True,
        "renders": s.renders,
        "render_limit": s.settings["max_renders"],
        "actions": sum(e["kind"] == "action" for e in s.events),
        "action_limit": s.settings["max_actions"],
        "review_fps": s.settings["review_fps"],
        "review_coverage": {
            k: {a: b for a, b in v.items() if a != "seen"}
            for k, v in s.review_coverage.items()
        },
    }
    # Keep the latest user instruction intact; older corrections stay in the digest/source history.
    applied = [m for m in s.messages if m.get("status") == "applied"]
    latest = applied[-1:] or s.feedback[-1:]
    # Keep current facts ahead of potentially truncated history/plan sections.
    # Assistant action summaries are intent, never evidence that a tool succeeded.
    last_outcome = next(
        (e for e in reversed(s.events) if e.get("kind") in ("observation", "error")),
        {},
    )
    live = {
        "video": s.plan.get("video"),
        "fps": s.plan.get("fps"),
        "global_refs": s.plan.get("global_refs", []),
        "latest_tool": last_outcome.get("tool"),
        "latest_tool_failed": last_outcome.get("kind") == "error",
        "confirmed_changes": (last_outcome.get("result") or {}).get("changed"),
        "already_applied": (last_outcome.get("result") or {}).get("already_applied"),
        "next_step": (last_outcome.get("result") or {}).get("next_step"),
        "instruction": "Use actual reference filenames from the current plan/inspect_source, never filenames from old assistant claims. Successful changes are already saved; do not repeat them. After a tool error change the request according to the error. Segmentation overlays are visual evidence; inspect them instead of repeating unchanged segment. temporal_expand is mask dilation, not proof that tracking is correct across the video.",
    }
    base = (
        system
        + "\nTOOLS:\n"
        + json.dumps(tools, ensure_ascii=False)
        + "\nAUTHORITATIVE RECOVERY (follow before older history):\n"
        + json.dumps(
            text_context(
                {
                    "error": s.observation.get("error"),
                    "recovery": s.observation.get("recovery"),
                    "instruction": "Use explicit integer shot for inspect_result. Review ALL active shots. Presented frames are coverage, not proof of visual correctness. If current_plan_rendered=true inspect results rather than render again. Finish requires combine first. User corrections override prior assistant claims of success; analyze attached screenshots and locate the defects before combining.",
                }
            ),
            ensure_ascii=False,
        )
        + "\nCURRENT VISUAL INPUT (ordered, actual attached images):\n"
        + json.dumps(
            text_context(s.observation.get("image_order", [])), ensure_ascii=False
        )
        + "\nImages are letterboxed to square: coordinates for segment refer to the original image content, excluding padding. Only describe what is visible; uncertainty is allowed. Never claim an action succeeded before its observation.\n"
        + "\nTASK:\n"
        + s.task
        + "\nLATEST USER INPUT:\n"
        + json.dumps(text_context(latest), ensure_ascii=False)
        + "\nGENERATION STATE:\n"
        + json.dumps(state, ensure_ascii=False)
        + "\nAUTHORITATIVE LIVE FACTS AND LAST TOOL OUTCOME:\n"
        + json.dumps(text_context(live), ensure_ascii=False)
    )
    if text_room <= tokens(base):
        s.context_usage = {
            "capacity": capacity,
            "capacity_source": source,
            "estimated": True,
            "text_tokens": tokens(base),
            "image_tokens": vision,
            "output_tokens": output,
            "safety_tokens": reserve,
            "total_tokens": tokens(base) + vision + output + reserve,
            "blocked": True,
        }
        raise ContextBudgetError(
            "Working context is too small for instructions, images and the decision budget. Increase Context window only within the model's supported limit, or reduce decision tokens/images/task length."
        )
    through = (s.context_summary or {}).get("through_event", 0)
    recent = s.events[through:]
    raw_size = tokens(json.dumps(text_context(recent), ensure_ascii=False))
    if len(recent) > 24 or raw_size > max(500, (text_room - tokens(base)) // 3):
        cutoff = max(through, len(s.events) - 6)
        if cutoff > through:
            compact(s, cutoff)
            recent = s.events[cutoff:]
    plan = {
        k: v
        for k, v in s.plan.items()
        if k not in ("auto_prompts", "ref_details", "picture_details", "cast")
    }
    if len(plan.get("segs", [])) > 8:
        plan["segs"] = plan["segs"][:8]
        plan["note"] = "More overrides saved; use inspect_plan for resolved shots."
    sections = [
        ("LATEST OBSERVATION", text_context(s.observation), 0.29),
        ("CURRENT PLAN", text_context(plan), 0.23),
        ("PERSISTENT SUMMARY", s.context_summary, 0.18),
        ("CHECKLIST", s.checklist, 0.10),
        ("EARLIER USER FEEDBACK", s.feedback[-8:], 0.10),
        ("RECENT HISTORY", text_context(recent), 0.10),
    ]
    remaining = max(0, (text_room - tokens(base) - 200) * 3)
    prompt = base
    for name, value, share in sections:
        content = json.dumps(value, ensure_ascii=False)
        limit = max(80, int(remaining * share))
        if len(content.encode("utf-8")) > limit:
            content = (
                content.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
                + " [excerpt; inspect_history/inspect_plan for details]"
            )
        prompt += "\n" + name + ":\n" + content
    prompt += (
        "\nImages follow the observation order. Return one registered tool action JSON."
    )
    estimate = tokens(prompt)
    s.context_usage = {
        "capacity": capacity,
        "capacity_source": source,
        "estimated": True,
        "text_tokens": estimate,
        "image_tokens": vision,
        "output_tokens": output,
        "safety_tokens": reserve,
        "total_tokens": estimate + vision + output + reserve,
        "compactions": (s.context_summary or {}).get("compactions", 0),
        "note": "Text bytes/3 and 512 tokens/image estimates; actual template, image and KV usage may differ.",
    }
    if s.context_usage["total_tokens"] > capacity:
        raise ContextBudgetError(
            "Working context estimate still exceeds the limit after compaction. Reduce the decision budget or images; the full history is saved."
        )
    return prompt
