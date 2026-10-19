"""Read-only VLM diagnostics. Never allocate tensors or probe memory with a generation."""

import inspect


def inspect_vlm(clip, settings, checkpoint=None):
    import torch

    report = {
        "checkpoint": checkpoint,
        "generation": callable(getattr(clip, "generate", None)),
        "images": "unknown",
        "thinking_control": "unknown",
        "thinking_checkpoint": "unknown (not established by the tokenizer API)",
        "context_tokens": None,
        "agent_output_cap": 16384,
        "requested_output_tokens": settings["max_tokens"],
        "thinking_enabled": settings["thinking"],
        "warnings": [],
        "gpu": None,
        "kv_estimate": None,
    }
    root = getattr(clip, "cond_stage_model", None)
    modules = list(root.modules()) if callable(getattr(root, "modules", None)) else []
    transformer = next(
        (m for m in modules if hasattr(m, "model_type") and hasattr(m, "model")), None
    )
    if transformer is None:
        transformer = next(
            (m for m in modules if hasattr(getattr(m, "model", None), "config")), None
        )
    report["architecture"] = getattr(transformer, "model_type", None) or (
        type(transformer).__name__ if transformer is not None else None
    )
    if any(
        getattr(m, "visual", None) is not None
        or getattr(m, "vision_model", None) is not None
        for m in modules
    ):
        report["images"] = "vision encoder detected"
    tokenizer = getattr(clip, "tokenizer", None)
    try:
        params = inspect.signature(tokenizer.tokenize_with_weights).parameters
        if "thinking" in params:
            report["thinking_control"] = "tokenizer supports the thinking switch"
    except (AttributeError, TypeError, ValueError):
        pass
    config = getattr(getattr(transformer, "model", None), "config", None)
    context = getattr(config, "max_position_embeddings", None)
    if isinstance(context, int) and 0 < context < 10_000_000:
        report["context_tokens"] = context
        report["context_source"] = (
            "loaded backend configuration; checkpoint limit may differ"
        )
    else:
        report["warnings"].append(
            "Context limit is not exposed by this backend; tokenizer padding limits are not model limits."
        )
    patcher = getattr(clip, "patcher", None)
    device = getattr(patcher, "load_device", None)
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_initialized():
        try:
            free, total = torch.cuda.mem_get_info(device)
            remaining_weights = max(0, patcher.model_size() - patcher.loaded_size())
            reserve = max(2 * 1024**3, int(total * 0.15))
            report["gpu"] = {
                "name": torch.cuda.get_device_name(device),
                "free_gib": round(free / 1024**3, 2),
                "total_gib": round(total / 1024**3, 2),
                "vlm_weights_not_resident_gib": round(remaining_weights / 1024**3, 2),
                "reserve_gib": round(reserve / 1024**3, 2),
            }
            # Only estimate the known full-attention Llama2_ cache layout. Hybrid/recurrent
            # backends and custom implementations must not receive a fabricated capacity.
            model = getattr(transformer, "model", None)
            if (
                type(model).__name__ == "Llama2_"
                and type(model).__module__ == "comfy.text_encoders.llama"
            ):
                dims = [
                    getattr(config, k, None)
                    for k in ("num_hidden_layers", "num_key_value_heads", "head_dim")
                ]
                if all(isinstance(v, int) and v > 0 for v in dims):
                    layers, heads, head_dim = dims
                    bytes_per_token = (
                        2 * layers * heads * head_dim * 4
                    )  # conservative FP32 K+V, batch=1
                    capacity = (
                        max(0, free - remaining_weights - reserve) // bytes_per_token
                    )
                    if report["context_tokens"]:
                        capacity = min(capacity, report["context_tokens"])
                    report["kv_estimate"] = {
                        "total_tokens": int(capacity),
                        "requested_output_cache_mib": round(
                            settings["max_tokens"] * bytes_per_token / 1024**2, 1
                        ),
                        "assumptions": "KV-only estimate, batch 1, FP32 cache; subtracts unloaded weights and a memory reserve. Input text, image tokens and output share this capacity. Prefill/vision/workspace peaks are not measured. Not a guaranteed fit or maximum output length.",
                    }
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            report["warnings"].append("GPU estimate unavailable: " + type(exc).__name__)
    if report["kv_estimate"] is None:
        report["warnings"].append(
            "A reliable GPU token capacity estimate is unavailable for this backend/device."
        )
    report["warnings"].append(
        "Thinking and the final action share the output budget. Images also consume context. No settings were changed automatically."
    )
    return report
