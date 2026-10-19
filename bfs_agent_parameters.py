"""Bounded edits of existing generation nodes; no graph rewiring or model loading."""

import math
import re
from decimal import Decimal

# (minimum, maximum, integer); the installed node schema may tighten these bounds.
NUMERIC = {
    "seed": (0, 2**64 - 1, True),
    "noise_seed": (0, 2**64 - 1, True),
    "steps": (1, 200, True),
    "cfg": (0, 30, False),
    "denoise": (0.0001, 1, False),
    "strength_model": (-20, 20, False),
    "strength_clip": (-20, 20, False),
    "sigma_min": (0, 5000, False),
    "sigma_max": (0, 5000, False),
    "rho": (0.0001, 100, False),
    "mu": (-10, 10, False),
    "beta": (0.0001, 50, False),
    "alpha": (0.0001, 50, False),
    "beta_d": (0.0001, 5000, False),
    "beta_min": (0.0001, 5000, False),
    "eps_s": (0.0001, 0.9999, False),
}
FIELDS = {
    "KSampler": {"seed", "steps", "cfg", "denoise", "sampler_name", "scheduler"},
    "KSamplerAdvanced": {"noise_seed", "steps", "cfg", "sampler_name", "scheduler"},
    "RandomNoise": {"noise_seed"},
    "CFGGuider": {"cfg"},
    "SamplerCustom": {"noise_seed", "cfg"},
    "KSamplerSelect": {"sampler_name"},
    "BasicScheduler": {"scheduler", "steps", "denoise"},
    "KarrasScheduler": {"steps", "sigma_min", "sigma_max", "rho"},
    "ExponentialScheduler": {"steps", "sigma_min", "sigma_max"},
    "PolyexponentialScheduler": {"steps", "sigma_min", "sigma_max", "rho"},
    "LaplaceScheduler": {"steps", "sigma_min", "sigma_max", "mu", "beta"},
    "BetaSamplingScheduler": {"steps", "alpha", "beta"},
    "VPScheduler": {"steps", "beta_d", "beta_min", "eps_s"},
    "SDTurboScheduler": {"steps", "denoise"},
    "ManualSigmas": {"sigmas"},
    "LoraLoader": {"strength_model", "strength_clip"},
    "LoraLoaderModelOnly": {"strength_model"},
    "BFSLoraSurgery": {"strength_model"},
}


def schema_inputs(kind):
    try:
        import nodes

        cls = nodes.NODE_CLASS_MAPPINGS.get(kind)
        if cls is None:
            return {}
        schema = cls.INPUT_TYPES()
        return {**schema.get("required", {}), **schema.get("optional", {})}
    except (ImportError, AttributeError):
        return {}


def controls(node):
    kind = node.get("class_type")
    schema = schema_inputs(kind)
    result = {}
    for key in sorted(FIELDS.get(kind, ())):
        if key not in node.get("inputs", {}) and key not in schema:
            continue
        spec = schema.get(key, ())
        item = {
            "value": node.get("inputs", {}).get(key),
            "connected": isinstance(node.get("inputs", {}).get(key), list),
        }
        if key in NUMERIC:
            low, high, integer = NUMERIC[key]
            limits = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            item.update(
                min=max(low, limits.get("min", low)),
                max=min(high, limits.get("max", high)),
                integer=integer,
            )
        elif key == "sigmas":
            item["format"] = (
                "2..201 finite nonnegative descending numbers, ending in zero"
            )
        else:
            options = spec[0] if spec and isinstance(spec[0], (list, tuple)) else None
            if options is None:
                try:
                    import comfy.samplers

                    options = (
                        comfy.samplers.SAMPLER_NAMES
                        if key == "sampler_name"
                        else comfy.samplers.SCHEDULER_NAMES
                    )
                except (ImportError, AttributeError):
                    options = []
            item["options"] = list(options)
        result[key] = item
    return result


def manual_sigmas(value):
    if isinstance(value, str) and len(value) <= 4096:
        value = [x for x in re.split(r"[,\s]+", value.strip()) if x]
    if not isinstance(value, list) or not 2 <= len(value) <= 201:
        raise ValueError("Use 2..201 descending sigmas ending in zero")
    try:
        if any(isinstance(x, bool) for x in value):
            raise ValueError()
        numbers = [float(x) for x in value]
        if any(not math.isfinite(x) or not 0 <= x <= 5000 for x in numbers):
            raise ValueError()
        if (
            numbers[0] <= 0
            or numbers[-1] != 0
            or any(a <= b for a, b in zip(numbers, numbers[1:]))
        ):
            raise ValueError()
        # ManualSigmas' parser does not support exponent notation.
        text = ", ".join(format(Decimal(str(x)), "f") for x in numbers)
        if len(text) > 4096:
            raise ValueError()
        return text
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(
            "Sigmas must be finite, strictly descending, nonnegative, and end in zero"
        ) from exc


def validated_changes(node, changes):
    if (
        not isinstance(changes, dict)
        or not changes
        or set(changes) - FIELDS.get(node.get("class_type"), set())
    ):
        raise ValueError(
            "Unsupported generation setting; inspect_generation lists supported controls"
        )
    available = controls(node)
    out = {}
    for key, value in changes.items():
        if key not in available:
            raise ValueError("Setting is not available on this installed node: " + key)
        item = available[key]
        if item["connected"]:
            raise ValueError(
                "Setting is connected; change its source node instead: " + key
            )
        if key == "sigmas":
            out[key] = manual_sigmas(value)
        elif key in NUMERIC:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Finite numeric setting required: " + key)
            if item["integer"] and not isinstance(value, int):
                raise ValueError("Integer setting required: " + key)
            if not item["min"] <= value <= item["max"]:
                raise ValueError(f"{key} must be within {item['min']}..{item['max']}")
            out[key] = value
        else:
            if not isinstance(value, str) or value not in item["options"]:
                raise ValueError("Choose a value listed by inspect_generation: " + key)
            out[key] = value
    merged = {**node.get("inputs", {}), **out}
    if {"sigma_min", "sigma_max"} <= merged.keys():
        low, high = merged["sigma_min"], merged["sigma_max"]
        if (
            not isinstance(low, list)
            and not isinstance(high, list)
            and not 0 < low < high
        ):
            raise ValueError("Require 0 < sigma_min < sigma_max")
    return out


def generation_nodes(workflow):
    return {
        key: {**node, "controls": controls(node)}
        for key, node in (workflow or {}).items()
        if node.get("class_type") in FIELDS
    }
