"""ComfyUI inference lifecycle for Agent calls made outside normal node execution."""

import importlib

CUDA_FAULTED = False


def ensure_cuda_healthy():
    if CUDA_FAULTED:
        raise RuntimeError(
            "CUDA context fault recorded; restart ComfyUI before running Agent inference again"
        )


def mark_cuda_fault():
    global CUDA_FAULTED
    CUDA_FAULTED = True


def cuda_context_fault(exc):
    message = str(exc).lower()
    return any(
        text in message
        for text in (
            "illegal memory access",
            "cuda_error_illegal_address",
            "cudaerrorillegaladdress",
            "device-side assert",
            "misaligned address",
            "unspecified launch failure",
            "cuda context fault",
        )
    )


def cleanup_inference():
    # Match ComfyUI's per-node lifecycle when its dynamic allocator is available.
    # Do not disable or reconfigure the user's compiler/allocator globally.
    try:
        memory = importlib.import_module("comfy.memory_management")
    except ImportError:
        return
    if not getattr(memory, "aimdo_enabled", False):
        return
    prefetch = importlib.import_module("comfy.model_prefetch")
    management = importlib.import_module("comfy.model_management")
    prefetch.cleanup_prefetch_queues()
    management.reset_cast_buffers()
    vbar = importlib.import_module("comfy_aimdo.model_vbar")
    vbar.vbars_reset_watermark_limits()


def run_inference(fn):
    import torch

    # inference_mode is thread-local: ComfyUI's queue context is not inherited
    # by asyncio/executor threads, so enter it where the work actually runs.
    ensure_cuda_healthy()
    failed_cuda = False
    with torch.inference_mode():
        try:
            return fn()
        except Exception as exc:
            failed_cuda = cuda_context_fault(exc)
            if failed_cuda:
                mark_cuda_fault()
            raise
        finally:
            # A poisoned CUDA context cannot be repaired by freeing/loading models.
            # In particular, do not enter allocator cleanup after a reported fatal fault.
            if not failed_cuda:
                cleanup_inference()
