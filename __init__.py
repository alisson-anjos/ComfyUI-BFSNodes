from .bfs_h3_cast_bindings import NODE_CLASS_MAPPINGS as H3CAST_NODE_CLASS_MAPPINGS
from .bfs_h3_cast_bindings import NODE_DISPLAY_NAME_MAPPINGS as H3CAST_NODE_DISPLAY_NAME_MAPPINGS
from .minimax_h3_token_weights import NODE_CLASS_MAPPINGS as H3TOK_NODE_CLASS_MAPPINGS
from .minimax_h3_token_weights import NODE_DISPLAY_NAME_MAPPINGS as H3TOK_NODE_DISPLAY_NAME_MAPPINGS
from .minimax_h3_reference_binding import NODE_CLASS_MAPPINGS as H3BIND_NODE_CLASS_MAPPINGS
from .minimax_h3_reference_binding import NODE_DISPLAY_NAME_MAPPINGS as H3BIND_NODE_DISPLAY_NAME_MAPPINGS
from .bfs_h3_native_shot_loop import NODE_CLASS_MAPPINGS as H3LOOP_NODE_CLASS_MAPPINGS
from .bfs_h3_native_shot_loop import NODE_DISPLAY_NAME_MAPPINGS as H3LOOP_NODE_DISPLAY_NAME_MAPPINGS
from .nodes import NODE_CLASS_MAPPINGS as BFS_NODE_CLASS_MAPPINGS
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as BFS_NODE_DISPLAY_NAME_MAPPINGS
from .ltxv_editanything import NODE_CLASS_MAPPINGS as LTXV_EA_NODE_CLASS_MAPPINGS
from .ltxv_editanything import NODE_DISPLAY_NAME_MAPPINGS as LTXV_EA_NODE_DISPLAY_NAME_MAPPINGS
from .headswap_node import NODE_CLASS_MAPPINGS as HEADSWAP_NODE_CLASS_MAPPINGS
from .headswap_node import NODE_DISPLAY_NAME_MAPPINGS as HEADSWAP_NODE_DISPLAY_NAME_MAPPINGS
from .anime2real_node import NODE_CLASS_MAPPINGS as A2R_NODE_CLASS_MAPPINGS
from .anime2real_node import NODE_DISPLAY_NAME_MAPPINGS as A2R_NODE_DISPLAY_NAME_MAPPINGS
from .amv_guide_node import NODE_CLASS_MAPPINGS as AMV_NODE_CLASS_MAPPINGS
from .amv_guide_node import NODE_DISPLAY_NAME_MAPPINGS as AMV_NODE_DISPLAY_NAME_MAPPINGS
from .ltx_identity_overlap import NODE_CLASS_MAPPINGS as IDT_NODE_CLASS_MAPPINGS
from .ltx_identity_overlap import NODE_DISPLAY_NAME_MAPPINGS as IDT_NODE_DISPLAY_NAME_MAPPINGS
from .ltx_multiref_slots import NODE_CLASS_MAPPINGS as MRSLOT_NODE_CLASS_MAPPINGS
from .ltx_multiref_slots import NODE_DISPLAY_NAME_MAPPINGS as MRSLOT_NODE_DISPLAY_NAME_MAPPINGS
from .ltx_multiple_controls import NODE_CLASS_MAPPINGS as MC_NODE_CLASS_MAPPINGS
from .ltx_multiple_controls import NODE_DISPLAY_NAME_MAPPINGS as MC_NODE_DISPLAY_NAME_MAPPINGS
from .color_mask_node import NODE_CLASS_MAPPINGS as CM_NODE_CLASS_MAPPINGS
from .color_mask_node import NODE_DISPLAY_NAME_MAPPINGS as CM_NODE_DISPLAY_NAME_MAPPINGS
try:
    from .minimax_h3_singleframe_vae import NODE_CLASS_MAPPINGS as MMVAE_NODE_CLASS_MAPPINGS
    from .minimax_h3_singleframe_vae import NODE_DISPLAY_NAME_MAPPINGS as MMVAE_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] MiniMax-H3 VAE loaders not loaded: {_e!r}")
    MMVAE_NODE_CLASS_MAPPINGS, MMVAE_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
try:
    from .sharpest_frame import NODE_CLASS_MAPPINGS as SHARP_NODE_CLASS_MAPPINGS
    from .sharpest_frame import NODE_DISPLAY_NAME_MAPPINGS as SHARP_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] Sharpest Frame node not loaded: {_e!r}")
    SHARP_NODE_CLASS_MAPPINGS, SHARP_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
try:
    from .minimax_h3_direct_decode import NODE_CLASS_MAPPINGS as MMDD_NODE_CLASS_MAPPINGS
    from .minimax_h3_direct_decode import NODE_DISPLAY_NAME_MAPPINGS as MMDD_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] MiniMax-H3 direct decode node not loaded: {_e!r}")
    MMDD_NODE_CLASS_MAPPINGS, MMDD_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
try:
    from .save_h3_latent import NODE_CLASS_MAPPINGS as SVH3_NODE_CLASS_MAPPINGS
    from .save_h3_latent import NODE_DISPLAY_NAME_MAPPINGS as SVH3_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] Save H3 Latent node not loaded: {_e!r}")
    SVH3_NODE_CLASS_MAPPINGS, SVH3_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
try:
    from .edit_prompt_fixer import NODE_CLASS_MAPPINGS as EPF_NODE_CLASS_MAPPINGS
    from .edit_prompt_fixer import NODE_DISPLAY_NAME_MAPPINGS as EPF_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] Edit Anything Prompt Fixer not loaded: {_e!r}")
    EPF_NODE_CLASS_MAPPINGS, EPF_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
try:
    from .ltx_identity_multiangle import NODE_CLASS_MAPPINGS as MA_NODE_CLASS_MAPPINGS
    from .ltx_identity_multiangle import NODE_DISPLAY_NAME_MAPPINGS as MA_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] LTX Identity Multiple Angles node not loaded: {_e!r}")
    MA_NODE_CLASS_MAPPINGS, MA_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
try:
    from .ltx_identity_gemma_vision import NODE_CLASS_MAPPINGS as GV_NODE_CLASS_MAPPINGS
    from .ltx_identity_gemma_vision import NODE_DISPLAY_NAME_MAPPINGS as GV_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] LTX Identity Gemma-Vision node not loaded: {_e!r}")
    GV_NODE_CLASS_MAPPINGS, GV_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
try:
    from .multiref_sheet_node import NODE_CLASS_MAPPINGS as MRS_NODE_CLASS_MAPPINGS
    from .multiref_sheet_node import NODE_DISPLAY_NAME_MAPPINGS as MRS_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] Multi-Ref Sheet Builder node not loaded: {_e!r}")
    MRS_NODE_CLASS_MAPPINGS, MRS_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
try:
    from .ltx_multishot_prompt import NODE_CLASS_MAPPINGS as MSP_NODE_CLASS_MAPPINGS
    from .ltx_multishot_prompt import NODE_DISPLAY_NAME_MAPPINGS as MSP_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] LTX Multishot Prompt node not loaded: {_e!r}")
    MSP_NODE_CLASS_MAPPINGS, MSP_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
# CAN / AdaLN node disabled: empirically the AdaLN modulation degrades the video (the identity
# gain came from the projector + LoRA, not the CAN). Kept the file but not registered.
CAN_NODE_CLASS_MAPPINGS, CAN_NODE_DISPLAY_NAME_MAPPINGS = {}, {}

try:
    from .bfs_h3_side_panel import NODE_CLASS_MAPPINGS as PANEL_NODE_CLASS_MAPPINGS
    from .bfs_h3_side_panel import NODE_DISPLAY_NAME_MAPPINGS as PANEL_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] H3 side panel nodes unavailable: {_e}")
    PANEL_NODE_CLASS_MAPPINGS, PANEL_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
try:
    from .bfs_shot_loop import NODE_CLASS_MAPPINGS as SHOT_NODE_CLASS_MAPPINGS
    from .bfs_shot_loop import NODE_DISPLAY_NAME_MAPPINGS as SHOT_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] Shot loop nodes not loaded: {_e!r}")
    SHOT_NODE_CLASS_MAPPINGS, SHOT_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
try:
    from .bfs_shot_noise import NODE_CLASS_MAPPINGS as SNOISE_NODE_CLASS_MAPPINGS
    from .bfs_shot_noise import NODE_DISPLAY_NAME_MAPPINGS as SNOISE_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] BFS Shot Noise node not loaded: {_e!r}")
    SNOISE_NODE_CLASS_MAPPINGS, SNOISE_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
try:
    from .minimax_h3_native_temporal import NODE_CLASS_MAPPINGS as H3TEMP_NODE_CLASS_MAPPINGS
    from .minimax_h3_native_temporal import NODE_DISPLAY_NAME_MAPPINGS as H3TEMP_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:
    print(f"[BFSNodes] H3 native temporal nodes not loaded: {_e!r}")
    H3TEMP_NODE_CLASS_MAPPINGS, H3TEMP_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
try:
    from .minimax_h3_downscaled_guide import NODE_CLASS_MAPPINGS as H3GUIDE_NODE_CLASS_MAPPINGS
    from .minimax_h3_downscaled_guide import NODE_DISPLAY_NAME_MAPPINGS as H3GUIDE_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] MiniMax-H3 downscaled guide nodes not loaded: {_e!r}")
    H3GUIDE_NODE_CLASS_MAPPINGS, H3GUIDE_NODE_DISPLAY_NAME_MAPPINGS = {}, {}

try:
    from .headswap_ltx_masked_sampler import NODE_CLASS_MAPPINGS as HSMS_NODE_CLASS_MAPPINGS
    from .headswap_ltx_masked_sampler import NODE_DISPLAY_NAME_MAPPINGS as HSMS_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] BFS Sampler not loaded: {_e!r}")
    HSMS_NODE_CLASS_MAPPINGS, HSMS_NODE_DISPLAY_NAME_MAPPINGS = {}, {}

try:
    from .bfs_lora_surgery import NODE_CLASS_MAPPINGS as LSURG_NODE_CLASS_MAPPINGS
    from .bfs_lora_surgery import NODE_DISPLAY_NAME_MAPPINGS as LSURG_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] LoRA Surgery node not loaded: {_e!r}")
    LSURG_NODE_CLASS_MAPPINGS, LSURG_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
try:
    from .bfs_video_compare import NODE_CLASS_MAPPINGS as VCMP_NODE_CLASS_MAPPINGS
    from .bfs_video_compare import NODE_DISPLAY_NAME_MAPPINGS as VCMP_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] Video Compare node not loaded: {_e!r}")
    VCMP_NODE_CLASS_MAPPINGS, VCMP_NODE_DISPLAY_NAME_MAPPINGS = {}, {}
try:
    from .bfs_segment_picker import NODE_CLASS_MAPPINGS as SEGPICK_NODE_CLASS_MAPPINGS
    from .bfs_segment_picker import NODE_DISPLAY_NAME_MAPPINGS as SEGPICK_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] Segment Picker node not loaded: {_e!r}")
    SEGPICK_NODE_CLASS_MAPPINGS, SEGPICK_NODE_DISPLAY_NAME_MAPPINGS = {}, {}

try:
    from .minimax_h3_clip_video import NODE_CLASS_MAPPINGS as H3CLIP_NODE_CLASS_MAPPINGS
    from .minimax_h3_clip_video import NODE_DISPLAY_NAME_MAPPINGS as H3CLIP_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] MiniMax H3 CLIP video context not loaded: {_e!r}")
    H3CLIP_NODE_CLASS_MAPPINGS, H3CLIP_NODE_DISPLAY_NAME_MAPPINGS = {}, {}

try:
    from .bfs_h3_native_workflow import NODE_CLASS_MAPPINGS as H3WORKFLOW_NODE_CLASS_MAPPINGS
    from .bfs_h3_native_workflow import NODE_DISPLAY_NAME_MAPPINGS as H3WORKFLOW_NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa
    print(f"[BFSNodes] H3 native workflow helpers not loaded: {_e!r}")
    H3WORKFLOW_NODE_CLASS_MAPPINGS, H3WORKFLOW_NODE_DISPLAY_NAME_MAPPINGS = {}, {}

NODE_CLASS_MAPPINGS = {
    **H3CAST_NODE_CLASS_MAPPINGS,
    **H3TOK_NODE_CLASS_MAPPINGS,
    **H3BIND_NODE_CLASS_MAPPINGS,
    **H3LOOP_NODE_CLASS_MAPPINGS,
    **H3WORKFLOW_NODE_CLASS_MAPPINGS,
    **H3CLIP_NODE_CLASS_MAPPINGS,
    **H3GUIDE_NODE_CLASS_MAPPINGS,
    **H3TEMP_NODE_CLASS_MAPPINGS,
    **SHOT_NODE_CLASS_MAPPINGS,
    **PANEL_NODE_CLASS_MAPPINGS,
    **GV_NODE_CLASS_MAPPINGS,
    **CAN_NODE_CLASS_MAPPINGS,
    **BFS_NODE_CLASS_MAPPINGS,
    **LTXV_EA_NODE_CLASS_MAPPINGS,
    **HEADSWAP_NODE_CLASS_MAPPINGS,
    **A2R_NODE_CLASS_MAPPINGS,
    **AMV_NODE_CLASS_MAPPINGS,
    **IDT_NODE_CLASS_MAPPINGS,
    **MC_NODE_CLASS_MAPPINGS,
    **MRSLOT_NODE_CLASS_MAPPINGS,
    **MA_NODE_CLASS_MAPPINGS,
    **MRS_NODE_CLASS_MAPPINGS,
    **CM_NODE_CLASS_MAPPINGS,
    **MSP_NODE_CLASS_MAPPINGS,
    **MMVAE_NODE_CLASS_MAPPINGS,
    **SHARP_NODE_CLASS_MAPPINGS,
    **SNOISE_NODE_CLASS_MAPPINGS,
    **MMDD_NODE_CLASS_MAPPINGS,
    **SVH3_NODE_CLASS_MAPPINGS,
    **EPF_NODE_CLASS_MAPPINGS,
    **HSMS_NODE_CLASS_MAPPINGS,
    **LSURG_NODE_CLASS_MAPPINGS,
    **VCMP_NODE_CLASS_MAPPINGS,
    **SEGPICK_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **H3CAST_NODE_DISPLAY_NAME_MAPPINGS,
    **H3TOK_NODE_DISPLAY_NAME_MAPPINGS,
    **H3BIND_NODE_DISPLAY_NAME_MAPPINGS,
    **H3LOOP_NODE_DISPLAY_NAME_MAPPINGS,
    **H3WORKFLOW_NODE_DISPLAY_NAME_MAPPINGS,
    **H3CLIP_NODE_DISPLAY_NAME_MAPPINGS,
    **H3GUIDE_NODE_DISPLAY_NAME_MAPPINGS,
    **H3TEMP_NODE_DISPLAY_NAME_MAPPINGS,
    **SHOT_NODE_DISPLAY_NAME_MAPPINGS,
    **PANEL_NODE_DISPLAY_NAME_MAPPINGS,
    **GV_NODE_DISPLAY_NAME_MAPPINGS,
    **CAN_NODE_DISPLAY_NAME_MAPPINGS,
    **BFS_NODE_DISPLAY_NAME_MAPPINGS,
    **LTXV_EA_NODE_DISPLAY_NAME_MAPPINGS,
    **HEADSWAP_NODE_DISPLAY_NAME_MAPPINGS,
    **A2R_NODE_DISPLAY_NAME_MAPPINGS,
    **AMV_NODE_DISPLAY_NAME_MAPPINGS,
    **IDT_NODE_DISPLAY_NAME_MAPPINGS,
    **MC_NODE_DISPLAY_NAME_MAPPINGS,
    **MRSLOT_NODE_DISPLAY_NAME_MAPPINGS,
    **MA_NODE_DISPLAY_NAME_MAPPINGS,
    **MRS_NODE_DISPLAY_NAME_MAPPINGS,
    **CM_NODE_DISPLAY_NAME_MAPPINGS,
    **MSP_NODE_DISPLAY_NAME_MAPPINGS,
    **MMVAE_NODE_DISPLAY_NAME_MAPPINGS,
    **SHARP_NODE_DISPLAY_NAME_MAPPINGS,
    **SNOISE_NODE_DISPLAY_NAME_MAPPINGS,
    **MMDD_NODE_DISPLAY_NAME_MAPPINGS,
    **SVH3_NODE_DISPLAY_NAME_MAPPINGS,
    **EPF_NODE_DISPLAY_NAME_MAPPINGS,
    **HSMS_NODE_DISPLAY_NAME_MAPPINGS,
    **LSURG_NODE_DISPLAY_NAME_MAPPINGS,
    **VCMP_NODE_DISPLAY_NAME_MAPPINGS,
    **SEGPICK_NODE_DISPLAY_NAME_MAPPINGS,
}

# Character swap with marked guides (Video Segment Picker -> H3 Character Swap Conditioning), keyframe planning, and
# the loop / sweep helpers. Each module loads on its own: a missing optional dependency only skips that module.
for _mod in ("bfs_keyframe_marks", "bfs_video_segment_picker", "bfs_h3_swap_conditioning", "bfs_keyframe_planner",
             "bfs_frame_loop", "bfs_xy_sweep", "bfs_workflow_loop", "bfs_h3_image_swap_prompt",
             "bfs_h3_image_swap_conditioning", "bfs_h3_image_swap_tools", "bfs_model_surgery",
             "bfs_h3_pdd_surgery"):
    try:
        import importlib as _il
        _m = _il.import_module(f".{_mod}", __package__)
        NODE_CLASS_MAPPINGS.update(_m.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_m.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as _e:  # noqa: BLE001
        print(f"[BFSNodes] {_mod} not loaded: {_e!r}")

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
