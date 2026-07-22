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
from .ltx_multiple_controls import NODE_CLASS_MAPPINGS as MC_NODE_CLASS_MAPPINGS
from .ltx_multiple_controls import NODE_DISPLAY_NAME_MAPPINGS as MC_NODE_DISPLAY_NAME_MAPPINGS
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
# CAN / AdaLN node disabled: empirically the AdaLN modulation degrades the video (the identity
# gain came from the projector + LoRA, not the CAN). Kept the file but not registered.
CAN_NODE_CLASS_MAPPINGS, CAN_NODE_DISPLAY_NAME_MAPPINGS = {}, {}

NODE_CLASS_MAPPINGS = {
    **GV_NODE_CLASS_MAPPINGS,
    **CAN_NODE_CLASS_MAPPINGS,
    **BFS_NODE_CLASS_MAPPINGS,
    **LTXV_EA_NODE_CLASS_MAPPINGS,
    **HEADSWAP_NODE_CLASS_MAPPINGS,
    **A2R_NODE_CLASS_MAPPINGS,
    **AMV_NODE_CLASS_MAPPINGS,
    **IDT_NODE_CLASS_MAPPINGS,
    **MC_NODE_CLASS_MAPPINGS,
    **MA_NODE_CLASS_MAPPINGS,
    **MRS_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **GV_NODE_DISPLAY_NAME_MAPPINGS,
    **CAN_NODE_DISPLAY_NAME_MAPPINGS,
    **BFS_NODE_DISPLAY_NAME_MAPPINGS,
    **LTXV_EA_NODE_DISPLAY_NAME_MAPPINGS,
    **HEADSWAP_NODE_DISPLAY_NAME_MAPPINGS,
    **A2R_NODE_DISPLAY_NAME_MAPPINGS,
    **AMV_NODE_DISPLAY_NAME_MAPPINGS,
    **IDT_NODE_DISPLAY_NAME_MAPPINGS,
    **MC_NODE_DISPLAY_NAME_MAPPINGS,
    **MA_NODE_DISPLAY_NAME_MAPPINGS,
    **MRS_NODE_DISPLAY_NAME_MAPPINGS,
}
