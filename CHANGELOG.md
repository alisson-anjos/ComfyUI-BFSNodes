# Changelog

All notable changes to **ComfyUI-BFSNodes** are documented in this file.

## [1.119.1] - 2026-11-09

### Fixed
- **BFS Video Segment Picker (`BFSVideoSegmentPicker`)**:
  - Prevented subjects picked on later frames from being falsely propagated or re-anchored onto other people in earlier frames before `first_pick`.
  - Restricted Molmo2 tracking (`molmo_relocate`) to search strictly forward from `first_pick` (`image[first_pick:]`), highlighting the selected person on the initial frame (`t=0.0s`) and dims background distractions.
  - Added visual candidate verification (`molmo_verify_candidate`): compares the reference person crop (`Picture 1`) against each candidate person crop (`Picture 2`) with Molmo2, checks RGB/chroma histogram similarity (`person_color_sim`), and rejects candidates that overlap masks already claimed by other subjects (`claimed_at`).
  - Bounded backward SAM 3 tracking before `first_pick` to the immediately preceding strip frame (`prev_strip`) with frame-to-frame spatial (`IoU >= 0.15`), non-overlap, and color continuity checks.
  - Enforced `gc.collect()` and `mm.soft_empty_cache(True)` when freeing GPU models after segmentation/tracking.
- **BFS H3 Character Swap Conditioning (`BFSH3KeyframeConditioning`)**:
  - Added `_free_gpu()` to explicitly unload `MiniMaxH3TEModel_` (15 GB) and `MiniMaxH3VideoVAE` (5 GB) and flush the CUDA cache after `MiniMaxH3ReferenceToVideo` and `MiniMaxH3AddGuide`, preventing partial-unload VRAM retention and CUDA OOM when `MiniMaxH3` (20 GB) starts sampling.

---

## [1.119.0] - 2026-11-09

### Added
- **BFS Video Segment Picker (`BFSVideoSegmentPicker`)**:
  - Interactive multi-subject (1–4) SAM 3.1 video segmentation and shot-by-shot propagation with automatic camera cut detection.
  - Optional Molmo2 re-anchoring across camera cuts and lost stretches, with per-shot and per-subject caching.
  - Interactive timeline scrubber displaying per-frame subject coverage and lost stretches.
  - Hybrid guide video output supporting `normal map`, `openpose + face` (DWPose skeleton + face landmarks over neutral gray), and texture-free styles (`flat color`, `shading`, `blur`, `pixelate`, `gray detail`, `edges`).
  - Configurable `pose_region` (`convex hull` or `person outline`) and `pose_gray_opacity`.
- **BFS H3 Character Swap Conditioning (`BFSH3KeyframeConditioning`)**:
  - Unified conditioning node supporting `keyframe pins`, `normal map guide`, and `DWPose guide` modes with up to 4 reference pictures (`Picture 1`–`Picture 4`) mapped to red, blue, green, and yellow subject marks.
- **Keyframe & Loop Helpers**:
  - Added `BFSKeyframePlanner`, `BFSKeyframeMarks`, `BFSTrackFromFrame`, `BFSWorkflowLoop`, `BFSXYSweep`, and `BFSFrameLoop`.

---

## [1.118.1] - 2026-11-07

### Fixed
- **Panel Server Routes**: Wrapped interactive panel button model executions under `torch.inference_mode()` to match standard ComfyUI node execution.

---

## [1.118.0] - 2026-11-04

### Added
- **BFS Segment Picker (`BFSSegmentPicker`)**: Added interactive brush strokes, negative/remove points, and eraser mode on the segmentation canvas.

---

## [1.117.0] - 2026-11-03

### Added
- **BFS Video Compare (`BFSVideoCompare`)**: Added current-frame PNG download button in the comparison UI and automatic PNG output when comparing single images.

---

## [1.116.0] - 2026-11-03

### Added
- **BFS Segment Picker (`BFSSegmentPicker`)**: Interactive SAM 3.1 object picker canvas with automatic concept detection, text search, and point prompts.
