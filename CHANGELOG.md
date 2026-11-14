# Changelog

All notable changes to **ComfyUI-BFSNodes** are documented in this file.

## [1.122.0] - 2026-11-14

### Changed
- **Shot Planner panel reorganized**: four tabs in working order (Video → Pictures & prompt → Shots → Run). The shot editor's selection is one list under **What changes**: *Subject 1* is the shot's own selection and, with Colored Mark on, each row says which picture it becomes (**→ Picture N**); **+ Add subject** adds rows. The five generation-mode cards are one **Generate** dropdown. Rare settings (cut detector, frame grid, fps, VLM, picture descriptions, memory, second pass, mask settings) are collapsed. Shorter labels and tooltips; Portuguese, Spanish and Chinese updated.

### Added
- **Shot Planner Colored Mark styles**: *translucent* (as before), *hatching + contour* and *normal map*, the BFS Video Segment Picker styles. The normal map takes MoGe normals of the shot's frames from the new optional `normals` input of BFS Shot H3 Conditioning (both nodes). The prompt names the style of the marks.
- **Shot Planner MolmoPoint**: optional `point_model` input (MolmoPoint-8B). While connected, a text selection is pointed by MolmoPoint and SAM 3 tracks from that point, so phrases like "the man on the left" pick the right subject. Falls back to SAM 3 text when MolmoPoint finds nothing.

### Removed
- **Shot Planner shot filters and face cast**: *Analyse people & faces*, the person/face/dark/static filters, the per-shot *filters decide / always run / always skip* selector, *Find people*, *Split where the main person changes* and *Only run shots with a detected person*, with their server routes. Use **⏸ Disable** to turn a shot off. Saved plans still load: a shot skipped by hand opens disabled, and the old filter and cast settings are ignored.

## [1.121.3] - 2026-11-14

### Fixed
- **BFS Shot H3 Conditioning (`BFSShotH3ConditioningSimple`)**: Reset newly added number and toggle widgets when loading workflows saved with the older `use_ref_2` layout, preventing shifted values such as `blur` from being parsed as `clip_video_fps`.

## [1.121.2] - 2026-11-13

### Added
- **Segment Picker / Video Segment Picker**: with an object selected, the brush edits that object's mask instead of adding a new object on top: a stroke grows the selected mask it starts on (or the nearest selected one), shift trims it. The chip shows ✎ and ↺ drops the edits. The brush stays on after ✓ Apply painted area.
- **Video Segment Picker**: `clean_masks` (default 0 = off) cleans the masks where subjects touch: on every frame, and on the picked frames before tracking, any piece smaller than that % of its subject's area joins what surrounds it (stray specks of one color on another subject switch to it, holes are filled, stray bits on the background go away). A subject's largest piece always stays; overlaps elsewhere are left as they were.

## [1.121.1] - 2026-11-13

### Fixed
- **Video Segment Picker**: a region painted with the brush no longer disappears when you switch to another color and back. Strokes not applied yet are applied to their color before the color or frame changes or a run starts, and the tool and brush size stay selected.

### Changed
- **Video Segment Picker**: objects added on a frame (brush, points, text) are shared by every color there; each color selects its own. Removing an added object removes it for every color.
- **Segment Picker / Video Segment Picker**: hold **shift** while painting to do the opposite of the tool: the brush erases from the mark, the eraser paints back. Erasing and painting over the same spot keeps whatever was painted last.

## [1.121.0] - 2026-11-12

### Added
- **H3 Image Auto Subjects conditioning**: reads the selected colors and guide style from the Picker report, includes only the corresponding character references and generates a matching prompt. Supports non-contiguous colors, an optional full `prompt_override`, and a blocked first pass while the user makes selections.
- **Optional Reference Image**: an unused `[none]` slot skips loading/resizing. Active images use native proportional resizing with editable `megapixels` and `scale_method` controls (defaults: 4 MP, Lanczos).
- **H3 Image Swap Comparison**: shows the original, marked guide, active references with their Picture numbering, and result; unused reference branches remain lazy.
- **H3 Image Character Swap Prompt**: standalone prompt generation, including the new hatching format.
- **Model Surgery**: inspect model parameter names and shapes, filter layers, and apply reversible scaling rules by glob, regex or row range using native ModelPatcher patches. Includes a browser panel.
- **H3 PDD Surgery**: experimental independent scaling of the shared output head and PDD offset blocks, without changing sigmas or allocating full-size deltas. It does not undo distillation.

### Changed
- **Video Segment Picker**: adds colored hatching with a matching contour, editable line width/spacing and protected-object handling. The existing report output now includes the active subject IDs and mark style for automatic downstream conditioning.
- Register the new modules independently so unavailable optional features do not prevent the rest of BFSNodes from loading.

## [1.120.0] - 2026-11-10

### Added
- **BFS H3 Image Character Swap Conditioning**: native H3 Image conditioning and automatic still-image prompts. The guide is Picture 1; red, blue and green references are Pictures 2, 3 and 4. Supports normal maps, DWPose, overlays and outlines. Requires ComfyUI 0.45.0 or later.
- **Add Guide for MiniMax H3 Image (BFS)**: experimental frozen guide reference aligned to the target image positions through a cloned ModelPatcher. Uses native H3 Image latents without an audio stream.
- **Video Segment Picker**: optional `point_model` input and MolmoPoint-8B tracking method. Connect the loader from ComfyUI-Molmo2-H3.
- Protected-object selections, a normal-map-inside-mask mode with feathered edges, translucent overlays and colored outlines.

### Fixed
- Bound picker mask caches by memory size, keep cached masks as boolean tensors and preserve existing masks when filling empty stretches.
- Track empty stretches and nearest-anchor windows instead of repeatedly recomputing the full clip.
- Correct SAM inference tensor handling for point edits, rendering invalidation, normal-map lazy input requests and Apply fix messages.
- Stabilize compare preview sizing; expose memory cleanup after comparison.

## [1.119.5] - 2026-11-09

### Changed
- **BFS Video Segment Picker (`BFSVideoSegmentPicker`)**: Restored the classic Molmo2 full-video text tracking as the default, with the current SAM 3 tracking available in a dropdown.
- Added repair scopes for one frame, the active frame onward, recomputing all frames, and filling only empty masks. Filling empty masks is the default and preserves existing non-empty results.
- Preserve the last propagated preview while adding selections, disable auto-propagation by default, and hide the manual eraser in the video picker while retaining SAM-guided remove points.
- Shortened the package description to fit Comfy Registry metadata validation.

## [1.119.4] - 2026-11-09

### Added
- **BFS Video Segment Picker (`BFSVideoSegmentPicker`)**:
  - **Automatic Unmarking / Subject Limiting (`max_subjects`)**: Added `"max_subjects"` parameter (`["auto", "1", "2", "3"]`, default `"auto"`). In `"auto"` mode, the node dynamically inspects the execution workflow graph (`prompt`) to count connected reference pictures (`picture_N`) on downstream conditioning nodes (e.g. `BFSH3KeyframeConditioning`). If only 2 pictures are connected, Subject 3 (`3 · green`) is automatically excluded from tracking, masking, and hybrid guide rendering, preventing unmatched green skeletons or normal maps from reaching MiniMax H3.
  - **One-Click Subject Clear (`✕`)**: Added a clear button `✕` on each subject badge (`1 · red`, `2 · blue`, `3 · green`) in the UI header. Clicking it instantly removes all picks for that subject across all frames and triggers auto-propagation.
  - **Explicit Text Tracking (`molmo_descriptions`)**: Added optional `"molmo_descriptions"` multiline input (line $k$ = Subject $k$). When provided, Molmo2 tracks using the exact text descriptions across camera cuts and shots instead of guessing from low-resolution crops, avoiding tracking errors.

---

## [1.119.3] - 2026-11-09

### Changed
- **BFS Video Segment Picker (`BFSVideoSegmentPicker`) & BFS H3 Character Swap Conditioning (`BFSH3KeyframeConditioning`)**:
  - Limited selectable subjects and reference pictures to 3 (`1 · red`, `2 · blue`, `3 · green` / `picture_1`..`picture_3`) to match the 3-subject training distribution of the Normal Map and DWPose Character Swap LoRAs.

---

## [1.119.2] - 2026-11-09

### Fixed
- **BFS H3 Character Swap Conditioning (`BFSH3KeyframeConditioning`)**:
  - Added runtime VRAM estimation and peak activation patching (`_patch_vram_estimate`) onto the `MiniMaxH3` model instance so ComfyUI accounts for `minimax_keyframes` (`guide_video`), `minimax_refs` (`Picture 1`–`Picture 4`), and `cross_attn` tokens when computing `minimum_memory_required`, and releases intermediate `q, k, v` and `h` buffers early in `DiTBlock` / `Attention` without modifying ComfyUI core.
  - Cleared cached DWPose detector/pose sessions (`global_cached_dwpose`) safely in `_free_gpu()` without triggering lazy-import errors on unrelated modules in `sys.modules`.
- **BFS Video Segment Picker (`BFSVideoSegmentPicker`)**:
  - Separated multi-subject DWPose guide rendering into two passes (erase all subject regions to neutral gray first, then draw all subject skeletons on top) so overlapping gray regions of adjacent/hugging subjects never erase each other's skeletons.
  - Added per-frame DWPose matching exclusion (`exclude`), relaxed overlap fallback (`min_inside=0.25`), and neighbor-frame pose fallback (`±2` frames) so closely interacting subjects do not lose their skeleton on intermediate frames.

---

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
