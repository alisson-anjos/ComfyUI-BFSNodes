# ComfyUI-BFSNodes

This repository includes general video composition helpers and the **LTXV Edit Anything** nodes required by Edit Anything LoRAs trained for LTX-Video 2.3 style `video_to_video_ref_adaln` workflows. The first public reference LoRA is expected to be `edit_anything_reference_v0.1_r128_12000.safetensors`.

## Nodes

### LTXV Edit Anything (Apply)

A unified conditioning node for LTXV Edit Anything LoRAs. It combines three conditioning paths from one LoRA checkpoint:

- **IC-LoRA sequence conditioning**: appends a reference image and optional guide video frames as clean conditioning frames.
- **Role embedding**: injects the learned reference-role token embedding before `patchify_proj`, when the LoRA contains role embedding weights.
- **AdaLN reference conditioning**: pools the encoded reference image and injects a global appearance/style condition into the LTXV timestep path, when the LoRA contains the AdaLN projector weights.

Use this node when a LoRA was trained with the Edit Anything / `video_to_video_ref_adaln` strategy and requires reference-image conditioning beyond a standard LoRA load.

### LTXV Apply Neutral Mask

Replaces masked-out regions in an image or video batch with a clean solid background (`white`, `neutral_gray`, or `black`). This is useful for preparing reference images or guide frames before passing them to Edit Anything conditioning.

### LTXV Resize Reference By Mask

Resizes a reference object onto a clean canvas using a mask bounding box as the target object size. This helps match the scale expected by the guide/video workflow while keeping the reference image clean.

### Video Composition Nodes

The repository also includes the existing BFS video composition utilities:

- **Reserved Region Frame Composer**
- **Frame Ranged Face Loader**
- **Face Sequence Batch**

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alisson-anjos/ComfyUI-BFSNodes.git
```

Install the Python dependencies if your ComfyUI environment does not already provide them:

```bash
cd ComfyUI-BFSNodes
pip install -r requirements.txt
```

Optional: `pip install insightface` with `onnxruntime` or `onnxruntime-gpu`, only for LTX Identity. The Shot Planner's
*Find people* / Cast no longer needs it (1.93.0): it uses OpenCV's YuNet + SFace (Apache-2.0, downloaded once), and uses
InsightFace only when it is installed. They are not in the requirements because insightface pins packages
that conflict with many environments; without them everything else works and those features say what to install.

Restart ComfyUI after installation.

## LoRA Placement

Put your Edit Anything LoRA checkpoint in ComfyUI's LoRA folder, for example:

```text
ComfyUI/models/loras/
```

Subfolders are supported by ComfyUI, so organized paths such as this are fine:

```text
ComfyUI/models/loras/ltx-2/2.3/edit_anything/edit_anything_reference_v0.1_r128_12000.safetensors
```

## Basic Workflow

The expected graph order is:

```text
LTXV model loader
  -> standard ComfyUI Load LoRA
  -> LTXV Edit Anything (Apply)
  -> KSampler / sampler node
```

Connect the node inputs as follows:

- `model`: the model output after the standard LoRA loader.
- `positive` / `negative`: your conditioning.
- `vae`: the LTXV VAE.
- `latent`: the target video latent.
- `ref_image`: the appearance/reference image.
- `lora_name`: the same Edit Anything LoRA checkpoint.
- `guide_frames` optional: frames used for motion or structure guidance.

The node returns the patched model, updated positive/negative conditioning, updated latent, and a preview of the resized reference image.

## Recommended Starting Values

Start with:

```text
guide_strength: 1.0
ref_strength: 1.0
role_strength: 1.0
adaln_scale: 1.0
enable_adaln: true
enable_role_embedding: true
resize_mode: pad_to_fit
```

If the model ignores the reference image, try increasing `role_strength`. If the output copies color/texture too strongly, reduce `adaln_scale`.

## Compatibility Notes

These nodes target ComfyUI's LTXV/LTX-Video model implementation. The main apply node monkey-patches LTXV internals at runtime so the reference-role embedding and AdaLN conditioning match the training-time conditioning path.

A plain LoRA loader alone is not enough for Edit Anything LoRAs that include role embedding and/or AdaLN reference conditioning. Use the standard LoRA loader first, then pass the result through **LTXV Edit Anything (Apply)**.

## Reference temporal offset

Nodes that inject clean appearance references expose a reference-only temporal offset in VAE
latent steps. The default is `0`, matching the standard frame-zero placement used by existing
LoRAs and preserving their expected behavior. Negative values are an experimental way to reduce
the overlap frame-0 artifact where the reference composition briefly appears at the start of the
generated video.

- `0`: standard frame-zero placement (default)
- `-1`: one latent step before frame zero (`-8` pixel frames with the standard LTX VAE)
- `-2`: two latent steps before frame zero (stronger separation, possibly weaker identity)

The offset is applied only to appearance/reference inputs. Guide videos, masks, identity masks,
and target video positions remain unchanged. In **LTX Multiple Controls**, the field is named
`identity_temporal_offset_latents` and affects only `identity_image`.

## Shot Planner / Shot Loop (1.48.0)

Split a long video into shots a video model can follow (at camera cuts via PySceneDetect, fixed
length, or by hand on a timeline), set a reference image and prompt per shot, run the rest of the
workflow once per shot, and join the results back with the original timing and soundtrack.

- **BFS Shot Planner** — interactive timeline (thumbnails, cut detector curve, draggable
  boundaries, per-shot reference gallery and prompt). Outputs the shots as a ComfyUI list, so
  every node downstream runs once per shot.
- **BFS Shot Unpack** — guide frames, references, prompt and length of one shot, for any model.
- **BFS Shot H3 Conditioning** — MiniMax H3 conditioning for one shot with the native nodes
  (aligned guide, native reference video, or aligned guide + CLIP video context).
- **MiniMax H3 Reference to Video — Guide 0 CLIP Context (BFS)** — lets Qwen see the video
  guide at 2 fps without native guide video-reference latents; use Add Guide separately for motion.
  See [H3 guide modes](MINIMAX_H3_GUIDES.md#video-context-in-clip-with-aligned-guide-latents).
- **BFS Shot Join** — concatenates, trims to the true lengths, cross-fades soft joins, returns audio. With `comparison` on, its `guide` output is the original video frame for frame.
- **BFS Video Compare** — one player for two videos (Join's `guide` and `images`): wipe ↔ / ↕, overlay or side by side, any of guide / target / difference on top, plus a per-pixel difference map and the difference per frame (click to jump). 🖼 PNG downloads the frame on screen as shown; with two images (one frame) ⬇ save PNG downloads it from lossless PNGs at the compared size.

**Cast (1.49.0)**: *Find people* tracks faces through the video (InsightFace) and groups them
into people. Link each person to a reference and every shot uses the reference of the person
with the most screen time in it; optionally split where the main person changes and run only the
shots where a linked person appears.

Two run modes: *auto loop* (all shots in one run) and *queue loop* (one shot per run, stored on
disk, re-queued automatically; nodes after the join only run on the last shot). See
[`web/js/docs/BFSShotPlanner.md`](web/js/docs/BFSShotPlanner.md).

Shots can also be cropped to a **SAM 3 mask** (text or points per shot) and pasted back by the join, continue
from the previous shot's result, and get **VLM suggestions** (what to segment, a shot description for the
prompt, run/skip) from a Qwen3-VL connected to the planner.

## H3 Duet / Side Panel (1.50.0)

Training-free split-screen generation for MiniMax H3 (TSC's latent pin): a panel (the source clip
or a reference) is pinned beside the video with a noise mask, the video is generated in sync with
it, and the panel is cropped off before decoding. **BFS H3 Duet** does it all in one node, **BFS Shot H3 Duet** renders each shot of the Shot Loop that way (or with an aligned guide for body-swap LoRAs);
**BFS H3 Side Panel** + **BFS H3 Side Panel Crop** are the building blocks. Optional aligned latent
guide. 1.84.0: `panel_static` (TV static over the face / person of the pinned clip, SAM 3) and a `setting`
picture (**BFS Setting Picture**) for same-world swaps. See [`web/js/docs/BFSH3Duet.md`](web/js/docs/BFSH3Duet.md).

## Requirements

- ComfyUI
- PyTorch
- safetensors
- An LTXV/LTX-Video model and compatible VAE
- An Edit Anything LoRA trained for this conditioning strategy

## License

This project follows the repository license. See [LICENSE](LICENSE).

## MiniMax-H3 downscaled latent guides (1.46.0)

**MiniMax-H3 Downscaled Latent Guide (BFS)** supports aligned lower-resolution
image/video guides for H3 LoRAs trained with `target_grid_stride_v1`. It encodes
the smaller guide and patches only the returned model's packed coordinates.
Connect both its **MODEL** and **positive CONDITIONING** outputs to sampling.

**MiniMax-H3 Guide Target — Image / Video (BFS)** creates an exact output canvas:
use one frame for images or 73 frames for approximately three seconds at 24 fps.
At output 1024 x 768 and factor 4, the guide preview is 256 x 192.

See [setup, workflow template, compatibility and validation](MINIMAX_H3_GUIDES.md).

BFSNodes **1.47.0** adds optional MiniMax-H3 overlap/sidecar source-phase RoPE and
**MiniMax-H3 Identity Reference + RoPE (BFS)**. See [the H3 guide](MINIMAX_H3_GUIDES.md#experimental-reference-layouts-and-source-phase-1470).

Experimental native temporal motion-transfer inference (standalone node and Shot Conditioning mode): see [H3_NATIVE_TEMPORAL.md](H3_NATIVE_TEMPORAL.md). Requires the matching experimental LoRA; preserves native image references and applies temporal RoPE alignment plus soft temporal attention.

### H3 Flash Partitions

[Illustrated explanation: native temporal alignment with compile-free Flash Partitions](docs/H3_FLASH_PARTITIONS.md) covers the exact temporal rule, positive partition decomposition, log-sum-exp merging, measured performance, guide scaling and current backend limitations.

Experimental temporal mode automatically recognizes **Model Attention Backend** (Kitchen/PyTorch) and, from 1.115.14, **Model Sparse Attention → sol-attn**. Version 1.115.22 automatically selects verified prebuilts for Linux SM80/86/89/90/100/120 and Windows SM86/89/120 (x86_64, CUDA13-capable driver); cross-platform GPU validation remains experimental. Version 1.115.16+ can use the [one-pass Kitchen extension](H3_NATIVE_TEMPORAL.md#one-pass-native-kitchen-kernels-111516) for dense INT8 and sol-attn. The earlier compatibility path repeats frame-wise calls and was slower than Flash Partitions; see [setup, measurements and limitations](H3_NATIVE_TEMPORAL.md#model-sparse-attention-sol-attn-compatibility-111514). SLA/VSA remain unsupported.

![H3 temporal weight decomposition](docs/illustrations/h3_temporal_weight_decomposition.png)

### Native H3 auto-prompt workflow

[Automatic reference descriptions, six-section prompt writing and source-derived video size/frame count](H3_NATIVE_AUTO_PROMPT.md). Includes [visual workflow](examples/h3_tara_native_auto_prompt.comfy.json) and [API graph](examples/h3_tara_native_auto_prompt.api.json), with native Generate Text stages and public BFS helpers; no private benchmark nodes.

TARA kernel downloads, including **RTX4090 (SM89), H100 (SM90) and RTX5090 (SM120)**: [direct download table and setup checks](H3_NATIVE_TEMPORAL.md#kernel-downloads).

[Animated TARA parameter guide](TARA_PARAMETERS.md): free window, falloff, strength and cap, with an offline interactive version.

### Native reference image preview (1.115.25)

**BFS H3 Describe References (Generate Text per Picture)** adds a third output,
`reference_images` (`IMAGE` list). Connect it to **Preview Image** to view every
attached reference in Picture 1..N order. Original sizes and pixels are preserved;
images of different sizes are not resized into a batch. The existing reference
bundle and text outputs retain their indices. This preview connection does not
change which images are sent to the encoder.

### Native shot loop without Planner (1.115.27, experimental)

Separate Start, Continuity and End/Join nodes now have a [native auto-prompt loop example](examples/h3_tara_native_auto_prompt_loop.comfy.json). See [wiring, manual cuts, continuity and validation limits](H3_NATIVE_SHOT_LOOP.md).

### Reference Attention Binding (1.115.28, experimental)

A separate model node adds per-Picture attention gains with optional tracked target
masks, composed with TARA. See [controls and limitations](H3_REFERENCE_BINDING.md)
and [example workflow](examples/h3_tara_reference_binding.comfy.json). The initial
prototype uses Flash Partitions; it does not yet use fused Kitchen reference gains.

### Click-to-Picture SAM 3.1 bindings and token weights (1.115.29, experimental)

Pause the guide and select a performer for Picture X, track it with native SAM 3.1,
and apply regional reference attention. A separate token/word editor offers
clickable forces with embedding-magnitude and text-attention modes. See
[workflows, setup and validation limits](H3_CAST_AND_TOKEN_CONTROLS.md).

### Phrase weights in prompts (1.115.30)

The Token / Word Weight Editor accepts `(phrase:1.3)` directives in
`weighted_phrases`, or annotations parsed before native encoding. See
[syntax, exact matching and override rules](H3_CAST_AND_TOKEN_CONTROLS.md#phrase-weights-in-the-prompt-111530).

### Optional SAM / loop workflow (1.115.31)

[One workflow](examples/h3_tara_optional_sam_loop.comfy.json) now has independent
SAM 3.1 and shot-loop switches. Both default OFF. See
[setup, disabled branches and status](H3_CAST_AND_TOKEN_CONTROLS.md#one-workflow-with-optional-sam-and-loop-111531).

### BFS Segment Picker (1.116.0)

Segment an image into objects with SAM 3.1 and pick them with the mouse. Connect a
Load Checkpoint node with `sam3.1_multiplex_fp16.safetensors` (its MODEL and CLIP),
press **▶ Segment** (runs only this node) and the objects appear on a canvas in the
node: hover to see one, click to select it. The automatic pass looks for the
`concepts` (default `person, animal`); add what it missed with **Find** (a SAM 3 text
prompt) or with keep / exclude **points**. Outputs: the selection's mask, one mask per
selected object, the image with the selection tinted (`mark_color` at `mark_opacity`,
magenta at 0.3 by default, the Colored Mark convention), a numbered overlay and the
objects as JSON. The selection and the added objects are saved with the workflow and
segmented again on load.

Hand edits (1.118.0): **🖌 Brush** paints extra area into the mark (e.g. hair SAM left
out), **➖ Remove** cuts out what SAM segments at the points you place, and **🧽 Eraser**
paints what to cut out; **shift** while painting does the opposite of the tool (the brush
erases, the eraser paints back). Each becomes a chip; unselect it to switch it off. Brush and
eraser strokes belong to the image they were painted on and are dropped when the image
changes. Fixed: clicking inside the node no longer shrinks its canvas while the side
panel is open.

### Character swap with marked guides (1.119.0)

Mark who is replaced in the guide video, then condition MiniMax H3 Character Swap on it.

**BFS Video Segment Picker (SAM 3)** — the Segment Picker for a video. A strip shows
frames of the batch; click a frame, pick the subject tab (1 red, 2 blue, 3 green,
4 yellow = Picture 1–4) and click the person. **↔ Propagate** tracks every subject from
its picked frames to the whole video with SAM 3, shot by shot (camera cuts are detected
and never tracked across). Optional `molmo_clip` (Molmo2 H3 encoder): Molmo2 describes
the picked person, follows them through the video and re-anchors shots that have no
pick, so a person is found again after a cut. Picking runs only segment the strip;
tracking, Molmo2, MoGe and DWPose run on Propagate. Repeated runs reuse what did not
change (per subject and per shot). After the first Propagate, picking on a frame
propagates by itself (**auto**). The **Whole video** scrubber shows every frame with
its masks, the lost stretches in red; **＋ add this frame** puts any frame in the strip
to fix it. A status line with a progress bar and a log shows what runs.

Hand fixes on a picked frame (1.121.1): **🖌 Brush** paints into the active color's
mark and **shift** while painting erases from it (e.g. a stray piece of mask over a
hand). Strokes not applied yet are applied when you change color or frame, or press
Apply fix, so they are never lost. What any color adds on a frame (brush, points, text)
is shared by every color there: each color selects from the same objects, and ✕ on an
added object removes it for all of them.

With an object selected, the brush edits that object's own mask (1.121.2): a stroke grows
the selected mask it starts on (or the nearest selected one) and shift trims it, with no
new object on top. The chip shows ✎; ↺ drops the edits.

`clean_masks` cleans the subject masks where people touch (arms around each other, hands
crossing): on every frame, and on the picked frames before tracking, any piece smaller
than that % of its subject's area joins what surrounds it. A speck of blue on the red
arm becomes red, a hole in a mask is filled, a stray bit on the background goes away;
each subject's main body always stays. 0 = off; 0.5–2 is usually enough.

Outputs: `masks`, `marked` (tinted), `overlay`, `report`, and `hybrid`: the guide
with each subject drawn in its color, by `mark_style`:
- `normal map` — its MoGe normal map in gray detail, tinted (connect `normals`:
  Run MoGe Inference → Render MoGe Geometry, a normal mode);
- `openpose + face` — erased to gray (`pose_region`: convex hull or person outline,
  `pose_gray_opacity`) and drawn as its DWPose skeleton with the eyes and the mouth
  opening (connect `pose_keypoint`: DWPose Estimator with body, hands and face);
- texture-free alternatives: flat color, shading, blur, pixelate, gray detail, edges.

`hybrid_grow` sets the margin around each subject; `hybrid_base` stacks pickers.
`free_after` unloads the models from the GPU when the node is done (default: all).

**BFS H3 Character Swap Conditioning** — `mode`:
- `keyframe pins`: the source video plus one Add Guide per edited keyframe at its own
  frame (`keyframes` + `keyframe_spec` from BFS Keyframe Planner, or `keyframe_N` /
  `time_N`). `keyframe_until` drops the keyframes (or every guide) below that sigma on
  the returned `model`.
- `normal map guide` / `DWPose guide`: the Video Segment Picker's `hybrid` as the guide.

Pictures 1–4 are the references in color order; the prompt is written from the mode
and the number of Pictures (or `prompt_override`). Outputs: model, positive, latent,
prompt, keyframe_video and `pictures` (the Pictures as a list, e.g. for BFS Video
Compare's `reference`).

Also: **BFS Keyframe Planner** (choose the keyframe frames on the video and export
them with their spec), **BFS Keyframe Marks** / **BFS Track From Frame (SAM 3)** (a
tracked person's mask at each keyframe), **BFS Workflow Loop** (run a part of the
graph once per item, injecting the item and changing widgets per iteration),
**BFS XY Sweep** (a grid of any two widgets' values) and **BFS Frame Loop Start/End**.

### MiniMax H3 Image character swap

**Auto Subjects:** connect the Picker's `hybrid` and `report` outputs to **BFS H3 Image Character Swap • Auto Subjects**, then connect references to the fixed red, blue and green slots. The node reads the actual selected colors; no manual subject count is needed. The guide is Picture 1 and active references follow from Picture 2 in color order. Blue or green may be used without red. Leave `prompt_override` empty for the generated prompt, or provide a complete replacement prompt; reference selection remains automatic.

**BFS Optional Reference Image** loads a reference or returns no image for `[none]`. Both the target `megapixels` and interpolation `scale_method` are editable; 4 MP + Lanczos are defaults. Aspect ratio is preserved, and empty references skip resizing. Native H3 encoding still applies its own reference-size limits.

The Picker's **colored hatching + contour** mode preserves the source pixels between colored lines, with `hatch_width` and `hatch_spacing` controls. MoGe is not executed for hatching. For normal-map guidance, connect rendered MoGe normals to the Picker and keep its normal convention consistent. The automatic prompt follows the Picker mode. **BFS H3 Image Swap Comparison** shows only active references and labels their actual Picture numbers.

Requires **ComfyUI 0.45.0+** and the H3 Image model, image VAE and matching text encoder.

Use **BFS H3 Image Character Swap Conditioning** with the Picker's single-image `hybrid` as `guide_image`. Connect character references in red, blue and green order. The generated prompt treats the guide as **Picture 1** and the characters as **Pictures 2–4**. Connect `positive` and `latent` to the sampler; the node also outputs the prompt for inspection. Normal map, DWPose, translucent overlay and outline guides are supported.

**Add Guide for MiniMax H3 Image (BFS)** is an optional experimental alignment step. Connect the model, positive conditioning, native image latent, image VAE and marked guide; use its returned MODEL and positive with the sampler. It aligns the guide's frozen reference positions to the target canvas. It does not use the video/audio Add Guide latent format. Structural checks passed; visual equivalence to a video-trained guide LoRA is not guaranteed.

For **MolmoPoint-8B** in Video Segment Picker, connect `point_model` from the loader in [ComfyUI-Molmo2-H3](https://github.com/alisson-anjos/ComfyUI-Molmo2-H3), select `MolmoPoint-8B` in `propagation_method`, and use **Apply fix** with the desired repair scope. SAM 3 still produces the masks.

### Model surgery

**BFS Model Surgery** takes a MODEL and returns a cloned MODEL with reversible weight scaling. Execute once to populate the layer panel; filter names and add scale rules. Rules are a JSON list using `pattern` (glob by default, or `regex: true`), `scale`, optional `enabled`, and optional `rows: [start, exclusive_stop]`. Overlapping rules multiply in list order. Empty rules preserve the model. Connect after LoRA loaders to scale the materialized weights, including their preceding patches. This scales the model weights themselves, rather than just the LoRA delta; CLIP and VAE are separate.

**BFS H3 PDD Surgery** provides experimental shared-base and per-offset controls for native H3 PDD output heads. All strengths at 1 preserve the original model. Setting offsets to zero does **not** recover a pre-distillation checkpoint, and sigmas stay unchanged.
