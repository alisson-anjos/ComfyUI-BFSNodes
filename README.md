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
  (aligned guide and/or native reference video).
- **BFS Shot Join** — concatenates, trims to the true lengths, cross-fades soft joins, returns audio. With `comparison` on, its `guide` output is the original video frame for frame.
- **BFS Video Compare** — one player for two videos (Join's `guide` and `images`): wipe ↔ / ↕, overlay or side by side, any of guide / target / difference on top, plus a per-pixel difference map and the difference per frame (click to jump).

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

### Experimental Planner Agent

The `feat/planner-agentic` branch adds an automatic tool agent to Shot Planner, with SAM masks, visual review, optional H3 intermediate previews, web/directory context and pause/resume controls. See [setup, tools and local GPU validation](docs/PLANNER_AGENT.md).
