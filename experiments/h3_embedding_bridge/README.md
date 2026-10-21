# Experimental Qwen3.5 9B thinking states -> H3 conditioning

Branch: `experiment/h3-9b-thinking-embeddings`. No changes to main or the live
OpenJutsu training run. This is a prepared research prototype, not a trained
encoder replacement. GPU execution and real workflow validation are pending.

## What is actually tested

The full Qwen3.5-9B sees numbered pictures and chronological guide frames, plus
an unchanged copy of the request. It generates with thinking off/on. A forward
hook captures its last decoder-block activations BEFORE final normalization:
one prefill bank and a separate bank of generated-token states. The sampled
last token may not have been forwarded. Reasoning text is not rewritten into an
H3 prompt, not saved, and not fed through the teacher. Token states themselves
are the bridge input. A completed `</think>` is required in thinking mode;
truncated thinking is not silently labelled successful.

A shared trainable bridge consumes these states. Position/modality queries
produce a variable-length sequence with the original H3 teacher's exact layout,
5120-wide unnormalized features and visual/text tags. The bridge never sees
teacher feature values as INPUT. Only its reconstruction loss uses them.
The 9B and H3 encoders stay frozen; H3 DiT/VAE are not loaded for calibration.
Teacher is the existing converted H3 encoder (50 layers), so no 66-GB full
teacher download is needed.

This prototype requires a cached teacher layout for each evaluation case. It is
NOT yet a general online encoder, a speedup demonstration, or an identity/motion
quality improvement. Distilling ordinary H3 embeddings may suppress useful
thinking information. Embedding error alone cannot prove thinking benefits.

## What is ready and what is not

Prepared: disjoint input manifest, teacher/student cache scripts, a modality-
balanced shared resampling bridge, train/validation/test reconstruction metrics,
held-out conditioning exports, and a guarded Comfy cached-case embedding swap.
Only syntax and CPU file preflight are checked. No tensor, GPU, generation or
quality tests have been run, per the user's local testing preference. No bridge
weights have been trained/exported. No new model download or environment upgrade
is performed automatically. GPU entrypoints refuse to run while the dataset
reviewer is active or GPU memory is occupied; there is no auto-start watcher.

Full local Qwen3.5-9B weights and a compatible Transformers environment are
required. Existing Comfy caption-model support does not guarantee a matching
Transformers model directory. Keep the experimental environment separate from
Comfy/training; use current official Qwen requirements rather than upgrading
the active environments. Generation uses a complete HF model, not GGUF and not
the headless H3 encoder.

## CPU preparation (safe during dataset processing)

```bash
cd /data/bfs-dev-vlm-reasoning/experiments/h3_embedding_bridge
python make_manifest.py --output /data/reports/h3_9b_embedding_bridge/manifest.jsonl
python preflight.py --manifest /data/reports/h3_9b_embedding_bridge/manifest.jsonl \
  --output /data/reports/h3_9b_embedding_bridge/preflight.json
```

Default: 32 calibration-train, 8 validation and 8 held-out cases; no exact
reference-asset hash reuse across these cases. Inputs are original training
split captions/guides, capped at four pictures for the initial cost experiment.
They are UNFILTERED calibration requests, NOT approved H3 training examples and
NOT a clean identity-quality benchmark. No targets, original captions or
production caches are modified. Their source captions may be inaccurate; this
limits downstream identity conclusions. Reference-image hashes prevent exact
asset overlap, not all appearances of the same real identity.

Each probe uses at most 73 native 24-fps frames (~3.04 s), source-time resampling,
384-pixel width and 2-fps Qwen context. Teacher and student use the same media
interval and picture order, with their respective native vision processors.
This small probe cannot establish full-video identity stability.

## Deferred GPU stages (after the dataset reviewer exits)

Use absolute paths or set `PROBE`, `CALIB`, `H3PY`, `QWENPY` in your shell.
`H3PY` is a compatible Comfy Python environment; `QWENPY` is an isolated
Transformers environment supporting the complete Qwen3.5 model. Use an empty
output directory. Models are loaded sequentially, not concurrently.

```bash
PROBE=/data/bfs-dev-vlm-reasoning/experiments/h3_embedding_bridge
CALIB=/data/reports/h3_9b_embedding_bridge
H3PY=/data/comfyui/.venv/bin/python
# Set QWENPY to your separate compatible environment, and QWEN_MODEL to local weights.
PYTHONPATH=/tmp/caption-runtime-site "$H3PY" "$PROBE/cache_teacher.py" \
  --manifest "$CALIB/manifest.jsonl" --cache "$CALIB/cache"
"$QWENPY" "$PROBE/cache_student.py" --model "$QWEN_MODEL" \
  --manifest "$CALIB/manifest.jsonl" --cache "$CALIB/cache" --max-new-tokens 512
"$QWENPY" "$PROBE/train_bridge.py" --manifest "$CALIB/manifest.jsonl" \
  --cache "$CALIB/cache" --output "$CALIB/run_001" --steps 500
```

Cache only complete, matching input keys. Teacher weights stay in their native
conversion/quantization format. If the local Comfy Python cannot load that format,
fix the isolated runtime first. The teacher stage has not been executed here.
Student HF paths are local-only; partial thinking halts with a metadata record
and can be retried with a larger token budget. The scripts do not resume training
optimizer state; choose a new output directory for each trial.

Validation selects one shared bridge using both thinking modes. Test outputs
contain `thinking_0`, `thinking_1`, and `thinking_ablated` (same thinking-on
prefill, generated states zeroed, same state-bank quota). Never use test teacher
features to fit the bridge. The per-channel output scale is estimated from the
training split only. There is no video/identity/reward objective in this first
calibration. Production use might require H3 adaptation, broader calibration,
and a trained task-specific objective.

## Local Comfy experiment after calibration

Copy held-out exported `*.thinking_*.safetensors` and
`*.thinking_ablated.safetensors` into `ComfyUI/models/h3_embedding_bridge/`.
Do not select `bridge.safetensors` there: it contains weights, not conditioning.
Restart Comfy with this experimental BFS branch. The node is:

**BFS H3 Experimental 9B Embedding Bridge (Cached Case)**

First build native H3 conditioning with the exact SAME encoder, prompt, ordered
pictures, source interval/resampling, dimensions and 2-fps guide context used in
that export. Feed its `positive` through the experimental node before the
sampler; retain the normal latent/Add Guide/reference-VAE path. The node replaces
only embeddings and retains existing metadata, native reference latents and
aligned guide latents. A teacher-baseline comparison rejects mismatched cached
cases/tags; do not raise tolerance to bypass wrong inputs. This baseline pass
means the prototype does not save encoder compute.

Render four matched arms with identical seed, guide, references, LoRA, CFG,
steps and sampler: native teacher, 9B thinking off, 9B thinking on, generated-state
ablation. Judge wrong-person assignments, identity exchanges after crossings or
occlusions, reference wardrobe fidelity and motion separately. A difference in
output, or a lower embedding reconstruction error, does not establish a benefit.
Use curated longer multi-person clips before changing the production encoder.

Official Qwen model/config: https://huggingface.co/Qwen/Qwen3.5-9B
Original H3 encoder: https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/FL2VA/text_encoder
