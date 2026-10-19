# Planner Agent experiment

Branch: `feat/planner-agentic`, based on `origin/main` at `488cb77`.

This adds a server-side tool agent to BFS Shot Planner. The Agent tab accepts a video task,
starts the local VLM, edits the Planner's own plan, uses SAM, renders the workflow, inspects
outputs, retries shots and combines a reviewed result. The browser displays activity and can
follow the tab and shot involved in each action. Closing the browser does not stop the worker.

## Local setup

1. Install this branch as the BFSNodes checkout in your local ComfyUI and restart ComfyUI.
2. Open a working Planner → conditioning → sampler → decode → Shot Join → save workflow.
   Select the video and reference pictures as usual. Connect a vision-language CLIP to
   Planner's `vlm` input. Choose a model checkpoint capable of thinking if enabling Thinking.
3. Leave Planner's optional external `prompt` input disconnected/empty. The Agent writes
   global and per-shot prompts in the plan. Keep only one Planner in this generation chain.
4. Open **Agent**, describe the task and preservation requirements, set render/action budgets,
   and click **Start automatic task**. No preliminary full video render is needed: an internal
   context-only queue loads the connected VLM first.

The generation workflow is snapshotted when starting a task. Graph changes afterward require
a new task. The agent can change seed/steps/cfg on supported generation nodes through its
`update_sampler` tool. It cannot edit arbitrary graph connections or execute shell commands.
The worker restricts rendering to the selected Planner's downstream outputs and their dependencies.

## Intermediate H3 previews

Insert **BFS Agent Preview Sampler (H3)** between a SAMPLER selector and SamplerCustom /
SamplerCustomAdvanced. It is an explicit wrapper; regular KSampler nodes with an internal
sampler are not intercepted. Existing generation without this wrapper still supports final-shot
inspection and correction, but has no agent intermediate previews.

- Put a compatible `taeh3*` checkpoint in `ComfyUI/models/vae_approx`, **or** connect a
  compatible tiny/light H3 VAE to `preview_vae`. The experiment does not download weights.
- Session ID and the Planner shot link are bound automatically in the Agent's queue snapshot.
- Defaults: inspect every 5 sampling steps, at least 15 seconds apart, using 4 temporal positions.
  The final step is captured even if the interval has not expired.
- Captures use **denoised x0**, converted through the generation model's latent format.
  Each selected position is decoded with a contiguous temporal neighborhood. Positions are
  approximate latent-time observations, not a full decoded video or exact frame timestamps.
- With `review_with_vlm` on, reviews begin after 35% progress. A VLM may request a retry
  for a major failure after 65%; that interrupts this generation and returns control to the
  agent, which can adjust the plan and render again within its budget.
- Source frames and reference images are supplied with the intermediate previews. The VLM
  is instructed not to judge fine identity from early previews.
- Review runs synchronously at a sampler boundary; generation waits while decoding/reviewing.
  Previously loaded models are restored before the next denoising step. This is not concurrent
  VLM and diffusion inference, and its overhead and compatibility require local GPU testing.

## Tools

| Tool | Operation |
|---|---|
| `set_checklist` | Visible step-by-step plan with pending/in progress/done states |
| `inspect_plan` | Resolved shots, active filters, cached render status, sampler settings |
| `inspect_source` | Ordered source frames and assigned references |
| `update_plan`, `update_shot`, `split_shot` | Planner settings, shot prompts/references/modes and boundaries |
| `segment`, `preview_mask` | SAM text/positive/negative points, temporal tracking and mask overlays |
| `write_prompt`, `describe_refs` | Existing visual prompt writer and reference descriptions |
| `render`, `inspect_result`, `inspect_preview` | Queue generation and visually inspect final/intermediate frames |
| `retry_shot`, `restore_take` | Archive the previous shot files, invalidate a shot/dependents, restore archived files and their full render settings snapshot |
| `update_sampler` | Bounded seed, steps, cfg or scheduler denoise changes on supported nodes |
| `combine`, `finish` | Assemble only after all active current shots were inspected; then finish |
| `list_directory`, `read_file`, `inspect_media` | Context files/images/videos under input, output or configured project root |
| `search_web`, `fetch_url` | Public HTTPS text context; search uses DuckDuckGo HTML, no API key |

SAM supports up to eight correction keyframes per shot through `segment.keyframes`.
Coordinates are normalized; keys are frame indices relative to the shot. The first keyframe
tracks backward and forward; subsequent corrections apply from their keyframe onward.
Global mask shaping settings are shared across shots, as in the existing Planner.

Directory tools are read-only. Resolved paths stay inside their configured roots, including
symlinks; hidden/credential paths are excluded. Web tools read bounded public HTTPS text
and reject local/private network destinations and non-text downloads. Search providers can
rate-limit or return challenge pages; those observations are returned to the agent.
Files/pages are context, not instructions or extra tool definitions.

## Controls, persistence and limits

- **Follow tabs** moves the panel to the tool's tab and selected shot. A persistent activity
  bar has Pause, Resume and Stop even while another tab is open.
- **Pause** stops further decisions after the current tool. A render already in the queue
  may finish; Resume picks it up using its exact ComfyUI prompt ID and accepts manual plan edits.
- **Stop** removes this session's queued render or interrupts it if it is currently running.
  It does not cancel unrelated queue entries.
- Sessions and workflow snapshots are written under `ComfyUI/output/bfs_agent/<id>.json`.
  The session ID is saved in the Planner plan. Reloading the panel reconnects to the worker.
  After a server restart, the session restores paused and must be explicitly resumed; a lost
  render ID is reported instead of silently launching duplicate work.
- One active Agent session at a time. Agent actions wait when another ComfyUI prompt runs.
- Default limits: 40 tool decisions, 4 full render attempts, 4096 generated tokens per decision.
  Three consecutive invalid responses/tool errors stop automatic decisions.
  On a paused/errored/budget-limited session, adjust the limits and click Resume to continue.
- Changes invalidate prior review and assembly. Cached results must be reconciled by a new
  render before being accepted. Disabled/filtered timeline shots map to active render indices.
- Intermediate/final VLM judgements are experimental. The system does not guarantee perfect
  identity, motion, temporal consistency or successful thinking on every checkpoint.

## Validation and GPU handoff

CPU checks exercise action validation, budgets, pause/cancel boundaries, session persistence,
SAM dispatch, mask correction boundaries, directory confinement, sampler settings, preview
callback x0 selection/model restoration and a complete simulated server queue cycle.
These use synthetic frames and mocked models/queue; they do not establish visual quality.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -p test_planner_agent.py -v
```

For GPU validation start with one short shot, two references, at most two render attempts,
then test SAM overlays, final review, previews without review, previews with thinking review,
and interruption/retry. Verify frame shape/range, identity compared to the full VAE, model
restoration after each review, memory peaks and time added per inspection. Repeat with several
shots and pause/resume, then close/reopen the browser during a render.

No VLM, SAM, preview decoding or diffusion inference was run on the training GPU during development.

The existing Planner suite also reports two failures reproduced against untouched `origin/main`:
`CastTest.test_plan_splits_assigns_and_skips_by_person` (empty cast reference under the existing test harness),
and `FillDetailsTest.test_render_node_fills_details_with_its_vlm` (test mock accepts six arguments
while the implementation passes seven). They are outside this experiment.

## Connected VLM diagnostics

After Start loads the connected VLM, the Agent tab shows its checkpoint (when exposed by the
loader), architecture, detected vision encoder, Thinking control, backend context limit,
requested output budget and the Agent's separate 16,384-token output cap. Resume refreshes
the report and applies Thinking/Web changes as well as numeric budgets.

Tokenizer support for a Thinking switch does not prove that a checkpoint was trained for
reasoning. Unknown capabilities remain unknown; tokenizer padding limits are never reported
as context limits. The backend context configuration may differ from a checkpoint's documented limit.

On initialized CUDA devices, the report includes free/total VRAM, unloaded VLM weights and
a memory reserve. Known full-attention Llama2_ backends also receive a conservative FP32
KV-cache estimate for one sequence. This is a total context estimate shared by input text,
images and output, not a guaranteed maximum output budget. It does not measure vision/prefill
or workspace peaks, and custom/hybrid backends receive no token-capacity estimate.
The diagnostic does not load/offload models, allocate probe tensors or change settings.

Numeric settings update during typing so periodic session polls do not overwrite an edit
before the field loses focus.

## Thinking that never produces a final action

With Thinking enabled, each decision reserves up to one quarter of its output-token budget
(capped at 2,048 tokens) for finalization. If the first call returns no final action, the Agent
logs a `finalize_action` event and requests one action with Thinking off, using the original
task, observations and images. It does not feed back or persist truncated private reasoning.
The combined generation limits of both calls stay inside the configured decision budget:
8,192 tokens means at most 6,144 for the first call and 2,048 for finalization.
A valid first response needs no second call. The Thinking setting stays enabled for future
decisions. Pause/Stop between calls skips finalization; tool validation and render budgets
still apply. If finalization also fails, the Agent reports an error rather than inventing a tool action.

## H3 packed AV preview latents

The Agent sampler runs before ComfyUI's outer callback unpacks AV latents. It uses the
model's `latent_shapes` to extract the first (video) stream from flat `[B,1,N]` x0,
validating stream sizes before decoding. Dense and nested H3 video tensors remain supported;
audio is never decoded as video. Native callbacks and generation output retain their original tensors.

Both taeh3 and the normal H3 video VAE can be connected as `preview_vae`. Known full
`MiniMaxH3VideoVAE` decoders use their single-latent image path for approximate snapshots;
tiny decoders use contiguous temporal neighborhoods. Full-VAE previews cost more time and VRAM.
Preview shape/decode/review errors log a visible warning and disable intermediate preview
for that render while generation continues. Explicit VLM retry decisions and user stop/interrupt
still interrupt sampling. These paths have CPU contract tests; full-VAE and taeh3 GPU fidelity
and memory behavior still require local validation.

## Agent inference lifecycle and CUDA faults

Agent background work uses a dedicated single-thread executor and enters `torch.inference_mode()`
in that thread. Each direct TextGenerate call, including Thinking finalization, runs the
native per-node cleanup for ComfyUI's dynamic allocator when available: prefetch/compiler
state, cast buffers and vbar watermarks. This fixes differences from ordinary queued nodes;
it does not prove that a particular checkpoint/driver/kernel combination is stable on a GPU.

A detected illegal-address/device-assert CUDA fault stops automatic retries immediately.
Further Agent starts/resumes are blocked until the process is restarted. Preview faults of
this kind are fatal too: the preview adapter must not attempt to reload models or continue
sampling with a poisoned context. Non-CUDA preview errors still degrade to a visible warning.
Native CUDA aborts can terminate Python before any Python error handler can run.

VRAM pressure and illegal CUDA accesses are separate diagnoses. Dynamic model loading logs
do not establish peak memory use. CPU tests cover lifecycle/order, inference context and fault
handling; reproduction on the user's actual GPU/backend is still needed.

## Kijai's decoder-only taeh3 checkpoint

The 9.79 MB Kijai `MiniMax-H3-TAE/vae_approx/taeh3.safetensors` is a 2D decoder-only
checkpoint (24 latent channels, 96/64-wide blocks, four spatial upsamples). It is not
a normal VAE checkpoint and the stock VAELoader may reject it. For Agent previews, keep
it in `models/vae_approx`, disconnect the stock loader from `preview_vae`, and let the
Agent sampler load it automatically. The adapter strictly loads all 81 tensors and decodes
independent snapshots, one at a time. Other temporal/native VAE formats still use ComfyUI's
VAE loader. Keep the full H3 video VAE on the final video decode branch.

An invalid-VAE error from the workflow is treated as a configuration failure: the Agent
stops instead of asking the VLM to repair a loader/model pairing it cannot change. Fix
the graph and start a new task so its workflow snapshot includes the change.

The per-call lifecycle guard also covers the common Planner TextGenerate helper (prompt
writing, reference descriptions and preview review), not only Agent decisions. This addresses
a repeated-generation failure mode reported upstream in ComfyUI issue #16441:
https://github.com/Comfy-Org/ComfyUI/issues/16441
Repeated-generation GPU stability still needs verification on the affected installation.

### Decision explanations and render evidence

Actions may include concise public `observed`, `reason`, `expected`, and
`uncertainty` explanations. The activity panel displays these alongside the tool
and arguments. A decision-start event reports the current phase and token budget;
private model reasoning is not displayed or persisted.

Intermediate captures are written as JPEGs under
`output/bfs_agent/<session>/previews/<attempt>/` before VLM review. Session JSON
stores the capture URLs, reviews, and render outcomes. Render history preserves
captures across retries, distinguishes technical failures from preview rejection,
and explicitly labels approximate previews. Failed renders with no captured
preview are recorded without invented images. A completed render is not a visual
quality approval. Persistence errors disable the optional preview adapter using
its existing error path. Captures survive later failures; nothing can recover a
preview that was never decoded and saved. Existing sessions without these fields
remain readable. This history is session-specific, not automatic cross-video
memory or context summarization. No external classifier service is enabled.

### Correcting a result with feedback

In the Agent conversation, write a correction and click **Send feedback and continue**.
Messages can be sent while the agent is working; they are queued for the next safe
tool boundary. Completed sessions can continue from feedback directly. Choose
**Question** to request an answer without withdrawing result approvals.
Identify the affected shot/person, the incorrect change, and what must remain
unchanged. For example: “Only replace the soldier on the left. You changed both
people. Segment the left soldier, check the mask, and preserve the person on the
right.”

Feedback is saved in the same session, included in every subsequent decision and
preview review, and linked to the last rendered revision's evidence. It clears
previous result approvals and the current combined-output approval while keeping
render snapshots, cached takes, previews, settings, and the original task. The
agent is instructed to inspect the rejected result and correct affected shots,
including checking the SAM overlay and other people's preservation for a
single-target swap. These instructions do not guarantee visual success.

Action/render counters are preserved: raise the existing limits when necessary.
Feedback cannot bypass a CUDA fault, invalid workflow snapshot, or another active
agent session. Stopped sessions need a new task. Each message is limited to 4000
characters; a session accepts at most 40 messages / 16000 total feedback characters.
This is persistent session feedback, not automatic learning across videos.

### Comparing generation parameters

`inspect_generation` lists the editable controls on existing workflow nodes,
including the installed sampler/scheduler choices and inputs connected to other
nodes. `update_sampler` now also supports sampler and scheduler names, standard
LoRA model/CLIP strengths, BFS LoRA Surgery overall strength, and controls on
standard sigma scheduler nodes (Karras, exponential, polyexponential, Laplace,
beta, VP, SD Turbo). An existing ManualSigmas node accepts a descending schedule
ending in zero, as a string or numeric list. Unsupported custom nodes are not
modified. Connected values must be adjusted at a supported source node; graph
connections and node types are preserved. Numeric values must be finite and
respect both agent limits and installed schema limits. LoRA files are not changed.

Before changing generation parameters, the agent archives current successful
active takes and the full plan/workflow snapshot. `list_takes` exposes the saved
settings and video paths for review; `restore_take` can choose a specific
`take_id` rather than only the latest archive. A restored take must be reconciled
with the cache and reviewed before combining. Each comparison still consumes
normal action/render budgets. Keep seed, prompt, references, shot ranges and
other controls fixed, change one variable at a time, inspect the candidates and
reserve budget for restoring the selected take. The VLM judges visual quality;
this is not a standardized benchmark or a guarantee of the best parameters.

### Paged, targeted result review

**Review fps** defaults to 2 sampled frames per second, including shot endpoints.
Set it to **0** to require every output frame. Each `inspect_result` call returns
up to six generated/source pairs at matching shot-relative times plus references.
Subsequent calls automatically advance to unreviewed required frames. The panel
reports coverage, and combining remains blocked until every active shot's required
frames have been presented. Dense review consumes more model calls/actions;
increase the action limit when needed. Sampling can miss brief defects, and even
full-frame coverage does not guarantee the VLM notices every visual error.

Both `inspect_source` and `inspect_result` accept exact `frame_indices` or an
inclusive `start_frame`/`end_frame` range with `count` (1..6). Indices are relative
to the selected shot. `region: [left,top,right,bottom]` magnifies normalized full-
frame coordinates; these views decode at source resolution before cropping. They
provide additional evidence without counting as full-frame review. Inspect the
source at the same key frame before selecting SAM points. Mask coordinates always
refer to the full source frame. The UI prioritizes final result inspection over
an older intermediate preview. Image base64 is supplied through the image input,
not repeated as text in the model context.

`update_shot` now advertises its exact partial-update fields and returns actionable
recovery instructions for unsupported fields. Use `segment` for masks, `refs` for
reference filenames and `target` for a source-person description. For point-based
SAM selection, include a positive point on the replacement target; negative points
exclude protected people in that same selection. Negative-only point selections
are rejected before changing the plan. These checks do not establish that a mask
is visually correct: inspect its overlay before rendering.

### Loading paths written in the task

An existing connected Planner/generation workflow can load server-side media from
labelled lines at Start:

```text
Guide: "/videos/battle scene.mp4"
Ref 1: "/pictures/goblin.png"
Replace only the soldier on the left. Preserve the other soldier.
```

Paths must exist on the **ComfyUI server**; a browser-local path does not upload a
file. Relative paths resolve under ComfyUI's input folder. References support
`Ref 1` through `Ref 12`. Quoted paths may contain spaces. Files outside input are
registered under `input/bfs_agent_imports` with stable generated names, normally
using symlinks (copy fallback where symlinks are unavailable); originals are kept.
Reference images are validated before registration. These imports and the selected
plan are recorded in session activity and reflected in the Planner UI.

A different guide clears the previous source's shot/cast/timing overrides and
uses bounded fixed shots (at most 15 seconds initially), retaining generation/model
configuration. Explicit references replace old per-shot reference overrides.
The agent can then inspect and refine the plan. Loading paths does not build model,
conditioning, sampler, or output nodes: the workflow must already be connected.
Use a new Start to select a different guide or refresh graph connections.

### Conversation, resources and context

The Agent tab now has Conversation, Plan, Renders, Activity and Settings views.
Public decision explanations appear as assistant messages; tool cards distinguish
requests from completed and failed operations, with arguments and evidence in
expandable details. Earlier saved activity can be paged back into the conversation.
The compact Agent bar remains available in other Planner tabs. Following tool tabs
is optional and starts off so the conversation stays visible.

Send corrections or questions while the agent works. Inputs are persisted
immediately, then applied between tools. A render is allowed to finish. If input
arrives while the VLM is choosing an action, that stale action is discarded before
execution and the next decision uses the new input. Pause/Stop still take priority.
A correction withdraws previous visual approvals; a Question leaves them intact.
`reply` answers without ending the autonomous task. Messages cannot bypass invalid
workflow snapshots, CUDA faults or execution budgets. A stopped session needs a
new task. At most 16 messages may be queued and 100 messages stored per session;
the existing correction text budget still applies.

Attach screenshots with the file picker, drag/drop, or paste. Up to four images per
message are resized to at most 1280px, stored as JPEG under
`output/bfs_agent/<session>/feedback/`, and supplied to the VLM when the input is
applied. Screenshots may also accompany the initial task. `inspect_feedback` reopens
saved screenshots later. Text inside an image is visual data, not authority to
execute instructions. Failed sends retain the local draft and attachments.

Resource cards show read-only GPU/RAM snapshots, execution budgets, current context
and cumulative tracked token estimates. GPU information is unavailable until CUDA
is already initialized; sampling resources never loads a model or initializes
CUDA. A fallback VLM-load snapshot is labelled as such. Resource snapshots refresh
while the Agent tab is visible. Token accounting uses text bytes/3 and a heuristic
512 tokens per supplied image: these are **estimates**, not exact tokenizer counts,
billing usage or measured KV memory. Returned final/reasoning text contributes only
a numeric output estimate; raw private reasoning is never saved. Calls without
returned output or failed calls mark the accumulated figure as incomplete (`+`).
Legacy calls made before tracking was introduced cannot be reconstructed.

The working **Context window** defaults to 32768 and is capped by the detected
backend limit when available. Input estimates, reserved decision output and a
safety margin share that window. Session cumulative usage is separate and is not
subtracted from it. When activity grows, the harness writes a grounded, bounded
summary of recorded requests, observations and failures, keeps recent events and
live state, and continues. This compaction requires no additional GPU generation.
The original task and latest input remain intact; full events and evidence remain
saved and can be recovered through `inspect_history`/`inspect_feedback`. Very long
optional sections are explicitly marked as excerpts. If fixed instructions and
images cannot fit even after compaction, execution stops with a clear context
error instead of silently exceeding the working limit. Increasing this setting
does not extend the model's capabilities or guarantee VRAM fit.
