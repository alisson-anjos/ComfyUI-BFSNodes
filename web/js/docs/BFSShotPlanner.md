# BFS Shot Planner / Shot Loop

Turn a long video into shots a video model can actually follow, give every shot its own
reference image and prompt, run the rest of the workflow once per shot, and join the results
back into one video with the original timing and soundtrack.

## Why

A video model only follows a guide reliably inside the clip length it was trained on. An H3
body-swap LoRA trained on 107-frame clips keeps the scene, framing and motion at 4.5 s and
loses them at 10 s, where it starts summarising the clip instead of following it frame by
frame. Splitting the source into model-sized shots, at camera cuts when there are any, keeps
every shot inside that range.

## Nodes

| node | what it does |
|---|---|
| **BFS Shot Planner** | Pick a video, split it on a timeline, set a reference and prompt per shot. Outputs the shots as a ComfyUI **list**. |
| **BFS Shot Unpack** | Opens one shot into plain values: guide frames, reference pictures (a list), prompt, length, first frame, audio, size, index. Use it with any model. |
| **BFS Shot Repack** | Puts edited pieces back into a shot: Unpack's `shot` output goes to Repack's `shot`, the edited piece (e.g. the reference with its background removed) to its input. Unconnected inputs keep their values; timing, cuts and audio are unchanged. |
| **BFS Shot H3 Conditioning** | The common MiniMax H3 path with the important controls only: aligned Add Guide and/or native video reference, references, optional audio conditioning, the Planner's inpaint mode and source-subject blur. |
| **BFS Shot H3 Conditioning (Advanced)** | The full conditioning node. Adds the setting reference, the first-frame anchor and the experimental mask-guide controls; its `model` input passes through to `model` (older workflows routed the model through it). The duet lives in **BFS Shot H3 Duet** (1.113 took it out of this node). |
| **BFS Shot Join** | Concatenates the decoded shots in order, trims each to its true length, cross-fades soft joins and returns the soundtrack. With *comparison* on it also returns a side-by-side video (original shot \| references \| result) with the shot's info and your *label* on top and its prompt below, ready for Create Video. *save to = mp4 file* writes the video (and the comparison) straight to MP4 files instead, one shot in memory at a time (long videos); `video_path` gives the file. |
| **BFS Shot H3 Duet** | Renders one shot with H3: the shot's clip pinned beside the video (duet, no LoRA), the shot as an aligned guide (for body-swap LoRAs), or both. |

## The panel

Use the language selector in the top bar to switch the interface between English, Portuguese, Spanish and Simplified
Chinese. The choice is saved in this browser and does not alter the workflow plan or prompt text. **Tour** opens an
optional step-by-step introduction when clicked; it never appears automatically.

The planner's panel has four tabs, in the order you work. The timeline stays on top of every tab.

| tab | what is there |
|---|---|
| 🎞 **Video** | pick or upload the video, the range, how it is split into shots (cuts or fixed length), generation size. The cut detector, frame grid and fps are under *More*. |
| 📝 **Pictures & prompt** | the pictures and prompt every shot uses, 🤖 Auto prompt, and (collapsed) the VLM and the picture descriptions (`{details}`) |
| 🎬 **Shots** | one card per shot (scroll sideways, **← →** move between shots), the preview player and the selected shot's editor: pictures & prompt, **What changes** (the selection, Colored Mark, how it is generated, `{target}`), continuity, **Copy to other shots**; then the mask settings shared by every shot (collapsed) |
| ▶ **Run** | **Checks** (what to fix before running, with a button to each shot), how the shots run (shot by shot / queue loop / auto loop) and the stored shots |

Shot cards show badges for what each shot has: 🎯 mask, 🎭 / ✂ generation mode, 🧑 `{target}`, ⛓ continuity and ⚠ problems.
The header's ⚠ count opens the checks.

**Copy to other shots** (in the shot editor) copies the chosen parts of the selected shot to all shots, or to the
shots after it: references, prompt, mask text, mask points (each shot gets them on its frame at the same relative
position), generation mode, `{target}`, continuity.

## How the loop works

ComfyUI runs any node that receives a list once per item and pairs several lists by index. The
planner's `shots` output is a list, so everything downstream of it (conditioning, sampler,
decode) runs once per shot, and shot *i* always gets guide *i*, reference *i* and prompt *i*.
Single values such as the model and VAE are shared by every run. The join takes the whole list
back (it is an `INPUT_IS_LIST` node) and puts the video together.

Three run modes:

- **One run, shot by shot (default)**: one Queue renders every shot **in order inside the same run**. Each shot is
  stored on disk; then BFS Shot Join expands the part of the graph between the planner and itself (ComfyUI node
  expansion) for the next shot, so that shot is conditioned *after* the previous one exists and continuity
  (*head* / *history*) works. No browser re-queue (the tab can be closed), Save Video fires once with the full video,
  and after a failure a new Queue resumes from the stored shots (*Redo all* regenerates every shot). Previews connected inside
  the shot's part of the graph show every shot. Needs exactly one BFS Shot Planner before the join. Same idea as the
  recursive loop of Ethanfel's MiniMax H3 Context Loop.
- **Queue loop (one shot per run)**: each run generates the next pending shot and stores it on
  disk. The join blocks every node after it until the last shot, then outputs the full video,
  so a Save Video node only fires once. With *Auto-queue the next shot* on (default), the panel re-queues
  the workflow after each shot, so one click still renders the whole video (keep the ComfyUI tab open: the panel
  does the re-queue). *Redo all* regenerates every shot. Changing the plan starts a new run. It resumes from the stored
  shots after a failure, and continuity *head* / *history* needs it: each shot reads the result of the one before.
- **Auto loop (all shots at once)**: every shot in one queue run, each node runs for every shot before the next node.
  Holds all decoded shots in memory. Continuity *head* / *history* does not work here (all shots are conditioned
  before any is sampled).

### Memory

- **Status at the top (🧠)** and a **Memory** card in the Run tab. BFS Shot Join records every shot it receives on
  this GPU: its size (latent tokens = generated frames, with the continuity head, × resolution) and the peak VRAM of
  its run. Before running, every planned shot is compared with the **biggest shot that already ran on this GPU**:
  *✓* within it, *a bit bigger* (it will probably fit, maybe slower), or *⚠ may run out of memory* (also listed in
  Checks). There is no fixed "GB per frame" for H3 (ComfyUI offloads weights as needed), so nothing is guessed: until
  a first shot has run, the status says *no memory data yet*. The record lives in ComfyUI's `user/bfs_shotloop_vram.json`.
- **Between shots** (Run tab, shot by shot and queue loop): *keep models loaded* (fastest) or *free memory*, which
  unloads every model and empties the GPU after each shot so the next one starts clean (low VRAM, or a later shot
  that runs out of memory); models load again for every shot, so it is slower.
- What makes a shot lighter: lower **Megapixels**, a shorter **Max seconds / shot**, **head 5** instead of 22 (the
  head adds its frames to every shot), **Crop** (the generation is the subject's box), and ComfyUI's own options
  (`--lowvram`, fp8 / GGUF weights, tiled VAE decode).
- **The final join** (after the last shot) is where long videos used to run out of **RAM** (not VRAM: the process is
  killed, e.g. `Terminated` under WSL). It now reads one stored shot at a time into the final video. Measured on 11
  shots at 960×544 (~50 s): **+15.8 GB before, +8.2 GB now** with *save to = images* (that is the video itself, which an
  IMAGE output must hold), and **+2.1 GB with *save to = mp4 file***, which streams the video to ffmpeg and stays at
  about one shot whatever the length (also with the comparison, which is streamed 16 frames at a time: +2.1 GB
  measured at *comparison_height* 1024, against ~32 GB as images). For long videos use *mp4 file* (BFS Shot Join), or
  the panel's **Combine**.
- **RAM guard:** before an *images* join, BFS Shot Join asks ComfyUI's cache to drop earlier outputs and estimates
  what the video (and the comparison) need; when that does not fit in the free RAM it writes MP4 files instead and
  says so in the console, instead of the whole ComfyUI being killed.
- **The outputs are never empty:** with *mp4 file* (chosen or by the RAM guard) the `images` and `comparison` outputs
  still carry the video, read back from the written files at a size that fits the free RAM (the console says the
  scale; the files keep the full size), the join shows the files on its node, and `video_path` gives the path.
- **Default: save_to = images** (the result comes out inside ComfyUI, for the next nodes). The panel's Checks warn when
  the video is long for this machine (as images it needs about 12 bytes per pixel per frame, ~3.5x with the
  comparison, against the free RAM) and advise *mp4 file* for that video.
- **comparison_height** is the height of each column: the comparison is three columns wide plus text bars, so it
  costs about 3× the video at the same height. 360–512 is plenty to review a long video.

### Stored shots (cache, redo, combine)

In the **shot by shot** and **queue loop** runs every generated shot is stored, and the Run tab's **Stored shots**
card manages them.

- **Shot cache:** *Keep* (default) stores them in ComfyUI's output folder, `output/bfs_shotloop/<run>`: they survive a
  restart, so a stopped run resumes and a single shot can be redone. *Temp* uses ComfyUI's temp folder, which ComfyUI
  **empties at every start** (before 1.104 every run was stored there).
- **Shot storage** (the format of each stored shot, measured on 145 frames at 1280×720):

  | Storage | Size | Quality |
  |---|---|---|
  | **Lossless** (FFV1, default) | ~34 MB | exact: the final video is unchanged |
  | Smaller (H.264 RGB, crf 8) | ~21 MB | visually lossless (PSNR ~52 dB) |
  | Smallest (H.264 RGB, crf 14) | ~9 MB | light loss (PSNR ~48 dB) |
  | Raw frames (`.pt`) | ~401 MB | exact; needs no ffmpeg |

  The final video of a stored run is assembled from the stored shots, and the continuity head reads them: a lossy
  storage reaches the output (the panel warns). Without ffmpeg a shot falls back to raw frames. With *comparison* on,
  the shot's source frames for the comparison's original column are kept as `shot_NNNN_orig.mkv` (visually lossless:
  they are only shown); the `.pt` itself holds only the shot's info (a few KB).
- **One folder per source video.** Each shot remembers the settings it was made with (its range, prompt, references,
  mask, crop, continuity, the generation size, the global prompt / references and the references or prompt connected
  to the planner). The next Queue **runs only the shots that are not stored or whose settings changed**, plus the
  shots that **continue from a shot that was made again** (head, history, previous shot as reference / first
  frame), so every seam matches. Shot cards show 💾 (stored, up to date) or ⏳ (runs next time).
- **Model, LoRA, seed, sampler (1.112.0):** the planner reads them from the queued workflow (every node that feeds
  BFS Shot Join's `images`: loaders, LoRAs, sampler, seed, conditioning settings), so changing any of them runs every
  shot again. The join's own settings (crossfade, comparison) don't. A seed set to *randomize* changes on every Queue,
  so every Queue then runs every shot; fix the seed to keep the stored shots.
- **↻ Redo** a shot (Stored shots card): it runs again on the next Queue, and the shots that continue from it run
  after it. Tick *↻ redoes only that shot* to keep the next shots as they are (the seam after it may then not match).
- **🎬 Combine stored shots:** puts the stored shots together into an MP4 in the output folder **now, without
  generating** (one shot in memory at a time), on the plan's timeline: the soundtrack, the original video where a
  shot is skipped, and also where a shot is not stored yet or out of date, so it previews a run in progress. A link
  to the file appears in the card.
- When every shot is stored, a Queue still runs the last shot again so the join can output the video: use Combine to
  only assemble.
- **Recovering a run outside ComfyUI** (e.g. ComfyUI was killed during the final join):
  `python tools/bfs_assemble_shots.py output/bfs_shotloop/<run> video.mp4 --audio input/<source video>` (ComfyUI's
  Python, ffmpeg on the PATH) does the same as Combine with one shot in memory; it reads the old raw `.pt` shots too.

## Guide frames and lengths

Each shot is generated at a length the model accepts (frame grid: H3 `17n+5`, LTX/Wan `8n+1`,
Wan `4n+1`, or any). When a shot is shorter than that length, the extra guide frames come from
the video that follows it, so the guide never repeats or stretches. The join cuts each result
back to the shot's own frames, which keeps the timing identical to the source. Those extra
frames are also a real overlap, which the join uses to cross-fade boundaries that are not
camera cuts.

## Splitting

- **Camera cuts** (default): cuts from **PySceneDetect** (`adaptive` or `content`; install
  `scenedetect`), or the built-in detector when the package is missing. Shots shorter than the
  minimum merge into a neighbour when the merge still fits; shots longer than the maximum split
  into equal parts.
- **Fixed length**: equal parts no longer than the maximum.
- **One shot (whole range)**: the whole selected range as a single shot, whatever its
  length; *Max seconds / shot* is not applied. For "N seconds in one piece" set *Range length, seconds* = N, right
  next to the mode (it moves the end frame).
  For pieces of at most X seconds use *Fixed length* with *Max seconds / shot* = X.
- **By hand**: drag the white handles on the timeline to move a boundary, double-click the
  shot bar to split, *⇥ Merge* to join, or select a shot and press **Delete** to remove the cut at its
  start (it merges into the previous shot). Your edits are what runs.

**Range** (🎞 Video tab, with its own player): *Start frame* / *End frame* (timeline frames; end 0 = the end of the
video) limit the plan to a part of the video, before anything is split, analysed or generated. Play or scrub the player
(or click the timeline) and press *⇤ start here* / *⇥ end here* to take its exact frame. The
timeline dims what is outside; the join and the soundtrack cover only the range.

*Max seconds / shot* is converted to frames and snapped down to the grid (4.5 s at 24 fps =
107 frames for H3). *Max shots* in the Video tab caps how many shots are used.

## Audio

The source card shows whether the video has a usable audio track (rate, channels) or why not. *Use the
video's audio* or *silent track*. Without a usable track (none, unreadable, or silent chosen) the planner's
`audio`, BFS Shot Join's `audio` and the H3 Duet nodes return a **silent track of the right length** instead
of nothing, so Create Video / Save Video always work. Audio is read with ffmpeg, or with PyAV when the ffmpeg
binary is missing.

## Preview

*Play shot* plays only the selected shot, from its first to its last frame, and stops (or loops with
*loop shot*); every shot card has its own ▶. *Play all* plays the whole plan and jumps over shots
that will not run. The panel shows the shot's start, end and current time (and frame), a yellow
playhead follows on the timeline, and clicking the timeline seeks. The source video streams from
the input folder, so only what you play is loaded.

## Turning shots off

**⏸ Disable** in a shot's editor turns a shot off. Shots that are off do not run. Connect the planner's `timeline`
output to BFS Shot Join and they are filled with the original video, keeping the full duration and soundtrack, or
removed (with the matching audio) when *Skipped shots in the output* is set to remove them.

The shot filters (person / face / dark / static detection) and the face cast were removed in 1.122: a plan that
skipped a shot by hand opens with that shot disabled, and the old filter and cast settings are ignored.

## References and prompts

Up to **12 reference pictures** (1.113.0): the global list starts with one slot and **+** adds more; they are
`<Picture 1>` to `<Picture 12>`, in order. Click a slot to upload a picture (the **+** takes several files at once),
or drop image files on a slot (replaces it) or anywhere on the box (adds them). A filled slot has **✕**: a global
picture is taken out (the next ones move up), a shot's own picture falls back to the global one.

Every shot can have its own pictures and prompt. A shot's empty slot uses the global picture at the same position
(shown faded), so a shot can change only `<Picture 2>` and keep the rest. **Copy to other shots** gives a shot's
pictures to every shot. The last ten pictures you picked are shown under the shot editor: click = picture 1,
shift+click = add it as the next picture, or drag it onto any slot.

Under every prompt (global and per shot) there is a chip per picture: **PictureN** inserts `<Picture N>` at the
cursor, **📝** inserts `{details@Picture N}` (that picture's description, below).

The node no longer has `ref_image` / `ref_image_2` inputs (the pictures live in the panel); a workflow saved with
them connected still runs, and the log says the images are ignored. A connected `prompt` input still overrides the
global prompt. The planner outputs `refs` (every picture the shots use, without repeats, as a list) and `ref_count`
where `ref_image` / `ref_image_2` were. **BFS Shot Unpack** outputs `refs`, every picture of the shot as a list (the next node runs once per picture,
e.g. a background remover), and `ref_count`; **BFS Shot Repack**'s `refs` input takes that list back and gives each
shot its own pictures (a workflow from before 1.113 that wired `ref_image` still works). **BFS Shot H3 Conditioning** passes every picture of the shot (the `use_ref_2` switch left in 1.114: the pictures are chosen in the planner); H3's own
node stops at 9 and LoRAs trained like OpenJutsu saw up to 9, so the log notes a shot with more.

## Mask & crop (SAM 3)

Generate only part of the frame. In a shot's editor (**What changes**), type what to select in English
(`person in white`, `woman with red hair`, `red car`; up to 32 tokens, commas for several things) or click
**🎯 Points** to pick positive / negative points on any frame of the shot, then **👁 Preview** (red = mask,
yellow = crop box). Then pick how the shot is generated in **Generate**:

| mode | what is generated | crop / uncrop |
|---|---|---|
| **Full frame** | the whole frame; the mask only feeds `{target}` and the setting picture | no |
| **🧩 Frame + paste** | the whole frame (so the model follows the guide's pose freely and the new person may be bigger); BFS Shot Join then pastes only the person onto the original frame: the shot's mask plus the new subject's outline (SAM 3 on the result with the shot's *new subject* text, default `person`; `paste_new_outline`) | no crop; paste in the Join |
| **🎭 Mask only** | only the mask, on the whole frame; everything else stays exactly as it was | no |
| **✂ Crop** | a box around the mask (more pixels for a small person); BFS Shot Join pastes it back | yes |
| **✂🎭 Crop + mask** | only the mask, inside the crop: the most detail with the background kept | yes |

*Frame + paste* is the safest for a different body (e.g. a woman in profile replaced by a man facing the camera):
*Mask only* binds the new person to the old outline, *Frame + paste* does not. The two mask modes need **BFS Shot H3 Conditioning → inpaint = per shot (planner)** (the default; *only the mask*
forces it on every shot, *off* never). The crop is one box around the mask (the union over all its frames, so it
does not shake), at the source resolution, sent to the model at the generation size; **BFS Shot Join** pastes the
result back, feathered by the mask (or the whole box). **BFS Shot Unpack** also outputs the shot's mask (`mask`),
e.g. for an inpainting model. In **BFS Shot Join**'s comparison video, cropped shots show their mask (red) and crop
box (yellow) over the original column (`comparison_mask`, on by default).

**Mask strength** (mask modes, per shot, slider under **Generate**): the value of the white inside the generation mask.
1 regenerates the masked area completely. A grey mask (e.g. 0.85) keeps part of the original there: H3 puts those
rows at *opacity* × the noise level, so they start from the original partly visible. 0.8-0.9 keeps pose, outline and lighting while still swapping; too low copies the original
person. With `guide_mode = aligned guide` the model also sees the whole original shot as its guide, so it follows
the motion either way.

**🎨 Colored Mark** is an optional translucent overlay on the source video: each selected subject (a person, an object,
a vehicle) is tinted with the color of the picture it becomes, and the prompt gets the mapping ("the person marked in red
becomes the character from `<Picture 3>`"), for any model or LoRA. Every picture has its **Mark color** (under it in **Pictures & prompt**:
green, purple, red, blue, yellow, magenta, cyan, orange, white, brown, pink, olive by default); change it when it is
too close to a color in the video. In a shot, turn on Colored Mark: every row under it is a subject, with its text or
🎯 points and **→ Picture N**, the picture it becomes. *Subject 1* is the shot's own selection; **+ Add subject** adds
another row (the points window has one point set per subject). This is only a visual marker: it does not create an inpainting mask
or change **Full frame**, **Mask only** or crop behavior. More than one marked subject needs Crop off, and the
shot must have every picture the marks use. The overlay keeps movement and facial
expression visible. Plans before 1.113 (a second region for ref 2) open with that region as `<Picture 2>`.

```text
body_swap: subject_definitions: <Subject 1> is the character shown in <Picture 1>, whose identity, face, head, hair or fur, skin, body shape, proportions and clothing define the green-marked replacement. <Subject 2> is the character shown in <Picture 2>, whose identity, face, head, hair or fur, skin, body shape, proportions and clothing define the purple-marked replacement. In the guide video, the translucent green and purple tints are temporary editing markers, not character appearance. Replace the green-tinted person only with <Subject 1> and the purple-tinted person only with <Subject 2>. The guide video provides the scene and the complete performance of both people, including their body poses, gestures, head motion, gaze, speech and facial expressions.

summary: [video generation + references] Replace the green-tinted guide person with <Subject 1> from <Picture 1> and the purple-tinted guide person with <Subject 2> from <Picture 2>, preserving the scene, camera and complete motion and expression of both people.

detailed_description: [Shot 1] Replace the person marked by the translucent green tint with <Subject 1> from <Picture 1>, and the person marked by the translucent purple tint with <Subject 2> from <Picture 2>. Each character follows that person's exact position, timing, body pose, gestures, head motion, gaze, speech and facial expressions in the guide. Remove both temporary color markers completely. Keep the two reference identities distinct and stable in every frame. Preserve their interaction, the background, lighting, camera and all unselected content from the guide video. Prevent identity drift, clothing changes, texture crawling and flicker. The result is a real, photographic video.
```

The VLM's shot analysis suggests a SAM 3 text selection from the original frames; it does not assign guide colors or
decide that two detections are the same identity. **🧑 Describe** only describes the first shot selection. For
two-color swaps, choose both selections and identities yourself, and use the explicit mapping above (or let
🤖 Auto prompt write it: it is told which tinted person becomes which picture).

**Mask guide** (experimental, *Mask only* shots, BFS Shot H3 Conditioning → `mask_guide`): the shot with only the
masked region visible, the rest grey, so the model looks at the subject's pose and outline on its own while the mask
limits where it generates. *+ extra guide* adds it as a second aligned guide next to the normal one (the model sees
both); *instead of the full guide* replaces the normal one; *+ reference video* gives it as a native reference video
(`<Video n>`, after the shot's own one when guide_mode uses it; write `{mask_video}` in the prompt where its tag goes, otherwise a sentence is added). `mask_guide_look` sets what it shows of the subject: *grey blurred* (default: volume, light and head direction, no colours or face, so the model follows the motion without copying the old subject), *colour* (as it is; can make the model copy the old subject), *silhouette* (flat shape), *edges* (outlines) or *pose (people)* (the skeleton of the people in the mask, OpenPose colours; needs ultralytics and a YOLO pose model in `models/ultralytics`, e.g. `pose/yolov8m-pose.pt`). `mask_ref_size` makes that reference video smaller (default 1/2, ~1/4 of the tokens): it only has to show the pose and outline. Compare with
*off*: the LoRAs were trained with one full guide, and it can also pull the old subject's look.

### Masks from a video (rotoscoping) instead of SAM 3

A mask made elsewhere (After Effects Roto Brush, DaVinci Resolve Magic Mask, another ComfyUI workflow…) can replace
SAM 3. It is a black and white video, **white = the subject**, and it works with every mode (Mask only, Crop,
Crop + mask), `{target}`, the setting picture and the comparison overlay. Three places, the most specific wins:

| where | covers | wins over |
|---|---|---|
| shot editor → **🎞 mask video** | only that shot (its first frame = the shot's first frame) | everything |
| the shot's own SAM 3 text / points | that shot | the two below |
| the planner's optional **`mask`** input (MASK or IMAGE batch, e.g. Load Video + Convert Image to Mask) | the whole video, one mask per source frame | the panel's mask video |
| Shots → **Mask settings** → **🎞 Mask video (whole video)** | the whole video, matched by time | SAM 3 for shots without a mask |

Any size and frame rate: it is matched to the shot by time and resized. The console says which mask each shot used.

**Stitch finishing** (BFS Shot Join, cropped shots only; adapted from Neko (Nekodificador)'s *NKD Inpaint Stitch*, MIT, built on AbleJones's workflow and nodes):

| Option | What it does |
|---|---|
| `edge_hardness` (0-1) | hardens the soft edge of the paste; raise it when a faint ghost of the original person shows around the new one |
| `match_colors` (0-1) | corrects the colour / brightness drift of the generated patch (Reinhard in LAB), with statistics over the whole shot so it does not flicker |
| `match_region` | *around the subject (swap)*: measured on a ring of background around the mask, so a new person keeps their own colours. *inside the subject*: measured inside the mask, for retouching the same content |
| `seamless_edges` | Poisson blend (OpenCV) for stubborn seams; slower |

**Which mode.** A generation mask alone keeps the background exact but generates the person at the frame's own size,
so a small person gets few pixels; a crop alone gives the person more resolution, but the whole box is regenerated
and pasted back (that is what the finishing above is for). *Mask only* is the simplest when the person fills a good
part of the frame; *Crop + mask* is best for small people. In the mask modes the latent starts from the shot's own
frames and its SAM 3 mask (grown by *expand* plus one latent cell) becomes the H3 generation mask. Raise *expand*
when the new person is bigger than the old one (longer hair, wider body): the generated area cannot go past the
grown mask; their shadow or reflection outside the mask stays from the original.

Credit to **Neko (Nekodificador)** and **AbleJones**, whose workflow and nodes the crop + generation mask follows.
ComfyUI builds without native H3 generation masks need a per-row mask patch on the model
(e.g. ComfyUI-MiniMaxH3-PerRowMasking).

**Mask settings (every shot)**, at the bottom of the Shots tab, holds the global mask settings, with defaults that work as they are: fill holes on,
temporal expand 2 frames (less flicker), expand 16 px, feather 12 px, padding 15 %, paste the whole box with feathered edges, blockify off
(16 aligns the mask to H3's latent grid), threshold 0.5, 4 objects. *show mask previews* (top of the Shots tab) overlays the preview
on the shot cards. SAM 3 uses the official `sam3.1_multiplex_fp16.safetensors` in `models/checkpoints`,
downloaded from Comfy-Org/sam3.1 the first time.

## VLM suggestions

Connect a vision-language model to the planner's `vlm` input (CLIPLoader with `qwen3vl_4b` or `qwen3vl_8b`).
It looks at a few frames of every shot and suggests what to segment, a description of the shot (camera,
framing, action). In the **Pictures & prompt** tab (*VLM*): *Use the VLM when the workflow runs* applies the
suggestions at run time (selection text for shots without one; `{shot}` in a prompt becomes that shot's
description), *Analyse shots* asks from the panel (after one run with the VLM connected), and each shot's
editor shows its suggestion with buttons to apply it. The summary output lists them too.

## Describe references ({details})

The VLM can describe the references, for any task (not tied to swaps), and any subject: a person, a creature, a robot,
an animal or a cartoon character (it says what the subject is first). In the **Pictures & prompt** tab (*Describe references*) pick an instruction
preset: *object* (a noun phrase for a prop, product, vehicle...: what it is, shape, materials, colors, text), *short* (the default: one noun phrase of at most ~20 words that fits inside a sentence, e.g. `green goblin with
long pointed ears in a torn brown leather vest`, for `<Subject 1> is the {details} shown in <Picture 1>`), *full body* (face, hair, skin, age, build, clothing piece by piece), *head / face*, *face attributes* (a
short comma-separated list), *outfit*, or *custom* (your own instruction). **📝 Describe refs** writes one
description per picture (see below), saved in the plan and editable. Write **`{details}`** anywhere in a prompt and
every shot gets the description of its own first picture; a picture without a description is described by the VLM
at run time (no click needed).

**Image max side (VLM)** controls the longest side of reference and shot images prepared for VLM analysis (1024 px by
default). Aspect ratio is preserved; the model's own image processor may resize them further. Increase it when fine
reference details matter, or lower it to reduce analysis cost. This does not change the generated video size.

When a shot has two references, the planner expands a one-reference template at run time to include `<Subject 2>` and
`{details2}` for `<Picture 2>` (not when the prompt already names `<Picture 2>` or uses `{details@Picture n}`). 
Existing one-reference templates remain unchanged in shots with only one reference. You can also use `{details2}`
explicitly in any template.

### 🤖 Auto prompt (1.113.0)

**Visual edits and color bindings (1.115.4):** the guide fixes the assigned performers' motion, action timing,
head movement, gaze and expressions. Explicitly request the target style, background, lighting, weather, outfit
or camera changes. For example: `Replace the man with the goblin from <Picture 1>, render the whole scene in anime
style, and use the hall from <Picture 2> as the setting. Keep the guide performance and timing.` A setting can also
be described in text. Use **full frame** generation for changes to the entire scene; masked generation only changes
the generated region.

References can have different roles. In the example above, Picture 1 defines the character and Picture 2 defines
the environment. To describe the latter, choose **scene / setting** in that picture's description preset. For several
characters, bind colors only to their character references: `The subject marked in green uses <Picture 1>; the subject
marked in purple uses <Picture 2>; <Picture 3> defines the setting.` Picture indices stay unchanged. A marking color
identifies the same performer across position changes, occlusion and cuts; it is not a target skin/clothing color.
Facial highlights are performance annotations. Prompt wording is generic and does not require a specific 2D/3D marking
implementation. This update changes prompt writing, not the existing marking renderer or tracking behavior.

After updating, restart ComfyUI, hard-refresh the browser and click **Write all prompts** to regenerate previously
written prompts with these rules. Existing written prompts are preserved until rewritten.

**1.115.5:** a setting replacement alone does not request a new rendering style. The writer preserves the source
medium unless the actual request or an explicitly designated style reference changes it; instructional examples
must not supply unrequested target attributes. For background replacement, use full-frame generation, disable
paste/crop-only composition and inpainting outside a subject mask, set **first_frame** to **none**, and leave
**setting_ref** off (that option adds the original source location). The full aligned guide still contains source
scene geometry, so a correct prompt alone does not guarantee complete background replacement by a checkpoint.

In **Pictures & prompt → 🤖 Auto prompt**, *Write the prompts with the VLM* turns the prompt boxes (global and per shot)
into **requests** ("replace the bottle with the red can from Picture 2", "Picture 1 dances in the rain"), and the VLM
writes each shot's MiniMax H3 prompt in the official full-reference format (MiniMaxAI/MiniMax-H3
`docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`): `subject_definitions`, `summary` with its task type, `retention_analysis`,
`detailed_description` (`[Shot 1]`, the official camera vocabulary, sounds in time), `overall_soundscape` and
`non_diegetic_music`. It looks at:

- **key frames of the shot** with their times (*Key frames*: auto = about 1.5 per second, 3 to 8);
- **the shot's pictures** and their descriptions (Describe references);
- **the request**, the shot's **Colored Marks** (which tinted person becomes which picture; the mapping is added if the
  writer leaves a color out), its selection / `{target}` and its mode (*Mask only*: only the subject is regenerated;
  *Crop*: the frames are a crop);
- **how the source is given**: *guide video* (aligned guide, no tag) or `<Video 1>` (native reference video) — match the
  conditioning node's `guide_mode`.

**Template:** with Auto prompt on, the planner's `prompt` input (e.g. the template a LoRA was trained with,
`body_swap: subject_definitions: ...`) is the **template** and the panel's text is the **request** applied to it
("Troque a mulher no vídeo pela mulher da <Picture 1>"). The template is kept as it is: same sentences, same order, its
trigger word first; the writer applies the request, adds the pictures' visible features and, after `[Shot 1]`, this
shot's camera, place, light and actions. The panel's 🤖 Write uses the template of the last run (shown in the card).
A request that is itself a whole prompt (H3 sections) is used as the template the same way. **Style** picks how a prompt without template is written: *H3
official* (the six sections) or *OpenJutsu captions*, the short style of the OpenJutsu training captions ("Replace the
dog in the video with the gaunt goblin … from <Picture 1>. Replace the cat … from <Picture 2>. Keep the same background
and location as in the video, and maintain the motion, camera angle, and timing."). Without a template, a
short request gets the full six-section prompt. Without a VLM, the template (else the request) is the prompt.

**Cuts:** the writer gets the shot's real camera cuts (the planner's cut detector) and writes `[Shot N] At MM:SS.mmm`
exactly there, with at least one key frame in each piece between cuts; a shot with no cut gets no timestamp at all.
Labels always keep their `< >`.

**🤖 Write all prompts** (or *only the missing*) writes them now; each shot shows its written prompt in the Shots tab,
editable, with **🤖 Write / Write again**. A shot without one is written when the workflow runs (with the VLM on the
planner, before any SAM 3 work) and the panel keeps it, so the next run uses the same text. When the request,
pictures, marks, selection or mode change after a prompt was written, the shot says it is *older*; it is still used
until you write it again. A written prompt is part of the stored shot's signature, so editing it re-renders the shot.
Without a VLM and without a written prompt, the request is used as the prompt.

### One description per picture (1.113.0)

Descriptions are per picture: **📝 Describe refs** describes every picture the plan uses by itself, once per image.
`{details}` is picture 1's description, `{details2}` picture 2's and **`{details@Picture n}`** (or
`{details@Picture3}`) picture n's, so each shot gets the descriptions of *its* pictures. The dropdown next to the button
sets what is described for every picture; a picture can have its own (e.g. *full body* for a character sheet, *head /
face* for a close-up, *outfit*, or *custom* with its own instruction), and **🔄** describes only that picture again. A
picture without a description is described at run time by the planner's VLM, else by the conditioning node's VLM, else
it becomes "character". Descriptions saved by older plans (per reference set) are read as those pictures'.

**Fit the prompt** (on by default, next to 📝 Describe refs): the VLM gets the sentence of the prompt that holds the
marker, with a blank in its place (e.g. `<Subject 1> is the ___ shown in <Picture 1>`), and writes the description for
it, so the description adapts to the prompt and not the other way round (no "the A goblin", no paragraph inside a
sentence). The panel buttons use the global prompt (or the shot's own); at run time the prompt the shot really uses.

The VLM buttons in the panel work after the workflow has run once with the VLM connected (ComfyUI only hands
models to nodes when they run). Tested with Qwen3-VL 8B; some smaller or modified models return empty answers
for some wordings: the planner retries with a reworded request and with sampling, but if a preset stays empty,
use the 8B or another preset.

## Settings reference

Every setting in the panel has a hover tooltip (ⓘ). The main ones:

| setting | what it does |
|---|---|
| Mode | Camera cuts (shots start at cuts, long ones split evenly), fixed length, or one shot (the whole range) |
| Detector / Sensitivity | PySceneDetect adaptive or content, or the built-in one; higher sensitivity finds more cuts |
| Frame grid | frame counts the model accepts (H3 17n+5, LTX/Wan 8n+1, Wan 4n+1) |
| Max / Min seconds per shot | longest shot sent to the model; shorter shots merge into a neighbour |
| Timeline fps | the frame rate the planner works on (generation and audio use it) |
| Max shots | test on the first shots only |
| Generation size | Megapixels (default), source video resolution, or custom width and height; dimensions snap to the size multiple |
| Run | one run, shot by shot (default) / queue loop (one shot per run) / auto loop (all shots at once) |
| Skipped shots in the output | keep the original video there or remove those shots |
| Mask: padding / expand / feather / temporal expand / blockify | crop context, paste-back growth and softness, flicker hold, square blocks |
| Mask: threshold / max objects / paste back | SAM 3 text matching, objects tracked, paste only the mask or the whole box |
| VLM: frames per shot / max tokens / image max side | how much the VLM sees, its answer limit, and the longest side of images sent for analysis |

## One seed per shot: BFS Shot Seed (1.94.0)

A seed widget on the sampler gives every shot the SAME seed (the node runs once per shot with the same value).
**BFS Shot Seed** (shot in, seed out) makes it a choice: *same for every shot*, *increment per shot* (seed, seed+1 ...)
or *random per shot* (unrelated but repeatable seeds from the base seed). `overrides` pins single shots, e.g.
`3:12345, 5:777`, to keep the shots that came out well and re-roll only the others.

## Structured starting noise: BFS Shot Noise (1.94.0, experimental)

Optional. **BFS Shot Noise** gives the NOISE for SamplerCustomAdvanced; with every amount at 0 it is exactly RandomNoise.

| Input | What it does |
|---|---|
| `shot_seed` (+ `shot`, `overrides`) | the seed of each shot: same / increment / random per shot, as BFS Shot Seed |
| `shared_base` | the coarse pattern of the noise comes from `base_seed`: the same for every shot, at any resolution and any length (it is a tape along the timeline); the fine detail stays per shot. Put the seed of a result that worked in `base_seed` to keep its coarse noise |
| `distribution` | gaussian (default), laplacian, student-t or uniform, all at unit variance |
| `spectrum_tilt` | 1/f^alpha spectrum: > 0 puts more energy in the large blobs ("fractal" noise), < 0 in the fine grain |
| `pyramid` | coarser layers of noise added on top |
| `static_share` | a part of the noise identical in every frame |
| `guide_structure` (`vae`, `guide_frames`, `mask`) | outside the mask, the guide's latent is mixed into the noise |
| `ref_tint` (`vae`, `ref_image`, `mask`) | inside the mask, the noise leans toward the reference's average latent |

All mixes keep unit variance. The model was trained on plain noise: use small amounts and compare with 0.

## {details} filled automatically (1.92.0)

A `{details}` in a shot's prompt is filled without pressing 📝 Describe refs: the planner's VLM describes the shot's
references before any SAM 3 work; with no VLM on the planner, the slot is left for **BFS Shot H3 Conditioning** or
**BFS Shot H3 Duet**, whose `vlm` input (the same Qwen3-VL CLIPLoader) describes them (the *short* preset, fitted to
the sentence). Edited descriptions always win. With no VLM anywhere it becomes "character" and the log says so.

## Source blur (1.91.0, finer in 1.110.0)

`source_blur` on **BFS Shot H3 Conditioning** (and its Simple version), **BFS Shot H3 Duet**, **BFS H3 Duet** and
**BFS H3 Duet Conditioning** blurs the source person in what the model *sees* of the source (aligned guide, native
reference video, duet panel): the model still reads the pose and the motion but has no face to copy, so the identity
comes from the reference. The kept area of *Mask only* and the output are never blurred. Trick shared by JalenBrunson
(Sapiens2 there) for reliable ref2v swaps.

Modes read `<region> (<what stays sharp>)`:

| region | what is blurred |
|---|---|
| `face` | the face (and neck) |
| `head + hands` | also the hair and the hands |
| `person` | the whole body: only for big changes, the pose then comes from the blurred shape |
| `masked area` | the shot's own mask (the person being replaced, with its expand); no extra segmentation. Without a mask on the shot (or in the plain Duet nodes) it blurs the whole person |

| sharp | |
|---|---|
| `keep mouth` | lips, teeth and tongue stay sharp (the lips carry a lot of the likeness and the lip sync) |
| `keep lips + tongue` | the teeth are blurred too (always uses Sapiens2 for the lips) |
| `mouth too` | nothing stays sharp |

`source_blur_segmenter`: **SAM 3.1** (text prompts, already used by the planner; the 1.91 behaviour) or **Sapiens2**
(Meta's body-part segmentation, 28 classes with lips, teeth and tongue apart from the face; one pass for every part,
finer around the mouth). Sapiens2 marks every person in the frame, like the SAM prompts here. Its weights
(`facebook/sapiens2-seg-0.4b`, 1.6 GB) download to `models/sapiens2/` the first time; it runs on the GPU in half
precision and goes back to RAM after each shot. Sapiens2 License.

`source_blur_effect`: **blur** (no detail left) or **invert colours** (keeps the shapes and the expression but not the
skin tone or the colours; some users get better likeness with it).

## Big videos (upload)

ComfyUI refuses any single request bigger than `--max-upload-size` (100 MB by default), which is why big videos fail
to upload in most nodes. The planner's ⬆ Upload (source video, mask videos, pictures) sends files of 32 MB or more in
8 MB pieces that are written straight to disk and assembled in the input folder, so any size works without changing
how ComfyUI is started (a dropped piece is sent again). For other nodes, start ComfyUI with a higher limit, e.g.
`--max-upload-size 4096`, or copy the file into `ComfyUI/input` and pick it from the list.

A newly selected or uploaded source starts as **one shot covering at most its first 15 seconds**. This is only the
initial plan: widen the range or choose *Camera cuts* / *Fixed length* and press *Auto split* when you want multiple
shots. Existing saved plans keep their own range and split mode.

## Continuity between shots

Each shot (after the first) can continue from the **previous shot's generated result**: in the shot editor,
*Continuity* = *previous shot as reference* adds a frame of it as one more `<Picture n>` after the shot's own
references (good across camera cuts, keeps the person consistent), and *previous shot as first frame* anchors it
at frame 0 of the shot (for continuous action without a cut). Pick which frame: the previous result's first,
middle or last. **Copy to other shots** with *continuity* ticked applies it to every shot after the first.

It needs the previous result to exist: it works **shot by shot** and in the **queue loop** (the planner reads the stored result) and in
the **auto loop with BFS Shot H3 Duet** (it renders shot by shot and keeps the last result). BFS Shot Unpack also
outputs it as `previous_result`. This is different from H3 Conditioning's *first_frame*, which anchors the shot's
own frame from the source video.

### History: a long take, window by window (1.87.0)

*Continuity* = **previous shot's end as history** carries the **last 5, 22, 39 or 56 frames of the previous shot's
result** into the next shot as clean frames placed **just before frame 0 on H3's timeline**: the model sees them as
the immediate past of the video it is generating, so motion, camera, light and the new look continue across the join
(instead of one still frame). Idea from Akatz's [H3 Relay](https://github.com/akatz-ai/h3-relay) windowed edit and
Ethanfel's MiniMax H3 Context Loop; this is BFS's own implementation (a process-local patch of H3's packed layout
that moves the video, its audio and the other guides past the history; ComfyUI files are not edited).

- Plans saved before keep what they had; *head* (below) is now the default for new shots.
- **Same take only.** A shot that starts at a camera cut skips it (nothing to continue). For one long take, split it
  with *Mode = fixed length* (or let *Camera cuts* split long shots) and set history on every shot after the first
  with **Copy to other shots → continuity**. Shots still exist: they are the windows, and cuts reset the history.
- **Every mode:** full frame, *Frame + paste*, *Mask only* (the mask regenerates the target; the history is a
  condition outside it), *Crop* (the history is cut with this shot's box) and the **duet canvas** (the source frames
  under the history go into the panel strip). Not with duet *shifted RoPE*. The aligned guide still drives this shot's
  own frames, so sync stays frame-exact.
- **BFS Shot H3 Conditioning and BFS Shot H3 Duet** both use it (duet canvas or aligned guide).
- **Shot by shot / queue loop:** shots run in order and each reads the stored result of the one before. Auto loop: taken
  from the previous render when it matches.
- Video only for now (the previous audio is not carried). History can use 5, 22, 39 or 56 frames; longer windows use more memory.
- **Only 17k+5 frames work.** H3's VAE encodes 17-frame clips into 5 latent steps and drops the last 3 steps of the
  whole video, so 17 frames come back as just their *first 5*: the old 17 / 34 / 51 options left a 12-frame (0.5 s)
  hole between the history and the shot, seen as a jump or drift at every join. Plans saved with 17 / 34 / 51 are
  read as 22 / 39 / 56.

### Head: the shot starts on the previous result (default)

*Continuity* = **previous shot's end as this shot's start** generates the shot from **5, 22 or 39 frames earlier**:
those first frames are the previous shot's result, encoded **into this shot's latent** and kept by the sampler
(noise mask 0), so they are **not generated or swapped again**, and the new frames continue straight from them.
BFS Shot Join cuts them off, so the timing is unchanged. Same masked-prefix continuation as the Context Loop's
*masked AV* mode, whose tested default is 22.

- **On by default, 22 frames**, for every shot after the first. Set *Continuity = off* (or *history*) to change it.
- **Same take only:** skipped at a camera cut, after a skipped shot, and on grids other than H3's.
- **Shot by shot or queue loop.** In the auto loop the previous result does not exist yet when the shot is conditioned,
  so the planner generates the shot without a head (and says so in the console).
- Works with full frame, *Frame + paste*, *Mask only* / *Crop + mask* (the head is kept whatever the mask says), *Crop*
  (the head is cut with this shot's box) and the duet canvas. Not with duet *shifted RoPE*. Pinned in
  **BFS Shot H3 Conditioning**; other workflows still generate and cut the extra frames.
- Costs those frames of generation per shot (a 60-frame shot is generated at 90 frames instead of 73).
- **Keep colors** (BFS Shot Join, on by default). A shot that starts on the previous result comes out a little more
  saturated right after it, and the next shot inherits that: measured on a 5-shot take, the new subject's saturation
  went 75 → 88 → 105 → 124, with almost no change *inside* each shot (the jump is at every join). With *keep_colors*
  the first shot of each take records its subject's colours, and every shot that continues from the one before has
  its subject's chroma (CIELAB a, b: one transform per shot, so no flicker) matched to them before it is stored; the
  next shot then starts from the anchored colours and nothing accumulates (same take: 69–73 on every shot). The
  subject is where the result differs from the source video (soft edge); the background and the lightness are not
  touched. A camera cut or a shot without continuity starts a new take (new anchor). *keep_colors_strength* blends it
  (1 = fully). It applies to stored runs (shot by shot, queue loop), where continuity exists.

## MiniMax H3 with an aligned guide (example)

```
BFS Shot Planner ── shots ──► BFS Shot H3 Conditioning (clip, vae, guide_mode = aligned guide)
                                 └─ positive, latent ─► CFGGuider / sampler ─► VAE Decode ─┐
BFS Shot Join ◄── images ─────────────────────────────────────────────────────────────────┘
      └─► Create Video (fps, audio) ─► Save Video
```

For other models, use **BFS Shot Unpack** and wire `guide_frames`, `refs`, `prompt` and
`length` into that model's own guide and reference nodes; the list mapping works the same way.

## Choosing who is replaced (`{target}`), step by step

With several people on screen, a prompt that says "replace the person" leaves the model to guess. `{target}`
puts a short description of the right person into the prompt of every shot ("the young woman with dark hair in a
pink top"), so the model knows WHICH one to swap. The selection is made with SAM 3, and the description by the
VLM, so it works for anyone or anything you can click on.

### What you need

- The planner's **`vlm`** input connected to a Qwen3-VL (CLIPLoader, type *stable_diffusion*, `qwen3vl_8b` recommended).
- The workflow run **once** with it connected. ComfyUI only hands models to nodes when they run; after that the
  panel buttons can use it.
- Without a VLM it still works: you get the cut-out and type the description yourself.

### 1. Put `{target}` in the prompt

Write `{target}` wherever the prompt names the person being replaced: in the global prompt (all shots) or in a
shot's own prompt. For example, the swap caption the H3 LoRAs were trained with:

```
... only the face, body and clothing of {target} are replaced.  summary: [video generation + reference] Replace
{target} in the guide video with <Subject 1>, keeping the scene, camera and the complete motion and facial
performance of the guide video.  detailed_description: [Shot 1] <Subject 1> takes the exact place of {target}
and moves with the same timing, body pose, gestures, ...
```

`{target}` can be combined with `{details}` (the description of the references) and `{shot}` (the VLM's
description of the shot).

### 2. Select the person

Open a shot and click **🎯 Points…**. The modal shows one frame of the shot; the slider below picks the frame.

- **Click on the person's body** (torso), not only the face. A click on the face selects the head, and the
  description then misses the clothes.
- **Right-click** (or shift+click) puts an *exclude* point, e.g. on a second person standing close.
- **👁 Segment** shows what SAM 3 picked (red). Add or remove points until only that person is red.
- Instead of points you can just type what to segment (`woman in pink`, `man with glasses`, `dog`): no clicking is
  needed, SAM 3 finds and tracks it through the shot. Points are for when the text is ambiguous (two similar people).
  Points win when both are set. **Copy to other shots** with *mask text* (and *mode*) ticked gives the text (and
  the generation mode) to every shot.

Then save:

| Button | What it does |
|---|---|
| **Save (this shot)** | keeps the selection on this shot only |
| **Save → all shots** | applies the same points to every shot, each on its frame at the same relative position as the frame you clicked on (e.g. clicked in the middle of this shot → every shot uses its middle frame) |

**Save → all shots** is for a person who stays in the same place across shots (a sequential video, a fixed camera).

**🎯 Track → whole video** (in the points window) is for a person who moves around: the selection is tracked by SAM 3
through the whole video (or the range), forward and backward from the frame you clicked on, and every shot without
its own selection takes its part of that track. Also type a short text (e.g. `woman in pink top`): after a camera cut
the tracker uses it to find the subject again. **Mask settings** (Shots tab) shows the track and removes it. The first run
tracks the whole range (a minute or two per few hundred frames); the result is cached until the selection changes.

### 3. Describe the person

Click **🧑 Describe** next to `{target}` in the shot's editor:
- SAM 3 cuts the person out of the selected frame, and a thumbnail of the cut-out appears next to the field.
- The VLM writes a short phrase into the **`{target}`** field.
- The field stays editable: correct it or write your own. Keep it short and visual (who, hair, main clothes with
  colours) and start it with "the".
- **Copy to other shots** with *{target}* ticked copies the description to every shot. Use it when the same person is replaced throughout;
  otherwise describe shot by shot.

### 4. Check and fix

- **👁 Preview** on a few shots shows the selection on several frames of each shot.
- In a shot where the selection picked the wrong thing, **✕ Clear** removes its points, text and mode. Select
  again in that shot only (**Save (this shot)**) and describe again.
- **Copy to other shots** with *mask points* ticked copies that shot's points to every shot without opening the modal.

### What happens when the workflow runs

Every shot's prompt gets its own `{target}` text: the one you wrote or made with 🧑 Describe; when the field is
empty but the shot has a selection (mask text, points or a mask video) and the VLM is connected, the subject is cut out
and described automatically (written for the prompt's `{target}` sentence). A shot with neither gets "the person". The selection is
only used to make the description: it does not crop or mask anything unless the shot's mode is *Mask only*, *Crop*
or *Crop + mask*.

### Examples

- **Two people, swap only one:** in the first shot, click on the woman's torso, put an exclude point on the man,
  then **Save → all shots** and **🧑 Describe** → "the young woman with long black hair in a white blouse".
  **Copy to other shots** with *{target}* ticked.
- **The person changes between shots:** describe shot by shot, e.g. shot 1 "the man in the grey suit", shot 2
  "the woman in the red dress". Each shot keeps its own text.
- **Animals:** it works the same way, e.g. "the small black dog in front".

### Troubleshooting

| Problem | Fix |
|---|---|
| "No VLM yet" | connect the `vlm` input and run the workflow once, or type the description |
| "Select the person first" | the shot has no points or mask text: use **🎯 Points…** |
| "SAM 3 found nothing" | click again on the person (on the torso), or try another frame with the slider |
| Only the face or the hair is described | the click was on the head: click the body |
| The other person is described | add an exclude point (right-click) on the other person, **👁 Segment**, save, describe again |
| The swap still hits the wrong person | make the description more specific (position: "on the left", clothing colours) |

### Cancel a pending redo

After marking a stored shot for redo, click **Undo redo** to keep its previous take.
**Clear redo selection** cancels all pending redo selections and preserves stored files and loop progress.
Once a shot has been regenerated, its previous take cannot be restored through this control.
**Refresh status** rereads the cache status; it cannot recover files deleted by older versions of Reset loop.
Changing the seed, model or other render settings still invalidates stored shots even after cancelling redo.

### Experimental native temporal motion transfer

`BFS Shot H3 Conditioning` now has **native temporal alignment (experimental)** in `guide_mode`. Connect MODEL in and route its MODEL output to the guider. This uses the matching experimental LoRA, the shot as native `<Video 1>`, temporal RoPE alignment and soft temporal attention; it does not use Add Guide. Use 24 fps and disable continuity/first-frame/mask guides and VSA. Simple appends a MODEL output without moving existing outputs. The standalone **BFS MiniMax H3 Native Temporal Alignment (Experimental)** node provides the same patch for native workflows. See `H3_NATIVE_TEMPORAL.md` for the full contract and validation limits.

Native temporal performance (1.115.11): `native_temporal_backend` defaults to `flash partitions (no compile)`, preserving the same temporal penalty. `native_guide_scale` can reduce only guide resolution (try 0.5); output dimensions and frame count stay unchanged. The original Flex backend remains available for comparison.

Prompt bindings (1.115.12): completed short captions are no longer expanded into legacy two-subject templates. Auto-written prompts whose request/reference/mark signature is outdated are regenerated with a connected Planner VLM; otherwise use Write again before rendering. The writer rejects picture indices beyond the connected references after its bounded retry. Character outfits come from identity references, while requested setting changes override conflicting source scenery/lighting.
