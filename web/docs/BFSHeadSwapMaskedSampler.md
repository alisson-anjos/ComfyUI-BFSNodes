# BFS Sampler (crop · mask · loop)

Runs a guide-driven LTX IC-LoRA and adds three things around it, each optional.
Connect only what you need — the node degrades to whatever is wired.

| what you connect | what you get |
|---|---|
| `guide_video` | the LoRA over the whole clip, one pass |
| `+ identity_image` | for a LoRA that takes a reference (head swap, identity transfer) |
| `+ subject_mask` | only the masked region is denoised |
| `+ crop_mode` | the edit runs inside a box around the subject, then is pasted back |
| `+ temporal_tile_size` | long clips sampled in chunks with overlap |
| `+ auto_config` | crop, mask and feather amounts measured off the subject itself |

`identity_image` is optional as of 1.38.0. A head-swap LoRA wants one; an
instruction-edit, restyle or sharpening LoRA works from the guide alone, and
with the slot empty it is simply not packed — the guide becomes the only
reference. Everything else in the node is indifferent to which kind it is.

The LoRA is never asked to understand a mask or a crop. It keeps doing its one
job: the guide rides on `source_id 1`, the identity on `source_id 2`. Everything
else here is plumbing around it.

## `auto_config`: let the mask set the amounts

Every amount in this node is in pixels — `mask_grow`, `mask_blur`,
`uncrop_feather`, and the latent dilation — and a pixel only means something
relative to how big the head is in the frame. The same `mask_grow: 8` is
generous slack on a head 90 px wide and invisible on one 900 px wide. That
mismatch is what leaves a visible seam when the crop goes back into the frame:
the amounts were tuned on one shot and the next shot has the subject at a
different distance.

Turn `auto_config` on and the node measures the subject in `subject_mask` — its
width, its height, how far it travels, how much its size changes, how fast it
moves between frames — and sets all of it from those numbers:

| knob | derived as | why |
|---|---|---|
| `crop_mode` | `off` when the head is ≥45% of the frame width; `zoomed` when its size changes >35%; `tracked` when it travels more than a quarter of the box; else `combined` | a crop buys resolution the head does not already have, and nothing else |
| `crop_scale` | 1.8, clamped so the box still fits in the frame | head plus neck and shoulders — a face-tight crop is a framing the LoRA never saw |
| `mask_grow` | 6% of the head's **width**, plus the capped headroom slack — **per side**: up ≫ sideways ≫ down | hair overflows upward and sideways; growing down only eats the neck and collar, which stay the guide's |
| `mask_blur` | 3% of the head's width | paste-back softness (the denoise mask is binarised anyway) |
| `uncrop_feather` | half the margin between mask and crop border, floored | **the seam fix**: a ramp wider than that margin fades the head itself |
| `latent_mask_dilate` | left at 0 | the pixel grow already carries the slack, and the pixel→latent reduction is a MAX, so any cell the mask touches is editable already — adding cells here stacks a whole 32 px on every side |
| `latent_mask_dilate_frames` | 1 when the head moves more than 15% of its width between frames | a fast head needs slack along time too |

### The mask is the old head, not the new one

This is the failure the mode is really for. `subject_mask` marks the head that
is being *replaced*. The reference head can be wider, or have more hair, or a
taller cut — and wherever it lands outside the mask, the swap is cut off at the
mask edge. Growing the mask by a fixed 6% does not fix that; it just moves the
clip.

Two things are done about it. The reference's **proportions** are measured —
`identity_image`'s aspect against the mask's own — and used to decide *where*
the slack goes: a relatively wider reference gets it at the sides, a taller one
above. The reference's **size** cannot be measured at all: a cropped head
carries no scale, and nothing in the graph says whether that crop is a 200 px
head or a 900 px one. That part is `identity_headroom`, default 1.15 — "the new
head may be up to 15% bigger than the old one". Raise it for big hair or a
visibly larger head, drop it to 1.0 when the two heads match.

Growth is never symmetric: down is held at the base 6% no matter what, because
the neck and collar below the jaw have to stay the guide's own pixels.

Every slack is a fraction of the head's **width** and capped there (25% sideways,
30% up). Width is the stable dimension of a head mask — height swings with how
much neck the mask happened to take — and scaling the upward slack by height,
as 1.37.0 did, produced holes far larger than the head, with the new head
generated floating inside the regenerated area. Fixed in 1.37.1.

It needs `subject_mask`; without one there is nothing to measure and the widget
values are kept. The widgets it overrides stay visible but are ignored, and the
`debug` output prints exactly what it chose:

```
auto: head 120x160px in 1920x1080 (6% of the width), seen in 73/73 frames;
reference aspect 1.00 vs mask 0.75 (tilt 1.33), headroom 1.15 -> crop tracked @ 1.8
(the subject travels 1296px, more than a quarter of the box); grow 19px sideways,
32px up, 7px down, blur 4px, feather 14px, latent dilate 1 cell(s), 0 frame(s)
```

Read that line before trusting it. It is a starting point measured from your own
shot, not a guarantee — a mask that leaks onto the shoulders makes the "head"
wider and every amount grows with it.

## The mask is native inpainting, not a LoRA feature

The mask becomes `latent["noise_mask"]`, which the guider consumes as
`denoise_mask` — ComfyUI's own inpainting path. Pixels outside the mask keep the
guide's own content, so **the original face stays visible to the model**. This is
deliberately different from a masked-hole recipe, where the region is painted out
before the model sees it.

When a mask is connected the node seeds the sampling latent with the
**VAE-encoded guide**, not with the empty latent you wired in — inpainting keeps
the initial latent outside the mask, and an empty one leaves grey where the video
should be. The connected latent still sets the size (and carries the audio
stream); its video content is replaced.

Masks are reduced to the latent grid with **max**, not with ComfyUI's trilinear
resize. Trilinear blurs a mask across frames and lets the original content bleed
through the edit. Max keeps a latent cell that any masked pixel touches fully
editable.

### Does the mask need to be semi-transparent so the LoRA can see the expression?

No — and it would cost you the swap. The expression does not travel under the
mask: the whole source clip, original face included, is in the conditioning as an
aligned reference on `source_id 1`, frame by frame. The model reads the
performance from there whatever the mask does. That is what the LoRA was trained
on: the new head follows the old head's acting.

Partial mask values do something else. In ComfyUI's inpaint path they blend the
original latent back in, so you keep the original geometry in pixels **and the
original identity with it** — the result drifts toward an average of both faces.
That is what `mask_strength` below 1.0 buys, and it is a deliberate choice, not a
way to get expressions. Soft edges are for hiding the seam; the middle of the
mask should stay at 1.0.

### Hard for the sampler, soft for the paste

`mask_blur` exists for compositing, where a soft edge hides the seam. In the
**denoise** mask a soft edge means *partial* denoising, which blends the original
latent — and the original identity — back in right at the edge of the head, so
the swap is weakest exactly where it is most visible. `mask_hard_for_inpaint`
(on by default) binarises the mask before it becomes the denoise mask, leaving
the soft version for the paste-back alone.

`latent_mask_dilate` grows the mask by whole **latent cells** after the
reduction. One cell is 32 px and one latent frame covers 8 video frames, so this
is far coarser than `mask_grow` in pixels — and it is what guarantees the head
sits inside editable blocks rather than clipping at a cell boundary. One cell
took a test mask from 11% to 25% of the grid; two took it to 44%. Watch
`latent_mask` and stop at the first value that covers the head.

**Watch the `latent_mask` output.** It has one frame per *latent* frame, not per
video frame, and that is the real resolution of the edit. A tight outline around
a face can nearly vanish at that scale — and then nothing changes, no matter how
good the LoRA is. If the mask looks thin there, raise `mask_grow`.

## Applying a LoRA to one region and putting it back

The shape of the thing: cut the region out as its own small clip, run the LoRA
on it, composite it back. What makes the result honest is that the composite
goes into the **original frames**, so every pixel outside the mask is the
source's own — it never went through the VAE at all.

```
subject_mask            connected
inpaint_with_mask       true     # outside the mask the latent IS the guide, not noise
mask_hard_for_inpaint   true     # a soft DENOISE mask means partial denoising at the edge
mask_strength           1.0      # below this the original is blended back INSIDE the mask
paste_back              true
paste_confine_to_mask   true     # only the mask reaches the frame
latent_mask_dilate      0        # one cell is 32 px of extra regenerated area, per side
mask_grow               the least that fits the new content
mask_blur               3-5% of the subject's width — the paste's soft edge
```

Since 1.39.0 this works with `crop_mode: off` too. Before that, with no crop the
node returned the decoded frame whole: nothing *changed* outside the mask, but
everything had been through an encode/decode round trip and came back softer.
Now the mask composites against the source either way, and the crop goes back to
being purely a resolution decision.

**Two passes on the region.** Set `paste_back: false` and the images and latent
stay in the crop's own space, so you can upscale and refine there — where the
region actually has pixels — and composite at the end with **BFS Paste Back**,
feeding it the `crop_bboxes` output. That is the whole point of `crop_bboxes`
being a socket.

## Why crop, and which mode

At 512×288 a person filling a fifth of the frame leaves a face about 25 px tall.
No LoRA recovers identity from 25 px — it is missing pixels, not missing
training. Cropping the head region and sampling it full-frame gives that same
face 200–300 px, and the result is feathered back into untouched frames.

| `crop_mode` | behaviour |
|---|---|
| `off` | the whole frame is sampled. Use it first, to confirm the swap works at all. |
| `combined` | one static box around the subject's whole travel, held for the clip. No dependency. |
| `tracked` | constant-size box that stays still until the subject would leave it, then moves as little as possible. |
| `zoomed` | the box also follows the subject's size, planned over the whole clip, every crop resampled to one output size. |

`tracked` and `zoomed` use a planner vendored from
[drozbay/MaskVidExperiments](https://github.com/drozbay/MaskVidExperiments)
(GPL-3.0, the same licence this pack carries) — nothing extra to install. Those
boxes hold still through mask noise and occlusion, and **a jittering crop reads
to a video model as camera motion**, which is what a naive per-frame crop around
a mask produces.

`crop_scale` is the box as a multiple of the subject. At 1.5 the subject occupies
two thirds and the rest is margin. Keep neck and shoulders inside: a face-tight
crop is a framing the LoRA never saw in training, and it will show.

## Sizing, which is the one fiddly part

**Since 1.41.1 the shape is handled for you.** The crop is resized into the
connected latent, so the box has to share its *aspect* — the sampler now asks
the planner for exactly the latent's aspect ratio and says so in `debug`:

```
crop shaped to the latent's 1.500 aspect
```

Without it a mask around a face gives a portrait box, sampling it in a landscape
latent stretches it, and the paste-back squeezes the result back — the face comes
out deformed, and under `crop_mode: zoomed` by a different amount every frame.

The latent's **size** is deliberately left to you, because that is the sampling
resolution and it is the entire reason to crop: a face 200 px tall in the source
sampled at 768 px wide is a face with pixels to work with, and the paste-back
brings it home at the box's size. Choose the size for the quality you want; the
shape is no longer yours to get right.

**BFS Crop Size** is still there for when you want the number, or the per-frame
boxes, upstream of the sampler — building the latent at the right size skips the
rebuild, and the boxes feed BFS Paste Back directly:

```
subject_mask ──┬─→ BFS Crop Size ──→ width / height ──→ EmptyLTXVLatentVideo ──┐
               │                                                               │
               └───────────────────────────────────────────→ BFS Sampler ←─────┘
```

Keep its three crop widgets identical to the sampler's — the planner is
deterministic, so equal inputs give the same box, and a mismatch means the
sampler silently resizes the crop into a latent that does not fit it.

That resize is worth naming, because it does not look like a sizing problem when
it happens. A mask around a face in a 1920x1080 frame gives a portrait box —
512x672, say. Sampling it into a 768x512 landscape latent stretches the crop
horizontally, and the paste-back then squeezes the result back into the portrait
box. The face comes out visibly deformed, and with `crop_mode: zoomed` the
amount of deformation changes from frame to frame.

### Doing it by hand



The connected `latent` sets the sampled size. When you crop, the box is smaller
than the frame, so the latent should match the box — the `debug` output prints
the size to use:

```
sample size 416x768 (connect EmptyLTXVLatentVideo at this size to avoid a resize)
```

If they disagree the node resamples the crop on the way back, which costs exactly
the sharpness the crop was there to buy. Run once, read `debug`, set
`EmptyLTXVLatentVideo`, run again.

**The latent must come from the graph.** LTX‑2.5 latents are AV — video and audio
nested together — and the node cannot fabricate that structure. Without one it
builds a plain video latent, warns, and the guide may not reach the model at all.

## Chunking

`temporal_tile_size` at 0 samples the clip in one pass. Set it to the length the
LoRA trained at (73 for the LTX head-swap recipe) for longer clips, and the node
slices the guide, the mask and the overlap together per chunk.

Chunked sampling with an AV latent is refused rather than silently corrupted —
slicing a nested latent in time is not a thing the node does yet.

## Decoding

`decode` chooses how the sampled latent becomes frames:

- **full** — one shot. Right for a single pass at the size the sampler ran at.
- **tiled** — for a latent too large to decode whole, e.g. after a 2x upscaler.
  A full decode there pins VRAM at 99% and shuttles tensors until it looks like
  a hang.
- **none** — skip it and return the latent alone.

**For anything beyond a single pass, prefer `none` and decode outside.** Your own
VAE Decode (Tiled) is tunable, visible in the graph, and reused by the rest of
the workflow. The only reason decoding lives in this node at all is the
paste-back, which needs pixels — so with `none` the compositing is skipped too,
and **BFS Head Swap Paste Back** does it after your decode.

## Second pass at higher resolution

With cropping on, `latent` is the **crop's** latent, not the frame's — which is
what you want to refine, because that is where the face has pixels. To chain a
second pass:

```
Head Swap Sampler (paste_back: off)
   ├─ latent      → LTXV upscaler / second sampler → VAE Decode ─┐
   └─ crop_bboxes ───────────────────────────────────────────────┤
guide_video (original frames) ────────────────────────────────────┤
                                                                  ▼
                                              BFS Head Swap Paste Back → frames
```

With `paste_back: off` the node stops compositing, so `images` and `latent` stay
in the crop's own space. Refine there, then bring the result home with **BFS Head
Swap Paste Back**, which takes the crops, the original frames and `crop_bboxes`.
It does the same edge-aware feather and optional mask confinement the sampler
would have done.

Refining before compositing is the right order: upscaling the crop puts the
pixels where the identity is, and the frame around it never needed a second pass.

## Reading the outputs

| output | what it answers |
|---|---|
| `images` | the result, with the crop pasted back |
| `mask_over_source` | is the mask where I think it is? Red over the original frames, green box for the crop |
| `cropped_guide` | exactly what the model was fed. If this is wrong, nothing downstream can be right |
| `crop_mask` | the mask after grow/blur, in the crop's pixel space |
| `latent_mask` | the mask as the sampler sees it — where a thin mask disappears |
| `latent` | the sampled latent — the CROP's, when cropping is on |
| `crop_bboxes` | the boxes, for Head Swap Paste Back after a second pass |
| `debug` | crop mode and box, mask ops, tiling, the size to build the latent at |

## If a second run behaves differently from the first

The `guider` is an object another node builds, and ComfyUI keeps it in the
execution cache between runs. This node has to write the reference specs into
its options — the guider captured its own dict by reference when it was built,
so patching the model alone never reaches the forward — and those specs hold
the encoded reference **latents**.

Since 1.37.0 the write is undone in a `finally` when sampling ends, including
when it ends by raising. Before that the specs stayed on the cached guider:
the latents could not be freed while that guider lived, and a later run that
did not overwrite the same keys — a graph with no references, an aborted run,
another sampler sharing the same guider — sampled with the *previous* clip's
specs, whose shapes no longer fit. If you saw runs that worked once and then
started failing or ignoring the reference until you restarted ComfyUI, that
was it.

Chunked clips also used to keep every sampled chunk on the sampling device
until the whole clip was done; they now move to the offload device as they come
out, so VRAM no longer grows with clip length while the model is still loaded.

## Suggested order when something looks wrong

1. `crop_mode: off`, no mask, `temporal_tile_size: 0`. This is the plain swap; if
   it fails, the problem is the recipe or the LoRA, not this node.
2. Add the mask. Check `latent_mask` before judging the result.
3. Add the crop. Check `mask_over_source` for the box, and set the latent to the
   size `debug` reports.
4. Only then raise the frame count and turn on chunking.

Sampling settings follow the LoRA's recipe: CFG 3–5 with a real negative, and no
LightX2V (CFG 1.0 means no amplification, and the reference stops being
amplified).
