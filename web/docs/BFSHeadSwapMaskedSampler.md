# BFS Head Swap Sampler (crop · mask · loop)

Runs an LTX head-swap LoRA and adds three things around it, each optional.
Connect only what you need — the node degrades to whatever is wired.

| what you connect | what you get |
|---|---|
| `guide_video` + `identity_image` | plain head swap, one pass |
| `+ subject_mask` | only the masked region is denoised |
| `+ crop_mode` | the swap runs inside a box around the subject, then is pasted back |
| `+ temporal_tile_size` | long clips sampled in chunks with overlap |

The LoRA is never asked to understand a mask or a crop. It keeps doing its one
job: the guide rides on `source_id 1`, the identity on `source_id 2`. Everything
else here is plumbing around it.

## The mask is native inpainting, not a LoRA feature

The mask becomes `latent["noise_mask"]`, which the guider consumes as
`denoise_mask` — ComfyUI's own inpainting path. Pixels outside the mask keep the
guide's own content, so **the original face stays visible to the model**. This is
deliberately different from a masked-hole recipe, where the region is painted out
before the model sees it.

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

**Watch the `latent_mask` output.** It has one frame per *latent* frame, not per
video frame, and that is the real resolution of the edit. A tight outline around
a face can nearly vanish at that scale — and then nothing changes, no matter how
good the LoRA is. If the mask looks thin there, raise `mask_grow`.

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

## Reading the outputs

| output | what it answers |
|---|---|
| `images` | the result, with the crop pasted back |
| `mask_over_source` | is the mask where I think it is? Red over the original frames, green box for the crop |
| `cropped_guide` | exactly what the model was fed. If this is wrong, nothing downstream can be right |
| `crop_mask` | the mask after grow/blur, in the crop's pixel space |
| `latent_mask` | the mask as the sampler sees it — where a thin mask disappears |
| `latent` | the sampled latent |
| `debug` | crop mode and box, mask ops, tiling, the size to build the latent at |

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
