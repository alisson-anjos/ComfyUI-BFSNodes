// 🎞 Video tab: the source video, how it is split into shots, and the generation size.
import { h, GRIDS, fld, select, section, pill, hint, fmtT } from "./common.js";

export function videoTab(c) {
  const { plan, an, files, size, busy, detectorUsed, maxLen } = c;
  const fps = plan.fps;
  const num = (k, step = 0.1, min = 0) => h("input", { type: "number", step, min, value: plan[k],
    onChange: e => c.setPlan(k, parseFloat(e.target.value) || 0,
      ["max_s", "min_s"].includes(k) ? () => c.autoSplit(true) : undefined) });
  const sel_ = (k, opts, after) => select(plan[k], opts, v =>
    c.setPlan(k, v, after || (k === "mode" ? () => c.autoSplit(true) : undefined)));
  const au = an.value?.audio;

  const source = section("Source video", [
    h("div", { class: "row" }, [
      h("div", { style: "flex:1;min-width:180px" }, [h("select", { value: plan.video, onChange: e => c.setVideo(e.target.value) },
        [h("option", { value: "" }, "— choose a video from the input folder —"), ...files.videos.map(v => h("option", { value: v }, v))])]),
      h("button", { class: plan.video ? "" : "pri", onClick: () => c.pickFile("video/*", c.setVideo) }, "⬆ Upload"),
      h("button", { onClick: c.refreshFiles, title: "refresh the input folder" }, "↻"),
    ]),
    an.value ? h("div", { class: "row", style: "margin-top:6px" }, [
      pill(`${an.value.width}×${an.value.height}`), pill(`${an.value.fps_src.toFixed(2)} fps`), pill(`${an.value.duration.toFixed(2)} s`),
      pill(`timeline ${fps} fps · ${an.value.n} frames`), pill(`generate at ${size.w}×${size.h}`, "acc"),
      au?.has_audio
        ? pill(`🔊 audio · ${au.sample_rate ? (au.sample_rate / 1000).toFixed(1) + " kHz" : "?"} · ${au.channels || "?"} ch`, "ok", au.codec || "")
        : pill(`🔇 ${au?.reason || "no audio"} → silent track`, "warn", "The outputs carry a silent track of the right length, so Create Video works"),
      h("select", { value: plan.audio_mode, style: "width:auto", disabled: !au?.has_audio, title: "Audio of the planner's and the join's outputs",
        onChange: e => c.setPlan("audio_mode", e.target.value) },
        [h("option", { value: "auto" }, "use the video's audio"), h("option", { value: "silent" }, "silent track")]),
    ]) : plan.video ? null : h("div", { class: "empty" }, [h("b", "Pick or upload a video to start"),
      "It is split into shots the model can generate; every shot then gets its own reference, prompt and mask."]),
  ]);

  const R = c.range.value, N = c.n.value;
  const playF = Math.min(N - 1, Math.max(0, c.play.frame || 0));
  // a player right here, so the range is picked by watching (scrub it, or click the timeline above)
  const rng = section("Range", [
    h("div", { class: "player" }, [
      h("video", { ref: c.vid, src: c.viewUrl(plan.video), preload: "metadata", controls: true, playsinline: true,
        style: "max-height:240px" }),
      h("div", { class: "pinfo" }, [
        h("div", { class: "tc" }, [
          h("div", ["now   ", h("b", fmtT(playF, fps)), `  (frame ${playF})`]),
          h("div", ["start ", h("b", fmtT(R.lo, fps)), `  (frame ${R.lo})`]),
          h("div", ["end   ", h("b", fmtT(R.hi, fps)), `  (frame ${R.hi})`]),
        ]),
        h("div", { class: "rgrid" }, [
          h("button", { class: "pri", title: "the range starts at the player's frame", onClick: () => c.setRange("start_f", playF) }, "⇤ start here"),
          fld("Start frame", h("input", { type: "number", step: 1, min: 0, max: N - 1, value: R.lo,
            onChange: e => c.setRange("start_f", e.target.value) }), fmtT(R.lo, fps)),
          h("button", { class: "pri", title: "the range ends after the player's frame", onClick: () => c.setRange("end_f", playF + 1) }, "⇥ end here"),
          fld("End frame (0 = end)", h("input", { type: "number", step: 1, min: 0, max: N, value: R.hi >= N ? 0 : R.hi,
            onChange: e => c.setRange("end_f", e.target.value) }), fmtT(R.hi, fps)),
          (R.lo > 0 || R.hi < N) ? h("button", { class: "ghost", onClick: () => { plan.start_f = 0; c.setRange("end_f", 0); } }, "✕ whole video") : h("span"),
        ]),
        hint(`Play or scrub the video (or click the timeline), then ⇤ / ⇥. ${R.hi - R.lo} frames (${((R.hi - R.lo) / fps).toFixed(2)} s) of ${N}: only this part is split, analysed, generated and joined.`),
      ]),
    ]),
  ], { sub: "work on a part of the video" });

  const split = section("Split into shots", [
    h("div", { class: "grid" }, [
      fld("Mode", sel_("mode", [["shots", "Camera cuts"], ["fixed", "Fixed length"], ["single", "One shot (whole range)"]]),
        plan.mode === "single" ? "ignores Max seconds / shot" : ""),
      // the range in seconds, here next to the mode: "15 s in one shot" is set without leaving this tab
      fld("Range length, seconds (0 = to the end)", h("input", { type: "number", step: 0.5, min: 0,
        value: c.range.value.hi < c.n.value ? +((c.range.value.hi - c.range.value.lo) / plan.fps).toFixed(2) : 0,
        onChange: e => { const v = parseFloat(e.target.value) || 0;
          c.setRange("end_f", v > 0 ? c.range.value.lo + Math.round(v * plan.fps) : 0); } }),
        `from the start frame · ${c.range.value.hi - c.range.value.lo} frames`),
      fld("Max seconds / shot", num("max_s", 0.1, 0.2), `= ${maxLen.value} frames (${(maxLen.value / fps).toFixed(2)} s)`),
      fld("Max shots (0 = all)", num("max_parts", 1, 0), "only the first N shots"),
    ]),
    // the cut detector and the frame grid rarely change
    h("details", { style: "margin-top:6px" }, [h("summary", { class: "hint", style: "cursor:pointer" }, "More: cut detector, frame grid, fps"),
      h("div", { class: "grid", style: "margin-top:6px" }, [
        fld("Detector", sel_("detector", [["adaptive", "PySceneDetect adaptive"], ["content", "PySceneDetect content"], ["builtin", "Built-in"]], () => c.autoSplit(true))),
        fld(`Sensitivity ${Number(plan.sensitivity).toFixed(2)}`, h("input", { type: "range", min: 0, max: 1, step: 0.05, value: plan.sensitivity,
          onInput: e => { plan.sensitivity = parseFloat(e.target.value); }, onChange: () => { c.save(); c.autoSplit(true); } })),
        fld("Min seconds / shot", num("min_s", 0.1, 0)),
        fld("Frame grid", sel_("grid", Object.keys(GRIDS))),
        fld("Timeline fps", num("fps", 1, 1), "re-analyses"),
      ])]),
    h("div", { class: "row", style: "margin-top:8px" }, [
      h("button", { class: "pri", disabled: !plan.video || !!busy.value, onClick: () => c.autoSplit(true) }, "✂ Auto split"),
      h("button", { disabled: !plan.video, onClick: c.analyze }, "↻ Re-analyse"),
      detectorUsed.value ? hint(`detector: ${detectorUsed.value}`) : null,
      c.allSegCount.value > c.segs.value.length ? hint(`Max shots = ${plan.max_parts}: ${c.allSegCount.value - c.segs.value.length} `
        + `more shot(s) on the timeline are left out. Set Max shots to 0 to use them all.`, "color:#f0b429") : null,
    ]),
  ], { sub: "camera cuts or fixed length, on the model's frame grid" });

  const updateSize = () => c.autoSplit(false);
  const genFields = [
    fld("Size mode", select(plan.generation_size_mode || "megapixels", [
      ["megapixels", "Megapixel target"], ["original", "Original video"], ["custom", "Custom dimensions"],
    ], v => c.setPlan("generation_size_mode", v, updateSize))),
  ];
  if ((plan.generation_size_mode || "megapixels") === "megapixels")
    genFields.push(fld("Megapixels", h("input", { type: "number", step: 0.01, min: 0.02, value: plan.megapixels,
      onChange: e => c.setPlan("megapixels", parseFloat(e.target.value) || 0.5, updateSize) }), size.w ? `${size.w}×${size.h}` : ""));
  if (plan.generation_size_mode === "custom") {
    for (const [key, label] of [["custom_width", "Custom width"], ["custom_height", "Custom height"]])
      genFields.push(fld(label, h("input", { type: "number", step: 32, min: 32, value: plan[key],
        onChange: e => c.setPlan(key, Math.max(32, parseInt(e.target.value) || 1024), updateSize) })));
    if (size.w) genFields.push(fld("Output size", `${size.w}×${size.h}`));
  }
  if (plan.generation_size_mode === "original") genFields.push(h("div", { class: "hint" }, "Original resolution is rounded to the size multiple for model compatibility."));
  genFields.push(fld("Size multiple", h("input", { type: "number", step: 8, min: 8, value: plan.multiple,
    onChange: e => c.setPlan("multiple", Math.max(8, parseInt(e.target.value) || 32), updateSize) })));
  const gen = section("Generation size", h("div", { class: "grid" }, genFields), { sub: "cropped shots use it for the crop" });

  return h("div", { class: "card" }, [source, an.value ? rng : null, an.value ? split : null, an.value ? gen : null]);
}
