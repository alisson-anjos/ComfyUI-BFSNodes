// 🎬 Shots tab: one card per shot, then the selected shot's editor (pictures & prompt, what changes, continuity, copy
// to other shots), then the mask settings shared by every shot.
import { h, hue, fmtT, segKey, hasMask, maskSource, pill, hint, check, section, select, fld, refBox, promptChips, pictureColor,
         markName, MAX_REFS } from "./common.js";
import { h as rawH } from "../vendor/vue.esm-browser.prod.mjs";

const bdg = (text, kind = "", title = "") => h("span", { class: ["bdg", kind], title }, text);

function shotCard(c, s, i) {
  const { plan, sel, maskPrev, showMasks } = c;
  const fps = plan.fps, why = c.skipWhy(s), issues = c.checksFor(i);
  const warnN = issues.filter(x => x.lvl === "warn").length;
  const mp = maskPrev.value[segKey(s)];
  return h("div", { class: ["sc", i === sel.value && "sel", why && "skip"], onClick: () => { sel.value = i; } }, [
    h("button", { class: "play", title: "play this shot", onClick: e => { e.stopPropagation(); c.playShot(i); } }, "▶"),
    h("div", { class: "bar", style: `background:${hue(i)}` }),
    h("div", { class: "row" }, [h("b", `#${i + 1}`), s.cut ? pill("cut") : null,
      why ? pill("off", "warn", why) : pill("run", "ok")]),
    h("div", { class: "badges" }, [
      { video: bdg("🎞 mask video", "", s.mask.video), global: bdg("🎞 global mask", "", "the plan's mask video for the whole video"),
        track: bdg("🎯 track", "", "its part of the whole-video SAM 3 track"),
        sam: bdg("🎯 " + (s.mask.text ? s.mask.text.slice(0, 14) : `${(s.mask.points || []).length} pts`), "", s.mask.text || `${(s.mask.points || []).length} points`) }[maskSource(s)] || null,
      s.crop || s.inpaint || s.paste ? bdg({ paste: "🧩 frame+paste", mask: "🎭 mask only", crop: "✂ crop", cropmask: "✂🎭 crop+mask" }[c.modeOf(s)] + (s.inpaint && s.strength < 1 ? ` ${Number(s.strength).toFixed(2)}` : ""), "on", c.modeName(s)) : null,
      s.color_guide ? bdg(`🎨 ${1 + s.regions.length}`, "on", [s.color_guide_picture, ...s.regions.map(r => r.picture)]
        .map(k => `Picture ${k} · ${markName(pictureColor(c.plan, k))}`).join("\n") + ` · opacity ${s.color_guide_opacity ?? 0.35}`) : null,
      s.target ? bdg("🧑 target", "on", s.target) : null,
      i > 0 && s.chain !== "off" ? bdg("⛓", "on", s.chain === "history" ? `continues the previous shot's last ${s.chainHistory} frames (history)` : s.chain === "head" ? `starts with the previous shot's last ${Math.min(s.chainHistory, 39)} frames, kept as they are (head)` : `continues from the previous shot's ${s.chainFrame} frame as ${s.chain}`) : null,
      { ok: bdg("💾", "on", "stored and up to date: the next Queue does not generate it again (Run tab → Stored shots to redo it)"),
        run: bdg("⏳", "", "not stored yet, or its settings changed: the next Queue generates it") }[c.cacheOf(s)] || null,
      warnN ? bdg(`⚠ ${warnN}`, "warn", issues.map(x => x.text).join("\n")) : null,
    ]),
    h("div", { class: "t" }, `${fmtT(s.start, fps)} → ${fmtT(s.end, fps)} · ${s.len}f → ${s.gen}f`),
    h("div", { class: "thumbs" }, [
      ...(s.names.length ? s.names.slice(0, 3).map((name, k) => h("img", { class: "rt", src: c.viewUrl(name),
          title: s.refs.includes(name) ? name : `${name} (global)`, style: s.refs.includes(name) ? "" : "opacity:.45" }))
        : [h("div", { class: "rt ph2" }, "ref")]),
      s.names.length > 3 ? h("div", { class: "rt ph2", title: s.names.slice(3).join("\n") }, `+${s.names.length - 3}`) : null,
      h("img", { class: "rt", style: "width:56px", src: showMasks.value && mp?.frames?.length
        ? mp.frames[Math.floor(mp.frames.length / 2)].src : c.thumbFor(s.start + Math.floor(s.len / 2)) }),
    ]),
    s.prompt ? rawH("div", { class: "p" }, s.prompt)
      : h("div", { class: "p" }, plan.global_prompt ? "↳ global prompt" : "— no prompt —"),
  ]);
}

function player(c, cur) {
  const { plan, segs, play, vid, sel, n } = c;
  const S = segs.value, ps = S[play.idx] || cur, fps = plan.fps, N = n.value;
  return h("div", { class: "player" }, [
    h("video", { ref: vid, src: c.viewUrl(plan.video), preload: "metadata", muted: false, playsinline: true,
      onPause: () => { if (play.mode) play.mode = ""; } }),
    h("div", { class: "pinfo" }, [
      h("div", { class: "row" }, [
        h("button", { class: "pri", disabled: !cur, onClick: () => c.playShot(sel.value) }, `▶ Play #${sel.value + 1}`),
        h("button", { onClick: c.playAll }, "▶ Play all"),
        h("button", { onClick: c.stop }, "■ Stop"),
        check(play.loop, v => { play.loop = v; }, "loop shot"),
        play.mode ? pill(play.mode === "all" ? "playing all" : "playing shot", "warn") : null,
      ]),
      ps ? h("div", { class: "tc" }, [
        h("div", ["Shot ", h("b", `#${S.indexOf(ps) + 1}`), c.skipWhy(ps) ? `  (skipped: ${c.skipWhy(ps)})` : ""]),
        h("div", ["start ", h("b", fmtT(ps.start, fps)), `  (frame ${ps.start})`]),
        h("div", ["end   ", h("b", fmtT(ps.end, fps)), `  (frame ${ps.end - 1})  ·  ${(ps.len / fps).toFixed(2)}s`]),
        h("div", ["now   ", h("b", fmtT(Math.min(N, play.frame), fps)), `  (frame ${Math.min(N, play.frame)})`]),
      ]) : null,
      hint("Click the timeline to seek. Play all skips shots that will not run."),
    ]),
  ]);
}

function editor(c, cur) {
  const { plan, sel, segs, maskPrev, targetCrop, vlmSug, busy, usedRefs, copyOpts, COPY_FIELDS } = c;
  const i = sel.value, S = segs.value, k = segKey(cur), mp = maskPrev.value[k], g = vlmSug.value[k];
  const issues = c.checksFor(i);

  const head = h("div", { class: "edhd" }, [
    h("button", { class: "ghost", disabled: i === 0, title: "previous shot (←)", onClick: () => c.select(i - 1) }, "◀"),
    h("span", { class: "big" }, `Shot #${i + 1}`),
    h("button", { class: "ghost", disabled: i >= S.length - 1, title: "next shot (→)", onClick: () => c.select(i + 1) }, "▶"),
    c.skipWhy(cur) ? pill("disabled", "warn") : pill("runs", "ok"),
    hint(`frames ${cur.start}–${cur.end - 1} · ${cur.len} → generate ${cur.gen}`),
    h("span", { class: "grow" }),
    h("button", { onClick: () => c.setMeta(i, "enabled", !cur.enabled) }, cur.enabled ? "⏸ Disable" : "▶ Enable"),
    h("button", { title: "split this shot in two", onClick: () => c.splitAt(cur.start + Math.floor(cur.len / 2)) }, "✂ Split"),
    h("button", { disabled: i >= S.length - 1, title: "merge with the next shot", onClick: () => c.mergeNext(i) }, "⇥ Merge"),
  ]);
  const warnsHere = issues.filter(x => x.lvl === "warn");
  const problems = warnsHere.length ? h("div", { class: "checks", style: "margin-bottom:6px" }, warnsHere.map(x =>
    h("div", { class: "ck" }, [h("span", { class: "lvl" }, x.lvl === "warn" ? "⚠" : "ℹ"), h("span", x.text)]))) : null;

  const refPrompt = section("Pictures & prompt", [
    refBox(c, i, cur.refs, plan.global_refs),
    h("div", { class: "pbox", style: "margin-top:6px" }, [
      h("textarea", { placeholder: plan.auto_prompt?.enabled
        ? "Request for this shot (empty = the global request); 🤖 Auto prompt writes the H3 prompt below from it"
        : "Prompt for this shot (empty = global prompt). Placeholders: {target} {details} {details@Picture n} {shot} {setting}",
        value: cur.prompt, onChange: e => c.setMeta(i, "prompt", e.target.value) }),
      promptChips(c, cur.names, v => c.setMeta(i, "prompt", v)),
      !cur.prompt && plan.global_prompt ? rawH("span", { class: "hint" }, `↳ uses the global prompt: ${plan.global_prompt.slice(0, 120)}${plan.global_prompt.length > 120 ? "…" : ""}`) : null,
    ]),
    plan.auto_prompt?.enabled ? (() => {
      const w = c.writtenOf(cur), old = w?.text && w.for !== c.autoFor(cur);
      return h("div", { class: "note", style: "margin-top:6px;padding:6px 8px" }, [
        h("div", { class: "row", style: "gap:6px;align-items:center;margin-bottom:4px" }, [
          h("b", "🤖 Written prompt"), old ? pill("older than the request / pictures / marks", "warn") : null, h("span", { class: "grow" }),
          h("button", { class: w?.text ? "" : "pri", disabled: !!busy.value, onClick: () => c.writePrompts([i]) }, w?.text ? "🤖 Write again" : "🤖 Write"),
          w?.text ? h("button", { class: "dng", title: "forget it: the next run writes a new one", onClick: () => c.setWritten(cur, "") }, "✕") : null,
        ]),
        h("textarea", { style: "min-height:140px;font-size:11px", value: w?.text || "",
          placeholder: "not written yet: 🤖 Write, or the planner's VLM writes it when the workflow runs",
          onChange: e => c.setWritten(cur, e.target.value) }),
      ]);
    })() : null,
    usedRefs.value.length ? h("div", { class: "recent" }, [hint("recent (click = picture 1, shift+click = add, or drag onto a slot):"),
      ...usedRefs.value.map(n => h("img", { src: c.viewUrl(n), title: n, draggable: true,
        onDragstart: e => e.dataTransfer.setData("text/bfs-ref", n),
        onClick: e => (e.shiftKey ? c.putRefs(i, cur.refs.length, [n]) : c.setRef(i, 0, n)) })),
      h("button", { class: "sm", title: "Forget the recent pictures (the ones this plan still uses stay)", onClick: () => c.clearRecent() }, "✕ clear"),
      h("button", { class: "sm dng", title: "Remove EVERY reference from this plan (global and all shots) and the recent list", onClick: () => c.clearAllRefs() }, "✕ clear all")]) : null,
    h("div", { class: "row", style: "margin-top:6px" }, [
      h("button", { class: "dng", title: "clear this shot's pictures and prompt: it uses the global ones",
        onClick: () => { Object.assign(c.meta(i), { refs: [], prompt: "" }); c.save(); } }, "Use global"),
    ]),
  ], { sub: "faded = the global ones (Pictures & prompt tab)" });

  // which picture a marked person becomes: "Picture 3 · red" with its thumbnail (the color is the picture's)
  const picturePick = (k, onPick) => {
    const n = Math.max(cur.names.length, k), color = pictureColor(plan, k);
    return h("span", { class: "row", style: "gap:4px;align-items:center" }, [
      cur.names[k - 1] ? h("img", { class: "mthumb", src: c.viewUrl(cur.names[k - 1]), style: `--mark:${color}`, title: cur.names[k - 1] })
        : h("span", { class: "mthumb", style: `--mark:${color};display:inline-block;background:${color}` }),
      h("select", { value: String(k), style: "width:auto", title: "The picture this subject becomes; the mark is drawn in its color",
        onChange: e => onPick(+e.target.value) },
        Array.from({ length: Math.min(MAX_REFS, n) }, (_, q) => h("option", { value: String(q + 1) },
          `Picture ${q + 1} · ${markName(pictureColor(plan, q + 1))}${cur.names[q] ? "" : " (no picture)"}`))),
    ]);
  };
  // what changes in this shot: one row per subject. Row 1 is the shot's own selection (its SAM 3 text / points);
  // with Colored Mark on, every row becomes a picture (drawn in that picture's color) and more rows can be added
  const cg = cur.color_guide, regionsN = 1 + cur.regions.length;
  const label = n => (cg ? h("span", { class: "hint", style: "white-space:nowrap;min-width:58px" }, `Subject ${n}`) : null);
  const becomes = (k, onPick) => (cg ? [h("span", { class: "hint" }, "→"), picturePick(k, onPick)] : []);
  const subject1 = h("div", { class: "mreg" }, [
    label(1),
    h("input", { type: "text", value: cur.mask.text || "", style: "flex:1;min-width:160px",
      placeholder: "what to select, in English: woman in pink top, red car, black dog… (Enter = preview)",
      title: "A short phrase (noun + one or two traits); SAM 3 finds and tracks it through the shot. Points, when set, win over the text.",
      onInput: e => c.setMask(i, "text", e.target.value),
      onKeydown: e => { if (e.key === "Enter") { e.preventDefault(); c.setMask(i, "text", e.target.value); if (e.target.value || (cur.mask.points || []).length) c.previewMask(i); } } }),
    h("button", { title: "Click positive / negative points on a frame (more precise than text when subjects look alike)",
      onClick: () => c.openPoints(i) }, (cur.mask.points || []).length ? `🎯 Points (${cur.mask.points.length})` : "🎯 Points…"),
    h("button", { class: hasMask(cur) && !mp ? "pri" : "", disabled: !!busy.value || !hasMask(cur),
      title: "Segment this shot and show a few frames", onClick: () => c.previewMask(i) }, "👁 Preview"),
    ...becomes(cur.color_guide_picture, k => { c.setMarkPicture(i, 1, k); if (maskPrev.value[segKey(cur)]) c.previewMask(i); }),
    h("button", { class: "dng", disabled: !(cur.mask.text || (cur.mask.points || []).length || cur.mask.video || cur.crop || cur.inpaint),
      title: "Clear this shot's selection (text, points, mask video) and its mode", onClick: () => c.clearMask(i) }, "✕"),
  ]);
  const more = cg ? cur.regions.map((r, j) => h("div", { class: "mreg" }, [
    label(j + 2),
    h("input", { type: "text", value: r.text || "", style: "flex:1;min-width:160px", placeholder: "what to select, e.g. the man on the left",
      onChange: e => c.setMarkRegion(i, j, "text", e.target.value) }),
    h("button", { onClick: () => c.openPoints(i, j + 2) }, (r.points || []).length ? `🎯 Points (${r.points.length})` : "🎯 Points…"),
    h("button", { disabled: !!busy.value || !(r.text || (r.points || []).length),
      onClick: () => c.previewMask(i, { text: r.text || "", points: r.points || [], key: r.key || 0, max_objects: 1,
        preview_color: pictureColor(plan, r.picture) }) }, "👁 Preview"),
    ...becomes(r.picture, k => c.setMarkRegion(i, j, "picture", k)),
    h("button", { class: "dng", title: "remove this subject", onClick: () => c.removeMarkRegion(i)(j) }, "✕"),
  ])) : [];
  const MODE_INFO = {
    frame: "The whole frame is regenerated.",
    paste: "The whole frame is generated; BFS Shot Join pastes only the subject back onto the original (background exact).",
    mask: "Only the selection is regenerated; the rest of the frame stays exactly as it was.",
    crop: "A box around the selection is regenerated at full size (more detail for a small subject), then pasted back.",
    cropmask: "Inside the crop, only the selection is regenerated: the most detail, background kept.",
  };
  const curMode = c.modeOf(cur);
  const who = section("What changes", [
    h("div", { class: "row", style: "align-items:center;gap:10px;margin-bottom:4px" }, [
      check(!!cg, v => c.setMeta(i, "color_guide", v), "🎨 Colored Mark",
        "Paint each selected subject with the color of the picture it becomes (a translucent marker); the prompt says which color becomes which picture. It does not turn on inpainting."),
      cg ? select(cur.color_guide_style, [["overlay", "translucent"], ["hatching", "hatching + contour"], ["normal map", "normal map (needs normals)"]],
        v => c.setMarkStyle(i, v), { style: "width:auto", title: "How the marks are drawn on the guide (the same styles as BFS Video Segment Picker)" }) : null,
      cg ? h("span", { class: "row", style: "gap:6px;align-items:center" }, [hint(`opacity ${Number(cur.color_guide_opacity ?? 0.35).toFixed(2)}`),
        h("input", { type: "range", min: 0.05, max: cur.color_guide_style === "overlay" ? 0.8 : 1, step: 0.01, value: cur.color_guide_opacity ?? 0.35, style: "width:140px",
          onInput: e => { c.meta(i).color_guide_opacity = parseFloat(e.target.value); }, onChange: c.save })]) : null,
    ]),
    cg && cur.color_guide_style === "normal map" ? h("div", { class: "note", style: "margin:2px 0 4px;padding:6px 8px" },
      "Normal map: run MoGe on BFS Shot Unpack's frames (Render MoGe Geometry, normal_directx) and connect it to BFS Shot H3 Conditioning → normals. The preview shows a tint.") : null,
    subject1, ...more,
    c.files.pointModel ? hint("📍 MolmoPoint is connected: a text selection is pointed by MolmoPoint, then SAM 3 tracks it.", "display:block;margin-top:4px") : null,
    cg ? h("div", { class: "row", style: "margin:4px 0;align-items:center;gap:8px" }, [
      h("button", { disabled: regionsN >= MAX_REFS, onClick: () => c.addMarkRegion(i) }, "+ Add subject"),
      cur.names.length <= regionsN ? hint("add pictures above to give each subject its own") : null,
    ]) : null,
    cg && !hasMask(cur) ? h("div", { class: "err" }, "Colored Mark needs a selection: a text, points or a mask video.") : null,
    h("div", { class: "row", style: "margin-top:6px;align-items:center" }, [
      h("span", { class: "hint", style: "white-space:nowrap", title: "A black/white video (white = the subject) for this shot only; it replaces SAM 3." }, "or a mask video"),
      h("select", { value: cur.mask.video || "", style: "flex:1;min-width:160px", onChange: e => c.setMask(i, "video", e.target.value) },
        [h("option", { value: "" }, cur.extMask ? "— none: the whole-video mask —" : "— none —"), ...c.files.videos.map(v => h("option", { value: v }, v))]),
      h("button", { title: "upload a mask video for this shot", onClick: () => c.pickFile("video/*", name => c.setMask(i, "video", name)) }, "⬆ Upload"),
    ]),
    maskSource(cur) === "video" ? hint("This shot uses its mask video: the text and points are ignored.", "display:block;margin-top:2px")
      : maskSource(cur) === "global" || maskSource(cur) === "track" ? hint("No selection of its own: this shot uses the whole-video mask (Mask settings, below).", "display:block;margin-top:2px") : null,
    mp ? h("div", { class: "mstrip" }, [
      ...mp.frames.map(f => h("img", { src: f.src, title: `frame ${f.f}` })),
      hint(mp.empty ? "nothing found: the shot runs without a mask" : `mask covers ${(mp.coverage * 100).toFixed(1)}%${cur.crop ? " · yellow = crop box" : ""}`),
    ]) : null,
    h("div", { class: "row", style: "margin-top:8px;align-items:center" }, [
      h("span", { style: "white-space:nowrap" }, "Generate"),
      select(curMode, c.MODES.map(m => [m[0], m[1]]), v => c.setMode(i, v), { style: "width:auto" }),
      hint(MODE_INFO[curMode]),
    ]),
    (cur.crop || cur.inpaint || cur.paste) && !hasMask(cur) ? h("div", { class: "err" }, `${c.modeName(cur)} needs a selection on this shot (text, points or a mask video).`) : null,
    cur.inpaint ? h("div", { class: "row", style: "margin-top:6px;align-items:center",
      title: "How much of the selection is regenerated. 0.8-0.9 keeps pose, outline and lighting while still changing it; too low copies the original. Needs BFS Shot H3 Conditioning → inpaint = per shot (planner)." }, [
      h("span", { style: "white-space:nowrap" }, `Mask strength ${Number(cur.strength).toFixed(2)}`),
      h("input", { type: "range", min: 0.3, max: 1, step: 0.01, value: cur.strength, style: "flex:1;max-width:320px",
        onInput: e => { c.meta(i).strength = parseFloat(e.target.value); }, onChange: c.save }),
      h("button", { class: "ghost", disabled: cur.strength >= 1, onClick: () => c.setMeta(i, "strength", 1) }, "reset"),
    ]) : null,
    cur.paste ? h("div", { class: "row", style: "margin-top:6px;align-items:center",
      title: "What the NEW subject is, in English, so its outline on the result is pasted too (a bigger silhouette is not cut). Empty = person." }, [
      h("span", { style: "white-space:nowrap" }, "new subject"),
      h("input", { type: "text", value: cur.pasteText || "", placeholder: "person (or dog, car… what the result shows)", style: "flex:1;max-width:420px",
        onChange: e => c.setMeta(i, "paste_text", e.target.value) }),
    ]) : null,
    h("div", { class: "row", style: "margin-top:8px;align-items:center" }, [
      targetCrop.value[k] ? h("img", { src: targetCrop.value[k], style: "height:56px;border-radius:6px" }) : null,
      h("span", { class: "hint", style: "white-space:nowrap" }, "{target}"),
      h("input", { type: "text", value: cur.target, style: "flex:1;min-width:200px",
        placeholder: "what is replaced, for {target} in the prompt (e.g. the young woman in a pink top)",
        title: "Write {target} in the prompt and it becomes this text, so the model knows WHICH subject changes.",
        onChange: e => c.setMeta(i, "target", e.target.value) }),
      h("button", { disabled: !!busy.value || !hasMask(cur), title: "SAM 3 cuts the selection out and the VLM describes it",
        onClick: () => c.describeTarget(i) }, "🧑 Describe"),
    ]),
    g ? h("div", { class: "vsug" }, [
      h("div", ["🤖 ", h("b", "select: "), g.segment || "—", g.segment ? h("button", { onClick: () => c.applySug(i, "segment") }, "Use") : null]),
      h("div", [h("b", "shot: "), g.shot || "—", g.shot ? h("button", { title: "copy (the {shot} placeholder in the prompt gets it automatically at run time)",
        onClick: () => navigator.clipboard?.writeText(g.shot) }, "Copy") : null]),
    ]) : null,
  ], { sub: "the selection, its picture and how it is generated" });

  const cont = section("Continuity", h("div", { class: "row" }, [
    h("select", { value: cur.chain, style: "width:auto", disabled: i === 0,
      title: "reference: the previous result's frame becomes one more <Picture n> after this shot's own references. first frame: it is anchored at frame 0 of this shot. history: the END of the previous result (5/22/39/56 frames) goes just before this shot on H3's timeline, so a long take continues its motion and look (same take only, not across a cut). head: the END of the previous result (5/22/39 frames) becomes the first frames of this shot, kept exactly as they are (not swapped again), and the new frames continue from them; the join cuts them off (same take only, H3 grid).",
      onChange: e => c.setMeta(i, "chain", e.target.value) },
      [h("option", { value: "off" }, "off"), h("option", { value: "reference" }, "previous shot as reference"),
       h("option", { value: "first frame" }, "previous shot as first frame"),
       h("option", { value: "history" }, "previous shot's end as history (long take)"),
       h("option", { value: "head" }, "previous shot's end as this shot's start (kept, long take)")]),
    cur.chain === "history" || cur.chain === "head"
      ? h("select", { value: String(cur.chainHistory), style: "width:auto", disabled: i === 0,
          title: "How much of the previous result's end goes before this shot. 5 frames is a short context; longer windows carry more motion and use more memory.",
          onChange: e => c.setMeta(i, "chain_history", Number(e.target.value)) },
          (cur.chain === "head" ? [5, 22, 39] : [5, 22, 39, 56]).map(n => h("option", { value: String(n) }, `last ${n} frames (${(n / 24).toFixed(1)} s)`)))
      : h("select", { value: cur.chainFrame, style: "width:auto", disabled: i === 0 || cur.chain === "off",
          title: "Which frame of the previous shot's result", onChange: e => c.setMeta(i, "chain_frame", e.target.value) },
          [h("option", { value: "first" }, "its first frame"), h("option", { value: "middle" }, "its middle frame"), h("option", { value: "last" }, "its last frame")]),
    hint(i === 0 ? "the first shot uses only its references"
      : cur.chain === "head" ? (cur.cut ? "⚠ a camera cut starts this shot: head is skipped (it only carries a continuing take)"
        : "this shot is generated from earlier: its first frames are the PREVIOUS shot's result, pinned in the latent, and the join cuts them (queue loop: shots run in order). Costs those frames of generation per shot.")
      : cur.chain === "history" ? (cur.cut ? "⚠ a camera cut starts this shot: history is skipped (it only carries a continuing take)"
        : "the end of the PREVIOUS shot's result continues into this one (queue loop: shots run in order). Works with every mask mode, crop and the duet canvas.")
      : "uses a frame of the PREVIOUS shot's result (queue loop, or auto loop with BFS Shot H3 Duet)"),
  ]), { open: cur.chain !== "off", sub: cur.chain !== "off" ? cur.chain : "off" });

  const copy = section("Copy to other shots", [
    h("div", { class: "copy" }, Object.entries(COPY_FIELDS).map(([f, label]) =>
      check(copyOpts[f], v => { copyOpts[f] = v; }, label, {
        ref: "every reference picture", prompt: "this shot's prompt", mask_text: "the mask text (shots keep their own points)",
        points: "the points, on each shot's frame at the same relative position (a subject that stays in place)",
        mask_video: "the shot's mask video (it covers only its own shot: usually you want a different one per shot)",
        crop: "the generation mode (Full frame / Frame + paste / Mask only / Crop / Crop + mask) and its mask strength (shots with a mask)", target: "the {target} description",
        chain: "continuity (not on the first shot)" }[f]))),
    h("div", { class: "row", style: "margin-top:6px" }, [
      h("div", { class: "seg-btns" }, [["all", "all shots"], ["after", "shots after this"]].map(([v, l]) =>
        h("button", { class: copyOpts.scope === v && "on", onClick: () => { copyOpts.scope = v; } }, l))),
      h("button", { class: "pri", disabled: !Object.keys(COPY_FIELDS).some(f => copyOpts[f]) || S.length < 2,
        onClick: e => { const n = c.copyFrom(i); const b = e.currentTarget; b.textContent = `✓ copied to ${n} shot(s)`; setTimeout(() => { b.textContent = "📋 Copy"; }, 1400); } }, "📋 Copy"),
    ]),
  ], { open: false, sub: Object.entries(COPY_FIELDS).filter(([f]) => copyOpts[f]).map(([, l]) => l).join(", ") });

  return h("div", { class: "card" }, [head, problems, refPrompt, who, cont, copy]);
}

// settings every shot's mask shares (SAM 3, crop and paste-back), and the optional mask for the whole video
function maskSettings(c) {
  const { plan } = c;
  const MC = plan.mask_cfg || {};
  const mnum = (k, step, label, help, scale = 1) => fld(label, h("input", { type: "number", step, min: 0, value: +(MC[k] * scale).toFixed(3),
    onChange: e => c.setMaskCfg(k, (parseFloat(e.target.value) || 0) / scale) }), help);
  return h("div", { class: "card" }, [section("Mask settings (every shot)", [
    h("div", { class: "row", style: "margin-bottom:8px;align-items:center" }, [
      h("span", { style: "white-space:nowrap", title: "A black/white video of the WHOLE source (white = the subject). Shots without their own selection use it instead of SAM 3. The planner's mask input does the same from the graph and wins over this." }, "🎞 Mask video (whole video)"),
      h("select", { value: plan.mask_video || "", style: "flex:1;min-width:180px", onChange: e => c.setGlobalMaskVideo(e.target.value) },
        [h("option", { value: "" }, "— none —"), ...c.files.videos.map(v => h("option", { value: v }, v))]),
      h("button", { onClick: () => c.pickFile("video/*", c.setGlobalMaskVideo) }, "⬆ Upload"),
      plan.mask_video ? h("button", { onClick: () => c.setGlobalMaskVideo("") }, "✕") : null,
    ]),
    plan.track?.points?.length ? h("div", { class: "note", style: "margin-bottom:8px;display:flex;gap:8px;align-items:center" }, [
      h("span", { class: "grow" }, [h("b", "🎯 Whole-video track: "), `${plan.track.points.length} point(s) at frame ${plan.track.frame}`,
        plan.track.text ? ` · re-found by "${plan.track.text}" after cuts` : " · no text: after a cut the subject may be lost",
        ". Shots without their own selection use it."]),
      h("button", { class: "ghost", onClick: c.clearTrack }, "✕ remove"),
    ]) : null,
    h("div", { class: "row", style: "gap:14px;margin-bottom:6px" }, [
      check(MC.fill_holes, v => c.setMaskCfg("fill_holes", v), "Fill holes", "Fill enclosed gaps so each region is solid"),
      check(MC.invert, v => c.setMaskCfg("invert", v), "Invert", "Use everything except the selection"),
    ]),
    h("div", { class: "grid" }, [
      mnum("expand", 1, "Expand (px)", "grow the mask"),
      mnum("temporal_expand", 1, "Temporal expand (frames)", "less flicker"),
      mnum("threshold", 0.05, "Detection threshold"),
      mnum("max_objects", 1, "Max objects", "matches of the text tracked"),
      mnum("padding", 1, "Crop padding (%)", "crop modes", 100),
      mnum("feather", 1, "Feather (px)", "crop modes"),
      fld("Paste back", h("select", { value: MC.paste, onChange: e => c.setMaskCfg("paste", e.target.value) },
        [h("option", { value: "mask" }, "only the mask (feathered)"), h("option", { value: "box" }, "the whole box (feathered)")]), "crop modes"),
      mnum("blockify", 1, "Blockify (px, 0 = off)"),
    ]),
  ], { open: false, sub: "SAM 3, crop and paste-back" })]);
}

export function shotsTab(c) {
  const { segs, sel, showMasks } = c;
  const S = segs.value, cur = S[sel.value];
  if (!S.length) return h("div", { class: "card empty" }, [h("b", "No shots yet"), "Pick a video in the Video tab."]);
  return h("div", [
    h("div", { class: "card" }, [
      h("div", { class: "row", style: "margin-bottom:6px" }, [h("span", { class: "sect" }, "Shots"),
        hint(`${S.length} · scroll sideways · ← → to move`), h("span", { class: "grow" }),
        check(showMasks.value, v => { showMasks.value = v; }, "show mask previews", "Overlay the mask preview on the shot cards")]),
      h("div", { class: "shots" }, S.map((s, i) => shotCard(c, s, i))),
    ]),
    cur ? h("div", { class: "card" }, [player(c, cur)]) : null,
    cur ? editor(c, cur) : null,
    maskSettings(c),
  ]);
}
