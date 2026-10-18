// 🎬 Shots tab: one card per shot, then the selected shot's editor (preview, reference & prompt, who is replaced,
// continuity, copy to other shots).
import { h, hue, fmtT, segKey, hasMask, maskSource, pill, hint, check, section, refBox, promptChips, pictureColor, markName,
         MAX_REFS } from "./common.js";
import { h as rawH } from "../vendor/vue.esm-browser.prod.mjs";

const bdg = (text, kind = "", title = "") => h("span", { class: ["bdg", kind], title }, text);

function shotCard(c, s, i) {
  const { plan, sel, maskPrev, showMasks } = c;
  const fps = plan.fps, why = c.skipWhy(s), who = c.whoIn(s), st = c.statFor(s), issues = c.checksFor(i);
  const warnN = issues.filter(x => x.lvl === "warn").length;
  const mp = maskPrev.value[segKey(s)];
  return h("div", { class: ["sc", i === sel.value && "sel", why && "skip"], onClick: () => { sel.value = i; } }, [
    h("button", { class: "play", title: "play this shot", onClick: e => { e.stopPropagation(); c.playShot(i); } }, "▶"),
    h("div", { class: "bar", style: `background:${hue(i)}` }),
    h("div", { class: "row" }, [h("b", `#${i + 1}`), s.cut ? pill("cut") : null,
      why ? pill("skip", "warn", why) : pill("run", "ok"), s.force !== "auto" ? pill(s.force) : null]),
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
    who ? h("div", { class: "who" }, who.people.length ? who.people.map(id => h("img", {
      src: c.personOf(id)?.thumb || "", title: `Person ${id}${id === who.main ? " (main)" : ""}${c.included(id) ? " · included" : ""}`,
      class: [id === who.main && "main", c.included(id) && "lk"] })) : [h("span", { class: "t" }, "no faces")]) : null,
    st ? h("div", { class: "t", style: "margin-top:2px" },
      `👤 ${st.persons} · ${(st.person_area * 100).toFixed(1)}% · 🙂 ${st.faces} · ☀ ${(st.brightness * 100).toFixed(0)}%`) : null,
    why ? h("div", { class: "t", style: "color:#ffc46b" }, why) : null,
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
    c.skipWhy(cur) ? pill(`skip · ${c.skipWhy(cur)}`, "warn") : pill("runs", "ok"),
    hint(`frames ${cur.start}–${cur.end - 1} · ${cur.len} → generate ${cur.gen}`),
    h("span", { class: "grow" }),
    h("select", { value: cur.force, style: "width:auto", title: "Override the content filters for this shot",
      onChange: e => c.setMeta(i, "force", e.target.value) },
      [h("option", { value: "auto" }, "filters decide"), h("option", { value: "run" }, "always run"), h("option", { value: "skip" }, "always skip")]),
    h("button", { onClick: () => c.setMeta(i, "enabled", !cur.enabled) }, cur.enabled ? "⏸ Disable" : "▶ Enable"),
    h("button", { title: "split this shot in two", onClick: () => c.splitAt(cur.start + Math.floor(cur.len / 2)) }, "✂ Split"),
    h("button", { disabled: i >= S.length - 1, title: "merge with the next shot", onClick: () => c.mergeNext(i) }, "⇥ Merge"),
  ]);
  const warnsHere = issues.filter(x => x.lvl === "warn");
  const problems = warnsHere.length ? h("div", { class: "checks", style: "margin-bottom:6px" }, warnsHere.map(x =>
    h("div", { class: "ck" }, [h("span", { class: "lvl" }, x.lvl === "warn" ? "⚠" : "ℹ"), h("span", x.text)]))) : null;

  const refPrompt = section("Reference & prompt", [
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
      h("button", { class: "dng", title: "clear this shot's references and prompt: it uses the global ones",
        onClick: () => { Object.assign(c.meta(i), { refs: [], prompt: "" }); c.save(); } }, "Use global"),
      hint("empty slots use the global references (Prompts & refs tab)"),
    ]),
  ], { sub: "who the shot becomes" });

  const mode = (on, title, text, onClick) => h("div", { class: ["mode", on && "on"], onClick }, [h("b", title), h("span", text)]);
  // which picture a marked person becomes: "Picture 3 · red" with its thumbnail (the color is the picture's)
  const picturePick = (k, onPick) => {
    const n = Math.max(cur.names.length, k), color = pictureColor(plan, k);
    return h("span", { class: "row", style: "gap:4px;align-items:center" }, [
      cur.names[k - 1] ? h("img", { class: "mthumb", src: c.viewUrl(cur.names[k - 1]), style: `--mark:${color}`, title: cur.names[k - 1] })
        : h("span", { class: "mthumb", style: `--mark:${color};display:inline-block;background:${color}` }),
      h("select", { value: String(k), style: "width:auto", title: "The picture this person becomes; the mark is drawn in its color",
        onChange: e => onPick(+e.target.value) },
        Array.from({ length: Math.min(MAX_REFS, n) }, (_, q) => h("option", { value: String(q + 1) },
          `Picture ${q + 1} · ${markName(pictureColor(plan, q + 1))}${cur.names[q] ? "" : " (no picture)"}`))),
    ]);
  };
  const who = section("Select region", [
    h("div", { class: "row", style: "margin-top:4px;align-items:center;gap:8px" }, [
      check(!!cur.color_guide, v => c.setMeta(i, "color_guide", v), "🎨 Colored Mark",
        "Color each selected person in the source video with the color of the picture they become, as a translucent marker; the prompt gets 'the person marked in <color> becomes the character from <Picture n>'. It does not create or enable an inpainting mask."),
      cur.color_guide ? h("span", { class: "hint" }, "this selection becomes") : null,
      cur.color_guide ? picturePick(cur.color_guide_picture, k => { c.setMarkPicture(i, 1, k); if (maskPrev.value[segKey(cur)]) c.previewMask(i); }) : null,
    ]),
    cur.color_guide ? h("div", { class: "note", style: "margin:4px 0;padding:6px 8px" }, [
      ...cur.regions.map((r, j) => h("div", { class: "mreg" }, [
        h("span", { class: "hint", style: "white-space:nowrap" }, `Person ${j + 2}`),
        picturePick(r.picture, k => c.setMarkRegion(i, j, "picture", k)),
        h("input", { type: "text", value: r.text || "", style: "flex:1;min-width:160px", placeholder: "who, in English: e.g. the man on the left",
          onChange: e => c.setMarkRegion(i, j, "text", e.target.value) }),
        h("button", { disabled: !!busy.value || !(r.text || (r.points || []).length),
          onClick: () => c.previewMask(i, { text: r.text || "", points: r.points || [], key: r.key || 0, max_objects: 1,
            preview_color: pictureColor(plan, r.picture) }) }, "👁 Preview"),
        h("button", { onClick: () => c.openPoints(i, j + 2) }, c.keyFramesOf(r).length ? `🎯 Points (${c.keyFramesOf(r).length} key frame${c.keyFramesOf(r).length > 1 ? "s" : ""})` : "🎯 Points…"),
        h("button", { class: "dng", title: "remove this person's mark", onClick: () => c.removeMarkRegion(i)(j) }, "✕"),
      ])),
      h("div", { class: "row", style: "margin-top:6px;align-items:center;gap:8px" }, [
        h("button", { disabled: 1 + cur.regions.length >= MAX_REFS, onClick: () => c.addMarkRegion(i) }, "+ Mark another person"),
        hint(cur.names.length > 1 + cur.regions.length ? "each person: a picture (its color) and a text or points"
          : "add more pictures (Reference & prompt) to give each person their own"),
      ]),
    ]) : null,
    cur.color_guide ? h("details", { style: "margin:4px 0" }, [h("summary", { class: "hint", style: "cursor:pointer" }, `Opacity · ${Number(cur.color_guide_opacity ?? 0.35).toFixed(2)}`),
      h("div", { class: "row", style: "margin-top:6px;gap:8px;align-items:center" }, [h("input", { type: "range", min: 0.05, max: 0.8, step: 0.01, value: cur.color_guide_opacity ?? 0.35, style: "flex:1;max-width:320px", onInput: e => { c.meta(i).color_guide_opacity = parseFloat(e.target.value); }, onChange: c.save }), hint("Keeps movement and expression visible.")])]) : null,
    h("div", { class: "row", style: "align-items:center" }, [
      h("input", { type: "text", value: cur.mask.text || "", style: "flex:1;min-width:160px",
        placeholder: "mask: what to segment, in English (woman in pink top, black dog…) · Enter = preview",
        title: "What to segment, in English: a short phrase (noun + one or two traits), e.g. 'woman in pink top'; commas for several things. Enough on its own - no points needed: SAM 3 finds and tracks it through the shot (every match, up to Max objects). Press Enter to preview. Points, when set, take priority.",
        onInput: e => c.setMask(i, "text", e.target.value),
        onKeydown: e => { if (e.key === "Enter") { e.preventDefault(); c.setMask(i, "text", e.target.value); if (e.target.value || (cur.mask.points || []).length) c.previewMask(i); } } }),
      h("button", { title: "Pick positive / negative points on a frame of this shot (more precise than text when several people look alike)",
        onClick: () => c.openPoints(i) }, c.keyFramesOf(cur.mask).length ? `🎯 Points (${c.keyFramesOf(cur.mask).length} key frame${c.keyFramesOf(cur.mask).length > 1 ? "s" : ""})` : "🎯 Points…"),
      h("button", { class: hasMask(cur) && !mp ? "pri" : "", disabled: !!busy.value || !hasMask(cur),
        title: (cur.mask.points || []).length ? "Segment this shot (the points win over the text)" : "Segment this shot with the text and show a few frames",
        onClick: () => c.previewMask(i) }, "👁 Preview"),
      h("button", { class: "dng", disabled: !(cur.mask.text || (cur.mask.points || []).length || cur.mask.video || cur.crop || cur.inpaint), title: "Remove this shot's mask (points, text, mask video) and mode - for shots where it picked the wrong thing",
        onClick: () => c.clearMask(i) }, "✕ Clear"),
    ]),
    h("div", { class: "row", style: "margin-top:6px;align-items:center" }, [
      h("span", { class: "hint", style: "white-space:nowrap", title: "A black/white video (white = the subject), e.g. rotoscoped in another tool. It replaces SAM 3 for this shot and covers only this shot: its first frame is the shot's first frame." }, "🎞 mask video"),
      h("select", { value: cur.mask.video || "", style: "flex:1;min-width:160px",
        onChange: e => c.setMask(i, "video", e.target.value) },
        [h("option", { value: "" }, cur.extMask ? "— none: the global mask video (People / Filters / Masks) —" : "— none: SAM 3 (text / points) —"),
         ...c.files.videos.map(v => h("option", { value: v }, v))]),
      h("button", { title: "upload a mask video for this shot", onClick: () => c.pickFile("video/*", name => c.setMask(i, "video", name)) }, "⬆ Upload"),
      cur.mask.video ? h("button", { title: "back to SAM 3 / the global mask", onClick: () => c.setMask(i, "video", "") }, "✕") : null,
    ]),
    maskSource(cur) === "video" ? hint("This shot uses its mask video: the text and points are ignored.", "display:block;margin-top:2px")
      : maskSource(cur) === "global" ? hint("No mask of its own: this shot uses the global mask video (People / Filters / Masks tab).", "display:block;margin-top:2px")
      : maskSource(cur) === "track" ? hint("No mask of its own: this shot uses its part of the whole-video track (People / Filters / Masks tab).", "display:block;margin-top:2px") : null,
    mp ? h("div", { class: "mstrip" }, [
      ...mp.frames.map(f => h("img", { src: f.src, title: `frame ${f.f}` })),
      hint(mp.empty ? "nothing found: the shot runs without a mask" : `mask covers ${(mp.coverage * 100).toFixed(1)}%${cur.crop ? " · yellow = crop box" : ""}`),
    ]) : null,
    h("div", { class: "modes", style: "margin-top:8px" }, [
      ["frame", "Full frame", "The whole frame is regenerated. The mask, if any, only feeds {target} / the setting picture."],
      ["paste", "🧩 Frame + paste", "The whole frame is generated (the pose follows the guide, the new subject may be bigger); BFS Shot Join pastes only the subject onto the original: background exact."],
      ["mask", "🎭 Mask only", "Only the mask is regenerated, on the whole frame: the rest stays exactly as it was. No crop / uncrop."],
      ["crop", "✂ Crop", "A box around the mask is regenerated (more pixels for a small subject); BFS Shot Join pastes it back."],
      ["cropmask", "✂🎭 Crop + mask", "Inside the crop, only the mask is regenerated: the most detail with the background kept."],
    ].map(([k, t, d]) => mode(c.modeOf(cur) === k, t, d, () => c.setMode(i, k)))),
    cur.color_guide && !hasMask(cur) ? h("div", { class: "err" }, "Colored Mark needs a region selection: enter a description, add points, or choose a mask video, then preview it.") : null,
    (cur.crop || cur.inpaint || cur.paste) && !hasMask(cur) ? h("div", { class: "err" }, `${c.modeName(cur)} needs a mask on this shot: a mask text, points or a mask video (or the planner's mask input).`) : null,
    cur.inpaint ? h("div", { class: "row", style: "margin-bottom:6px;align-items:center",
      title: "Opacity of the generation mask: the value of the white inside the mask (1 = fully white, the default). H3 reads a grey mask value as 'regenerate this much': at 0.85 the masked area starts at 85% of the noise and keeps ~15% of the original there. 0.8-0.9 keeps pose, outline and lighting while still swapping; too low copies the original person." }, [
      h("span", { style: "white-space:nowrap" }, `Mask opacity ${Number(cur.strength).toFixed(2)}`),
      h("input", { type: "range", min: 0.3, max: 1, step: 0.01, value: cur.strength, style: "flex:1;max-width:320px",
        onInput: e => { c.meta(i).strength = parseFloat(e.target.value); }, onChange: c.save }),
      h("button", { class: "ghost", disabled: cur.strength >= 1, onClick: () => c.setMeta(i, "strength", 1) }, "reset"),
      hint(cur.strength >= 1 ? "regenerates the masked area completely" : `keeps ~${Math.round(100 - cur.strength * 100)}% of the original inside the mask`),
    ]) : null,
    cur.inpaint ? h("div", { class: "note", style: "margin-bottom:6px" }, [h("b", "Mask only / Crop + mask: "),
      "set BFS Shot H3 Conditioning → inpaint to ", h("b", "per shot (planner)"), " (the default). Raise Expand (People / Filters / Masks tab) when the new subject is bigger than the old one."]) : null,
    cur.paste ? h("div", { class: "row", style: "margin-bottom:6px;align-items:center",
      title: "What the NEW subject is, in English, for SAM 3 on the result: BFS Shot Join also pastes its outline (when it is bigger than the old one). E.g. person, dog, cat, car. Empty = person." }, [
      h("span", { style: "white-space:nowrap" }, "new subject"),
      h("input", { type: "text", value: cur.pasteText || "", placeholder: "person (or dog, cat, car… what the result shows)", style: "flex:1;max-width:420px",
        onChange: e => c.setMeta(i, "paste_text", e.target.value) }),
      hint("its outline on the result is pasted too, so a bigger silhouette is not cut"),
    ]) : null,
    h("div", { class: "row", style: "margin-top:4px;align-items:center" }, [
      targetCrop.value[k] ? h("img", { src: targetCrop.value[k], style: "height:56px;border-radius:6px" }) : null,
      h("span", { class: "hint", style: "white-space:nowrap" }, "{target}"),
      h("input", { type: "text", value: cur.target, style: "flex:1;min-width:200px",
        placeholder: "who is replaced, e.g. the young woman in a pink crop top (empty = the person)",
        title: "Write {target} in the prompt: it becomes this description, so the model knows WHICH person to replace (useful with several people on screen).",
        onChange: e => c.setMeta(i, "target", e.target.value) }),
      h("button", { disabled: !!busy.value || !hasMask(cur),
        title: "Uses this shot's selection (🎯 Points… on the person — click the BODY, not only the face, so the outfit is described — or the mask text): SAM 3 cuts the person out and the VLM describes them",
        onClick: () => c.describeTarget(i) }, "🧑 Describe target"),
    ]),
    g ? h("div", { class: "vsug" }, [
      h("div", ["🤖 ", h("b", "segment: "), g.segment || "—", g.segment ? h("button", { onClick: () => c.applySug(i, "segment") }, "Use as mask") : null]),
      h("div", [h("b", "shot: "), g.shot || "—", g.shot ? h("button", { title: "copy (the {shot} placeholder in the prompt gets it automatically at run time)",
        onClick: () => navigator.clipboard?.writeText(g.shot) }, "Copy") : null]),
      h("div", [h("b", "recommend: "), `${g.recommend}${g.people != null ? " · " + g.people + " people" : ""}${g.reason ? " · " + g.reason : ""}`,
        g.recommend === "skip" ? h("button", { onClick: () => c.applySug(i, "skip") }, "Skip this shot") : null]),
      g.raw ? hint("unparsed answer: " + g.raw.slice(0, 200)) : null,
    ]) : null,
  ], { sub: "SAM 3 mask (text or points), crop, {target}" });

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
      hint("replaces the old “→ all” buttons"),
    ]),
  ], { open: false, sub: Object.entries(COPY_FIELDS).filter(([f]) => copyOpts[f]).map(([, l]) => l).join(", ") });

  return h("div", { class: "card" }, [head, problems, refPrompt, who, cont, copy]);
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
  ]);
}
