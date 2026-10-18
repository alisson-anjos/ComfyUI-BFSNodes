// The busy bar and the points window (🎯 Points…: click on a frame to pick the subject for SAM 3).
import { Teleport } from "../vendor/vue.esm-browser.prod.mjs";
import { h, hint, t, pictureColor, markName } from "./common.js";

export function busyBar(c, msg) {
  const { busySince, now, upPct, status } = c;
  const secs = busySince.value ? Math.max(0, Math.round((now.value - busySince.value) / 1000)) : 0;
  const pct = upPct.value >= 0 ? upPct.value : (status.total > 0 ? Math.min(1, status.done / status.total) : -1);
  const stage = status.stage && !msg.startsWith(status.stage) ? status.stage : "";
  return h("div", { class: "busybar" }, [
    h("div", { class: "brow" }, [h("span", { class: "spin" }), h("b", msg), stage ? hint(`· ${stage}`) : null,
      h("span", { class: "grow" }), pct >= 0 ? h("span", `${Math.round(pct * 100)}%`) : null, hint(`${secs}s`)]),
    h("div", { class: ["bprog", pct < 0 && "ind"] }, [h("div", { style: pct >= 0 ? `width:${(pct * 100).toFixed(1)}%` : "" })]),
  ]);
}

export function pointsModal(c) {
  const { modal, segs, plan, api } = c;
  const ms = segs.value[modal.idx];
  if (!modal.open || !ms) return null;
  // Colored Mark: region 1 = the shot's own selection, 2.. = its extra regions; each one is a picture, in its color
  const regionPicture = r => (r === 1 ? ms.color_guide_picture : ms.regions[r - 2]?.picture) || 1;
  const regionColor = r => (ms.color_guide ? pictureColor(plan, regionPicture(r)) : "#3ccf6b");
  const regionsN = ms.color_guide ? 1 + ms.regions.length : 1;
  const regions = Array.from({ length: regionsN }, (_, q) => q + 1);
  // the points of a region on the frame shown (the one being edited are in modal.points)
  const pointsAt = r => (r === modal.region ? modal.points : modal.groups[r]?.keys.find(k => k.key === modal.key)?.points || []);
  const keysOf = r => {
    const ks = new Set((modal.groups[r]?.keys || []).map(k => k.key));
    if (r === modal.region) { if (modal.points.length) ks.add(modal.key); else ks.delete(modal.key); }
    return [...ks].sort((a, b) => a - b);
  };
  const edited = () => { modal.stale = !!modal.prev; };
  const pointOverlays = regions.flatMap(r => pointsAt(r).map((p, k) => h("div", {
    class: ["pt", ms.color_guide && "cg", r !== modal.region && "other", p.label ? "pos" : "neg"],
    style: `left:${p.x * 100}%;top:${p.y * 100}%;--point-color:${regionColor(r)}`,
    title: r === modal.region ? "click to remove" : `Picture ${regionPicture(r)}`,
    onPointerdown: e => { if (r !== modal.region) return; e.stopPropagation(); modal.points = modal.points.filter((_, j) => j !== k); edited(); },
  })));
  const regionButton = r => {
    const k = regionPicture(r), name = ms.names[k - 1];
    return h("button", { class: modal.region === r ? "pri" : "", onClick: () => c.switchPointRegion(r),
      title: `Region ${r}: becomes <Picture ${k}> (${markName(regionColor(r))})` }, [
      name ? h("img", { class: "mthumb", src: c.viewUrl(name), style: `--mark:${regionColor(r)}` }) : null,
      ` Picture ${k} · ${markName(regionColor(r))} (${keysOf(r).length} key frame${keysOf(r).length === 1 ? "" : "s"})`]);
  };
  // after Segment: the mask over every frame of the shot (modal.prev.frames, one per frame)
  const masked = modal.prev && modal.showMask ? modal.prev.frames.find(f => f.f === ms.start + modal.key) : null;
  const play = () => {
    if (modal.playing) { clearInterval(modal.playing); modal.playing = 0; return; }
    modal.playing = setInterval(() => {
      if (modal.key >= ms.len - 1 || !modal.open) { clearInterval(modal.playing); modal.playing = 0; return; }
      c.showFrame(modal.key + 1);
    }, 1000 / Math.min(24, plan.fps || 24));
  };
  const marks = h("div", { class: "kmarks" }, regions.flatMap(r => keysOf(r).map(f => h("span", {
    class: ["kmark", r === modal.region && "on"], style: `left:${(f / Math.max(1, ms.len - 1)) * 100}%;--mark:${regionColor(r)}`,
    title: `key frame ${ms.start + f}`, onClick: () => c.showFrame(f) }))));
  return h(Teleport, { to: "body" }, h("div", { class: "bsl", style: "background:none;padding:0" }, h("div", { class: "mmodal",
    onKeydown: e => {
      e.stopPropagation();
      if (e.target.tagName === "INPUT" && e.target.type === "text") return;
      if (e.key === "ArrowLeft") c.showFrame(modal.key - 1);
      if (e.key === "ArrowRight") c.showFrame(modal.key + 1);
    }, onPointerdown: e => { if (e.target === e.currentTarget) modal.open = false; } }, [
    h("div", { class: "mbox", tabindex: 0 }, [
      h("div", { class: "row", style: "margin-bottom:6px" }, [h("span", { class: "sect" }, `Shot #${modal.idx + 1} · points`),
        ms.color_guide ? h("div", { class: "row" }, regions.map(regionButton)) : null,
        hint("click = keep · right-click or shift+click = exclude · click a point to remove it · points on another frame = a new key frame: the mask is tracked from each key frame up to the next one")]),
      h("div", { class: "mstage" }, [
      h("div", { class: "mimg", onContextmenu: e => e.preventDefault(), onPointerdown: e => {
        const box = e.currentTarget.getBoundingClientRect();
        const x = (e.clientX - box.left) / box.width, y = (e.clientY - box.top) / box.height;
        modal.points = [...modal.points, { x: +x.toFixed(4), y: +y.toFixed(4), label: (e.button === 2 || e.shiftKey) ? 0 : 1 }];
        edited();
      } }, [
        h("img", { draggable: false, src: masked ? masked.src : api.apiURL(`/bfs/shotloop/frame?video=${encodeURIComponent(plan.video)}&fps=${plan.fps}&f=${ms.start + modal.key}&w=900`) }),
        ...pointOverlays,
      ])]),
      h("div", { class: "row", style: "margin-top:6px;align-items:center" }, [
        h("button", { title: "play the shot (with the mask after Segment)", onClick: play }, modal.playing ? "⏸" : "▶"),
        h("button", { title: "previous frame (←)", onClick: () => c.showFrame(modal.key - 1) }, "◀"),
        h("button", { title: "next frame (→)", onClick: () => c.showFrame(modal.key + 1) }, "▶|"),
        hint(`frame ${ms.start + modal.key} (${modal.key + 1}/${ms.len})`),
        h("div", { class: "kslider" }, [
          h("input", { type: "range", min: 0, max: ms.len - 1, step: 1, value: modal.key, style: "width:100%",
            onInput: e => c.showFrame(parseInt(e.target.value)) }),
          marks,
        ]),
        modal.prev ? h("label", { class: "row", style: "gap:4px" }, [h("input", { type: "checkbox", checked: modal.showMask,
          onChange: e => { modal.showMask = e.target.checked; } }), "mask"]) : null,
      ]),
      h("div", { class: "row", style: "margin-top:4px;gap:4px;flex-wrap:wrap" }, [
        hint("key frames:"),
        ...(keysOf(modal.region).length ? keysOf(modal.region).map(f => h("span", { class: ["kchip", f === modal.key && "on"] }, [
          h("button", { class: "sm", onClick: () => c.showFrame(f) }, `frame ${ms.start + f}`),
          h("button", { class: "sm dng", title: "remove this key frame's points", onClick: () => { c.dropKeyFrame(f); edited(); } }, "✕")]))
          : [hint("none yet: click on the frame to add points")]),
      ]),
      h("div", { class: "row", style: "margin-top:6px" }, [
        h("input", { type: "text", value: modal.text, style: "flex:1", placeholder: "text prompt (used when there are no points)",
          onChange: e => { modal.text = e.target.value; edited(); } }),
        h("button", { title: "remove the points of this frame", onClick: () => { modal.points = []; edited(); } }, "Clear frame"),
        h("button", { class: "pri", disabled: !!modal.busy || !(keysOf(modal.region).length || modal.text),
          onClick: () => c.previewMask(modal.idx, { ...c.pointSpec(), ...(modal.region > 1 ? { max_objects: 1 } : {}),
            preview_color: regionColor(modal.region) }, true) }, modal.prev && modal.stale ? "👁 Segment again" : "👁 Segment"),
      ]),
      modal.busy ? busyBar(c, modal.busy) : null,
      modal.prev ? hint(modal.prev.empty ? "nothing found" : `covers ${(modal.prev.coverage * 100).toFixed(1)}% of the frames`
        + (modal.stale ? " · the points changed: Segment again to see the new mask" : " · play or scrub to check every frame; where it slips, add points on that frame"),
        "display:block;margin-top:4px") : null,
      h("div", { class: "row", style: "margin-top:8px;justify-content:flex-end" }, [
        h("button", { onClick: () => { modal.open = false; } }, "Cancel"),
        !ms.color_guide ? h("button", { title: "Apply this frame's points to EVERY shot, each on its frame at the same relative position (a subject that stays in place)",
          onClick: c.savePointsAll }, "Save → all shots") : null,
        h("button", { disabled: !modal.points.length || ms.color_guide,
          title: "Track this frame's selection with SAM 3 through the whole video (or the range), forward and backward from this frame; every shot without its own selection takes its part of the track. Also type a short text (e.g. 'woman in pink top') so the subject is found again after camera cuts.",
          onClick: c.saveTrack }, "🎯 Track → whole video"),
        h("button", { class: "pri", onClick: c.savePoints }, "Save (this shot)"),
      ]),
    ]),
  ])));
}

const TOUR_STEPS = [
  ["video", "Video", "Choose a video, set the frame range, split at camera cuts, then set the generation size."],
  ["shots", "Shots", "Review each shot, assign its references and prompt, and choose how the result is applied."],
  ["people", "People", "Track detected face identities to filter or split shots. Set masks here or per shot."],
  ["prompts", "Prompts", "Set shared references, prompts and optional VLM assistance. Written prompts are kept exactly as authored."],
  ["shots", "Colored Mark", "Colored Mark adds a translucent color overlay to a selected region in the source video. It does not enable inpainting. Choose the color before marking the region with points or a description."],
  ["run", "Run", "Check warnings, choose auto or queue loop, and set filters or quick-run limits."],
];

export function tourModal(c) {
  const { tour, tab } = c;
  const step = tour.value - 1;
  if (step < 0 || step >= TOUR_STEPS.length) return null;
  const [targetTab, title, body] = TOUR_STEPS[step];
  return h(Teleport, { to: "body" }, h("div", { class: "bsl", style: "background:none;padding:0" },
    h("div", { class: "tourback", role: "dialog", "aria-modal": "true", "aria-label": t("Shot Planner tour"),
      onKeydown: e => { if (e.key === "Escape") tour.value = 0; }, onPointerdown: e => { if (e.target === e.currentTarget) tour.value = 0; } }, [
    h("div", { class: "tourbox" }, [
      h("div", { class: "row" }, [h("b", t("Shot Planner tour")), h("span", { class: "grow" }), h("span", { class: "hint" }, `${step + 1} / ${TOUR_STEPS.length}`),
        h("button", { class: "ghost", title: t("Close tour"), "aria-label": t("Close tour"), onClick: () => { tour.value = 0; } }, "✕")]),
      h("h3", t(title)),
      h("p", t(body)),
      h("p", { class: "hint" }, t("This tour is optional. Use Next to visit each main tab; close it whenever you like.")),
      h("div", { class: "row", style: "justify-content:flex-end;margin-top:12px" }, [
        step > 0 ? h("button", { onClick: () => { tour.value--; tab.value = TOUR_STEPS[step - 1][0]; } }, t("Back")) : null,
        h("button", { class: "pri", onClick: () => {
          if (step === TOUR_STEPS.length - 1) tour.value = 0;
          else { tour.value++; tab.value = TOUR_STEPS[step + 1][0]; }
        } }, t(step === TOUR_STEPS.length - 1 ? "Close tour" : "Next")),
      ]),
    ]),
  ])));
}
