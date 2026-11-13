/**
 * BFS Video Segment Picker: a strip with every frame of the batch on top; click a frame to show it in the image
 * Segment Picker's canvas (the same Panel, one selection per frame). The strip marks each frame: ✓ picked by hand,
 * ↔ tracked from a picked frame, ⚠ lost or nothing. State: the hidden `selection` widget, {"frames": {"i": selection}}.
 * What a color adds on a frame (brush, points, text) is shared by every color there: each one selects from the same objects.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { wheelToCanvas } from "./bfs_wheel.js";
import { Panel, styles as pickerStyles } from "./bfs_segment_picker.js";
import { createApp, h, ref, reactive } from "./vendor/vue.esm-browser.prod.mjs";

const fileUrl = f => api.apiURL(`/view?filename=${encodeURIComponent(f.filename)}&subfolder=${encodeURIComponent(f.subfolder || "")}&type=${f.type || "temp"}`);

const SUBJECTS = [["1", "#eb2828", "red"], ["2", "#236eeb", "blue"], ["3", "#3ccd3c", "green"]];
function parse(text) {   // {frame: {subject: selection}}; a bare selection (older state) is subject 1
  let s = {};
  try { s = JSON.parse(text || "{}") || {}; } catch { s = {}; }
  const f = (s.frames && typeof s.frames === "object") ? s.frames : {};
  const out = {};
  for (const [k, v] of Object.entries(f)) {
    if (v && (v.selected || v.added)) out[k] = { "1": v };
    else if (v && typeof v === "object") {
      out[k] = Object.fromEntries(Object.entries(v).filter(([sk]) => sk === "lock" || (+sk >= 1 && +sk <= SUBJECTS.length)));
    } else out[k] = {};
  }
  return out;
}

let styled = false;
function styles() {
  if (styled) return;
  styled = true;
  const css = document.createElement("style");
  css.textContent = `
.bvsp{--card:#1b1d21;--line:#2e3138;--muted:#8b919c;--text:#e3e6eb;--accent:#3b82f6;--ok:#22c55e;--trk:#60a5fa;--bad:#f43f5e;
  position:absolute;inset:0;display:flex;flex-direction:column;gap:6px;padding:6px;box-sizing:border-box;font:12px system-ui,sans-serif;color:var(--text);overflow:hidden}
.bvsp .card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 8px;display:flex;flex-direction:column;gap:6px;flex:0 0 auto}
.bvsp .row{display:flex;align-items:center;gap:8px;min-width:0}
.bvsp .sp{flex:1}
.bvsp .muted{color:var(--muted);font-size:11px}
.bvsp .btn{background:#2a2d34;border:1px solid #3a3e47;color:var(--text);border-radius:6px;padding:4px 10px;cursor:pointer;font:inherit;white-space:nowrap}
.bvsp .btn:hover{background:#343843}
.bvsp .btn.pri{background:var(--accent);border-color:#4b8ff7;color:#fff;font-weight:600}
.bvsp .btn.pri:hover{background:#4b8ff7}
.bvsp .chk{display:flex;align-items:center;gap:4px;color:var(--muted);font-size:11px;cursor:pointer;user-select:none}
.bvsp .subj{display:flex;gap:4px}
.bvsp .subj .t{display:flex;align-items:center;gap:5px;border:1px solid var(--line);background:#23262c;border-radius:999px;padding:3px 10px 3px 6px;cursor:pointer;color:var(--muted);font-weight:600}
.bvsp .subj .t .clr{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;border-radius:50%;margin-left:4px;font-size:10px;line-height:1;background:rgba(255,255,255,.15);color:#fff;cursor:pointer}
.bvsp .subj .t .clr:hover{background:#f43f5e;color:#fff}
.bvsp .subj .t i{width:10px;height:10px;border-radius:50%;display:block}
.bvsp .subj .t.on{color:#fff;border-color:currentColor;background:#2b2f37}
.bvsp .subj .t.lock.on{color:#fbbf24;background:#44351b}
.bvsp .stat{display:flex;align-items:center;gap:8px;font-size:11px;color:#cfd6e0}
.bvsp .stat .pb{flex:0 0 110px;height:5px;background:#2a2d34;border-radius:3px;overflow:hidden}
.bvsp .stat .pb i{display:block;height:100%}
.bvsp .stat .txt{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bvsp .stat .tog{cursor:pointer;color:var(--trk);user-select:none}
.bvsp .log{max-height:90px;overflow:auto;background:#141518;border:1px solid var(--line);border-radius:6px;padding:4px 6px;font:11px ui-monospace,monospace;color:#aab}
.bvsp .counts{display:flex;gap:10px;font-size:11px;color:var(--muted)}
.bvsp .counts b{font-weight:700}
.bvsp .strip{display:flex;gap:5px;overflow-x:auto;padding-bottom:2px}
.bvsp .fr{position:relative;flex:0 0 auto;width:96px;height:58px;border-radius:6px;background:#111 center/cover no-repeat;cursor:pointer;border:2px solid transparent;outline:1px solid var(--line)}
.bvsp .fr.on{border-color:#f5b041}
.bvsp .fr .b{position:absolute;left:3px;bottom:3px;font-size:10px;background:rgba(0,0,0,.7);padding:0 4px;border-radius:3px}
.bvsp .fr .s{position:absolute;right:3px;top:3px;font-size:11px;font-weight:700;padding:0 4px;border-radius:3px;background:rgba(0,0,0,.7)}
.bvsp .fr .s.picked{color:var(--ok)}.bvsp .fr .s.tracked{color:var(--trk)}.bvsp .fr .s.lost{color:var(--bad)}
.bvsp .fr .dots{position:absolute;left:3px;top:3px;display:flex;gap:2px}
.bvsp .fr .dots i{width:8px;height:8px;border-radius:50%;border:1px solid #000;display:block}
.bvsp .tl{display:flex;gap:10px;align-items:center}
.bvsp .tl img{height:72px;border-radius:6px;border:1px solid var(--line);flex:0 0 auto}
.bvsp .tl .scrub{flex:1;display:flex;flex-direction:column;gap:6px;min-width:140px}
.bvsp .tl .track{position:relative;height:18px}
.bvsp .tl .track .bg{position:absolute;left:0;right:0;top:7px;height:4px;border-radius:2px}
.bvsp .tl .track input{position:absolute;inset:0;width:100%;margin:0;background:transparent;-webkit-appearance:none;appearance:none}
.bvsp .tl .track input::-webkit-slider-runnable-track{background:transparent;height:18px}
.bvsp .tl .track input::-moz-range-track{background:transparent}
.bvsp .tl .track input::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;margin-top:2px;border-radius:50%;background:#fff;border:2px solid var(--accent);cursor:pointer}
.bvsp .tl .track input::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:#fff;border:2px solid var(--accent);cursor:pointer}
.bvsp .tl .fn{font-variant-numeric:tabular-nums;color:var(--text);font-weight:600}
.bvsp .chips{display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.bvsp .chips .c{display:flex;align-items:center;gap:4px;border:1px solid #6b2a37;background:#2e1a1f;color:#fecdd3;border-radius:999px;padding:1px 8px;cursor:pointer;font-size:11px}
.bvsp .chips .c:hover{background:#45222b}
.bvsp .chips .c i{width:8px;height:8px;border-radius:50%;display:block}
.bvsp .chips .okk{color:var(--ok);font-size:11px}
.bvsp .host{position:absolute;inset:0}
.bvsp .empty{color:var(--muted);text-align:center;padding:12px}
`;
  document.head.appendChild(css);
}

// A run the panel starts for picking ("Segment again", "+ frame") only segments the strip: the selection it sends is marked
// segment_only, so the node skips tracking and Molmo2. "Propagate" and a normal workflow run track as usual.
const propagating = new Set();
let wrapped = false;
function wrapQueue() {
  if (wrapped) return;
  wrapped = true;
  const orig = api.queuePrompt.bind(api);
  api.queuePrompt = async function (number, p, opts) {
    const targets = opts?.partialExecutionTargets || [];
    for (const id of targets) {
      const out = p?.output?.[String(id)];
      if (out?.class_type !== "BFSVideoSegmentPicker" || propagating.has(String(id))) continue;
      try {
        const sel = JSON.parse(out.inputs.selection || "{}");
        sel.segment_only = true;
        out.inputs.selection = JSON.stringify(sel);
      } catch { /* leave it */ }
    }
    return orig(number, p, opts);
  };
}

app.registerExtension({
  name: "BFSNodes.VideoSegmentPicker",
  async nodeCreated(node) {
    if (node.comfyClass !== "BFSVideoSegmentPicker") return;
    pickerStyles(); styles();
    const widget = node.widgets?.find(w => w.name === "selection");
    if (widget) {
      widget.type = "hidden"; widget.hidden = true; widget.computeSize = () => [0, -4];
      widget.options = { ...(widget.options || {}), hidden: true, serialize: true };
      widget.serializeValue = () => widget.value;
    }
    const frames = reactive({ v: parse(widget?.value) });
    let focusFrame = 0;
    const scopeWidget = node.widgets?.find(w => w.name === "propagate_scope");
    const validScopes = ["all frames", "only this frame", "from active frame onward", "from first selected frame onward", "fill only empty masks"];
    const scope = ref(validScopes.includes(scopeWidget?.value) ? scopeWidget.value : "fill only empty masks");
    if (scopeWidget) scopeWidget.value = scope.value;
    const save = () => { if (widget) widget.value = JSON.stringify({ frames: frames.v, active_frame: focusFrame }); node.graph?.setDirtyCanvas(true); };
    const hasPicks = k => Object.values(frames.v).some(f => f?.[k]?.selected?.length);
    const clearSubject = k => {
      let changed = false;
      for (const f of Object.keys(frames.v)) {
        if (frames.v[f]?.[k]) {
          delete frames.v[f][k];
          changed = true;
        }
      }
      if (changed) {
        scope.value = "all frames";
        if (scopeWidget) scopeWidget.value = "all frames";
        save();
        mountPanel();
        if (propagatedOnce()) propagate();
      }
    };
    const run = reactive({ data: null });          // {frames: [panel data per frame], report: [...]}
    const active = ref(0);
    const subject = ref("1");                      // which subject the clicks select (its color = Picture number)

    const root = document.createElement("div");
    root.style.cssText = "position:relative;width:100%;height:100%;min-width:440px;min-height:420px";
    node.addDOMWidget("bfs_video_segment_picker", "div", root, { serialize: false, hideOnZoom: false, hideInPanel: true });
    wheelToCanvas(root);

    // the image picker's Panel for the active frame, remounted when the frame changes
    let panelApp = null, handle = {};
    const ui = { mode: "select", brush: .02 };       // the tool and brush size survive a color or frame change
    const panelHost = document.createElement("div");
    panelHost.className = "host";
    // paint strokes not applied yet are applied (to the color they were painted with) before the panel changes
    const switchPanel = async change => { try { await handle.commit?.(); } catch { /* shown in the panel */ } change(); mountPanel(); };
    const mountPanel = () => {
      panelApp?.unmount(); panelHost.innerHTML = ""; handle = {};
      const i = run.data?.indices?.[active.value] ?? active.value;   // the real frame index of the strip entry
      focusFrame = i;
      const sj = subject.value;
      const el = document.createElement("div"); el.style.cssText = "position:absolute;inset:0"; panelHost.appendChild(el);
      panelApp = createApp({ setup: () => Panel({
        node,
        get: () => JSON.stringify(frames.v[String(i)]?.[sj] || {}),
        set: v => {
          focusFrame = i;
          const before = JSON.stringify(frames.v[String(i)]?.[sj]?.selected || []);
          try { frames.v[String(i)] = { ...(frames.v[String(i)] || {}), [sj]: JSON.parse(v) }; } catch { /* keep */ }
          save();
          if (JSON.stringify(frames.v[String(i)]?.[sj]?.selected || []) !== before) autoPropagate();
        },
        expose: o => Object.assign(handle, o),
      }, { showEraser: false, showRemove: false, autoPointOnClick: sj === "lock",
           // ids stay unique on the frame, since every color sees the objects the others added
           entryPrefix: sj === "lock" ? "lock_" : sj === "1" ? "" : `s${sj}_`,
           maskColor: sj === "lock" ? [251, 191, 36] : null,
           mode: sj === "lock" ? "select" : ui.mode, brush: ui.brush,
           onMode: m => { if (sj !== "lock") ui.mode = m; }, onBrush: b => { ui.brush = b; },
           // added objects live in the run's frame data too, so the next mount (another color, back here) still has them
           onObjects: objs => { const fd = run.data?.frames?.[active.value]; if (fd) run.data.frames[active.value] = { ...fd, objects: objs }; },
           onRemoveEntry: id => {      // an added object goes for every color: no color keeps selecting a gone object
             for (const v of Object.values(frames.v[String(i)] || {})) {
               if (!v || typeof v !== "object") continue;
               if (v.added) v.added = v.added.filter(a => a.id !== id);
               if (v.selected) v.selected = v.selected.filter(s => s !== id && !s.startsWith(`${id}.`));
             }
             save();
           } }) });
      panelApp.mount(el);
      const d = run.data?.frames?.[active.value];
      if (d) setTimeout(() => handle.executed?.(d), 0);
    };

    const pickedSubjects = i => Object.entries(frames.v[String(run.data?.indices?.[i] ?? i)] || {}).filter(([, v]) => (v?.selected || []).length).map(([k]) => k);
    const status = i => {
      const own = (frames.v[String(run.data?.indices?.[i] ?? i)]?.[subject.value]?.selected || []).length > 0;
      const r = subject.value === "lock"
        ? (run.data?.protected_frames?.[i] ? "tracked protected" : "")
        : (run.data?.report?.[i]?.subjects?.[subject.value] || run.data?.report?.[i]?.source || "");
      if (own) return ["picked", "✓"];
      if (r.startsWith("tracked")) return ["tracked", "↔"];
      if (r.startsWith("lost")) return ["lost", "⚠"];
      return ["", ""];
    };
    const busy = ref("");
    const live = reactive({ text: "", progress: null, elapsed: 0, log: [], open: false, running: false });
    api.addEventListener("bfs_vsegpick_status", e => {
      const d = e.detail || {};
      if (String(d.node) !== String(node.id)) return;
      live.text = d.text; live.elapsed = d.elapsed || 0; live.running = (d.progress ?? 0) < 1;
      if (d.progress !== null && d.progress !== undefined) live.progress = d.progress;
      if (d.log) { live.log.push(`[${(d.elapsed || 0).toFixed(1)}s] ${d.text}`); if (live.log.length > 60) live.log.shift(); }
    });
    wrapQueue();
    const segmentOnly = async () => {      // a picking run: the new frame is segmented, nothing is tracked
      busy.value = "segmenting…";
      try { await handle.commit?.(); } catch { /* shown in the panel */ }
      try { const p = await app.graphToPrompt(); await api.queuePrompt(0, p, { partialExecutionTargets: [String(node.id)] }); }
      catch (e) { busy.value = String(e.message || e); }
    };
    const propagate = async () => {
      busy.value = "propagating…";
      try { await handle.commit?.(); } catch { /* shown in the panel */ }
      try {
        const p = await app.graphToPrompt();
        propagating.add(String(node.id));
        try { await api.queuePrompt(0, p, { partialExecutionTargets: [String(node.id)] }); } finally { propagating.delete(String(node.id)); }
      } catch (e) { busy.value = String(e.message || e); }
    };
    // after the first Propagate, a pick that changes a frame's selection propagates by itself (1.5 s after the last
    // click): only the shots whose picks changed are tracked again, the rest comes from the cache
    const auto = ref(localStorage.getItem("bfs_vsegpick_auto") === "1");
    let autoTimer = null;
    const propagatedOnce = () => (run.data?.report || []).some(r => /tracked|lost|Molmo/.test(JSON.stringify(r)));
    const autoPropagate = () => {
      if (!auto.value || !propagatedOnce()) return;
      clearTimeout(autoTimer);
      busy.value = "auto-propagate in 1.5 s…";
      autoTimer = setTimeout(propagate, 1500);
    };
    const addFrame = f => {          // put a frame in the strip (an empty pick) and run, so it can be picked on
      const n = run.data?.count || 0;
      f = Math.max(0, Math.min(n - 1, Math.round(+f)));
      if (!Number.isFinite(f)) return;
      focusFrame = f;
      if (!frames.v[String(f)]) { frames.v[String(f)] = {}; save(); }
      else save();
      segmentOnly();
    };
    const SUBJ = { "1": "red", "2": "blue", "3": "green" };
    const tlPos = ref(0);             // index into run.data.timeline (whole-video scrubber)
    const counts = () => {
      const r = run.data?.report || []; const n = run.data?.frames?.length || 0;
      let picked = 0, tracked = 0, lost = 0;
      for (let i = 0; i < n; i++) { const [c] = status(i); if (c === "picked") picked++; else if (c === "tracked") tracked++; else if (c === "lost") lost++; }
      return { picked, tracked, lost, n };
    };
    const tlAt = () => run.data.timeline[Math.min(tlPos.value, run.data.timeline.length - 1)];
    const lostTrack = () => {     // the scrubber's track: lost stretches in red over the video's length
      const n = Math.max(1, (run.data?.count || 1) - 1);
      const runs = [...(run.data?.lost || [])];
      if (!runs.length) return "#3a3e47";
      const stops = ["#3a3e47 0%"];
      for (const r of runs.sort((a, b) => a.start - b.start)) {
        const a = (100 * r.start / n).toFixed(2), b = (100 * Math.max(r.end, r.start + 0.5) / n).toFixed(2);
        stops.push(`#3a3e47 ${a}%`, `#f43f5e ${a}%`, `#f43f5e ${b}%`, `#3a3e47 ${b}%`);
      }
      return `linear-gradient(90deg, ${stops.join(", ")}, #3a3e47 100%)`;
    };
    createApp({
      setup: () => () => h("div", { class: "bvsp" }, [
        h("div", { class: "card" }, [
          h("div", { class: "row" }, [
            h("div", { class: "subj" }, [...SUBJECTS.map(([k, col, name]) => h("span", {
              class: ["t", subject.value === k ? "on" : ""], style: subject.value === k ? { color: col } : null,
              title: `Subject ${k} (${name}) = Picture ${k}: clicks select this subject`,
              onClick: () => { if (subject.value !== k) switchPanel(() => { subject.value = k; }); } },
              [
                h("i", { style: { background: col } }),
                h("span", { style: "color:#fff" }, `${k} · ${name}`),
                hasPicks(k) ? h("span", {
                  class: "clr",
                  title: `Clear all picks for Subject ${k} (${name})`,
                  onClick: (e) => { e.stopPropagation(); clearSubject(k); }
                }, "✕") : null,
              ])), h("span", {
                class: ["t", "lock", subject.value === "lock" ? "on" : ""],
                title: "Everything outside the selected subject is already protected. Use this padlock only for an object inside that subject's mask; click it to segment with SAM 3.",
                onClick: () => { if (subject.value !== "lock") switchPanel(() => { subject.value = "lock"; }); },
              }, [
                h("span", null, "🔒"),
                hasPicks("lock") ? h("span", {
                  class: "clr", title: "Clear all protected-object picks",
                  onClick: e => { e.stopPropagation(); clearSubject("lock"); },
                }, "✕") : null,
              ])]),
            h("span", { class: "sp" }),
            h("select", { value: scope.value, title: "Choose how a new frame selection changes the propagated masks",
              style: "max-width:190px;background:#23262c;color:#e3e6eb;border:1px solid #3a3e47;border-radius:6px;padding:4px;font:inherit",
              onChange: e => { scope.value = e.target.value; if (scopeWidget) scopeWidget.value = e.target.value; node.graph?.setDirtyCanvas(true); } },
              ["from active frame onward", "only this frame", "fill only empty masks", "all frames", "from first selected frame onward"].map(v => h("option", { value: v },
                v === "from active frame onward" ? "From this frame onward" : v === "only this frame" ? "Only this frame"
                  : v === "fill only empty masks" ? "Fill only empty masks" : v === "all frames" ? "Recompute all frames" : "From earliest selected frame"))),
            h("label", { class: "chk", title: "After a selection changes, apply the selected fix mode automatically after 1.5 seconds" }, [
              h("input", { type: "checkbox", checked: auto.value, onChange: e => { auto.value = e.target.checked;
                try { localStorage.setItem("bfs_vsegpick_auto", e.target.checked ? "1" : "0"); } catch { /* ignore */ } } }), "auto"]),
            h("button", { class: "btn pri", onClick: propagate, title: "Apply the selected fix mode to the current selection" }, "↔ Apply fix"),
          ]),
          live.text || busy.value ? h("div", { class: "stat" }, [
            h("span", { class: "pb" }, [h("i", { style: { width: `${Math.round(100 * (live.progress ?? 0))}%`,
              background: live.running ? "#3b82f6" : "#22c55e" } })]),
            h("span", { class: "txt", title: live.text }, busy.value && !live.running ? busy.value
              : `${live.running ? "⏳" : "✓"} ${live.text} · ${live.elapsed}s`),
            h("span", { class: "tog", onClick: () => { live.open = !live.open; } }, live.open ? "▾ log" : "▸ log"),
          ]) : h("div", { class: "muted" }, subject.value === "lock"
            ? "🔒 Outside the subject is already protected. Click an object inside the subject mask to keep it unchanged, then Apply fix."
            : "Pick a subject, click a frame below, then click its person. Subject N = Picture N."),
          live.open ? h("div", { class: "log" }, live.log.length ? live.log.map((l, i) => h("div", { key: i }, l)) : "no log yet") : null,
        ]),
        h("div", { class: "card" }, [
          h("div", { class: "row" }, [
            h("span", { style: "font-weight:600" }, "Frames"),
            run.data ? h("span", { class: "counts" }, (() => { const c = counts(); return [
              h("span", null, [h("b", { style: "color:#22c55e" }, "✓ "), `${c.picked} picked`]),
              h("span", null, [h("b", { style: "color:#60a5fa" }, "↔ "), `${c.tracked} tracked`]),
              h("span", null, [h("b", { style: "color:#f43f5e" }, "⚠ "), `${c.lost} lost`])]; })()) : null,
          ]),
          run.data?.frames?.length
            ? h("div", { class: "strip" }, run.data.frames.map((d, i) => {
                const [cls, mark] = status(i);
                return h("div", { class: ["fr", i === active.value ? "on" : ""], key: i, title: run.data.report?.[i]?.source || "",
                  style: { backgroundImage: `url("${fileUrl(d.image)}")` },
                  onClick: () => { if (active.value !== i) switchPanel(() => { active.value = i; focusFrame = run.data.indices?.[i] ?? i; save(); }); } },
                  [h("span", { class: "b" }, `f${run.data.indices?.[i] ?? i}`), mark ? h("span", { class: ["s", cls] }, mark) : null,
                   h("span", { class: "dots" }, pickedSubjects(i).map(k => k === "lock"
                     ? h("span", { title: "Protected object picked", style: "color:#fbbf24;background:#181818;border-radius:3px;padding:0 2px" }, "🔒")
                     : h("i", { style: { background: SUBJECTS[(+k - 1) % SUBJECTS.length][1] } })))]);
              }))
            : h("div", { class: "empty" }, "Run the node once: the frames appear here. Click a frame, then the person."),
        ]),
        run.data?.timeline?.length ? h("div", { class: "card" }, [
          h("div", { class: "tl" }, [
            h("img", { src: fileUrl(tlAt()), title: "every frame with its masks: scrub to find a wrong or missing mark" }),
            h("div", { class: "scrub" }, [
              h("div", { class: "row" }, [
                h("span", { style: "font-weight:600" }, "Whole video"),
                h("span", { class: "fn" }, `f${tlAt().frame}`),
                h("span", { class: "muted" }, `/ ${(run.data.count || 1) - 1}`),
                h("span", { class: "sp" }),
                h("button", { class: "btn", onClick: () => addFrame(tlAt().frame),
                  title: "put this frame in the strip to pick or fix the person on it" }, "＋ add this frame"),
              ]),
              h("div", { class: "track" }, [
                h("div", { class: "bg", style: { background: lostTrack() } }),
                h("input", { type: "range", min: 0, max: run.data.timeline.length - 1, value: tlPos.value,
                  onInput: e => { tlPos.value = +e.target.value; } }),
              ]),
              h("div", { class: "chips" }, (run.data.lost || []).length
                ? (run.data.lost || []).slice(0, 12).map(r => h("span", { class: "c", title: "add the middle frame of this stretch to the strip",
                    onClick: () => addFrame((r.start + r.end) / 2) },
                    [h("i", { style: { background: SUBJECTS[(+r.subject - 1) % SUBJECTS.length]?.[1] || "#999" } }), `f${r.start}–${r.end}`]))
                : [h("span", { class: "okk" }, "✓ no lost stretches")]),
            ]),
          ]),
        ]) : null,
        h("div", { ref: el => { if (el && !el.contains(panelHost)) { el.appendChild(panelHost); if (!panelApp) mountPanel(); } }, style: "flex:1;position:relative;min-height:260px;border:1px solid #2e3138;border-radius:8px;overflow:hidden" }),
      ]),
    }).mount(root);

    const onExecuted = node.onExecuted;
    node.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      const d = output?.bfs_vsegpick?.[0];
      if (!d) return;
      run.data = d; busy.value = "";
      const focusIndex = d.indices?.indexOf?.(focusFrame) ?? -1;
      if (focusIndex >= 0) active.value = focusIndex;
      if (active.value >= d.frames.length) active.value = 0;
      mountPanel();
    };
    const onConfigure = node.onConfigure;
    node.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      setTimeout(() => {
        let saved = {};
        try { saved = JSON.parse(widget?.value || "{}"); } catch { /* ignore */ }
        frames.v = parse(widget?.value); focusFrame = Number.isInteger(+saved.active_frame) ? +saved.active_frame : 0;
        if (scopeWidget?.value) scope.value = scopeWidget.value;
        mountPanel();
      }, 0);
      return r;
    };
    node.size = [Math.max(node.size[0], 780), Math.max(node.size[1], 1180)];
  },
});
