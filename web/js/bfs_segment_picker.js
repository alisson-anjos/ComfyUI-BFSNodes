/**
 * BFS Segment Picker: the canvas of the node. It shows the objects SAM 3 found in the image, colored; hover one to
 * see it, click to select it. Objects the automatic pass missed are added by text (a SAM 3 prompt), by points
 * (keep / exclude), segmented at once by /bfs/segpick/add, or painted with the brush (to grow the mark by hand).
 * ➖ Remove (points) and 🧽 Eraser (brush) make the same kinds of objects, cut out of the mark instead of added to it.
 * Shift while painting does the opposite of the mode: with the brush it erases the mark, with the eraser it paints back.
 * With an object selected, the brush edits that object's own mask (grows it, shift trims it) instead of adding a new one:
 * the selected object the stroke starts on, otherwise the nearest selected one. ↺ on its chip drops the edits.
 *
 * All state lives in the hidden `selection` STRING widget (JSON: selected ids and the added entries), so a workflow
 * saves, reloads and shares the selection. The objects themselves come from the node's last run (onExecuted).
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { wheelToCanvas } from "./bfs_wheel.js";
import { createApp, h, ref, reactive, computed, watch, nextTick, onMounted, onBeforeUnmount } from "./vendor/vue.esm-browser.prod.mjs";

// same order as PALETTE in bfs_segment_picker.py, so the panel and the overlay output agree
const PALETTE = [[230, 70, 60], [45, 125, 225], [50, 180, 90], [235, 205, 35], [205, 55, 180], [30, 200, 205],
  [240, 130, 35], [135, 75, 210], [120, 200, 60], [225, 95, 150], [80, 160, 230], [190, 150, 90]];
const CUT = [25, 25, 25];        // remove / eraser objects: a dark hole, unlike any PALETTE color
const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const DEFAULT_W = 760, DEFAULT_H = 820;

const fileUrl = f => api.apiURL(`/view?filename=${encodeURIComponent(f.filename)}&subfolder=${encodeURIComponent(f.subfolder || "")}&type=${f.type || "temp"}`);

function parseState(text) {
  let s = {};
  try { s = JSON.parse(text || "{}") || {}; } catch { s = {}; }
  return { selected: Array.isArray(s.selected) ? s.selected.map(String) : [], added: Array.isArray(s.added) ? s.added : [],
           next: Number(s.next) || (Array.isArray(s.added) ? s.added.length + 1 : 1) };
}

const STYLE_ID = "bfs-segpick-style";
export function styles() {
  if (document.getElementById(STYLE_ID)) return;
  const css = document.createElement("style");
  css.id = STYLE_ID;
  css.textContent = `
.bsp{position:absolute;inset:0;display:flex;flex-direction:column;gap:6px;padding:6px;box-sizing:border-box;font:12px system-ui,sans-serif;color:#ddd}
.bsp .bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.bsp button{background:#333;color:#ddd;border:1px solid #555;border-radius:4px;padding:3px 8px;cursor:pointer;font:inherit}
.bsp button:hover:not(:disabled){background:#444}
.bsp button:disabled{opacity:.45;cursor:default}
.bsp button.pri{background:#2d5a8a;border-color:#3d6fa5;color:#fff}
.bsp button.on{background:#7a3d8a;border-color:#9a55aa;color:#fff}
.bsp input[type=text]{background:#222;color:#ddd;border:1px solid #555;border-radius:4px;padding:3px 6px;font:inherit;flex:1;min-width:120px}
.bsp .stage{flex:1;min-height:200px;position:relative;display:flex;align-items:center;justify-content:center;background:#151515;border-radius:4px;overflow:hidden}
.bsp .frame{position:relative}
.bsp .frame img,.bsp .frame canvas{position:absolute;inset:0;width:100%;height:100%;user-select:none}
.bsp .frame canvas{cursor:pointer}
.bsp .frame.points canvas,.bsp .frame.remove canvas{cursor:crosshair}
.bsp .frame.brush canvas,.bsp .frame.eraser canvas{cursor:none}
.bsp .chip.cut{border-style:dashed}
.bsp .frame.remove .pt.pos{background:#e74c3c}.bsp .frame.remove .pt.neg{background:#2ecc71}
.bsp input[type=range]{width:90px}
.bsp .pt{position:absolute;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;border:2px solid #fff;pointer-events:auto;cursor:pointer;box-shadow:0 0 3px #000}
.bsp .pt.pos{background:#2ecc71}.bsp .pt.neg{background:#e74c3c}
.bsp .empty{color:#888;text-align:center;padding:20px;line-height:1.6}
.bsp .tip{position:absolute;left:6px;top:6px;background:rgba(0,0,0,.75);padding:2px 6px;border-radius:3px;pointer-events:none}
.bsp .list{display:flex;flex-wrap:wrap;gap:4px;max-height:96px;overflow:auto}
.bsp .chip{display:flex;align-items:center;gap:4px;padding:2px 6px;border-radius:10px;border:1px solid #444;background:#222;cursor:pointer}
.bsp .chip.sel{border-color:#fff;background:#3a3a3a}
.bsp .chip .sw{width:10px;height:10px;border-radius:2px}
.bsp .chip .x{color:#999;padding:0 2px}.bsp .chip .x:hover{color:#f66}
.bsp .hint{color:#888}.bsp .err{color:#f77}
`;
  document.head.appendChild(css);
}

/** One loaded object: a colored canvas to draw and its mask alpha for hit tests, at the preview resolution. */
function loadMask(o, color) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const w = img.naturalWidth, hh = img.naturalHeight;
      const c = document.createElement("canvas"); c.width = w; c.height = hh;
      const g = c.getContext("2d");
      g.drawImage(img, 0, 0);
      const alpha = g.getImageData(0, 0, w, hh).data;
      const hit = new Uint8Array(w * hh);
      for (let i = 0; i < hit.length; i++) hit[i] = alpha[i * 4 + 3] > 127 ? 1 : 0;
      g.globalCompositeOperation = "source-in";
      g.fillStyle = rgb(color); g.fillRect(0, 0, w, hh);
      // the outline: mask pixels with a background pixel within r, so it stays visible once scaled down
      const r = Math.max(1, Math.round(Math.max(w, hh) / 400));
      const e = document.createElement("canvas"); e.width = w; e.height = hh;
      const eg = e.getContext("2d"), px = eg.createImageData(w, hh), [cr, cg, cb] = color;
      for (let y = 0; y < hh; y++) for (let x = 0; x < w; x++) {
        if (!hit[y * w + x]) continue;
        let edge = false;
        for (let dy = -r; dy <= r && !edge; dy++) for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= hh || !hit[yy * w + xx]) { edge = true; break; }
        }
        if (edge) { const i = (y * w + x) * 4; px.data[i] = cr; px.data[i + 1] = cg; px.data[i + 2] = cb; px.data[i + 3] = 255; }
      }
      eg.putImageData(px, 0, 0);
      resolve({ canvas: c, edge: e, hit, w, h: hh });
    };
    img.onerror = reject;
    img.src = fileUrl(o.mask);
  });
}

export function Panel(io, options = {}) {   // exported: BFS Video Segment Picker shows one per frame
  const showEraser = options.showEraser !== false;
  const showRemove = options.showRemove !== false;
  const state = reactive(parseState(io.get()));
  const data = ref(null);            // {session, width, height, image, objects} from the last run
  const loaded = new Map();          // object id -> loadMask result (not reactive: canvases and big arrays)
  const ready = ref(0);              // bumps when masks finish loading, to redraw
  const hover = ref(null);
  const mode = ref(options.mode || "select");        // "select" | "points" | "brush" | "remove" (points) | "eraser" (brush)
  const points = ref([]);            // the point entry being built
  const strokes = ref([]);           // the brush entry being built: [{r, pts: [[x, y], ...]}], relative to the image
  const brush = ref(options.brush || .02);   // brush radius, relative to the image width
  let painting = false, cursor = null, shiftHeld = false;
  const text = ref("");
  const busy = ref(""); const error = ref("");
  const stageEl = ref(null); const canvasEl = ref(null);
  const fit = reactive({ w: 0, h: 0 });

  const save = () => io.set(JSON.stringify({ selected: state.selected, added: state.added, next: state.next }));
  const entryId = kind => `${options.entryPrefix || ""}${kind}${state.next}`;
  const objects = computed(() => data.value?.objects || []);
  const colorOf = id => {
    const i = Math.max(0, objects.value.findIndex(o => o.id === id));
    return objects.value[i]?.remove ? CUT : PALETTE[i % PALETTE.length];
  };
  const isSel = id => state.selected.includes(id);
  const pointing = () => mode.value === "points" || mode.value === "remove";
  const brushing = () => mode.value === "brush" || mode.value === "eraser";
  const cutting = () => mode.value === "remove" || mode.value === "eraser";
  function setMode(m) {
    if (m !== mode.value) { points.value = []; strokes.value = []; }   // unfinished points / strokes mean something else in another mode
    mode.value = m; hover.value = null;
    options.onMode?.(m);
  }

  async function loadAll() {
    const objs = objects.value;
    await Promise.all(objs.map(async (o, i) => {
      const key = `${o.id}|${o.mask.filename}`;
      if (loaded.get(o.id)?.key === key) return;
      try { loaded.set(o.id, { key, ...(await loadMask(o, o.remove ? CUT : (options.maskColor || PALETTE[i % PALETTE.length]))) }); } catch { /* shown as missing */ }
    }));
    ready.value++;
  }

  function relayout() {
    const el = stageEl.value, d = data.value;
    if (!el || !d) return;
    const W = el.clientWidth - 2, H = el.clientHeight - 2, s = Math.min(W / d.width, H / d.height);
    fit.w = Math.max(1, Math.floor(d.width * s)); fit.h = Math.max(1, Math.floor(d.height * s));
  }

  function draw() {
    const c = canvasEl.value, d = data.value;
    if (!c || !d) return;
    const first = loaded.values().next().value;
    const W = first?.w || fit.w, H = first?.h || fit.h;
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const g = c.getContext("2d");
    g.clearRect(0, 0, W, H);
    // biggest first, so small objects stay visible on top of the ones around them
    const order = [...objects.value].sort((a, b) => b.area - a.area);
    // the image itself is never tinted: an object is an outline, filled only while hovered (light) or selected
    for (const o of order) {
      const m = loaded.get(o.id);
      if (!m) continue;
      const sel = isSel(o.id), hov = hover.value === o.id;
      if (sel || hov) { g.globalAlpha = sel ? (hov ? .6 : .45) : .3; g.drawImage(m.canvas, 0, 0, W, H); }
      if (mode.value !== "select" && !sel && !hov) continue;
      g.globalAlpha = sel || hov ? 1 : .7;
      g.drawImage(m.edge, 0, 0, W, H);
    }
    g.globalAlpha = 1;
    if (!brushing()) return;
    g.lineCap = g.lineJoin = "round";
    for (const st of strokes.value) {
      g.strokeStyle = g.fillStyle = cutting() !== !!st.erase ? "rgba(20,20,20,.6)" : "rgba(255,0,255,.55)";
      const [x0, y0] = st.pts[0];
      g.beginPath(); g.arc(x0 * W, y0 * H, st.r * W, 0, 2 * Math.PI); g.fill();
      if (st.pts.length < 2) continue;
      g.lineWidth = 2 * st.r * W; g.beginPath(); g.moveTo(x0 * W, y0 * H);
      for (const [x, y] of st.pts.slice(1)) g.lineTo(x * W, y * H);
      g.stroke();
    }
    if (cursor) {
      g.lineWidth = Math.max(1, W / 500); g.strokeStyle = cutting() !== shiftHeld ? "#f43f5e" : "#fff";
      g.beginPath(); g.arc(cursor.x * W, cursor.y * H, brush.value * W, 0, 2 * Math.PI); g.stroke();
    }
  }
  watch([ready, hover, mode, brush, () => strokes.value.length, () => [...state.selected], () => fit.w, () => fit.h], () => nextTick(draw));

  function at(e) {
    const r = canvasEl.value.getBoundingClientRect();
    return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) };
  }
  function objectAt(p) {
    let best = null;
    for (const o of objects.value) {
      const m = loaded.get(o.id);
      if (!m) continue;
      const i = Math.min(m.h - 1, Math.floor(p.y * m.h)) * m.w + Math.min(m.w - 1, Math.floor(p.x * m.w));
      if (m.hit[i] && (!best || o.area < best.area)) best = o;    // the smallest one under the mouse: inner objects stay pickable
    }
    return best;
  }

  function toggle(id) {
    state.selected = isSel(id) ? state.selected.filter(i => i !== id) : [...state.selected, id];
    save();
  }
  function onPointer(e) {
    if (!data.value) return;
    e.preventDefault(); e.stopPropagation();
    const p = at(e);
    if (pointing()) {
      points.value = [...points.value, { x: +p.x.toFixed(4), y: +p.y.toFixed(4), label: (e.button === 2 || e.shiftKey) ? 0 : 1 }];
      return;
    }
    if (brushing()) {
      canvasEl.value.setPointerCapture(e.pointerId);
      painting = true;
      const st = { r: +brush.value.toFixed(4), pts: [[+p.x.toFixed(4), +p.y.toFixed(4)]] };
      if (e.shiftKey) st.erase = true;       // shift: the opposite of the mode (erase with the brush, paint back with the eraser)
      strokes.value = [...strokes.value, st];
      return;
    }
    const o = objectAt(p);
    if (options.autoPointOnClick && (!o || !o.entry)) {
      add({ id: entryId("p"), points: [{ x: +p.x.toFixed(4), y: +p.y.toFixed(4), label: 1 }] });
    } else if (o) toggle(o.id);
  }
  function onMove(e) {
    if (data.value && brushing()) {
      cursor = at(e); shiftHeld = e.shiftKey;
      const pts = strokes.value.at(-1)?.pts, last = pts?.at(-1);
      // a point every ~quarter radius keeps the stroke smooth without bloating the saved JSON
      if (painting && Math.hypot(cursor.x - last[0], cursor.y - last[1]) > brush.value / 4) pts.push([+cursor.x.toFixed(4), +cursor.y.toFixed(4)]);
      draw();
      return;
    }
    if (!data.value || pointing()) { hover.value = null; return; }
    hover.value = objectAt(at(e))?.id ?? null;
  }

  // every brush edit on one object, in order: the video picker passes the ones every color made on the frame
  const editsOf = target => (options.allAdded?.() || state.added).filter(a => a.target === target);
  async function editObject(target, edits) {
    const r = await api.fetchApi("/bfs/segpick/edit", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: data.value.session, target, edits }) });
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || r.statusText);
    const o = j.objects[0];
    data.value = { ...data.value, objects: data.value.objects.map(x => (x.id === o.id ? o : x)) };
    options.onObjects?.(data.value.objects);
    await loadAll();
  }
  async function add(entry) {
    if (!data.value) { error.value = "Run the node first (▶ Segment)."; return; }
    if (entry.target) {
      busy.value = "Painting on the mask…"; error.value = "";
      try {
        await editObject(entry.target, [...editsOf(entry.target).filter(a => a.id !== entry.id), entry]);
        state.added = [...state.added.filter(a => a.id !== entry.id), entry];
        state.next = Math.max(state.next, Number(entry.id.match(/(\d+)$/)?.[1] || 0) + 1);
        if (!isSel(entry.target)) state.selected = [...state.selected, entry.target];
        save();
      } catch (err) { error.value = String(err.message || err); } finally { busy.value = ""; }
      return;
    }
    busy.value = entry.text ? `Finding “${entry.text}”…` : entry.strokes ? "Adding the painted area…" : "Segmenting the points…"; error.value = "";
    try {
      const r = await api.fetchApi("/bfs/segpick/add", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session: data.value.session, entry }) });
      const j = await r.json();
      if (!r.ok || j.error) throw new Error(j.error || r.statusText);
      if (!j.objects.length) { error.value = entry.text ? `Nothing found for “${entry.text}”: try other words or a lower threshold.` : entry.strokes ? "The painted area is empty." : "The points gave an empty mask."; return; }
      state.added = [...state.added.filter(a => a.id !== entry.id), entry];
      state.next = Math.max(state.next, Number(entry.id.match(/(\d+)$/)?.[1] || 0) + 1);
      data.value = { ...data.value, objects: [...data.value.objects.filter(o => o.entry !== entry.id), ...j.objects] };
      state.selected = [...state.selected, ...j.objects.map(o => o.id).filter(id => !isSel(id))];   // what you add is what you want
      save();
      options.onObjects?.(data.value.objects);
      await loadAll();
    } catch (err) {
      error.value = String(err.message || err);
    } finally { busy.value = ""; }
  }
  const addText = () => { const t = text.value.trim(); if (t) { add({ id: entryId("t"), text: t }); text.value = ""; } };
  const addPoints = async () => {
    if (!points.value.some(p => p.label)) return;
    await add(cutting() ? { id: entryId("r"), remove: true, points: points.value } : { id: entryId("p"), points: points.value });
    if (!error.value) setMode("select");
  };
  // The strokes become up to two entries, in painting order: what they add (cut strokes painted as holes in it) and what
  // they cut (painting strokes after a cut painted back), so erasing and painting over the same spot ends as painted last.
  // The object a brush stroke edits: the selected one it starts on (the smallest), otherwise the nearest selected one
  const targetFor = st => {
    const [x, y] = st.pts[0];
    const sel = objects.value.filter(o => isSel(o.id) && !o.remove && loaded.get(o.id));
    if (!sel.length) return null;
    const under = sel.filter(o => { const m = loaded.get(o.id); return m.hit[Math.min(m.h - 1, Math.floor(y * m.h)) * m.w + Math.min(m.w - 1, Math.floor(x * m.w))]; });
    if (under.length) return under.sort((a, b) => a.area - b.area)[0].id;
    const dist = o => { const [bx, by, bw, bh] = o.box || [0, 0, 1, 1]; return Math.hypot(Math.max(bx - x, 0, x - bx - bw), Math.max(by - y, 0, y - by - bh)); };
    return sel.sort((a, b) => dist(a) - dist(b))[0].id;
  };
  const commitStrokes = async () => {
    const all = strokes.value;
    if (!all.length || !data.value) return;
    const target = mode.value === "brush" ? targetFor(all[0]) : null;
    if (target) {          // one edit of that object's mask, the strokes in painting order
      strokes.value = [];
      await add({ id: entryId("g"), image: data.value.image_key, target, strokes: all.map(st => (st.erase ? { r: st.r, pts: st.pts, erase: true } : { r: st.r, pts: st.pts })) });
      return;
    }
    const cutOf = st => cutting() !== !!st.erase;
    const part = cutPart => {
      const k = all.findIndex(st => cutOf(st) === cutPart);
      return k < 0 ? [] : all.slice(k).map(st => (cutOf(st) === cutPart ? { r: st.r, pts: st.pts } : { r: st.r, pts: st.pts, erase: true }));
    };
    const keep = part(false), cut = part(true), image = data.value.image_key;
    strokes.value = [];
    if (keep.length) await add({ id: entryId("b"), image, strokes: keep });
    if (cut.length) await add({ id: entryId("e"), remove: true, image, strokes: cut });
  };
  const addBrush = () => commitStrokes();     // the brush stays on: painting usually goes on
  async function resetEdits(o) {          // ↺: the object as segmented, without the brush edits
    const ids = editsOf(o.id).map(a => a.id);
    state.added = state.added.filter(a => !ids.includes(a.id));
    save();
    ids.forEach(id => options.onRemoveEntry?.(id, { keepSelection: true }));
    busy.value = "Restoring the mask…"; error.value = "";
    try { await editObject(o.id, []); } catch (err) { error.value = String(err.message || err); } finally { busy.value = ""; }
  }
  function removeEntry(entryId) {
    state.added = state.added.filter(a => a.id !== entryId);
    const gone = objects.value.filter(o => o.entry === entryId).map(o => o.id);
    state.selected = state.selected.filter(i => !gone.includes(i));
    if (data.value) data.value = { ...data.value, objects: data.value.objects.filter(o => o.entry !== entryId) };
    gone.forEach(i => loaded.delete(i));
    save(); ready.value++;
    options.onRemoveEntry?.(entryId);
    if (data.value) options.onObjects?.(data.value.objects);
  }

  async function runNode() {
    error.value = "";
    try {
      const p = await app.graphToPrompt();
      await api.queuePrompt(0, p, { partialExecutionTargets: [String(io.node.id)] });
      busy.value = "Segmenting…";
    } catch (err) { error.value = String(err.message || err); }
  }

  io.expose({
    executed: d => {
      busy.value = "";
      if (!d) return;
      if (data.value?.session !== d.session) loaded.clear();
      if (data.value && data.value.image_key !== d.image_key) strokes.value = [];
      // brush entries belong to the image they were painted on: another image drops them (text and points segment again)
      const stale = state.added.filter(a => a.strokes && a.image && a.image !== d.image_key).map(a => a.id);
      if (stale.length) { state.added = state.added.filter(a => !stale.includes(a.id)); save(); }
      data.value = d;
      // ids of objects that no longer exist (another image, other concepts) leave the selection
      const ids = new Set(d.objects.map(o => o.id));
      const keep = state.selected.filter(i => ids.has(i));
      if (keep.length !== state.selected.length) { state.selected = keep; save(); }
      nextTick(() => { relayout(); loadAll(); });
    },
    reload: () => { Object.assign(state, parseState(io.get())); ready.value++; },
    commit: commitStrokes,        // paint strokes not added yet become entries (the video picker calls it before it remounts)
  });

  let ro;
  const done = () => { busy.value = ""; };      // a failed or fully cached run sends no output to this node
  onMounted(() => {
    ro = new ResizeObserver(() => relayout()); ro.observe(stageEl.value);
    api.addEventListener("execution_error", done); api.addEventListener("execution_success", done);
  });
  onBeforeUnmount(() => {
    ro?.disconnect();
    api.removeEventListener("execution_error", done); api.removeEventListener("execution_success", done);
  });
  watch(data, () => nextTick(relayout));

  return () => {
    const d = data.value, objs = objects.value;
    const hovered = objs.find(o => o.id === hover.value);
    return h("div", { class: "bsp", onKeydown: e => e.stopPropagation() }, [
      h("div", { class: "bar" }, [
        h("button", { class: "pri", onClick: runNode, title: "Run this node only (and what feeds it) to segment the image" }, d ? "↻ Segment again" : "▶ Segment"),
        h("button", { class: mode.value === "select" ? "on" : "", onClick: () => setMode("select") }, "👆 Select"),
        h("button", { class: mode.value === "points" ? "on" : "", disabled: !d, onClick: () => setMode("points"),
          title: "Place points on something the automatic pass missed" }, "📍 Points"),
        h("button", { class: mode.value === "brush" ? "on" : "", disabled: !d, onClick: () => setMode("brush"),
          title: "Paint extra area into the mark by hand (e.g. hair SAM left out)" }, "🖌 Brush"),
        showRemove ? h("button", { class: mode.value === "remove" ? "on" : "", disabled: !d, onClick: () => setMode("remove"),
          title: "Place points on something to cut out of the mark (SAM segments it)" }, "➖ Remove") : null,
        showEraser ? h("button", { class: mode.value === "eraser" ? "on" : "", disabled: !d, onClick: () => setMode("eraser"),
          title: "Paint an area to subtract manually from the selected mask" }, "🧽 Eraser") : null,
        h("span", { class: "hint" }, d ? `${objs.length} objects · ${state.selected.length} selected` : ""),
        h("span", { style: "flex:1" }),
        h("button", { disabled: !objs.length, onClick: () => { state.selected = objs.map(o => o.id); save(); } }, "All"),
        h("button", { disabled: !state.selected.length, onClick: () => { state.selected = []; save(); } }, "None"),
      ]),
      h("div", { class: "bar" }, [
        h("input", { type: "text", value: text.value, placeholder: "Find by text (SAM 3 prompt), e.g. red umbrella, the man on the left",
          disabled: !d, onInput: e => { text.value = e.target.value; }, onKeydown: e => { e.stopPropagation(); if (e.key === "Enter") addText(); } }),
        h("button", { disabled: !d || !text.value.trim() || !!busy.value, onClick: addText }, "🔎 Find"),
        pointing() ? [
          h("button", { class: "pri", disabled: !points.value.some(p => p.label) || !!busy.value, onClick: addPoints },
            `${cutting() ? "✓ Remove this" : "✓ Create object"} (${points.value.length})`),
          h("button", { disabled: !points.value.length, onClick: () => { points.value = []; } }, "Clear points"),
        ] : null,
        brushing() ? [
          h("input", { type: "range", min: .003, max: .1, step: .001, value: brush.value, title: "Brush size",
            onInput: e => { brush.value = +e.target.value; options.onBrush?.(brush.value); } }),
          h("button", { class: "pri", disabled: !strokes.value.length || !!busy.value, onClick: addBrush },
            `${cutting() ? "✓ Erase painted area" : "✓ Apply painted area"} (${strokes.value.length})`),
          h("button", { disabled: !strokes.value.length, onClick: () => { strokes.value = strokes.value.slice(0, -1); } }, "Undo"),
          h("button", { disabled: !strokes.value.length, onClick: () => { strokes.value = []; } }, "Clear"),
        ] : null,
      ]),
      h("div", { class: "stage", ref: stageEl, onContextmenu: e => e.preventDefault() }, d ? [
        h("div", { class: ["frame", mode.value !== "select" && mode.value], style: `width:${fit.w}px;height:${fit.h}px` }, [
          h("img", { src: fileUrl(d.image), draggable: false }),
          h("canvas", { ref: canvasEl, onPointerdown: onPointer, onPointermove: onMove, onPointerup: () => { painting = false; },
            onPointerleave: () => { hover.value = null; cursor = null; if (!painting) draw(); } }),
          ...points.value.map((p, k) => h("div", { class: ["pt", p.label ? "pos" : "neg"], style: `left:${p.x * 100}%;top:${p.y * 100}%`,
            title: "click to remove", onPointerdown: e => { e.stopPropagation(); points.value = points.value.filter((_, j) => j !== k); } })),
        ]),
        hovered ? h("div", { class: "tip" }, `${hovered.id} · ${hovered.label}${hovered.score != null ? ` · ${hovered.score}` : ""}`) : null,
        mode.value === "points" ? h("div", { class: "tip" }, "click = keep · right-click or shift+click = exclude · click a point to remove it") : null,
        mode.value === "remove" ? h("div", { class: "tip" }, "click on what to cut out of the mark · right-click or shift+click = keep it in · then ✓ Remove this") : null,
        mode.value === "brush" ? h("div", { class: "tip" }, "paint to grow the selected mask the stroke starts on (or the nearest) · shift = erase from it · then ✓ Apply painted area") : null,
        mode.value === "eraser" ? h("div", { class: "tip" }, "paint what to cut out of the mark · shift = paint back · then ✓ Erase painted area") : null,
      ] : h("div", { class: "empty" }, ["Connect an image and press ▶ Segment (or run the workflow).", h("br"),
        "Then click objects to select them; add what was missed by text or 📍 points."])),
      busy.value ? h("div", { class: "hint" }, busy.value) : null,
      error.value ? h("div", { class: "err" }, error.value) : null,
      objs.length ? h("div", { class: "list" }, objs.map(o => h("div", {
        class: ["chip", isSel(o.id) && "sel", o.remove && "cut"], title: `${o.label}${o.score != null ? ` · score ${o.score}` : ""} · ${(o.area * 100).toFixed(1)}% of the image`,
        onClick: () => toggle(o.id), onPointerenter: () => { hover.value = o.id; }, onPointerleave: () => { hover.value = null; },
      }, [
        h("span", { class: "sw", style: `background:${rgb(colorOf(o.id))}` }),
        `${o.remove ? "➖ " : ""}${o.id} ${o.label}${o.edits ? " ✎" : ""}`,
        o.edits ? h("span", { class: "x", title: "undo the brush edits on this object", onClick: e => { e.stopPropagation(); resetEdits(o); } }, "↺") : null,
        o.entry ? h("span", { class: "x", title: "remove this added object", onClick: e => { e.stopPropagation(); removeEntry(o.entry); } }, "×") : null,
      ]))) : null,
      d ? h("div", { class: "hint" }, "The outputs follow the selection on the next run.") : null,
    ]);
  };
}

app.registerExtension({
  name: "BFSNodes.SegmentPicker",
  async nodeCreated(node) {
    if (node.comfyClass !== "BFSSegmentPicker") return;
    styles();
    const widget = node.widgets?.find(w => w.name === "selection");
    if (widget) {   // the JSON stays a widget (saved with the workflow) but is never shown
      widget.type = "hidden"; widget.hidden = true; widget.computeSize = () => [0, -4];
      widget.options = { ...(widget.options || {}), hidden: true, serialize: true };
      widget.serializeValue = () => widget.value;
    }
    const host = document.createElement("div");
    host.style.cssText = "position:relative;width:100%;height:100%;min-width:420px;min-height:460px";
    // hideInPanel: the side panel's legacy renderer writes its own width into the widget, shrinking the canvas in the node
    node.addDOMWidget("bfs_segment_picker", "div", host, { serialize: false, hideOnZoom: false, hideInPanel: true });
    wheelToCanvas(host);
    const handle = {};
    createApp({
      setup: () => Panel({
        node,
        get: () => widget?.value ?? "{}",
        set: v => { if (widget) widget.value = v; node.graph?.setDirtyCanvas(true); },
        expose: o => Object.assign(handle, o),
      }),
    }).mount(host);
    const onExecuted = node.onExecuted;
    node.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      handle.executed?.(output?.bfs_segpick?.[0]);
    };
    const onConfigure = node.onConfigure;
    node.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      setTimeout(() => handle.reload?.(), 0);
      return r;
    };
    node.size = [Math.max(node.size[0], DEFAULT_W), Math.max(node.size[1], DEFAULT_H)];
  },
});
