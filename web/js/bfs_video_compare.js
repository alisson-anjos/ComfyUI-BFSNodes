/**
 * BFS Video Compare: one player for two videos.
 *
 * Wipe (drag the divider), overlay (drag to fade) or side by side, with any of guide / target / difference on top
 * or below. The strip under the scrubber is the difference per frame, measured by the node on the real frames:
 * click it to jump to a frame. The node sends three temporary MP4s and that list when it runs.
 *
 * ⬇ save plays the video once from the start and downloads it as shown (same layout, divider, labels); ● rec records
 * whatever you do in the player (dragging the divider, pausing, scrubbing) until ■ stop. Both draw the two videos into
 * a canvas and record it in the browser, in real time. The browser's file (WebM, or a fragmented MP4 many apps won't
 * play) is then converted by ComfyUI to the chosen format: MP4 (H.264), a small MP4 (under 10 MB) or a GIF.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { wheelToCanvas } from "./bfs_wheel.js";

const SOURCES = ["guide", "target", "difference"];
const MODES = [["wipe", "wipe ↔"], ["wipe-v", "wipe ↕"], ["overlay", "overlay"], ["side", "side by side"]];
const FORMATS = [["mp4", "MP4"], ["small", "MP4 small (<10 MB)"], ["gif", "GIF"], ["webm", "browser file (WebM)"]];
const FORMAT_KEY = "bfs.videoCompare.format";
const PART = 8 * 1024 * 1024;   // upload pieces, under ComfyUI's request limit

function styles() {
  if (document.getElementById("bfs-vcmp-css")) return;
  const el = document.createElement("style");
  el.id = "bfs-vcmp-css";
  el.textContent = `
.bfsvc{display:flex;flex-direction:column;gap:6px;height:100%;box-sizing:border-box;padding:6px;
  font:12px/1.4 var(--font-family,system-ui,sans-serif);color:var(--descrip-text,#c9c9cf);
  background:var(--comfy-input-bg,#1b1b1f);border-radius:8px}
.bfsvc .stage{position:relative;flex:1 1 0;min-height:160px;background:#0e0e11;border-radius:6px;overflow:hidden;
  cursor:ew-resize;touch-action:none;user-select:none}
.bfsvc .stage.v{cursor:ns-resize}.bfsvc .stage.o{cursor:col-resize}.bfsvc .stage.s{cursor:default}
.bfsvc video{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none}
.bfsvc .stage.s video{width:calc(50% - 2px)}.bfsvc .stage.s video.r{left:auto;right:0}
.bfsvc .refs{flex:0 0 auto;height:84px;display:flex;gap:6px;justify-content:center;overflow-x:auto;padding:4px;
  background:#0e0e11;border-radius:6px}
.bfsvc .refs figure{margin:0;height:100%;position:relative;flex:none}
.bfsvc .refs img{height:100%;border-radius:4px;box-shadow:0 0 0 1px #000a;background:#000;display:block}
.bfsvc .refs figcaption{position:absolute;left:3px;top:2px;font-size:10px;padding:0 4px;border-radius:3px;background:#000a;color:#fff}
.bfsvc .line{position:absolute;background:#fff;box-shadow:0 0 0 1px #0008;pointer-events:none}
.bfsvc .tag{position:absolute;top:6px;padding:1px 7px;border-radius:4px;background:#000a;color:#fff;font-size:11px;
  pointer-events:none}
.bfsvc .row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.bfsvc select,.bfsvc button{background:#131317;border:1px solid #34343d;color:#e6e6ea;border-radius:4px;
  padding:2px 6px;font:inherit;cursor:pointer}
.bfsvc button:hover,.bfsvc select:hover{border-color:#3a86ff}
.bfsvc button.rec{border-color:#e0573a;color:#ff7a5c}
.bfsvc button:disabled{opacity:.45;cursor:default}
.bfsvc label{display:flex;gap:3px;align-items:center;cursor:pointer}
.bfsvc input[type=range]{flex:1;accent-color:#3a86ff;min-width:80px}
.bfsvc canvas{width:100%;height:26px;cursor:pointer;border-radius:3px;background:#131317;display:block}
.bfsvc .info{font-family:ui-monospace,monospace;font-size:11px;white-space:nowrap}
.bfsvc .empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#8b8b93;
  text-align:center;padding:12px}
`;
  document.head.appendChild(el);
}

const url = f => api.apiURL(`/view?filename=${encodeURIComponent(f.filename)}&subfolder=${encodeURIComponent(f.subfolder || "")}`
  + `&type=${f.type || "temp"}&t=${Date.now()}`);

function el(tag, attrs = {}, kids = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") e.style.cssText = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of [].concat(kids)) e.append(c);
  return e;
}

function buildPlayer(host) {
  const st = { data: null, top: "target", bottom: "guide", mode: "wipe", pos: 0.5, playing: false, muted: true, frame: 0,
    labels: true, refs: true };
  const vTop = el("video", { playsinline: "", preload: "auto", loop: "" });
  const vBot = el("video", { playsinline: "", preload: "auto", loop: "" });
  vTop.muted = vBot.muted = true;
  const line = el("div", { class: "line" });
  const tagTop = el("div", { class: "tag" }), tagBot = el("div", { class: "tag" });
  const empty = el("div", { class: "empty" }, "Run the node to load the guide and the target.");
  const refBox = el("div", { class: "refs", style: "display:none" });
  const stage = el("div", { class: "stage", tabindex: "0" }, [vBot, vTop, line, tagTop, tagBot, empty]);
  const sel = (opts, val, on) => {
    const s = el("select", {}, opts.map(([v, l]) => { const o = el("option", { value: v }, l); if (v === val) o.selected = true; return o; }));
    s.addEventListener("change", () => on(s.value));
    return s;
  };
  const srcOpts = () => SOURCES.map(s => [s, s === "guide" ? (st.data?.labels?.[0] || "guide") : s === "target" ? (st.data?.labels?.[1] || "target") : "difference"]);
  let selTop, selBot;
  const selMode = sel(MODES, st.mode, v => { st.mode = v; layout(); });
  const refreshSelects = () => {
    const t = sel(srcOpts(), st.top, v => { st.top = v; load(); }), b = sel(srcOpts(), st.bottom, v => { st.bottom = v; load(); });
    selTop ? selTop.replaceWith(t) : null; selBot ? selBot.replaceWith(b) : null;
    selTop = t; selBot = b;
  };
  refreshSelects();
  const btnSwap = el("button", { title: "Swap top and bottom", onclick: () => { [st.top, st.bottom] = [st.bottom, st.top]; refreshSelects(); load(); } }, "⇅ swap");
  const btnPlay = el("button", { title: "Play / pause (space)", onclick: () => toggle() }, "▶");
  const btnPrev = el("button", { title: "Previous frame (←)", onclick: () => step(-1) }, "◀|");
  const btnNext = el("button", { title: "Next frame (→)", onclick: () => step(1) }, "|▶");
  const btnMute = el("button", { title: "Sound (from the target)", onclick: () => { st.muted = !st.muted; sound(); } }, "🔇");
  const btnFull = el("button", { title: "Full screen", onclick: () => (document.fullscreenElement ? document.exitFullscreen() : host.requestFullscreen?.())?.catch?.(() => {}) }, "⛶");
  const btnSave = el("button", { title: "Play the video once from the start and download it as shown", onclick: () => saveFull() }, "⬇ save");
  let format = "mp4";
  try { format = localStorage.getItem(FORMAT_KEY) || format; } catch { }
  if (!FORMATS.some(([v]) => v === format)) format = "mp4";
  const selFormat = sel(FORMATS, format, v => { format = v; try { localStorage.setItem(FORMAT_KEY, v); } catch { } });
  selFormat.title = "Format of what save / rec download. MP4: H.264 + AAC, plays and previews anywhere. MP4 small: "
    + "the same, sized to stay under 10 MB. GIF: no sound, under 10 MB. browser file: the recording as it is, no conversion.";
  const btnRec = el("button", { title: "Record what you do in the player until stop, then download it", onclick: () => (rec.mr ? stopRec() : startRec()) }, "● rec");
  const chkLabels = el("input", { type: "checkbox" });
  chkLabels.checked = true;
  chkLabels.addEventListener("change", () => { st.labels = chkLabels.checked; });
  const chkRefs = el("input", { type: "checkbox" });
  chkRefs.checked = true;
  const showRefs = () => { refBox.style.display = st.refs && refBox.childElementCount ? "flex" : "none"; };
  chkRefs.addEventListener("change", () => { st.refs = chkRefs.checked; showRefs(); });
  const lblRefs = el("label", { title: "Show the reference pictures in a strip below the videos (also in what save / rec download)", style: "display:none" }, [chkRefs, "refs"]);
  const scrub = el("input", { type: "range", min: "0", max: "0", value: "0", step: "1" });
  scrub.addEventListener("input", () => seek(+scrub.value));
  const info = el("span", { class: "info" }, "");
  const strip = el("canvas", { title: "Difference per frame: click to jump" });
  strip.addEventListener("pointerdown", e => {
    const r = strip.getBoundingClientRect(), n = st.data?.frames || 0;
    if (n) seek(Math.min(n - 1, Math.max(0, Math.floor((e.clientX - r.left) / r.width * n))));
  });
  const notes = el("div", {style:"white-space:pre-wrap;font-size:12px;padding:6px;color:#e6b566"});
  host.append(el("div", { class: "bfsvc" }, [
    el("div", { class: "row" }, [el("span", {}, "top"), selTop, btnSwap, el("span", {}, "bottom"), selBot, selMode, btnFull]),
    stage,
    refBox,
    el("div", { class: "row" }, [btnPrev, btnPlay, btnNext, scrub, btnMute, info]),
    el("div", { class: "row" }, [btnSave, btnRec, selFormat, el("label", { title: "Draw the names on the saved / recorded video" }, [chkLabels, "labels"]), lblRefs]),
    strip,
    notes,
  ]));

  const fps = () => st.data?.fps || 24;
  const frameOf = t => Math.min((st.data?.frames || 1) - 1, Math.max(0, Math.floor(t * fps() + 1e-3)));
  const timeOf = f => (f + 0.5) / fps();
  const videos = () => [vBot, vTop];

  function sound() {
    btnMute.textContent = st.muted ? "🔇" : "🔊";
    for (const v of [vTop, vBot]) v.muted = true;
    if (!st.muted) { const v = st.bottom === "target" ? vBot : st.top === "target" ? vTop : null; if (v) v.muted = false; }
  }

  function layout() {
    const m = st.mode, p = st.pos;
    stage.className = "stage" + (m === "wipe-v" ? " v" : m === "overlay" ? " o" : m === "side" ? " s" : "");
    vTop.style.clipPath = m === "wipe" ? `inset(0 ${(1 - p) * 100}% 0 0)` : m === "wipe-v" ? `inset(0 0 ${(1 - p) * 100}% 0)` : "none";
    vTop.style.opacity = m === "overlay" ? String(p) : "1";
    line.style.display = m === "wipe" || m === "wipe-v" ? "block" : "none";
    if (m === "wipe") line.style.cssText = `display:block;left:calc(${p * 100}% - 1px);top:0;bottom:0;width:2px`;
    if (m === "wipe-v") line.style.cssText = `display:block;top:calc(${p * 100}% - 1px);left:0;right:0;height:2px`;
    stage.prepend(vTop); stage.prepend(vBot);   // under the divider and tags; side by side: bottom left, top right
    vTop.classList.toggle("r", m === "side");
    const name = s => s === "guide" ? (st.data?.labels?.[0] || "guide") : s === "target" ? (st.data?.labels?.[1] || "target") : "difference";
    tagTop.textContent = name(st.top) + (m === "overlay" ? ` ${Math.round(p * 100)}%` : "");
    tagBot.textContent = name(st.bottom);
    tagTop.style.cssText = m === "side" ? "right:6px" : "left:6px";
    tagBot.style.cssText = m === "side" ? "left:6px" : m === "wipe-v" ? "left:6px;top:auto;bottom:6px" : "right:6px";
    if (m === "overlay") tagBot.style.cssText = "right:6px";
  }

  function drag(e) {
    if (st.mode === "side" || !st.data) return;
    const r = stage.getBoundingClientRect();
    st.pos = Math.min(1, Math.max(0, st.mode === "wipe-v" ? (e.clientY - r.top) / r.height : (e.clientX - r.left) / r.width));
    layout();
  }
  stage.addEventListener("pointerdown", e => { stage.setPointerCapture(e.pointerId); drag(e); });
  stage.addEventListener("pointermove", e => { if (stage.hasPointerCapture(e.pointerId)) drag(e); });
  stage.addEventListener("keydown", e => {
    if (e.key === " ") { toggle(); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { step(1); e.preventDefault(); }
  });

  function drawStrip() {
    const d = st.data, ctx = strip.getContext("2d");
    const w = strip.width = Math.max(50, strip.clientWidth * devicePixelRatio), h = strip.height = 26 * devicePixelRatio;
    ctx.clearRect(0, 0, w, h);
    if (!d?.mae?.length) return;
    // from the least to the most different frame, so the changes along the video stand out
    const mx = Math.max(...d.mae), mn = Math.min(...d.mae), span = Math.max(mx - mn, 1e-6), n = d.mae.length;
    ctx.fillStyle = "#e0573a";
    d.mae.forEach((v, i) => { const bh = 2 + (v - mn) / span * (h - 4); ctx.fillRect(i / n * w, h - bh, Math.max(1, w / n), bh); });
    ctx.fillStyle = "#3a86ff";
    ctx.fillRect(st.frame / n * w, 0, Math.max(2, w / n), h);
  }

  function show(f) {
    st.frame = f;
    scrub.value = String(f);
    const d = st.data;
    const t = (f / fps());
    info.textContent = d ? `frame ${f}/${d.frames - 1}  ${t.toFixed(2)}s  diff ${(d.mae?.[f] ?? 0).toFixed(1)}/255` : "";
    drawStrip();
  }

  function seek(f) {
    if (!st.data) return;
    if (st.playing) toggle();
    for (const v of videos()) v.currentTime = timeOf(f);
    show(f);
  }
  const step = k => seek(Math.min(st.data ? st.data.frames - 1 : 0, Math.max(0, st.frame + k)));

  function toggle() {
    if (!st.data) return;
    st.playing = !st.playing;
    btnPlay.textContent = st.playing ? "⏸" : "▶";
    if (st.playing) { vTop.currentTime = vBot.currentTime; for (const v of videos()) v.play().catch(() => {}); tick(); }
    else for (const v of videos()) { v.pause(); v.playbackRate = 1; }
  }

  function tick() {   // the bottom video is the clock; the top one follows it within half a frame
    if (!st.playing) return;
    // small drift: nudge the top one's speed (seeking it every frame kept it seeking, so it stuttered and drew blank)
    const drift = vTop.currentTime - vBot.currentTime;
    if (!vTop.seeking) {
      if (Math.abs(drift) > 0.25) { vTop.currentTime = vBot.currentTime; vTop.playbackRate = 1; }
      else vTop.playbackRate = Math.abs(drift) > 0.5 / fps() ? Math.min(1.05, Math.max(0.95, 1 - 2 * drift)) : 1;
    }
    show(frameOf(vBot.currentTime));
    requestAnimationFrame(tick);
  }

  function load() {
    const d = st.data;
    if (!d) return;
    const f = st.frame, was = st.playing;
    if (was) toggle();
    let pending = 2;
    const ready = () => { if (--pending === 0) { for (const v of videos()) v.currentTime = timeOf(f); show(f); sound(); if (was) toggle(); } };
    for (const [v, s] of [[vTop, st.top], [vBot, st.bottom]]) {
      v.addEventListener("loadeddata", ready, { once: true });
      v.src = url(d[s]);
      v.load();
    }
    layout();
  }

  // --- save / record: the two videos drawn into a canvas the way the stage shows them, encoded by MediaRecorder
  const rec = { mr: null, chunks: [], raf: 0, t0: 0, full: false, onEnd: null, type: "", busy: "" };

  function pickType(audio) {   // name an audio codec only with a sound track: Firefox stalls on one it never gets
    const list = audio ? ["video/mp4;codecs=avc1.640028,mp4a.40.2", "video/mp4", "video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"]
      : ["video/mp4;codecs=avc1.640028", "video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
    return list.find(t => window.MediaRecorder?.isTypeSupported?.(t)) || "";
  }

  function fit(aw, ah, x, y, w, h) {   // the largest aw:ah box centred in (x, y, w, h)
    const k = Math.min(w / aw, h / ah), bw = aw * k, bh = ah * k;
    return [x + (w - bw) / 2, y + (h - bh) / 2, bw, bh];
  }

  function label(ctx, text, x, y, size, right = false, bottom = false) {
    ctx.font = `${size}px system-ui,sans-serif`;
    const pad = size * 0.45, w = ctx.measureText(text).width + 2 * pad, h = size * 1.5;
    const lx = right ? x - w : x, ly = bottom ? y - h : y;
    ctx.fillStyle = "rgba(0,0,0,.65)"; ctx.fillRect(lx, ly, w, h);
    ctx.fillStyle = "#fff"; ctx.textBaseline = "middle"; ctx.fillText(text, lx + pad, ly + h / 2);
  }

  const last = new Map();   // video -> its last whole frame, drawn while it seeks (a seeking video draws blank)
  function pic(v) {
    if (v.readyState < 2 || v.seeking) return last.get(v) || v;
    let c = last.get(v);
    if (!c) last.set(v, c = el("canvas"));
    if (c.width !== v.videoWidth || c.height !== v.videoHeight) { c.width = v.videoWidth; c.height = v.videoHeight; }
    c.getContext("2d").drawImage(v, 0, 0);
    return c;
  }

  const refImages = () => (st.refs ? [...refBox.querySelectorAll("img")].filter(i => i.complete && i.naturalWidth) : []);
  // save / rec: the pictures in a strip below the videos, as in the player (22% of the video's height)
  const stripHeight = h => (refImages().length ? Math.round(h * 0.22) & ~1 : 0);
  function drawRefs(ctx, top) {
    const ims = refImages();
    if (!ims.length) return;
    const cw = ctx.canvas.width, sh = ctx.canvas.height - top, gap = Math.max(2, Math.round(sh * 0.05));
    let h = sh - 2 * gap;
    const widths = () => ims.map(im => Math.round(im.naturalWidth * h / im.naturalHeight));
    let total = widths().reduce((a, b) => a + b, 0) + gap * (ims.length - 1);
    if (total > cw - 2 * gap) { h = Math.floor(h * (cw - 2 * gap) / total); total = widths().reduce((a, b) => a + b, 0) + gap * (ims.length - 1); }
    ctx.fillStyle = "#0e0e11"; ctx.fillRect(0, top, cw, sh);
    let x = Math.round((cw - total) / 2);
    const y = top + Math.round((sh - h) / 2), fs = Math.max(10, Math.round(h * 0.12));
    ims.forEach((im, k) => {
      const w = Math.round(im.naturalWidth * h / im.naturalHeight);
      ctx.drawImage(im, x, y, w, h);
      if (st.labels) label(ctx, `Picture ${k + 1}`, x + 3, y + 3, fs);
      x += w + gap;
    });
  }

  function compose(ctx) {
    const sh = ctx.canvas.height - (ctx.canvas.videoHeight || ctx.canvas.height);
    drawVideos(ctx, ctx.canvas.height - sh);
    drawRefs(ctx, ctx.canvas.height - sh);
  }

  function drawVideos(ctx, area) {
    const cw = ctx.canvas.width, ch = area ?? ctx.canvas.height, m = st.mode, p = st.pos;
    const B = pic(vBot), T = pic(vTop);
    const aw = vBot.videoWidth || st.data.width, ah = vBot.videoHeight || st.data.height;
    const name = s => s === "guide" ? (st.data?.labels?.[0] || "guide") : s === "target" ? (st.data?.labels?.[1] || "target") : "difference";
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cw, ch);
    if (m === "side") {   // bottom left, top right, as on the stage
      const gap = Math.round(cw * 0.004), half = (cw - gap) / 2;
      const a = fit(aw, ah, 0, 0, half, ch), b = fit(aw, ah, half + gap, 0, half, ch);
      ctx.drawImage(B, ...a); ctx.drawImage(T, ...b);
      if (st.labels) {
        const fs = Math.max(12, Math.round(a[3] * 0.032)), o = fs * 0.5;
        label(ctx, name(st.bottom), a[0] + o, a[1] + o, fs); label(ctx, name(st.top), b[0] + o, b[1] + o, fs);
      }
      return;
    }
    const [x, y, w, h] = fit(aw, ah, 0, 0, cw, ch);
    ctx.drawImage(B, x, y, w, h);
    ctx.save();
    if (m === "wipe" || m === "wipe-v") { ctx.beginPath(); m === "wipe" ? ctx.rect(x, y, w * p, h) : ctx.rect(x, y, w, h * p); ctx.clip(); }
    if (m === "overlay") ctx.globalAlpha = p;
    ctx.drawImage(T, x, y, w, h);
    ctx.restore();
    const lw = Math.max(2, Math.round(h / 360));
    ctx.fillStyle = "#fff";
    if (m === "wipe") ctx.fillRect(x + w * p - lw / 2, y, lw, h);
    if (m === "wipe-v") ctx.fillRect(x, y + h * p - lw / 2, w, lw);
    if (st.labels) {
      const fs = Math.max(12, Math.round(h * 0.032)), o = fs * 0.5;
      label(ctx, name(st.top) + (m === "overlay" ? ` ${Math.round(p * 100)}%` : ""), x + o, y + o, fs);
      if (m === "wipe-v") label(ctx, name(st.bottom), x + o, y + h - o, fs, false, true);
      else label(ctx, name(st.bottom), x + w - o, y + o, fs, true);
    }
  }

  function stem() {
    const d = st.data, clean = s => String(s || "").replace(/[^\w.-]+/g, "_").slice(0, 40);
    const stamp = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
    return `bfs_compare_${clean(d?.labels?.[0])}_vs_${clean(d?.labels?.[1])}_${st.mode}_${stamp}`;
  }

  function download(href, name) {
    const a = el("a", { href, download: name });
    document.body.append(a); a.click(); a.remove();
  }

  async function exportBlob(blob, ext, seconds) {   // the browser's file -> ComfyUI converts it -> download
    const name = stem();
    if (format === "webm") {
      const href = URL.createObjectURL(blob);
      download(href, `${name}.${ext}`);
      setTimeout(() => URL.revokeObjectURL(href), 60000);
      return;
    }
    const id = [...crypto.getRandomValues(new Uint8Array(8))].map(b => b.toString(16).padStart(2, "0")).join("");
    rec.busy = "uploading"; buttons();
    try {
      for (let off = 0; off < blob.size; off += PART) {
        const r = await api.fetchApi(`/bfs/compare/part?id=${id}&offset=${off}`, { method: "POST", body: blob.slice(off, off + PART) });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `upload failed (${r.status})`);
      }
      rec.busy = "converting"; buttons();
      const r = await api.fetchApi("/bfs/compare/export", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, format, seconds, name }) });
      const f = await r.json();
      if (!r.ok || f.error) throw new Error(f.error || `conversion failed (${r.status})`);
      download(url(f), f.filename);
    } catch (e) {
      alert(`BFS Video Compare: could not convert the recording (${e.message}). Choose "browser file" to download it as recorded.`);
    } finally {
      rec.busy = ""; buttons();
    }
  }

  function buttons() {
    if (rec.busy) {   // uploading / converting the finished recording
      btnRec.classList.remove("rec");
      btnRec.disabled = btnSave.disabled = selFormat.disabled = true;
      btnSave.textContent = rec.busy === "uploading" ? "⬆ uploading…" : "⚙ converting…";
      btnRec.textContent = "● rec";
      return;
    }
    btnRec.disabled = selFormat.disabled = false;
    const busy = !!rec.mr;
    btnRec.classList.toggle("rec", busy);
    btnRec.textContent = busy ? (rec.full ? "■ stop" : `■ stop ${((performance.now() - rec.t0) / 1000).toFixed(0)}s`) : "● rec";
    btnSave.disabled = busy;
    btnSave.textContent = busy && rec.full ? `⬇ ${Math.round(100 * (vBot.currentTime / (vBot.duration || 1)))}%` : "⬇ save";
  }

  function startRec(full = false) {
    if (!st.data || rec.mr || rec.busy) return false;
    if (!window.MediaRecorder || !HTMLCanvasElement.prototype.captureStream) { alert("This browser can't record a canvas (MediaRecorder)."); return false; }
    const aw = vBot.videoWidth || st.data.width, ah = vBot.videoHeight || st.data.height;
    const side = st.mode === "side", canvas = el("canvas");
    // the size of the layout it starts in (side by side is twice as wide); a later mode change is fitted into it
    canvas.width = (side ? 2 * aw + Math.round(2 * aw * 0.004) : aw) & ~1;
    canvas.videoHeight = ah & ~1;                       // the videos; the reference strip goes below them
    canvas.height = canvas.videoHeight + stripHeight(canvas.videoHeight);
    const ctx = canvas.getContext("2d");
    compose(ctx);
    const stream = canvas.captureStream(Math.max(30, Math.round(fps())));
    // the soundtrack is on the target. Not Firefox's mozCaptureStream: it takes the element's picture away from the canvas
    const vt = st.top === "target" ? vTop : st.bottom === "target" ? vBot : null;
    try { for (const t of vt?.captureStream?.()?.getAudioTracks?.() || []) stream.addTrack(t); } catch { }
    const type = pickType(stream.getAudioTracks().length > 0);
    if (!type) { for (const t of stream.getTracks()) t.stop(); alert("This browser can't record MP4 or WebM."); return false; }
    rec.type = type; rec.chunks = []; rec.full = full; rec.t0 = performance.now();
    rec.mr = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 16e6 });
    rec.mr.ondataavailable = e => { if (e.data?.size) rec.chunks.push(e.data); };
    rec.mr.onstop = () => {
      for (const t of stream.getTracks()) t.stop();
      const chunks = rec.chunks, seconds = (performance.now() - rec.t0) / 1000;
      rec.mr = null; rec.chunks = []; buttons();
      if (chunks.length) exportBlob(new Blob(chunks, { type: rec.type.split(";")[0] }), rec.type.includes("mp4") ? "mp4" : "webm", seconds);
    };
    rec.mr.start(1000);
    const frame = () => { if (!rec.mr) return; compose(ctx); buttons(); rec.raf = requestAnimationFrame(frame); };
    rec.raf = requestAnimationFrame(frame);
    buttons();
    return true;
  }

  function stopRec() {
    if (!rec.mr) return;
    cancelAnimationFrame(rec.raf);
    if (rec.onEnd) { vBot.removeEventListener("ended", rec.onEnd); rec.onEnd = null; }
    if (rec.full) { for (const v of videos()) v.loop = true; if (st.playing) toggle(); }
    if (rec.mr.state !== "inactive") rec.mr.stop();
  }

  async function saveFull() {   // once from the first frame to the last, then stop and download
    if (!st.data || rec.mr || rec.busy) return;
    if (st.playing) toggle();
    for (const v of videos()) v.loop = false;
    const seeked = v => new Promise(res => { const t = setTimeout(res, 2000); v.addEventListener("seeked", () => { clearTimeout(t); res(); }, { once: true }); });
    const wait = Promise.all(videos().map(seeked));
    seek(0);
    await wait;
    if (!startRec(true)) { for (const v of videos()) v.loop = true; return; }
    rec.onEnd = () => { rec.onEnd = null; stopRec(); };
    vBot.addEventListener("ended", rec.onEnd, { once: true });
    toggle();
  }

  return {
    setData(d) {
      stopRec();
      last.clear();
      refBox.replaceChildren(...(d.refs || []).map((f, k) => el("figure", { title: `Picture ${k + 1}` },
        [el("img", { src: url(f), crossorigin: "anonymous" }), el("figcaption", {}, `Picture ${k + 1}`)])));
      lblRefs.style.display = d.refs?.length ? "flex" : "none";
      showRefs();
      st.data = d;
      notes.textContent = (d.notes || []).join("\n");
      empty.style.display = "none";
      scrub.max = String(Math.max(0, d.frames - 1));
      refreshSelects();
      st.frame = Math.min(st.frame, d.frames - 1);
      load();
    },
    layout, drawStrip,
  };
}

app.registerExtension({
  name: "BFSNodes.VideoCompare",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "BFSVideoCompare") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      styles();
      const host = el("div", { style: "width:100%;height:100%;min-height:320px" });
      this._bfsCompare = buildPlayer(host);
      wheelToCanvas(host);   // the wheel zooms the canvas over the player too
      this.addDOMWidget("bfs_video_compare", "div", host, { serialize: false, hideOnZoom: false, getMinHeight: () => 320 });
      this.setSize([Math.max(this.size[0], 560), Math.max(this.size[1], 620)]);
      new ResizeObserver(() => this._bfsCompare?.drawStrip()).observe(host);
      return r;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      executed?.apply(this, arguments);
      const d = message?.bfs_compare?.[0];
      if (d) this._bfsCompare?.setData(d);
    };
  },
});
