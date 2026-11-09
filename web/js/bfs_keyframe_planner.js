/**
 * BFS Keyframe Planner: the panel of the node. A small player of the guide (frames from the node's last run), a timeline
 * with the keyframes on it, one card per keyframe (source frame + an optional edited frame, dropped or uploaded) and up
 * to 4 reference pictures.
 *
 * All state lives in the hidden `plan` STRING widget (JSON: keyframe frames, edited frame and picture paths in input/),
 * so a workflow saves, reloads and shares the plan. Uploaded images go to input/bfs_keyframes/.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { wheelToCanvas } from "./bfs_wheel.js";
import { createApp, h, ref, reactive, computed, watch, onBeforeUnmount } from "./vendor/vue.esm-browser.prod.mjs";

const DEFAULT_W = 560, DEFAULT_H = 640, MAX_PICTURES = 4, SUBFOLDER = "bfs_keyframes";

const tempUrl = (d, f) => api.apiURL(`/view?filename=${encodeURIComponent(f)}&subfolder=${encodeURIComponent(d.subfolder)}&type=${d.type || "temp"}`);
const inputUrl = p => {
  const i = p.lastIndexOf("/");
  return api.apiURL(`/view?filename=${encodeURIComponent(p.slice(i + 1))}&subfolder=${encodeURIComponent(i < 0 ? "" : p.slice(0, i))}&type=input`);
};

function parsePlan(text) {
  let s = {};
  try { s = JSON.parse(text || "{}") || {}; } catch { s = {}; }
  const keyframes = (Array.isArray(s.keyframes) ? s.keyframes : []).filter(k => Number.isFinite(+k.frame))
    .map(k => ({ frame: Math.round(+k.frame), edit: k.edit || null }));
  const pictures = (Array.isArray(s.pictures) ? s.pictures : []).slice(0, MAX_PICTURES).map(p => p || null);
  while (pictures.length < MAX_PICTURES) pictures.push(null);
  return { keyframes, pictures };
}

async function upload(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("subfolder", SUBFOLDER);
  body.append("type", "input");
  const r = await api.fetchApi("/upload/image", { method: "POST", body });
  if (r.status !== 200) throw new Error(`upload failed (${r.status})`);
  const j = await r.json();
  return j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
}

function pickFile() {
  return new Promise(resolve => {
    const inp = document.createElement("input");
    inp.type = "file"; inp.accept = "image/*";
    inp.onchange = () => resolve(inp.files?.[0] || null);
    inp.click();
  });
}

let styled = false;
function styles() {
  if (styled) return;
  styled = true;
  const css = document.createElement("style");
  css.textContent = `
.bkp{position:absolute;inset:0;display:flex;flex-direction:column;gap:6px;padding:6px;box-sizing:border-box;font:12px system-ui,sans-serif;color:#ddd}
.bkp button{background:#333;color:#ddd;border:1px solid #555;border-radius:4px;padding:3px 8px;cursor:pointer;font:inherit;line-height:1.2}
.bkp button:hover:not(:disabled){background:#444}
.bkp button:disabled{opacity:.4;cursor:default}
.bkp button.pri{background:#2d5a8a;border-color:#3d6fa5;color:#fff}
.bkp button.danger:hover:not(:disabled){background:#6a2a2a}
.bkp .stage{flex:1;min-height:150px;position:relative;display:flex;align-items:center;justify-content:center;background:#121212;border-radius:6px;overflow:hidden}
.bkp .stage img{max-width:100%;max-height:100%;object-fit:contain;user-select:none;-webkit-user-drag:none}
.bkp .stage.onkf{box-shadow:inset 0 0 0 2px #e0a030}
.bkp .badge{position:absolute;left:6px;top:6px;background:rgba(0,0,0,.7);padding:2px 7px;border-radius:10px;font-variant-numeric:tabular-nums}
.bkp .badge.kf{left:auto;right:6px;background:#e0a030;color:#111;font-weight:600}
.bkp .empty{color:#888;text-align:center;padding:16px;line-height:1.7}
.bkp .ctl{display:flex;gap:4px;align-items:center}
.bkp .ctl .sp{flex:1}
.bkp .time{font-variant-numeric:tabular-nums;color:#bbb;min-width:110px;text-align:center}
.bkp .tl{position:relative;height:26px;background:#1d1d1d;border:1px solid #333;border-radius:5px;cursor:pointer;touch-action:none}
.bkp .tl .fill{position:absolute;left:0;top:0;bottom:0;background:#2a3a4d;border-radius:5px 0 0 5px;pointer-events:none}
.bkp .tl .head{position:absolute;top:-2px;bottom:-2px;width:2px;margin-left:-1px;background:#fff;pointer-events:none}
.bkp .tl .mk{position:absolute;top:3px;width:10px;height:18px;margin-left:-5px;border-radius:3px;background:#e0a030;border:1px solid #111;cursor:pointer}
.bkp .tl .mk.done{background:#3fbf6a}
.bkp .tl .mk.cur{outline:2px solid #fff}
.bkp .sec{display:flex;align-items:center;gap:6px;color:#999;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.bkp .sec .n{color:#ccc}
.bkp .row{display:flex;gap:6px;overflow-x:auto;padding-bottom:3px;min-height:96px}
.bkp .card{flex:0 0 auto;display:flex;flex-direction:column;gap:3px;background:#1d1d1d;border:1px solid #383838;border-radius:6px;padding:4px}
.bkp .card.cur{border-color:#e0a030}
.bkp .card .pair{display:flex;gap:3px}
.bkp .thumb{width:72px;height:54px;border-radius:4px;background:#111 center/cover no-repeat;position:relative;cursor:pointer;flex:0 0 auto}
.bkp .thumb .lbl{position:absolute;left:2px;bottom:2px;font-size:9px;background:rgba(0,0,0,.7);padding:0 3px;border-radius:2px}
.bkp .drop{white-space:pre-line;width:72px;height:54px;border:1px dashed #555;border-radius:4px;display:flex;align-items:center;justify-content:center;text-align:center;color:#888;font-size:10px;cursor:pointer;line-height:1.2;flex:0 0 auto}
.bkp .drop:hover,.bkp .drop.over{border-color:#3fbf6a;color:#cfc}
.bkp .card .foot{display:flex;align-items:center;gap:4px;font-size:11px;font-variant-numeric:tabular-nums}
.bkp .card .foot .sp{flex:1}
.bkp .x{color:#999;cursor:pointer;padding:0 3px;border-radius:3px}.bkp .x:hover{color:#f66;background:#2a1a1a}
.bkp .pics{display:flex;gap:6px}
.bkp .pic{position:relative}
.bkp .pic .thumb{width:60px;height:60px}
.bkp .pic .drop{width:60px;height:60px}
.bkp .pic .x{position:absolute;right:1px;top:1px;background:rgba(0,0,0,.7)}
.bkp .pic .tag{font-size:10px;color:#888;text-align:center}
.bkp .hint{color:#777;font-size:11px}.bkp .err{color:#f77;font-size:11px}
`;
  document.head.appendChild(css);
}

function Panel(io) {
  const plan = reactive(parsePlan(io.get()));
  const strip = ref(null);              // {subfolder, type, files, frames, count, fps, width, height} from the last run
  const pos = ref(0);                   // index into strip.frames
  const playing = ref(false);
  const err = ref("");
  const over = ref("");                 // drop target being hovered
  let timer = null;

  const save = () => io.set(JSON.stringify({
    keyframes: [...plan.keyframes].sort((a, b) => a.frame - b.frame),
    pictures: plan.pictures.map(p => p || null),
  }));
  watch(plan, save, { deep: true });

  const count = computed(() => strip.value?.count || 0);
  const fps = computed(() => strip.value?.fps || io.fps() || 24);
  const frame = computed(() => strip.value ? strip.value.frames[pos.value] : 0);
  const sorted = computed(() => [...plan.keyframes].sort((a, b) => a.frame - b.frame));
  const curKf = computed(() => plan.keyframes.find(k => k.frame === frame.value) || null);
  const nearest = f => {                // strip index of the frame nearest to f
    const fr = strip.value?.frames || [];
    let best = 0;
    for (let i = 1; i < fr.length; i++) if (Math.abs(fr[i] - f) < Math.abs(fr[best] - f)) best = i;
    return best;
  };
  const frameUrl = f => strip.value ? tempUrl(strip.value, strip.value.files[nearest(f)]) : "";
  const fmt = f => `${(f / fps.value).toFixed(2)}s`;

  const stop = () => { playing.value = false; if (timer) { clearInterval(timer); timer = null; } };
  const play = () => {
    if (!strip.value) return;
    if (playing.value) return stop();
    playing.value = true;
    const fr = strip.value.frames;
    const step = fr.length > 1 ? (fr[fr.length - 1] - fr[0]) / (fr.length - 1) : 1;
    timer = setInterval(() => { pos.value = (pos.value + 1) % fr.length; }, 1000 * step / fps.value);
  };
  const seekIdx = i => { if (strip.value) pos.value = Math.max(0, Math.min(strip.value.frames.length - 1, i)); };
  const seekFrame = f => seekIdx(nearest(f));
  onBeforeUnmount(stop);

  const addHere = () => {
    if (!strip.value || curKf.value) return;
    plan.keyframes.push({ frame: frame.value, edit: null });
  };
  const removeKf = k => { plan.keyframes.splice(plan.keyframes.indexOf(k), 1); };
  const jumpKf = d => {
    const list = sorted.value;
    if (!list.length) return;
    const f = frame.value;
    const t = d > 0 ? list.find(k => k.frame > f) || list[0] : [...list].reverse().find(k => k.frame < f) || list[list.length - 1];
    seekFrame(t.frame);
  };

  const withFile = async (file, apply) => {
    if (!file) return;
    err.value = "";
    try { apply(await upload(file)); } catch (e) { err.value = String(e.message || e); }
  };
  const dropProps = (key, apply) => ({
    class: ["drop", over.value === key ? "over" : ""],
    onDragover: e => { e.preventDefault(); e.stopPropagation(); over.value = key; },
    onDragleave: () => { if (over.value === key) over.value = ""; },
    onDrop: e => { e.preventDefault(); e.stopPropagation(); over.value = ""; withFile(e.dataTransfer?.files?.[0], apply); },
    onClick: async e => { e.stopPropagation(); withFile(await pickFile(), apply); },
  });

  // timeline: click / drag to seek
  const tlDown = e => {
    if (!strip.value) return;
    const el = e.currentTarget, r = el.getBoundingClientRect();
    const go = ev => {
      const x = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
      seekFrame(Math.round(x * Math.max(0, count.value - 1)));
    };
    stop(); go(e);
    el.setPointerCapture?.(e.pointerId);
    const move = ev => go(ev);
    const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); };
    el.addEventListener("pointermove", move); el.addEventListener("pointerup", up);
  };
  const pct = f => `${count.value > 1 ? (100 * f / (count.value - 1)) : 0}%`;

  io.expose({
    executed: d => {
      if (!d) return;
      const keep = frame.value;
      strip.value = d;
      seekFrame(keep);
      for (const f of d.files) { const im = new Image(); im.src = tempUrl(d, f); }   // preload for smooth playback
    },
    reload: () => { Object.assign(plan, parsePlan(io.get())); },
  });

  return () => {
    const s = strip.value;
    const cards = sorted.value.map((k, i) => h("div", { class: ["card", k.frame === frame.value ? "cur" : ""], key: k.frame }, [
      h("div", { class: "pair" }, [
        h("div", { class: "thumb", title: "source frame: click to go there", style: s ? { backgroundImage: `url("${frameUrl(k.frame)}")` } : {},
          onClick: () => { stop(); seekFrame(k.frame); } }, [h("span", { class: "lbl" }, "source")]),
        k.edit
          ? h("div", { class: "thumb", title: "edited frame: click to replace", style: { backgroundImage: `url("${inputUrl(k.edit)}")` },
              onClick: async () => withFile(await pickFile(), p => { k.edit = p; }) }, [h("span", { class: "lbl" }, "edited")])
          : h("div", { ...dropProps(`kf${k.frame}`, p => { k.edit = p; }), title: "drop or click: the edited frame for this time (optional)" },
              "drop edited\nframe"),
      ]),
      h("div", { class: "foot" }, [
        h("span", null, `#${i + 1} · ${fmt(k.frame)}`), h("span", { class: "sp" }),
        k.edit ? h("span", { class: "x", title: "clear the edited frame", onClick: () => { k.edit = null; } }, "⟲") : null,
        h("span", { class: "x", title: "remove this keyframe", onClick: () => removeKf(k) }, "✕"),
      ]),
    ]));
    const linked = j => !!io.node.inputs?.find(x => x.name === `picture_${j}`)?.link;
    const pics = plan.pictures.map((p, j) => h("div", { class: "pic", key: j }, [
      linked(j + 1)
        ? h("div", { class: "drop", title: "this picture comes from the connected input" }, "input")
        : p
          ? h("div", { class: "thumb", style: { backgroundImage: `url("${inputUrl(p)}")` }, title: "click to replace",
              onClick: async () => withFile(await pickFile(), q => { plan.pictures[j] = q; }) })
          : h("div", { ...dropProps(`pic${j}`, q => { plan.pictures[j] = q; }), title: "drop or click: a reference picture" }, "+ picture"),
      p && !linked(j + 1) ? h("span", { class: "x", title: "remove", onClick: () => { plan.pictures[j] = null; } }, "✕") : null,
      h("div", { class: "tag" }, `Picture ${j + 1}`),
    ]));
    return h("div", { class: "bkp" }, [
      h("div", { class: ["stage", curKf.value ? "onkf" : ""] }, s
        ? [h("img", { src: tempUrl(s, s.files[pos.value]), draggable: false }),
           h("span", { class: "badge" }, `${fmt(frame.value)} · frame ${frame.value}/${count.value - 1}`),
           curKf.value ? h("span", { class: "badge kf" }, curKf.value.edit ? "keyframe ✓" : "keyframe") : null]
        : [h("div", { class: "empty" }, ["Queue this node once to load the guide frames.", h("br"),
            "Then scrub, press ＋ Keyframe, and drop the edited frames on the cards."])]),
      h("div", { class: "ctl" }, [
        h("button", { disabled: !s, title: "previous keyframe", onClick: () => { stop(); jumpKf(-1); } }, "⏮"),
        h("button", { disabled: !s, title: "previous frame", onClick: () => { stop(); seekIdx(pos.value - 1); } }, "◀"),
        h("button", { disabled: !s, title: playing.value ? "pause" : "play", onClick: play }, playing.value ? "⏸" : "▶"),
        h("button", { disabled: !s, title: "next frame", onClick: () => { stop(); seekIdx(pos.value + 1); } }, "▶|"),
        h("button", { disabled: !s, title: "next keyframe", onClick: () => { stop(); jumpKf(1); } }, "⏭"),
        h("span", { class: "time" }, s ? `${fmt(frame.value)} / ${fmt(count.value - 1)}` : "--"),
        h("span", { class: "sp" }),
        curKf.value
          ? h("button", { class: "danger", onClick: () => removeKf(curKf.value) }, "－ Keyframe")
          : h("button", { class: "pri", disabled: !s, onClick: addHere, title: "mark the current frame as a keyframe" }, "＋ Keyframe"),
      ]),
      h("div", { class: "tl", onPointerdown: tlDown }, s ? [
        h("div", { class: "fill", style: { width: pct(frame.value) } }),
        ...sorted.value.map(k => h("div", { class: ["mk", k.edit ? "done" : "", k.frame === frame.value ? "cur" : ""], key: k.frame,
          style: { left: pct(k.frame) }, title: `${fmt(k.frame)}${k.edit ? " (edited)" : " (no edited frame yet)"}`,
          onPointerdown: e => { e.stopPropagation(); stop(); seekFrame(k.frame); } })),
        h("div", { class: "head", style: { left: pct(frame.value) } }),
      ] : []),
      h("div", { class: "sec" }, ["Keyframes ", h("span", { class: "n" }, String(plan.keyframes.length)),
        plan.keyframes.length ? h("span", null, ` · ${plan.keyframes.filter(k => k.edit).length} edited`) : null]),
      h("div", { class: "row" }, cards.length ? cards : [h("div", { class: "hint" }, "No keyframes yet.")]),
      h("div", { class: "sec" }, "References"),
      h("div", { class: "pics" }, pics),
      err.value ? h("div", { class: "err" }, err.value) : h("div", { class: "hint" }, "Outputs follow the plan on the next run. Without edited frames, source_frames are ready to edit."),
    ]);
  };
}

app.registerExtension({
  name: "BFSNodes.KeyframePlanner",
  async nodeCreated(node) {
    if (node.comfyClass !== "BFSKeyframePlanner") return;
    styles();
    const widget = node.widgets?.find(w => w.name === "plan");
    if (widget) {   // the JSON stays a widget (saved with the workflow) but is never shown
      widget.type = "hidden"; widget.hidden = true; widget.computeSize = () => [0, -4];
      widget.options = { ...(widget.options || {}), hidden: true, serialize: true };
      widget.serializeValue = () => widget.value;
    }
    const fpsW = node.widgets?.find(w => w.name === "fps");
    const host = document.createElement("div");
    host.style.cssText = "position:relative;width:100%;height:100%;min-width:440px;min-height:520px";
    node.addDOMWidget("bfs_keyframe_planner", "div", host, { serialize: false, hideOnZoom: false, hideInPanel: true });
    wheelToCanvas(host);
    const handle = {};
    createApp({
      setup: () => Panel({
        node,
        get: () => widget?.value ?? "{}",
        set: v => { if (widget) widget.value = v; node.graph?.setDirtyCanvas(true); },
        fps: () => Number(fpsW?.value) || 24,
        expose: o => Object.assign(handle, o),
      }),
    }).mount(host);
    const onExecuted = node.onExecuted;
    node.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      handle.executed?.(output?.bfs_kfplan?.[0]);
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
