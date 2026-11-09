/**
 * BFS XY Sweep: the panel. Lists every node upstream of the `result` input and its widgets (like TinyTerra's
 * advanced xyPlot); for X and Y pick a node + widget and its values. Combo widgets (LoRA names, samplers...) get a
 * filterable checklist; numbers take "0.5, 0.7, 0.9" or a range "start:stop:step"; text takes one value per line.
 * State lives in the hidden `sweep` STRING widget (JSON), saved with the workflow.
 */
import { app } from "../../scripts/app.js";
import { wheelToCanvas } from "./bfs_wheel.js";
import { createApp, h, reactive, computed, watch } from "./vendor/vue.esm-browser.prod.mjs";

const SKIP_TYPES = new Set(["hidden", "converted-widget", "button"]);
const SKIP_NAMES = new Set(["control_after_generate", "upload", "image_upload", "videopreview", "sweep"]);

function upstreamNodes(node) {
  const graph = node.graph || app.graph;
  const inp = node.inputs?.find(i => i.name === "result");
  const link = inp?.link != null ? graph.links[inp.link] ?? graph.links.get?.(inp.link) : null;
  if (!link) return [];
  const seen = new Set(), order = [], stack = [link.origin_id];
  while (stack.length) {
    const id = stack.pop();
    if (seen.has(id)) continue;
    seen.add(id);
    const n = graph.getNodeById(id);
    if (!n) continue;
    order.push(n);
    for (const i of n.inputs || []) {
      if (i.link == null) continue;
      const l = graph.links[i.link] ?? graph.links.get?.(i.link);
      if (l) stack.push(l.origin_id);
    }
  }
  return order.sort((a, b) => a.id - b.id);
}

function widgetsOf(n) {
  return (n.widgets || []).filter(w => w.name && !SKIP_TYPES.has(w.type) && !SKIP_NAMES.has(w.name) && !w.hidden
    && !(w.options && w.options.hidden));
}

function kindOf(w) {
  if (!w) return "text";
  if (w.type === "combo" || Array.isArray(w.options?.values)) return "combo";
  if (w.type === "toggle" || typeof w.value === "boolean") return "bool";
  if (w.type === "number" || typeof w.value === "number") return "number";
  return "text";
}

function parseNumbers(text) {
  const out = [];
  for (const part of String(text).split(/[,\n]/).map(s => s.trim()).filter(Boolean)) {
    const r = part.split(":").map(Number);
    if (r.length === 3 && r.every(Number.isFinite) && r[2] !== 0) {
      for (let v = r[0], i = 0; (r[2] > 0 ? v <= r[1] + 1e-9 : v >= r[1] - 1e-9) && i < 256; v += r[2], i++) out.push(+v.toFixed(6));
    } else if (Number.isFinite(+part)) out.push(+part);
  }
  return out;
}

function parseState(text) {
  let s = {};
  try { s = JSON.parse(text || "{}") || {}; } catch { s = {}; }
  const axis = a => ({ node: a?.node != null ? String(a.node) : "", widget: a?.widget || "", values: Array.isArray(a?.values) ? a.values : [],
                       text: a?.text ?? (Array.isArray(a?.values) ? a.values.join(", ") : ""), title: a?.title || "", filter: "" });
  return { x: axis(s.x), y: axis(s.y) };
}

let styled = false;
function styles() {
  if (styled) return;
  styled = true;
  const css = document.createElement("style");
  css.textContent = `
.bxy{position:absolute;inset:0;display:flex;flex-direction:column;gap:8px;padding:6px;box-sizing:border-box;font:12px system-ui,sans-serif;color:#ddd;overflow:auto}
.bxy .ax{background:#1d1d1d;border:1px solid #363636;border-radius:6px;padding:6px;display:flex;flex-direction:column;gap:5px}
.bxy .hd{display:flex;align-items:center;gap:6px}
.bxy .tag{font-weight:700;background:#2d5a8a;color:#fff;border-radius:4px;padding:1px 7px}
.bxy .tag.y{background:#7a3d8a}
.bxy select,.bxy input,.bxy textarea{background:#262626;color:#ddd;border:1px solid #4a4a4a;border-radius:4px;padding:3px 5px;font:inherit;min-width:0}
.bxy select{flex:1}
.bxy textarea{resize:vertical;min-height:40px}
.bxy .opts{max-height:120px;overflow:auto;display:flex;flex-direction:column;gap:1px;border:1px solid #333;border-radius:4px;padding:3px}
.bxy .opts label{display:flex;gap:5px;align-items:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;cursor:pointer}
.bxy .opts label:hover{background:#2a2a2a}
.bxy .chips{display:flex;flex-wrap:wrap;gap:3px}
.bxy .chip{background:#2a2a2a;border:1px solid #444;border-radius:9px;padding:0 6px;font-size:11px}
.bxy .hint{color:#888;font-size:11px}
.bxy .foot{display:flex;align-items:center;gap:6px;color:#bbb}
.bxy .foot .n{font-weight:700;color:#fff}
.bxy .warn{color:#f7a}
.bxy button{background:#333;color:#ddd;border:1px solid #555;border-radius:4px;padding:2px 8px;cursor:pointer;font:inherit}
.bxy button:hover{background:#444}
`;
  document.head.appendChild(css);
}

function Panel(io) {
  const st = reactive(parseState(io.get()));
  const tick = reactive({ n: 0 });           // bumps on graph changes so the node lists refresh
  const nodes = computed(() => (tick.n, upstreamNodes(io.node)));
  const nodeById = id => nodes.value.find(n => String(n.id) === String(id));
  const widgetOf = a => widgetsOf(nodeById(a.node) || {}).find(w => w.name === a.widget);

  const save = () => {
    const out = {};
    for (const k of ["x", "y"]) {
      const a = st[k];
      if (!a.node || !a.widget) continue;
      const kind = kindOf(widgetOf(a));
      let values = a.values;
      if (kind === "number") values = parseNumbers(a.text);
      else if (kind === "text") values = String(a.text).split("\n").map(s => s.trim()).filter(Boolean);
      else if (kind === "bool") values = [false, true].filter(v => a.values.includes(v));
      const n = nodeById(a.node);
      out[k] = { node: a.node, widget: a.widget, values, text: a.text, title: n ? (n.title || n.type) : a.title };
    }
    io.set(JSON.stringify(out));
  };
  watch(st, save, { deep: true });

  const count = computed(() => {
    const len = a => {
      if (!a.node || !a.widget) return 0;
      const kind = kindOf(widgetOf(a));
      if (kind === "number") return parseNumbers(a.text).length;
      if (kind === "text") return String(a.text).split("\n").filter(s => s.trim()).length;
      return a.values.length;
    };
    const x = len(st.x), y = len(st.y);
    return x * (y || 1);
  });

  io.expose({ refresh: () => { tick.n++; }, reload: () => { Object.assign(st, parseState(io.get())); tick.n++; } });

  const axisView = (k, label) => {
    const a = st[k];
    const n = nodeById(a.node);
    const ws = n ? widgetsOf(n) : [];
    const w = widgetOf(a);
    const kind = kindOf(w);
    const pickNode = e => { a.node = e.target.value; a.widget = ""; a.values = []; a.text = ""; };
    const pickWidget = e => {
      a.widget = e.target.value; a.values = []; const ww = widgetOf(a);
      a.text = ww && kindOf(ww) !== "combo" ? String(ww.value ?? "") : "";
    };
    let editor = null;
    if (w && kind === "combo") {
      const all = (w.options?.values || []).map(String);
      const shown = a.filter ? all.filter(v => v.toLowerCase().includes(a.filter.toLowerCase())) : all;
      editor = [
        h("input", { placeholder: "filter options…", value: a.filter, onInput: e => { a.filter = e.target.value; } }),
        h("div", { class: "opts" }, shown.slice(0, 400).map(v => h("label", { title: v }, [
          h("input", { type: "checkbox", checked: a.values.includes(v),
            onChange: e => { a.values = e.target.checked ? [...a.values, v] : a.values.filter(x => x !== v); } }), v]))),
        a.values.length ? h("div", { class: "chips" }, a.values.map(v => h("span", { class: "chip", title: v }, v.split(/[\\/]/).pop()))) : null,
      ];
    } else if (w && kind === "bool") {
      editor = h("div", { class: "chips" }, [false, true].map(v => h("label", null, [
        h("input", { type: "checkbox", checked: a.values.includes(v),
          onChange: e => { a.values = e.target.checked ? [...a.values, v] : a.values.filter(x => x !== v); } }), String(v)])));
    } else if (w && kind === "number") {
      editor = [h("input", { value: a.text, placeholder: "0.5, 0.7, 0.9   or   0.5:0.9:0.2", onInput: e => { a.text = e.target.value; } }),
                h("div", { class: "hint" }, `→ ${parseNumbers(a.text).join(", ") || "no values"}  (now ${w.value})`)];
    } else if (w) {
      editor = [h("textarea", { value: a.text, placeholder: "one value per line", onInput: e => { a.text = e.target.value; } })];
    }
    return h("div", { class: "ax" }, [
      h("div", { class: "hd" }, [h("span", { class: ["tag", k] }, label),
        h("select", { value: a.node, onChange: pickNode }, [h("option", { value: "" }, k === "y" ? "— none —" : "— pick a node —"),
          ...nodes.value.map(nd => h("option", { value: String(nd.id) }, `[${nd.id}] ${nd.title || nd.type}`))]),
      ]),
      n ? h("select", { value: a.widget, onChange: pickWidget }, [h("option", { value: "" }, "— pick a widget —"),
        ...ws.map(ww => h("option", { value: ww.name }, `${ww.name}  (${String(ww.value).split(/[\\/]/).pop().slice(0, 40)})`))]) : null,
      ...(Array.isArray(editor) ? editor : [editor]),
    ]);
  };

  return () => h("div", { class: "bxy" }, [
    nodes.value.length ? null : h("div", { class: "warn" }, "Connect the image to compare to `result`; its upstream nodes appear here."),
    axisView("x", "X"),
    axisView("y", "Y"),
    h("div", { class: "foot" }, [h("span", null, ["cells: ", h("span", { class: "n" }, String(count.value))]),
      count.value > 64 ? h("span", { class: "warn" }, "max 64") : null, h("span", { style: "flex:1" }),
      h("button", { onClick: () => { tick.n++; }, title: "re-read the graph" }, "↻")]),
    h("div", { class: "hint" }, "Only the nodes that depend on the swept widgets run once per cell; models and the rest run once."),
  ]);
}

app.registerExtension({
  name: "BFSNodes.XYSweep",
  async nodeCreated(node) {
    if (node.comfyClass !== "BFSXYSweep") return;
    styles();
    const widget = node.widgets?.find(w => w.name === "sweep");
    if (widget) {
      widget.type = "hidden"; widget.hidden = true; widget.computeSize = () => [0, -4];
      widget.options = { ...(widget.options || {}), hidden: true, serialize: true };
      widget.serializeValue = () => widget.value;
    }
    const host = document.createElement("div");
    host.style.cssText = "position:relative;width:100%;height:100%;min-width:340px;min-height:380px";
    node.addDOMWidget("bfs_xy_sweep", "div", host, { serialize: false, hideOnZoom: false, hideInPanel: true });
    wheelToCanvas(host);
    const handle = {};
    createApp({ setup: () => Panel({ node, get: () => widget?.value ?? "{}",
      set: v => { if (widget) widget.value = v; node.graph?.setDirtyCanvas(true); }, expose: o => Object.assign(handle, o) }) }).mount(host);
    const onConn = node.onConnectionsChange;
    node.onConnectionsChange = function () { const r = onConn?.apply(this, arguments); handle.refresh?.(); return r; };
    const onConfigure = node.onConfigure;
    node.onConfigure = function () { const r = onConfigure?.apply(this, arguments); setTimeout(() => handle.reload?.(), 0); return r; };
    node.size = [Math.max(node.size[0], 380), Math.max(node.size[1], 520)];
  },
});
