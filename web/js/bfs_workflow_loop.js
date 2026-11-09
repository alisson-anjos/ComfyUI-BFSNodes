/**
 * BFS Workflow Loop: the panel. Lists the nodes upstream of `result`. "Inject" picks a node whose output is replaced
 * by the current item (or its mask) on every iteration; "Change per iteration" picks widgets with one value per
 * iteration (one per line, or "a, b, c" / "start:stop:step" for numbers). State lives in the hidden `loop` widget.
 */
import { app } from "../../scripts/app.js";
import { wheelToCanvas } from "./bfs_wheel.js";
import { createApp, h, reactive, computed, watch } from "./vendor/vue.esm-browser.prod.mjs";

const SKIP_TYPES = new Set(["hidden", "converted-widget", "button"]);
const SKIP_NAMES = new Set(["control_after_generate", "upload", "image_upload", "videopreview", "loop"]);

function upstreamNodes(node) {
  const graph = node.graph || app.graph;
  const getLink = id => graph.links[id] ?? graph.links.get?.(id);
  const inp = node.inputs?.find(i => i.name === "result");
  const link = inp?.link != null ? getLink(inp.link) : null;
  if (!link) return [];
  const seen = new Set(), order = [], stack = [link.origin_id];
  while (stack.length) {
    const id = stack.pop();
    if (seen.has(id)) continue;
    seen.add(id);
    const n = graph.getNodeById(id);
    if (!n) continue;
    order.push(n);
    for (const i of n.inputs || []) if (i.link != null) { const l = getLink(i.link); if (l) stack.push(l.origin_id); }
  }
  return order.sort((a, b) => a.id - b.id);
}
const widgetsOf = n => (n?.widgets || []).filter(w => w.name && !SKIP_TYPES.has(w.type) && !SKIP_NAMES.has(w.name) && !w.hidden);
const kindOf = w => (w?.type === "combo" || Array.isArray(w?.options?.values)) ? "combo"
  : (w?.type === "toggle" || typeof w?.value === "boolean") ? "bool" : (w?.type === "number" || typeof w?.value === "number") ? "number" : "text";

function parseValues(text, kind) {
  if (kind === "number") {
    const out = [];
    for (const part of String(text).split(/[,\n]/).map(s => s.trim()).filter(Boolean)) {
      const r = part.split(":").map(Number);
      if (r.length === 3 && r.every(Number.isFinite) && r[2] !== 0) {
        for (let v = r[0], i = 0; (r[2] > 0 ? v <= r[1] + 1e-9 : v >= r[1] - 1e-9) && i < 256; v += r[2], i++) out.push(+v.toFixed(6));
      } else if (Number.isFinite(+part)) out.push(+part);
    }
    return out;
  }
  if (kind === "bool") return String(text).split(/[,\n]/).map(s => s.trim().toLowerCase()).filter(Boolean).map(s => ["1", "true", "yes", "on"].includes(s));
  return String(text).split("\n").map(s => s.trim()).filter(Boolean);
}

function parseState(text) {
  let s = {};
  try { s = JSON.parse(text || "{}") || {}; } catch { s = {}; }
  return {
    inject: (s.inject || []).map(i => ({ node: String(i.node ?? ""), slot: +i.slot || 0, source: i.source || "items" })),
    params: (s.params || []).map(p => ({ node: String(p.node ?? ""), widget: p.widget || "", text: p.text ?? (p.values || []).join("\n") })),
  };
}

let styled = false;
function styles() {
  if (styled) return;
  styled = true;
  const css = document.createElement("style");
  css.textContent = `
.bwl{position:absolute;inset:0;display:flex;flex-direction:column;gap:8px;padding:6px;box-sizing:border-box;font:12px system-ui,sans-serif;color:#ddd;overflow:auto}
.bwl .sec{display:flex;align-items:center;gap:6px;color:#9ab;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.bwl .row{background:#1d1d1d;border:1px solid #363636;border-radius:6px;padding:5px;display:flex;flex-direction:column;gap:4px}
.bwl .line{display:flex;gap:4px;align-items:center}
.bwl select,.bwl textarea{background:#262626;color:#ddd;border:1px solid #4a4a4a;border-radius:4px;padding:3px 5px;font:inherit;min-width:0;flex:1}
.bwl textarea{resize:vertical;min-height:38px}
.bwl button{background:#333;color:#ddd;border:1px solid #555;border-radius:4px;padding:2px 8px;cursor:pointer;font:inherit}
.bwl button:hover{background:#444}
.bwl .x{color:#999;cursor:pointer;padding:0 4px}.bwl .x:hover{color:#f66}
.bwl .hint{color:#888;font-size:11px}.bwl .warn{color:#f7a}
.bwl .foot{color:#bbb}.bwl .foot b{color:#fff}
`;
  document.head.appendChild(css);
}

function Panel(io) {
  const st = reactive(parseState(io.get()));
  const tick = reactive({ n: 0 });
  const nodes = computed(() => (tick.n, upstreamNodes(io.node)));
  const byId = id => nodes.value.find(n => String(n.id) === String(id));
  const save = () => io.set(JSON.stringify({
    inject: st.inject.filter(i => i.node).map(i => ({ node: i.node, slot: i.slot, source: i.source })),
    params: st.params.filter(p => p.node && p.widget).map(p => {
      const w = widgetsOf(byId(p.node)).find(x => x.name === p.widget);
      return { node: p.node, widget: p.widget, values: parseValues(p.text, kindOf(w)), text: p.text };
    }),
  }));
  watch(st, save, { deep: true });
  io.expose({ refresh: () => { tick.n++; }, reload: () => { Object.assign(st, parseState(io.get())); tick.n++; } });

  const nodeSelect = (obj, onPick) => h("select", { value: obj.node, onChange: e => { obj.node = e.target.value; onPick?.(); } },
    [h("option", { value: "" }, "— pick a node —"), ...nodes.value.map(n => h("option", { value: String(n.id) }, `[${n.id}] ${n.title || n.type}`))]);

  const injectRow = (i, idx) => {
    const n = byId(i.node);
    const outs = (n?.outputs || []).map((o, k) => ({ k, label: `${o.name || o.type} (${o.type})` }));
    return h("div", { class: "row", key: "i" + idx }, [
      h("div", { class: "line" }, [nodeSelect(i, () => { i.slot = 0; }), h("span", { class: "x", onClick: () => st.inject.splice(idx, 1) }, "✕")]),
      n ? h("div", { class: "line" }, [
        h("select", { value: String(i.slot), onChange: e => { i.slot = +e.target.value; } }, outs.map(o => h("option", { value: String(o.k) }, `output: ${o.label}`))),
        h("select", { value: i.source, onChange: e => { i.source = e.target.value; } },
          [h("option", { value: "items" }, "← item i (items)"), h("option", { value: "items_b" }, "← item i (items_b)"),
           h("option", { value: "item_masks" }, "← mask i (item_masks)")]),
      ]) : null,
    ]);
  };
  const paramRow = (p, idx) => {
    const n = byId(p.node);
    const w = widgetsOf(n).find(x => x.name === p.widget);
    const kind = kindOf(w);
    const vals = parseValues(p.text, kind);
    return h("div", { class: "row", key: "p" + idx }, [
      h("div", { class: "line" }, [nodeSelect(p, () => { p.widget = ""; p.text = ""; }), h("span", { class: "x", onClick: () => st.params.splice(idx, 1) }, "✕")]),
      n ? h("select", { value: p.widget, onChange: e => { p.widget = e.target.value; const ww = widgetsOf(n).find(x => x.name === p.widget); p.text = ww ? String(ww.value ?? "") : ""; } },
        [h("option", { value: "" }, "— pick a widget —"), ...widgetsOf(n).map(x => h("option", { value: x.name }, `${x.name}  (${String(x.value).split(/[\\/]/).pop().slice(0, 36)})`))]) : null,
      w ? h("textarea", { value: p.text, placeholder: kind === "number" ? "1, 2, 3   or   1:10:1" : "one value per line (iteration 1, 2, 3...)",
        onInput: e => { p.text = e.target.value; } }) : null,
      w ? h("div", { class: "hint" }, kind === "combo" ? `${vals.length} values; options: ${(w.options?.values || []).slice(0, 6).map(v => String(v).split(/[\\/]/).pop()).join(", ")}…`
        : `${vals.length} values${vals.length ? ": " + vals.slice(0, 8).join(", ") + (vals.length > 8 ? "…" : "") : ""}`) : null,
    ]);
  };
  const itemsLinked = () => !!io.node.inputs?.find(x => x.name === "items")?.link;

  return () => h("div", { class: "bwl" }, [
    nodes.value.length ? null : h("div", { class: "warn" }, "Connect what one iteration produces to `result`; its upstream nodes appear here."),
    h("div", { class: "sec" }, ["Inject the current item into", h("span", { style: "flex:1" }),
      h("button", { onClick: () => st.inject.push({ node: "", slot: 0, source: "items" }) }, "＋")]),
    ...st.inject.map(injectRow),
    st.inject.length ? null : h("div", { class: "hint" }, "e.g. the Load Image that stands for “the frame”: on iteration i it gets item i."),
    h("div", { class: "sec" }, ["Change per iteration", h("span", { style: "flex:1" }),
      h("button", { onClick: () => st.params.push({ node: "", widget: "", text: "" }) }, "＋")]),
    ...st.params.map(paramRow),
    st.params.length ? null : h("div", { class: "hint" }, "e.g. a seed or a prompt line per iteration (cycled when shorter than the items)."),
    h("div", { class: "foot" }, [itemsLinked() ? "Iterations: one per item." : "No items connected: iterations = the longest value list (or count).",
      h("span", { style: "flex:1" }), h("button", { onClick: () => { tick.n++; }, title: "re-read the graph" }, "↻")]),
  ]);
}

app.registerExtension({
  name: "BFSNodes.WorkflowLoop",
  async nodeCreated(node) {
    if (node.comfyClass !== "BFSWorkflowLoop") return;
    styles();
    const widget = node.widgets?.find(w => w.name === "loop");
    if (widget) {
      widget.type = "hidden"; widget.hidden = true; widget.computeSize = () => [0, -4];
      widget.options = { ...(widget.options || {}), hidden: true, serialize: true };
      widget.serializeValue = () => widget.value;
    }
    const host = document.createElement("div");
    host.style.cssText = "position:relative;width:100%;height:100%;min-width:340px;min-height:360px";
    node.addDOMWidget("bfs_workflow_loop", "div", host, { serialize: false, hideOnZoom: false, hideInPanel: true });
    wheelToCanvas(host);
    const handle = {};
    createApp({ setup: () => Panel({ node, get: () => widget?.value ?? "{}",
      set: v => { if (widget) widget.value = v; node.graph?.setDirtyCanvas(true); }, expose: o => Object.assign(handle, o) }) }).mount(host);
    const onConn = node.onConnectionsChange;
    node.onConnectionsChange = function () { const r = onConn?.apply(this, arguments); handle.refresh?.(); return r; };
    const onConfigure = node.onConfigure;
    node.onConfigure = function () { const r = onConfigure?.apply(this, arguments); setTimeout(() => handle.reload?.(), 0); return r; };
    node.size = [Math.max(node.size[0], 380), Math.max(node.size[1], 440)];
  },
});
