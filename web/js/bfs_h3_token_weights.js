import { app } from "../../scripts/app.js";
import { createApp, h, ref, reactive, computed } from "./vendor/vue.esm-browser.prod.mjs";

function parsePromptWeights(source) {
  function parse(text) {
    let out = "", spans = [], i = 0;
    while (i < text.length) {
      const ch = text[i];
      if (ch === "\\" && i + 1 < text.length && "():\\".includes(text[i + 1])) {
        out += text[i + 1]; i += 2; continue;
      }
      if (ch !== "(") { out += ch; i += 1; continue; }
      let depth = 1, j = i + 1, colon = -1;
      while (j < text.length && depth > 0) {
        if (text[j] === "\\" && j + 1 < text.length) { j += 2; continue; }
        if (text[j] === "(") depth += 1;
        else if (text[j] === ")") depth -= 1;
        else if (text[j] === ":" && depth === 1) colon = j;
        j += 1;
      }
      if (depth > 0) { out += ch; i += 1; continue; }
      const stop = j - 1;
      const inside = text.slice(i + 1, stop);
      const clock = /(?:^|[^\w])\d+:\d{2}(?::\d{2})?(?:\.\d+)?\s*$/.test(inside);
      const numericBody = colon >= 0 && /^[\d\s:+.\-]+$/.test(text.slice(i + 1, colon));
      let gain = null;
      if (colon >= 0 && !clock && !numericBody) {
        const suffix = text.slice(colon + 1, stop).trim();
        if (/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(suffix)) {
          const val = Number(suffix);
          if (Number.isFinite(val) && val >= 0.05 && val <= 3) gain = val;
        }
      }
      const sub = parse(gain !== null ? text.slice(i + 1, colon) : inside);
      const prefix = gain !== null ? "" : "(";
      const piece = prefix + sub.text + (gain !== null ? "" : ")");
      const base = out.length + prefix.length;
      for (const s of sub.spans) spans.push({ start: base + s.start, end: base + s.end, gain: s.gain });
      if (gain !== null && sub.text.trim()) spans.push({ start: out.length, end: out.length + sub.text.length, gain });
      out += piece;
      i = j;
    }
    return { text: out, spans };
  }
  return parse(source || "");
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function mapPhrasesToWeights(text, tokens, prompt) {
  if (!text || !text.trim() || !tokens?.length) return {};
  const { text: clean, spans } = parsePromptWeights(text);
  if (!spans.length) return {};
  let mapped = spans;
  if (clean !== prompt && prompt) {
    mapped = [];
    for (const s of spans) {
      const phrase = clean.slice(s.start, s.end);
      if (!phrase) continue;
      const pre = /[\w]/.test(phrase[0]) ? "(?<!\\w)" : "";
      const post = /[\w]/.test(phrase[phrase.length - 1]) ? "(?!\\w)" : "";
      const re = new RegExp(pre + escapeRegex(phrase) + post, "g");
      for (const m of prompt.matchAll(re)) mapped.push({ start: m.index, end: m.index + phrase.length, gain: s.gain });
    }
  }
  const out = {};
  for (const s of mapped) {
    for (const t of tokens) {
      if (t.char_start == null || t.char_end == null) continue;
      if (t.char_start < s.end && t.char_end > s.start) {
        const g = Math.min(3, Math.max(0.05, Math.round(((out[t.index] ?? 1) * s.gain) * 100) / 100));
        out[t.index] = g;
      }
    }
  }
  for (const k of Object.keys(out)) if (out[k] === 1) delete out[k];
  return out;
}

function buildAnnotatedPrompt(prompt, tokens, weights) {
  if (!prompt || !tokens?.length) return prompt || "";
  const entries = Object.entries(weights || {})
    .map(([k, v]) => [Number(k), Number(v)])
    .filter(([i, g]) => Number.isFinite(i) && Number.isFinite(g) && g !== 1 && tokens[i]?.char_start != null)
    .sort((a, b) => a[0] - b[0]);
  if (!entries.length) return prompt;
  const groups = [];
  for (const [idx, gain] of entries) {
    const prev = groups[groups.length - 1];
    if (prev && prev.b + 1 === idx && prev.gain === gain) prev.b = idx;
    else groups.push({ a: idx, b: idx, gain });
  }
  let out = "", cursor = 0;
  for (const { a, b, gain } of groups) {
    const ca = tokens[a].char_start, cb = tokens[b].char_end;
    if (ca < cursor) continue;
    const raw = prompt.slice(ca, cb);
    const lead = raw.slice(0, raw.length - raw.trimStart().length);
    const trail = raw.slice(raw.trimEnd().length);
    const core = raw.trim() || raw;
    const gStr = String(Number(gain.toFixed(2)));
    out += prompt.slice(cursor, ca) + `${lead}(${core}:${gStr})${trail}`;
    cursor = cb;
  }
  return out + prompt.slice(cursor);
}

app.registerExtension({
  name: "BFS.H3.TokenWeightEditor",
  async beforeRegisterNodeDef(type, data) {
    if (data.name !== "BFSH3TokenWeightEditor") return;
    const created = type.prototype.onNodeCreated;
    type.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments), node = this;
      const host = document.createElement("div");
      host.style.cssText = "width:100%;height:100%;min-width:480px;min-height:370px;box-sizing:border-box";

      const view = ref(null);
      const filter = ref("");
      const status = ref("Queue once in inspect-only mode to load tokens. Changes apply on the next queued run.");
      const selected = reactive(new Set());
      const docState = ref({});
      let syncing = false, phrasesHooked = false;

      const widget = () => node.widgets?.find(w => w.name === "weights_json");
      const phrasesWidget = () => node.widgets?.find(w => w.name === "weighted_phrases");
      function readDoc() { try { return JSON.parse(widget()?.value || "{}"); } catch { return {}; } }

      function save(d, updatePhrases = true) {
        syncing = true;
        try {
          docState.value = d;
          const w = widget();
          if (w) { w.value = JSON.stringify(d); w.callback?.(w.value); }
          if (updatePhrases && view.value?.prompt) {
            const pw = phrasesWidget();
            if (pw) {
              const nextText = buildAnnotatedPrompt(view.value.prompt, view.value.tokens, d.weights || {});
              if (pw.value !== nextText) {
                pw.value = nextText;
                if (pw.inputEl && pw.inputEl.value !== nextText) pw.inputEl.value = nextText;
                pw.callback?.(pw.value);
              }
            }
          }
          node.setDirtyCanvas(true, true);
        } finally { syncing = false; }
      }

      function syncFromPhrases(text) {
        if (syncing || !view.value?.prompt) return;
        const weights = mapPhrasesToWeights(text, view.value.tokens, view.value.prompt);
        save({ prompt_sha256: view.value.prompt_sha256, weights }, false);
        status.value = "Synced from weighted_phrases. Queue to apply.";
      }

      function hookPhrasesWidget() {
        if (phrasesHooked) return;
        const pw = phrasesWidget();
        if (!pw) return;
        phrasesHooked = true;
        const origCb = pw.callback;
        pw.callback = function (val) { origCb?.apply(this, arguments); syncFromPhrases(val); };
        if (pw.inputEl) pw.inputEl.addEventListener("input", e => syncFromPhrases(e.target.value));
      }

      function runAction(action) {
        if (!view.value) return;
        let d = readDoc();
        if (action === "clear") {
          d = { prompt_sha256: view.value.prompt_sha256, weights: {} };
          selected.clear();
        } else {
          if (d.prompt_sha256 && d.prompt_sha256 !== view.value.prompt_sha256) {
            status.value = "Prompt changed. Reset all before weighting the new text.";
            return;
          }
          d = { ...d, prompt_sha256: view.value.prompt_sha256, weights: { ...d.weights } };
          for (const i of selected) {
            const base = Number(view.value.tokens.find(t => t.index === i)?.base_gain ?? 1);
            const gain = action === "reset" ? base : Math.min(3, Math.max(0.05, Math.round((Number(d.weights[i] ?? base) + (action === "plus" ? 0.1 : -0.1)) * 100) / 100));
            if (gain === base) delete d.weights[i]; else d.weights[i] = gain;
          }
        }
        save(d, true);
        status.value = "Weights changed. Turn inspect_only off and queue to test. Other parameters and seed should stay fixed.";
      }

      const vueApp = createApp({
        setup() {
          const filteredTokens = computed(() => {
            if (!view.value?.tokens) return [];
            const q = filter.value.toLowerCase();
            return q ? view.value.tokens.filter(t => `${t.label} ${t.word}`.toLowerCase().includes(q)) : view.value.tokens;
          });
          const maxNorm = computed(() => Math.max(1, ...(view.value?.tokens || []).map(t => t.norm)));
          const isDocValid = computed(() => !docState.value.prompt_sha256 || docState.value.prompt_sha256 === view.value?.prompt_sha256);

          return () => h("div", {
            style: "width:100%;height:100%;box-sizing:border-box;padding:12px;background:#101725;color:#d9e6ff;border-radius:10px;font:12px system-ui;display:flex;flex-direction:column;gap:8px"
          }, [
            h("b", "H3 · Text token weights"),
            h("div", "Original tokenizer labels · synced with weighted_phrases"),
            h("input", {
              value: filter.value,
              onInput: e => { filter.value = e.target.value; },
              placeholder: "Filter token / word…",
              style: "color:#eee;background:#1b2940;padding:5px;border:1px solid #375071"
            }),
            h("div", { style: "display:flex;gap:6px;align-items:center" }, [
              h("button", { type: "button", onClick: () => runAction("minus") }, "−0.1"),
              h("button", { type: "button", onClick: () => runAction("plus") }, "+0.1"),
              h("button", { type: "button", onClick: () => runAction("reset") }, "Reset selected"),
              h("button", { type: "button", onClick: () => runAction("clear") }, "Reset overrides"),
            ]),
            h("div", `${selected.size} selected · click tokens; double-click selects matching word fragments`),
            h("div", {
              style: "flex:1;overflow:auto;min-height:150px;display:flex;flex-wrap:wrap;gap:4px;align-content:flex-start"
            }, filteredTokens.value.map(token => {
              const gain = isDocValid.value ? Number(docState.value.weights?.[token.index] ?? token.base_gain ?? 1) : Number(token.base_gain ?? 1);
              return h("button", {
                key: token.index,
                type: "button",
                title: `Token ${token.index} · ID ${token.token_id}\nOriginal word/fragment: ${token.word}\nEmbedding norm: ${token.norm.toFixed(3)}\nNorm is not attention or importance. Double-click selects this word's occurrences.`,
                style: `font:12px ui-monospace,monospace;white-space:pre-wrap;padding:6px;border-radius:5px;border:1px solid ${selected.has(token.index) ? "#fff" : gain > 1 ? "#53c9b2" : gain < 1 ? "#df9d68" : "#304769"};background:linear-gradient(to top,#233d58 ${token.norm / maxNorm.value * 100}%,#172235 0);color:${gain > 1 ? "#9ff2d9" : gain < 1 ? "#ffca9c" : "#cbd9eb"};cursor:pointer`,
                onClick: e => {
                  if (!e.ctrlKey && !e.metaKey) selected.clear();
                  if (selected.has(token.index)) selected.delete(token.index); else selected.add(token.index);
                },
                onDblclick: () => {
                  selected.clear();
                  for (const t of view.value.tokens) if (t.word === token.word) selected.add(t.index);
                }
              }, `${token.label.replaceAll("\n", "↵") || "∅"} ×${gain.toFixed(2)}`);
            })),
            h("div", { style: "color:#91b2d9" }, status.value)
          ]);
        }
      });
      vueApp.mount(host);

      this._bfsTokenPanel = {
        update(d) {
          view.value = d;
          selected.clear();
          status.value = d.message;
          docState.value = readDoc();
          hookPhrasesWidget();
          const pw = phrasesWidget();
          if (pw && d.prompt && !pw.value?.trim()) {
            const weights = {};
            for (const t of d.tokens) if (t.gain != null && t.gain !== 1) weights[t.index] = t.gain;
            syncing = true;
            try {
              const nextText = buildAnnotatedPrompt(d.prompt, d.tokens, weights);
              pw.value = nextText;
              if (pw.inputEl) pw.inputEl.value = nextText;
            } finally { syncing = false; }
          }
        }
      };

      const onRemoved = node.onRemoved;
      node.onRemoved = function () { vueApp.unmount(); return onRemoved?.apply(this, arguments); };

      this.addDOMWidget("bfs_h3_token_panel", "div", host, { serialize: false, hideOnZoom: false, getMinHeight: () => 370 });
      this.setSize([Math.max(this.size[0], 550), Math.max(this.size[1], 660)]);
      return result;
    };
    const executed = type.prototype.onExecuted;
    type.prototype.onExecuted = function (message) { executed?.apply(this, arguments); const d = message?.bfs_h3_tokens?.[0]; if (d) this._bfsTokenPanel?.update(d); };
  }
});
