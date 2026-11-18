import { app } from "../../scripts/app.js";

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
      host.style.cssText = "width:100%;height:100%;box-sizing:border-box;padding:12px;background:#101725;color:#d9e6ff;border-radius:10px;font:12px system-ui;display:flex;flex-direction:column;gap:8px";
      host.innerHTML = `<b>H3 · Text token weights</b><div>Original tokenizer labels · synced with weighted_phrases</div><input class="search" placeholder="Filter token / word…" style="color:#eee;background:#1b2940;padding:5px;border:1px solid #375071"><div class="actions" style="display:flex;gap:6px;align-items:center"><button data-action="minus">−0.1</button><button data-action="plus">+0.1</button><button data-action="reset">Reset selected</button><button data-action="clear">Reset overrides</button></div><div class="selection">Click a token; Ctrl/Cmd-click to select several.</div><div class="tokens" style="flex:1;overflow:auto;min-height:150px;display:flex;flex-wrap:wrap;gap:4px;align-content:flex-start"></div><div class="status" style="color:#91b2d9">Queue once in inspect-only mode to load tokens. Changes apply on the next queued run.</div>`;
      const box = host.querySelector(".tokens"), status = host.querySelector(".status"), selected = new Set();
      let view = null, syncing = false, phrasesHooked = false;
      const widget = () => node.widgets?.find(w => w.name === "weights_json");
      const phrasesWidget = () => node.widgets?.find(w => w.name === "weighted_phrases");
      function doc() { try { return JSON.parse(widget()?.value || "{}"); } catch { return {}; } }
      function save(d, updatePhrases = true) {
        syncing = true;
        try {
          const w = widget();
          if (w) { w.value = JSON.stringify(d); w.callback?.(w.value); }
          if (updatePhrases && view?.prompt) {
            const pw = phrasesWidget();
            if (pw) {
              const nextText = buildAnnotatedPrompt(view.prompt, view.tokens, d.weights || {});
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
        if (syncing || !view?.prompt) return;
        const weights = mapPhrasesToWeights(text, view.tokens, view.prompt);
        const d = { prompt_sha256: view.prompt_sha256, weights };
        save(d, false);
        status.textContent = "Synced from weighted_phrases. Queue to apply.";
        render();
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
      function render() {
        if (!view) return;
        const d = doc(), valid = !d.prompt_sha256 || d.prompt_sha256 === view.prompt_sha256;
        const filter = host.querySelector(".search").value.toLowerCase(); box.replaceChildren();
        const max = Math.max(1, ...view.tokens.map(t => t.norm));
        for (const token of view.tokens) {
          if (filter && !`${token.label} ${token.word}`.toLowerCase().includes(filter)) continue;
          const gain = valid ? Number(d.weights?.[token.index] ?? token.base_gain ?? 1) : Number(token.base_gain ?? 1);
          const button = document.createElement("button"); button.type = "button";
          button.textContent = `${token.label.replaceAll("\n", "↵") || "∅"} ×${gain.toFixed(2)}`;
          button.title = `Token ${token.index} · ID ${token.token_id}\nOriginal word/fragment: ${token.word}\nEmbedding norm: ${token.norm.toFixed(3)}\nNorm is not attention or importance. Double-click selects this word's occurrences.`;
          button.style.cssText = `font:12px ui-monospace,monospace;white-space:pre-wrap;padding:6px;border-radius:5px;border:1px solid ${selected.has(token.index) ? "#fff" : gain > 1 ? "#53c9b2" : gain < 1 ? "#df9d68" : "#304769"};background:linear-gradient(to top,#233d58 ${token.norm / max * 100}%,#172235 0);color:${gain > 1 ? "#9ff2d9" : gain < 1 ? "#ffca9c" : "#cbd9eb"};cursor:pointer`;
          button.onclick = e => { if (!e.ctrlKey && !e.metaKey) selected.clear(); if (selected.has(token.index)) selected.delete(token.index); else selected.add(token.index); render(); };
          button.ondblclick = () => { selected.clear(); for (const t of view.tokens) if (t.word === token.word) selected.add(t.index); render(); }; box.append(button);
        }
        host.querySelector(".selection").textContent = `${selected.size} selected · click tokens; double-click selects matching word fragments`;
      }
      host.querySelector(".search").oninput = render;
      host.querySelector(".actions").onclick = e => {
        const action = e.target.dataset.action; if (!action || !view) return;
        let d = doc();
        if (action === "clear") { d = { prompt_sha256: view.prompt_sha256, weights: {} }; selected.clear(); }
        else {
          if (d.prompt_sha256 && d.prompt_sha256 !== view.prompt_sha256) { status.textContent = "Prompt changed. Reset all before weighting the new text."; return; }
          d = { ...d, prompt_sha256: view.prompt_sha256, weights: { ...d.weights } };
          for (const i of selected) {
            const base = Number(view.tokens.find(t => t.index === i)?.base_gain ?? 1);
            const gain = action === "reset" ? base : Math.min(3, Math.max(0.05, Math.round((Number(d.weights[i] ?? base) + (action === "plus" ? 0.1 : -0.1)) * 100) / 100));
            if (gain === base) delete d.weights[i]; else d.weights[i] = gain;
          }
        }
        save(d, true); status.textContent = "Weights changed. Turn inspect_only off and queue to test. Other parameters and seed should stay fixed."; render();
      };
      this._bfsTokenPanel = {
        update(d) {
          view = d; selected.clear(); status.textContent = d.message;
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
          render();
        }
      };
      this.addDOMWidget("bfs_h3_token_panel", "div", host, { serialize: false, hideOnZoom: false, getMinHeight: () => 370 });
      this.setSize([Math.max(this.size[0], 550), Math.max(this.size[1], 660)]);
      return result;
    };
    const executed = type.prototype.onExecuted;
    type.prototype.onExecuted = function (message) { executed?.apply(this, arguments); const d = message?.bfs_h3_tokens?.[0]; if (d) this._bfsTokenPanel?.update(d); };
  }
});
