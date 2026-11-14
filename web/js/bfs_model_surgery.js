import { app } from '../../scripts/app.js';

app.registerExtension({
  name: 'BFS.ModelSurgery',
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== 'BFSModelSurgery') return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      const node = this;
      const root = document.createElement('div');
      root.style.cssText = 'padding:8px;box-sizing:border-box;background:#202024;color:#ddd;overflow:auto;height:100%;font:12px sans-serif';
      const note = document.createElement('p');
      note.textContent = 'Run once to inspect the connected MODEL. Connect after LoRAs. Overlapping rules multiply.';
      const search = document.createElement('input');
      search.placeholder = 'Filter layers / glob for new rule';
      search.style.width = '95%';
      const scale = document.createElement('input');
      scale.type = 'number'; scale.value = '1'; scale.step = '0.05'; scale.style.width = '65px';
      const add = document.createElement('button'); add.textContent = 'Add pattern rule';
      const summary = document.createElement('div');
      const list = document.createElement('div');
      const rulesView = document.createElement('div');
      root.append(note, search, scale, add, summary, rulesView, list);
      let layers = [];
      const rulesWidget = () => node.widgets?.find(w => w.name === 'rules');
      const read = () => { const v = JSON.parse(rulesWidget()?.value || '[]'); if (!Array.isArray(v)) throw Error('Rules must be a list'); return v; };
      const write = rules => { const w = rulesWidget(); w.value = JSON.stringify(rules, null, 2); w.callback?.(w.value); node.graph?.setDirtyCanvas(true, true); renderRules(); };
      const run = fn => { try { fn(); } catch (e) { summary.textContent = e.message; } };
      function renderRules() {
        rulesView.replaceChildren();
        run(() => read().forEach((rule, index) => {
          const row = document.createElement('div'); row.style.cssText = 'padding:4px 0;border-bottom:1px solid #444';
          const enabled = document.createElement('input'); enabled.type = 'checkbox'; enabled.checked = rule.enabled !== false;
          enabled.onchange = () => run(() => { const rules = read(); rules[index].enabled = enabled.checked; write(rules); });
          const label = document.createElement('span'); label.textContent = `${rule.pattern}${rule.rows ? ' rows '+rule.rows.join(':') : ''} × `;
          const value = document.createElement('input'); value.type = 'number'; value.step = '0.05'; value.value = rule.scale ?? 1; value.style.width = '65px';
          value.onchange = () => run(() => { const rules = read(); rules[index].scale = Number(value.value); write(rules); });
          const remove = document.createElement('button'); remove.textContent = '×';
          remove.onclick = () => run(() => { const rules = read(); rules.splice(index, 1); write(rules); });
          row.append(enabled, label, value, remove); rulesView.append(row);
        }));
      }
      function renderLayers() {
        list.replaceChildren();
        const query = search.value.toLowerCase().replaceAll('*', '');
        const matches = layers.filter(v => v.key.toLowerCase().includes(query));
        summary.textContent = `${matches.length} matching parameters; displaying up to 150.`;
        for (const layer of matches.slice(0, 150)) {
          const row = document.createElement('div'); row.style.cssText = 'padding:4px;border-bottom:1px solid #333;overflow-wrap:anywhere';
          const name = document.createElement('span'); name.textContent = `${layer.key} [${layer.shape.join(' × ')}] `;
          const button = document.createElement('button'); button.textContent = 'Add';
          button.onclick = () => run(() => write([...read(), { pattern: layer.key, scale: Number(scale.value) }]));
          row.append(name, button); list.append(row);
        }
      }
      search.oninput = renderLayers;
      add.onclick = () => run(() => write([...read(), {pattern: search.value || '*', scale: Number(scale.value)}]));
      this.addDOMWidget('model_surgery_panel', 'custom', root, {serialize: false, getMinHeight: () => 250});
      this.size[0] = Math.max(this.size[0], 540);
      const onExecuted = this.onExecuted;
      this.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        if (message.surgery_layers) layers = message.surgery_layers;
        renderRules(); renderLayers();
      };
      const onConfigure = this.onConfigure;
      this.onConfigure = function () { onConfigure?.apply(this, arguments); renderRules(); };
      renderRules();
    };
  }
});
