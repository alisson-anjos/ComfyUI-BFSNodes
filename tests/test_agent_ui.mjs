import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = (await readFile(new URL('../web/js/shotloop/view_agent.js', import.meta.url), 'utf8'))
  .replace(/^import .*;\s*$/gm, '');
const stubs = `const reactive = x => x; const onMounted = () => {}; const onBeforeUnmount = () => {};
const h = (tag, props = {}, children = []) => ({tag, props, children});\n`;
const { createAgent, agentTab } = await import('data:text/javascript;base64,' + Buffer.from(stubs + source).toString('base64'));
function find(node, predicate) {
  if (!node || typeof node !== 'object') return;
  if (predicate(node)) return node;
  for (const child of Array.isArray(node.children) ? node.children : []) {
    const found = find(child, predicate);
    if (found) return found;
  }
}
function context() {
  const c = {plan: {}, save() {}, api: {fetchApi: async () => ({ok: true, json: async () => ({
    id: 'test', task: 'existing task', status: 'running', settings: {max_tokens: 4096}, events: [],
  })})}};
  c.agent = createAgent(c, {}, {});
  c.agent.a.task = 'existing task';
  c.agent.a.id = 'test';
  return c;
}
test('token edits survive a poll before blur, including an empty draft', async () => {
  const c = context();
  const label = find(agentTab(c), n => n.tag === 'label' && n.children[0] === 'Decision token budget');
  const input = label.children[1];
  assert.equal(typeof input.props.onInput, 'function');
  input.props.onInput({target: {value: '8192'}});
  await c.agent.poll();
  assert.equal(c.agent.a.settings.max_tokens, 8192);
  input.props.onInput({target: {value: ''}});
  await c.agent.poll();
  assert.equal(c.agent.a.settings.max_tokens, '');
});
test('Thinking checkbox still uses its checked state', () => {
  const c = context();
  const label = find(agentTab(c), n => n.tag === 'label' && n.children[0] === 'Thinking');
  label.children[1].props.onChange({target: {checked: false}});
  assert.equal(c.agent.a.settings.thinking, false);
});

test('invalid VAE workflow disables Resume and permits starting a fresh task', async () => {
  const c = context();
  c.api.fetchApi = async () => ({ok: true, json: async () => ({id: 'test', status: 'error',
    requires_new_task: true, observation: {error: 'Workflow VAE loader rejected its checkpoint.'}})});
  await c.agent.poll();
  const tree = agentTab(c);
  const resume = find(tree, n => n.tag === 'button' && n.children === 'Resume');
  const start = find(tree, n => n.tag === 'button' && n.children === '▶ Start automatic task');
  assert.equal(resume.props.disabled, true);
  assert.equal(start.props.disabled, false);
});

test('feedback survives polling, sends separately from task, and clears only on success', async () => {
  const c = context();
  c.agent.a.status = 'complete';
  let sent;
  c.api.fetchApi = async (url, options) => {
    if(options)sent=JSON.parse(options.body);
    return {ok:true,json:async()=>({id:'test',status:'running',events:[],feedback:[{text:'Preserve the other person'}]})};
  };
  const input=find(agentTab(c),n=>n.tag==='textarea'&&n.props.maxLength===4000);
  input.props.onInput({target:{value:'Preserve the other person'}});
  await c.agent.poll();
  assert.equal(c.agent.a.feedback,'Preserve the other person');
  c.agent.a.status='complete';
  assert.equal(find(agentTab(c),n=>n.tag==='button'&&n.children==='Send feedback and continue').props.disabled,false);
  await c.agent.sendFeedback();
  assert.equal(sent.action,'feedback');
  assert.equal(sent.feedback,'Preserve the other person');
  assert.equal(c.agent.a.task,'existing task');
  assert.equal(c.agent.a.feedback,'');
  assert.equal(c.agent.a.feedbackHistory.length,1);
  c.agent.a.feedback='Retry this correction';
  c.api.fetchApi=async()=>({ok:false,json:async()=>({error:'Wait for current action'})});
  await c.agent.sendFeedback();
  assert.equal(c.agent.a.feedback,'Retry this correction');
  assert.match(c.agent.a.error,/Wait for current action/);
});

test('feedback requires pausing active sessions and cannot bypass invalid workflow',()=>{
  const c=context();c.agent.a.feedback='Fix only the target';
  for(const status of ['running','waiting_render','stopped']){
    c.agent.a.status=status;
    assert.equal(find(agentTab(c),n=>n.tag==='button'&&n.children==='Send feedback and continue').props.disabled,true);
  }
  c.agent.a.status='paused';c.agent.a.requiresNewTask=true;
  assert.equal(find(agentTab(c),n=>n.tag==='button'&&n.children==='Send feedback and continue').props.disabled,true);
});
