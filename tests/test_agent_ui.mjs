import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = (await readFile(new URL('../web/js/shotloop/view_agent.js', import.meta.url), 'utf8'))
  .replace(/^import .*;\s*$/gm, '');
const stubs = `const reactive = x => x; const onMounted = () => {}; const onBeforeUnmount = () => {};
function h(tag,props={},children){if(Array.isArray(props)||typeof props==="string"){children=props;props={};}return {tag,props,children:children??[]};}\n`;
const { createAgent, agentTab } = await import('data:text/javascript;base64,' + Buffer.from(stubs + source).toString('base64'));
function find(node, predicate) {
  if (!node || typeof node !== 'object') return;
  if (predicate(node)) return node;
  for (const child of Array.isArray(node.children) ? node.children : typeof node.children==='object' ? [node.children] : []) {
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
  c.agent.a.view='settings';
  return c;
}
test('polling preserves manual video, references and prompts while accepting unrelated agent edits', async () => {
  const c=context();
  let plan={video:'old.mp4',global_refs:['old.png'],global_prompt:'old prompt',segs:[{prompt:'old shot'}],vlm_cfg:{temperature:0.5,max_tokens:512},fps:16};
  c.plan=structuredClone(plan);
  c.api.fetchApi=async()=>({ok:true,json:async()=>({id:'test',status:'running',plan:structuredClone(plan)})});
  await c.agent.poll();
  Object.assign(c.plan,{video:'new.mp4',global_refs:['new.png'],global_prompt:'my prompt',segs:[{prompt:'my shot'}]});
  c.plan.vlm_cfg.temperature=0.9;
  await c.agent.poll();
  plan={...plan,global_prompt:'agent prompt',fps:24,vlm_cfg:{temperature:0.7,max_tokens:1024}};
  await c.agent.poll();
  assert.equal(c.plan.video,'new.mp4');
  assert.deepEqual(c.plan.global_refs,['new.png']);
  assert.equal(c.plan.global_prompt,'my prompt');
  assert.equal(c.plan.segs[0].prompt,'my shot');
  assert.deepEqual(c.plan.vlm_cfg,{temperature:0.9,max_tokens:1024});
  assert.equal(c.plan.fps,24);
  // Completed sessions used to restore the old plan on every poll too.
  c.api.fetchApi=async()=>({ok:true,json:async()=>({id:'test',status:'complete',plan})});
  await c.agent.poll();
  assert.equal(c.plan.video,'new.mp4');
});
test('reconnecting to an old session keeps the saved workflow plan', async () => {
  const c=context();
  c.plan={video:'new.mp4',global_refs:['new.png'],global_prompt:'new prompt'};
  c.api.fetchApi=async()=>({ok:true,json:async()=>({id:'test',status:'complete',plan:{video:'old.mp4',global_refs:[],global_prompt:'old'}})});
  await c.agent.poll();await c.agent.poll();
  assert.deepEqual({...c.plan,agent_session_id:undefined},{video:'new.mp4',global_refs:['new.png'],global_prompt:'new prompt',agent_session_id:undefined});
});
test('agent changes still reach untouched fields and removed fields while paused drafts survive Resume', async () => {
  const c=context();
  let status='running', plan={video:'clip.mp4',global_prompt:'old',global_refs:['ref.png'],obsolete:true};
  c.plan=structuredClone(plan);
  c.api.fetchApi=async(url,options)=>{
    if(options){plan=JSON.parse(options.body).plan;status='running';}
    return {ok:true,json:async()=>({id:'test',status,plan:structuredClone(plan)})};
  };
  await c.agent.poll();
  plan={video:'clip.mp4',global_prompt:'agent edit',global_refs:['ref2.png']};
  await c.agent.poll();
  assert.equal(c.plan.global_prompt,'agent edit');
  assert.deepEqual(c.plan.global_refs,['ref2.png']);
  assert.equal('obsolete' in c.plan,false);
  status='paused';await c.agent.poll();
  c.plan.global_prompt='manual paused edit';c.plan.global_refs=[];
  await c.agent.poll();await c.agent.control('resume');await c.agent.poll();
  assert.equal(plan.global_prompt,'manual paused edit');
  assert.equal(c.plan.global_prompt,'manual paused edit');
  assert.deepEqual(c.plan.global_refs,[]);
  plan.global_prompt='agent after resume';await c.agent.poll();
  assert.equal(c.plan.global_prompt,'agent after resume');
});
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
  const fresh = find(tree, n => n.tag === 'button' && n.children === 'New task');
  assert.equal(resume.props.disabled, true);
  assert.equal(fresh.props.disabled, false);
});

test('feedback survives polling, sends separately from task, and clears only on success', async () => {
  const c = context();
  c.agent.a.status = 'complete';
  let sent;
  c.api.fetchApi = async (url, options) => {
    if(options)sent=JSON.parse(options.body);
    return {ok:true,json:async()=>({id:'test',status:'running',events:[],messages:[{id:'m',text:'Preserve the other person',kind:'feedback',status:'queued'}],pending_messages:['m']})};
  };
  const input=find(agentTab(c),n=>n.tag==='textarea'&&n.props['aria-label']==='Message to the agent');
  input.props.onInput({target:{value:'Preserve the other person'}});
  await c.agent.poll();
  assert.equal(c.agent.a.feedback,'Preserve the other person');
  c.agent.a.status='complete';
  assert.equal(find(agentTab(c),n=>n.tag==='button'&&n.children==='Send feedback and continue').props.disabled,false);
  await c.agent.sendFeedback();
  assert.equal(sent.kind,'feedback');
  assert.equal(sent.text,'Preserve the other person');
  assert.equal(c.agent.a.task,'existing task');
  assert.equal(c.agent.a.feedback,'');
  assert.equal(c.agent.a.messages.length,1);
  assert.equal(c.agent.a.pendingMessages.length,1);
  c.agent.a.feedback='Retry this correction';
  c.api.fetchApi=async()=>({ok:false,json:async()=>({error:'Wait for current action'})});
  await c.agent.sendFeedback();
  assert.equal(c.agent.a.feedback,'Retry this correction');
  assert.match(c.agent.a.error,/Wait for current action/);
});

test('chat accepts feedback during active work and cannot bypass stopped or invalid sessions',()=>{
  const c=context();c.agent.a.feedback='Fix only the target';
  for(const status of ['running','waiting_render']){
    c.agent.a.status=status;
    assert.equal(find(agentTab(c),n=>n.tag==='button'&&n.children==='Send feedback and continue').props.disabled,false);
  }
  c.agent.a.status='stopped';
  assert.equal(find(agentTab(c),n=>n.tag==='button'&&n.children==='Send feedback and continue').props.disabled,true);
  c.agent.a.status='paused';c.agent.a.requiresNewTask=true;
  assert.equal(find(agentTab(c),n=>n.tag==='button'&&n.children==='Send feedback and continue').props.disabled,true);
});

test('zero review fps survives polling and the UI prioritizes final review over stale previews',async()=>{
  const c=context();
  const label=find(agentTab(c),n=>n.tag==='label'&&n.children[0]==='Review fps (0 = every frame)');
  assert.equal(label.children[1].props.min,0);
  label.children[1].props.onInput({target:{value:'0'}});
  await c.agent.poll();
  assert.equal(c.agent.a.settings.review_fps,0);
  c.agent.a.view='chat';
  c.agent.a.preview={frames:[{src:'old-preview'}]};
  c.agent.a.observation={frames:[{src:'final-review',label:'generated frame 179'}]};
  const tree=agentTab(c);
  assert.ok(find(tree,n=>n.tag==='img'&&n.props.src==='final-review'));
  assert.equal(find(tree,n=>n.tag==='img'&&n.props.src==='old-preview'),undefined);
});

test('attachments travel with queued feedback and failed sends preserve the draft',async()=>{
  const c=context();c.agent.a.status='waiting_render';c.agent.a.feedback='Look at the right face';
  c.agent.a.attachments=[{name:'frame.png',src:'data:image/png;base64,test'}];
  let body;
  c.api.fetchApi=async(url,options)=>{body=JSON.parse(options.body);return{ok:false,json:async()=>({error:'connection lost'})};};
  await c.agent.sendFeedback();
  assert.equal(body.attachments[0].name,'frame.png');
  assert.equal(c.agent.a.attachments.length,1);
  assert.equal(c.agent.a.feedback,'Look at the right face');
});

test('resource snapshots update without changing a numeric setting draft',async()=>{
  const c=context();c.agent.a.settings.max_tokens='';
  c.api.fetchApi=async()=>({ok:true,json:async()=>({time:123,ram:{percent:40},gpu:null})});
  await c.agent.pollResources(true);
  assert.equal(c.agent.a.resources.ram.percent,40);
  assert.equal(c.agent.a.settings.max_tokens,'');
  assert.equal(c.agent.a.resources.gpu,null);
});

test('new task clears a stale invalid-workflow lock and old accounting',()=>{
  const c=context();c.agent.a.status='error';c.agent.a.requiresNewTask=true;c.agent.a.actions=80;
  c.agent.a.contextUsage={capacity:32768};c.agent.a.tokenUsage={calls:20};
  c.agent.newTask();
  assert.equal(c.agent.a.requiresNewTask,false);
  assert.equal(c.agent.a.id,'');
  assert.equal(c.agent.a.actions,0);
  assert.equal(c.agent.a.tokenUsage,null);
  c.agent.a.task='Retry the fixed workflow';
  assert.equal(find(agentTab(c),n=>n.tag==='button'&&n.children==='▶ Start automatic task').props.disabled,false);
});

test('typing the next message during a send does not lose the new draft',async()=>{
  const c=context();c.agent.a.feedback='First correction';
  let finish;
  c.api.fetchApi=async()=>{await new Promise(resolve=>{finish=resolve;});return{ok:true,json:async()=>({id:'test',status:'running',events:[]})};};
  const pending=c.agent.sendFeedback();
  c.agent.a.feedback='Next correction still being written';
  finish();await pending;
  assert.equal(c.agent.a.feedback,'Next correction still being written');
});

test('polling retains activity that rolls out of the server event window',async()=>{
  const c=context();
  c.agent.a.events=[{time:1,kind:'status',summary:'older'},{time:2,kind:'status',summary:'still current'}];
  c.api.fetchApi=async()=>({ok:true,json:async()=>({id:'test',status:'running',event_count:3,events:[{time:2,kind:'status',summary:'still current'},{time:3,kind:'status',summary:'new'}]})});
  await c.agent.poll();
  assert.equal(c.agent.a.historyEvents[0].summary,'older');
  assert.equal(c.agent.a.events.at(-1).summary,'new');
});

test('copy exports unloaded history and tool failures without changing the visible filter', async () => {
  const c=context();let copied='';
  Object.defineProperty(globalThis,'navigator',{configurable:true,value:{clipboard:{writeText:async text=>{copied=text;}}}});
  c.api.fetchApi=async url=>({ok:true,json:async()=>url.includes('/history/')?{before:0,events:[{time:1,kind:'action',tool:'segment',arguments:{shot:0}}]}:{id:'test',task:'Swap only one person',event_count:2,messages:[{time:1.5,text:'Preserve the other person',kind:'feedback'}],events:[{time:2,kind:'error',tool:'segment',error:'Example failure'}]}});
  await c.agent.copyTranscript('tools');
  assert.match(copied,/"shot": 0/);assert.match(copied,/Example failure/);
  assert.equal(c.agent.a.logFilter,'all');assert.equal(c.agent.a.copying,false);
  await c.agent.copyTranscript('conversation');assert.match(copied,/Preserve the other person/);
  await c.agent.copyTranscript('errors');assert.match(copied,/Example failure/);assert.doesNotMatch(copied,/"shot": 0/);
});

test('copy reports failed history retrieval instead of copying an incomplete transcript', async () => {
  const c=context();let writes=0;
  Object.defineProperty(globalThis,'navigator',{configurable:true,value:{clipboard:{writeText:async()=>{writes++;}}}});
  c.api.fetchApi=async url=>({ok:!url.includes('/history/'),json:async()=>url.includes('/history/')?{error:'History unavailable'}:{event_count:2,events:[]}});
  await c.agent.copyTranscript('all');assert.equal(writes,0);assert.match(c.agent.a.error,/History unavailable/);assert.equal(c.agent.a.copying,false);
});
