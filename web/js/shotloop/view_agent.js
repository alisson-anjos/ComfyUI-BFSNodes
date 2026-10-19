import { reactive, onMounted, onBeforeUnmount } from '../vendor/vue.esm-browser.prod.mjs';
import { h } from './common.js';

export function createAgent(c, io, app) {
  const a = reactive({ task:'', id:'', status:'idle', checklist:[], events:[], observation:{}, preview:null, vlmInfo:null, settings:{thinking:true,web:true,max_actions:40,max_renders:4,max_tokens:4096,project_root:''}, follow:true, error:'', working:false, combined:null });
  let timer = null, dead = false, polling = false;
  const endpoint = '/bfs/shotloop/agent';
  async function post(path, data) {
    const r=await c.api.fetchApi(endpoint+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const j=await r.json();if(!r.ok || j.error)throw new Error(j.error || r.statusText);return j;
  }
  function apply(j) {
    if(!a.task){a.task=j.task||'';Object.assign(a.settings,j.settings||{});}
    a.vlmInfo=j.vlm_info||null;
    a.checklist=j.checklist||[];a.id=j.id;a.status=j.status;a.events=j.events||[];a.observation=j.observation||{};a.preview=j.preview;a.combined=j.combined;
    // Paused manual edits are sent back to the server on Resume.
    if (j.plan && ['running','waiting_render','complete','budget_exhausted','error'].includes(j.status)) {
      Object.assign(c.plan,j.plan);c.save();
    }
    const last=[...(j.events||[])].reverse().find(e=>e.kind==='action');
    if(last && ['segment','preview_mask'].includes(last.tool) && j.observation?.frames) {
      const shot=c.segs.value[last.arguments?.shot];
      if(shot)c.maskPrev.value={...c.maskPrev.value,[`${shot.start}-${shot.end}`]:j.observation};
    }
    if (j.id) { c.plan.agent_session_id=j.id;c.save(); }
  }
  async function poll() {
    if(dead || !a.id || polling)return;
    polling=true;
    try {
      const r=await c.api.fetchApi(`${endpoint}/session/${encodeURIComponent(a.id)}`);const j=await r.json();
      if(!r.ok || j.error)throw new Error(j.error||r.statusText);apply(j);
    } catch(e){a.error=String(e.message||e);}finally{polling=false;}
  }
  async function start() {
    a.error='';a.working=true;
    try {
      if(['running','waiting_render'].includes(a.status))throw new Error('Pause or stop the current session first');
      c.save();const graph=await app.graphToPrompt();
      const plannerId=String(io.nodeId);
      if(graph.output?.[plannerId]?.class_type!=='BFSShotPlanner') {
        throw new Error('The current Planner is missing from the executable workflow. Enable this Planner and connect it to the generation/output chain, then try again.');
      }
      apply(await post('/start',{task:a.task,plan:JSON.parse(JSON.stringify(c.plan)),settings:{...a.settings},workflow:graph.output,planner_id:plannerId,client_id:c.api.clientId}));
    } catch(e){a.error=String(e.message||e);}finally{a.working=false;}
  }
  async function control(action) {
    a.error='';try{apply(await post('/control',{id:a.id,action,settings:{...a.settings},plan:JSON.parse(JSON.stringify(c.plan))}));}catch(e){a.error=String(e.message||e);}
  }
  const event=e=>{
    if(e.detail?.id!==a.id)return;
    const v=e.detail.event;a.events=[...a.events,v].slice(-100);
    if(a.follow && v.tab && v.tab!=='agent') {
      c.tab.value=v.tab;
      const i=v.arguments?.shot;if(Number.isInteger(i))c.sel.value=i;
    }
  };
  const preview=e=>{if(e.detail?.id===a.id)a.preview=e.detail.preview;};
  onMounted(()=>{
    a.id=c.plan.agent_session_id||'';poll();timer=setInterval(poll,1800);
    c.api.addEventListener('bfs-agent-event',event);c.api.addEventListener('bfs-agent-preview',preview);
  });
  onBeforeUnmount(()=>{dead=true;clearInterval(timer);c.api.removeEventListener('bfs-agent-event',event);c.api.removeEventListener('bfs-agent-preview',preview);});
  return {a,start,control,poll};
}

export function agentBar(c) {
  const {a,control}=c.agent;
  if(!a.id)return null;
  const last=[...a.events].reverse().find(e=>e.summary||e.error);
  return h('div',{style:'padding:8px;border:1px solid #7461b5;border-radius:8px'},[
    h('div',{class:'row',style:'gap:8px'},[
      h('button',{onClick:()=>{c.tab.value='agent';}},'🤖 '+a.status),
      h('span',{class:'grow'},last?.summary||last?.error||'Agent'),
      h('label',[h('input',{type:'checkbox',checked:a.follow,onChange:e=>{a.follow=e.target.checked;}}),' Follow tabs']),
      h('button',{onClick:()=>control('pause'),disabled:!['running','waiting_render'].includes(a.status)},'Pause'),
      h('button',{onClick:()=>control('resume'),disabled:!['paused','error','budget_exhausted'].includes(a.status)},'Resume'),
      h('button',{onClick:()=>control('stop'),disabled:['stopped','complete'].includes(a.status)},'Stop'),
    ]),
    a.preview ? h('details',{open:a.status==='waiting_render'},[
      h('summary',`Preview · ${a.preview.step}/${a.preview.total_steps} steps · approximate`),
      a.preview.error?h('div',{class:'err'},'Intermediate preview disabled; generation continues: '+a.preview.error):null,
      h('div',{class:'row'},(a.preview.frames||[]).map(f=>h('img',{src:f.src,title:`latent ${f.latent_index}`,style:'max-width:140px;max-height:100px'}))),
      a.preview.review?h('div',a.preview.review.reason||''):null,
    ]):null,
  ]);
}

export function agentTab(c) {
  const {a,start,control}=c.agent;
  const setting=(name,label,type='number')=>h('label',{style:'display:flex;gap:8px;align-items:center'},[
    label,h('input',{type,min:type==='number'?(name==='max_tokens'?512:1):undefined,max:type==='number'?({max_tokens:16384,max_actions:200,max_renders:20}[name]):undefined,value:a.settings[name],checked:!!a.settings[name],onInput:type==='checkbox'?undefined:e=>{a.settings[name]=type==='number'&&e.target.value!==''?Number(e.target.value):e.target.value;},onChange:type==='checkbox'?e=>{a.settings[name]=e.target.checked;}:undefined})]);
  const frames=a.preview?.frames||a.observation.frames||[];
  return h('div',{style:'padding:12px;display:grid;gap:12px'},[
    h('h3','Agent · experimental'),
    h('textarea',{value:a.task,placeholder:'Describe the complete video task, references, what must change and what must be preserved.',rows:5,style:'width:100%;box-sizing:border-box',onInput:e=>{a.task=e.target.value;}}),
    h('div',{class:'row',style:'gap:16px;flex-wrap:wrap'},[
      setting('thinking','Thinking','checkbox'),setting('web','Web context','checkbox'),
      setting('max_actions','Action limit'),setting('max_renders','Render limit'),setting('max_tokens','Decision token budget'),
    ]),
    setting('project_root','Project context folder (optional)','text'),
    a.vlmInfo?h('details',{open:true},[
      h('summary','Connected VLM · capabilities and memory'),
      h('div',`Checkpoint: ${a.vlmInfo.checkpoint||'not exposed'} · Architecture: ${a.vlmInfo.architecture||'unknown'}`),
      h('div',`Generation API: ${a.vlmInfo.generation?'available':'not detected'} · Images: ${a.vlmInfo.images||'unknown'} · Thinking control: ${a.vlmInfo.thinking_control||'unknown'}`),
      h('div',`Checkpoint Thinking capability: ${a.vlmInfo.thinking_checkpoint||'unknown'}`),
      h('div',`Backend context: ${a.vlmInfo.context_tokens??'unknown'} tokens · Agent output cap: ${a.vlmInfo.agent_output_cap??'unknown'} · Requested output: ${a.vlmInfo.requested_output_tokens??'unknown'}`),
      a.vlmInfo.context_source?h('div',{class:'hint'},a.vlmInfo.context_source):null,
      a.vlmInfo.finalization_reserve_tokens?h('div',{class:'hint'},`Thinking finalization reserve: ${a.vlmInfo.finalization_reserve_tokens} tokens within the decision budget; used only if the first call returns no final action.`):null,
      a.vlmInfo.gpu?h('div',`${a.vlmInfo.gpu.name}: ${a.vlmInfo.gpu.free_gib}/${a.vlmInfo.gpu.total_gib} GiB free · VLM weights not resident: ${a.vlmInfo.gpu.vlm_weights_not_resident_gib} GiB · Reserve: ${a.vlmInfo.gpu.reserve_gib} GiB`):null,
      a.vlmInfo.kv_estimate?h('div',[
        h('div',`Estimated total KV capacity: ${a.vlmInfo.kv_estimate.total_tokens} tokens · Requested output KV: ${a.vlmInfo.kv_estimate.requested_output_cache_mib} MiB`),
        h('div',{class:'hint'},a.vlmInfo.kv_estimate.assumptions),
      ]):null,
      ...(a.vlmInfo.warnings||[]).map(w=>h('div',{class:'hint'},w)),
    ]):h('div',{class:'hint'},'VLM capabilities and GPU memory diagnostics appear after Start loads the connected VLM.'),
    h('div',{class:'row'},[
      h('button',{onClick:start,disabled:a.working||!a.task.trim()||['running','waiting_render'].includes(a.status)},'▶ Start automatic task'),
      h('button',{onClick:()=>control('pause'),disabled:!a.id},'Pause'),
      h('button',{onClick:()=>control('resume'),disabled:!a.id},'Resume'),
      h('button',{onClick:()=>control('stop'),disabled:!a.id},'Stop'),
    ]),
    h('div',{class:'hint'},'Runs on the server, including when this tab is closed. Pause stops after the current tool; a queued render may finish. Stop cancels this session’s render. For intermediate H3 previews, connect BFS Agent Preview Sampler between sampler selection and SamplerCustom; session binding is automatic.'),
    a.checklist.length?h('ol',a.checklist.map(step=>h('li',`${{pending:'○',in_progress:'▶',done:'✓'}[step.status]} ${step.title}`))):null,
    a.error?h('div',{class:'err'},a.error):null,
    a.preview?h('div',`Intermediate preview · ${a.preview.step}/${a.preview.total_steps} steps · approximate`):null,
    a.preview?.error?h('div',{class:'err'},'Intermediate preview disabled; generation continues: '+a.preview.error):null,
    h('div',{class:'row',style:'flex-wrap:wrap'},frames.map(f=>h('figure',{style:'margin:4px;max-width:220px'},[
      h('img',{src:f.src,style:'max-width:220px;max-height:200px'}),h('figcaption',f.label||`latent ${f.latent_index??f.f??''}`)]))),
    a.preview?.review?h('pre',{style:'white-space:pre-wrap'},JSON.stringify(a.preview.review,null,2)):null,
    a.combined?h('a',{href:c.api.apiURL(`/view?filename=${encodeURIComponent(a.combined.filename)}&subfolder=${encodeURIComponent(a.combined.subfolder||'')}&type=output`),target:'_blank'},'Open assembled video'):null,
    h('details',{open:true},[h('summary','Activity and tool calls'),h('div',{style:'max-height:420px;overflow:auto'},a.events.slice(-60).reverse().map(e=>h('div',{style:'padding:6px;border-bottom:1px solid #444'},[
      h('b',`${new Date(e.time*1000).toLocaleTimeString()} · ${e.tool||e.kind}`),
      h('div',e.summary||e.error||''),
      e.arguments?h('pre',{style:'white-space:pre-wrap'},JSON.stringify(e.arguments,null,2)):null,
    ])))]),
    h('details',[h('summary','Latest observation'),h('pre',{style:'white-space:pre-wrap;max-height:300px;overflow:auto'},JSON.stringify(a.observation,(k,v)=>k==='src'?'[image]':v,2))]),
  ]);
}
