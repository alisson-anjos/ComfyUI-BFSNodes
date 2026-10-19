import { reactive, onMounted, onBeforeUnmount } from '../vendor/vue.esm-browser.prod.mjs';
import { h } from './common.js';
import { AGENT_CSS } from './agent_styles.js';

export function createAgent(c, io, app) {
  const a = reactive({ task:'', feedback:'', feedbackHistory:[], reviewCoverage:{}, messages:[], attachments:[], pendingMessages:[], contextUsage:null, contextSummary:null, tokenUsage:null, resources:null, resourceError:'', view:'chat', logFilter:'all', inputKind:'feedback', lightbox:null, followChat:true, actions:0, renders:0, created:0, copying:false, copyNotice:'', historyEvents:[], historyBefore:0, eventCount:0, pendingRender:null, id:'', status:'idle', checklist:[], events:[], observation:{}, preview:null, previewHistory:[], vlmInfo:null, requiresNewTask:false, settings:{thinking:true,web:true,max_actions:40,max_renders:4,max_tokens:4096,review_fps:2,context_window:32768,project_root:''}, follow:false, error:'', working:false, combined:null });
  let timer = null, dead = false, polling = false, resourcesAt = 0, resourcePolling = false;
  let fileInput = null;
  let remotePlan = null, remoteSession = null;
  const clone = value => JSON.parse(JSON.stringify(value));
  const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
  // Apply server changes only where the user has not edited the last server value.
  // Arrays are atomic: shot indices and reference order can change together.
  function mergePlan(local, previous, incoming) {
    for (const key of new Set([...Object.keys(previous), ...Object.keys(incoming)])) {
      if (same(previous[key], incoming[key])) continue;
      if (same(local[key], previous[key])) {
        if (Object.hasOwn(incoming, key)) local[key] = clone(incoming[key]);
        else delete local[key];
      } else if ([local[key], previous[key], incoming[key]].every(value => value && typeof value === 'object' && !Array.isArray(value))) {
        mergePlan(local[key], previous[key], incoming[key]);
      }
    }
  }
  const endpoint = '/bfs/shotloop/agent';
  async function post(path, data) {
    const r=await c.api.fetchApi(endpoint+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const j=await r.json();if(!r.ok || j.error)throw new Error(j.error || r.statusText);return j;
  }
  function apply(j) {
    if(a.id===j.id&&a.events.length&&j.events?.length){
      const key=e=>`${e.time}|${e.kind}|${e.action_id||''}`;
      const incoming=new Set(j.events.map(key));
      const rolled=a.events.filter(e=>!incoming.has(key(e))&&e.time<=j.events[0].time);
      if(rolled.length)a.historyEvents=[...a.historyEvents,...rolled];
    }
    if(!a.task){a.task=j.task||'';Object.assign(a.settings,j.settings||{});}
    if(a.id&&j.id&&a.id!==j.id){a.historyEvents=[];a.historyBefore=0;}
    if(!a.historyEvents.length)a.historyBefore=Math.max(0,(j.event_count||j.events?.length||0)-(j.events?.length||0));
    a.eventCount=j.event_count||j.events?.length||0;a.pendingRender=j.pending_render||null;
    a.messages=j.messages||[];a.pendingMessages=j.pending_messages||[];
    a.contextUsage=j.context_usage||null;a.contextSummary=j.context_summary||null;a.tokenUsage=j.token_usage||null;
    a.actions=j.actions??(j.events||[]).filter(e=>e.kind==='action').length;a.renders=j.renders||0;a.created=j.created||a.created;
    a.reviewCoverage=j.review_coverage||{};
    a.feedbackHistory=j.feedback||[];
    a.previewHistory=j.preview_history||[];
    a.vlmInfo=j.vlm_info||null;
    a.requiresNewTask=!!j.requires_new_task||String(j.observation?.error||'').includes('Workflow VAE loader rejected');
    a.checklist=j.checklist||[];a.id=j.id;a.status=j.status;a.events=j.events||[];a.observation=j.observation||{};a.preview=j.preview;a.combined=j.combined;
    // Paused manual edits are sent back to the server on Resume.
    if (j.plan) {
      const video=c.plan.video, refs=JSON.stringify(c.plan.global_refs);
      if (remotePlan && remoteSession === j.id && ['running','waiting_render','complete','budget_exhausted','error'].includes(j.status)) {
        mergePlan(c.plan,remotePlan,j.plan);
      }
      // On reconnect, the saved workflow is the user's current plan. An old
      // session snapshot establishes the baseline without replacing that plan.
      remotePlan=clone(j.plan);remoteSession=j.id;
      const sourceChanged=video!==c.plan.video;
      const refsChanged=refs!==JSON.stringify(c.plan.global_refs);
      c.save();
      if(sourceChanged||refsChanged)c.refreshFiles?.();
      if(sourceChanged)c.analyze?.();
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
      const plan=clone(c.plan);
      const result=await post('/start',{task:a.task,plan,attachments:a.attachments.map(({src,name})=>({src,name})),settings:{...a.settings},workflow:graph.output,planner_id:plannerId,client_id:c.api.clientId});
      remotePlan=plan;remoteSession=result.id;apply(result);
      a.attachments=[];a.followChat=true;
    } catch(e){a.error=String(e.message||e);}finally{a.working=false;}
  }
  async function sendFeedback() {
    a.error='';a.working=true;
    try {
      const text=a.feedback.trim(), files=[...a.attachments];
      if(!text&&!a.attachments.length)throw new Error('Write a message or attach a screenshot.');
      apply(await post('/message',{id:a.id,text,kind:a.inputKind,attachments:files.map(({src,name})=>({src,name})),settings:{...a.settings}}));
      if(a.feedback.trim()===text)a.feedback='';a.attachments=a.attachments.filter(f=>!files.includes(f));a.followChat=true;
    }catch(e){a.error=String(e.message||e);}finally{a.working=false;}
  }
  async function loadHistory(){
    if(!a.id||!a.historyBefore)return;
    try{const r=await c.api.fetchApi(`${endpoint}/history/${encodeURIComponent(a.id)}?before=${a.historyBefore}&count=100`);const j=await r.json();if(!r.ok||j.error)throw new Error(j.error||r.statusText);a.historyEvents=[...(j.events||[]),...a.historyEvents];a.historyBefore=j.before;}catch(e){a.error=String(e.message||e);}
  }
  async function copyTranscript(mode='conversation'){
    if(a.copying)return;
    a.copying=true;a.copyNotice='';a.error='';
    try{
      // Read a bounded snapshot of the entire persisted history, including unloaded pages.
      const sessionId=a.id;
      let snapshot={task:a.task,messages:a.messages,feedback:a.feedbackHistory,events:a.events,event_count:a.eventCount};
      if(sessionId){
        const r=await c.api.fetchApi(`${endpoint}/session/${encodeURIComponent(sessionId)}`);
        snapshot=await r.json();if(!r.ok||snapshot.error)throw new Error(snapshot.error||r.statusText);
      }
      let events=[...(snapshot.events||[])];
      let before=Math.max(0,(snapshot.event_count||events.length)-events.length);
      while(before>0){
        const r=await c.api.fetchApi(`${endpoint}/history/${encodeURIComponent(sessionId)}?before=${before}&count=100`);
        const page=await r.json();if(!r.ok||page.error)throw new Error(page.error||r.statusText);
        if(!Number.isInteger(page.before)||page.before<0||page.before>=before)throw new Error('Could not load the complete history.');
        events=[...(page.events||[]),...events];before=page.before;
      }
      const clean=(key,value)=>typeof value==='string'&&value.startsWith('data:image/')?'[image omitted]':value;
      const stamp=time=>time?new Date(time*1000).toISOString():'';
      const lines=[`BFS Agent · ${mode} · session ${sessionId||'not started'}`,`Task: ${snapshot.task||''}`];
      if(mode==='conversation'){
        const entries=(snapshot.messages||[]).map(m=>({time:m.time,text:`You (${m.kind||'feedback'}, ${m.status||'applied'}): ${m.text||''}`+(m.attachments?.length?'\nAttachments: '+m.attachments.map(f=>f.name||f.src).join(', '):'')}));
        for(const e of events){
          if(['action','reply','error','context'].includes(e.kind))entries.push({time:e.time,text:JSON.stringify(e,clean,2)});
          else if(e.kind==='feedback'&&!(snapshot.messages||[]).some(m=>m.text===e.summary))entries.push({time:e.time,text:'You: '+e.summary});
        }
        entries.sort((x,y)=>x.time-y.time);lines.push(...entries.map(e=>`[${stamp(e.time)}] ${e.text}`));
      }else{
        const selected=events.filter(e=>mode==='errors'?e.kind==='error':mode==='tools'?['action','observation'].includes(e.kind)||(e.kind==='error'&&e.tool):true);
        lines.push(...selected.map(e=>`[${stamp(e.time)}] ${JSON.stringify(e,clean,2)}`));
      }
      const text=lines.join('\n\n');
      try{await navigator.clipboard.writeText(text);}catch{
        const field=document.createElement('textarea');field.value=text;field.style.cssText='position:fixed;left:-10000px';document.body.appendChild(field);
        const focus=document.activeElement;
        try{field.select();if(!document.execCommand('copy'))throw new Error('Clipboard access failed. Allow clipboard access and try again.');}
        finally{field.remove();focus?.focus();}
      }
      a.copyNotice=mode==='conversation'?'Conversation copied':`${mode[0].toUpperCase()+mode.slice(1)} log copied`;
    }catch(e){a.error=String(e.message||e);}finally{a.copying=false;}
  }
  async function pollResources(force=false){
    if(dead||resourcePolling||(!force&&Date.now()-resourcesAt<5000))return;
    if(!force&&c.tab&&c.tab.value!=='agent')return;
    resourcePolling=true;
    try{const r=await c.api.fetchApi(endpoint+'/resources');const j=await r.json();if(!r.ok||j.error)throw new Error(j.error||r.statusText);a.resources=j;a.resourceError='';resourcesAt=Date.now();}
    catch(e){a.resourceError=String(e.message||e);}finally{resourcePolling=false;}
  }
  async function addFiles(files){
    a.error='';
    for(const file of Array.from(files||[])){
      if(a.attachments.length>=4){a.error='Attach up to four screenshots.';break;}
      if(!['image/png','image/jpeg','image/webp'].includes(file.type)){a.error='Choose a PNG, JPEG or WebP screenshot.';continue;}
      if(file.size>12*1024*1024){a.error='Screenshot file must be under 12 MB before resizing.';continue;}
      try{
        const bitmap=await createImageBitmap(file);
        if(bitmap.width*bitmap.height>24000000){bitmap.close();throw new Error('Screenshot dimensions are too large.');}
        const scale=Math.min(1,1280/Math.max(bitmap.width,bitmap.height));
        const canvas=document.createElement('canvas');canvas.width=Math.max(1,Math.round(bitmap.width*scale));canvas.height=Math.max(1,Math.round(bitmap.height*scale));
        canvas.getContext('2d').drawImage(bitmap,0,0,canvas.width,canvas.height);bitmap.close();
        a.attachments.push({name:file.name||'pasted screenshot',src:canvas.toDataURL('image/jpeg',.86)});
      }catch(e){a.error=String(e.message||e);}
    }
  }
  function paste(event){const files=Array.from(event.clipboardData?.files||[]).filter(f=>f.type.startsWith('image/'));if(files.length){event.preventDefault();event.stopPropagation();addFiles(files);}}
  function pickImages(){fileInput?.click();}
  function setFileInput(el){fileInput=el;}
  function newTask(){if(['running','waiting_render'].includes(a.status))return;a.id='';a.task='';a.feedback='';a.messages=[];a.events=[];a.attachments=[];a.preview=null;a.previewHistory=[];a.combined=null;a.reviewCoverage={};a.status='idle';a.requiresNewTask=false;a.error='';a.actions=0;a.renders=0;a.contextUsage=null;a.contextSummary=null;a.tokenUsage=null;a.created=0;a.historyEvents=[];a.historyBefore=0;a.pendingMessages=[];a.feedbackHistory=[];c.plan.agent_session_id='';c.save();}
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
    if(!document.getElementById('bfs-agent-styles')){const style=document.createElement('style');style.id='bfs-agent-styles';style.textContent=AGENT_CSS;document.head.appendChild(style);}
    pollResources(true);
    a.id=c.plan.agent_session_id||'';poll();timer=setInterval(()=>{poll();pollResources();},1800);
    c.api.addEventListener('bfs-agent-event',event);c.api.addEventListener('bfs-agent-preview',preview);
  });
  onBeforeUnmount(()=>{dead=true;clearInterval(timer);c.api.removeEventListener('bfs-agent-event',event);c.api.removeEventListener('bfs-agent-preview',preview);});
  return {a,start,control,poll,sendFeedback,pollResources,addFiles,paste,pickImages,setFileInput,newTask,loadHistory,copyTranscript};
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
      h('button',{onClick:()=>control('resume'),disabled:a.requiresNewTask||!['paused','error','budget_exhausted'].includes(a.status)},'Resume'),
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

const shortNumber = value => Number.isFinite(value) ? (value>=1000?`${(value/1000).toFixed(1)}k`:String(Math.round(value))) : '—';
const timeLabel = value => value ? new Date(value*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) : '';
const stateLabels = {idle:'Ready',running:'Working',waiting_render:'Rendering',paused:'Paused',error:'Needs attention',complete:'Complete',budget_exhausted:'Limit reached',stopped:'Stopped'};

function toolState(event, events) {
  const index=events.indexOf(event);
  const following=events.slice(index+1);
  const outcome=following.find(e=>event.action_id?e.action_id===event.action_id:['observation','error','action'].includes(e.kind));
  if(outcome?.kind==='error')return {label:'Failed',kind:'error',error:outcome.error};
  if(outcome?.kind==='observation')return {label:event.tool==='render'?'Queued':'Completed',kind:'success'};
  return {label:'Requested',kind:'pending'};
}
function eventCard(e, events) {
  const status=toolState(e,events);
  return h('details',{class:'ag-tool'},[
    h('summary',[h('span',e.tool||e.kind),h('span',{class:'ag-badge','data-kind':status.kind},status.label),h('small',{style:'margin-left:auto'},timeLabel(e.time))]),
    h('div',{class:'ag-tool-body'},[
      status.error?h('div',{class:'ag-error'},status.error):null,
      ...[['observed','Observed'],['reason','Reason'],['expected','Expected'],['uncertainty','Uncertainty']].filter(([key])=>e[key]).map(([key,label])=>h('p',[h('strong',label+': '),e[key]])),
      e.arguments?h('pre',JSON.stringify(e.arguments,null,2)):null,
    ]),
  ]);
}
export function agentTab(c) {
  const agent=c.agent, {a,start,control,sendFeedback}=agent;
  const url=src=>src?.startsWith('/view?')?c.api.apiURL(src):src;
  const running=['running','waiting_render'].includes(a.status);
  const settingsLabel=(name,label,type='number')=>h('label',[
    label,h('input',{type,min:type==='number'?(name==='review_fps'?0:name==='max_tokens'?512:name==='context_window'?8192:1):undefined,
      max:type==='number'?({max_tokens:16384,max_actions:200,max_renders:20,review_fps:24,context_window:262144}[name]):undefined,
      value:a.settings[name],checked:!!a.settings[name],
      onInput:type==='checkbox'?undefined:e=>{a.settings[name]=type==='number'&&e.target.value!==''?Number(e.target.value):e.target.value;},
      onChange:type==='checkbox'?e=>{a.settings[name]=e.target.checked;}:undefined}),
  ]);
  const badge=h('span',{class:'ag-state','data-state':a.status,role:'status'},[h('span',{class:'ag-dot'}),stateLabels[a.status]||a.status]);
  const context=a.contextUsage, inputTokens=context?(context.text_tokens+context.image_tokens):null;
  const gpu=a.resources?.gpu||a.vlmInfo?.gpu;
  const ram=a.resources?.ram;
  const percent=(used,total)=>Number.isFinite(used)&&total>0?Math.min(100,Math.max(0,used/total*100)):0;
  const meter=(used,total)=>h('div',{class:'ag-meter','data-pressure':used/total>.8?'high':'normal'},[h('span',{style:{width:percent(used,total)+'%'}})]);
  const card=(label,value,note,bar)=>h('div',{class:'ag-resource'},[h('div',{class:'ag-resource-label'},label),h('div',{class:'ag-resource-value'},value),bar,h('div',{class:'ag-resource-note'},note)]);
  const events=[...(a.historyEvents||[]),...a.events].filter((e,i,all)=>all.findIndex(v=>v.time===e.time&&v.kind===e.kind&&v.action_id===e.action_id)===i);
  const frames=a.observation.frames||a.preview?.frames||[];
  const frameGrid=items=>h('div',{class:'ag-frames'},items.map((f,i)=>h('figure',{key:f.src||i},[
    h('img',{src:url(f.src),loading:'lazy',alt:f.label||'Video preview',tabindex:0,onClick:()=>{a.lightbox=url(f.src);},onKeydown:e=>{if(e.key==='Enter')a.lightbox=url(f.src);}}),
    h('figcaption',f.label||`Latent ${f.latent_index??i}`),
  ])));
  const message=(role,text,time,extra,key)=>h('article',{class:'ag-message','data-role':role,key},[
    h('div',{class:'ag-avatar','aria-hidden':'true'},role==='user'?'U':'B'),
    h('div',{class:'ag-bubble'},[h('div',{class:'ag-message-meta'},[h('strong',role==='user'?'You':'Planner'),h('span',timeLabel(time))]),
      text?h('div',{class:'ag-message-text'},text):null,extra]),
  ]);
  const items=[];
  if(a.id&&a.task)items.push({time:a.created||events[0]?.time||1,node:message('user',a.task,a.created,null,'task')});
  for(const m of a.messages)items.push({time:m.time,node:message('user',m.text,m.time,h('div',[
    m.status==='queued'?h('span',{class:'ag-pending'},'Queued · next safe step'):null,
    m.attachments?.length?frameGrid(m.attachments.map(f=>({src:f.src,label:f.name}))):null,
  ]),m.id)});
  for(const e of events){
    if(e.kind==='action')items.push({time:e.time,node:message('assistant',e.summary,e.time,eventCard(e,events),e.action_id||e.time)});
    else if(e.kind==='reply')items.push({time:e.time,node:message('assistant',e.summary,e.time,null,'reply'+e.time)});
    else if(e.kind==='feedback'&&!a.messages.some(m=>m.kind==='feedback'&&m.text===e.summary))items.push({time:e.time,node:message('user',e.summary,e.time,null,'feedback'+e.time)});
    else if(e.kind==='error'&&!events.some(v=>v.kind==='action'&&v.action_id&&v.action_id===e.action_id))items.push({time:e.time,node:message('assistant',null,e.time,h('div',{class:'ag-error'},e.error),'error'+e.time)});
    else if(e.kind==='context')items.push({time:e.time,node:h('div',{class:'ag-note',key:'context'+e.time},e.summary)});
  }
  items.sort((x,y)=>x.time-y.time);
  const last=events.at(-1);
  const phase=a.status==='waiting_render'?(a.pendingRender?.kind==='context'?'Loading the connected VLM…':'Rendering the current plan…'):
    last?.phase==='finalize_action'?'Finalizing the next action…':last?.kind==='action'?`Using ${last.tool}…`:'Reviewing evidence and choosing the next step…';
  const conversation=h('div',{class:'ag-thread',role:'log','aria-label':'Agent conversation',
    ref:el=>{if(el&&a.followChat)queueMicrotask(()=>{el.scrollTop=el.scrollHeight;});},
    onScroll:e=>{const el=e.target;a.followChat=el.scrollHeight-el.scrollTop-el.clientHeight<80;}},[
    a.historyBefore>0?h('button',{class:'ag-scroll-bottom',onClick:agent.loadHistory},'Load earlier activity'):null,
    !a.id?h('div',{class:'ag-empty'},[h('div',{class:'ag-logo',style:'margin:auto'},'↗'),h('h3','What would you like to change?'),
      h('p','Describe the edit and what must stay untouched. The Planner will inspect, edit, render and review while you follow along.'),
      h('div',{class:'ag-suggestions'},[
        h('button',{onClick:()=>{a.task='Replace only the person on the left with <Picture 1>. Preserve the other people, background and original motion. Verify the masks and results across the video.';}},'Swap one character'),
        h('button',{onClick:()=>{a.task='Compare supported sampler and scheduler settings for this shot. Keep the seed and references fixed, save the candidates and select the strongest result based on identity and motion.';}},'Compare generation settings'),
      ])]):null,
    ...items.map(i=>i.node),
    running?h('div',{class:'ag-live',role:'status'},[h('span',{class:'ag-live-dot'}),phase]):null,
    !a.followChat?h('button',{class:'ag-scroll-bottom',onClick:()=>{a.followChat=true;}},'Jump to latest'):null,
  ]);
  const settings=h('div',{class:'ag-content'},[
    h('div',{class:'ag-settings'},[
      settingsLabel('thinking','Thinking','checkbox'),settingsLabel('web','Web context','checkbox'),
      settingsLabel('review_fps','Review fps (0 = every frame)'),settingsLabel('max_actions','Action limit'),settingsLabel('max_renders','Render limit'),
      settingsLabel('max_tokens','Decision token budget'),settingsLabel('context_window','Context window (working limit)'),settingsLabel('project_root','Project context folder (optional)','text'),
    ]),
    h('p',{class:'ag-note',style:'margin-top:14px'},'Changes apply at the next message or Resume. The context window is a working budget, not a model capability upgrade. Dense frame review takes more actions. Your workflow must already have generation and output nodes connected.'),
    a.vlmInfo?h('details',{class:'ag-details'},[h('summary','Connected VLM · capabilities and memory'),h('div',{class:'ag-detail-content'},[
      h('p',`Checkpoint: ${a.vlmInfo.checkpoint||'not exposed'} · ${a.vlmInfo.architecture||'unknown architecture'}`),
      h('p',`Images: ${a.vlmInfo.images||'unknown'} · Thinking: ${a.vlmInfo.thinking_control||'unknown'}`),
      h('p',`Backend context: ${a.vlmInfo.context_tokens??'unknown'} tokens. ${a.vlmInfo.context_source||''}`),
      ...(a.vlmInfo.warnings||[]).map(w=>h('p',{class:'ag-muted'},w)),
    ])]):h('p',{class:'ag-note'},'VLM capabilities appear after the connected model is loaded.'),
    a.resources?h('details',{class:'ag-details'},[h('summary','Latest resource snapshot'),h('div',{class:'ag-detail-content'},[h('pre',JSON.stringify(a.resources,null,2))])]):null,
    a.contextUsage?h('details',{class:'ag-details'},[h('summary','Context and token accounting'),h('div',{class:'ag-detail-content'},[
      h('p','Session totals accumulate across calls; context usage is only the current decision. Counts are estimates, not billing or measured KV memory.'),h('pre',JSON.stringify({context:a.contextUsage,session:a.tokenUsage},null,2)),
    ])]):null,
    a.contextSummary?h('details',{class:'ag-details'},[h('summary',`Saved context summary · ${a.contextSummary.compactions} compactions`),h('div',{class:'ag-detail-content'},[h('p',a.contextSummary.note),h('pre',JSON.stringify(a.contextSummary.facts,null,2))])]):null,
    h('label',{class:'ag-row'},[h('input',{type:'checkbox',checked:a.follow,onChange:e=>{a.follow=e.target.checked;}}),'Follow tool tabs']),
  ]);
  const trace=h('div',{class:'ag-content'},[
    h('div',{class:'ag-log-filter'},['all','tools','errors'].map(filter=>h('button',{'aria-pressed':a.logFilter===filter,onClick:()=>{a.logFilter=filter;}},filter[0].toUpperCase()+filter.slice(1))).concat([h('button',{disabled:a.copying,onClick:()=>agent.copyTranscript(a.logFilter)},a.copying?'Copying…':'Copy log')])),
    h('div',{class:'ag-log'},events.filter(e=>a.logFilter==='errors'?e.kind==='error':a.logFilter==='tools'?e.kind==='action':!['decision','observation'].includes(e.kind)).slice().reverse().map(e=>
      e.kind==='action'?eventCard(e,events):h('div',{class:e.kind==='error'?'ag-error':'ag-note'},[h('small',`${timeLabel(e.time)} · ${e.kind}`),h('div',e.summary||e.error||JSON.stringify(e.review||{}))]))),
    a.historyBefore>0?h('button',{onClick:agent.loadHistory},'Load earlier activity'):null,
    h('details',{class:'ag-details',style:'margin-top:12px'},[h('summary','Latest tool observation'),h('div',{class:'ag-detail-content'},[h('pre',JSON.stringify(a.observation,(key,value)=>key==='src'?'[image]':value,2))])]),
  ]);
  const evidence=h('div',{class:'ag-content'},[
    frames.length?h('details',{class:'ag-details',open:true},[h('summary',a.observation.frames?'Current result inspection':'Intermediate preview · approximate'),h('div',{class:'ag-detail-content'},frameGrid(frames))]):null,
    ...Object.entries(a.reviewCoverage).map(([shot,v])=>h('div',{class:'ag-note'},`Shot ${shot}: ${v.covered_count}/${v.required_count} required frames presented${v.complete?' · coverage complete':''}`)),
    ...a.previewHistory.slice().reverse().map(t=>h('details',{class:'ag-details'},[
      h('summary',`Render ${t.render} · ${String(t.status).replaceAll('_',' ')}`),h('div',{class:'ag-detail-content'},[
        t.reason?h('p',t.reason):null,...(t.user_feedback||[]).map(f=>h('p','Your feedback: '+f.text)),
        ...(t.captures||[]).map(v=>h('div',[h('p',{class:'ag-evidence-title'},`Shot ${v.shot??'?'} · step ${v.step}/${v.total_steps} · approximate`),frameGrid(v.frames||[]),v.review?h('p',v.review.reason||''):null])),
        !t.captures?.length?h('p',{class:'ag-muted'},'No preview was captured before this render ended.'):null,
        t.generation_nodes?h('details',[h('summary','Generation settings'),h('pre',JSON.stringify(t.generation_nodes,null,2))]):null,
        t.render_status?.status_str!=='success'&&t.render_status?h('details',[h('summary','Render error details'),h('pre',JSON.stringify(t.render_status,null,2))]):null,
      ]),
    ])),
    !frames.length&&!a.previewHistory.length?h('p',{class:'ag-muted'},'Captured previews and result inspections appear here.'):null,
  ]);
  const plan=h('div',{class:'ag-content'},[
    a.checklist.length?h('ol',{class:'ag-checklist'},a.checklist.map(step=>h('li',{'data-status':step.status},[h('span',({pending:'○',in_progress:'↗',done:'✓'})[step.status]),h('span',step.title)]))):h('p',{class:'ag-muted'},'The agent will publish its plan after inspecting your task.'),
    h('p',{class:'ag-note',style:'margin-top:14px'},'The agent continues through tools and review within the action/render limits. Pause waits for the current tool; a queued render may finish. Stop cancels this session’s render.'),
  ]);
  const draft=a.id?a.feedback:a.task;
  const canSend=!!draft.trim()||a.attachments.length>0;
  const send=()=>a.id?sendFeedback():start();
  const composer=h('div',{class:'ag-composer-wrap'},[
    h('div',{class:'ag-composer',onDragover:e=>{e.preventDefault();e.stopPropagation();},onDrop:e=>{e.preventDefault();e.stopPropagation();agent.addFiles(e.dataTransfer.files);}},[
      h('textarea',{value:draft,rows:3,maxLength:a.id?4000:16000,'aria-label':a.id?'Message to the agent':'Video task',
        placeholder:a.id?'Describe what went wrong, ask a question, or paste a screenshot…':'Describe your edit. Optional paths: Guide: /path/video.mp4 and Ref 1: /path/reference.png on separate lines.',
        onInput:e=>{if(a.id)a.feedback=e.target.value;else a.task=e.target.value;},onPaste:agent.paste,
        onKeydown:e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)&&canSend&&!a.working){e.preventDefault();e.stopPropagation();send();}}}),
      a.attachments.length?h('div',{class:'ag-attachments'},a.attachments.map((f,i)=>h('div',{class:'ag-attachment',key:f.src},[
        h('img',{src:f.src,alt:f.name}),h('button',{'aria-label':'Remove '+f.name,onClick:()=>{a.attachments.splice(i,1);}},'×'),
      ]))):null,
      h('div',{class:'ag-compose-actions'},[
        h('div',{class:'ag-compose-left'},[
          h('input',{type:'file',multiple:true,accept:'image/png,image/jpeg,image/webp',style:'display:none',ref:agent.setFileInput,onChange:e=>{agent.addFiles(e.target.files);e.target.value='';}}),
          h('button',{onClick:agent.pickImages,title:'Attach or paste screenshots'},'+ Screenshot'),
          a.id?h('select',{value:a.inputKind,'aria-label':'Message type',onChange:e=>{a.inputKind=e.target.value;}},[h('option',{value:'feedback'},'Correction'),h('option',{value:'question'},'Question')]):null,
        ]),
        h('button',{class:'ag-primary',onClick:send,disabled:a.working||!canSend||a.requiresNewTask||(!!a.id&&a.status==='stopped')},a.id?'Send feedback and continue':'▶ Start automatic task'),
      ]),
    ]),
    h('div',{class:'ag-compose-hint'},a.id?'Messages are applied at the next tool boundary. The agent keeps working. Ctrl/⌘ + Enter to send.':'Use paths on the ComfyUI server. Screenshots can be attached, dropped or pasted. Ctrl/⌘ + Enter to start.'),
  ]);
  return h('section',{class:'bfs-agent','aria-label':'Planner Agent'},[
    h('header',{class:'ag-header'},[
      h('div',{class:'ag-brand'},[h('div',{class:'ag-logo','aria-hidden':'true'},'↗'),h('div',[h('div',{class:'ag-title'},'Planner Agent'),h('div',{class:'ag-subtitle'},'Experimental · edits, renders and reviews')])]),
      h('div',{class:'ag-controls'},[badge,
        h('button',{onClick:()=>control('pause'),disabled:!a.id||!running},'Pause'),
        h('button',{onClick:()=>control('resume'),disabled:!a.id||a.requiresNewTask||!['paused','error','budget_exhausted'].includes(a.status)},'Resume'),
        h('button',{class:'ag-danger',onClick:()=>control('stop'),disabled:!a.id||['stopped','complete'].includes(a.status)},'Stop'),
        h('button',{onClick:agent.newTask,disabled:running||a.working},'New task'),
      ]),
    ]),
    h('div',{class:'ag-resources'},[
      card('GPU / memory',gpu?`${gpu.free_gib} / ${gpu.total_gib} GiB free`:'Not measured',gpu?`${gpu.name}${!a.resources?.gpu?' · at VLM load':''}`:ram?`RAM ${ram.used_gib}/${ram.total_gib} GiB`:'Waiting for resource snapshot',gpu?meter(gpu.total_gib-gpu.free_gib,gpu.total_gib):null),
      card('Context · estimated',context?`${shortNumber(inputTokens)} / ${shortNumber(context.capacity)}`:'Not measured',context?`${shortNumber(context.output_tokens)} reserved output · ${context.compactions||0} summaries`:'Available after the first decision',context?meter(inputTokens+context.output_tokens,context.capacity):null),
      card('Session tokens · estimated',a.tokenUsage?shortNumber(a.tokenUsage.input_tokens+a.tokenUsage.output_tokens)+(a.tokenUsage.unreported_output_calls?'+':''):'—',a.tokenUsage?`${shortNumber(a.tokenUsage.input_tokens)} in · ${shortNumber(a.tokenUsage.output_tokens)} out · ${a.tokenUsage.calls} tracked calls`:'Totals accumulate across decisions',null),
      card('Execution budget',`${a.actions} / ${a.settings.max_actions||'—'} actions`,`${a.renders} / ${a.settings.max_renders||'—'} renders · ${ram?`RAM ${ram.percent}%`:'RAM unknown'}${a.resources?.disk?` · ${a.resources.disk.free_gib} GiB disk free`:''}`,meter(a.actions,Number(a.settings.max_actions))),
    ]),
    h('nav',{class:'ag-nav',role:'tablist','aria-label':'Agent views'},[
      ...[['chat','Conversation'],['plan','Plan'],['renders','Renders'],['activity','Activity'],['settings','Settings']].map(([view,label])=>h('button',{class:'ag-tab',role:'tab','aria-selected':a.view===view,onClick:()=>{a.view=view;}},label)),
      h('span',{class:'ag-nav-spacer'}),h('button',{class:'ag-tab',disabled:a.copying,onClick:()=>agent.copyTranscript('conversation')},a.copying?'Copying…':'Copy conversation'),a.pendingMessages.length?h('span',{class:'ag-pending'},`${a.pendingMessages.length} queued`):null,
      h('button',{class:'ag-tab',onClick:()=>agent.pollResources(true),title:a.resourceError||'Refresh resource snapshot'},a.resourceError?'↻ Unavailable':a.resources?.time?`↻ ${timeLabel(a.resources.time)}`:'↻ Resources'),
    ]),
    a.requiresNewTask?h('div',{class:'ag-error',style:'margin:12px 22px'},'Fix the VAE loader/connections, then start a new task. Resume uses the old workflow snapshot.'):null,
    a.copyNotice?h('div',{class:'ag-note',role:'status',style:'margin:8px 22px'},a.copyNotice):null,
    a.error?h('div',{class:'ag-error',role:'alert',style:'margin:12px 22px'},a.error):null,
    a.view==='chat'?conversation:a.view==='settings'?settings:a.view==='activity'?trace:a.view==='renders'?evidence:plan,
    a.view==='chat'&&frames.length?h('details',{class:'ag-details',style:'margin:0 22px 14px'},[h('summary',a.observation.frames?'Latest result inspection':'Latest approximate preview'),h('div',{class:'ag-detail-content'},frameGrid(frames))]):null,
    a.combined?h('div',{class:'ag-result'},[h('span','Assembled video ready to inspect'),h('a',{href:c.api.apiURL(`/view?filename=${encodeURIComponent(a.combined.filename)}&subfolder=${encodeURIComponent(a.combined.subfolder||'')}&type=output`),target:'_blank',rel:'noopener'},'Open video ↗')]):null,
    composer,
    a.lightbox?h('div',{class:'ag-lightbox',role:'dialog','aria-modal':'true','aria-label':'Screenshot or frame',tabindex:-1,ref:el=>el?.focus(),onClick:()=>{a.lightbox=null;},onKeydown:e=>{if(e.key==='Escape')a.lightbox=null;}},[h('button',{onClick:()=>{a.lightbox=null;}},'Close'),h('img',{src:a.lightbox,alt:'Expanded screenshot or frame'})]):null,
  ]);
}
