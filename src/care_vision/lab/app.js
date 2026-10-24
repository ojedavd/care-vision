'use strict';
const root = document.getElementById('care-app');
const el = id => root.querySelector('#' + id);
const escapeText = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (n, digits = 1) => typeof n === 'number' && Number.isFinite(n) ? n.toFixed(digits) : '—';
const running = s => ['STARTING', 'RUNNING'].includes(s.session_state);
const empty = () => ({session_state:'IDLE',agent_state:'UNKNOWN',runtime:{},counts:{},metrics:{},edge:{metrics:{},series:[]},commands:[],transitions:[],timeline:[],observation:{},vision_live:false});
let live = empty(), perspective = 'lab', mode = 'live', scenario = 'recovery', step = 0, attention = 'unassigned', selectedCommand = null, polling = false, busy = false, csrf = '';
const titles = {
  lab:['SYSTEM EXPLORER','See the loop in action.','Follow a frame from capture to the next agent decision.'],
  caregiver:['CARE WORKSPACE','Know what needs your attention.','Clear evidence, a clear next step, and a visible owner.'],
  family:['FAMILY WORKSPACE','Stay informed, without the noise.','Relevant updates and who is taking care of them.'],
  operations:['SERVICE OPERATIONS','Keep every connection visible.','Separate service health from events that need human review.'],
};
function demoState() {
  const states = ['NORMAL','SUSPICIOUS','VERIFYING','VERIFYING','VERIFYING','RECOVERY'];
  const s = empty();s.session_state='WALKTHROUGH';s.agent_state=states[step];s.runtime={phase:1,cool_verified:false};
  s.edge={metrics:{capture_fps:30,configured_capture_fps:30,transmit_fps:step<3?2:step===3?null:step===5?2:20,target_transmit_fps:step>=2&&step<5?20:2,buffer_frames:300,buffer_bytes:null,network:'GOOD'},series:[],fresh:true,preview_capture_seq:null};
  s.vision_live=true;
  s.commands=step<2?[]:[{index:1,action:'SET_TRANSMIT_FPS',requested_fps:20,source_seen:true,ack_applied:step>=3,new_perception:step>=4,reason:'Sustained motion requires denser visual evidence; request 20 FPS.',source:{capture_seq:241,motion_score:null}}];
  s.transitions=[{state:s.agent_state,reason:['Baseline visual sampling.','A change in the example requires more evidence.','The command is sent; application has not been confirmed.','Applied ACK received; new evidence is still required.','Post-command evidence was processed.','The example shows recovery; return to baseline.'][step]}];
  if(step===5 && scenario==='review'){s.agent_state='ASSISTANCE_REQUIRED';s.transitions=[{state:s.agent_state,reason:'This scripted episode remains unresolved and requests human review.'}];}
  if(step>=3 && scenario==='offline'){s.agent_state='UNKNOWN';s.vision_live=false;s.edge.fresh=false;s.edge.metrics.network='OFFLINE';s.edge.metrics.transmit_fps=null;s.commands[0].ack_applied=false;s.commands[0].new_perception=false;s.transitions=[{state:'UNKNOWN',reason:'Connection interrupted. Cloud analysis is unavailable; current human activity is unknown.'}];}
  return s;
}
const data = () => mode==='walkthrough'?demoState():live;
function status(s) {
  if(mode==='live' && !running(s))return {tone:'neutral',label:s.session_state==='IDLE'?'Ready to start':'Run '+s.session_state.toLowerCase()};
  if(!s.vision_live)return {tone:'warning',label:mode==='walkthrough'?'Monitoring interrupted':'Waiting for fresh evidence'};
  if(s.agent_state==='ASSISTANCE_REQUIRED')return {tone:'danger',label:'Human review needed'};
  if(['SUSPICIOUS','VERIFYING'].includes(s.agent_state))return {tone:'warning',label:'Verifying a change'};
  return {tone:'good',label:mode==='walkthrough'?'No open review request':'Evidence arriving'};
}
const badge = (label,tone='neutral') => `<span class="badge ${tone}">${escapeText(label)}</span>`;
const metric = (label,value,unit,context='') => `<div class="metric"><span class="metric-label">${label}</span><span class="metric-value">${escapeText(value)}<span class="metric-unit">${unit}</span></span>${context?`<span class="metric-context">${context}</span>`:''}</div>`;
const pair = (label,value) => `<div><dt>${label}</dt><dd>${escapeText(value)}</dd></div>`;
function chartHTML() {return '<svg id="rate-chart" class="chart" role="img" aria-label="Measured local capture and transmission rates over time"></svg><div class="legend"><span><i></i>Capture</span><span><i class="requested"></i>Requested transmit</span><span><i class="transmitted"></i>Measured transmit</span></div>';}
function lab(s) {
  const e=s.edge.metrics, o=s.observation, st=status(s), commands=s.commands||[], latest=commands.find(c=>c.index===selectedCommand)||commands.filter(c=>c.action!=='REQUEST_RECENT_BUFFER').at(-1)||commands.at(-1);
  const reason=latest?.reason||s.transitions?.at(-1)?.reason||'Start a run to see the motion policy respond to generated visual evidence.';
  const stages=[['Source seen',latest?.source_seen],['Command sent',!!latest],['ACK applied',latest?.ack_applied],[latest?.action==='REQUEST_RECENT_BUFFER'?'History processed':'Perceived again',latest?.action==='REQUEST_RECENT_BUFFER'?latest?.history_complete:latest?.new_perception]];
  const preview=mode==='walkthrough'?'<div class="preview-empty"><strong>Scenario walkthrough</strong><span>Scripted example · No visual inference is running</span></div>':s.edge.preview_capture_seq?'<img id="source-preview" src="/api/preview.jpg" alt="Generated moving-rectangle source at the latest transmitted capture">':'<div class="preview-empty"><strong>Your synthetic scene will appear here</strong><span>No camera, model download or AWS account required</span></div>';
  return `<div class="grid-two"><section class="panel panel-body"><div class="panel-title"><h2>From the Edge</h2>${badge(s.edge.fresh?'Recent telemetry':'No current telemetry',s.edge.fresh?'info':'neutral')}</div><div class="preview-shell">${preview}</div><div class="preview-caption"><span class="caption">Generated source preview · ${s.edge.preview_capture_seq?'capture #'+fmt(s.edge.preview_capture_seq,0):'640 × 360'}</span><span class="caption">Reconstructed source frame; not transmitted pixels</span></div><div class="metrics-row">${metric('Capture',fmt(e.capture_fps),'FPS','Measured capture window')}${metric('Transmit',fmt(e.transmit_fps),'FPS','Measured live-frame send rate')}${metric('Requested',fmt(e.target_transmit_fps),'FPS','Requested rate ≠ achieved rate')}</div></section><section class="panel panel-body"><div class="panel-title"><h2>The agent’s decision</h2>${badge(st.label,st.tone)}</div><p class="state-label">Current policy state</p><p class="agent-state">${escapeText(s.agent_state)}</p><p class="reason">${escapeText(reason)}</p><div class="command-picker" aria-label="Inspect a command">${commands.map(c=>`<button type="button" data-command="${c.index}" aria-pressed="${latest?.index===c.index}">Command ${c.index}</button>`).join('')}</div><div class="action-box"><span class="caption">${latest?'Selected command':'Next action'}</span><code>${latest?escapeText(latest.action)+(typeof latest.requested_fps==='number'?'('+latest.requested_fps+')':''):'KEEP_OBSERVING'}</code>${latest?.source?.capture_seq?`<span class="caption">Caused by source capture #${fmt(latest.source.capture_seq,0)}</span>`:''}</div><div class="chain">${stages.map(([label,done])=>`<div class="chain-step ${done?'done':''}">${done?'✓ ':''}${label}</div>`).join('')}</div><div class="notice">${latest?.action==='REQUEST_RECENT_BUFFER'?'Historical completion is verified separately from a new live-perception cycle.':'An applied ACK is not a completed cycle. Later valid, command-linked evidence must be processed.'}</div><dl class="data-list">${pair('Motion score',fmt(o.motion_score,4))}${pair('Observation validity',typeof o.valid==='boolean'?(o.valid?'Valid':'Insufficient context'):'Not available')}${pair('Vision processing',fmt(s.metrics.processing_ms)+' ms')}${pair('Runtime',s.runtime.opencv?'OpenCV '+s.runtime.opencv:'Not started')}</dl></section></div><div class="grid-two section-space"><section class="panel panel-body"><div class="panel-title"><h2>Capture stays steady. Sending adapts.</h2>${badge(mode==='walkthrough'?'No measured data':'Observed samples','info')}</div>${chartHTML()}<dl class="data-list">${pair('Frames captured',fmt(e.captured,0))}${pair('Live frames sent',fmt(e.sent,0))}${pair('Historical frames sent',fmt(e.historical_sent,0))}${pair('Received / processed',fmt(s.counts.received,0)+' / '+fmt(s.counts.processed,0))}${pair('Application bytes sent',typeof e.bytes_sent==='number'?fmt(e.bytes_sent/1024)+' KiB':'—')}${pair('Ring buffer',typeof e.buffer_frames==='number'?fmt(e.buffer_frames,0)+' frames':'—')}${pair('Measured network quality',e.network||'UNKNOWN')}${pair('Evidence freshness',s.vision_live?'Recent observations':'Not currently live')}</dl><p class="notice">${s.edge.fresh?'Rates come from Edge telemetry; cloud processing cadence is a separate measurement.':'Values are unavailable or retained from the last sample; they do not indicate an active stream.'}</p></section><section class="panel panel-body"><div class="panel-title"><h2>Policy activity</h2>${badge(fmt(s.causal_cycles||0,0)+' causal cycles','neutral')}</div><div class="timeline">${(s.transitions||[]).length?s.transitions.slice(-7).reverse().map(t=>`<div class="timeline-item"><span class="timeline-dot" aria-hidden="true"></span><div><strong>${escapeText(t.state)}</strong><p>${escapeText(t.reason)}</p></div></div>`).join(''):'<p class="subtext">State changes will appear here after the run starts.</p>'}</div>${s.report?`<div class="notice"><strong>${s.report.passed?'Local loop verified':'Loop verification failed'}</strong><br>${fmt(s.report.frames_sent,0)} sent · ${fmt(s.report.frames_received,0)} received · ${fmt(s.report.observations,0)} observations · Buffer ${s.report.buffer_verified?'verified':'not verified'}</div>`:''}<div class="notice">Motion-only fixture. Person detection, posture confidence, fall accuracy and COOL performance are not evaluated here.</div></section></div>`;
}
function careCase(s) {
  const sim=mode==='walkthrough', offline=sim&&scenario==='offline'&&step>=3, review=sim&&scenario==='review'&&step===5, verifying=s.vision_live&&['SUSPICIOUS','VERIFYING'].includes(s.agent_state);
  return {offline,review,verifying,tone:offline?'warning':review?'danger':verifying?'warning':'info',
    kicker:offline?'SERVICE INTERRUPTED':review?'REVIEW REQUEST':verifying?'GATHERING EVIDENCE':'NO REVIEW REQUEST',
    headline:offline?'Current activity is unknown.':review?'An episode needs a person’s review.':verifying?'A change is being verified.':sim?'No human review is requested.':running(s)?'The generated scene is being observed.':'No synthetic run is active.',
    copy:offline?'Cloud analysis has no fresh evidence. Check the service and verify the situation through another channel.':review?'This scripted episode is unresolved. It is a request to review the evidence, not a confirmed fall.':verifying?'The agent is requesting more visual evidence before resolving the episode.':sim?'There are no open requests in this example. Absence of an alert does not establish a person’s wellbeing.':'This local run contains a moving shape, not a person. It cannot determine anyone’s wellbeing.',
  };
}
function caseRows(s,c) {return `<div class="case-list"><div class="case-row"><span class="case-label">Detection state</span><span class="case-value">${escapeText(s.agent_state)}</span></div><div class="case-row"><span class="case-label">Human review</span><span class="case-value">${c.review?(attention==='closed'?'Completed in walkthrough':attention==='reviewing'?'Demo caregiver reviewing':'No owner assigned'):'Not requested'}</span></div><div class="case-row"><span class="case-label">Notification delivery</span>${badge('NOT SENT','neutral')}</div><div class="case-row"><span class="case-label">Evidence access</span><span class="case-value">Generated scene only</span></div></div>`;}
function caregiver(s) {
  const c=careCase(s);
  return `<div class="grid-two"><section class="panel panel-body"><div class="person-header"><span class="avatar" aria-hidden="true">DH</span><div><p class="person-name">Demonstration household</p><p class="person-details">Living area · ${mode==='walkthrough'?'Fictional case':'Generated local source'}</p></div></div><div class="care-hero ${c.tone}"><p class="hero-kicker">${c.kicker}</p><h2 class="hero-title">${c.headline}</h2><p class="hero-copy">${c.copy}</p>${c.review?`<div class="care-actions">${attention==='unassigned'?'<button type="button" class="button primary" data-case="reviewing">I’m reviewing · demo</button>':attention==='reviewing'?'<button type="button" class="button primary" data-case="closed">Complete human review · demo</button>':badge('HUMAN REVIEW COMPLETED','info')}</div><p class="review-note">Changes affect this walkthrough only. The detector’s state remains separate.</p>`:''}</div></section><section class="panel panel-body"><p class="eyebrow">CASE OWNERSHIP</p><h2>What happens next</h2>${caseRows(s,c)}<div class="notice">${c.offline?'A service interruption is a technical issue, not an incident classification.':'One clear owner. A visible delivery state. A separate record of human attention.'}</div></section></div>`;
}
function family(s) {
  const c=careCase(s);
  const owner=c.review?(attention==='unassigned'?'No demo caregiver has taken the case yet.':attention==='reviewing'?'A demo caregiver is reviewing the episode.':'The human walkthrough review is complete.'):'No human attention has been requested.';
  return `<div class="family-layout"><section class="panel panel-body"><div class="panel-title"><div><p class="person-name">Demonstration household</p><p class="person-details">${mode==='walkthrough'?'Fictional family view':'Synthetic source · no person being monitored'}</p></div>${badge(c.offline?'NO RECENT INFORMATION':c.review?'REVIEW REQUEST':c.verifying?'VERIFYING':'NO OPEN REQUEST',c.tone||'neutral')}</div><div class="care-hero ${c.tone}"><p class="hero-kicker">LATEST UPDATE</p><h2 class="hero-title">${c.headline}</h2><p class="hero-copy">${c.copy}</p></div><div class="family-handoff"><span class="avatar" aria-hidden="true">${c.review&&attention!=='unassigned'?'DC':'—'}</span><div><h3>Who is taking care of it?</h3><p class="subtext">${owner}</p></div></div><div class="case-row"><span class="case-label">Latest information</span><span class="case-value">${c.offline?'No longer current':mode==='walkthrough'?'Scripted example':s.vision_live?'Recent synthetic observations':'No active observations'}</span></div><div class="case-row"><span class="case-label">External messages</span><span class="case-value">No messages have been sent</span></div></section></div>`;
}
function operations(s) {
  const c=careCase(s), sim=mode==='walkthrough';
  const households=sim?[['Demo household A','Scripted scenario',c.offline?'Interrupted':c.review?'Human review':c.verifying?'Verifying':'Baseline',c.offline?'No fresh evidence':'Scripted evidence',c.tone||'neutral'],['Demo household B','Fictional household','Baseline','Scripted evidence','neutral'],['Demo household C','Fictional household','Paused','Monitoring inactive','neutral']]:[['Local synthetic source','Owned demo process',s.session_state,s.vision_live?'Fresh evidence':'No active evidence',s.vision_live?'info':'neutral']];
  return `<div class="ops-summary"><div class="ops-stat"><span class="metric-label">${sim?'Illustrative households':'Local sources'}</span><strong>${households.length}</strong></div><div class="ops-stat"><span class="metric-label">Technical issues</span><strong>${c.offline?'1':sim?'0':s.session_state==='FAILED'?'1':'0'}</strong></div><div class="ops-stat"><span class="metric-label">Unassigned review requests</span><strong>${c.review&&attention==='unassigned'?'1':'0'}</strong></div></div><section class="panel panel-body"><div class="panel-title"><h2>Service overview</h2>${badge(sim?'SCRIPTED FLEET':'LOCAL PROCESS','info')}</div><table class="home-table"><thead><tr><th scope="col">Source</th><th scope="col">Status</th><th scope="col">Evidence</th></tr></thead><tbody>${households.map(([name,description,state,evidence,tone])=>`<tr><td><span class="home-name">${escapeText(name)}</span><span class="caption">${escapeText(description)}</span></td><td>${badge(state,tone)}</td><td>${escapeText(evidence)}</td></tr>`).join('')}</tbody></table></section><div class="grid-two section-space"><section class="panel panel-body"><p class="eyebrow">SERVICE HEALTH</p><h2>${c.offline?'Restore visibility first.':'Know where the stream stops.'}</h2><p class="reason">${c.offline?'Check camera, Edge, network and cloud worker separately. Historical telemetry must never appear as an active connection.':'Capture, transmission, receipt and processing are distinct stages. Their counts and freshness are shown in the Lab workspace.'}</p></section><section class="panel panel-body"><p class="eyebrow">REVIEW OWNERSHIP</p><h2>${c.review?(attention==='unassigned'?'This case needs an owner.':'Human review is recorded.'):'No human review is pending.'}</h2><p class="reason">${c.review?(attention==='unassigned'?'The demo request has not been taken. Assigning attention is distinct from notification delivery.':'A caregiver action is reflected across perspectives; it does not change the detector’s classification.'):'A technical failure must remain distinct from a possible incident.'}</p><div class="notice">Production organization permissions, persistence, delivery receipts and alert routing are not implemented in this local lab.</div></section></div>`;
}
function render() {
  const s=data();const title=titles[perspective];
  el('workspace-label').textContent=title[0];el('page-title').textContent=title[1];el('page-description').textContent=title[2];
  root.querySelectorAll('[data-role]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.role===perspective)));
  el('live-mode').setAttribute('aria-pressed',String(mode==='live'));el('walkthrough-mode').setAttribute('aria-pressed',String(mode==='walkthrough'));
  el('walkthrough-controls').hidden=mode!=='walkthrough';el('start-run').hidden=mode!=='live'||perspective!=='lab';el('stop-run').hidden=mode!=='live'||perspective!=='lab';
  el('start-run').disabled=busy||running(live);el('stop-run').disabled=busy||!running(live);
  el('step-label').textContent=`Event ${step+1} of 6`;el('previous-step').disabled=step===0;el('next-step').disabled=step===5;
  el('source-status').textContent=mode==='walkthrough'?'SCRIPTED SCENARIO · not runtime evidence':live.session_state==='IDLE'?'Ready when you are':`${live.session_state} · ${live.vision_live?'Fresh evidence arriving':'No current visual evidence'}`;
  el('scope-label').textContent=mode==='walkthrough'?'Scenario simulation · No inference running in this view':'Generated motion only · Standard OpenCV · NOT AWS / COOL';
  const focusedCommand=document.activeElement?.dataset?.command;
  el('workspace').innerHTML=({lab,caregiver,family,operations}[perspective])(s);drawChart();
  if(focusedCommand)root.querySelector('[data-command="'+focusedCommand+'"]')?.focus({preventScroll:true});
}
function drawChart() {
  const svg=el('rate-chart');if(!svg)return;
  const series=data().edge.series||[],width=Math.max(250,svg.getBoundingClientRect().width),height=190,left=38,right=12,top=18,bottom=34;
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  if(!series.length){svg.innerHTML=`<text x="${width/2}" y="86" text-anchor="middle">${mode==='walkthrough'?'Measured chart is available in a live run.':'Waiting for measured Edge samples.'}</text><text x="${width/2}" y="108" text-anchor="middle">${mode==='walkthrough'?'No illustrative rates are plotted as measurements.':'Start the synthetic run above.'}</text>`;return;}
  const maxX=Math.max(1,...series.map(p=>p.seconds)),maxY=Math.max(5,...series.flatMap(p=>[p.capture,p.requested,p.transmitted].filter(v=>typeof v==='number')))*1.12;
  const x=t=>left+t/maxX*(width-left-right),y=v=>height-bottom-v/maxY*(height-top-bottom);
  let markup=`<title>Measured capture, requested transmit and measured transmit FPS</title><desc>Samples from local Edge status records. Elapsed seconds use a single local host clock.</desc>`;
  for(let i=0;i<4;i++){const value=maxY*i/3;markup+=`<line class="axis-line" x1="${left}" y1="${y(value)}" x2="${width-right}" y2="${y(value)}"/><text x="${left-8}" y="${y(value)+4}" text-anchor="end">${fmt(value,0)}</text>`;}
  for(let i=0;i<4;i++){const value=maxX*i/3;markup+=`<text x="${x(value)}" y="${height-16}" text-anchor="${i===0?'start':i===3?'end':'middle'}">${fmt(value,0)}s</text>`;}
  markup+=`<text x="${left}" y="12">FPS</text><text x="${width-right}" y="${height-1}" text-anchor="end">Elapsed time</text>`;
  for(const [key,cls] of [['capture','capture-line'],['requested','requested-line'],['transmitted','transmitted-line']]){let drawing=false,path='';for(const p of series){if(typeof p[key]!=='number'){drawing=false;continue;}path+=(drawing?'L':'M')+x(p.seconds).toFixed(2)+' '+y(p[key]).toFixed(2)+' ';drawing=true;}markup+=`<path class="${cls}" fill="none" d="${path}"/>`;}
  svg.innerHTML=markup;
}
function announce(message){el('announcement').textContent=message;}
function setMode(value){mode=value;selectedCommand=null;render();announce(value==='live'?'Live synthetic workspace selected.':'Scripted scenario selected.');}
root.addEventListener('click',event=>{
  const button=event.target.closest('button');if(!button)return;
  if(button.dataset.role){perspective=button.dataset.role;render();announce(titles[perspective][1]);}
  if(button.dataset.command){selectedCommand=Number(button.dataset.command);render();}
  if(button.dataset.case){attention=button.dataset.case;render();announce('Walkthrough human review updated. No notification was sent.');}
});
el('live-mode').onclick=()=>setMode('live');el('walkthrough-mode').onclick=()=>setMode('walkthrough');
el('scenario').onchange=e=>{scenario=e.target.value;attention='unassigned';render();};
el('previous-step').onclick=()=>{step=Math.max(0,step-1);render();};el('next-step').onclick=()=>{step=Math.min(5,step+1);render();};
async function request(path,mutate=false) {
  const options={cache:'no-store',credentials:'same-origin'};
  if(mutate)Object.assign(options,{method:'POST',headers:{'X-CSRF-Token':csrf,'Content-Type':'application/json'},body:'{}'});
  const response=await fetch(path,options);const value=await response.json();
  if(!response.ok)throw new Error(value.error||'Local request failed.');return value;
}
async function refresh() {
  if(polling)return;polling=true;
  try{live=await request('/api/state');csrf=live.csrf;el('error').hidden=true;if(mode==='live')render();}
  catch(error){el('error').textContent='Local connection unavailable. Displayed values are historical. '+error.message;el('error').hidden=false;live.vision_live=false;live.edge.fresh=false;if(mode==='live'){render();el('source-status').textContent='CONNECTION UNAVAILABLE · retained values';}}
  finally{polling=false;}
}
async function control(action) {
  busy=true;render();
  try{await request('/api/'+action,true);await refresh();announce(action==='start'?'Synthetic run started.':'Synthetic run stopped.');}
  catch(error){el('error').textContent=error.message;el('error').hidden=false;}
  finally{busy=false;render();}
}
el('start-run').onclick=()=>control('start');el('stop-run').onclick=()=>control('stop');
new ResizeObserver(drawChart).observe(el('workspace'));
render();refresh();setInterval(refresh,1000);
