'use strict';
const root = document.getElementById('care-app');
const el = id => root.querySelector('#' + id);
const escapeText = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (n, digits = 1) => typeof n === 'number' && Number.isFinite(n) ? n.toFixed(digits) : '—';
const running = s => ['STARTING', 'RUNNING'].includes(s.session_state);
const empty = () => ({session_state:'IDLE',agent_state:'UNKNOWN',runtime:{},counts:{},metrics:{},edge:{metrics:{},series:[]},commands:[],transitions:[],timeline:[],observation:{},vision_live:false});
let live = empty(), perspective = 'lab', mode = 'walkthrough', scenario = 'recovery', step = 0, attention = 'unassigned', selectedCommand = null, polling = false, busy = false, csrf = '';
const titles = {
  lab:['SYSTEM EXPLORER','See the loop in action.','Follow a frame from capture to the next agent decision.'],
  caregiver:['CARE WORKSPACE','Review the evidence. See what happened.','Saved episodes, body pose and controller outcomes, without household camera imagery.'],
  family:['FAMILY WORKSPACE','Stay informed, without the noise.','Relevant updates and who is taking care of them.'],
  operations:['SERVICE OPERATIONS','Keep every connection visible.','Separate service health from events that need human review.'],
};
function demoState() {
  const states = ['NORMAL','SUSPICIOUS','VERIFYING','VERIFYING','VERIFYING','RECOVERY'];
  const active=step>=2&&(step<5||scenario==='review');
  const s = empty();s.session_state='WALKTHROUGH';s.agent_state=states[step];s.runtime={phase:1,cool_verified:false};
  s.edge={metrics:{capture_fps:30,configured_capture_fps:30,transmit_fps:step<3?2:step===3?null:active?20:2,target_transmit_fps:active?20:2,buffer_frames:300,buffer_bytes:null,network:'GOOD'},series:[],fresh:true,preview_capture_seq:null};
  s.vision_live=true;
  s.commands=step<2?[]:[{index:1,action:'SET_TRANSMIT_FPS',requested_fps:20,source_seen:true,ack_applied:step>=3,new_perception:step>=4,reason:'Sustained motion requires denser visual evidence; request 20 FPS.',source:{capture_seq:241,motion_score:null}}];
  s.transitions=[{state:s.agent_state,reason:['Baseline visual sampling.','A change in the example requires more evidence.','The command is sent; application has not been confirmed.','Applied ACK received; new evidence is still required.','Post-command evidence was processed.','The example shows recovery; return to baseline.'][step]}];
  if(step===5 && scenario==='review'){s.agent_state='ASSISTANCE_REQUIRED';s.transitions=[{state:s.agent_state,reason:'This scripted episode remains unresolved and requests human review.'}];}
  if(step>=3 && scenario==='offline'){s.agent_state='UNKNOWN';s.vision_live=false;s.edge.fresh=false;s.edge.metrics.network='OFFLINE';s.edge.metrics.transmit_fps=null;s.commands[0].ack_applied=false;s.commands[0].new_perception=false;s.transitions=[{state:'UNKNOWN',reason:'Connection interrupted. Cloud analysis is unavailable; current human activity is unknown.'}];}
  return s;
}
const data = () => mode==='walkthrough'?demoState():live;
function status(s) {
  if(mode==='live' && !running(s))return {tone:s.session_state==='FAILED'?'danger':'neutral',label:s.session_state==='IDLE'?'Ready to start':s.session_state==='FAILED'?'Run failed':'Session finished'};
  if(!s.vision_live)return {tone:'warning',label:mode==='walkthrough'?'Monitoring interrupted':'Waiting for fresh evidence'};
  if(s.agent_state==='ASSISTANCE_REQUIRED')return {tone:'danger',label:'Human review needed'};
  if(['SUSPICIOUS','VERIFYING'].includes(s.agent_state))return {tone:'warning',label:'Verifying a change'};
  return {tone:'good',label:mode==='walkthrough'?'No open review request':'Evidence arriving'};
}
const badge = (label,tone='neutral') => `<span class="badge ${tone}">${escapeText(label)}</span>`;
const metric = (label,value,unit,context='') => `<div class="metric"><span class="metric-label">${label}</span><span class="metric-value">${escapeText(value)}<span class="metric-unit">${unit}</span></span>${context?`<span class="metric-context">${context}</span>`:''}</div>`;
const pair = (label,value) => `<div><dt>${label}</dt><dd>${escapeText(value)}</dd></div>`;
function chartHTML() {return '<svg id="rate-chart" class="chart" role="img" aria-label="Measured local capture and transmission rates over time"></svg><div class="legend"><span><i></i>Capture</span><span><i class="requested"></i>Requested transmit</span><span><i class="transmitted"></i>Measured transmit</span></div>';}
function networkLabel(s) {
  if(mode==='walkthrough')return s.edge.metrics.network||'UNKNOWN';
  if(s.session_state==='IDLE')return 'Not started';
  if(!running(s))return s.session_state==='FAILED'?'Run failed · no active stream':'Session finished · stream closed';
  return s.edge.fresh?(s.edge.metrics.network||'UNKNOWN'):s.edge.metrics.network==='OFFLINE'?'OFFLINE · no recent sample':'Unknown · no recent network sample';
}
function seatedLabel(s) {
  if(s.target!=='graviton')return 'Disabled · local passive pose reference';
  const active=s.profile?.seated_closure_seconds, configured=s.capabilities?.graviton_seated_closure_seconds;
  if(typeof active==='number')return (running(s)?'Enabled':'Last run')+' · '+fmt(active)+' s · reviewed scene';
  return typeof configured==='number'?'Configured · '+fmt(configured)+' s · awaiting worker verification':'Disabled · no opt-in in cloud profile';
}
function careCase(s) {
  const sim=mode==='walkthrough', offline=sim&&scenario==='offline'&&step>=3, review=sim&&scenario==='review'&&step===5, verifying=s.vision_live&&['SUSPICIOUS','VERIFYING'].includes(s.agent_state);
  if(s.pipeline==='full'&&!sim)return {offline:running(s)&&!s.vision_live,review:false,verifying,tone:s.vision_live?'info':'warning',kicker:s.vision_live?'EXPERIMENTAL OBSERVATION':'NO FRESH EVIDENCE',headline:running(s)?'The person/pose pipeline is running.':'No pipeline is active.',copy:'Review the System workspace for measured detections and decisions. This prototype has no caregiver delivery or validated incident classification.'};
  return {offline,review,verifying,tone:offline?'warning':review?'danger':verifying?'warning':'info',
    kicker:offline?'SERVICE INTERRUPTED':review?'REVIEW REQUEST':verifying?'GATHERING EVIDENCE':'NO REVIEW REQUEST',
    headline:offline?'Current activity is unknown.':review?'An episode needs a person’s review.':verifying?'A change is being verified.':sim?'No human review is requested.':running(s)?(s.source?.kind==='camera'?'Local camera motion is being observed.':'The generated scene is being observed.'):'No synthetic run is active.',
    copy:offline?'Cloud analysis has no fresh evidence. Check the service and verify the situation through another channel.':review?'This scripted episode is unresolved. It is a request to review the evidence, not a confirmed fall.':verifying?'The agent is requesting more visual evidence before resolving the episode.':sim?'There are no open requests in this example. Absence of an alert does not establish a person’s wellbeing.':(s.source?.kind==='camera'?'This local camera diagnostic measures motion only. It cannot establish a person’s wellbeing.':'This local run contains a moving shape, not a person. It cannot determine anyone’s wellbeing.'),
  };
}
function caseRows(s,c) {return `<div class="case-list"><div class="case-row"><span class="case-label">Detection state</span><span class="case-value">${escapeText(s.agent_state)}</span></div><div class="case-row"><span class="case-label">Human review</span><span class="case-value">${c.review?(attention==='closed'?'Completed in walkthrough':attention==='reviewing'?'Demo caregiver reviewing':'No owner assigned'):'Not requested'}</span></div><div class="case-row"><span class="case-label">Notification delivery</span>${badge('NOT SENT','neutral')}</div><div class="case-row"><span class="case-label">Evidence access</span><span class="case-value">${s.pipeline==='full'?'Local source preview only':'Scenario illustration only'}</span></div></div>`;}
function caregiver(s) {
  const c=careCase(s);
  return `<div class="grid-two"><section class="panel panel-body"><div class="person-header"><span class="avatar" aria-hidden="true">DH</span><div><p class="person-name">Demonstration household</p><p class="person-details">Living area · ${mode==='walkthrough'?'Fictional case':s.pipeline==='full'?'Experimental person/pose pipeline':s.source?.kind==='camera'?'Local camera motion diagnostic':'Generated local source'}</p></div></div><div class="care-hero ${c.tone}"><p class="hero-kicker">${c.kicker}</p><h2 class="hero-title">${c.headline}</h2><p class="hero-copy">${c.copy}</p>${c.review?`<div class="care-actions">${attention==='unassigned'?'<button type="button" class="button primary" data-case="reviewing">I’m reviewing · demo</button>':attention==='reviewing'?'<button type="button" class="button primary" data-case="closed">Complete human review · demo</button>':badge('HUMAN REVIEW COMPLETED','info')}</div><p class="review-note">Changes affect this walkthrough only. The detector’s state remains separate.</p>`:''}</div></section><section class="panel panel-body"><p class="eyebrow">CASE OWNERSHIP</p><h2>What happens next</h2>${caseRows(s,c)}<div class="notice">${c.offline?'A service interruption is a technical issue, not an incident classification.':'One clear owner. A visible delivery state. A separate record of human attention.'}</div></section></div>`;
}
function family(s) {
  const c=careCase(s);
  const owner=c.review?(attention==='unassigned'?'No demo caregiver has taken the case yet.':attention==='reviewing'?'A demo caregiver is reviewing the episode.':'The human walkthrough review is complete.'):'No human attention has been requested.';
  return `<div class="family-layout"><section class="panel panel-body"><div class="panel-title"><div><p class="person-name">Demonstration household</p><p class="person-details">${mode==='walkthrough'?'Fictional family view':s.pipeline==='full'?'Experimental person/pose pipeline · No external messages':s.source?.kind==='camera'?'Local motion diagnostic · no person classification':'Synthetic source · no person being monitored'}</p></div>${badge(c.offline?'NO RECENT INFORMATION':c.review?'REVIEW REQUEST':c.verifying?'VERIFYING':'NO OPEN REQUEST',c.tone||'neutral')}</div><div class="care-hero ${c.tone}"><p class="hero-kicker">LATEST UPDATE</p><h2 class="hero-title">${c.headline}</h2><p class="hero-copy">${c.copy}</p></div><div class="family-handoff"><span class="avatar" aria-hidden="true">${c.review&&attention!=='unassigned'?'DC':'—'}</span><div><h3>Who is taking care of it?</h3><p class="subtext">${owner}</p></div></div><div class="case-row"><span class="case-label">Latest information</span><span class="case-value">${c.offline?'No longer current':mode==='walkthrough'?'Scripted example':s.vision_live?'Illustrated evidence':'No active observations'}</span></div><div class="case-row"><span class="case-label">External messages</span><span class="case-value">No messages have been sent</span></div></section></div>`;
}
function operations(s) {
  const c=careCase(s), sim=mode==='walkthrough';
  const households=sim?[['Demo household A','Scripted scenario',c.offline?'Interrupted':c.review?'Human review':c.verifying?'Verifying':'Baseline',c.offline?'No fresh evidence':'Scripted evidence',c.tone||'neutral'],['Demo household B','Fictional household','Baseline','Scripted evidence','neutral'],['Demo household C','Fictional household','Paused','Monitoring inactive','neutral']]:[[s.pipeline==='full'?'Person/pose pipeline':s.source?.kind==='camera'?'Local camera diagnostic':'Local synthetic source',s.target==='graviton'?'Your AWS Graviton':'Owned local process',s.session_state,s.vision_live?'Fresh evidence':'No active evidence',s.vision_live?'info':'neutral']];
  return `<div class="ops-summary"><div class="ops-stat"><span class="metric-label">${sim?'Illustrative households':'Local sources'}</span><strong>${households.length}</strong></div><div class="ops-stat"><span class="metric-label">Technical issues</span><strong>${c.offline?'1':sim?'0':s.session_state==='FAILED'?'1':'0'}</strong></div><div class="ops-stat"><span class="metric-label">Unassigned review requests</span><strong>${c.review&&attention==='unassigned'?'1':'0'}</strong></div></div><section class="panel panel-body"><div class="panel-title"><h2>Service overview</h2>${badge(sim?'SCRIPTED FLEET':'LOCAL PROCESS','info')}</div><table class="home-table"><thead><tr><th scope="col">Source</th><th scope="col">Status</th><th scope="col">Evidence</th></tr></thead><tbody>${households.map(([name,description,state,evidence,tone])=>`<tr><td><span class="home-name">${escapeText(name)}</span><span class="caption">${escapeText(description)}</span></td><td>${badge(state,tone)}</td><td>${escapeText(evidence)}</td></tr>`).join('')}</tbody></table></section><div class="grid-two section-space"><section class="panel panel-body"><p class="eyebrow">SERVICE HEALTH</p><h2>${c.offline?'Restore visibility first.':'Know where the stream stops.'}</h2><p class="reason">${c.offline?'Check camera, Edge, network and cloud worker separately. Historical telemetry must never appear as an active connection.':'Capture, transmission, receipt and processing are distinct stages. Their counts and freshness are shown in the System workspace.'}</p></section><section class="panel panel-body"><p class="eyebrow">REVIEW OWNERSHIP</p><h2>${c.review?(attention==='unassigned'?'This case needs an owner.':'Human review is recorded.'):'No human review is pending.'}</h2><p class="reason">${c.review?(attention==='unassigned'?'The demo request has not been taken. Assigning attention is distinct from notification delivery.':'A caregiver action is reflected across perspectives; it does not change the detector’s classification.'):'A technical failure must remain distinct from a possible incident.'}</p><div class="notice">These perspectives illustrate a workflow. No caregiver accounts, external notifications or production alert routing are created here.</div></section></div>`;
}

function illustration(s) {
  const offline=scenario==='offline'&&step>=3, low=step>=2&&(step<5||scenario==='review');
  const body=low?'<circle cx="246" cy="202" r="15"/><path d="M265 207 L323 219 L369 216 M303 215 L322 240 L367 242 M278 211 L257 234"/>':'<circle cx="304" cy="84" r="16"/><path d="M304 107 L303 169 M303 126 L275 153 M304 127 L332 151 M303 169 L280 227 M303 169 L329 227"/>';
  return `<div><svg class="illustration" viewBox="0 0 540 280" role="img" aria-label="Illustrated scenario, not recorded camera footage"><rect class="room" x="30" y="18" width="480" height="240" rx="16"/><path d="M48 239H492" stroke="var(--line)" stroke-width="2"/><rect x="83" y="145" width="103" height="15" rx="4" fill="var(--line)"/><path d="M95 160V236 M175 160V236" stroke="var(--line)" stroke-width="7"/><g class="person" opacity="${offline?.25:1}">${body}</g>${step>=2&&!offline?'<rect class="roi" x="228" y="51" width="165" height="195" rx="8"/>':''}<text class="hint" x="48" y="46">${offline?'No fresh visual evidence':step===0?'Baseline observation':step===1?'A change requires verification':step===5?(scenario==='review'?'Human review remains open':'Return to baseline'):'Observe again before deciding'}</text></svg></div>`;
}
let autoplay=null, caregiverMode='recorded';
const episodeReview=CareEpisodeReview.create(el('workspace'),window.CARE_DATA,announce);
function pauseScenario(){if(autoplay!==null)clearInterval(autoplay);autoplay=null;el('play-scenario').textContent='Play scenario';}
function render() {
  const s=data(),title=titles[perspective], recorded=perspective==='caregiver'&&caregiverMode==='recorded';
  episodeReview.pause();
  el('caregiver-mode').hidden=perspective!=='caregiver';
  root.querySelectorAll('[data-care-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.careMode===caregiverMode)));
  root.querySelector('.run-actions').hidden=recorded;
  root.querySelector('.walkthrough-controls').hidden=recorded;
  root.querySelector('.judge-callout').hidden=perspective!=='lab';
  root.querySelector('.system-intro').hidden=perspective!=='lab';
  el('context-badge').textContent=recorded?'SAVED EPISODE REVIEW':'SCENARIO WALKTHROUGH';
  el('source-status').textContent=recorded?'Historical data · no live monitoring or notifications':'Illustrative scenario · no camera is active';
  el('workspace-label').textContent=title[0];el('page-title').textContent=title[1];el('page-description').textContent=title[2];
  if(perspective==='caregiver'&&!recorded){el('page-title').textContent='Know what needs your attention.';el('page-description').textContent='An illustrated human-review workflow. Actions affect this scenario only.';}
  root.querySelectorAll('[data-role]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.role===perspective)));
  el('step-label').textContent=`Event ${step+1} of 6`;
  el('previous-step').disabled=step===0;el('next-step').disabled=step===5;
  const focus=document.activeElement?.dataset?.command;
  if(recorded) {episodeReview.mount();return;}
  el('workspace').innerHTML=({lab:publicLab,caregiver,family,operations}[perspective])(s);
  if(focus)root.querySelector('[data-command="'+focus+'"]')?.focus({preventScroll:true});
  const chart=el('rate-chart');if(chart)chart.innerHTML='<text x="50%" y="80" text-anchor="middle">Scenario values are illustrative.</text><text x="50%" y="105" text-anchor="middle">Measured results are in the judge review.</text>';
}
function announce(message){el('announcement').textContent=message;}
root.addEventListener('click',event=>{
  const button=event.target.closest('button');if(!button)return;
  if(button.dataset.role){window.location.hash=button.dataset.role==='lab'?'how-it-works':button.dataset.role;}
  if(button.dataset.careMode){pauseScenario();episodeReview.pause();caregiverMode=button.dataset.careMode;render();announce(caregiverMode==='recorded'?'Saved episode review.':'Scripted human-review walkthrough.');}
  if(button.dataset.command){selectedCommand=Number(button.dataset.command);render();}
  if(button.dataset.case){attention=button.dataset.case;render();announce('Example human review updated. No notification was sent.');}
});
el('scenario').onchange=e=>{pauseScenario();scenario=e.target.value;step=0;attention='unassigned';selectedCommand=null;render();};
el('previous-step').onclick=()=>{pauseScenario();step=Math.max(0,step-1);render();};
el('next-step').onclick=()=>{pauseScenario();step=Math.min(5,step+1);render();};
el('reset-scenario').onclick=()=>{pauseScenario();step=0;attention='unassigned';selectedCommand=null;render();};
el('play-scenario').onclick=()=>{if(autoplay!==null){pauseScenario();return;}if(step===5){step=0;attention='unassigned';}el('play-scenario').textContent='Pause scenario';autoplay=setInterval(()=>{step=Math.min(5,step+1);if(step===5)pauseScenario();render();},2200);render();};
document.addEventListener('visibilitychange',()=>{if(document.hidden){pauseScenario();episodeReview.pause();}});
function navigateWorkspace(){
  pauseScenario();episodeReview.pause();
  const key=window.location.hash.slice(1);
  perspective=['caregiver','family','operations'].includes(key)?key:'lab';
  render();announce(titles[perspective][1]);
}
window.addEventListener('hashchange',navigateWorkspace);
navigateWorkspace();
function publicLab(s) {
  const latest=s.commands.at(-1), st=status(s);
  const phases=[['Source observed',!!latest],['Command selected',!!latest],['ACK applied',latest?.ack_applied],['Perceived again',latest?.new_perception]];
  const reasons=['The Edge captures continuously while only selected frames are sent.','A change raises uncertainty. The next step is to collect more evidence.','The controller requests a higher transmission rate. A request alone is not proof that it was applied.','The Edge confirms application. The controller still needs a later observation.','New evidence completes the perception–decision–action loop.','Recovery returns sensing to baseline; an unresolved case remains visible for human review.'];
  const interrupted=scenario==='offline'&&step>=3;
  const action=step===5&&scenario==='recovery'?'RETURN_TO_BASELINE':latest?latest.action+'('+latest.requested_fps+')':'KEEP_OBSERVING';
  if(interrupted)reasons[step]='The cloud has no fresh evidence. The Edge retains recent context locally while awaiting reconnection; current activity stays unknown.';
  const events=['Baseline observation','A change raises uncertainty','Select the next observation',scenario==='offline'?'Connection interrupted':'Confirm Edge application',scenario==='offline'?'Retain recent context locally':'Process new visual evidence',scenario==='review'?'Request human review':scenario==='offline'?'Await reconnection; activity unknown':'Recovery and baseline'];
  return `<div class="grid-two"><section class="panel panel-body"><div class="panel-title"><h2>What the system observes</h2>${badge('ILLUSTRATED SCENE','info')}</div>${illustration(s)}<p class="scene-caption">Illustration, not camera footage. The blue region illustrates targeted observation.</p><div class="metrics-row">${metric('Capture',fmt(s.edge.metrics.capture_fps),'FPS','Example continuous capture')}${metric('Transmit',fmt(s.edge.metrics.transmit_fps),'FPS','Illustrated transmission')}${metric('Requested',fmt(s.edge.metrics.target_transmit_fps),'FPS','Selected observation policy')}</div></section><section class="panel panel-body"><div class="panel-title"><h2>The agent’s decision</h2>${badge(st.label,st.tone)}</div><p class="state-label">Scenario state</p><p class="agent-state">${escapeText(s.agent_state)}</p><p class="reason">${escapeText(s.transitions[0].reason)}</p><div class="action-box"><span class="caption">Selected action in this example</span><code>${escapeText(action)}</code></div><div class="chain">${phases.map(([label,done])=>`<div class="chain-step ${done?'done':''}">${done?'✓ ':''}${label}</div>`).join('')}</div><div class="notice">${scenario==='offline'&&step>=3?'Cloud analysis is unavailable. Missing evidence stays visible as UNKNOWN.':'A command needs an applied ACK and a later, valid observation to complete the loop.'}</div></section></div><div class="grid-two section-space"><section class="panel panel-body"><p class="eyebrow">ACTIVE PERCEPTION</p><h2>Observe more when the evidence needs it.</h2><p class="reason">${reasons[step]}</p><dl class="data-list">${pair('Current representation',interrupted?'No fresh evidence':step>=2&&(step<5||scenario==='review')?'Targeted observation example':'Full-scene context')}${pair('Network in this scenario',s.edge.metrics.network)}${pair('Local ring buffer','Recent visual context')}${pair('Inference and controller','AWS Graviton · OpenCV 5 + COOL')}</dl><p class="notice">Capture FPS and transmission FPS are different controls. The numbers above illustrate a policy; they are not benchmark measurements.</p></section><section class="panel panel-body"><div class="panel-title"><h2>One loop. Six visible events.</h2>${badge('Event '+(step+1)+' of 6')}</div><div class="timeline">${events.map((text,i)=>`<div class="timeline-item"><span class="timeline-dot" aria-hidden="true" style="opacity:${i<=step?1:.25}"></span><div><strong>${i+1}. ${text}</strong>${i===step?'<p>Current event in the walkthrough</p>':''}</div></div>`).join('')}</div></section></div>`;
}
