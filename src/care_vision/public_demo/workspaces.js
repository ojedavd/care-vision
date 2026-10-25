/* Family and Operations consume only the existing, reviewed public episode export. */
(() => {
  'use strict';
  const esc=v=>String(v ?? '—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=(n,d=0)=>Number.isFinite(n)?n.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
  const badge=(label,tone='info')=>`<span class="badge ${tone}">${esc(label)}</span>`;
  const stat=(label,value,context)=>`<div class="board-stat"><span>${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(context)}</small></div>`;
  const nearest=(values,p)=>values.length?[...values].sort((a,b)=>a-b)[Math.max(0,Math.ceil(values.length*p)-1)]:null;
  function commandTrace(trial) {
    return trial.events.filter(e=>e.kind==='command').map(event=>({
      ...event, nextCapture:Number.isFinite(event.applied)?trial.frames.find(f=>f.t>event.applied):undefined,
    }));
  }
  function summary(trial) {
    const commands=commandTrace(trial), processing=trial.frames.map(f=>f.processing).filter(Number.isFinite);
    const opened=trial.episodes.find(e=>e.open), closed=trial.episodes.find(e=>!e.open);
    const alert=trial.episodes.find(e=>e.alert);
    return {captured:trial.captured,sent:trial.sent,observations:trial.frames.length,
      bytes:trial.frames.at(-1)?.bytes ?? null,commands,acks:commands.filter(e=>Number.isFinite(e.applied)).length,
      p50:nearest(processing,.5),p95:nearest(processing,.95),
      state:trial.states.at(-1)?.state ?? 'UNKNOWN',
      original:(trial.recordedStates||trial.states).at(-1)?.state ?? 'UNKNOWN',
      opened:opened?.t,outcome:closed?.t ?? alert?.t ?? null,
      episodeSpan:opened && closed?closed.t-opened.t:null,alert:!!alert,
      rates:trial.frames.reduce((a,f)=>{if(a.at(-1)!==f.fps)a.push(f.fps);return a;},[])};
  }
  function aggregate(trials) {
    return trials.map(summary).reduce((a,s)=>({captured:a.captured+s.captured,sent:a.sent+s.sent,
      observations:a.observations+s.observations,commands:a.commands+s.commands.length,
      acks:a.acks+s.acks,alerts:a.alerts+Number(s.alert)}),
    {captured:0,sent:0,observations:0,commands:0,acks:0,alerts:0});
  }
  function milestones(trial) {
    return [{t:0,title:'Baseline observation',copy:'Only selected captures are sent for cloud analysis.'},
      ...trial.events.filter(e=>e.kind==='episode').map(e=> {
        const label=e.label.toLowerCase();
        if(label.startsWith('episode opens'))return {t:e.t,title:'A change needed verification',copy:'An episode opened and the system requested more evidence.'};
        if(label.startsWith('horizontal'))return {t:e.t,title:'A horizontal posture was observed',copy:'The tracked episode remained open for further evidence.'};
        if(label.startsWith('recovery'))return {t:e.t,title:'Recovery closed the episode',copy:'The recorded controller returned to NORMAL.'};
        if(label.startsWith('seated closure'))return {t:e.t,title:'Seated closure in replay',copy:'Replay returned to NORMAL. The original recording remained UNRESOLVED.'};
        if(label.startsWith('test alert'))return {t:e.t,title:'A test alert requested review',copy:'No recovery was recorded before this alert. No external notification was sent.'};
        if(label.startsWith('end'))return {t:e.t,title:'The capture ended with an open episode',copy:'The saved trial ends here; no later outcome is available in this review.'};
        return {t:e.t,title:e.label,copy:'Saved episode milestone.'};
      })];
  }
  function familyStatus(trial,s) {
    if(trial.replay && s.state==='NORMAL' && s.episode && !s.episode.open)
      return {tone:'good',label:'CLOSED IN REPLAY',title:'The seated episode closes in replay.',copy:'The original recorded episode stayed unresolved. Both outcomes remain visible in the evidence review.'};
    if(s.alert)return {tone:'danger',label:'REVIEW REQUESTED',title:'This episode called for a person’s review.',copy:'A test alert followed an open episode without recovery. This trial did not send an external notification.'};
    if(s.episode && !s.episode.open)return {tone:'good',label:'EPISODE CLOSED',title:'Recovery brought the episode to a close.',copy:'The recorded controller returned to NORMAL. This describes the saved episode, not someone’s current wellbeing.'};
    if(s.episode)return {tone:'warning',label:'VERIFICATION OPEN',title:'The system was still gathering evidence.',copy:'At this point the episode was open. A final outcome had not yet been reached.'};
    return {tone:'info',label:'BASELINE',title:'No episode was open at this point.',copy:'This is an earlier moment in the saved trial. Select a later milestone to follow what happened.'};
  }
  function chart(trial,field,label,id) {
    const frames=trial.frames.filter(f=>Number.isFinite(f[field])), width=640, height=190;
    const max=field==='fps'?6:Math.max(1,...frames.map(f=>f[field]))*1.15;
    const x=t=>44+t/trial.duration*576,y=v=>154-v/max*126;
    let path='';frames.forEach((f,i)=>{path+=i?(field==='fps'?`H${x(f.t)}V${y(f[field])}`:`L${x(f.t)},${y(f[field])}`):`M${x(f.t)},${y(f[field])}`;});
    if(field==='fps'&&frames.length)path+=`H${x(trial.duration)}`;
    const levels=field==='fps'?[0,2,5]:[0,max/2,max];
    return `<svg class="board-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(label)}">${levels.map(v=>`<path class="chart-grid" d="M44 ${y(v)}H620"/><text x="34" y="${y(v)+4}" text-anchor="end">${num(v)}</text>`).join('')}${[0,15,30,45,trial.duration].map(t=>`<text x="${x(t)}" y="178" text-anchor="middle">${num(t)} s</text>`).join('')}<path class="chart-signal" d="${path}"/><line id="${id}" class="chart-cursor" y1="23" y2="154"/></svg>`;
  }
  function lanes(trial) {
    const lane=(label,rows)=>`<div class="board-lane"><span>${label}</span><div>${rows.map((row,i)=> {
      const end=rows[i+1]?.t??trial.duration;if(end<=row.t)return '';
      return `<span class="board-segment state-${row.state}" style="width:${(end-row.t)/trial.duration*100}%" title="${esc(row.state)}: ${num(row.t,3)}–${num(end,3)} s">${esc(row.state)}</span>`;
    }).join('')}<i class="board-playhead"></i></div></div>`;
    return lane(trial.replay?'Replay state':'Recorded state',trial.states)+(trial.replay?lane('Original state',trial.recordedStates):'');
  }
  function create(container,data,announce=()=>{},onSelect=()=>{}) {
    let index=0, view='family', time=0, filter='all', selectedCommand=0;
    const $=id=>container.querySelector('#'+id);
    const trial=()=>data.trials[index];
    function select(key) {const i=data?.trials?.findIndex(t=>t.key===key);if(i>=0){index=i;filter='all';return true;}return false;}
    const selectedKey=()=>data?.trials?.[index]?.key;
    function empty() {container.innerHTML='<section class="panel panel-body"><h2>Saved episode data unavailable</h2><p class="reason">No live service is connected to this view.</p><a class="button section-space" href="/trials/">Open episode evidence ↗</a></section>';}
    function choices() {
      return `<div class="care-episodes" role="group" aria-label="Family episode selection">${data.trials.map((t,i)=> {
        const s=summary(t);
        return `<button type="button" data-family-trial="${i}" aria-pressed="${i===index}"><span class="board-card-top"><strong>${esc(t.title)}</strong>${badge(s.alert?'Review':t.replay?'Replay':'Closed',s.alert?'danger':t.replay?'info':'good')}</span><span>${t.replay?'Seated closure with original state retained':s.alert?'Open episode · test alert':'Recorded recovery · episode closed'}</span><small>${t.replay?'REPLAY OUTCOME':'RECORDED OUTCOME'} · ${num(s.outcome,2)} s</small></button>`;
      }).join('')}</div>`;
    }
    function mountFamily() {
      const t=trial(), sum=summary(t), moments=milestones(t);time=t.focus;
      container.innerHTML=`${choices()}<section class="panel family-overview"><div class="family-summary-copy"><div class="board-card-top"><p class="eyebrow">EPISODE UPDATE</p><span id="family-status"></span></div><h2 id="family-headline"></h2><p id="family-copy"></p><div class="family-summary-meta"><span>${esc(t.title)} · ${t.replay?'Controller replay':'Recorded outcome'}</span><span id="family-time"></span></div></div><div class="family-summary-facts">${stat('Saved capture',num(t.duration,0)+' s','Source-relative episode review')}${stat('External notification','Not sent','Test-only alerting in these trials')}${stat('Human review','Not recorded','No real caregiver assignment is provided')}</div></section>
        <div class="family-board-grid section-space"><section class="panel panel-body"><div class="panel-title"><div><p class="eyebrow">EVENT JOURNEY</p><h2>What happened, step by step</h2></div>${badge('SELECT A MOMENT')}</div><p class="subtext">Times are relative to the start of this saved capture.</p><ol class="family-journey">${moments.map((m,i)=>`<li><button type="button" data-family-moment="${i}" aria-pressed="false"><span class="journey-number">${i+1}</span><span><strong>${esc(m.title)}</strong><small>${esc(m.copy)}</small></span><time>${num(m.t,2)} s</time></button></li>`).join('')}</ol></section>
        <aside class="family-sidebar"><section class="panel panel-body"><p class="eyebrow">CARE &amp; COMMUNICATION</p><h2>The system decision is one step.</h2><div class="communication-list"><div><span class="communication-icon">1</span><span><strong id="family-decision"></strong><small id="family-original"></small></span></div><div><span class="communication-icon neutral">2</span><span><strong>Caregiver attention</strong><small>No human-review record is included in these trials.</small></span></div><div><span class="communication-icon neutral">3</span><span><strong>Family notification</strong><small>No external message was sent.</small></span></div></div><a class="button primary" href="#caregiver/${esc(t.key)}">Review this episode in Caregiver ↗</a></section>
          <section class="panel panel-body"><p class="eyebrow">VISUAL PRIVACY</p><h2>Context without household footage.</h2><p class="subtext">Only body joints can be shown here. Faces, room images, video and audio are excluded.</p><details id="family-pose-details" class="family-pose-details"><summary>Show body pose at this moment</summary><div id="family-pose" class="care-pose"></div><p id="family-pose-caption" class="caption"></p></details><p class="caption board-footnote">${t.poseSource==='recorded_cloud'?'Sitting reuses saved cloud pose.':'Pose was prepared offline for presentation; it was not used by the recorded controller.'} Missing or over-one-second-old pose stays absent.</p></section>
        </aside></div>`;
      container.querySelectorAll('[data-family-trial]').forEach(b=>b.onclick=()=>{
        index=Number(b.dataset.familyTrial);mountFamily();onSelect(view,trial().key);container.querySelector(`[data-family-trial="${index}"]`).focus({preventScroll:true});announce(`${trial().title} family update selected.`);
      });
      container.querySelectorAll('[data-family-moment]').forEach(b=>b.onclick=()=>{time=moments[Number(b.dataset.familyMoment)].t;updateFamily();});
      updateFamily();
    }
    function updateFamily() {
      const t=trial(), s=CareEpisodeReview.snapshot(data,t,time), info=familyStatus(t,s);
      $('family-status').innerHTML=badge(info.label,info.tone);
      $('family-headline').textContent=info.title;$('family-copy').textContent=info.copy;
      $('family-time').textContent=`Selected moment · ${num(time,2)} s`;
      $('family-decision').textContent=s.state==='ASSISTANCE_REQUIRED'?'Test alert recorded':s.episode&&!s.episode.open?(t.replay?'Closed in replay':'Recovery recorded'):'Observation in progress';
      $('family-original').textContent=t.replay?`Replay ${s.state} · original ${s.original}`:`Recorded state · ${s.state}`;
      $('family-pose').innerHTML=CareEpisodeReview.poseSVG(data,s,false);
      $('family-pose-caption').textContent=s.count?`${s.count}/12 body joints · ${num(time,2)} s · ${t.poseSource==='recorded_cloud'?'saved cloud pose':'offline presentation pose'}`:'No fresh, supported pose at this selected moment.';
      const moments=milestones(t), active=moments.findLastIndex(m=>m.t<=time);
      container.querySelectorAll('[data-family-moment]').forEach((b,i)=>{b.setAttribute('aria-pressed',String(i===active));b.closest('li').classList.toggle('journey-past',i<active);});
      container.querySelector('.family-overview').dataset.tone=info.tone;
    }
    function mountOperations() {
      const all=aggregate(data.trials), t=trial(), sum=summary(t);time=t.focus;selectedCommand=Math.max(0,sum.commands.length-1);
      const visible=data.trials.map((t,i)=>({t,i})).filter(({t})=>filter==='all'||(filter==='replay'?t.replay:!t.replay));
      container.innerHTML=`<nav class="ops-jump-nav" aria-label="Operations sections"><button type="button" class="button compact" data-ops-jump="comparison">Compare trials ↓</button><button type="button" class="button compact" data-ops-jump="policy">Observation policy ↓</button><button type="button" class="button compact" data-ops-jump="trace">Command trace ↓</button></nav><div class="board-stats">${stat('Saved trials',num(data.trials.length),'Two recorded outcomes · one replay')}${stat('Cloud observations',num(all.observations),'Selected historical observations')}${stat('Applied command ACKs',`${all.acks} / ${all.commands}`,'From the original physical trials')}${stat('Test alerts',num(all.alerts),'No external notification')}</div>
        <section class="panel panel-body section-space"><div class="panel-title"><div><p class="eyebrow">TRANSPORT OVERVIEW</p><h2>Keep capture, transmission and analysis distinct.</h2></div>${badge('THREE SAVED TRIALS')}</div><div class="operations-flow">${[['01','Edge capture',num(all.captured),'locally captured frames'],['02','Selected & sent',num(all.sent),'transmitted frames'],['03','Cloud observations',num(all.observations),'saved observation rows'],['04','Edge applied ACKs',num(all.acks),'acknowledged commands']].map(([n,title,value,label])=>`<div><span>${n}</span><h3>${title}</h3><strong>${value}</strong><small>${label}</small></div>`).join('')}</div><p class="caption board-footnote">Captured frames and transmitted frames are different counts. These totals are not a fixed-versus-adaptive performance benchmark.</p></section>
        <section id="ops-comparison" class="panel panel-body section-space"><div class="panel-title"><div><p class="eyebrow">RUN COMPARISON</p><h2>Select a saved trial</h2></div><label class="board-filter">Show <select id="ops-filter"><option value="all">All outcomes</option><option value="recorded">Recorded decisions</option><option value="replay">Controller replay</option></select></label></div><div class="board-table-scroll"><table class="operations-table"><thead><tr><th>Trial / outcome</th><th>Capture / sent</th><th>Observations</th><th>Applied ACKs</th><th>Image payload</th></tr></thead><tbody>${visible.map(({t:row,i})=>{const s=summary(row);return `<tr class="${i===index?'selected':''}"><th scope="row"><button type="button" data-ops-trial="${i}" aria-pressed="${i===index}"><strong>${esc(row.title)}</strong><span>${row.replay?'Replay '+esc(s.state)+' · original '+esc(s.original):esc(s.state)}</span></button></th><td>${num(s.captured)} / ${num(s.sent)}</td><td>${num(s.observations)}</td><td>${s.acks} / ${s.commands.length}</td><td>${Number.isFinite(s.bytes)?num(s.bytes/1048576,2)+' MiB':'Unavailable'}</td></tr>`;}).join('')}</tbody></table></div></section>
        <section id="ops-policy" class="panel panel-body section-space"><div class="panel-title"><div><p class="eyebrow">SELECTED RUN</p><h2>${esc(t.title)} · observation policy</h2></div>${badge(t.replay?'REPLAY + ORIGINAL DATA':'RECORDED PHYSICAL TRIAL')}</div><div class="board-stats selected-stats">${stat('Capture median',num(t.captureMedian,2)+' FPS','Measured historical capture rate')}${stat('Transmit settings',sum.rates.map(n=>num(n)).join(' → ')+' FPS','Selected settings, not achieved FPS')}${stat('Cloud processing P50',num(sum.p50,1)+' ms','Nearest-rank saved sample percentile')}${stat('Cloud processing P95',num(sum.p95,1)+' ms','Excludes later presentation pose')}</div><div class="board-charts"><div><h3>Selected transmission setting · FPS</h3>${chart(t,'fps','Saved transmission settings over source time','ops-rate-cursor')}</div><div><h3>Saved cloud processing · ms</h3>${chart(t,'processing','Saved cloud processing samples over source time','ops-latency-cursor')}</div></div><div class="ops-time-controls"><label for="ops-seek">Inspect source time</label><input id="ops-seek" type="range" min="0" max="${t.duration}" step="0.01"><output id="ops-clock"></output></div>${lanes(t)}<div class="ops-state-readout"><span>At selected time</span><strong id="ops-state"></strong><span id="ops-original"></span></div></section>
        <div class="operations-detail-grid section-space"><section id="ops-trace" class="panel panel-body"><div class="panel-title"><div><p class="eyebrow">COMMAND TRACE</p><h2>Request → ACK → later evidence</h2></div>${badge(sum.commands.length+' COMMANDS')}</div><div class="ops-command-picker" role="group" aria-label="Recorded command selection">${sum.commands.map((c,i)=>`<button type="button" data-ops-command="${i}" aria-pressed="${i===selectedCommand}"><span>${String(i+1).padStart(2,'0')}</span><strong>${esc(c.label.replace(' · recorded',''))}</strong><small>${num(c.t,3)} s</small></button>`).join('')}</div><div id="ops-command-detail"></div><p class="caption board-footnote">The later observation is the next saved capture after the ACK time. This ordering alone does not verify ROI content or resolution compliance.</p></section>
        <aside class="panel panel-body"><p class="eyebrow">REVIEW &amp; SERVICE CONTEXT</p><h2>Preserve unresolved outcomes.</h2><div class="ops-review-list">${data.trials.map(row=>{const s=summary(row);return `<div><strong>${esc(row.title)}</strong>${badge(s.alert?'OPEN · TEST ALERT':row.replay?'ORIGINAL UNRESOLVED':'CLOSED',s.alert?'danger':row.replay?'warning':'good')}<p>${row.replay?'Replay closes to NORMAL; the original recorded episode remains unresolved.':s.alert?'The saved capture ends with an open episode. No caregiver review record is supplied.':'Recovery closes the recorded episode to NORMAL.'}</p></div>`;}).join('')}</div><div class="notice">This is saved-run analysis. Live service health, caregiver assignment and message delivery are not inferred from these records.</div><a class="button section-space" href="/live">Open technical telemetry ↗</a></aside></div>`;
      container.querySelectorAll('[data-ops-jump]').forEach(b=>b.onclick=()=>$('ops-'+b.dataset.opsJump).scrollIntoView({block:'start'}));
      $('ops-filter').value=filter;
      $('ops-filter').onchange=e=>{filter=e.target.value;const first=data.trials.findIndex(t=>filter==='all'||(filter==='replay'?t.replay:!t.replay));if(first<0)filter='all';else if(!visibleForFilter(trial()))index=first;mountOperations();onSelect(view,trial().key);$('ops-filter').focus({preventScroll:true});};
      container.querySelectorAll('[data-ops-trial]').forEach(b=>b.onclick=()=>{index=Number(b.dataset.opsTrial);mountOperations();onSelect(view,trial().key);container.querySelector(`[data-ops-trial="${index}"]`).focus({preventScroll:true});announce(`${trial().title} operational details selected.`);});
      container.querySelectorAll('[data-ops-command]').forEach(b=>b.onclick=()=>{selectedCommand=Number(b.dataset.opsCommand);const c=summary(trial()).commands[selectedCommand];time=c.nextCapture?.t??c.applied??c.t;updateOperations();});
      $('ops-seek').oninput=e=>{time=Number(e.target.value);updateOperations();};
      updateOperations();
    }
    const visibleForFilter=t=>filter==='all'||(filter==='replay'?t.replay:!t.replay);
    function updateOperations() {
      const t=trial(), s=CareEpisodeReview.snapshot(data,t,time), c=summary(t).commands[selectedCommand];
      $('ops-seek').value=String(time);$('ops-clock').textContent=`${num(time,2)} / ${num(t.duration,2)} s`;
      $('ops-state').textContent=(t.replay?'Replay ':'Recorded ')+s.state;
      $('ops-original').textContent=t.replay?'Original recorded state: '+s.original:'';
      ['ops-rate-cursor','ops-latency-cursor'].forEach(id=>{const x=44+time/t.duration*576;$(id).setAttribute('x1',x);$(id).setAttribute('x2',x);});
      container.querySelectorAll('.board-playhead').forEach(n=>n.style.left=Math.min(99.8,time/t.duration*100)+'%');
      container.querySelectorAll('[data-ops-command]').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===selectedCommand)));
      $('ops-command-detail').innerHTML=c?`<div class="command-checkpoints">${[['Request',c.t,'Controller selected the command'],['Applied ACK',c.applied,'Edge application acknowledged'],['Later capture',c.nextCapture?.t,'Saved cloud observation by source time']].map(([label,t,copy])=>`<div class="${Number.isFinite(t)?'checkpoint-known':''}"><span>${label}</span><strong>${Number.isFinite(t)?num(t,3)+' s':'Not recorded'}</strong><small>${copy}</small></div>`).join('')}</div><dl class="data-list">${Number.isFinite(c.applied)?`<div><dt>Request to ACK</dt><dd>${num((c.applied-c.t)*1000,1)} ms</dd></div>`:''}${c.nextCapture?`<div><dt>Next saved capture after ACK</dt><dd>${num((c.nextCapture.t-c.applied)*1000,1)} ms</dd></div>`:''}</dl>`:'<p class="notice">No command is recorded in this trial.</p>';
    }
    function mount(name) {view=name;if(!data?.trials?.length){empty();return;}if(view==='family')mountFamily();else mountOperations();}
    return {mount,select,selectedKey};
  }
  const api={summary,aggregate,commandTrace,milestones,familyStatus,create};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else globalThis.PublicWorkspaces=api;
})();
