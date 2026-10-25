/* Public presentation only. Input is the already reviewed /trials/data.js export. */
(() => {
  'use strict';
  const links = [[0,1],[0,2],[2,4],[1,3],[3,5],[0,6],[1,7],[6,7],[6,8],[8,10],[7,9],[9,11]];
  const esc = v => String(v ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (n, d=2) => Number(n).toFixed(d);
  function last(rows, time) {
    let lo=0, hi=rows.length;
    while(lo<hi) { const m=(lo+hi)>>1; if(rows[m].t<=time+1e-8) lo=m+1; else hi=m; }
    return rows[lo-1];
  }
  function snapshot(data, trial, time) {
    const frame=last(trial.frames,time), episode=last(trial.episodes,time);
    const state=last(trial.states,time)?.state ?? 'UNKNOWN';
    const original=last(trial.recordedStates || trial.states,time)?.state ?? 'UNKNOWN';
    const fresh=!!frame && time-frame.t<=1;
    const points=frame?.pose?.bodyPoints;
    const supported=Array.from({length:12},(_,i)=> {
      const p=points?.[i];
      return !!(fresh && frame.confirmed && frame.pose.status==='ok' && points.length===12
        && p?.length===3 && p.every(Number.isFinite) && p[2]>=data.poseDisplayResponse
        && p[0]>=0 && p[0]<data.sourceWidth && p[1]>=0 && p[1]<data.sourceHeight);
    });
    return {frame, episode, state, original, fresh, points, supported,
      count:supported.filter(Boolean).length, alert:!!episode?.alert};
  }
  function poseSVG(data, s, box) {
    let shapes='';
    let view=[0,0,data.sourceWidth,data.sourceHeight];
    if(s.count) {
      const body=s.points.filter((_,i)=>s.supported[i]);
      const xs=body.map(p=>p[0]), ys=body.map(p=>p[1]);
      if(s.frame.bbox?.length===4 && s.frame.bbox.every(Number.isFinite)) {
        const [x,y,w,h]=s.frame.bbox; xs.push(x,x+w);ys.push(y,y+h);
      }
      const w=Math.min(data.sourceWidth,Math.max(320,Math.max(...xs)-Math.min(...xs)+120));
      const h=Math.min(data.sourceHeight,Math.max(240,Math.max(...ys)-Math.min(...ys)+120));
      view=[Math.max(0,Math.min(data.sourceWidth-w,(Math.min(...xs)+Math.max(...xs)-w)/2)),
        Math.max(0,Math.min(data.sourceHeight-h,(Math.min(...ys)+Math.max(...ys)-h)/2)),w,h];
    }
    if(box && s.fresh && s.frame?.confirmed && s.frame.bbox?.length===4
      && s.frame.bbox.every(Number.isFinite)) {
      const [x,y,w,h]=s.frame.bbox;
      shapes+=`<rect class="care-box" x="${x}" y="${y}" width="${w}" height="${h}"/>`;
    }
    links.forEach(([a,b])=> {
      if(!s.supported[a] || !s.supported[b]) return;
      shapes+=`<line x1="${s.points[a][0]}" y1="${s.points[a][1]}" x2="${s.points[b][0]}" y2="${s.points[b][1]}"/>`;
    });
    s.supported.forEach((show,i)=> {
      if(show) shapes+=`<circle cx="${s.points[i][0]}" cy="${s.points[i][1]}" r="7"><title>${esc(data.bodyJointNames[i])}</title></circle>`;
    });
    return `<svg viewBox="${view.join(' ')}" role="img" aria-label="Saved body pose, centered on the person; neutral background and no camera imagery">${shapes}</svg>`;
  }
  function explanation(trial,s) {
    if(trial.replay && s.state==='NORMAL' && s.episode && !s.episode.open)
      return ['Seated closure in replay.','The replay closes this episode. The original recorded controller remained UNRESOLVED.'];
    if(s.alert) return ['A person’s review was requested.','The recorded controller emitted a test alert after no recovery. No external notification was sent.'];
    if(s.episode && !s.episode.open) return ['Recovery closed the episode.','The recorded controller returned to NORMAL after recovery evidence.'];
    if(s.state==='VERIFYING') return ['More evidence was requested.','The controller is verifying the change before resolving the episode.'];
    if(s.state==='UNRESOLVED') return ['The episode remains open.','The saved evidence is insufficient for closure at this point.'];
    return ['Baseline observation.','No episode has opened at this point in the saved trial.'];
  }
  function create(container,data,announce=()=>{},onSelect=()=>{}) {
    let index=0, time=data?.trials?.[0]?.focus ?? 0, playing=false, tickId=null, previous=null, box=false;
    const $=id=>container.querySelector('#care-'+id);
    const trial=()=>data.trials[index];
    function pause() {
      playing=false; previous=null;
      if(tickId!==null) cancelAnimationFrame(tickId);
      tickId=null;
      if($('play')) $('play').textContent='Play from here';
    }
    function seek(t) { pause(); time=Math.max(0,Math.min(trial().duration,t)); update(); }
    function update() {
      const t=trial(), s=snapshot(data,t,time), [headline,copy]=explanation(t,s);
      $('pose').innerHTML=poseSVG(data,s,box);
      $('clock').textContent=`${fmt(time)} / ${fmt(t.duration)} s`;
      $('seek').value=String(time);
      $('state').textContent=s.state;
      $('state').className='badge '+(s.alert?'danger':s.state==='NORMAL'?'good':'warning');
      $('state-origin').textContent=t.replay?'Controller replay':'Recorded controller';
      $('original').textContent=s.original;
      $('headline').textContent=headline; $('copy').textContent=copy;
      $('result').className='care-hero '+(s.alert?'danger':s.state==='NORMAL'?'':'warning');
      $('episode-state').textContent=!s.episode?'Not opened yet':s.episode.open?'Open':'Closed';
      $('alert').textContent=s.alert?'Emitted · test only':'Not emitted';
      $('fps').textContent=s.frame?`${s.frame.fps} FPS`:'No observation yet';
      $('age').textContent=s.frame?`Saved observation ${fmt(s.frame.t,3)} s · age ${fmt(time-s.frame.t)} s`:'No saved observation yet';
      $('pose-caption').textContent=!s.fresh?'No fresh pose at this time · samples over 1 s old are hidden':s.count?`${s.count}/12 body joints · ${t.poseSource==='recorded_cloud'?'saved cloud pose':'offline presentation pose'}`:s.frame?.pose?.status==='not_requested'?'No pose was requested for this saved observation':'No supported pose in this observation';
      container.querySelectorAll('[data-care-event]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.careEvent)<=time && Number(b.dataset.careEvent)===last(t.events,time)?.t)));
    }
    function mount() {
      pause();
      if(!data?.trials?.length) {
        container.innerHTML='<section class="panel panel-body"><h2>Episode data unavailable</h2><p class="reason">Open the public episode review to inspect saved evidence. No live connection is active.</p><a class="button section-space" href="/trials/">Open episode evidence ↗</a></section>';
        return;
      }
      const t=trial();
      container.innerHTML=`<div class="care-episodes" role="group" aria-label="Saved episode selection">${data.trials.map((row,i)=>`<button type="button" data-care-trial="${i}" aria-pressed="${i===index}"><strong>${esc(row.title)}</strong><span>${esc(row.subtitle)}</span><small>${row.replay?'CONTROLLER REPLAY':'RECORDED DECISIONS'}</small></button>`).join('')}</div>
        <div class="grid-two care-recorded-grid">
          <section class="panel panel-body"><div class="panel-title"><div><p class="eyebrow">SAVED EVIDENCE</p><h2>${esc(t.title)}</h2></div><span class="badge info">BODY POSE</span></div>
            <div id="care-pose" class="care-pose"></div><p id="care-pose-caption" class="caption care-pose-caption"></p>
            <div class="care-player"><button type="button" class="button primary compact" id="care-play">Play from here</button><button type="button" class="button compact" id="care-start">From start</button><label>Speed <select id="care-speed"><option value="1">1×</option><option value="0.5">0.5×</option><option value="2">2×</option></select></label><output id="care-clock"></output></div>
            <label class="sr-only" for="care-seek">Episode source time in seconds</label><input id="care-seek" class="care-seek" type="range" min="0" max="${t.duration}" step="0.01" value="${time}">
            <div class="care-pose-options"><label><input type="checkbox" id="care-show-box" ${box?'checked':''}> Detection box</label><span id="care-age" class="caption"></span></div>
            <div class="care-moments" aria-label="Episode milestones">${t.events.filter(e=>e.kind==='episode'&&!e.label.startsWith('End')).map(e=>`<button type="button" class="button compact" data-care-event="${e.t}">${esc(e.label)}<small>${fmt(e.t,3)} s</small></button>`).join('')}<button type="button" class="button compact" data-care-seek="${t.duration}">End</button></div>
          </section>
          <section class="panel panel-body"><div class="panel-title"><p class="eyebrow" id="care-state-origin"></p><span id="care-state"></span></div><div id="care-result" class="care-hero"><p class="hero-kicker">AT THE SELECTED TIME</p><h2 class="hero-title" id="care-headline"></h2><p class="hero-copy" id="care-copy"></p></div>
            <dl class="case-list care-facts"><div class="case-row"><dt>Episode</dt><dd id="care-episode-state"></dd></div><div class="case-row"><dt>Original recorded state</dt><dd id="care-original"></dd></div><div class="case-row"><dt>Test alert</dt><dd id="care-alert"></dd></div><div class="case-row"><dt>External notification</dt><dd>Not sent</dd></div><div class="case-row"><dt>Selected transmission setting</dt><dd id="care-fps"></dd></div></dl>
            <div class="notice">No caregiver was assigned through this review. Human-review buttons are available in the separate Scenario walkthrough.</div><a class="button section-space" href="/trials/">Inspect decisions, commands &amp; ACKs ↗</a>
          </section>
        </div>
        <section class="panel panel-body section-space care-provenance"><h2>Where this evidence comes from</h2><p>${t.poseSource==='recorded_cloud'?'Sitting uses the saved cloud pose. Its closure is a controller replay; the original observations and UNRESOLVED state remain visible.':'The body pose was calculated offline from the archived recording for presentation. It was not used by the recorded agent. Controller decisions and transmission settings come from the original trial.'}</p><p>Only body joints are shown. No camera images, video, audio or facial landmarks are included. Missing pose stays missing; estimated joints may be wrong under occlusion. This view cannot establish physical seat support or validate fall-detection accuracy.</p></section>`;
      container.querySelectorAll('[data-care-trial]').forEach(b=>b.onclick=()=> {
        pause(); index=Number(b.dataset.careTrial); time=trial().focus; mount();onSelect(trial().key);
        container.querySelector(`[data-care-trial="${index}"]`).focus({preventScroll:true});
        announce(`${trial().title} saved episode selected.`);
      });
      container.querySelectorAll('[data-care-event]').forEach(b=>b.onclick=()=>seek(Number(b.dataset.careEvent)));
      container.querySelectorAll('[data-care-seek]').forEach(b=>b.onclick=()=>seek(Number(b.dataset.careSeek)));
      $('seek').oninput=e=>seek(Number(e.target.value));
      $('show-box').onchange=e=>{box=e.target.checked;update();};
      $('start').onclick=()=>seek(0);
      $('play').onclick=()=> {
        if(playing) {pause();return;}
        if(time>=trial().duration) time=0;
        playing=true; previous=null; $('play').textContent='Pause playback';
        tickId=requestAnimationFrame(tick);
      };
      update();
    }
    function tick(now) {
      if(!playing) return;
      if(previous!==null) time=Math.min(trial().duration,time+(now-previous)/1000*Number($('speed').value));
      previous=now; update();
      if(time>=trial().duration) pause(); else tickId=requestAnimationFrame(tick);
    }
    function select(key) {
      const i=data?.trials?.findIndex(t=>t.key===key);
      if(i>=0){pause();index=i;time=trial().focus;return true;}
      return false;
    }
    return {mount,pause,select,selectedKey:()=>data?.trials?.[index]?.key};
  }
  const api={snapshot,poseSVG,create};
  if(typeof module!=='undefined' && module.exports) module.exports=api;
  else globalThis.CareEpisodeReview=api;
})();
