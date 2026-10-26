const el = id => document.getElementById(id);
let csrf = '', signedIn = false, polling = false;
const text = (id, value) => { el(id).textContent = value; };
const fmt = (v, suffix = '', digits = 1) => typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) + suffix : '—';
async function request(path, body) {
  const options = {credentials: 'same-origin', cache: 'no-store'};
  if (body !== undefined) Object.assign(options, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrf}, body: JSON.stringify(body)});
  const response = await fetch(path, options);
  const value = await response.json();
  if (!response.ok) { if (response.status === 401) signOut(); throw new Error(value.error || 'Request failed'); }
  return value;
}
function signOut() { signedIn = false; csrf = ''; el('panel').hidden = true; el('login').hidden = false; text('connection', 'Signed out'); }
function render(s) {
  csrf = s.csrf; signedIn = true; el('panel').hidden = false; el('login').hidden = true;
  const synthetic = s.mode === 'SYNTHETIC_LOCAL_NOT_COOL';
  text('mode', synthetic ? 'Synthetic local demonstration · NOT COOL' : 'AWS Graviton / COOL session');
  text('session', s.session_state); text('connection', s.vision_live ? 'Evidence arriving' : 'No fresh visual evidence');
  el('review').hidden = s.review?.required !== true;
  text('scope', synthetic ? 'Generated motion fixture; no person/fall validation and no AWS/COOL claim.' : (s.person_detector_configured ? 'Guided private person-detection session; experimental decisions.' : 'COOL motion/transport probe only. Person detection and fall verification are not configured.'));
  text('agent', s.agent_state); text('processed', s.counts.processed); text('fps', fmt(s.processed_fps));
  text('target', fmt(s.metrics.target_transmit_fps)); text('latency', fmt(s.metrics.processing_ms, ' ms'));
  text('cycles', s.causal_cycles);
  text('runtime', `OpenCV ${s.runtime.opencv || 'not available'} · COOL verified: ${s.runtime.cool_verified ? 'yes' : 'no'} · phase ${s.runtime.phase ?? '—'}`);
  const seated = s.profile?.seated_closure_seconds;
  text('profile', typeof seated === 'number' ? `Experimental seated-pattern closure: ${fmt(seated, ' s')}. Requires a reviewed scene configuration; physical seat support is unverified.` : 'Seated-pattern closure is not active in this session.');
  text('freshness', s.vision_live ? `Last observation ${fmt(s.observation_age_seconds, ' s')} ago` : 'Analysis is not currently receiving fresh evidence. No safety conclusion can be drawn.');
  text('resources', `Live processed: ${s.counts.live_processed} · Historical processed: ${s.counts.historical_processed} · Process CPU: ${fmt(s.metrics.cpu_percent_one_core, '% of one core')} · RSS: ${fmt(typeof s.metrics.rss_bytes === 'number' ? s.metrics.rss_bytes / 1048576 : null, ' MiB')} · Received: ${s.counts.received} · Explicit drops: ${s.counts.dropped}`);
  el('start').hidden = !s.operator; el('stop').hidden = !s.operator;
  const running = ['STARTING','RUNNING'].includes(s.session_state);
  el('start').disabled = running; el('stop').disabled = !running;
  el('operator-info').hidden = !s.operator || !s.cloud_port;
  text('operator-info', s.cloud_port ? `Operator: cloud loopback WebSocket port ${s.cloud_port}. Connect only through the authorized SSH tunnel.` : '');
  el('timeline').replaceChildren();
  for (const item of [...s.timeline].reverse()) {
    const row = document.createElement('div'); row.className = 'event';
    row.textContent = `${item.index ? '#' + item.index + ' · ' : ''}${item.stage.replaceAll('_',' ')} · ${item.action}`;
    el('timeline').append(row);
  }
}
async function refresh() {
  if (polling) return; polling = true;
  try { render(await request('/api/state')); text('error', ''); }
  catch (e) { if (signedIn) { text('connection','Connection unavailable'); text('freshness','Connection unavailable. Displayed metrics are historical; live analysis is unconfirmed.'); text('error',e.message); } }
  finally { polling = false; }
}
el('login-form').addEventListener('submit', async event => {
  event.preventDefault(); const credential = el('credential').value; el('credential').value = '';
  try { const result = await request('/api/login',{credential}); csrf = result.csrf; await refresh(); }
  catch (e) { text('error', e.message); }
});
for (const action of ['start','stop']) el(action).addEventListener('click',async () => {
  el(action).disabled = true;
  try { await request('/api/' + action,{}); await refresh(); } catch (e) { text('error',e.message); }
});
el('logout').addEventListener('click',async () => { try { await request('/api/logout',{}); signOut(); } catch(e) { text('error',e.message); } });
setInterval(() => { if (signedIn) refresh(); },1000);
refresh();
