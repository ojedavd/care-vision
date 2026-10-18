'use strict';
const narration = document.getElementById('narration');
const cues = window.NARRATION_CUES;
let lastCue = -1;
const minuteClock = t => Math.floor(t / 60) + ':' + String(Math.floor(t % 60)).padStart(2, '0');
function lockPanel(locked) {
  document.querySelectorAll('button,input,select').forEach(el => {
    if (!el.closest('#rehearsal')) el.disabled = locked;
  });
}
function syncNarration(force = false) {
  const t = narration.currentTime;
  const cueIndex = cues.findLastIndex(c => c.time <= t);
  document.getElementById('narrate-time').textContent = minuteClock(t) + ' / 4:00';
  document.getElementById('narrate-seek').value = t;
  if (cueIndex < 0 || (!force && cueIndex === lastCue)) return;
  lastCue = cueIndex;
  const cue = cues[cueIndex];
  choose(cue.trial);
  index = rows.findIndex(r => r.seq === cue.seq);
  if (index < 0) throw new Error('Missing narrated observation');
  draw();
  document.getElementById('narrate-title').textContent = cue.title;
  document.getElementById('narrate-moment').textContent = cue.moment + ' · fuente ' + fmt(cue.sourceTime, 3) + ' s';
  document.getElementById('narrate-text').textContent = cue.text;
  lockPanel(!narration.paused);
}
function audioState() {
  const done = narration.ended;
  document.getElementById('narrate-play').textContent = done ? 'Volver a escuchar' :
    narration.paused ? (narration.currentTime > 0 ? 'Reanudar ensayo' : 'Iniciar ensayo narrado') : 'Pausar ensayo';
  document.getElementById('narrate-status').textContent = done ? 'Ensayo finalizado' :
    narration.paused ? 'En pausa' : 'Narración en curso';
  lockPanel(!narration.paused);
}
document.getElementById('narrate-play').onclick = async () => {
  if (!narration.paused) { narration.pause(); return; }
  if (narration.ended) narration.currentTime = 0;
  stop();
  syncNarration(true);
  try {
    await narration.play();
    document.getElementById('rehearsal').scrollIntoView({block: 'start'});
  } catch (error) {
    document.getElementById('narrate-error').textContent = 'No se pudo iniciar el audio local: ' + error.message;
    audioState();
  }
};
document.getElementById('narrate-reset').onclick = () => {
  narration.pause(); narration.currentTime = 0; syncNarration(true); audioState();
};
document.getElementById('narrate-seek').oninput = event => {
  narration.currentTime = Number(event.target.value); syncNarration(true); audioState();
};
narration.addEventListener('timeupdate', () => syncNarration());
narration.addEventListener('play', audioState);
narration.addEventListener('pause', audioState);
narration.addEventListener('ended', () => { syncNarration(); audioState(); });
narration.addEventListener('error', () => {
  document.getElementById('narrate-error').textContent = 'No se pudo cargar narration.m4a.';
  audioState();
});
syncNarration(true);
