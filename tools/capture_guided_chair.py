"""Bounded camera capture with local spoken prompts; no Edge CV inference."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from care_vision.edge.source import CameraSource

base=Path(sys.argv[1]);base.mkdir(parents=True,exist_ok=False)
(base/'capture.pid').write_text(str(os.getpid()))
prompts=[
(0,'baseline','La grabación empezó. Quedate de pie, mirando hacia la cámara, con el cuerpo completo visible.'),
(15,'sit','Ahora sentate normalmente en la silla. Quedate sentado hasta la siguiente indicación.'),
(30,'stand','Ahora levantate normalmente y quedate de pie.'),
(43,'bend','Agachate a recoger un objeto liviano o simulá recogerlo. Después volvé a quedar de pie.'),
(57,'walk_behind','Caminá despacio por detrás de la silla y volvé a tu posición de pie.'),
(72,'controlled_descent','Ahora bajá lentamente hasta acostarte en el suelo detrás de la silla. No te dejes caer. Cuando estés acostado, quedate allí hasta que te indique levantarte.'),
(88,'floor_rest','Seguí acostado y tranquilo. Todavía no te levantes.'),
(100,'recovery','Ahora incorporáte y levantate despacio. Después quedate de pie dentro del encuadre.'),
(116,'finish_hold','Estamos terminando. Quedate de pie unos segundos más.')]
plan={'seconds':120,'camera_index':0,'capture_fps':30,'width':1280,'height':720,
      'prompts':[{'target_seconds':t,'action':a,'text':s} for t,a,s in prompts],
      'labels_are_instructions_not_ground_truth':True,'alerts_enabled':False,
      'cv_on_edge':False,'aws_host':'54.81.194.90','thresholds_frozen':True}
(base/'protocol.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
root=Path.cwd();frozen={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
 for p in (root/'src/care_vision/cloud').glob('*.py')}
(base/'frozen-cloud-source.json').write_text(json.dumps(frozen,indent=2)+'\n')
def say(text):return subprocess.Popen(['/usr/bin/say','-v','Paulina','-r','145',text])
voice=say('Vamos a empezar en ocho segundos. Preparáte frente a la cámara.');voice.wait(timeout=15)
time.sleep(3)
source=CameraSource(index=0,seconds=120,fps=30,width=1280,height=720,record=base/'demo.avi')
source.start();start=None;next_prompt=0;spoken=[]
try:
 while True:
  capture,count,done,error=source.snapshot()
  if error:raise RuntimeError(error)
  if capture and start is None:
   start=capture.mono_ns;print('CAPTURE_STARTED',flush=True)
   (base/'first-frame.jpg').write_bytes(capture.jpeg)
  elapsed=0 if start is None else (time.monotonic_ns()-start)/1e9
  if start is not None and next_prompt<len(prompts) and elapsed>=prompts[next_prompt][0]:
   target,action,text=prompts[next_prompt]
   if voice.poll() is None:voice.terminate();voice.wait(timeout=3)
   voice=say(text)
   spoken.append({'target_seconds':target,'actual_prompt_seconds':elapsed,'action':action,'text':text,
                  'capture_seq_at_prompt':capture.seq,'prompt_mono_ns':time.monotonic_ns()})
   (base/'spoken-prompts.json').write_text(json.dumps(spoken,ensure_ascii=False,indent=2)+'\n')
   print('PROMPT',round(elapsed,2),action,'frames',count,flush=True);next_prompt+=1
  if done:break
  time.sleep(.03)
finally:
 source.close()
 if voice.poll() is None:voice.terminate();voice.wait(timeout=3)
_,count,done,error=source.snapshot()
if error:raise RuntimeError(error)
assert done and count>0
log=[json.loads(s) for s in (base/'demo.captures.jsonl').read_text().splitlines()]
duration=(log[-1]['captured_mono_ns']-log[0]['captured_mono_ns'])/1e9
report={'frames':count,'measured_span_seconds':duration,'measured_capture_fps':(count-1)/duration,
        'source_sha256':hashlib.sha256((base/'demo.avi').read_bytes()).hexdigest(),
        'camera_closed':True,'spoken_prompts':len(spoken),'cv_inference_performed':False}
(base/'capture-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
say('Terminó la grabación. Ya podés relajarte. Ahora voy a revisar la evidencia y analizarla en AWS.').wait(timeout=20)
