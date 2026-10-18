import os
import hashlib
import io
import json
import shlex
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from care_vision.aws_demo import SSH, READ_FILE
from care_vision.common.contracts import envelope, pack_frame, utc, validate
from care_vision.edge.commands import CommandHandler
from care_vision.edge.representation import VisualPolicy
from care_vision.edge.sampler import Sampler

root=Path.cwd();capture=Path(sys.argv[1]).resolve();out=capture/'aws-evaluation';out.mkdir(exist_ok=False)
ssh=SSH(os.environ['CARE_VISION_SSH_HOST'],Path(os.environ['CARE_VISION_SSH_IDENTITY']))
prior=json.loads((root/'artifacts/occlusion-roi-20260927/trial-3/plan.json').read_text())
manifest=json.loads((capture/'manifest.json').read_text())
prior['frames']=[{'frame_index':f['frame_index'],'time':f['source_time_seconds'],'sha256':f['sha256']} for f in manifest['frames']]
prior['semantic_frames']=[f['frame_index'] for f in manifest['frames'] if f['capture_seq'] in set(manifest['schedules']['2'])]
images={f['frame_index']:f for f in manifest['frames']}
frozen=json.loads((capture/'frozen-cloud-source.json').read_text())
for name,sha in frozen.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
files=list(root.glob('src/**/*.py'))+[root/'tools/occlusion_command_cloud.py',Path(__file__).resolve()]
plan={'model':prior['model'],'model_sha256':prior['model_sha256'],
      'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
      'scope':'Real Mac/AWS SSH roundtrips using recorded capture time, lockstep (not real-time FPS).'}
(out/'plan.json').write_text(json.dumps(plan)+'\n')
remote='/home/ubuntu/care-vision-experiments/new-chair-'+str(int(time.time()))
release=json.loads((root/'artifacts/pose-occlusion-20260927/deployment.json').read_text())['existing_release']
(out/'deployment.json').write_text(json.dumps({'experiment':remote,'release':release})+'\n')
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as a:
 for p in files:a.add(p,arcname=str(p.relative_to(root)))
 a.add(out/'plan.json',arcname='plan.json')
ssh.python('''import io,pathlib,sys,tarfile
base=pathlib.Path(sys.argv[1]);base.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as a:
 for m in a.getmembers():
  p=pathlib.PurePosixPath(m.name)
  if not m.isfile() or p.is_absolute() or '..' in p.parts:raise ValueError('bad path')
  f=base/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(a.extractfile(m).read())
''',remote,payload=b.getvalue())
command=['env','PYTHONNOUSERSITE=1',f'PYTHONPATH={remote}/src:/opt/cool/python_3.11/site-packages',
         'LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib','timeout','900',release+'/.venv/bin/python',
         '-u',remote+'/tools/occlusion_command_cloud.py',remote]
identity={'edge_id':'mac-recorded-roi','boot_id':'occlusion-replay','session_id':'new-chair-'+str(int(time.time()))}
visual=VisualPolicy(1280,720,1280,720);sampler=Sampler(2);handler=CommandHandler(identity,sampler,30,visual_policy=visual)
semantic=set(prior['semantic_frames']);rows=[];acks=[];jpeg_bytes=0;wire_bytes=0
started=time.monotonic()
with (out/'cloud.stderr.log').open('w') as err:
 process=subprocess.Popen([*ssh.args,ssh.host,shlex.join(command)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
 def exchange(payload):
  global wire_bytes
  wire=struct.pack('!I',len(payload))+payload;wire_bytes+=len(wire)
  process.stdin.write(wire);process.stdin.flush()
  line=process.stdout.readline()
  if not line:raise RuntimeError((out/'cloud.stderr.log').read_text())
  return json.loads(line)
 try:
  for f in prior['frames']:
   if sampler.fps==2 and f['frame_index'] not in semantic:continue
   meta=images[f['frame_index']];original=(capture/'evidence'/meta['filename']).read_bytes()
   assert hashlib.sha256(original).hexdigest()==f['sha256']
   ns=round(f['time']*1e9);capture_seq=f['frame_index']+1
   jpeg,view=visual.render(original,ns);jpeg_bytes+=len(jpeg)
   header=envelope('FRAME',identity,frame_id=f"{identity['boot_id']}:{capture_seq}",capture_seq=capture_seq,
       captured_mono_ns=ns,sampled_mono_ns=ns,captured_at_utc=utc(),encoding='jpeg',width=1280,height=720,
       payload_bytes=len(jpeg),applied_command_id=handler.command_id,policy_seq=handler.policy_seq,
       target_transmit_fps=sampler.fps,evidence_kind='live',**view)
   response=exchange(pack_frame(header,jpeg));rows.append({'header':header,'state':response['state']})
   for c in response['commands']:
    ack=handler.apply(c,ns,capture_seq);validate(ack)
    assert ack['result']=='applied'
    assert exchange(json.dumps(ack).encode())['acknowledged']==c['command_id'];acks.append(ack)
    print(round(f['time'],2),c['type'],c.get('transmit_fps',c.get('context_fps')),flush=True)
   if len(rows)%100==0:print('FRAMES',len(rows),'source time',round(f['time'],2),flush=True)
  end=exchange(json.dumps({'control':'finish'}).encode());process.stdin.close();assert process.wait(timeout=30)==0
 except BaseException:
  process.kill();process.wait();raise
elapsed=time.monotonic()-started
(out/'edge.json').write_text(json.dumps({'rows':rows,'acks':acks,'jpeg_bytes':jpeg_bytes,
    'framed_edge_to_cloud_bytes':wire_bytes,'wall_seconds':elapsed,'finish':end})+'\n')
(out/'result.json').write_bytes(ssh.python(READ_FILE,remote+'/result.json'))
print(json.dumps(end), 'wall_seconds',elapsed,flush=True)
