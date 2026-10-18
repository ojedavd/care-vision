import os
import hashlib,io,json,shlex,tarfile,time
from pathlib import Path
from care_vision.aws_demo import SSH,READ_FILE
root=Path.cwd();out=root/'artifacts/visibility-activation-20260927/aws-controls';out.mkdir(exist_ok=False)
prior=json.loads((root/'artifacts/occlusion-continuity-20260927/verified-v5/plan.json').read_text());jobs=[j for j in prior['jobs'] if not j['id'].startswith('chair-') or j['id']=='chair-nanodet-adaptive']
files=list(root.glob('src/**/*.py'))
worker=(root/'tools/evaluate_occlusion_fragments.py').read_text().replace('tracker = OcclusionFragments()','tracker = OcclusionFragments(visibility_probes=True)')
plan={'jobs':jobs,'visibility_probes':True,'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'worker_sha256':hashlib.sha256(worker.encode()).hexdigest(),'scope':'AWS COOL image regression with original full frames and cached AWS 2FPS person observations; not ROI command-loop or fresh NanoDet inference.'}
(out/'plan.json').write_text(json.dumps(plan)+'\n');(out/'worker.py').write_text(worker)
remote='/home/ubuntu/care-vision-experiments/visibility-controls-'+str(int(time.time()));release=json.loads((root/'artifacts/pose-occlusion-20260927/deployment.json').read_text())['existing_release'];(out/'deployment.json').write_text(json.dumps({'experiment':remote,'release':release})+'\n')
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as a:
 for p in files:a.add(p,arcname=str(p.relative_to(root)))
 a.add(out/'worker.py',arcname='worker.py');a.add(out/'plan.json',arcname='plan.json')
ssh=SSH(os.environ['CARE_VISION_SSH_HOST'],Path(os.environ['CARE_VISION_SSH_IDENTITY']))
ssh.python('''import io,pathlib,sys,tarfile
base=pathlib.Path(sys.argv[1]);base.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as a:
 for m in a.getmembers():
  p=pathlib.PurePosixPath(m.name)
  if not m.isfile() or p.is_absolute() or '..' in p.parts:raise ValueError('bad path')
  f=base/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(a.extractfile(m).read())
''',remote,payload=b.getvalue())
cmd=['env','PYTHONNOUSERSITE=1',f'PYTHONPATH={remote}/src:/opt/cool/python_3.11/site-packages','LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib','timeout','180',release+'/.venv/bin/python','-u',remote+'/worker.py',remote]
print(ssh.python('import subprocess,sys\np=subprocess.run(sys.argv[1:],capture_output=True,text=True)\nprint(p.stdout)\nprint(p.stderr,file=sys.stderr)\nraise SystemExit(p.returncode)',*cmd,timeout=200).decode(),flush=True)
(out/'result.json').write_bytes(ssh.python(READ_FILE,remote+'/result.json'))
print('CONTROL_RESULTS_SAVED',flush=True)
