import os
import ast,io,json,tarfile
from pathlib import Path
from care_vision.aws_demo import SSH,READ_FILE
root=Path.cwd();base=root/'artifacts/new-chair-validation-20260927/trial-1/visibility-evaluation';deployment=json.loads((base/'deployment.json').read_text());remote=deployment['experiment'];release=deployment['release'];ssh=SSH(os.environ['CARE_VISION_SSH_HOST'],Path(os.environ['CARE_VISION_SSH_IDENTITY']))
node=ast.parse((root/'tools/debug_visibility_pixels.py').read_text());worker=next(ast.literal_eval(n.value) for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='worker' for t in n.targets))
worker=worker.replace("records=[];cv2.setNumThreads(1)","records=[];rows=[];cv2.setNumThreads(1)")
worker=worker.replace(" assert state['episodes']==row['episodes'],(row['time'],state['episodes'],row['episodes'])"," rows.append(state)")
worker=worker.replace(" if 81.4<=row['time']<=86.6 and row['people'] is not None:"," if False:")
worker+="\n(base/'analysis-v2-result.json').write_text(json.dumps({'rows':rows,'episodes':[g['ledger'].finish() for g in tracker.groups]}))\nprint(json.dumps([g['ledger'].snapshot() for g in tracker.groups]))\n"
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as a:
 for p in root.glob('src/**/*.py'):a.add(p,arcname=str(p.relative_to(root)))
ssh.python('''import io,pathlib,sys,tarfile
base=pathlib.Path(sys.argv[1]);base.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as a:
 for m in a.getmembers():
  p=pathlib.PurePosixPath(m.name)
  if not m.isfile() or p.is_absolute() or '..' in p.parts:raise ValueError('bad path')
  f=base/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(a.extractfile(m).read())
''',remote+'/analysis-v2',payload=b.getvalue())
ssh.python('import pathlib,sys;pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())',remote+'/analysis-v2-worker.py',payload=worker.encode())
print(ssh.python('import subprocess,sys\np=subprocess.run(sys.argv[1:],capture_output=True,text=True)\nprint(p.stdout)\nprint(p.stderr,file=sys.stderr)\nraise SystemExit(p.returncode)','env','PYTHONNOUSERSITE=1',f'PYTHONPATH={remote}/analysis-v2/src:/opt/cool/python_3.11/site-packages','LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib',release+'/.venv/bin/python',remote+'/analysis-v2-worker.py',remote,timeout=120).decode(),flush=True)
(base/'analysis-v2-result.json').write_bytes(ssh.python(READ_FILE,remote+'/analysis-v2-result.json'))
