import os
import hashlib,io,json,tarfile,time
from pathlib import Path
import sys
from care_vision.aws_demo import SSH,READ_FILE
from care_vision.edge.representation import VisualPolicy
root=Path.cwd();capture=root/'artifacts/new-chair-validation-20260927/trial-1';base=capture/'visibility-evaluation';out=base/'pixel-debug';out.mkdir(exist_ok=False)
r=json.loads((base/'result.json').read_text());mf={f['capture_seq']:f for f in json.loads((capture/'manifest.json').read_text())['frames']};
remote=json.loads((base/'deployment.json').read_text())['experiment'];release=json.loads((base/'deployment.json').read_text())['release']
# Reconstruct exactly the representations by saved command metadata; check lengths.
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as a:
 for row in r['rows']:
  h=row['header'];f=mf[h['capture_seq']];original=(capture/'evidence'/f['filename']).read_bytes();assert hashlib.sha256(original).hexdigest()==f['sha256']
  roi=[h['roi_'+k] for k in ('x','y','width','height')] if h['view_kind']=='roi' else None
  v=VisualPolicy(1280,720,1280,720,view_kind=h['view_kind'],roi=tuple(roi) if roi else None)
  jpg,_=v.render(original);assert len(jpg)==h['payload_bytes']
  info=tarfile.TarInfo(f"received/{h['capture_seq']:06d}.jpg");info.size=len(jpg);a.addfile(info,io.BytesIO(jpg))
worker=r'''import json,sys,math
from pathlib import Path
import cv2,numpy as np
from care_vision.cloud.occlusion_fragments import OcclusionFragments
base=Path(sys.argv[1]);r=json.loads((base/'result.json').read_text());tracker=OcclusionFragments(visibility_probes=True);debug=base/'pixel-debug';debug.mkdir(exist_ok=True);records=[];cv2.setNumThreads(1)
for row in r['rows']:
 h=row['header'];image=cv2.imread(str(base/'received'/f"{h['capture_seq']:06d}.jpg"));visible=None
 if h['view_kind']=='roi':
  x,y,w,hh=[h['roi_'+k] for k in ('x','y','width','height')];s=min(1280/w,720/hh);rw,rh=round(w*s),round(hh*s);left=(1280-rw)//2;top=(720-rh)//2;crop=cv2.resize(image[top:top+rh,left:left+rw],(w,hh));canvas=np.zeros((720,1280,3),np.uint8);canvas[y:y+hh,x:x+w]=crop;visible=[x,y,w,hh]
 else:canvas=image
 state,small,fg=tracker.update(canvas,row['time'],row['frame_index'],row['people'],visible_roi=visible)
 assert state['episodes']==row['episodes'],(row['time'],state['episodes'],row['episodes'])
 if 81.4<=row['time']<=86.6 and row['people'] is not None:
  g=tracker.groups[-1];x,y,w,hh=g['ledger'].reference_bbox;x0=max(0,round(x+w/2-.75*hh));x1=min(640,round(x+w/2+.75*hh));y0=max(0,round(y+hh+.05*hh));y1=min(360,round(y+1.6*hh));low=fg[y0:y1,x0:x1];ys,xs=np.where(low>0)
  item={'time':row['time'],'seq':h['capture_seq'],'region':[x0,y0,x1-x0,y1-y0],'count':len(xs),'min_area':.015*w*hh,'min_width':.25*hh,'width':int(xs.max()-xs.min()+1) if len(xs) else 0,'height':int(ys.max()-ys.min()+1) if len(xs) else 0,'aspect':float((xs.max()-xs.min()+1)/(ys.max()-ys.min()+1)) if len(xs) else None,'episodes':state['episodes']};records.append(item)
  cv2.rectangle(small,(x0,y0),(x1,y1),(255,255,0),2);name=f"{h['capture_seq']:06d}.jpg";cv2.imwrite(str(debug/name),np.hstack([small,cv2.cvtColor(fg,cv2.COLOR_GRAY2BGR)]));print(json.dumps(item),flush=True)
(base/'pixel-debug.json').write_text(json.dumps(records))
'''
ssh=SSH(os.environ['CARE_VISION_SSH_HOST'],Path(os.environ['CARE_VISION_SSH_IDENTITY']))
ssh.python('''import io,pathlib,sys,tarfile
base=pathlib.Path(sys.argv[1])
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as a:
 for m in a.getmembers():
  p=pathlib.PurePosixPath(m.name)
  if not m.isfile() or p.is_absolute() or '..' in p.parts:raise ValueError('bad path')
  f=base/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(a.extractfile(m).read())
''',remote,payload=b.getvalue(),timeout=180)
ssh.python('import pathlib,sys;pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())',remote+'/debug-worker.py',payload=worker.encode())
print(ssh.python('import subprocess,sys\np=subprocess.run(sys.argv[1:],capture_output=True,text=True)\nprint(p.stdout)\nprint(p.stderr,file=sys.stderr)\nraise SystemExit(p.returncode)','env','PYTHONNOUSERSITE=1',f'PYTHONPATH={remote}/src:/opt/cool/python_3.11/site-packages','LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib',release+'/.venv/bin/python',remote+'/debug-worker.py',remote,timeout=120).decode())
data=ssh.python(READ_FILE,remote+'/pixel-debug.json');(out/'measurements.json').write_bytes(data)
for x in json.loads(data):
 name=f"{x['seq']:06d}.jpg";(out/name).write_bytes(ssh.python(READ_FILE,remote+'/pixel-debug/'+name))
print('SAVED_DEBUG',flush=True)
