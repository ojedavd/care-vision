"""Export recorded frames/timestamps for AWS evaluation; no local inference."""
import hashlib
import json
import sys
from pathlib import Path
import cv2
from care_vision.edge.sampler import Sampler

base=Path(sys.argv[1]);target=base/'evidence';target.mkdir(exist_ok=False)
log=[json.loads(x) for x in (base/'demo.captures.jsonl').read_text().splitlines()]
video=cv2.VideoCapture(str(base/'demo.avi'));origin=log[0]['captured_mono_ns']
samplers={n:Sampler(n) for n in (2,20)};schedules={str(n):[] for n in samplers};frames=[]
for i,record in enumerate(log):
 ok,image=video.read();assert ok
 ns=record['captured_mono_ns']-origin;seq=record['capture_seq']
 selected=False
 for fps,sampler in samplers.items():
  if sampler.select(seq,ns):schedules[str(fps)].append(seq);selected=True
 if not selected:continue
 name=f'{seq:06d}.jpg';ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,80]);assert ok
 data=encoded.tobytes();(target/name).write_bytes(data)
 frames.append({'capture_seq':seq,'frame_index':i,'source_time_seconds':ns/1e9,
                'captured_mono_ns':record['captured_mono_ns'],'filename':name,
                'width':1280,'height':720,'sha256':hashlib.sha256(data).hexdigest()})
assert not video.read()[0];video.release()
report={'frames':frames,'schedules':schedules,'source_sha256':hashlib.sha256((base/'demo.avi').read_bytes()).hexdigest(),
        'method':'MJPEG recording decoded and encoded as JPEG80; actual capture timestamps retained.'}
(base/'manifest.json').write_text(json.dumps(report)+'\n')
print('Prepared',len(frames),'JPEGs; 2FPS',len(schedules['2']),'20FPS',len(schedules['20']))
