"""Remote preparation scripts for an isolated locked environment, never an SDK installer."""

PREPARE_RUNTIME = r"""
import json,os,pathlib,platform,subprocess,sys
base,python,uv=map(pathlib.Path,sys.argv[1:4])
if platform.system()!='Linux' or platform.machine() not in {'aarch64','arm64'}:
 raise SystemExit('requires Linux ARM64 Graviton')
if 'OPENCV_FORCE_DNN_ENGINE' in os.environ:
 raise SystemExit('unset OPENCV_FORCE_DNN_ENGINE; no override accepted')
if not all(p.is_absolute() and p.is_file() for p in (python,uv)):
 raise SystemExit('explicit absolute Python and uv executables required')
if not pathlib.Path('/opt/cool/python_3.11/site-packages/cv2').is_dir():
 raise SystemExit('preinstalled COOL Python 3.11 SDK required; not installed by this tool')
version=subprocess.check_output([str(python),'-I','-c',
 'import json,sys;print(json.dumps(list(sys.version_info[:2])))'],text=True)
if json.loads(version)!=[3,11]:raise SystemExit('COOL requires Python 3.11')
uv_version=subprocess.check_output([str(uv),'--version'],text=True).split()
if len(uv_version)<2 or uv_version[1]!=sys.argv[4]:
 raise SystemExit('uv version must match pinned tool version')
env={**os.environ,'UV_PROJECT_ENVIRONMENT':str(base/'.venv'),
 'PYTHONNOUSERSITE':'1','UV_PYTHON_DOWNLOADS':'never'}
env.pop('PYTHONPATH',None)
env.pop('LD_LIBRARY_PATH',None)
subprocess.run([str(uv),'sync','--locked','--no-default-groups',
 '--python',str(python)],cwd=base,env=env,check=True,stdout=sys.stderr)
probe_env={**env,'PYTHONPATH':'/opt/cool/python_3.11/site-packages',
 'LD_LIBRARY_PATH':'/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib'}
probe='''import json,platform,sys,cv2
from care_vision.common.cool import verify_runtime
from care_vision.cloud.yolox import validate_dnn_engine
validate_dnn_engine('classic')
print(json.dumps({'python':sys.version,'machine':platform.machine(),
 'opencv':cv2.__version__,'cool':verify_runtime(),'classic_available':True}))'''
evidence=json.loads(subprocess.check_output([str(base/'.venv/bin/python'),'-c',probe],
 cwd=base,env=probe_env,text=True))
evidence.update(uv_version=uv_version[1],python_requested=str(python),
 python_environment=str(base/'.venv/bin/python'),sdk_installed_by_tool=False)
print(json.dumps(evidence))
"""

START = r"""
import json,os,pathlib,subprocess,sys
base=pathlib.Path(sys.argv[1]);run=base/sys.argv[2];run.mkdir(exist_ok=False)
python=base/'.venv/bin/python'
if not python.is_file():raise SystemExit('prepared workspace Python required')
env={**os.environ,'PYTHONNOUSERSITE':'1',
 'PYTHONPATH':str(base/'src')+':/opt/cool/python_3.11/site-packages',
 'LD_LIBRARY_PATH':'/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib'}
if 'OPENCV_FORCE_DNN_ENGINE' in env:raise SystemExit('DNN override rejected')
cmd=[str(python),'-u','-m','care_vision.cloud','--phase','3','--port','0',
 '--log',str(run/'cloud.jsonl'),'--run-seconds','100',
 '--person-model','models/person.onnx','--person-detector','yolox','--person-input-size','640',
 '--experimental-s-only-5fps','--experimental-weak-association',
 '--pose-shadow-config',str(base/sys.argv[3]),'--experimental-pose-observation',
 '--experimental-episode-continuity','--episode-state-file',str(run/'episode.json'),
 '--experimental-yolox-classic-engine']
if sys.argv[4]=='test20':cmd+=['--experimental-alert-after-seconds','20']
elif sys.argv[4] not in {'baseline','pose'}:raise SystemExit('unknown replay mode')
with (run/'cloud.stderr').open('w') as log:
 p=subprocess.Popen(cmd,cwd=base,env=env,stdin=subprocess.DEVNULL,
 stdout=log,stderr=log,start_new_session=True)
(run/'pid').write_text(str(p.pid));print(json.dumps({'pid':p.pid,'argv':cmd}))
"""

STATE_HASHES = r"""
import hashlib,json,pathlib,sys
base=pathlib.Path(sys.argv[1])
print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
 for p in sorted(base.glob('*/episode.json'))}))
"""

CLEANUP = r"""
import json,pathlib,shutil,sys
p=pathlib.Path(sys.argv[1]);assert p.parent==pathlib.Path(sys.argv[2])
assert p.name.startswith('care-vision-portable-classic-')
shutil.rmtree(p);print(json.dumps({'workspace_removed':not p.exists()}))
"""
