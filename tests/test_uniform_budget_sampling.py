from pathlib import Path
import importlib.util,json,os
from types import SimpleNamespace
from care_vision.edge.sampler import Sampler
B=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('uniform_adapter',B/'tools/diagnostic_edge.py')
from care_vision.edge.runtime import Edge
saved={n:getattr(Edge,n) for n in ['send','report','select_live_capture']}
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
for name,method in saved.items():setattr(Edge,name,method)

def edge():
 return SimpleNamespace(sampler=Sampler(2),network=SimpleNamespace(transmission_cap_fps=5),visual=SimpleNamespace(view_kind='full_frame',policy_seq=0),identity={'session_id':'one'},network_limited_frames=0)

def test_regular_grid_uses_actual_capture_timestamps_and_keeps_temporal_window():
 asset=os.environ.get('CARE_VISION_ASSET_ROOT')
 rows=([json.loads(s) for s in (Path(asset)/'artifacts/tapo-weak-live-20261005/recovery-2/demo.captures.jsonl').read_text().splitlines()] if asset else [{'capture_seq':i+1,'captured_mono_ns':i*40_000_000} for i in range(1500)])
 e=edge();selected=[]
 for r in rows:
  if adapter.uniform_select(e,r['capture_seq'],r['captured_mono_ns']):selected.append(r['captured_mono_ns'])
 gaps=[(b-a)/1e9 for a,b in zip(selected,selected[1:])]
 assert len(gaps)>50
 assert min(gaps)>=.65 and max(gaps)<=.8
 assert abs((len(selected)-1)/((selected[-1]-selected[0])/1e9)-1.34)<.03

def test_command_change_cannot_claim_achieved_five_fps_or_duplicate_capture():
 e=edge();chosen=[]
 for i in range(1000):
  ns=i*40_000_000
  if i==300:e.sampler.set_rate(5,ns)
  if adapter.uniform_select(e,i+1,ns):chosen.append(ns)
  assert not adapter.uniform_select(e,i+1,ns)
 assert all(.7<=(b-a)/1e9<=.8 for a,b in zip(chosen,chosen[1:]))
 assert e.sampler.fps==5 and e.diagnostic_grid.fps==1.34

def test_roi_request_changes_sampling_then_full_frame_returns_to_cap():
 e=edge();counts={}
 for mode in ['full_frame','roi','full_frame']:
  e.visual.view_kind=mode;e.visual.policy_seq+=1
  chosen=[]
  offset=e.visual.policy_seq*500
  for i in range(offset,offset+250):
   if adapter.uniform_select(e,i+1,i*40_000_000):chosen.append(i)
  counts[e.visual.policy_seq]=len(chosen)
 assert counts[2]>counts[1]*1.3
 assert abs(counts[3]-counts[1])<=1
