import asyncio,hashlib,os,runpy,time
import cv2,numpy as np
from care_vision.common.contracts import pack_frame,unpack_frame
from care_vision.edge.runtime import Edge
from care_vision.edge.sampler import Sampler
FULL_FRAME_CAP=1.34

def uniform_select(self,seq,captured_ns):
 # The byte-constrained time grid is selected at acquisition, not after dropping
 # already-sampled half-second frames while the sender waits.
 requested_selected=self.sampler.select(seq,captured_ns)
 cap=min(self.sampler.fps,self.network.transmission_cap_fps or 30,
         FULL_FRAME_CAP if self.visual.view_kind=='full_frame' else 5)
 key=(self.identity['session_id'],cap,self.visual.view_kind,self.visual.policy_seq)
 if getattr(self,'diagnostic_grid_key',None)!=key:
  self.diagnostic_grid_key=key
  self.diagnostic_grid=Sampler(cap)
  self.diagnostic_grid.set_rate(cap,captured_ns)
 chosen=self.diagnostic_grid.select(seq,captured_ns)
 if requested_selected and not chosen:self.network_limited_frames+=1
 return chosen
Edge.select_live_capture=uniform_select
original_report=Edge.report
original_send=Edge.send
async def paced_send(self,ws):
 budget=int(os.environ.get('CARE_DIAGNOSTIC_BYTE_BUDGET','0'))
 if budget<=0:return await original_send(self,ws)
 send_original=ws.send
 due=0.0
 async def send(data,*a,**kw):
  nonlocal due
  if isinstance(data,bytes):
   original_bytes=len(data)
   header,jpeg=unpack_frame(data)
   image=cv2.imdecode(np.frombuffer(jpeg,dtype=np.uint8),cv2.IMREAD_COLOR)
   if image is None:raise ValueError('cannot decode diagnostic source JPEG')
   ok,encoded=cv2.imencode('.webp',image,[cv2.IMWRITE_WEBP_QUALITY,80])
   if not ok:raise RuntimeError('diagnostic JPEG encode failed')
   jpeg=encoded.tobytes()
   if header['evidence_kind']=='live':
    cap=min(header.get('effective_target_transmit_fps',header['target_transmit_fps']),FULL_FRAME_CAP if header.get('view_kind','full_frame')=='full_frame' else 5)
    header.update(effective_target_transmit_fps=cap,network_limited=cap<header['target_transmit_fps'],diagnostic_uniform_sampling_cap_fps=cap)
   header.update(encoding="webp",payload_bytes=len(jpeg),diagnostic_webp_quality=80,diagnostic_webp_sha256=hashlib.sha256(jpeg).hexdigest())
   data=pack_frame(header,jpeg)
   self.log.emit('diagnostic_webp_reencoded',**self.identity,frame_id=header['frame_id'],quality=80,original_wire_bytes=original_bytes,new_wire_bytes=len(data),jpeg_sha256=header['diagnostic_webp_sha256'])
   if self.sending_header is not None:self.sending_header.update(header)
   wait=max(0,due-time.monotonic())
   self.log.emit('diagnostic_wire_budget',**self.identity,bytes_per_second=budget,wait_ms=wait*1000,wire_bytes=len(data),requested_fps=self.sampler.fps)
   await asyncio.sleep(wait)
   await send_original(data,*a,**kw)
   # Original Edge.send adds original buffer length synchronously on return.
   # Correct the delta so its cumulative byte counter remains actual wire bytes.
   self.bytes_sent+=len(data)-original_bytes
   due=time.monotonic()+len(data)/budget
  else:await send_original(data,*a,**kw)
 ws.send=send
 try:await original_send(self,ws)
 finally:ws.send=send_original
Edge.send=paced_send
async def report(self,ws):
 original_ping=ws.ping
 async def ping(data=None):
  start=time.monotonic_ns()
  self.log.emit('transport_ping_started',**self.identity,write_buffer_bytes=ws.transport.get_write_buffer_size(),local_socket_only=True)
  pong=await original_ping(data)
  self.log.emit('transport_ping_enqueued',**self.identity,enqueue_ms=(time.monotonic_ns()-start)/1e6,write_buffer_bytes=ws.transport.get_write_buffer_size())
  def done(f):
   self.log.emit('transport_ping_finished',**self.identity,cancelled=f.cancelled(),rtt_ms=None if f.cancelled() or f.exception() else f.result()*1000)
  pong.add_done_callback(done)
  return pong
 ws.ping=ping
 try:await original_report(self,ws)
 finally:ws.ping=original_ping
Edge.report=report
if __name__=='__main__':runpy.run_module('care_vision.edge',run_name='__main__')
