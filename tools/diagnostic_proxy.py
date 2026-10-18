import asyncio,json,signal,sys,time
from pathlib import Path
async def main():
 target=int(sys.argv[1]);path=Path(sys.argv[2]);tasks=set();writers=set();stop=asyncio.Event()
 rate=131072;stats={'upstream_bytes':0,'downstream_bytes':0,'upstream_bytes_per_second':rate,'scope':'Local TCP leg before owned SSH tunnel; not WAN bandwidth shaping'}
 loop=asyncio.get_running_loop()
 for sig in [signal.SIGTERM,signal.SIGINT]:loop.add_signal_handler(sig,stop.set)
 async def pipe(reader,writer,limited):
  try:
   while data:=await reader.read(4096):
    if limited:await asyncio.sleep(len(data)/rate)
    writer.write(data);await writer.drain()
    stats['upstream_bytes' if limited else 'downstream_bytes']+=len(data)
  finally:writer.close()
 async def relay(reader,writer):
  task=asyncio.current_task();tasks.add(task);writers.add(writer)
  try:
   other_reader,other_writer=await asyncio.open_connection('127.0.0.1',target);writers.add(other_writer)
   await asyncio.gather(pipe(reader,other_writer,True),pipe(other_reader,writer,False))
  except (OSError,asyncio.CancelledError):pass
  finally:tasks.discard(task)
 server=await asyncio.start_server(relay,'127.0.0.1',0)
 stats['started_mono_ns']=time.monotonic_ns()
 print(json.dumps({'port':server.sockets[0].getsockname()[1]}),flush=True)
 try:await stop.wait()
 finally:
  server.close();await server.wait_closed()
  for writer in writers:writer.close()
  remaining=list(tasks)
  for task in remaining:task.cancel()
  await asyncio.gather(*remaining,return_exceptions=True)
  stats.update(stopped_mono_ns=time.monotonic_ns(),owned_relays_stopped=True)
  path.write_text(json.dumps(stats,indent=2)+'\n')
asyncio.run(main())
