"""Loopback-only TCP relay for bounded, explicitly logged transport experiments.

Records byte counts and monotonic times, never payloads. A one-shot upstream
hold models delivery delay/bunching, not packet loss or absolute AWS latency.
"""

import argparse
import asyncio
import json
import math
import select
import subprocess
import sys
import time
from contextlib import contextmanager

from care_vision.common.metrics import EventLog


class TransportProbe:
    def __init__(self, upstream_port, log, pause_at=12.0, pause_seconds=0.0):
        if not 1 <= upstream_port <= 65535:
            raise ValueError("invalid upstream port")
        if not math.isfinite(pause_at) or not 0 <= pause_at <= 180:
            raise ValueError("pause_at must be finite and within [0, 180]")
        if not math.isfinite(pause_seconds) or not 0 <= pause_seconds <= 2:
            raise ValueError("pause_seconds must be finite and within [0, 2]")
        self.upstream_port, self.log = upstream_port, log
        self.pause_at, self.pause_seconds = pause_at, pause_seconds
        self.started = None
        self.injected = False
        self.connections = 0
        self.handlers = set()

    async def pipe(self, reader, writer, direction, connection):
        while data := await reader.read(65536):
            read_ns = time.monotonic_ns()
            if (
                direction == "upstream" and self.pause_seconds and not self.injected
                and time.monotonic() >= self.started + self.pause_at
            ):
                self.injected = True
                self.log.emit("hold_started", connection=connection, bytes_held=len(data),
                              requested_seconds=self.pause_seconds)
                await asyncio.sleep(self.pause_seconds)
                self.log.emit("hold_released", connection=connection)
            writer.write(data)
            await writer.drain()
            self.log.emit(
                "bytes_forwarded", connection=connection, direction=direction,
                byte_count=len(data), read_mono_ns=read_ns,
                relay_elapsed_ms=(time.monotonic_ns() - read_ns) / 1e6,
            )

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.handlers.add(task)
        self.connections += 1
        connection = self.connections
        if self.started is None:
            self.started = time.monotonic()
            self.log.emit("hold_schedule", pause_at_seconds=self.pause_at,
                          pause_seconds=self.pause_seconds, anchor="first_connection")
        self.log.emit("connection_started", connection=connection)
        upstream = None
        workers = []
        try:
            other, upstream = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", self.upstream_port), timeout=10
            )
            workers = [
                asyncio.create_task(self.pipe(reader, upstream, "upstream", connection)),
                asyncio.create_task(self.pipe(other, writer, "downstream", connection)),
            ]
            done, _ = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED)
            for worker in done:
                worker.result()
        except (OSError, TimeoutError) as exc:
            self.log.emit("relay_error", connection=connection, error_type=type(exc).__name__,
                          reason=str(exc))
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            for stream in (writer, upstream):
                if stream is not None:
                    stream.close()
                    try:
                        await stream.wait_closed()
                    except OSError:
                        pass
            self.log.emit("connection_finished", connection=connection)
            self.handlers.discard(task)

    async def close(self):
        tasks = list(self.handlers)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@contextmanager
def probe_process(upstream_port, folder, pause_at=12.0, pause_seconds=0.0):
    """Start a separate relay process; the normal AWS runner does not use it."""
    with (folder / "transport-probe.stderr").open("w") as error:
        process = subprocess.Popen([
            sys.executable, "-m", "care_vision.transport_probe",
            "--upstream-port", str(upstream_port), "--log", str(folder / "transport.jsonl"),
            "--pause-at", str(pause_at), "--pause-seconds", str(pause_seconds),
        ], stdout=subprocess.PIPE, stderr=error)
        try:
            ready, _, _ = select.select([process.stdout], [], [], 10)
            if not ready:
                raise RuntimeError("transport probe readiness timeout")
            status = json.loads(process.stdout.readline())
            yield status["port"]
        finally:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()


async def run(args):
    log = EventLog("transport_probe", args.log)
    probe = TransportProbe(args.upstream_port, log, args.pause_at, args.pause_seconds)
    try:
        async with await asyncio.start_server(probe.handle, "127.0.0.1", 0) as server:
            print(json.dumps({"port": server.sockets[0].getsockname()[1]}), flush=True)
            await asyncio.sleep(300)  # Bound orphan lifetime if the runner disappears.
    finally:
        await probe.close()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-port", type=int, required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--pause-at", type=float, default=12)
    parser.add_argument("--pause-seconds", type=float, default=0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
