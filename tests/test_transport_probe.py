import asyncio
import json
import time

import pytest

from care_vision.common.metrics import EventLog
from care_vision.transport_probe import TransportProbe


@pytest.mark.parametrize("duration", [0, .12])
def test_probe_preserves_bytes_and_only_holds_upstream(tmp_path, duration):
    log = EventLog("probe", tmp_path / "probe.jsonl")
    payload = bytes(range(256)) * 1024

    async def echo(reader, writer):
        writer.write(b"READY")
        await writer.drain()
        try:
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def scenario():
        async with await asyncio.start_server(echo, "127.0.0.1", 0) as upstream:
            port = upstream.sockets[0].getsockname()[1]
            probe = TransportProbe(port, log, pause_at=0, pause_seconds=duration)
            try:
                async with await asyncio.start_server(probe.handle, "127.0.0.1", 0) as server:
                    reader, writer = await asyncio.open_connection(
                        "127.0.0.1", server.sockets[0].getsockname()[1]
                    )
                    assert await reader.readexactly(5) == b"READY"
                    writer.write(payload)
                    await writer.drain()
                    assert await asyncio.wait_for(reader.readexactly(len(payload)), 2) == payload
                    writer.close()
                    await writer.wait_closed()
            finally:
                await probe.close()
            assert not probe.handlers

    try:
        asyncio.run(scenario())
    finally:
        log.close()
    rows = [json.loads(s) for s in (tmp_path / "probe.jsonl").read_text().splitlines()]
    forwarded = [r for r in rows if r["event"] == "bytes_forwarded"]
    assert sum(r["byte_count"] for r in forwarded if r["direction"] == "upstream") == len(payload)
    assert sum(
        r["byte_count"] for r in forwarded if r["direction"] == "downstream"
    ) == len(payload) + 5
    starts = [r for r in rows if r["event"] == "hold_started"]
    ends = [r for r in rows if r["event"] == "hold_released"]
    if duration:
        assert len(starts) == len(ends) == 1
        assert (ends[0]["local_mono_ns"] - starts[0]["local_mono_ns"]) / 1e9 >= .08
        assert forwarded[0]["direction"] == "downstream"
    else:
        assert not starts and not ends


@pytest.mark.parametrize("at,duration", [(float('nan'), 0), (0, float('inf')), (-1, 0), (0, 3)])
def test_probe_rejects_unbounded_or_nonfinite_pauses(at, duration):
    with pytest.raises(ValueError):
        TransportProbe(1, None, at, duration)


def test_probe_schedule_is_not_restarted_after_reconnection(tmp_path):
    log = EventLog("probe", tmp_path / "reconnect.jsonl")
    probe = TransportProbe(1, log, 0, .1)
    probe.started = time.monotonic() - 5
    probe.injected = True

    class Reader:
        def __init__(self):
            self.reads = 0

        async def read(self, size):
            self.reads += 1
            return b"context" if self.reads == 1 else b""

    class Writer:
        def write(self, data):
            assert data == b"context"

        async def drain(self):
            pass

    try:
        asyncio.run(probe.pipe(Reader(), Writer(), "upstream", 2))
    finally:
        log.close()
    assert 'hold_started' not in (tmp_path / "reconnect.jsonl").read_text()
