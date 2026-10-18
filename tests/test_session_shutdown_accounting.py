import asyncio
import threading

import pytest
from conftest import frame, status
from test_incident_episode import opened
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from care_vision.check_loop import read_log
from care_vision.cloud.server import Session
from care_vision.cloud.vision import MotionVision
from care_vision.common.contracts import dumps, loads, pack_frame
from care_vision.common.metrics import EventLog


async def eventually(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


def assert_accounted(records, count):
    received = [r['frame_id'] for r in records if r['event'] == 'frame_received']
    observed = [r['frame_id'] for r in records if r['event'] == 'observation']
    dropped = [r['frame_id'] for r in records if r['event'] == 'frame_dropped']
    assert len(received) == count
    assert len(observed + dropped) == len(set(observed + dropped))
    assert set(received) == set(observed + dropped), 'received frames lack a terminal outcome'
    final = next(r for r in records if r['event'] == 'session_finished')
    assert final['frames_received'] == final['frames_processed'] + final['frames_dropped']


@pytest.mark.parametrize('close_code', [1000, 1012])
@pytest.mark.parametrize('count', [3, 10])
def test_socket_close_accounts_queue_and_inference_without_late_decisions(
    tmp_path, identity, monkeypatch, close_code, count,
):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    original = MotionVision.observe

    def blocked(self, *args):
        entered.set()
        assert release.wait(5), 'test did not release inference'
        result = original(self, *args)
        returned.set()
        return result

    monkeypatch.setattr(MotionVision, 'observe', blocked)

    async def scenario():
        path = tmp_path / 'cloud.jsonl'
        log = EventLog('cloud', path)
        finished = asyncio.Event()
        sessions = []

        async def handler(ws):
            assert loads(await ws.recv())['type'] == 'EDGE_STATUS'
            session = Session(ws, identity, log, experimental_s_only_5fps=True)
            session.controller = opened(identity)
            sessions.append(session)
            try:
                await session.run()
            except ConnectionClosed:
                pass
            finally:
                finished.set()

        try:
            async with serve(handler, '127.0.0.1', 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with connect(f'ws://127.0.0.1:{port}', proxy=None) as ws:
                    await ws.send(dumps(status(identity)))
                    for index in range(1, count + 1):
                        header, jpeg = frame(identity, index, 1_000_000_000 + index * 50_000_000)
                        await ws.send(pack_frame(header, jpeg))
                        if index == 1:
                            await eventually(entered.is_set)
                    await eventually(lambda: sessions[0].received == count)
                    await ws.close(code=close_code)
                    await asyncio.wait_for(finished.wait(), 2)
                before = read_log(path)
                release.set()
                await eventually(returned.is_set)
                await asyncio.sleep(0.025)
                assert read_log(path) == before, 'late thread result wrote after session end'
                assert_accounted(before, count)
                shutdown = [r for r in before if r['event'] == 'frame_dropped'
                            and r['reason'] == 'session_ended_before_observation']
                assert len(shutdown) == min(count, 5)
                assert {r['processing_stage'] for r in shutdown} == {'inference', 'queued'}
                assert all(not r['intentional_backpressure'] for r in shutdown)
                assert sessions[0].controller.episode is not None
                assert sessions[0].controller.state == 'UNRESOLVED'
                assert not any(r['event'] in {'observation', 'agent_action', 'test_alert'}
                               for r in before)
        finally:
            release.set()
            log.close()

    asyncio.run(scenario())


def test_frame_arriving_during_diagnostic_close_is_explicitly_dropped(tmp_path, identity):
    async def scenario():
        path = tmp_path / 'diagnostic.jsonl'
        log = EventLog('cloud', path)

        class Socket:
            def __init__(self):
                self.queue = asyncio.Queue()

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await self.queue.get()

            async def close(self, **kwargs):
                header, jpeg = frame(identity, 2, 1_100_000_000)
                self.queue.put_nowait(pack_frame(header, jpeg))
                await eventually(lambda: session.received == 2)

        socket = Socket()
        header, jpeg = frame(identity)
        socket.queue.put_nowait(pack_frame(header, jpeg))
        session = Session(socket, identity, log, experimental_s_only_5fps=True,
                          diagnostic_disconnect_check=lambda *_: True)
        session.controller = opened(identity)
        try:
            await asyncio.wait_for(session.run(), 3)
            records = read_log(path)
            assert_accounted(records, 2)
            assert len([r for r in records if r['event'] == 'observation']) == 1
            drop = next(r for r in records if r['event'] == 'frame_dropped')
            assert drop['capture_seq'] == 2 and drop['processing_stage'] == 'queued'
            assert session.controller.episode is not None
        finally:
            log.close()

    asyncio.run(scenario())


def test_inference_error_keeps_error_and_accounts_unobserved_frame(tmp_path, identity):
    async def scenario():
        path = tmp_path / 'inference-error.jsonl'
        log = EventLog('cloud', path)

        class Socket:
            async def __aiter__(self):
                header, jpeg = frame(identity)
                yield pack_frame(header, jpeg)
                await asyncio.Future()

        def broken(*_):
            raise OSError('controlled inference failure')

        session = Session(Socket(), identity, log)
        session.vision.observe = broken
        try:
            with pytest.raises(OSError, match='controlled inference failure') as error:
                await session.run()
            assert error.value.failed_worker == 'Session.perceive'
            records = read_log(path)
            assert_accounted(records, 1)
            assert not any(r['event'] == 'observation' for r in records)
            final = next(r for r in records if r['event'] == 'session_finished')
            assert final['frames_processed'] == 0 and final['shutdown_frames_dropped'] == 1
        finally:
            log.close()

    asyncio.run(scenario())
