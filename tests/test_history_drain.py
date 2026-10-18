import asyncio
import time
from types import SimpleNamespace

import pytest
from conftest import command
from websockets.asyncio.server import serve

from care_vision.check_history_delivery import check_history_delivery
from care_vision.check_loop import read_log
from care_vision.cloud.controller import Controller
from care_vision.cloud.server import CloudServer, Session
from care_vision.common import transport
from care_vision.common.contracts import ProtocolError, dumps, envelope, loads
from care_vision.common.metrics import EventLog
from care_vision.common.transport import Outbox
from care_vision.edge import runtime as edge_runtime
from care_vision.edge.runtime import Edge
from care_vision.edge.source import VideoSource
from care_vision.fixture import make_video


def test_history_wire_budget_and_live_priority_with_dynamic_eof_limit(monkeypatch):
    async def scenario():
        clock = [100.0]
        limits = [5, 1000]
        monkeypatch.setattr(transport, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        box = Outbox(history_limits=lambda: limits)
        box.put_history([(b"x" * 300, {}), (b"y" * 300, {}), ("done", None)], 20)
        assert (await box.get())[0] == b"x" * 300
        clock[0] += 0.21  # FPS alone allows it; byte budget still blocks history.
        pending = asyncio.create_task(box.get())
        await asyncio.sleep(0)
        assert not pending.done()
        box.put_control("ack")
        box.put_frame("live")
        assert await pending == "ack"
        assert await box.get() == "live"
        clock[0] = 100.31
        assert (await box.get())[0] == b"y" * 300
        # At EOF raise the budget without rewriting history sample metadata.
        limits[:] = [20, 10000]
        clock[0] = 100.37
        assert await box.get() == ("done", None)
        assert box.history_period == 0.05

    asyncio.run(scenario())


def test_cloud_drain_waits_for_inflight_work_and_history(tmp_path, identity):
    async def scenario():
        messages = []

        async def send(data):
            messages.append(loads(data))

        log = EventLog("cloud", tmp_path / "cloud.jsonl")
        session = Session(SimpleNamespace(send=send), identity, log)
        session.drain_request = envelope(
            "DRAIN_REQUEST", identity, drain_id="D", live_frames_sent=1
        )
        session.received = 1
        session.history.finished = False
        session.history.total_received = 2
        task = asyncio.create_task(session.confirm_drain())
        try:
            await asyncio.sleep(0.025)
            assert not messages
            session.processed = 1
            await asyncio.sleep(0.025)
            assert not messages
            session.history.total_processed = 2
            session.history.finished = True
            session.actions_inflight = True
            await asyncio.sleep(0.025)
            assert not messages
            session.actions_inflight = False
            async with asyncio.timeout(1):
                while not messages:
                    await asyncio.sleep(0.01)
            assert messages[0]["historical_processed"] == 2
            assert messages[0]["live_processed"] == 1
            await asyncio.sleep(0.025)
            assert len(messages) == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            log.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("damage", ["negative", "bool", "incomplete_live", "incomplete_history"])
def test_drain_contract_rejects_incomplete_or_invalid_counts(identity, damage):
    ack = envelope(
        "DRAINED",
        identity,
        drain_id="D",
        live_received=2,
        live_processed=2,
        live_dropped=0,
        historical_received=3,
        historical_processed=3,
    )
    if damage == "negative":
        ack["live_received"] = -1
    elif damage == "bool":
        ack["historical_processed"] = True
    elif damage == "incomplete_live":
        ack["live_processed"] = 1
    else:
        ack["historical_processed"] = 2
    with pytest.raises(ProtocolError):
        dumps(ack)


def test_edge_rejects_wrong_session_count_or_pending_history(tmp_path):
    log = EventLog("edge", tmp_path / "edge.jsonl")
    edge = Edge(VideoSource("unused"), "ws://unused", log)
    edge.outbox = Outbox()
    edge.session_sent = 2
    edge.session_historical_sent = 3
    edge.drain_request = {"drain_id": "D"}
    ack = envelope(
        "DRAINED",
        edge.identity,
        drain_id="D",
        live_received=2,
        live_processed=2,
        live_dropped=0,
        historical_received=3,
        historical_processed=3,
    )
    try:
        for change in ({"session_id": "other"}, {"drain_id": "other"}, {"live_received": 1}):
            with pytest.raises(ProtocolError):
                edge.acknowledge_drain({**ack, **change})
        edge.outbox.put_history([("pending", None)], 20)
        with pytest.raises(ProtocolError):
            edge.acknowledge_drain(ack)
        edge.outbox.history.clear()
        edge.acknowledge_drain(ack)
        assert edge.drain_done.is_set()
    finally:
        log.close()


def test_eof_timeout_is_bounded_and_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(edge_runtime, "DRAIN_TIMEOUT_SECONDS", 0.02)

    async def scenario():
        log = EventLog("edge", tmp_path / "edge.jsonl")
        edge = Edge(VideoSource("unused"), "ws://unused", log)
        edge.outbox = Outbox()
        try:
            with pytest.raises(RuntimeError, match="not confirmed"):
                await edge.drain()
            assert not edge.drain_done.is_set()
        finally:
            log.close()

    asyncio.run(scenario())
    assert any(r["event"] == "drain_timeout" for r in read_log(tmp_path / "edge.jsonl"))


def test_escalation_does_not_replace_active_history(tmp_path, identity):
    async def scenario():
        messages = []

        async def send(data):
            messages.append(loads(data))

        log = EventLog("cloud", tmp_path / "cloud.jsonl")
        session = Session(SimpleNamespace(send=send), identity, log)
        session.history.finished = False
        session.history.request = {"command_id": "old"}
        try:
            await session.dispatch(command(identity, deadline=time.monotonic_ns() + 10**9), {})
            assert [m["type"] for m in messages] == ["SET_TRANSMIT_FPS"]
            assert session.history.request["command_id"] == "old"
        finally:
            log.close()

    asyncio.run(scenario())
    assert any(r["event"] == "buffer_request_skipped" for r in read_log(tmp_path / "cloud.jsonl"))


def test_real_eof_drains_history_while_agent_remains_verifying(tmp_path, monkeypatch):
    video = make_video(tmp_path / "static.avi", static=True, seconds=2)

    def escalate_once(self, observation):
        if self.command_seq == 0 and observation["capture_seq"] >= 30 and observation["valid"]:
            return self._command(observation, 20, "test_late_history")
        return None

    monkeypatch.setattr(Controller, "observe", escalate_once)

    async def scenario():
        edge_log = EventLog("edge", tmp_path / "edge.jsonl")
        cloud_log = EventLog("cloud", tmp_path / "cloud.jsonl")
        try:
            async with serve(CloudServer(cloud_log).handler, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                edge = Edge(VideoSource(str(video)), f"ws://127.0.0.1:{port}", edge_log)
                await asyncio.wait_for(edge.run(), 6)
        finally:
            edge_log.close()
            cloud_log.close()

    asyncio.run(scenario())
    edge, cloud = read_log(tmp_path / "edge.jsonl"), read_log(tmp_path / "cloud.jsonl")
    assert [
        r["transmit_fps"]
        for r in cloud
        if r["event"] == "agent_action" and r["type"] == "SET_TRANSMIT_FPS"
    ] == [20]
    ack = next(
        r
        for r in edge
        if r["event"] == "command_ack" and r.get("command_type") == "REQUEST_RECENT_BUFFER"
    )
    done = next(r for r in edge if r["event"] == "drain_complete")
    start = next(r for r in edge if r["event"] == "drain_started")
    historical = [
        r for r in edge if r["event"] == "frame_sent" and r["evidence_kind"] == "historical"
    ]
    assert len(historical) == ack["selected_frames"] == done["historical_processed"] > 0
    assert any(r["local_mono_ns"] > start["local_mono_ns"] for r in historical)
    assert any(r["local_mono_ns"] < start["local_mono_ns"] for r in historical)
    assert not any(r["event"] in {"buffer_transfer_aborted", "drain_timeout"} for r in edge)
    assert check_history_delivery(edge, cloud)["passed"]
    incomplete = [r for r in cloud if r["event"] != "buffer_processed"]
    with pytest.raises(ValueError, match="completion missing"):
        check_history_delivery(edge, incomplete)
