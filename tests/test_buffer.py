import asyncio

import cv2
import numpy as np
import pytest
from conftest import command, frame

from care_vision.cloud.controller import Controller
from care_vision.cloud.history import HistoryTransfer
from care_vision.common.contracts import ProtocolError, envelope, unpack_frame, utc, validate
from care_vision.common.metrics import EventLog
from care_vision.common.transport import Outbox
from care_vision.edge.buffer import Capture, RingBuffer, encode
from care_vision.edge.commands import CommandHandler
from care_vision.edge.runtime import Edge
from care_vision.edge.sampler import Sampler
from care_vision.edge.source import VideoSource


def request(identity, **changes):
    base = command(identity, seq=2, command_id="B1", deadline=20_000_000_000)
    base.pop("transmit_fps")
    return {
        **base,
        "type": "REQUEST_RECENT_BUFFER",
        "from_edge_mono_ns": 5_000_000_000,
        "to_edge_mono_ns": 10_000_000_000,
        "history_fps": 20,
        **changes,
    }


def capture(seq, size=10):
    return Capture(seq, 1_000_000_000 + round((seq - 1) * 1e9 / 30), utc(), b"x" * size)


def test_ten_seconds_all_captures_and_twenty_fps_selection():
    ring = RingBuffer()
    for seq in range(1, 601):
        ring.append(capture(seq))
    assert len(ring.frames) == 300
    assert ring.frames[0].seq == 301
    end = ring.frames[-1].mono_ns
    selected = ring.select(end - 5_000_000_000, end, 20)
    assert len(selected) == 100
    assert all(b.seq > a.seq and b.mono_ns > a.mono_ns for a, b in zip(selected, selected[1:]))
    assert selected[0].mono_ns == end - 5_000_000_000
    assert selected[-1].mono_ns < end
    assert len(ring.frames) == 300  # Reading doesn't consume the buffer.


def test_capture_encoding_supports_high_resolution_without_aspect_distortion():
    portrait = np.full((800, 400, 3), 255, dtype=np.uint8)
    jpeg = encode(portrait, 1280, 720)
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (720, 1280, 3)
    assert image[:, :400].mean() < 2
    assert image[:, 470:810].mean() > 250


def test_byte_count_frame_count_and_time_limits_are_independent():
    ring = RingBuffer(max_bytes=25)
    for seq in range(1, 5):
        ring.append(capture(seq))
    assert ring.bytes == 20 and len(ring.frames) == 2
    ring.append(capture(5, 100))
    assert ring.bytes == 0 and not ring.frames
    ring.append(Capture(6, 30_000_000_000, utc(), b"ok"))
    assert ring.bytes == 2
    with pytest.raises(ValueError, match="increase"):
        ring.append(capture(6))
    ring = RingBuffer(max_frames=1)
    ring.append(capture(1))
    ring.append(capture(2))
    assert [f.seq for f in ring.frames] == [2]
    ring = RingBuffer(seconds=1)
    ring.append(capture(1))
    ring.append(capture(61))
    assert [f.seq for f in ring.frames] == [61]


@pytest.mark.parametrize(
    "changes",
    [
        {"from_edge_mono_ns": -1},
        {"to_edge_mono_ns": 5_000_000_000},
        {"to_edge_mono_ns": 15_000_000_001},
        {"history_fps": True},
        {"history_fps": float("nan")},
        {"history_fps": 31},
        {"command_seq": 0},
    ],
)
def test_invalid_buffer_request(identity, changes):
    with pytest.raises(ProtocolError):
        validate(request(identity, **changes))


def test_edge_snapshot_dedup_idempotency_busy_partial_empty_and_live_policy(tmp_path):
    source = VideoSource("unused")
    for seq in range(1, 301):
        source.buffer.append(capture(seq))
    log = EventLog("edge", tmp_path / "edge.jsonl")
    try:
        edge = Edge(source, "unused", log)
        edge.outbox = Outbox()
        identity = edge.identity
        edge.commands.apply(command(identity), 100, 1)
        edge.sent_seqs.extend(range(1, 301, 15))
        cmd = request(identity)
        ack = edge.commands.apply(cmd, 11_000_000_000, 301)
        assert ack["result"] == "applied" and ack["coverage"] == "full"
        assert ack["selected_frames"] + ack["deduplicated_frames"] == 100
        assert ack["deduplicated_frames"] == 10
        size = len(edge.outbox.history)
        assert edge.commands.apply(cmd, 12_000_000_000, 330)["result"] == "duplicate"
        assert len(edge.outbox.history) == size
        assert edge.commands.policy_seq == 1 and edge.sampler.fps == 20
        headers = [unpack_frame(data)[0] for data, header in edge.outbox.history if header]
        assert all(h["capture_seq"] not in edge.sent_seqs for h in headers)
        assert [h["history_index"] for h in headers] == list(range(90))
        assert all(h["evidence_kind"] == "historical" and h["policy_seq"] == 0 for h in headers)
        busy = request(identity, command_id="B2", command_seq=3)
        assert edge.commands.apply(busy, 12_000_000_000, 330)["reason"] == "buffer_busy"
        edge.outbox.history.clear()
        partial = request(identity, command_id="B3", command_seq=4, from_edge_mono_ns=0)
        assert edge.commands.apply(partial, 12_000_000_000, 330)["coverage"] == "partial"
        edge.outbox.history.clear()
        empty = request(
            identity, command_id="B4", command_seq=5, from_edge_mono_ns=0, to_edge_mono_ns=1
        )
        ack = edge.commands.apply(empty, 12_000_000_000, 330)
        assert ack["coverage"] == "empty" and ack["selected_frames"] == 0
    finally:
        log.close()


def test_buffer_command_fencing_expiry_reuse_and_future(identity):
    calls = []

    def enqueue(cmd):
        calls.append(cmd)
        return {
            "selected_frames": 0,
            "selected_bytes": 0,
            "deduplicated_frames": 0,
            "coverage": "empty",
        }

    handler = CommandHandler(identity, Sampler(), 30, enqueue)
    cmd = request(identity)
    assert handler.apply(cmd, 11_000_000_000, 301)["result"] == "applied"
    assert handler.apply({**cmd, "history_fps": 10}, 12_000_000_000, 330)["result"] == "rejected"
    assert (
        handler.apply(request(identity, command_id="old", command_seq=1), 12_000_000_000, 330)[
            "result"
        ]
        == "stale"
    )
    assert (
        handler.apply(
            request({**identity, "session_id": "old"}, command_id="other"), 12_000_000_000, 330
        )["result"]
        == "rejected"
    )
    assert (
        handler.apply(
            request(identity, command_id="expired", command_seq=3, valid_until_edge_mono_ns=1),
            12_000_000_000,
            330,
        )["result"]
        == "expired"
    )
    assert (
        handler.apply(request(identity, command_id="future", command_seq=4), 9_000_000_000, 270)[
            "reason"
        ]
        == "future_buffer_window"
    )
    assert len(calls) == 1 and handler.sampler.fps == 2


def test_control_and_live_frames_have_priority_over_paced_history():
    async def scenario():
        outbox = Outbox()
        outbox.put_history([(b"old1", {}), (b"old2", {}), ("complete", None)], 20)
        outbox.put_frame("live1")
        outbox.put_control("ack")
        assert await outbox.get() == "ack"
        assert await outbox.get() == "live1"
        assert await outbox.get() == (b"old1", {})
        waiting = asyncio.create_task(outbox.get())
        await asyncio.sleep(0)
        outbox.put_frame("live2")
        assert await waiting == "live2"
        assert await outbox.get() == (b"old2", {})
        assert await outbox.get() == ("complete", None)

    asyncio.run(scenario())


def test_history_can_be_deferred_until_live_returns_to_baseline():
    async def scenario():
        verifying = True
        outbox = Outbox(defer_history=lambda: verifying)
        outbox.put_history([(b"old", {})], 20)
        waiting = asyncio.create_task(outbox.get())
        await asyncio.sleep(0.01)
        assert not waiting.done()
        outbox.put_frame("live")
        assert await waiting == "live"
        waiting = asyncio.create_task(outbox.get())
        await asyncio.sleep(0.01)
        assert not waiting.done()
        verifying = False
        outbox.put_control("ack")
        assert await waiting == "ack"
        assert await outbox.get() == (b"old", {})

    asyncio.run(scenario())


def historical_frame(identity, index=0, at_ns=5_000_000_000):
    header, jpeg = frame(identity, seq=index + 1, at_ns=at_ns)
    header.update(
        evidence_kind="historical", buffer_request_id="B1", history_index=index, history_fps=20
    )
    return header, jpeg


def accepted_history(identity, log, count=2):
    history = HistoryTransfer(log)
    history.start(request(identity))
    _, jpeg = historical_frame(identity)
    history.acknowledge(
        {
            "command_type": "REQUEST_RECENT_BUFFER",
            "command_id": "B1",
            "command_seq": 2,
            "result": "applied",
            "selected_frames": count,
            "selected_bytes": len(jpeg) * count,
            "coverage": "full",
            "deduplicated_frames": 0,
        }
    )
    return history


def test_historical_bounds_order_budget_and_completion(identity, tmp_path):
    log = EventLog("cloud", tmp_path / "cloud.jsonl")
    try:
        history = accepted_history(identity, log)
        header, jpeg = historical_frame(identity)
        for changes, match in [
            ({"buffer_request_id": "other"}, "unsolicited"),
            ({"captured_mono_ns": 10_000_000_000}, "outside"),
            ({"history_index": 1}, "order"),
        ]:
            with pytest.raises(ProtocolError, match=match):
                history.receive({**header, **changes}, jpeg)
        history.receive(header, jpeg)
        with pytest.raises(ProtocolError, match="order"):
            history.receive(header, jpeg)
        complete = envelope("BUFFER_COMPLETE", identity, buffer_request_id="B1", frames_sent=2)
        with pytest.raises(ProtocolError, match="incomplete"):
            history.finish_receiving(complete)
        header2, jpeg2 = historical_frame(identity, 1, 5_500_000_000)
        history.receive(header2, jpeg2)
        with pytest.raises(ProtocolError, match="budget"):
            history.receive(*historical_frame(identity, 2, 6_000_000_000))
        history.finish_receiving(complete)
        with pytest.raises(ProtocolError, match="completed"):
            history.finish_receiving(complete)
        assert history.queue.qsize() == 3 and history.queue.maxsize == 301
    finally:
        log.close()


def test_historical_perception_does_not_age_out_or_drive_live_state(identity, tmp_path):
    from care_vision.check_loop import read_log

    async def scenario():
        log = EventLog("cloud", tmp_path / "cloud.jsonl")
        history = accepted_history(identity, log)
        history.receive(*historical_frame(identity))
        history.receive(*historical_frame(identity, 1, 5_500_000_000))
        history.finish_receiving(
            envelope("BUFFER_COMPLETE", identity, buffer_request_id="B1", frames_sent=2)
        )
        task = asyncio.create_task(history.perceive())
        try:
            async with asyncio.timeout(2):
                while not history.finished:
                    await asyncio.sleep(0.01)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            log.close()
        observations = [
            r for r in read_log(tmp_path / "cloud.jsonl") if r["event"] == "observation"
        ]
        assert len(observations) == 2 and observations[1]["valid"]
        assert observations[1]["reference_frame_id"] == observations[0]["frame_id"]
        controller = Controller(identity)
        before = dict(controller.__dict__)
        for obs in observations:
            assert controller.observe(obs) is None
        assert controller.__dict__ == before

    asyncio.run(scenario())


def test_historical_inference_waits_while_live_verification_has_priority(identity, tmp_path):
    from care_vision.check_loop import read_log

    async def scenario():
        blocked = True
        log = EventLog("cloud", tmp_path / "priority.jsonl")
        history = accepted_history(identity, log, count=1)
        history.defer_processing = lambda: blocked
        history.receive(*historical_frame(identity))
        history.finish_receiving(
            envelope("BUFFER_COMPLETE", identity, buffer_request_id="B1", frames_sent=1)
        )
        task = asyncio.create_task(history.perceive())
        try:
            await asyncio.sleep(0.03)
            assert history.processed == 0
            blocked = False
            async with asyncio.timeout(1):
                while not history.finished:
                    await asyncio.sleep(0.01)
            assert history.processed == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            log.close()

        records = read_log(tmp_path / "priority.jsonl")
        assert any(row["event"] == "buffer_processing_deferred" for row in records)
        assert any(row["event"] == "buffer_processing_resumed" for row in records)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes",
    [
        {"buffer_request_id": ""},
        {"history_index": -1},
        {"history_index": 300},
        {"history_index": True},
        {"policy_seq": 1, "applied_command_id": "future-policy"},
        {"evidence_kind": "live"},
    ],
)
def test_historical_header_cannot_masquerade_as_live_or_escape_bounds(identity, changes):
    header, _ = historical_frame(identity)
    with pytest.raises(ProtocolError):
        validate({**header, **changes})


def test_empty_transfer_has_explicit_completion(identity, tmp_path):
    async def scenario():
        log = EventLog("cloud", tmp_path / "empty.jsonl")
        history = accepted_history(identity, log, count=0)
        history.finish_receiving(
            envelope("BUFFER_COMPLETE", identity, buffer_request_id="B1", frames_sent=0)
        )
        task = asyncio.create_task(history.perceive())
        try:
            async with asyncio.timeout(1):
                while not history.finished:
                    await asyncio.sleep(0.01)
            assert history.processed == 0
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            log.close()

    asyncio.run(scenario())


def test_missing_historical_completion_times_out_explicitly(identity, tmp_path):
    import time

    from care_vision.cloud.server import Session

    class Socket:
        async def send(self, data):
            pytest.fail("An accepted transfer should not retry the request")

    async def scenario():
        log = EventLog("cloud", tmp_path / "timeout.jsonl")
        session = Session(Socket(), identity, log)
        session.history = accepted_history(identity, log)
        session.history.started = time.monotonic() - 31
        try:
            with pytest.raises(
                ProtocolError, match="historical transfer total deadline exceeded"
            ):
                await asyncio.wait_for(session.retry_commands(), timeout=2)
        finally:
            log.close()

    asyncio.run(scenario())


def test_history_can_progress_past_thirty_seconds_but_is_still_bounded(
    identity, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from care_vision.cloud import history as module

    clock = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    log = EventLog("cloud", tmp_path / "paced.jsonl")
    try:
        history = accepted_history(identity, log, count=100)
        deadline = history.timeout_seconds
        for i in range(100):
            clock[0] += 0.6
            history.check_timeout(clock[0])
            history.receive(*historical_frame(identity, i, 5_000_000_000 + i * 40_000_000))
        assert clock[0] - history.started > 30
        assert 60 < deadline <= 180
        prior_progress = history.last_progress
        history.acknowledge({**history.ack, "result": "duplicate"})
        assert history.last_progress == prior_progress
        assert history.timeout_seconds == deadline
        with pytest.raises(ProtocolError, match="total deadline"):
            history.check_timeout(history.started + deadline)
    finally:
        log.close()


def test_complete_but_unprocessed_history_still_has_idle_deadline(identity, tmp_path):
    log = EventLog("cloud", tmp_path / "stalled.jsonl")
    try:
        history = accepted_history(identity, log, count=100)
        history.complete = True
        with pytest.raises(ProtocolError, match="no progress"):
            history.check_timeout(history.last_progress + 30)
    finally:
        log.close()
