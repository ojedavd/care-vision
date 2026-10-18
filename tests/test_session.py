import asyncio
import time
from types import SimpleNamespace

import pytest
from conftest import command, frame, roi_command, status
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

import care_vision.cloud.server as server_module
from care_vision.check_loop import read_log
from care_vision.cloud.server import LIVE_QUEUE_CAPACITY, CloudServer, Session
from care_vision.common.contracts import (
    MAX_MESSAGE,
    ProtocolError,
    dumps,
    envelope,
    loads,
    pack_frame,
    unpack_frame,
    utc,
)
from care_vision.common.metrics import EventLog
from care_vision.common.transport import Outbox
from care_vision.edge.runtime import Edge
from care_vision.edge.source import VideoSource
from care_vision.fixture import make_video


def test_standalone_full_frame_dispatch_keeps_verification_and_fps(tmp_path, identity):
    class Socket:
        def __init__(self):
            self.sent = []

        async def send(self, data):
            self.sent.append(loads(data))

    async def scenario():
        socket = Socket()
        log = EventLog("cloud", tmp_path / "restore-context.jsonl")
        try:
            session = Session(socket, identity, log)
            controller = session.controller
            controller.state = "VERIFYING"
            controller.expected_command = "fps-20"
            controller.active_view_kind = "roi"
            request = controller.request_full_frame({
                "observation_id": "lost-person", "frame_id": "roi-frame",
                "captured_mono_ns": time.monotonic_ns(),
            }, reason="roi_person_unconfirmed")
            before = time.monotonic()
            await session.dispatch(request, {})
            assert socket.sent == [request]
            assert session.pending_since >= before
            assert session.control_last_sent[request["command_id"]] >= before
            assert controller.pending is None
            assert controller.visual_pending == request
            assert controller.expected_command == "fps-20"
            assert controller.state == "VERIFYING"
            assert session.history.request is None
        finally:
            log.close()
        actions = [r for r in read_log(tmp_path / "restore-context.jsonl")
                   if r["event"] == "agent_action"]
        assert len(actions) == 1 and actions[0]["agent_state"] == "VERIFYING"

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["command", "visual command"])
def test_retry_waits_for_each_send_without_extending_ack_deadline(
    tmp_path, identity, monkeypatch, kind
):
    clock = [100.0]
    monkeypatch.setattr(server_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    class Socket:
        def __init__(self):
            self.sent = []

        async def send(self, raw):
            self.sent.append(loads(raw))

    async def scenario():
        socket = Socket()
        log = EventLog("cloud", tmp_path / "paced-retry.jsonl")
        try:
            session = Session(socket, identity, log)
            request = command(identity) if kind == "command" else roi_command(identity)
            session.pending_since = 100.0
            session.control_last_sent[request["command_id"]] = 100.0
            clock[0] = 100.009  # The observed failure: global timer ticks 9 ms after send.
            await session.retry_control(request, kind)
            assert not socket.sent
            clock[0] = 101.01
            await session.retry_control(request, kind)
            assert socket.sent == [request]
            clock[0] = 101.5
            await session.retry_control(request, kind)
            assert socket.sent == [request]
            clock[0] = 105.0
            with pytest.raises(ProtocolError, match="unconfirmed"):
                await session.retry_control(request, kind)
            assert session.pending_since == 100.0
        finally:
            log.close()

    asyncio.run(scenario())


def test_outbox_bounds_and_control_priority():
    async def scenario():
        outbox = Outbox()
        for i in range(1000):
            outbox.put_frame(i)
        outbox.put_control("ACK")
        assert await outbox.get() == "ACK"
        assert await outbox.get() == 999
        assert outbox.dropped == 999

    asyncio.run(scenario())


def test_duplicate_frame_and_wrong_session_over_socket(tmp_path, identity):
    async def scenario():
        log = EventLog("cloud", tmp_path / "cloud.jsonl")
        cloud = CloudServer(log)
        try:
            async with serve(cloud.handler, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with connect(f"ws://127.0.0.1:{port}", proxy=None) as ws:
                    await ws.send(dumps(status(identity)))
                    header, jpeg = frame(identity)
                    await ws.send(pack_frame(header, jpeg))
                    await ws.send(pack_frame(header, jpeg))
                    other, jpeg = frame({**identity, "session_id": "old"}, 2)
                    await ws.send(pack_frame(other, jpeg))
                    await asyncio.wait_for(ws.wait_closed(), 2)
                    assert ws.close_code == 1008
        finally:
            log.close()
        records = read_log(tmp_path / "cloud.jsonl")
        assert len([r for r in records if r["event"] == "frame_received"]) == 1
        assert any(r["event"] == "duplicate_frame_rejected" for r in records)
        assert any("another session" in r.get("reason", "") for r in records)

    asyncio.run(scenario())


def test_slow_vision_drops_pending_frames_not_unbounded_queue(tmp_path, identity, monkeypatch):
    from care_vision.cloud.vision import MotionVision

    original = MotionVision.observe

    def slow(self, *args):
        time.sleep(0.1)
        return original(self, *args)

    monkeypatch.setattr(MotionVision, "observe", slow)

    async def scenario():
        log = EventLog("cloud", tmp_path / "slow.jsonl")
        try:
            async with serve(CloudServer(log).handler, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with connect(f"ws://127.0.0.1:{port}", proxy=None) as ws:
                    await ws.send(dumps(status(identity)))
                    for i in range(30):
                        header, jpeg = frame(identity, i + 1, 1_000_000_000 + i * 50_000_000)
                        await ws.send(pack_frame(header, jpeg))
                        await asyncio.sleep(0.001)
                    await asyncio.sleep(0.4)
        finally:
            log.close()
        records = read_log(tmp_path / "slow.jsonl")
        assert len([r for r in records if r["event"] == "frame_received"]) == 30
        assert len([r for r in records if r["event"] == "observation"]) < 10
        assert any(r["event"] == "frame_dropped" for r in records)

    asyncio.run(scenario())


def test_short_transport_burst_is_buffered_and_larger_backlog_is_explicit(tmp_path, identity):
    class BurstSocket:
        def __init__(self, messages):
            self.messages = iter(messages)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.messages)
            except StopIteration:
                raise StopAsyncIteration from None

    async def scenario():
        messages = []
        for index in range(LIVE_QUEUE_CAPACITY + 1):
            header, jpeg = frame(
                identity,
                index + 1,
                1_000_000_000 + index * 50_000_000,
            )
            messages.append(pack_frame(header, jpeg))
        log = EventLog("cloud", tmp_path / "burst.jsonl")
        session = Session(BurstSocket(messages), identity, log)
        try:
            await session.receive()
            assert session.frames.qsize() == LIVE_QUEUE_CAPACITY
            assert session.dropped == 1
            assert [item[0]["capture_seq"] for item in session.frames._queue] == [2, 3, 4, 5]
        finally:
            log.close()

    asyncio.run(scenario())
    records = read_log(tmp_path / "burst.jsonl")
    received = [record for record in records if record["event"] == "frame_received"]
    assert [record["live_queue_depth_before"] for record in received] == [0, 1, 2, 3, 4]
    dropped = [record for record in records if record["event"] == "frame_dropped"]
    assert dropped == [
        {
            **dropped[0],
            "frame_id": "boot:1",
            "capture_seq": 1,
            "replaced_by_frame_id": "boot:5",
            "replaced_by_capture_seq": 5,
            "reason": "vision_busy",
            "drop_policy": "latest_live_frame_wins",
            "intentional_backpressure": True,
            "live_queue_capacity": LIVE_QUEUE_CAPACITY,
        }
    ]


def test_ready_consumer_gets_first_frame_before_five_frame_arrival_burst(tmp_path, identity):
    class BurstSocket:
        def __init__(self):
            self.messages = []
            for i in range(5):
                header, jpeg = frame(identity, i + 1, 1_000_000_000 + i * 50_000_000)
                self.messages.append(pack_frame(header, jpeg))

        async def __aiter__(self):
            for message in self.messages:
                yield message  # All bytes already available, as in the observed AWS burst.

    async def scenario():
        log = EventLog("cloud", tmp_path / "fair-burst.jsonl")
        session = Session(BurstSocket(), identity, log)
        active = []
        release = asyncio.Event()

        async def consume():
            item = await session.frames.get()
            active.append(item[0]["capture_seq"])
            await release.wait()  # A frame is in inference while the remaining four arrive.

        worker = asyncio.create_task(consume())
        try:
            await asyncio.sleep(0)  # Consumer is ready and waiting on the empty queue.
            await session.receive()
            assert session.dropped == 0
            assert active == [1]
            assert [item[0]["capture_seq"] for item in session.frames._queue] == [2, 3, 4, 5]
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            log.close()

    asyncio.run(scenario())


def test_edge_reconnects_capture_continues_and_static_never_escalates(tmp_path):
    video = make_video(tmp_path / "static.avi", static=True, seconds=5)

    async def scenario():
        cloud_log = EventLog("cloud", tmp_path / "cloud.jsonl")
        edge_log = EventLog("edge", tmp_path / "edge.jsonl")
        cloud = CloudServer(cloud_log)
        source = VideoSource(str(video))
        try:
            async with serve(cloud.handler, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                edge = Edge(source, f"ws://127.0.0.1:{port}", edge_log)
                task = asyncio.create_task(edge.run())
                try:
                    await asyncio.sleep(1.5)
                    assert cloud.active
                    await cloud.active[0].close(code=1012, reason="test restart")
                    await asyncio.wait_for(task, timeout=8)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        finally:
            cloud_log.close()
            edge_log.close()
        edge_records = read_log(tmp_path / "edge.jsonl")
        connected = [r for r in edge_records if r["event"] == "connected"]
        assert len(connected) == 2
        assert len({r["boot_id"] for r in connected}) == 1
        assert len({r["session_id"] for r in connected}) == 2
        assert source.count == 150
        assert not source.thread.is_alive()
        actions = [r for r in read_log(tmp_path / "cloud.jsonl") if r["event"] == "agent_action"]
        assert all(r["reason"] == "recover_offline_evidence" for r in actions)

    asyncio.run(scenario())


def test_fixed_20_fps_survives_connection_reset_and_transmits_frames(tmp_path):
    video = make_video(tmp_path / "fixed20.avi", static=True, seconds=2)
    received = []
    initial_statuses = []

    async def scenario():
        async def fake_cloud(ws):
            initial = loads(await ws.recv())
            initial_statuses.append(initial)
            identity = {key: initial[key] for key in ("edge_id", "boot_id", "session_id")}
            async for raw in ws:
                if isinstance(raw, bytes):
                    received.append(unpack_frame(raw))
                else:
                    message = loads(raw)
                    if message["type"] == "DRAIN_REQUEST":
                        await ws.send(dumps(envelope(
                            "DRAINED", identity,
                            drain_id=message["drain_id"],
                            live_received=message["live_frames_sent"],
                            live_processed=message["live_frames_sent"],
                            live_dropped=0,
                            historical_received=0,
                            historical_processed=0,
                        )))

        log = EventLog("edge", tmp_path / "fixed20.jsonl")
        try:
            async with serve(fake_cloud, "127.0.0.1", 0, max_size=MAX_MESSAGE) as server:
                port = server.sockets[0].getsockname()[1]
                await Edge(
                    VideoSource(str(video)), f"ws://127.0.0.1:{port}", log,
                    initial_transmit_fps=20,
                ).run()
        finally:
            log.close()

    asyncio.run(scenario())
    assert initial_statuses[0]["target_transmit_fps"] == 20
    assert len(received) >= 30
    connected = [row for row in read_log(tmp_path / "fixed20.jsonl")
                 if row["event"] == "connected"]
    assert connected[0]["baseline_fps"] == 20


def test_real_edge_acknowledges_duplicate_and_rejects_expired(tmp_path):
    video = make_video(tmp_path / "commands.avi", static=True, seconds=2)
    answers = []

    async def scenario():
        async def fake_cloud(ws):
            initial = loads(await ws.recv())
            identity = {k: initial[k] for k in ("edge_id", "boot_id", "session_id")}
            now = time.monotonic_ns()
            cmd = command(identity, deadline=now + 5_000_000_000)
            await ws.send(dumps(cmd))
            await ws.send(dumps(cmd))
            await ws.send(dumps(command(identity, seq=2, command_id="old", fps=2, deadline=1)))
            async for raw in ws:
                if isinstance(raw, str):
                    message = loads(raw)
                    if message["type"] == "COMMAND_ACK":
                        answers.append(message)
                    elif message["type"] == "DRAIN_REQUEST":
                        await ws.send(
                            dumps(
                                envelope(
                                    "DRAINED",
                                    identity,
                                    drain_id=message["drain_id"],
                                    live_received=message["live_frames_sent"],
                                    live_processed=message["live_frames_sent"],
                                    live_dropped=0,
                                    historical_received=0,
                                    historical_processed=0,
                                )
                            )
                        )

        log = EventLog("edge", tmp_path / "edge.jsonl")
        try:
            async with serve(fake_cloud, "127.0.0.1", 0, max_size=MAX_MESSAGE) as server:
                port = server.sockets[0].getsockname()[1]
                await Edge(VideoSource(str(video)), f"ws://127.0.0.1:{port}", log).run()
        finally:
            log.close()

    asyncio.run(scenario())
    assert [a["result"] for a in answers] == ["applied", "duplicate", "expired"]
    assert all(a["active_transmit_fps"] == 20 for a in answers)


def test_real_edge_applies_roi_then_reduced_full_frame(tmp_path):
    video = make_video(tmp_path / "visual.avi", static=True, seconds=3)
    answers = []
    frames = []

    async def scenario():
        async def fake_cloud(ws):
            initial = loads(await ws.recv())
            identity = {key: initial[key] for key in ("edge_id", "boot_id", "session_id")}
            deadline = time.monotonic_ns() + 5_000_000_000
            fps = command(identity, deadline=deadline)
            roi = roi_command(identity, deadline=deadline)
            await ws.send(dumps(fps))
            await ws.send(dumps(roi))
            full_sent = False
            async for raw in ws:
                if isinstance(raw, str):
                    message = loads(raw)
                    if message["type"] == "COMMAND_ACK":
                        answers.append(message)
                    elif message["type"] == "DRAIN_REQUEST":
                        await ws.send(
                            dumps(
                                envelope(
                                    "DRAINED",
                                    identity,
                                    drain_id=message["drain_id"],
                                    live_received=len(frames),
                                    live_processed=len(frames),
                                    live_dropped=0,
                                    historical_received=0,
                                    historical_processed=0,
                                )
                            )
                        )
                else:
                    header, jpeg = unpack_frame(raw)
                    frames.append((header, jpeg))
                    if header.get("view_kind") == "roi" and not full_sent:
                        full = envelope(
                            "REQUEST_FULL_FRAME",
                            identity,
                            command_id="F1",
                            command_seq=3,
                            issued_at_utc=utc(),
                            caused_by_observation_id="O2",
                            based_on_frame_id=header["frame_id"],
                            output_width=320,
                            output_height=180,
                            reason="return_to_baseline",
                            valid_until_edge_mono_ns=deadline,
                        )
                        await ws.send(dumps(full))
                        full_sent = True

        log = EventLog("edge", tmp_path / "visual-edge.jsonl")
        try:
            async with serve(fake_cloud, "127.0.0.1", 0, max_size=MAX_MESSAGE) as server:
                port = server.sockets[0].getsockname()[1]
                await Edge(VideoSource(str(video)), f"ws://127.0.0.1:{port}", log).run()
        finally:
            log.close()

    asyncio.run(scenario())
    assert [answer["command_type"] for answer in answers[:3]] == [
        "SET_TRANSMIT_FPS",
        "REQUEST_ROI",
        "REQUEST_FULL_FRAME",
    ]
    roi_frames = [header for header, _ in frames if header.get("view_kind") == "roi"]
    reduced = [
        header
        for header, _ in frames
        if header.get("view_kind") == "full_frame" and header["width"] == 320
    ]
    assert roi_frames and reduced
    assert all(header["applied_visual_command_id"] == "R1" for header in roi_frames)
    assert all(header["applied_visual_command_id"] == "F1" for header in reduced)


def test_missing_ack_retries_same_command_then_fails_explicitly(tmp_path, identity):
    class Socket:
        def __init__(self):
            self.sent = []

        async def send(self, data):
            self.sent.append(loads(data))

    async def scenario():
        socket = Socket()
        log = EventLog("cloud", tmp_path / "retry.jsonl")
        session = Session(socket, identity, log)
        original = command(identity)
        session.controller.pending = original
        session.pending_since = time.monotonic() - 3.5
        try:
            with pytest.raises(ProtocolError, match="command unconfirmed"):
                await asyncio.wait_for(session.retry_commands(), 3)
            assert socket.sent == [original]
            assert session.controller.pending == original
        finally:
            log.close()

    asyncio.run(scenario())
