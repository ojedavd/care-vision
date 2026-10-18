import asyncio
import socket

import pytest
from conftest import frame
from websockets.asyncio.server import serve

from care_vision.check_loop import read_log
from care_vision.cloud.server import CloudServer
from care_vision.common.contracts import ProtocolError, dumps
from care_vision.common.metrics import EventLog
from care_vision.edge.buffer import Capture, RingBuffer
from care_vision.edge.network import NetworkMonitor
from care_vision.edge.runtime import Edge
from care_vision.edge.source import VideoSource
from care_vision.fixture import make_video


@pytest.mark.parametrize("baseline", [0.08, 0.17, 0.19])
def test_recovery_accounts_for_healthy_baseline_without_learning_degradation(baseline):
    monitor = NetworkMonitor()
    monitor.connected(0)
    for _ in range(6):
        monitor.observed_rtt(baseline)
    assert monitor.quality == "GOOD"
    threshold = monitor.recovery_threshold_ms
    assert 150 <= threshold <= 200
    for _ in range(20):
        monitor.observed_rtt(0.35)
    assert monitor.transmission_cap_fps == 5
    assert monitor.recovery_threshold_ms == threshold
    for _ in range(10):
        monitor.observed_rtt(0.23)
    assert monitor.transmission_cap_fps == 5  # A bad baseline cannot teach recovery.
    monitor.observed_rtt(baseline)
    assert monitor.transmission_cap_fps == 5
    monitor.observed_rtt(baseline)
    assert monitor.transmission_cap_fps is None
    monitor.disconnected(0)
    monitor.connected(0)
    assert monitor.recovery_threshold_ms == 150


def test_error_preserves_failed_session_worker():
    from care_vision.common.transport import run_until_first

    async def missing_pong():
        raise TimeoutError("pong timeout")

    async def scenario():
        with pytest.raises(TimeoutError, match="pong timeout") as caught:
            await run_until_first(missing_pong())
        assert caught.value.failed_worker.endswith("missing_pong")

    asyncio.run(scenario())


def test_severe_rtt_caps_before_waiting_for_a_second_slow_pong():
    monitor = NetworkMonitor()
    monitor.connected(0)
    for _ in range(6):
        monitor.observed_rtt(0.17)
    monitor.observed_rtt(2.84)
    assert monitor.transmission_cap_fps == 5
    assert monitor.quality == "DEGRADED"
    monitor.observed_rtt(0.17)
    assert monitor.transmission_cap_fps == 5
    monitor.observed_rtt(0.17)
    assert monitor.transmission_cap_fps is None


def test_network_quality_hysteresis_and_offline_capture_accounting():
    monitor = NetworkMonitor()
    assert monitor.quality == "OFFLINE"
    outage = monitor.connected(20, now=monitor.offline_since + 1)
    assert monitor.transmission_cap_fps is None
    assert outage["captured_while_offline"] == 20
    assert outage["offline_seconds"] == 1
    assert monitor.outage(20) == outage
    assert monitor.observed_rtt(0.3) == "DEGRADED"
    assert monitor.transmission_cap_fps is None
    assert monitor.observed_rtt(0.3) == "DEGRADED"
    assert monitor.transmission_cap_fps == 5
    assert monitor.observed_rtt(0.2) == "DEGRADED"
    assert monitor.observed_rtt(0.1) == "DEGRADED"
    assert monitor.observed_rtt(0.1) == "GOOD"
    assert monitor.transmission_cap_fps is None
    monitor.disconnected(45, now=10)
    assert monitor.quality == "OFFLINE" and monitor.rtt_ms is None
    assert monitor.connected(75, now=11) == {"offline_seconds": 1, "captured_while_offline": 30}
    assert monitor.outage(90) == {"offline_seconds": 1, "captured_while_offline": 30}
    with pytest.raises(ValueError):
        monitor.observed_rtt(float("nan"))


def test_measured_high_rtt_caps_only_escalated_transmission_and_recovers(tmp_path):
    source = VideoSource(str(tmp_path / "unused.avi"))
    log = EventLog("edge", tmp_path / "cap.jsonl")
    edge = Edge(source, "ws://127.0.0.1:1", log)
    try:
        start = 100_000_000_000
        edge.network.connected(0)
        edge.sampler.set_rate(20, start)
        normal = sum(
            edge.select_live_capture(seq, start + (seq - 1) * 1_000_000_000 // 30)
            for seq in range(1, 31)
        )
        edge.observe_network_rtt(0.3, start + 1_000_000_000)
        assert edge.network.transmission_cap_fps is None
        edge.observe_network_rtt(0.3, start + 1_000_000_000)
        assert edge.status()["effective_target_transmit_fps"] == 5
        assert edge.history_limits() == (2, 64 * 1024)
        limited = sum(
            edge.select_live_capture(seq, start + (seq - 1) * 1_000_000_000 // 30)
            for seq in range(31, 61)
        )
        edge.observe_network_rtt(0.1, start + 2_000_000_000)
        edge.observe_network_rtt(0.1, start + 2_000_000_000)
        assert edge.status()["effective_target_transmit_fps"] == 20
        assert edge.history_limits() == (5, 256 * 1024)
        recovered = sum(
            edge.select_live_capture(seq, start + (seq - 1) * 1_000_000_000 // 30)
            for seq in range(61, 91)
        )
        assert 18 <= normal <= 21
        assert 4 <= limited <= 6
        assert edge.network_limited_frames >= 12
        assert 18 <= recovered <= 21
        assert edge.sampler.fps == 20  # Cloud policy remains intact.
    finally:
        log.close()


def test_frame_contract_distinguishes_requested_and_network_limited_target(identity):
    header, _ = frame(identity, policy_seq=1, command_id="C1")
    limited = {
        **header,
        "effective_target_transmit_fps": 5,
        "network_limited": True,
    }
    dumps(limited)
    for change in (
        {"network_limited": False},
        {"effective_target_transmit_fps": 25},
    ):
        with pytest.raises(ProtocolError):
            dumps({**limited, **change})


def test_status_contract_rejects_quality_contradictions(tmp_path):
    source = VideoSource(str(tmp_path / "unused.avi"))
    log = EventLog("edge", tmp_path / "status.jsonl")
    try:
        edge = Edge(source, "ws://127.0.0.1:1", log)
        status = edge.status()
        assert status["network_quality"] == "OFFLINE"
        assert status["cloud_analysis_available"] is False
        assert status["captured_while_offline"] == 0
        dumps(status)
        for change in (
            {"network_quality": "BAD"},
            {"network_quality": "GOOD"},
            {"cloud_analysis_available": True},
            {"network_rtt_ms": -1},
            {"effective_target_transmit_fps": 5},
            {"network_cap_fps": 5, "effective_target_transmit_fps": 1},
            {"offline_frames_lost_at_reconnect": 1},
            {"recovery_from_edge_mono_ns": 10},
            {
                "recovery_from_edge_mono_ns": 10,
                "recovery_to_edge_mono_ns": 10,
                "recovery_available_frames": 1,
                "recovery_truncated": False,
            },
        ):
            with pytest.raises(ProtocolError):
                dumps({**status, **change})
    finally:
        log.close()


def test_eof_without_cloud_confirmation_exits_with_explicit_failure(tmp_path):
    video = make_video(tmp_path / "offline.avi", static=True, seconds=2)
    source = VideoSource(str(video))
    log = EventLog("edge", tmp_path / "edge.jsonl")

    async def scenario():
        with pytest.raises(RuntimeError, match="without cloud processing confirmation"):
            await asyncio.wait_for(Edge(source, "ws://127.0.0.1:1", log).run(), 6)

    try:
        asyncio.run(scenario())
    finally:
        log.close()
    records = read_log(tmp_path / "edge.jsonl")
    assert source.count == 60 and not source.thread.is_alive()
    assert any(r["event"] == "cloud_processing_unconfirmed_at_eof" for r in records)
    assert any(
        r["event"] == "network_transition" and r["network_quality"] == "OFFLINE" for r in records
    )


def test_initial_outage_capture_continues_then_cloud_drain_confirms(tmp_path):
    video = make_video(tmp_path / "recover.avi", static=True, seconds=4)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    source = VideoSource(str(video))
    edge_log = EventLog("edge", tmp_path / "edge.jsonl")
    cloud_log = EventLog("cloud", tmp_path / "cloud.jsonl")

    async def scenario():
        task = asyncio.create_task(Edge(source, f"ws://127.0.0.1:{port}", edge_log).run())
        try:
            await asyncio.sleep(1)
            async with serve(CloudServer(cloud_log).handler, "127.0.0.1", port):
                await asyncio.wait_for(task, 8)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        edge_log.close()
        cloud_log.close()
    records = read_log(tmp_path / "edge.jsonl")
    connected = next(r for r in records if r["event"] == "connected")
    assert connected["captured_while_offline"] > 0
    assert connected["offline_seconds"] >= 0.9
    assert any(
        r["event"] == "status_received" and r["captured_while_offline"] > 0
        for r in read_log(tmp_path / "cloud.jsonl")
    )
    cloud_records = read_log(tmp_path / "cloud.jsonl")
    initial = next(r for r in cloud_records if r["event"] == "status_received")
    assert initial["initial"] is True
    assert initial["offline_frames_retained_at_reconnect"] == connected["captured_while_offline"]
    assert initial["offline_frames_lost_at_reconnect"] == 0
    recovery = next(r for r in cloud_records if r["event"] == "offline_recovery_requested")
    assert recovery["available_frames"] > 0
    assert recovery["truncated"] is False
    assert any(
        r["event"] == "ack_received"
        and r["command_id"] == recovery["buffer_request_id"]
        and r["selected_frames"] > 0
        for r in cloud_records
    )
    historical = [
        r
        for r in cloud_records
        if r["event"] == "observation" and r["evidence_kind"] == "historical"
    ]
    assert historical
    historical_ids = {r["frame_id"] for r in historical}
    assert not any(
        r.get("based_on_frame_id") in historical_ids
        for r in cloud_records
        if r["event"] == "agent_action"
    )
    drained = next(r for r in cloud_records if r["event"] == "drain_confirmed")
    assert drained["historical_received"] == drained["historical_processed"] > 0
    assert source.count == 120
    assert any(r["event"] == "drain_complete" for r in records)
    assert not any(r["event"] == "cloud_processing_unconfirmed_at_eof" for r in records)


def test_recovery_window_reports_evicted_offline_prefix(tmp_path):
    source = VideoSource(str(tmp_path / "unused.avi"))
    for seq in range(1, 361):
        source.buffer.append(Capture(seq, seq * 100_000_000, "2026-09-22T00:00:00Z", b"x"))
    window = source.recovery_window(1, 360)
    assert window["recovery_truncated"] is True
    assert window["recovery_available_frames"] == 100
    assert window["recovery_from_edge_mono_ns"] == 26_100_000_000
    assert window["recovery_to_edge_mono_ns"] == 36_000_000_001
    assert window["offline_frames_retained_at_reconnect"] == 100
    assert window["offline_frames_lost_at_reconnect"] == 260
    context = source.recovery_window(311, 360, context_from_ns=20_000_000_000)
    assert context["recovery_includes_unconfirmed_context"] is True
    assert context["recovery_truncated"] is True
    assert context["recovery_available_frames"] == 100
    offline_only = source.recovery_window(311, 360)
    assert context["recovery_from_edge_mono_ns"] < offline_only["recovery_from_edge_mono_ns"]
    retained_context = source.recovery_window(311, 360, context_from_ns=28_000_000_000)
    assert retained_context["recovery_truncated"] is False


def test_recovery_accounts_for_outage_when_buffer_retains_nothing(tmp_path):
    source = VideoSource(str(tmp_path / "unused.avi"))
    source.buffer = RingBuffer(max_bytes=1)
    source.buffer.append(Capture(1, 1_000_000_000, "2026-09-22T00:00:00Z", b"xx"))
    assert source.recovery_window(1, 1) == {
        "recovery_window_available": False,
        "offline_frames_retained_at_reconnect": 0,
        "offline_frames_lost_at_reconnect": 1,
    }
