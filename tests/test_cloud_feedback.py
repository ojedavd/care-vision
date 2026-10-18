import asyncio

import pytest
from conftest import frame, status
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from care_vision.cloud.server import CloudServer
from care_vision.common.contracts import ProtocolError, dumps, envelope, loads, pack_frame, validate
from care_vision.common.metrics import EventLog
from care_vision.common.transport import Outbox
from care_vision.edge.cloud_feedback import CloudFeedback
from care_vision.edge.runtime import Edge
from care_vision.edge.source import VideoSource


def feedback(identity, seq=1, **changes):
    fields = dict(status_seq=seq, live_received=10, live_processed=9,
                  live_dropped=1, live_stale=0, queue_depth=0, queue_capacity=4)
    fields.update(changes)
    return envelope("CLOUD_STATUS", identity, **fields)


def test_freshness_known_gap_and_new_session(identity):
    monitor = CloudFeedback(identity)
    assert monitor.fields("GOOD", 1)["pipeline_quality"] == "UNKNOWN"
    monitor.accept(feedback(identity, live_processed=10, live_dropped=0), 1)
    assert monitor.fields("GOOD", 2)["pipeline_quality"] == "GOOD"
    assert monitor.fields("GOOD", 4.01)["pipeline_quality"] == "UNKNOWN"
    monitor.accept(feedback(identity, 2, live_received=11, live_processed=10), 5)
    assert monitor.fields("GOOD", 5)["pipeline_quality"] == "DEGRADED"
    fields = monitor.fields("GOOD", 10)
    assert fields["cloud_feedback_state"] == "STALE"
    assert fields["pipeline_quality"] == "DEGRADED"  # A stale report cannot erase known loss.
    assert monitor.fields("OFFLINE", 10)["pipeline_quality"] == "OFFLINE"
    assert CloudFeedback({**identity, "session_id": "new"}).fields("GOOD")[
        "cloud_live_dropped"] is None


@pytest.mark.parametrize("changes", [
    {"session_id": "old"}, {"status_seq": 1},
    {"live_received": 9, "live_processed": 8}, {"live_dropped": 0, "live_processed": 10},
])
def test_reject_stale_identity_sequence_and_counters(identity, changes):
    monitor = CloudFeedback(identity)
    monitor.accept(feedback(identity), 1)
    with pytest.raises(ProtocolError):
        monitor.accept({**feedback(identity, 2), **changes}, 2)
    assert monitor.received_at == 1


@pytest.mark.parametrize("changes", [
    {"live_stale": 10}, {"live_processed": 11}, {"queue_depth": 5},
    {"live_received": 15, "queue_depth": 0}, {"status_seq": True},
])
def test_reject_contradictory_wire_accounting(identity, changes):
    with pytest.raises(ProtocolError):
        validate(feedback(identity, **changes))


def test_stale_processing_is_visible_without_a_queue_drop(identity):
    monitor = CloudFeedback(identity)
    monitor.accept(feedback(identity, live_processed=10, live_dropped=0, live_stale=1))
    assert monitor.fields("GOOD")["pipeline_quality"] == "DEGRADED"


def test_edge_routes_feedback_without_changing_link_quality_or_rate(tmp_path):
    source = VideoSource(str(tmp_path / "unused.avi"))
    log = EventLog("edge", tmp_path / "edge.jsonl")
    edge = Edge(source, "ws://127.0.0.1:1", log)
    edge.outbox = Outbox()
    edge.network.connected(0)
    edge.network.observed_rtt(.1)
    edge.network.observed_rtt(.1)
    class Socket:
        def __aiter__(self):
            async def messages():
                yield dumps(feedback(edge.identity))
            return messages()
    try:
        asyncio.run(edge.receive(Socket()))
        state = validate(edge.status())
        assert state["pipeline_quality"] == "DEGRADED"
        assert state["network_quality"] == "GOOD"
        assert state["network_cap_fps"] is None
        assert state["target_transmit_fps"] == 2
        with pytest.raises(ProtocolError, match="GOOD pipeline"):
            validate({**state, "pipeline_quality": "GOOD"})
        with pytest.raises(ProtocolError, match="unknown cloud feedback"):
            validate({**state, "cloud_status_seq": None})
    finally:
        log.close()


def test_real_socket_reports_drop_and_final_counts_before_drain(tmp_path, identity, monkeypatch):
    import time

    from care_vision.cloud.vision import MotionVision
    original = MotionVision.observe
    def slow(self, *args):
        time.sleep(.05)
        return original(self, *args)
    monkeypatch.setattr(MotionVision, "observe", slow)

    async def scenario():
        log = EventLog("cloud", tmp_path / "cloud.jsonl")
        try:
            async with serve(CloudServer(log).handler, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with connect(f"ws://127.0.0.1:{port}", proxy=None) as ws:
                    await ws.send(dumps({**status(identity), "cloud_status_version": 1}))
                    for seq in range(1, 13):
                        header, jpeg = frame(identity, seq, 1_000_000_000 + seq * 50_000_000)
                        await ws.send(pack_frame(header, jpeg))
                    await ws.send(dumps(envelope("DRAIN_REQUEST", identity,
                                                 drain_id="done", live_frames_sent=12)))
                    monitor = CloudFeedback(identity)
                    while True:
                        message = loads(await asyncio.wait_for(ws.recv(), 3))
                        if message["type"] == "CLOUD_STATUS":
                            monitor.accept(message)
                        elif message["type"] == "DRAINED":
                            assert monitor.latest["live_dropped"] > 0
                            for field in ("live_received", "live_processed", "live_dropped"):
                                assert monitor.latest[field] == message[field]
                            assert monitor.fields("GOOD")["pipeline_quality"] == "DEGRADED"
                            break
                        else:
                            pytest.fail(f"unexpected message {message['type']}")
        finally:
            log.close()
    asyncio.run(scenario())
