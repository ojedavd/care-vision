import copy

import pytest

from care_vision.adaptive_evaluation import audit_adaptive_run


@pytest.fixture
def run():
    video = {
        "video_id": "test",
        "fps": 10,
        "frame_count": 30,
        "events": [],
        "segments": [{"start_frame": 0, "end_frame_exclusive": 30, "phase": "normal"}],
    }

    def frame(seq, kind="live", session="session"):
        return {
            "event": "frame_sent",
            "frame_id": f"boot:{seq}",
            "boot_id": "boot",
            "session_id": session,
            "capture_seq": seq,
            "captured_mono_ns": 100_000_000_000 + (seq - 1) * 100_000_000,
            "local_mono_ns": 200_000_000_000 + seq * 100_000_000,
            "evidence_kind": kind,
            "buffer_request_id": "request" if kind == "historical" else None,
            "payload_bytes": 100,
            "applied_command_id": None,
            "primary_track_id": "person",
            "primary_orientation": "vertical",
            "primary_motion_direction": "stable",
            "observation_id": f"obs:{seq}",
        }

    live1, live2, history = frame(1), frame(6), frame(2, "historical")

    def receive(r, t):
        return {**r, "event": "frame_received", "local_mono_ns": t}

    def observe(r, t):
        return {**r, "event": "observation", "local_mono_ns": t}

    edge = [
        live1,
        live2,
        history,
        {"event": "source_stopped", "captured": 30, "source_error": None},
    ]
    cloud = [
        {"event": "server_ready", "phase": 1, "runtime": {}, "local_mono_ns": 0},
        receive(live1, 1_000_000_000),
        observe(live1, 2_000_000_000),
        {
            "event": "agent_action",
            "type": "REQUEST_RECENT_BUFFER",
            "command_id": "request",
            "local_mono_ns": 3_000_000_000,
            "caused_by_observation_id": "obs:1",
            "based_on_frame_id": "boot:1",
            "session_id": "session",
            "history_fps": 20,
            "from_edge_mono_ns": 100_000_000_000,
            "to_edge_mono_ns": 100_500_000_000,
        },
        {
            "event": "ack_received",
            "command_id": "request",
            "local_mono_ns": 4_000_000_000,
            "result": "applied",
            "selected_frames": 1,
            "coverage": "full",
        },
        receive(live2, 5_000_000_000),
        observe(live2, 6_000_000_000),
        receive(history, 7_000_000_000),
        observe(history, 8_000_000_000),
        {
            "event": "buffer_processed",
            "buffer_request_id": "request",
            "local_mono_ns": 9_000_000_000,
        },
    ]
    return edge, cloud, video


def test_delayed_history_uses_processing_order_and_same_host_clock(run):
    report = audit_adaptive_run(*run)
    assert report["counts"] == {
        "sent": 3,
        "received": 3,
        "processed": 3,
        "sent_without_receipt": 0,
        "received_without_observation": 0,
    }
    assert report["actions"][0]["ack_round_trip_ms"] == 1000
    assert report["actions"][0]["first_history_processing_after_request_ms"] == 5000
    assert report["actions"][0]["completed_after_request_ms"] == 6000
    assert report["processing_timeline"][-1]["source_frame_index"] == 1
    assert report["processing_timeline"][-1]["latest_live_source_frame_at_processing"] == 5
    assert len(report["sequences"]) == 2
    assert not report["original_demo_check"]["passed"]


def test_disconnect_and_incomplete_transfer_are_results_not_hidden(run):
    edge, cloud, video = run
    cloud.pop()  # no BUFFER_COMPLETE processing record
    cloud.pop()  # received historical frame never processed
    cloud.append(
        {
            "event": "buffer_incomplete",
            "buffer_request_id": "request",
            "local_mono_ns": 9_000_000_000,
        }
    )
    edge.append({"event": "connection_error", "reason": "test outage"})
    report = audit_adaptive_run(edge, cloud, video)
    assert report["counts"]["received_without_observation"] == 1
    assert not report["actions"][0]["complete"]
    assert report["actions"][0]["first_history_processing_after_request_ms"] is None
    assert len(report["recorded_failures"]) == 2


def test_reconnection_has_separate_sequence_state_even_with_reused_track_id(run):
    edge, cloud, video = run
    for rows in (edge, cloud):
        for r in rows:
            if r.get("frame_id") == "boot:6":
                r["session_id"] = "reconnected"
    report = audit_adaptive_run(edge, cloud, video)
    assert report["sessions"] == 2
    assert {s["stream"] for s in report["sequences"]} == {
        "live:session",
        "live:reconnected",
        "historical:request",
    }


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate_send",
        "missing_send",
        "metadata",
        "source",
        "out_of_range",
        "cloud_order",
        "historical_window",
        "request_session",
        "command_causality",
        "ack_clock",
    ],
)
def test_rejects_corrupt_provenance_and_clock_order(run, damage):
    edge, cloud, video = copy.deepcopy(run)
    if damage == "duplicate_send":
        edge.append(copy.deepcopy(edge[0]))
    elif damage == "missing_send":
        edge.pop(0)
    elif damage == "metadata":
        cloud[1]["capture_seq"] = 3
    elif damage == "source":
        edge[-1]["captured"] = 29
    elif damage == "out_of_range":
        edge[0]["capture_seq"] = 31
    elif damage == "cloud_order":
        cloud[1], cloud[2] = cloud[2], cloud[1]
    elif damage == "historical_window":
        cloud[3]["to_edge_mono_ns"] = 100_100_000_000
    elif damage == "request_session":
        cloud[3]["session_id"] = "other"
    elif damage == "command_causality":
        cloud[3]["caused_by_observation_id"] = "obs:6"
    else:
        cloud[4]["local_mono_ns"] = 2_500_000_000
    with pytest.raises(ValueError):
        audit_adaptive_run(edge, cloud, video)


def test_unknown_boxes_are_counted_separately_from_negative_events(run):
    edge, cloud, video = run
    for r in cloud:
        if r["event"] == "observation":
            r.update(
                primary_track_id=None, primary_orientation=None, primary_motion_direction="unknown"
            )
    report = audit_adaptive_run(edge, cloud, video)
    assert all(r["bbox_available_frames"] == 0 for r in report["phase_coverage"])
    assert all(s["state"] == "UNKNOWN" for stream in report["sequences"] for s in stream["states"])
    assert all(not stream["episodes"] for stream in report["sequences"])


def test_second_perception_requires_new_policy_and_valid_observation(run):
    edge, cloud, video = run
    command = {
        "event": "agent_action",
        "type": "SET_TRANSMIT_FPS",
        "command_id": "fps",
        "transmit_fps": 20,
        "local_mono_ns": 2_500_000_000,
        "caused_by_observation_id": "obs:1",
        "based_on_frame_id": "boot:1",
    }
    cloud.insert(3, command)
    for records in (edge, cloud):
        for row in records:
            if row.get("frame_id") == "boot:6":
                row["applied_command_id"] = "fps"
                row["valid"] = False
    action = audit_adaptive_run(edge, cloud, video)["actions"][0]
    assert action["updated_live_observations"] == 1
    assert action["first_valid_new_perception_after_command_ms"] is None
    for row in cloud:
        if row["event"] == "observation" and row["frame_id"] == "boot:6":
            row["valid"] = True
    action = audit_adaptive_run(edge, cloud, video)["actions"][0]
    assert action["first_valid_new_perception_after_command_ms"] == 3500
