import copy

import pytest

from care_vision.cloud.event_context import EventContext, source_frame
from care_vision.event_context_evaluation import replay_event_contexts


def observation(t, bbox=None, kind="live", track="live-person", **extra):
    index = round(t * 100)
    return {
        "event": "observation",
        "local_mono_ns": round((t + 100) * 10**9),
        "edge_id": "edge",
        "boot_id": "boot",
        "session_id": "session",
        "frame_id": f"boot:{index + 1}",
        "capture_seq": index + 1,
        "captured_mono_ns": round(t * 10**9),
        "observation_id": f"obs:{index}",
        "width": 100,
        "height": 100,
        "source_width": 100,
        "source_height": 100,
        "view_kind": "full_frame",
        "persons": [{"bbox": bbox or [40, 20, 10, 30], "confidence": 0.9, "track_id": track}],
        "person_count": 1,
        "primary_track_id": track,
        "person_detection_status": "ok",
        "evidence_kind": kind,
        "buffer_request_id": "request" if kind == "historical" else None,
        "valid": True,
        **extra,
    }


def request():
    return {
        "event": "agent_action",
        "type": "REQUEST_RECENT_BUFFER",
        "command_id": "request",
        "local_mono_ns": 101_100_000_000,
        "edge_id": "edge",
        "boot_id": "boot",
        "session_id": "session",
        "based_on_frame_id": "boot:101",
        "caused_by_observation_id": "obs:100",
        "from_edge_mono_ns": 0,
        "to_edge_mono_ns": 1_000_000_000,
    }


def live_rows():
    return [
        observation(0, [40, 0, 10, 30]),
        observation(0.5, [40, 5, 10, 30]),
        observation(1),
        *[observation(t, [20, 40, 50, 20]) for t in (1.5, 2, 2.5, 3)],
    ]


def context():
    c = EventContext(request(), observation(1))
    for row in live_rows():
        c.add(row)
    return c


def test_rebuild_anchors_raw_detection_and_ignores_prior_motion_labels():
    c = context()
    result = c.snapshot(110_000_000_000)
    assert result["identity_status"] == "experimental_anchor_association"
    assert result["anchor_context_track_id"].startswith("context:request:")
    assert result["episodes"]
    other = EventContext(request(), observation(1, track="historical:unrelated"))
    for r in live_rows():
        r["persons"][0]["track_id"] = "historical:unrelated"
        r["primary_track_id"] = "historical:unrelated"
        r.update(primary_motion_direction="upward", primary_orientation="vertical", phase="normal")
        other.add(r)
    assert other.snapshot(110_000_000_000) == result


def test_roi_letterbox_projection_uses_capture_coordinates():
    row = observation(
        1, [70, 20, 20, 40], width=200, height=100, view_kind="roi", source_roi=[20, 10, 50, 50]
    )
    assert source_frame(row)["persons"][0]["bbox"] == [30, 20, 10, 20]


def test_late_history_revises_context_without_backdating_availability():
    c = context()
    before = c.snapshot(110_000_000_000)
    old = copy.deepcopy(before)
    c.add(observation(0.75, [40, 10, 10, 30], kind="historical", track="history-person"))
    after = c.snapshot(111_000_000_000)
    baseline = c.snapshot(111_000_000_000, include_history=False)
    assert before == old
    assert before["historical_frames"] == 0
    assert after["historical_frames"] == 1
    assert after["available_at_cloud_mono_ns"] == 111_000_000_000
    assert baseline["historical_frames"] == 0
    assert baseline["episodes"] == before["episodes"]
    with pytest.raises(ValueError, match="backwards"):
        c.snapshot(109_000_000_000)


def test_multiple_detections_retract_experimental_identity_instead_of_choosing_primary():
    c = context()
    before = c.snapshot(110_000_000_000)
    assert before["episodes"]
    row = observation(0.75, kind="historical")
    row["persons"].append({"bbox": [60, 10, 10, 30], "confidence": 0.8, "track_id": "other"})
    row["person_count"] = 2
    c.add(row)
    result = c.snapshot(111_000_000_000)
    assert result["identity_status"] == "ambiguous_multiple_detections"
    assert result["episodes"] == []
    assert before["episodes"]  # previous revision remains auditable, not mutated


@pytest.mark.parametrize(
    "damage", ["session", "boot", "dimensions", "request", "window", "duplicate"]
)
def test_rejects_cross_context_or_conflicting_frames(damage):
    c = context()
    row = observation(0.75, kind="historical")
    if damage == "session":
        row["session_id"] = "other"
    elif damage == "boot":
        row["boot_id"] = "other"
    elif damage == "dimensions":
        row["source_width"] = 200
    elif damage == "request":
        row["buffer_request_id"] = "other"
    elif damage == "window":
        row = observation(1, kind="historical")
    else:
        row = observation(1)
        row["persons"][0]["bbox"] = [10, 10, 10, 30]
    with pytest.raises(ValueError):
        c.add(row)


def test_memory_and_post_window_are_bounded_and_unknown_is_explicit():
    c = context()
    assert not c.add(observation(10))
    assert c.snapshot(111_000_000_000)["frame_count"] == 7
    c.maximum_frames = 7
    assert c.add(observation(0.75, kind="historical"))
    assert not c.add(observation(0.75, kind="historical"))
    assert c.snapshot(112_000_000_000)["identity_status"] == "capacity_exceeded"
    missing = observation(1, primary_track_id=None)
    assert (
        EventContext(request(), missing).snapshot(102_000_000_000)["identity_status"]
        == "anchor_unavailable"
    )


def test_anchor_requires_a_confirmed_rebuilt_track():
    c = EventContext(request(), observation(1))
    assert c.snapshot(102_000_000_000)["identity_status"] == "anchor_unconfirmed_in_context"


def test_replay_publishes_capacity_failure_before_history_completion(monkeypatch):
    monkeypatch.setattr(EventContext, "maximum_frames", 3)
    rows = live_rows()[:3] + [request()] + live_rows()[3:4]
    report = replay_event_contexts(rows)
    assert all(s["final"]["identity_status"] == "capacity_exceeded" for s in report["summaries"])
    assert not any(s["history_transfer_complete"] for s in report["summaries"])


def test_replay_future_append_preserves_all_previous_revisions():
    rows = live_rows()
    prefix = rows[:3] + [request()] + rows[3:]
    delayed = observation(0.75, kind="historical", local_mono_ns=111_000_000_000)
    a = replay_event_contexts(prefix)
    b = replay_event_contexts(prefix + [delayed])

    def stable(s):
        return {k: v for k, v in s.items() if k != "replay_processing_ms"}

    assert [stable(s) for s in a["snapshots"]] == [
        stable(s) for s in b["snapshots"][: len(a["snapshots"])]
    ]
    assert all(s["historical_frames"] == 0 for s in a["snapshots"])


@pytest.mark.parametrize(
    "damage", ["future_anchor", "reverse_clock", "unsolicited", "bad_completion"]
)
def test_replay_rejects_noncausal_or_incomplete_provenance(damage):
    rows = live_rows()[:3] + [request()]
    if damage == "future_anchor":
        rows = [request()] + live_rows()[:3]
    elif damage == "reverse_clock":
        rows[-1]["local_mono_ns"] = 100_000_000_000
    elif damage == "unsolicited":
        rows = [observation(0.75, kind="historical")]
    else:
        rows.append(
            {
                "event": "buffer_processed",
                "buffer_request_id": "request",
                "frames_processed": 10,
                "local_mono_ns": 111_000_000_000,
            }
        )
    with pytest.raises(ValueError):
        replay_event_contexts(rows)


def test_completion_uses_only_already_observed_history():
    rows = live_rows()[:3] + [
        request(),
        observation(0.75, kind="historical", local_mono_ns=111_000_000_000),
        {
            "event": "buffer_processed",
            "buffer_request_id": "request",
            "frames_processed": 1,
            "local_mono_ns": 112_000_000_000,
        },
    ]
    result = replay_event_contexts(rows)
    assert all(s["history_transfer_complete"] for s in result["summaries"])
    assert not result["snapshots"][0]["history_transfer_complete"]
