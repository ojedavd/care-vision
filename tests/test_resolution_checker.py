import copy

import pytest

from care_vision.check_loop import CheckFailed
from care_vision.check_resolution import check_resolution_records


def resolution_records():
    roi = {
        "event": "agent_action",
        "type": "REQUEST_ROI",
        "source_width": 1280,
        "source_height": 720,
        "roi_width": 400,
        "output_width": 640,
        "output_height": 360,
    }
    full = {
        "event": "agent_action",
        "type": "REQUEST_FULL_FRAME",
        "output_width": 320,
        "output_height": 180,
    }
    edge = [
        {"event": "history_transmission_deferred", "local_mono_ns": 10},
        {"event": "history_transmission_resumed", "local_mono_ns": 20},
    ]
    for index in range(10):
        edge.append(
            {
                "event": "frame_sent",
                "frame_id": f"b{index}",
                "evidence_kind": "live",
                "view_kind": "full_frame",
                "source_width": 1280,
                "source_height": 720,
                "width": 320,
                "height": 180,
                "payload_bytes": 5_000,
            }
        )
        edge.append(
            {
                "event": "frame_sent",
                "frame_id": f"r{index}",
                "evidence_kind": "live",
                "view_kind": "roi",
                "source_width": 1280,
                "source_height": 720,
                "width": 640,
                "height": 360,
                "payload_bytes": 20_000,
            }
        )
        edge.append(
            {
                "event": "frame_sent",
                "frame_id": f"h{index}",
                "evidence_kind": "historical",
                "width": 1280,
                "height": 720,
                "payload_bytes": 50_000,
                "local_mono_ns": 30 + index,
            }
        )
    return edge, [roi, full]


def test_resolution_checker_measures_reduction_and_sampling_gain():
    edge, cloud = resolution_records()
    report = check_resolution_records(edge, cloud)
    assert report["verified"]
    assert report["baseline_payload_reduction_percent"] == 90
    assert report["roi_sampling_gain"] == 6.4


def test_resolution_checker_rejects_low_resolution_history():
    edge, cloud = resolution_records()
    broken = copy.deepcopy(edge)
    next(record for record in broken if record.get("evidence_kind") == "historical")["width"] = 640
    with pytest.raises(CheckFailed, match="capture resolution"):
        check_resolution_records(broken, cloud)


def test_resolution_accepts_concurrent_history_but_rejects_wrong_roi_size():
    edge, cloud = resolution_records()
    edge = [r for r in edge if r["event"] == "frame_sent"]
    assert check_resolution_records(edge, cloud)["history_deferred_ms"] is None
    next(r for r in edge if r.get("view_kind") == "roi")["width"] = 320
    with pytest.raises(CheckFailed, match="ROI frames"):
        check_resolution_records(edge, cloud)
