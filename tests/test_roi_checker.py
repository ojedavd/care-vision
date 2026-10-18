import copy

import pytest

from care_vision.check_loop import CheckFailed
from care_vision.check_roi import check_roi_records


def roi_records():
    session = "session-1"
    roi_id, full_id = "roi-1", "full-1"
    bbox = [100, 30, 80, 200]
    observations = [
        {
            "event": "observation",
            "observation_id": "trigger",
            "frame_id": "f0",
            "session_id": session,
            "valid": True,
            "evidence_kind": "live",
            "view_kind": "full_frame",
            "source_width": 640,
            "source_height": 360,
            "confirmed_person_detected": True,
            "primary_tracked_bbox": bbox,
        },
        {
            "event": "observation",
            "observation_id": "return-trigger",
            "frame_id": "r0",
            "session_id": session,
            "valid": True,
            "evidence_kind": "live",
            "view_kind": "roi",
            "source_width": 640,
            "source_height": 360,
            "source_roi": bbox,
            "width": 640,
            "height": 360,
            "visual_policy_seq": 3,
            "applied_visual_command_id": roi_id,
        },
    ]
    edge = [{"event": "connected", "session_id": session}]
    cloud = [
        {
            "event": "agent_action",
            "type": "REQUEST_ROI",
            "session_id": session,
            "command_id": roi_id,
            "command_seq": 3,
            "caused_by_observation_id": "trigger",
            "based_on_frame_id": "f0",
            "source_width": 640,
            "source_height": 360,
            "roi_x": 100,
            "roi_y": 30,
            "roi_width": 80,
            "roi_height": 200,
            "output_width": 640,
            "output_height": 360,
            "local_mono_ns": 10,
        },
        {
            "event": "agent_action",
            "type": "REQUEST_FULL_FRAME",
            "session_id": session,
            "command_id": full_id,
            "command_seq": 5,
            "caused_by_observation_id": "return-trigger",
            "based_on_frame_id": "r0",
        },
    ]
    for command_id, seq, view in ((roi_id, 3, "roi"), (full_id, 5, "full_frame")):
        ack = {
            "command_id": command_id,
            "result": "applied",
            "active_view_kind": view,
            "active_visual_policy_seq": seq,
            "applied_visual_command_id": command_id,
            "effective_from_capture_seq": 1,
        }
        edge.append({"event": "command_ack", **ack})
        cloud.append({"event": "ack_received", "local_mono_ns": 20, **ack})
    for index in range(10):
        frame = {
            "frame_id": f"r{index}",
            "capture_seq": index + 1,
            "session_id": session,
            "evidence_kind": "live",
            "view_kind": "roi",
            "source_roi": bbox,
            "roi_x": bbox[0],
            "roi_y": bbox[1],
            "roi_width": bbox[2],
            "roi_height": bbox[3],
            "width": 640,
            "height": 360,
            "visual_policy_seq": 3,
            "applied_visual_command_id": roi_id,
        }
        edge.append({"event": "frame_sent", **frame})
        cloud.append({"event": "frame_received", **frame})
        if index:
            observations.append({"event": "observation", "observation_id": f"ro{index}", **frame})
    for index in range(10):
        observations.append(
            {
                "event": "observation",
                "observation_id": f"fo{index}",
                "frame_id": f"ff{index}",
                "session_id": session,
                "valid": True,
                "view_kind": "full_frame",
                "source_roi": None,
                "visual_policy_seq": 5,
                "applied_visual_command_id": full_id,
            }
        )
    cloud.extend(observations)
    return edge, cloud


def test_roi_checker_verifies_complete_visual_loop():
    edge, cloud = roi_records()
    report = check_roi_records(edge, cloud)
    assert report["verified"]
    assert report["roi_frames_sent"] == 10
    assert report["restored_full_frame_observations"] == 10


def test_roi_checker_rejects_incomplete_history_and_metadata_mismatch():
    edge, cloud = roi_records()
    broken = copy.deepcopy(cloud)
    broken.append({"event": "buffer_incomplete"})
    with pytest.raises(CheckFailed, match="incomplete history"):
        check_roi_records(edge, broken)
    broken = copy.deepcopy(cloud)
    next(record for record in broken if record.get("event") == "frame_received")["roi_x"] = 0
    with pytest.raises(CheckFailed, match="metadata"):
        check_roi_records(edge, broken)
