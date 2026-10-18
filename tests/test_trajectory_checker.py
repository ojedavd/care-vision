import copy

import pytest

from care_vision.check_loop import CheckFailed
from care_vision.check_trajectory import check_trajectory_records


def trajectory_records():
    records = []
    for index in range(20):
        records.append(
            {
                "event": "observation",
                "evidence_kind": "live",
                "view_kind": "full_frame" if index < 10 else "roi",
                "source_width": 1280,
                "source_height": 720,
                "posture_trajectory_status": "ok",
                "primary_source_bbox": [400, 80 + index, 320, 560],
                "primary_center_y_normalized": 0.5,
                "primary_bottom_y_normalized": 0.9,
                "primary_bbox_aspect_ratio": 0.57,
                "primary_orientation": "vertical",
                "primary_vertical_velocity_per_s": 0.2 if index == 5 else 0.0,
                "primary_motion_direction": "downward" if index == 5 else "stable",
                "primary_near_floor": True,
                "primary_time_near_floor_s": index / 20,
            }
        )
    return records


def test_trajectory_checker_verifies_views_and_signal_ranges():
    report = check_trajectory_records(trajectory_records())
    assert report["verified"]
    assert report["valid_full_frame_observations"] == 10
    assert report["valid_roi_observations"] == 10
    assert not report["fall_classification_enabled"]


def test_trajectory_checker_rejects_missing_field_and_invalid_source_bbox():
    missing = copy.deepcopy(trajectory_records())
    missing[0].pop("primary_motion_direction")
    with pytest.raises(CheckFailed, match="lacks"):
        check_trajectory_records(missing)
    invalid = copy.deepcopy(trajectory_records())
    invalid[0]["primary_source_bbox"] = [1200, 80, 320, 560]
    with pytest.raises(CheckFailed, match="bbox"):
        check_trajectory_records(invalid)
