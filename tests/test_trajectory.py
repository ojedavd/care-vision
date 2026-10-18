import pytest

from care_vision.cloud.trajectory import PostureTrajectoryAnalyzer, map_bbox_to_source


def header(seq, at_ns, **changes):
    return {
        "frame_id": f"F{seq}",
        "captured_mono_ns": at_ns,
        "width": 320,
        "height": 180,
        "source_width": 1280,
        "source_height": 720,
        "view_kind": "full_frame",
        **changes,
    }


def person(track_id="T1", bbox=(100, 20, 80, 140)):
    return {"primary_track_id": track_id, "primary_tracked_bbox": list(bbox)}


def test_full_frame_and_roi_boxes_map_to_stable_source_coordinates():
    assert map_bbox_to_source([100, 20, 80, 140], header(1, 1)) == [400, 80, 320, 560]
    roi_header = header(
        2,
        2,
        width=640,
        height=360,
        view_kind="roi",
        roi_x=400,
        roi_y=80,
        roi_width=320,
        roi_height=560,
    )
    # The 320x560 crop is letterboxed horizontally into 640x360.
    # Edge rounds the crop to 206x360 and places it at integer x=217.
    mapped = map_bbox_to_source([217, 0, 206, 360], roi_header)
    assert mapped == pytest.approx([400, 80, 320, 560], abs=0.01)


def test_trajectory_reports_downward_motion_orientation_and_floor_duration():
    analyzer = PostureTrajectoryAnalyzer()
    first = analyzer.update(person(bbox=(100, 0, 80, 100)), header(1, 1_000_000_000), "live:s")
    second = analyzer.update(person(bbox=(100, 60, 140, 100)), header(2, 1_500_000_000), "live:s")
    third = analyzer.update(person(bbox=(100, 60, 140, 100)), header(3, 2_000_000_000), "live:s")
    assert first["posture_trajectory_status"] == "insufficient_temporal_context"
    assert second["posture_trajectory_status"] == "ok"
    assert second["primary_motion_direction"] == "downward"
    assert second["primary_orientation"] == "horizontal"
    assert second["primary_near_floor"]
    assert third["primary_time_near_floor_s"] == 0.5


def test_missing_person_and_namespace_change_reset_temporal_context():
    analyzer = PostureTrajectoryAnalyzer()
    analyzer.update(person(), header(1, 1_000_000_000), "live:s1")
    assert (
        analyzer.update({}, header(2, 1_500_000_000), "live:s1")["posture_trajectory_status"]
        == "no_confirmed_person"
    )
    reset = analyzer.update(person(), header(3, 2_000_000_000), "historical:b1")
    assert reset["posture_trajectory_status"] == "insufficient_temporal_context"
