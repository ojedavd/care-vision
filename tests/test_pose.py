import numpy as np
import pytest

from care_vision.cloud.pose_signals import PoseTrajectoryAnalyzer, pose_scoring_row
from care_vision.cloud.rtmpose import RTMPoseEstimator


def points(shoulder=(200, 100), hip=(200, 200), score=0.8):
    result = np.zeros((17, 3), dtype=np.float64)
    for index in (5, 6):
        result[index] = [*shoulder, score]
    for index in (11, 12):
        result[index] = [*hip, score]
    return result


def test_affine_crop_centers_bbox_preserves_aspect_and_uses_sdk_rgb_order():
    image = np.full((500, 600, 3), [10, 20, 30], dtype=np.uint8)
    blob, center, scale = RTMPoseEstimator.preprocess(image, [100, 100, 100, 200])
    assert blob.shape == (1, 3, 256, 192)
    assert blob.dtype == np.float32
    assert center == pytest.approx([150, 200])
    assert scale == pytest.approx([187.5, 250])
    assert blob[0, :, 128, 96] == pytest.approx(
        (np.array([30, 20, 10]) - RTMPoseEstimator.mean) / RTMPoseEstimator.std
    )


def test_simcc_decoding_restores_coordinates_and_preserves_uncalibrated_scores():
    x, y = np.zeros((1, 17, 384)), np.zeros((1, 17, 512))
    x[:, :, 192], y[:, :, 256] = 1.7, 1.4
    decoded = RTMPoseEstimator.decode([y, x], np.array([150, 200]), np.array([187.5, 250]))
    assert decoded == pytest.approx(np.tile([150, 200, 1.4], (17, 1)))
    x[:, :, 192] = 0
    x[:, :, 0] = 0.8
    decoded = RTMPoseEstimator.decode([x, y], np.array([150, 200]), np.array([187.5, 250]))
    assert decoded[0] == pytest.approx([56.25, 200, 0.8])


@pytest.mark.parametrize("bbox", [[1, 2, 0, 3], [0, 0, -2, 3], [0, 0, float("nan"), 3]])
def test_invalid_pose_crop_is_rejected(bbox):
    with pytest.raises(ValueError):
        RTMPoseEstimator.preprocess(np.zeros((100, 100, 3), np.uint8), bbox)


def test_nonfinite_or_wrong_model_outputs_are_rejected():
    with pytest.raises(ValueError):
        RTMPoseEstimator.decode([np.zeros((1, 17, 1))], [0, 0], [1, 1])
    with pytest.raises(ValueError):
        RTMPoseEstimator.decode(
            [np.full((1, 17, 384), np.nan), np.zeros((1, 17, 512))], [0, 0], [1, 1]
        )


def test_pose_trajectory_uses_source_time_and_reports_direction():
    analyzer = PoseTrajectoryAnalyzer()
    first = analyzer.update(points(), "T1", 0, (500, 600), "live")
    second = analyzer.update(points((200, 200), (200, 300)), "T1", 500_000_000, (500, 600), "live")
    assert first["pose_orientation"] == "vertical"
    assert first["pose_motion_direction"] == "unknown"
    assert second["pose_vertical_velocity_per_s"] == pytest.approx(0.4)
    assert second["pose_motion_direction"] == "downward"


def test_pose_rotation_is_separate_from_upward_translation():
    analyzer = PoseTrajectoryAnalyzer()
    analyzer.update(points((300, 200), (200, 200)), "T1", 0, (500, 600), "live")
    current = analyzer.update(points((200, 150), (200, 250)), "T1", 500_000_000, (500, 600), "live")
    assert current["pose_torso_rotation_degrees_per_s"] == -180
    assert current["pose_straightening"]
    assert current["pose_motion_direction"] == "stable"


@pytest.mark.parametrize(
    "interruption", ["missing", "low_confidence", "new_track", "namespace", "gap"]
)
def test_pose_does_not_bridge_unobserved_or_unrelated_people(interruption):
    analyzer = PoseTrajectoryAnalyzer()
    analyzer.update(points(), "T1", 0, (500, 600), "live")
    track, namespace, at = "T1", "live", 500_000_000
    if interruption == "missing":
        analyzer.update(None, "T1", 100_000_000, (500, 600), "live")
    elif interruption == "low_confidence":
        missing = analyzer.update(points(score=0.2), "T1", 100_000_000, (500, 600), "live")
        assert missing["pose_orientation"] is None
    elif interruption == "new_track":
        track = "T2"
    elif interruption == "namespace":
        namespace = "historical"
    else:
        at = 1_000_000_000
    result = analyzer.update(points(), track, at, (500, 600), namespace)
    assert result["pose_motion_direction"] == "unknown"
    assert result["pose_straightening"] is None


def test_out_of_view_or_degenerate_torso_never_becomes_a_valid_posture():
    analyzer = PoseTrajectoryAnalyzer()
    assert (
        analyzer.update(points((-5, 10)), "T1", 0, (500, 600), "live")["pose_status"]
        == "torso_outside_image"
    )
    assert (
        analyzer.update(points((200, 200)), "T1", 1, (500, 600), "live")["pose_status"]
        == "degenerate_torso"
    )
    with pytest.raises(ValueError, match="increase"):
        analyzer.update(points(), "T1", 1, (500, 600), "live")


def test_scoring_keeps_unavailable_pose_unknown():
    row = {
        **PoseTrajectoryAnalyzer.empty("low_torso_confidence"),
        "frame_index": 10,
        "phase": "recovering",
    }
    projected = pose_scoring_row(row, straightening=True)
    assert projected["primary_motion_direction"] == "unknown"
    assert projected["primary_orientation"] is None
    assert projected["primary_near_floor"] is None
