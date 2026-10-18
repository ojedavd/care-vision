import numpy as np
import pytest

from care_vision.cloud.pose_evidence import assess_pose_support, pose_crop_bbox
from care_vision.cloud.rtmpose import RTMPoseEstimator


def body():
    points = np.zeros((17, 3))
    for left, right, y in [(5, 6, 100), (11, 12, 200), (13, 14, 300), (15, 16, 400)]:
        points[left] = [200, y, .8]
        points[right] = [230, y, .8]
    return points


def test_support_is_not_visibility_or_recovery_proof():
    result = assess_pose_support(body(), [180, 50, 100, 380], (500, 600))
    assert result["body_support_usable"]
    assert not result["physical_visibility_verified"]
    assert result["left_knee_angle_degrees"] == pytest.approx(180)


@pytest.mark.parametrize("failure", ["low_score", "outside", "degenerate_leg", "degenerate_torso"])
def test_missing_or_degenerate_body_evidence_is_unknown(failure):
    points = body()
    if failure == "low_score":
        points[15, 2] = .29
    elif failure == "outside":
        points[15, 0] = -1
    elif failure == "degenerate_leg":
        points[15] = points[13]
    else:
        points[5] = points[11]
        points[6] = points[12]
    result = assess_pose_support(points, [180, 50, 100, 380], (500, 600))
    assert not result["body_support_usable"]
    assert result["body_support_status"] == "INSUFFICIENT_KEYPOINT_SUPPORT"


def test_crop_with_other_detection_is_explicitly_ambiguous():
    result = assess_pose_support(body(), [180, 50, 100, 380], (500, 600), [[200, 100, 30, 60]])
    assert result["torso_supported"] and result["legs_supported"]
    assert not result["body_support_usable"]
    assert result["body_support_status"] == "AMBIGUOUS_CROP"


@pytest.mark.parametrize("box", [[10, 10, 60, 400], [10, 10, 400, 60]])
def test_diagnostic_crop_matches_existing_model_preprocessing(box):
    _, center, scale = RTMPoseEstimator.preprocess(np.zeros((500, 600, 3), np.uint8), box)
    expected = [*(center - scale / 2), *scale]
    assert pose_crop_bbox(box) == pytest.approx(expected)


def test_missing_pose_and_invalid_inputs():
    result = assess_pose_support(None, [0, 0, 100, 200], (500, 600))
    assert result["body_support_status"] == "NO_POSE"
    with pytest.raises(ValueError):
        assess_pose_support(np.zeros((3, 3)), [0, 0, 100, 200], (500, 600))
    with pytest.raises(ValueError):
        pose_crop_bbox([0, 0, 0, 200])
