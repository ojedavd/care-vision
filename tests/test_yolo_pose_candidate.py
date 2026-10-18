import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def tool(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / f"tools/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = tool("yolo_pose_candidate")
comparison = tool("compare_pose_cache")


def predictions(boxes):
    output = np.zeros((1, 56, 8400), np.float32)
    for index, box in enumerate(boxes):
        output[0, :4, index] = box
        output[0, 4, index] = 0.9
        output[0, 5:, index] = np.tile([192, 64, 0.7], 17)
    return output


def test_letterbox_preserves_crop_aspect_rgb_and_scale():
    crop = np.full((200, 100, 3), [10, 20, 30], np.uint8)
    blob, transform = candidate.letterbox(crop)
    assert transform == [3.2, 3.2, 160, 0]
    assert blob.shape == (1, 3, 640, 640)
    assert blob[0, :, 200, 320] == pytest.approx(np.array([30, 20, 10]) / 255)
    assert blob[0, :, 200, 0] == pytest.approx([114 / 255] * 3)


def test_decoding_and_restoration_keep_joint_coordinates_and_response():
    output = predictions([[272, 224, 160, 320]])
    points, metadata = candidate.decode(output, [3.2, 3.2, 160, 0], [10, 20, 50, 100])
    assert metadata["status"] == "ok"
    assert json.loads(json.dumps(metadata, allow_nan=False))["status"] == "ok"
    assert points[0] == pytest.approx([10, 20, 0.7])
    restored = candidate.restore_points(points, [100, 50, 200, 400], (200, 100, 3))
    assert restored[0] == pytest.approx([120, 90, 0.7])


def test_model_detection_of_other_person_is_not_silently_selected():
    points, metadata = candidate.decode(
        predictions([[500, 500, 100, 100]]), [1, 1, 0, 0], [100, 100, 100, 200]
    )
    assert points is None
    assert metadata["status"] == "no_corresponding_pose"


def test_ambiguous_pose_candidates_remain_unknown():
    points, metadata = candidate.decode(
        predictions([[125, 200, 100, 200], [175, 200, 100, 200]]),
        [1, 1, 0, 0],
        [100, 100, 100, 200],
    )
    assert points is None
    assert metadata["status"] == "ambiguous_pose_selection"


def test_missing_pose_is_not_normal_activity():
    points, metadata = candidate.decode(predictions([]), [1, 1, 0, 0], [0, 0, 100, 200])
    assert points is None
    assert metadata["status"] == "no_pose_detection"
    assert "activity" not in metadata


@pytest.mark.parametrize("output", [np.zeros((1, 57, 8400)), np.full((1, 56, 8400), np.nan)])
def test_wrong_model_or_nonfinite_outputs_rejected(output):
    with pytest.raises(ValueError):
        candidate.decode(output, [1, 1, 0, 0], [0, 0, 100, 200])


def test_shared_crop_restores_center_without_second_bbox_padding():
    from care_vision.cloud.rtmpose import RTMPoseEstimator

    class Net:
        def setInput(self, blob):
            self.blob = blob

        def getUnconnectedOutLayersNames(self):
            return ["x", "y"]

        def forward(self, names):
            x, y = np.zeros((1, 17, 384)), np.zeros((1, 17, 512))
            x[:, :, 192] = 0.8
            y[:, :, 256] = 0.7
            return [x, y]

    estimator = RTMPoseEstimator.__new__(RTMPoseEstimator)
    estimator.net = Net()
    crop, box = candidate.common_crop(
        np.full((500, 600, 3), [10, 20, 30], np.uint8), [100, 100, 100, 200]
    )
    local = comparison.rtm_on_crop(estimator, crop)
    points = candidate.restore_points(local, box, crop.shape)
    assert points == pytest.approx(np.tile([150, 200, 0.7], (17, 1)))
    assert estimator.net.blob[0, :, 128, 96] == pytest.approx(
        (np.array([30, 20, 10]) - estimator.mean) / estimator.std
    )
