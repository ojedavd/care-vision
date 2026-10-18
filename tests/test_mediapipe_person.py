import cv2
import numpy as np
import pytest

from care_vision.cloud.mediapipe_person import MediaPipePersonDetector


def test_mediapipe_person_adapter_decodes_outputs_independent_of_order(tmp_path, monkeypatch):
    class FakeNet:
        def setPreferableBackend(self, value):
            pass

        def setPreferableTarget(self, value):
            pass

        def setInput(self, value):
            assert value.shape == (1, 3, 224, 224)

        def getUnconnectedOutLayersNames(self):
            return ["scores", "boxes"]

        def forward(self, names):
            boxes = np.zeros((1, 2254, 12), dtype=np.float32)
            scores = np.full((1, 2254, 1), -100, dtype=np.float32)
            boxes[0, 0, :4] = [0, 0, 50, 100]
            scores[0, 0, 0] = 2
            return [scores, boxes]

    model = tmp_path / "mediapipe.onnx"
    model.write_bytes(b"test model")
    monkeypatch.setattr(cv2.dnn, "readNet", lambda path: FakeNet())
    detector = MediaPipePersonDetector(model)
    detections = detector.detect(np.zeros((224, 224, 3), dtype=np.uint8))

    assert detector.anchors.shape == (2254, 2)
    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(1 / (1 + np.exp(-2)))
    assert detections[0].bbox == (0, 0, 29, 54)


def test_mediapipe_person_padding_uses_normalized_zero(tmp_path, monkeypatch):
    class FakeNet:
        def setPreferableBackend(self, value):
            pass

        def setPreferableTarget(self, value):
            pass

    model = tmp_path / "mediapipe.onnx"
    model.write_bytes(b"test model")
    monkeypatch.setattr(cv2.dnn, "readNet", lambda path: FakeNet())
    detector = MediaPipePersonDetector(model)

    blob, pad_x, pad_y = detector._preprocess(np.full((200, 100, 3), 255, dtype=np.uint8))

    assert pad_x == 49
    assert pad_y == 0
    assert np.all(blob[:, :, :, :56] == 0)
    assert np.all(blob[:, :, :, 56:168] == 1)


@pytest.mark.parametrize("score,nms", [(0, 0.3), (1.1, 0.3), (0.3, 0)])
def test_mediapipe_person_adapter_rejects_invalid_thresholds(tmp_path, score, nms):
    model = tmp_path / "mediapipe.onnx"
    model.write_bytes(b"test model")
    with pytest.raises(ValueError, match="thresholds"):
        MediaPipePersonDetector(model, score_threshold=score, nms_threshold=nms)
