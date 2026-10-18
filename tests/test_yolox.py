import cv2
import numpy as np
import pytest

from care_vision.cloud.yolox import YoloXPersonDetector


def test_yolox_adapter_decodes_person_and_ignores_other_classes(tmp_path, monkeypatch):
    class FakeNet:
        def setPreferableBackend(self, value):
            pass

        def setPreferableTarget(self, value):
            pass

        def setInput(self, value):
            assert value.shape == (1, 3, 640, 640)

        def getUnconnectedOutLayersNames(self):
            return ["output"]

        def forward(self, names):
            output = np.zeros((1, 8400, 85), dtype=np.float32)
            output[0, 810, 4] = 0.9
            output[0, 810, 5] = 0.8
            output[0, 811, 4] = 0.95
            output[0, 811, 6] = 0.9
            return [output]

    model = tmp_path / "yolox.onnx"
    model.write_bytes(b"test model")
    monkeypatch.setattr(cv2.dnn, "readNet", lambda path: FakeNet())
    detector = YoloXPersonDetector(model)
    detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    assert detector.grids.shape == (1, 8400, 2)
    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.72)
    assert detections[0].bbox == (76, 76, 8, 8)


def test_yolox_adapter_supports_official_416_models(tmp_path, monkeypatch):
    class FakeNet:
        def setPreferableBackend(self, value):
            pass

        def setPreferableTarget(self, value):
            pass

        def setInput(self, value):
            assert value.shape == (1, 3, 416, 416)

        def getUnconnectedOutLayersNames(self):
            return ["output"]

        def forward(self, names):
            return [np.zeros((1, 3549, 85), dtype=np.float32)]

    model = tmp_path / "yolox_tiny.onnx"
    model.write_bytes(b"test model")
    monkeypatch.setattr(cv2.dnn, "readNet", lambda path: FakeNet())
    detector = YoloXPersonDetector(model, input_size=416)

    assert detector.detect(np.zeros((1080, 1920, 3), dtype=np.uint8)) == []
    assert detector.grids.shape == (1, 3549, 2)
    assert detector.backend.startswith("opencv-dnn-yolox-416x416:")


@pytest.mark.parametrize("score,nms", [(0, 0.5), (1.1, 0.5), (0.35, 0)])
def test_yolox_adapter_rejects_invalid_thresholds(tmp_path, score, nms):
    model = tmp_path / "yolox.onnx"
    model.write_bytes(b"test model")
    with pytest.raises(ValueError, match="thresholds"):
        YoloXPersonDetector(model, score_threshold=score, nms_threshold=nms)


@pytest.mark.parametrize("input_size", [0, 415, 416.0])
def test_yolox_adapter_rejects_invalid_input_size(tmp_path, input_size):
    model = tmp_path / "yolox.onnx"
    model.write_bytes(b"test model")
    with pytest.raises(ValueError, match="input_size"):
        YoloXPersonDetector(model, input_size=input_size)
