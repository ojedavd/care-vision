import cv2
import numpy as np
import pytest
from conftest import frame

from care_vision.cloud.nanodet import NanoDetPersonDetector
from care_vision.cloud.person import MAX_PERSONS, PersonDetection, observe_people
from care_vision.cloud.server import Session
from care_vision.cloud.vision import MotionVision
from care_vision.common.metrics import EventLog


class FixturePersonDetector:
    backend = "fixture-bright-region"

    def detect(self, image):
        mask = cv2.inRange(image, (180, 180, 180), (255, 255, 255))
        points = cv2.findNonZero(mask)
        if points is None:
            return []
        return [PersonDetection(cv2.boundingRect(points), 0.95)]


def test_observation_exposes_person_and_tentative_tracking_contract(identity):
    header, jpeg = frame(identity, x=40)
    observation = MotionVision(person_detector=FixturePersonDetector()).observe(header, jpeg)

    assert observation["observation_schema_version"] == 1
    assert observation["person_detection_status"] == "ok"
    assert observation["person_detector_backend"] == "fixture-bright-region"
    assert observation["person_detected"] is True
    assert observation["person_count"] == 1
    assert observation["primary_person_bbox"] == observation["persons"][0]["bbox"]
    assert observation["primary_person_confidence"] == 0.95
    person = observation["persons"][0]
    assert person["detection_id"] == f"{header['frame_id']}:person:0"
    assert person["label"] == "person"
    x, y, width, height = person["bbox"]
    assert 38 <= x <= 42 and 88 <= y <= 92
    assert 179 <= width <= 183 and 179 <= height <= 183
    assert person["track_id"] == f"live:{identity['session_id']}:track:1"
    assert person["track_state"] == "tentative"
    assert observation["confirmed_person_detected"] is False
    assert observation["primary_track_id"] is None


def test_unconfigured_detector_is_distinct_from_configured_no_person(identity):
    header, jpeg = frame(identity)
    unavailable = MotionVision().observe(header, jpeg)

    class EmptyDetector:
        backend = "empty-test-detector"

        def detect(self, image):
            return []

    empty = MotionVision(person_detector=EmptyDetector()).observe(header, jpeg)
    assert unavailable["person_detection_status"] == "not_configured"
    assert unavailable["person_detected"] is False
    assert empty["person_detection_status"] == "ok"
    assert empty["person_detected"] is False
    assert empty["person_count"] == 0


def test_detections_are_bounded_validated_and_sorted(identity):
    header, jpeg = frame(identity)
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    class OrderedDetector:
        backend = "ordered-test-detector"

        def detect(self, image):
            return [PersonDetection((20, 20, 10, 20), 0.4), PersonDetection((30, 30, 20, 40), 0.9)]

    result = observe_people(OrderedDetector(), image, header["frame_id"])
    assert [person["confidence"] for person in result["persons"]] == [0.9, 0.4]
    assert result["primary_person_bbox"] == [30, 30, 20, 40]

    class InvalidDetector:
        backend = "invalid-test-detector"

        def detect(self, image):
            return [PersonDetection((630, 350, 20, 20), 0.8)]

    error = observe_people(InvalidDetector(), image, header["frame_id"])
    assert error["person_detection_status"] == "error"
    assert "outside the frame" in error["person_detection_error"]
    assert error["persons"] == []


def test_detector_output_count_is_bounded(identity):
    header, jpeg = frame(identity)
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    class TooManyDetector:
        backend = "too-many-test-detector"

        def detect(self, image):
            return [PersonDetection((0, 0, 1, 1), 0.5)] * (MAX_PERSONS + 1)

    result = observe_people(TooManyDetector(), image, header["frame_id"])
    assert result["person_detection_status"] == "error"
    assert result["person_count"] == 0


def test_invalid_image_does_not_call_person_detector(identity):
    class MustNotRun:
        backend = "must-not-run"

        def detect(self, image):
            raise AssertionError("detector must not run for an invalid image")

    header, _ = frame(identity)
    observation = MotionVision(person_detector=MustNotRun()).observe(header, b"corrupt")
    assert observation["invalid_reason"] == "invalid_image"
    assert observation["person_detection_status"] == "invalid_image"


def test_live_and_historical_timelines_use_separate_detector_instances(tmp_path, identity):
    instances = []

    class StatefulDetector:
        backend = "stateful-test-detector"

        def detect(self, image):
            return []

    def factory():
        detector = StatefulDetector()
        instances.append(detector)
        return detector

    log = EventLog("cloud", tmp_path / "cloud.jsonl")
    try:
        session = Session(object(), identity, log, person_detector_factory=factory)
        session.history.start({"command_id": "buffer-1"})

        assert len(instances) == 2
        assert session.vision.person_detector is instances[0]
        assert session.history.vision.person_detector is instances[1]
    finally:
        log.close()


@pytest.mark.parametrize("grouped_outputs", [False, True])
def test_nanodet_adapter_decodes_and_unletterboxes_person(tmp_path, monkeypatch, grouped_outputs):
    class FakeNet:
        def setPreferableBackend(self, value):
            pass

        def setPreferableTarget(self, value):
            pass

        def setInput(self, value):
            assert value.shape == (1, 3, 416, 416)

        def getUnconnectedOutLayersNames(self):
            return [f"output-{index}" for index in range(8)]

        def forward(self, names):
            score_outputs = []
            box_outputs = []
            for stride in (8, 16, 32, 64):
                side = 416 // stride
                count = side * side
                scores = np.zeros((1, count, 80), dtype=np.float32)
                logits = np.full((1, count, 32), -10, dtype=np.float32)
                logits[:, :, 3::8] = 10
                if stride == 8:
                    scores[0, (side // 2) * side + (side // 2), 0] = 0.9
                score_outputs.append(scores)
                box_outputs.append(logits)
            if grouped_outputs:
                return score_outputs + box_outputs
            return [item for pair in zip(score_outputs, box_outputs) for item in pair]

    model = tmp_path / "nanodet.onnx"
    model.write_bytes(b"test model")
    monkeypatch.setattr(cv2.dnn, "readNet", lambda path: FakeNet())
    detector = NanoDetPersonDetector(model)
    detections = detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))

    assert detector.backend.startswith("opencv-dnn-nanodet-2022nov:")
    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.9)
    x, y, width, height = detections[0].bbox
    assert 280 <= x <= 300
    assert 140 <= y <= 160
    assert 65 <= width <= 85
    assert 65 <= height <= 85


@pytest.mark.parametrize("score,nms", [(0, 0.6), (1.1, 0.6), (0.35, 0)])
def test_nanodet_adapter_rejects_invalid_thresholds(tmp_path, score, nms):
    model = tmp_path / "nanodet.onnx"
    model.write_bytes(b"test model")
    with pytest.raises(ValueError, match="thresholds"):
        NanoDetPersonDetector(model, score_threshold=score, nms_threshold=nms)
