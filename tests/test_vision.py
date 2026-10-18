import pytest
from conftest import frame

from care_vision.cloud.vision import MotionVision


@pytest.mark.parametrize("fps", [2, 20])
def test_motion_uses_fixed_temporal_interval(identity, fps):
    vision = MotionVision()
    observations = []
    for i in range(fps * 2):
        header, jpeg = frame(identity, i + 1, round((1 + i / fps) * 1e9), x=20 + int(i / fps * 140))
        observations.append(vision.observe(header, jpeg))
    valid = [o for o in observations if o["valid"]]
    assert valid
    assert all(400 <= o["comparison_interval_ms"] <= 800 for o in valid)
    assert all(o["motion_score"] > 0.035 for o in valid)


def test_bad_frame_and_gap_do_not_become_valid_quiet(identity):
    vision = MotionVision()
    header, jpeg = frame(identity)
    assert not vision.observe(header, jpeg)["valid"]


def test_observation_contains_posture_trajectory_contract(identity):
    vision = MotionVision()
    header, jpeg = frame(identity)
    observed = vision.observe(header, jpeg)
    assert observed["posture_trajectory_status"] == "no_confirmed_person"
    assert observed["primary_source_bbox"] is None
    assert observed["primary_motion_direction"] == "unknown"
    header, jpeg = frame(identity, 2, 1_500_000_000)
    assert vision.observe(header, jpeg)["motion_score"] == 0
    assert vision.observe(header, b"corrupt")["invalid_reason"] == "invalid_image"
    header, jpeg = frame(identity, 3, 5_000_000_000)
    assert not vision.observe(header, jpeg)["valid"]


def test_experimental_weak_association_preserves_existing_track_only(identity):
    from care_vision.cloud.person import PersonDetection

    class Detector:
        backend = "test"
        confidence = 0.9

        def detect(self, image):
            return [PersonDetection((10, 10, 100, 200), self.confidence)]

    detector = Detector()
    vision = MotionVision(person_detector=detector, experimental_weak_association=True)
    for seq in (1, 2):
        header, jpeg = frame(identity, seq, int(seq * 0.5e9))
        original = vision.observe(header, jpeg)
    detector.confidence = 0.25
    header, jpeg = frame(identity, 3, 1_500_000_000)
    weak = vision.observe(header, jpeg)
    assert weak["primary_track_id"] == original["primary_track_id"]
    assert weak["persons"][0]["association_only"]
    # A fresh session cannot manufacture identity from the weak candidate.
    header, jpeg = frame(dict(identity, session_id="new"), 4, 2_000_000_000)
    new = vision.observe(header, jpeg)
    assert not new["confirmed_person_detected"]
    assert new["primary_person_bbox"] is None
