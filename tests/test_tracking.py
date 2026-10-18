import pytest

from care_vision.cloud.tracking import PersonTracker


def observation(*boxes):
    persons = [
        {
            "detection_id": f"frame:person:{index}",
            "label": "person",
            "confidence": 0.9 - index * 0.1,
            "bbox": list(bbox),
        }
        for index, bbox in enumerate(boxes)
    ]
    return {
        "person_detection_status": "ok",
        "person_detected": bool(persons),
        "person_count": len(persons),
        "persons": persons,
    }


def test_track_is_confirmed_after_two_associated_detections_and_bbox_is_smoothed():
    tracker = PersonTracker()
    first = tracker.update(observation((10, 10, 100, 200)), 1_000_000_000, "live:s1")
    second = tracker.update(observation((20, 10, 100, 200)), 1_050_000_000, "live:s1")

    assert first["persons"][0]["track_state"] == "tentative"
    assert first["confirmed_person_detected"] is False
    assert second["persons"][0]["track_id"] == first["persons"][0]["track_id"]
    assert second["persons"][0]["track_state"] == "confirmed"
    assert second["persons"][0]["tracked_bbox"] == [16, 10, 100, 200]
    assert second["persons"][0]["track_new"] is False
    assert second["persons"][0]["track_match_iou"] > 0.8
    assert second["confirmed_person_detected"] is True
    assert second["primary_track_id"] == second["persons"][0]["track_id"]


def test_confirmed_track_becomes_lost_then_expires_by_elapsed_capture_time():
    tracker = PersonTracker(max_gap_seconds=1)
    tracker.update(observation((10, 10, 100, 200)), 1_000_000_000, "live:s1")
    tracker.update(observation((10, 10, 100, 200)), 1_100_000_000, "live:s1")
    lost = tracker.update(observation(), 1_200_000_000, "live:s1")
    expired = tracker.update(observation(), 2_200_000_000, "live:s1")

    assert lost["confirmed_person_detected"] is False
    assert lost["tracks"][0]["state"] == "lost"
    assert lost["tracks"][0]["misses"] == 1
    assert expired["active_track_count"] == 0
    assert expired["tracks"] == []


def test_default_retention_reuses_track_across_a_one_and_a_half_second_gap():
    tracker = PersonTracker()
    tracker.update(observation((10, 10, 100, 200)), 1_000_000_000, "live:s1")
    confirmed = tracker.update(observation((10, 10, 100, 200)), 1_050_000_000, "live:s1")
    tracker.update(observation(), 2_000_000_000, "live:s1")
    reacquired = tracker.update(observation((10, 10, 100, 200)), 2_550_000_000, "live:s1")

    assert reacquired["persons"][0]["track_id"] == confirmed["persons"][0]["track_id"]
    assert reacquired["persons"][0]["track_new"] is False


def test_association_is_one_to_one_and_new_detection_gets_tentative_track():
    tracker = PersonTracker(confirmation_hits=1)
    first = tracker.update(observation((10, 10, 100, 200)), 1_000_000_000, "live:s1")
    second = tracker.update(
        observation((12, 10, 100, 200), (300, 20, 80, 160)),
        1_050_000_000,
        "live:s1",
    )

    assert second["persons"][0]["track_id"] == first["persons"][0]["track_id"]
    assert second["persons"][1]["track_id"] != first["persons"][0]["track_id"]
    assert len({person["track_id"] for person in second["persons"]}) == 2


def test_namespace_change_resets_track_identity_and_state():
    tracker = PersonTracker()
    live = tracker.update(observation((10, 10, 100, 200)), 1_000_000_000, "live:s1")
    historical = tracker.update(observation((10, 10, 100, 200)), 900_000_000, "historical:buffer-1")

    assert live["persons"][0]["track_id"] == "live:s1:track:1"
    assert historical["persons"][0]["track_id"] == "historical:buffer-1:track:1"
    assert historical["persons"][0]["track_state"] == "tentative"


def test_capture_time_must_increase_within_one_timeline():
    tracker = PersonTracker()
    tracker.update(observation(), 1_000_000_000, "live:s1")
    with pytest.raises(ValueError, match="capture time"):
        tracker.update(observation(), 1_000_000_000, "live:s1")


def test_center_and_scale_gate_preserves_track_when_iou_is_low():
    tracker = PersonTracker()
    first = tracker.update(observation((0, 0, 100, 200)), 1_000_000_000, "live:s1")
    second = tracker.update(observation((80, 0, 160, 200)), 1_050_000_000, "live:s1")

    assert second["persons"][0]["track_id"] == first["persons"][0]["track_id"]
    assert second["persons"][0]["track_match_iou"] < tracker.iou_threshold


def test_distant_detection_starts_new_track():
    tracker = PersonTracker()
    first = tracker.update(observation((0, 0, 50, 100)), 1_000_000_000, "live:s1")
    second = tracker.update(observation((500, 200, 50, 100)), 1_050_000_000, "live:s1")

    assert second["persons"][0]["track_id"] != first["persons"][0]["track_id"]
    assert second["persons"][0]["track_new"] is True


def test_recent_confirmed_track_can_be_reacquired_after_abrupt_box_change():
    tracker = PersonTracker()
    tracker.update(observation((0, 0, 100, 200)), 1_000_000_000, "live:s1")
    confirmed = tracker.update(observation((2, 0, 100, 200)), 1_050_000_000, "live:s1")
    tracker.update(observation(), 1_100_000_000, "live:s1")
    reacquired = tracker.update(observation((180, 0, 300, 200)), 1_150_000_000, "live:s1")

    assert reacquired["persons"][0]["track_id"] == confirmed["persons"][0]["track_id"]
    assert reacquired["persons"][0]["track_match_iou"] == 0


def test_abrupt_box_after_reacquisition_window_starts_new_track():
    tracker = PersonTracker(reacquire_seconds=0.1)
    tracker.update(observation((0, 0, 100, 200)), 1_000_000_000, "live:s1")
    confirmed = tracker.update(observation((2, 0, 100, 200)), 1_050_000_000, "live:s1")
    later = tracker.update(observation((180, 0, 300, 200)), 1_200_000_000, "live:s1")

    assert later["persons"][0]["track_id"] != confirmed["persons"][0]["track_id"]


def test_detector_failure_does_not_turn_a_track_into_a_visible_person():
    tracker = PersonTracker(confirmation_hits=1)
    tracker.update(observation((10, 10, 100, 200)), 1_000_000_000, "live:s1")
    failed = tracker.update(
        {"person_detection_status": "error", "persons": []},
        1_050_000_000,
        "live:s1",
    )

    assert failed["person_tracking_status"] == "error"
    assert failed["confirmed_person_detected"] is False
    assert failed["active_track_count"] == 1


def test_experimental_weak_detection_sustains_confirmed_track_but_cannot_start_one():
    tracker = PersonTracker(association_confidence=0.2)
    weak = observation((10, 10, 100, 200))
    weak["persons"][0]["confidence"] = 0.3
    first = tracker.update(weak, 1_000_000_000, "live:s1")
    assert first["persons"] == [] and first["active_track_count"] == 0
    tracker.update(observation((10, 10, 100, 200)), 1_500_000_000, "live:s1")
    high = tracker.update(observation((10, 10, 100, 200)), 2_000_000_000, "live:s1")
    low = tracker.update(weak, 2_500_000_000, "live:s1")
    assert low["primary_track_id"] == high["primary_track_id"]
    assert low["persons"][0]["association_only"]
    far = observation((400, 10, 100, 200))
    far["persons"][0]["confidence"] = 0.3
    assert tracker.update(far, 3_000_000_000, "live:s1")["persons"] == []


def test_experimental_weak_evidence_cannot_confirm_tentative_or_revive_expired_track():
    tracker = PersonTracker(association_confidence=0.2)
    tracker.update(observation((10, 10, 100, 200)), 1_000_000_000, "live:s1")
    weak = observation((10, 10, 100, 200))
    weak["persons"][0]["confidence"] = 0.3
    assert tracker.update(weak, 1_500_000_000, "live:s1")["persons"] == []
    tracker.update(observation((10, 10, 100, 200)), 2_000_000_000, "live:s1")
    assert tracker.update(weak, 5_000_000_000, "live:s1")["persons"] == []


def test_experimental_weak_detection_rejects_ambiguous_overlap():
    tracker = PersonTracker(association_confidence=0.2)
    both = observation((10, 10, 100, 200), (70, 10, 100, 200))
    tracker.update(both, 1_000_000_000, "live:s1")
    tracker.update(both, 1_500_000_000, "live:s1")
    weak = observation((40, 10, 100, 200))
    weak["persons"][0]["confidence"] = 0.3
    assert tracker.update(weak, 2_000_000_000, "live:s1")["persons"] == []


def test_experimental_strong_detection_takes_priority_over_weak_overlap():
    tracker = PersonTracker(association_confidence=0.2)
    tracker.update(observation((10, 10, 100, 200)), 1_000_000_000, "live:s1")
    prior = tracker.update(observation((10, 10, 100, 200)), 1_500_000_000, "live:s1")
    obs = observation((10, 10, 100, 200), (15, 10, 100, 200))
    obs["persons"][0]["confidence"] = 0.3
    obs["persons"][1]["confidence"] = 0.9
    result = tracker.update(obs, 2_000_000_000, "live:s1")
    assert len(result["persons"]) == 1
    assert result["primary_track_id"] == prior["primary_track_id"]
    assert not result["persons"][0]["association_only"]
