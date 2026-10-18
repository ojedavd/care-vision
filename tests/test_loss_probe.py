from care_vision.cloud.loss_probe import LossProbe


def observation(*persons):
    return {
        "person_detection_status": "ok",
        "person_tracking_status": "ok",
        "persons": list(persons),
        "confirmed_person_count": len(persons),
    }


def person(track_id):
    return {"track_state": "confirmed", "track_id": track_id, "tracked_bbox": [1, 2, 3, 4]}


def test_confirmed_loss_requests_once_without_current_box():
    probe = LossProbe()
    assert probe.observe(1_000_000_000, observation(person("a"))) is None
    assert probe.observe(1_500_000_000, observation()) is None
    assert probe.observe(2_000_000_000, observation()) == {
        "track_id": "a",
        "captured_mono_ns": 1_000_000_000,
        "bbox": [1, 2, 3, 4],
    }
    assert probe.observe(2_500_000_000, observation()) is None


def test_ambiguous_people_and_cooldown_do_not_repeat_request():
    probe = LossProbe()
    probe.observe(1_000_000_000, observation(person("a"), person("b")))
    assert probe.observe(2_000_000_000, observation()) is None
    probe.observe(3_000_000_000, observation(person("a")))
    assert probe.observe(4_000_000_000, observation()) is not None
    probe.observe(5_000_000_000, observation(person("b")))
    assert probe.observe(6_000_000_000, observation()) is None


def test_new_unrelated_track_does_not_hide_previous_person_loss():
    probe = LossProbe()
    assert probe.observe(1_000_000_000, observation(person("main"))) is None
    assert probe.observe(1_500_000_000, observation(person("other"))) is None
    assert probe.observe(2_000_000_000, observation(person("other"))) == {
        "track_id": "main",
        "captured_mono_ns": 1_000_000_000,
        "bbox": [1, 2, 3, 4],
    }
    assert probe.observe(2_500_000_000, observation(person("main"))) is None
    assert probe.last_seen["track_id"] == "main"
