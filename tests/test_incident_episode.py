from care_vision.cloud.controller import Controller, Policy


def sample(t, *, track="a", box=None, valid=True):
    return {
        "observation_id": f"o-{t}",
        "frame_id": f"f-{t}",
        "capture_seq": round(t * 30),
        "captured_mono_ns": round(t * 1e9),
        "valid": valid,
        "motion_score": 0,
        "applied_command_id": None,
        "width": 1280,
        "height": 720,
        "target_transmit_fps": 2,
        "person_detection_status": "ok",
        "person_tracking_status": "ok",
        "persons": []
        if box is None
        else [{"track_id": track, "track_state": "confirmed", "tracked_bbox": box}],
    }


def opened(identity):
    c = Controller(identity, Policy(experimental_s_only_5fps=True))
    c._open_episode(sample(1), "a", [600, 40, 150, 500])
    c.state = "UNRESOLVED"
    c.observe(sample(2, box=[550, 410, 330, 110]))
    return c


def test_same_track_recovery_requires_sustained_reference_posture(identity):
    c = opened(identity)
    for t in (51, 70, 89):
        c.observe(sample(t, box=[550, 410, 330, 110]))
        assert c.state == "UNRESOLVED"
    for t in (90, 90.5, 91, 91.5):
        c.observe(sample(t, box=[600, 40, 150, 500]))
        assert c.state == "UNRESOLVED"
    c.observe(sample(92, box=[600, 40, 150, 500]))
    assert c.state == "NORMAL"
    assert c.episode is None
    assert c.pop_episode_events()[-1]["incident_status"] == "recovery_observed"


def test_other_person_and_missing_evidence_never_close_episode(identity):
    c = opened(identity)
    for i in range(100):
        c.observe(sample(3 + i * 0.5, track="b", box=[600, 40, 150, 500]))
    c.observe(sample(100))
    assert c.state == "UNRESOLVED"
    assert c.finish_episode()["incident_status"] == "unresolved"
    assert c.episode["track_id"] == "a"


def test_gap_invalid_ambiguous_history_and_wrong_command_break_recovery(identity):
    for interruption in ("gap", "invalid", "ambiguous", "command", "history"):
        c = opened(identity)
        c.observe(sample(3, box=[600, 40, 150, 500]))
        item = sample(3.5, box=[600, 40, 150, 500])
        if interruption == "gap":
            item["captured_mono_ns"] = 4_000_000_000
        elif interruption == "invalid":
            item["valid"] = False
        elif interruption == "ambiguous":
            item["persons"].append(dict(item["persons"][0], track_id="b"))
        elif interruption == "command":
            item["applied_command_id"] = "unapplied"
        else:
            item["evidence_kind"] = "historical"
        c.observe(item)
        # At least one absence/gap after a rejected frame must prevent bridging.
        c.observe(sample(4.5, box=[600, 40, 150, 500]))
        c.observe(sample(5, box=[600, 40, 150, 500]))
        assert c.state == "UNRESOLVED"
        assert c.finish_episode() is not None


def alert_controller(identity, threshold=2):
    c = Controller(
        identity, Policy(experimental_s_only_5fps=True, experimental_alert_after_seconds=threshold)
    )
    c._open_episode(sample(1), "a", [600, 40, 150, 500])
    c.state = "UNRESOLVED"
    return c


def test_horizontal_evidence_alerts_once_and_recovery_after_alert_is_linked(identity):
    c = alert_controller(identity)
    for t in (2, 2.5, 3, 3.5):
        c.observe(sample(t, box=[550, 410, 330, 110]))
        assert c.state == "UNRESOLVED"
    c.observe(sample(4, box=[550, 410, 330, 110]))
    assert c.state == "ASSISTANCE_REQUIRED"
    for t in (4.5, 5, 5.5):
        c.observe(sample(t, box=[550, 410, 330, 110]))
    events = c.pop_episode_events()
    assert sum(e["reason"] == "TRIGGER_ALERT_TEST" for e in events) == 1
    alert = next(e for e in events if e["reason"] == "TRIGGER_ALERT_TEST")
    assert alert["horizontal_evidence_seconds"] == 2
    assert c.finish_episode()["episode_open"] is True
    for t in (6, 6.5, 7, 7.5, 8):
        c.observe(sample(t, box=[600, 40, 150, 500]))
    recovery = c.pop_episode_events()[-1]
    assert recovery["episode_id"] == alert["episode_id"]
    assert recovery["incident_status"] == "recovery_observed"
    assert recovery["test_alert_emitted"] is True
    assert c.state == "NORMAL"


def test_missing_invalid_other_track_and_history_do_not_count_as_horizontal_time(identity):
    c = alert_controller(identity)
    c.observe(sample(2, box=[550, 410, 330, 110]))
    c.observe(sample(2.5, box=[550, 410, 330, 110]))
    c.observe(sample(3, box=[550, 410, 330, 110], valid=False))
    c.observe(sample(3.5, track="b", box=[550, 410, 330, 110]))
    item = sample(4, box=[550, 410, 330, 110])
    item["evidence_kind"] = "historical"
    c.observe(item)
    c.observe(sample(50, box=[550, 410, 330, 110]))
    assert c.episode["horizontal_evidence_seconds"] == 0.5
    assert c.state == "UNRESOLVED"
    assert not c.episode["test_alert_emitted"]


def test_experimental_alert_threshold_is_validated_and_off_by_default(identity):
    import pytest

    for threshold in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Policy(experimental_s_only_5fps=True, experimental_alert_after_seconds=threshold)
    with pytest.raises(ValueError):
        Policy(experimental_alert_after_seconds=20)
    c = opened(identity)
    for i in range(200):
        c.observe(sample(3 + i * 0.5, box=[550, 410, 330, 110]))
    assert c.state == "UNRESOLVED"
    assert not c.episode["test_alert_emitted"]


def test_weak_candidate_does_not_open_new_s_only_episode(identity):
    c = Controller(identity, Policy(experimental_s_only_5fps=True))
    for t in (1, 1.5, 2, 2.5, 3):
        item = sample(t, box=[600, 40, 150, 500])
        c.observe(item)
    for t in (3.5, 4, 4.5, 5):
        item = sample(t, box=[550, 410, 330, 110])
        item["persons"][0]["association_only"] = True
        assert c.observe(item) is None
    assert c.episode is None
    assert c.state == "NORMAL"
