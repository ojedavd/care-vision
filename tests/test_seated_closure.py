import pytest

from care_vision.cloud.controller import Controller, Policy
from care_vision.cloud.server import CloudServer
from care_vision.common.metrics import EventLog

UPRIGHT = [600, 40, 150, 500]
SEATED = [600, 250, 170, 300]
HORIZONTAL = [550, 410, 330, 110]


def sample(t, *, track="a", box=SEATED, valid=True, fps=2):
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
        "target_transmit_fps": fps,
        "person_detection_status": "ok",
        "person_tracking_status": "ok",
        "persons": []
        if box is None
        else [{"track_id": track, "track_state": "confirmed", "tracked_bbox": box}],
    }


def diagnostic(controller, observation, candidate=True):
    return {
        "episode_id": controller.episode["episode_id"] if controller.episode else None,
        "observation_id": observation["observation_id"],
        "seated_pose_candidate": candidate,
    }


def opened(identity, *, origin="posture_change", seconds=5, **policy):
    c = Controller(
        identity,
        Policy(experimental_s_only_5fps=True, experimental_seated_closure_seconds=seconds,
               **policy),
    )
    c._open_episode(sample(1), "a", UPRIGHT, origin=origin)
    c.state = "UNRESOLVED"
    return c


def feed(c, times, *, box=SEATED, candidate=True, age=0.2):
    decisions = []
    for t in times:
        item = sample(t, box=box)
        c.observe(item)
        if c.episode is None:
            break
        decisions.append(
            c.consider_seated_closure(item, diagnostic(c, item, candidate), result_age_seconds=age)
        )
    return decisions


def seconds(start, end, step=0.5):
    return [start + i * step for i in range(round((end - start) / step) + 1)]


def test_sustained_seated_pattern_after_observed_change_closes_to_normal(identity):
    c = opened(identity)
    decisions = feed(c, seconds(2, 6.5))
    assert c.state == "UNRESOLVED"
    assert decisions[-1]["reason"] == "awaiting_sustained_seated_pattern"
    decisions = feed(c, [7])
    assert decisions[-1]["closed"] is True
    assert decisions[-1]["seat_support_verified"] is False
    assert c.state == "NORMAL"
    assert c.episode is None
    event = c.pop_episode_events()[-1]
    assert event["reason"] == "experimental_seated_pattern_cleared_posture_change"
    assert event["incident_status"] == "seated_posture_change_cleared_experimental"


def test_closure_transition_is_logged_from_unresolved(identity):
    c = opened(identity)
    feed(c, seconds(2, 7))
    transition = c.pop_transitions()[-1]
    assert (transition["from_state"], transition["to_state"], transition["reason"]) == (
        "UNRESOLVED", "NORMAL", "experimental_seated_closure"
    )


@pytest.mark.parametrize("origin", ["person_missing", "unspecified"])
def test_only_observed_posture_change_episodes_can_close(identity, origin):
    c = opened(identity, origin=origin)
    decisions = feed(c, seconds(2, 20))
    assert c.state == "UNRESOLVED"
    assert {d["reason"] for d in decisions} == {"episode_not_opened_by_observed_posture_change"}


def test_prior_horizontal_evidence_blocks_closure_for_whole_episode(identity):
    c = opened(identity)
    feed(c, [2], box=HORIZONTAL)
    decisions = feed(c, seconds(2.5, 20))
    assert c.state == "UNRESOLVED"
    assert decisions[-1]["reason"] == "prior_horizontal_or_alert_requires_incident_followup"


def test_alert_blocks_closure(identity):
    c = opened(identity)
    c.episode["test_alert_emitted"] = True
    c.state = "ASSISTANCE_REQUIRED"
    decisions = feed(c, seconds(2, 20))
    assert c.state == "ASSISTANCE_REQUIRED"
    assert decisions[-1]["reason"] == "prior_horizontal_or_alert_requires_incident_followup"


def test_unlinked_identity_blocks_closure(identity):
    c = opened(identity)
    c.episode["identity_link_status"] = "unresolved_after_reconnect"
    decisions = feed(c, seconds(2, 20))
    assert c.state == "UNRESOLVED"
    assert decisions[-1]["reason"] == "episode_identity_unlinked"


@pytest.mark.parametrize("interruption", ["gap", "invalid", "missing", "ambiguous", "other"])
def test_any_observation_break_permanently_blocks_seated_closure(identity, interruption):
    c = opened(identity)
    feed(c, [2, 2.5])
    if interruption == "gap":
        feed(c, [4])
    elif interruption == "invalid":
        c.observe(sample(3, valid=False))
    elif interruption == "missing":
        c.observe(sample(3, box=None))
    elif interruption == "ambiguous":
        item = sample(3)
        item["persons"].append(dict(item["persons"][0], track_id="b"))
        c.observe(item)
    else:
        c.observe(sample(3, track="b"))
    decisions = feed(c, seconds(4.5, 30))
    assert c.state == "UNRESOLVED"
    assert decisions[-1]["reason"] == "episode_observation_interrupted"
    # The standing-recovery route remains available after the interruption.
    for t in seconds(30.5, 33):
        c.observe(sample(t, box=UPRIGHT))
    assert c.state == "NORMAL"
    assert c.pop_episode_events()[-1]["incident_status"] == "posture_change_cleared"


def test_non_candidate_result_restarts_sustained_run(identity):
    c = opened(identity)
    feed(c, seconds(2, 5))
    feed(c, [5.5], candidate=False)
    decisions = feed(c, seconds(6, 10.5))
    assert c.state == "UNRESOLVED"
    assert decisions[-1]["seated_pattern_seconds"] == pytest.approx(4.5)
    feed(c, [11])
    assert c.state == "NORMAL"


def test_missing_diagnostic_results_restart_run_without_breaking_episode(identity):
    c = opened(identity)
    feed(c, seconds(2, 4))
    # Controller frames continue, but the pose diagnostic skips 1.5 s.
    for t in (4.5, 5):
        c.observe(sample(t))
    decisions = feed(c, seconds(5.5, 10))
    assert c.state == "UNRESOLVED"
    assert decisions[-1]["seated_pattern_seconds"] == pytest.approx(4.5)
    feed(c, [10.5])
    assert c.state == "NORMAL"


def test_stale_or_mismatched_diagnostics_never_count(identity):
    c = opened(identity)
    decisions = feed(c, seconds(2, 20), age=1.5)
    assert {d["reason"] for d in decisions} == {"stale_diagnostic"}
    item = sample(20.5)
    c.observe(item)
    wrong = dict(diagnostic(c, item), observation_id="other")
    assert c.consider_seated_closure(item, wrong, result_age_seconds=0)["reason"] == (
        "diagnostic_episode_or_observation_mismatch"
    )
    wrong = dict(diagnostic(c, item), episode_id="other")
    assert c.consider_seated_closure(item, wrong, result_age_seconds=0)["reason"] == (
        "diagnostic_episode_or_observation_mismatch"
    )
    assert c.state == "UNRESOLVED"


def test_closure_waits_for_baseline_state_and_no_pending_command(identity):
    c = opened(identity)
    c.state = "VERIFYING"
    decisions = feed(c, seconds(2, 8))
    assert c.state == "VERIFYING"
    assert decisions[-1]["reason"] == "closure_deferred_until_baseline_observation"
    c.state = "UNRESOLVED"
    c.pending = {"command_id": "pending"}
    item = sample(8.5)
    c.observe(item)
    assert c.consider_seated_closure(item, diagnostic(c, item), result_age_seconds=0)[
        "reason"] == "closure_deferred_until_baseline_observation"
    c.pending = None
    feed(c, [9])
    assert c.state == "NORMAL"


def test_controller_must_have_seen_the_person_recently(identity):
    c = opened(identity)
    feed(c, seconds(2, 6.5))
    item = sample(7.5)
    # Diagnostic arrives without the controller having processed recent frames.
    decision = c.consider_seated_closure(item, diagnostic(c, item), result_age_seconds=0)
    assert decision["reason"] == "controller_person_evidence_stale"
    assert c.state == "UNRESOLVED"


def test_restored_episode_cannot_close_by_seated_pattern(identity):
    c = opened(identity, experimental_episode_continuity=True)
    episode = dict(c.episode)
    restored = Controller(identity, c.policy)
    restored.restore_episode(episode)
    assert restored.episode["observed_continuously"] is False


def test_disabled_by_default_and_episode_unchanged(identity):
    c = Controller(identity, Policy(experimental_s_only_5fps=True))
    c._open_episode(sample(1), "a", UPRIGHT, origin="posture_change")
    assert "observed_continuously" not in c.episode
    assert "origin" not in c.episode
    with pytest.raises(ValueError):
        c.consider_seated_closure(sample(2), {}, result_age_seconds=0)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_policy_rejects_invalid_closure_time(value):
    with pytest.raises(ValueError):
        Policy(experimental_s_only_5fps=True, experimental_seated_closure_seconds=value)


def test_policy_and_server_require_s_only_and_pose_shadow(tmp_path):
    with pytest.raises(ValueError):
        Policy(experimental_seated_closure_seconds=5)
    log = EventLog("cloud", tmp_path / "log.jsonl")
    try:
        with pytest.raises(ValueError):
            CloudServer(log, experimental_s_only_5fps=True, experimental_seated_closure_seconds=5)
        with pytest.raises(ValueError):
            CloudServer(log, pose_shadow_factory=lambda log: None,
                        experimental_seated_closure_seconds=5)
    finally:
        log.close()


def switch_frame(t, **changes):
    item = sample(t, valid=False)
    item["invalid_reason"] = "insufficient_temporal_context"
    item.update(changes)
    return item


def test_motion_context_only_invalid_frame_keeps_continuity(identity):
    c = opened(identity)
    c.observe(switch_frame(1.5))
    assert c.episode["observed_continuously"] is True
    decisions = feed(c, seconds(2, 7))
    assert decisions[-1]["closed"] is True


@pytest.mark.parametrize("variant", ["other_reason", "gap", "no_person", "other_track",
                                     "two_people", "detection_failed", "edge_box"])
def test_other_invalid_frames_still_break_continuity(identity, variant):
    c = opened(identity)
    feed(c, [2])
    t = 4 if variant == "gap" else 2.5
    item = switch_frame(t)
    if variant == "other_reason":
        item["invalid_reason"] = "processing_stale"
    elif variant == "no_person":
        item["persons"] = []
    elif variant == "other_track":
        item["persons"][0]["track_id"] = "b"
    elif variant == "two_people":
        item["persons"].append(dict(item["persons"][0], track_id="b"))
    elif variant == "detection_failed":
        item["person_detection_status"] = "error"
    elif variant == "edge_box":
        item["persons"][0]["tracked_bbox"] = [0, 250, 170, 300]
    c.observe(item)
    assert c.episode["observed_continuously"] is False
    decisions = feed(c, seconds(t + 0.5, 20))
    assert decisions[-1]["reason"] == "episode_observation_interrupted"
