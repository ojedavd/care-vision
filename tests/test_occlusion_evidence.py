import pytest

from care_vision.cloud.occlusion_evidence import OcclusionEvidence


def candidate():
    e = OcclusionEvidence("original", 0, [100, 20, 60, 100])
    for i in range(21):
        e.update(i / 10, fragment_box=[100, 100, 120, 25])
    assert e.state == "OCCLUDED_INCIDENT_CANDIDATE"
    return e


def person(track="new", box=(100, 80, 40, 60), state="confirmed"):
    return {"track": track, "bbox": list(box), "state": state}


def test_partial_visibility_candidate_does_not_invent_hidden_pose():
    e = candidate()
    assert not e.snapshot()["hidden_pose_observed"]
    assert not e.snapshot()["alerts_enabled"]


def test_fragment_loss_cannot_be_bridged_by_later_standing_person():
    e = candidate()
    e.update(3, fragment_box=[100, 100, 120, 25], matches=[person()], people=[person()])
    assert e.state == "UNRESOLVED"
    e.update(3.5, matches=[person()], people=[person()])
    assert e.bound_track is None


def test_two_people_cannot_resolve_identity():
    e = candidate()
    e.update(2.1, fragment_box=[100, 100, 120, 25], matches=[person("a"), person("b")], people=[])
    assert e.state == "UNRESOLVED"


def test_one_match_does_not_bind_and_different_match_does_not_continue_it():
    e = candidate()
    e.update(2.1, fragment_box=[100, 100, 120, 25], matches=[person("a")], people=[])
    e.update(2.6, fragment_box=[100, 100, 120, 25], matches=[person("b")], people=[])
    assert e.bound_track is None


def bind(e):
    e.update(2.1, fragment_box=[100, 100, 120, 25], matches=[person(state="tentative")], people=[])
    e.update(2.6, fragment_box=[100, 100, 120, 25], matches=[person()], people=[])
    assert e.bound_track == "new"


def test_recovery_requires_rise_and_sustained_upright_of_bound_person():
    e = candidate()
    bind(e)
    for t in [3.1, 3.6, 4.1, 4.6, 5.1]:
        e.update(t, people=[person(box=[100, 20, 40, 120])])
    assert e.state == "RECOVERY_OBSERVED"
    assert e.rise_seen


def test_standing_without_rise_does_not_close_candidate():
    e = candidate()
    for t in [2.1, 2.6]:
        e.update(
            t, fragment_box=[100, 100, 120, 25], matches=[person(box=[100, 20, 40, 120])], people=[]
        )
    for t in [3.1, 3.6, 4.1, 4.6, 5.1]:
        e.update(t, people=[person(box=[100, 20, 40, 120])])
    assert e.state == "PERSON_REASSOCIATED"


def test_other_person_or_missing_observations_cannot_supply_recovery():
    for people in [[person("other", [100, 20, 40, 120])], None]:
        e = candidate()
        bind(e)
        e.update(3.5, people=people)
        assert e.state == "UNRESOLVED"


def test_bend_without_low_evidence_expires():
    e = OcclusionEvidence("original", 0, [100, 20, 60, 100])
    for i in range(52):
        e.update(i / 10)
    assert e.state == "NO_LOW_EVIDENCE"
    assert e.candidate_at is None


def test_low_evidence_must_be_contiguous():
    e = OcclusionEvidence("original", 0, [100, 20, 60, 100])
    for i in range(40):
        e.update(i / 10, fragment_box=None if i % 10 == 0 else [100, 100, 120, 25])
    assert e.candidate_at is None


def test_causal_chronology_is_enforced():
    e = candidate()
    with pytest.raises(ValueError):
        e.update(1)


def test_bending_is_cleared_when_same_person_returns_upright():
    e = OcclusionEvidence("original", 0, [100, 20, 60, 100])
    for i in range(11):
        t = i * 0.5
        e.update(
            t, people=[person("original", [100, 20 if t >= 1 else 60, 40, 100 if t >= 1 else 60])]
        )
    assert e.state == "POSTURE_CHANGE_CLEARED"
    assert e.candidate_at is None


def test_other_upright_person_cannot_clear_descent():
    e = OcclusionEvidence("original", 0, [100, 20, 60, 100])
    for i in range(9):
        e.update(i * 0.5, people=[person("other", [100, 20, 40, 100])])
    assert e.state == "VERIFYING_DESCENT"


def test_late_fragment_cannot_seed_an_old_descent():
    e = OcclusionEvidence("original", 0, [100, 20, 60, 100])
    e.update(5.1, fragment_box=[100, 100, 120, 25])
    assert e.state == "NO_LOW_EVIDENCE"


def test_end_of_clip_is_unresolved_not_recovery():
    e = candidate()
    assert e.finish()["state"] == "UNRESOLVED"
    assert e.events[-1]["reason"] == "end_of_evidence"


def test_rise_does_not_remain_fresh_forever():
    e = candidate()
    bind(e)
    for i in range(12):
        # A rise cue, but broad non-upright box cannot establish standing.
        e.update(3.1 + i * 0.5, people=[person(box=[100, 20, 200, 120])])
    assert e.state == "UNRESOLVED"
    assert e.events[-1]["reason"] == "rise_evidence_expired_without_sustained_upright"


def test_repeated_floor_fragment_detection_does_not_end_pixel_continuity():
    e = candidate()
    for t in [2.1, 2.6, 3.1]:
        e.update(
            t,
            fragment_box=[100, 100, 120, 25],
            matches=[person(box=[100, 100, 120, 25])],
            people=[],
        )
    assert e.bound_track is None
    assert e.state == "OCCLUDED_INCIDENT_CANDIDATE"
