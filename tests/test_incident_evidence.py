import pytest

from care_vision.cloud.incident_evidence import IncidentEvidence
from care_vision.cloud.temporal_candidate import SequenceObservation


def step(evaluator, t, orientation="vertical", motion="stable", track="a", extras=()):
    frame = round(t * 100)
    observations = [] if track is None else [
        SequenceObservation(t, frame, track, orientation, motion, None)
    ]
    observations.extend(SequenceObservation(t, frame, *extra, None) for extra in extras)
    return evaluator.update(t, frame, observations)


def candidate():
    evaluator = IncidentEvidence()
    step(evaluator, 0, motion="downward")
    for t in (.5, 1, 1.5):
        step(evaluator, t, "horizontal")
    assert evaluator.episodes[0]["candidate_time"] == 1.5
    return evaluator


def test_descent_loss_is_unresolved_even_when_another_person_is_upright():
    evaluator = IncidentEvidence()
    step(evaluator, 0, motion="downward")
    for i in range(1, 15):
        state = step(evaluator, i / 2, track="b")
    assert state["unresolved_count"] == 1
    assert state["tracks"][0]["state"] == "INSUFFICIENT_EVIDENCE"
    assert not state["fall_confirmed"]
    episode = evaluator.finish()[0]
    assert episode["candidate_time"] is None
    assert episode["outcome"] == "UNRESOLVED"
    assert episode["end_reason"] == "end_of_recording_unresolved"
    assert len(episode["losses"]) == 1


def test_horizontal_without_descent_and_normal_disappearance_are_not_candidates():
    evaluator = IncidentEvidence()
    for i in range(10):
        step(evaluator, i / 2, "horizontal")
    state = step(evaluator, 5, track=None)
    assert state["tracks"][0]["state"] == "PERSON_UNOBSERVED"
    assert evaluator.finish() == []


def test_missing_frame_breaks_horizontal_dwell_but_preserves_descent():
    evaluator = IncidentEvidence()
    step(evaluator, 0, motion="downward")
    step(evaluator, .5, "horizontal")
    step(evaluator, 1, track=None)
    for t in (1.5, 2):
        step(evaluator, t, "horizontal")
        assert evaluator.episodes[0]["candidate_time"] is None
    step(evaluator, 2.5, "horizontal")
    assert evaluator.episodes[0]["candidate_time"] == 2.5


def test_stale_descent_does_not_bridge_long_occlusion_into_candidate():
    evaluator = IncidentEvidence()
    step(evaluator, 0, motion="downward")
    for i in range(1, 13):
        step(evaluator, i / 2, track=None)
    for t in (6.5, 7, 7.5):
        step(evaluator, t, "horizontal")
    assert evaluator.finish()[0]["candidate_time"] is None


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_quiet_or_upright_alone_never_means_recovered(orientation):
    evaluator = candidate()
    for i in range(4, 15):
        step(evaluator, i / 2, orientation)
    assert evaluator.finish()[0]["outcome"] == "UNRESOLVED"


def test_recovery_requires_upward_then_contiguous_upright_same_identity():
    evaluator = candidate()
    step(evaluator, 2, motion="upward")
    for t in (2.5, 3, 3.5):
        state = step(evaluator, t)
        assert state["unresolved_count"] == 1
    state = step(evaluator, 4)
    assert state["events"] == [{"track_id": "a", "event": "recovery_proxy_observed"}]
    assert evaluator.finish()[0]["outcome"] == "RECOVERY_PROXY_OBSERVED"


@pytest.mark.parametrize("break_kind", ["missing", "gap", "downward", "new_identity"])
def test_broken_recovery_evidence_does_not_close_candidate(break_kind):
    evaluator = candidate()
    step(evaluator, 2, motion="upward")
    if break_kind == "missing":
        step(evaluator, 2.5, track=None)
    elif break_kind == "downward":
        step(evaluator, 2.5, motion="downward")
    for t in (3, 3.5, 4, 4.5, 5, 5.5):
        step(evaluator, t, track="b" if break_kind == "new_identity" else "a")
    assert evaluator.finish()[0]["outcome"] == "UNRESOLVED"


def test_two_people_are_evaluated_independently_in_same_frame():
    evaluator = IncidentEvidence()
    step(evaluator, 0, motion="downward", extras=(("b", "vertical", "stable"),))
    for t in (.5, 1, 1.5):
        step(evaluator, t, "horizontal", extras=(("b", "vertical", "upward"),))
    assert len(evaluator.episodes) == 1
    assert evaluator.episodes[0]["track_id"] == "a"
    assert evaluator.episodes[0]["recovery_time"] is None


def test_bad_batch_and_reordered_time_rejected_without_state_mutation():
    evaluator = IncidentEvidence()
    step(evaluator, 0)
    before = evaluator.last_time
    with pytest.raises(ValueError):
        step(evaluator, .5, extras=(("a", "vertical", "downward"),))
    assert evaluator.last_time == before
    with pytest.raises(ValueError):
        step(evaluator, 0)
    evaluator.finish()
    with pytest.raises(ValueError):
        step(evaluator, 1)


def test_expired_upward_cue_cannot_reuse_earlier_upright_dwell():
    evaluator = candidate()
    step(evaluator, 2, "intermediate", "upward")
    for i in range(5, 14):
        step(evaluator, i / 2, "intermediate")
    step(evaluator, 7)
    for t in (7.5, 8, 8.5):
        step(evaluator, t)
    step(evaluator, 9, motion="upward")
    assert evaluator.episodes[0]["recovery_time"] is None
    for t in (9.5, 10, 10.5):
        step(evaluator, t)
    step(evaluator, 11)
    assert evaluator.finish()[0]["recovery_time"] == 11
