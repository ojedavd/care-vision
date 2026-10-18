from copy import deepcopy

import pytest

from care_vision.cloud.seated_candidate import SeatContext, SeatedCandidate, SeatedPolicy


def observation(t, box=(400, 200, 160, 300)):
    return {
        "observation_id": f"o-{t}", "captured_mono_ns": round(t * 1e9),
        "capture_seq": round(t * 25), "edge_id": "edge", "boot_id": "boot",
        "session_id": "session", "width": 1280, "height": 720,
        "source_width": 1280, "source_height": 720, "view_kind": "full_frame",
        "valid": True, "evidence_kind": "live", "person_detection_status": "ok",
        "person_tracking_status": "ok",
        "persons": [{"track_id": "a", "track_state": "confirmed", "tracked_bbox": list(box)}],
    }


def evaluator(**changes):
    episode = {
        "episode_id": "episode", "track_id": "a", "started_ns": 1_000_000_000,
        "reference_bbox": [400 / 1280, 40 / 720, 160 / 1280, 500 / 720],
        "horizontal_observed": False, "test_alert_emitted": False,
    }
    episode.update(changes)
    context = SeatContext("edge", 1280, 720, (0.30, 0.40, 0.15, 0.15), "synthetic test")
    return SeatedCandidate(episode, observation(1), context)


def test_seated_candidate_without_standing_remains_shadow_only():
    e = evaluator()
    for t in (1, 1.5, 2, 2.5):
        assert not e.update(observation(t))["seated_like_candidate"]
    before = observation(3)
    original = deepcopy(before)
    r = e.update(before)
    assert r["seated_like_candidate"] and r["stable_seconds"] == 2
    assert not r["would_close_episode"]
    assert before == original
    assert e.update(observation(30))["seated_like_candidate"] is False  # No gap bridging.


@pytest.mark.parametrize("step", [0.2, 0.5])
def test_stability_uses_elapsed_time_at_different_transmission_rates(step):
    e = evaluator()
    for i in range(round(2 / step)):
        assert not e.update(observation(1 + i * step))["seated_like_candidate"]
    assert e.update(observation(3))["seated_like_candidate"]


@pytest.mark.parametrize("kind", [
    "missing", "other_track", "ambiguous", "tentative", "weak", "invalid", "historical",
    "roi", "session", "boot", "camera", "dimensions", "detection", "tracking", "nan",
])
def test_unavailable_or_ambiguous_evidence_breaks_stability(kind):
    e = evaluator()
    e.update(observation(1))
    e.update(observation(1.5))
    item = observation(2)
    if kind == "missing":
        item["persons"] = []
    elif kind == "other_track":
        item["persons"][0]["track_id"] = "b"
    elif kind == "ambiguous":
        item["persons"].append(dict(item["persons"][0], track_id="b"))
    elif kind == "tentative":
        item["persons"][0]["track_state"] = "tentative"
    elif kind == "weak":
        item["persons"][0]["association_only"] = True
    elif kind == "invalid":
        item["valid"] = False
    elif kind == "historical":
        item["evidence_kind"] = "historical"
    elif kind == "roi":
        item["view_kind"] = "roi"
    elif kind in {"session", "boot", "camera"}:
        item[{"session": "session_id", "boot": "boot_id", "camera": "edge_id"}[kind]] = "new"
    elif kind == "dimensions":
        item["source_width"] = 640
    elif kind == "nan":
        item["persons"][0]["tracked_bbox"][0] = float("nan")
    else:
        item[f"person_{kind}_status"] = "unavailable"
    assert not e.update(item)["seated_like_candidate"]
    for t in (2.5, 3, 3.5, 4):
        assert not e.update(observation(t))["seated_like_candidate"]


@pytest.mark.parametrize("flag", ["horizontal_observed", "test_alert_emitted"])
def test_prior_incident_evidence_cannot_be_cleared_by_sitting(flag):
    e = evaluator(**{flag: True})
    for i in range(30):
        r = e.update(observation(1 + i * 0.5))
        assert not r["seated_like_candidate"] and not r["would_close_episode"]


def test_horizontal_then_seated_blocks_benign_branch_for_whole_episode():
    e = evaluator()
    e.update(observation(1, (300, 400, 400, 150)))
    for i in range(1, 30):
        assert not e.update(observation(1 + i * 0.5))["seated_like_candidate"]


@pytest.mark.parametrize("box", [
    (400, 40, 160, 500),  # Standing in front of the region.
    (850, 200, 160, 300),  # Contracted outside the seat region.
    (400, 300, 160, 140),  # Excessive contraction.
    (400, 200, 300, 300),  # Too wide to establish the proposed seated proxy.
])
def test_posture_or_region_mismatch_never_generates_candidate(box):
    e = evaluator()
    for i in range(12):
        assert not e.update(observation(1 + i * 0.5, box))["seated_like_candidate"]


def test_continuous_motion_cannot_accumulate_stability():
    e = evaluator()
    for i in range(6):
        box = (400, 200 + (i % 2) * 60, 160, 300)
        assert not e.update(observation(1 + i * 0.5, box))["seated_like_candidate"]


def test_duplicate_or_reordered_frames_are_rejected():
    e = evaluator()
    e.update(observation(1))
    with pytest.raises(ValueError, match="increasing"):
        e.update(observation(1))


def test_invalid_context_and_policy_are_rejected():
    for region in [(0, 0, 0, 1), (0.9, 0, 0.2, 1), (float("nan"), 0, 1, 1)]:
        with pytest.raises(ValueError):
            SeatContext("edge", 1280, 720, region, "test")
    for settings in [{"stable_seconds": 0}, {"maximum_gap_seconds": 3},
                     {"minimum_height_ratio": 0.9}, {"maximum_aspect": 1.3}]:
        with pytest.raises(ValueError):
            SeatedPolicy(**settings)


def test_identical_geometry_cannot_distinguish_hypothesized_activities():
    """Logical ambiguity test, not a recorded crouching false positive.

    If two physical activities produce the same observed boxes, adding a label
    the evaluator cannot perceive cannot make it discriminate those activities.
    """
    seated, confuser = evaluator(), evaluator()
    for t in (1, 1.5, 2, 2.5, 3):
        a, b = observation(t), observation(t)
        a["hypothesized_activity"] = "sitting"
        b["hypothesized_activity"] = "crouching beside chair"
        result_a, result_b = seated.update(a), confuser.update(b)
        assert result_a == result_b
    assert result_a["seated_like_candidate"]
    assert not result_a["would_close_episode"]
