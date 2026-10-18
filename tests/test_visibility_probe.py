import math

import numpy as np
import pytest

from care_vision.cloud.occlusion_evidence import OcclusionEvidence
from care_vision.cloud.occlusion_fragments import OcclusionFragments
from care_vision.cloud.visibility_probe import VisibilityProbe, VisibilityProbePolicy


def people(box, track="a"):
    return (
        []
        if box is None
        else [
            {
                "track_id": track,
                "track_state": "confirmed",
                "tracked_bbox": box,
            }
        ]
    )


def start(probe):
    probe.update(0, 0, people([0, 0, 20, 100]))
    probe.update(0.5, 1, people([0, 4, 20, 96]))
    return probe.update(1, 2, people([0, 10, 20, 90]))


def test_early_probe_is_not_an_incident_or_recovery():
    a = VisibilityProbe()
    s = start(a)
    assert s["request_more_evidence"]
    assert s["started"][0]["reference_bbox"] == [0, 0, 20, 100]
    assert not s["started"][0]["incident_confirmed"]
    assert not s["started"][0]["recovery_observed"]


def test_missing_detections_preserve_search_reference_not_identity():
    a = VisibilityProbe()
    start(a)
    for i in range(3, 18):
        s = a.update(i / 2, i, [])
        assert s["request_more_evidence"]
        assert s["probes"][0]["reference_time"] == 0
        assert s["probes"][0]["upright_since"] is None
    s = a.update(9, 18, [])
    assert not s["request_more_evidence"]
    assert s["probes"][0]["state"] == "UNRESOLVED"
    assert not s["probes"][0]["recovery_observed"]


def test_other_person_cannot_clear_original_probe():
    a = VisibilityProbe()
    start(a)
    for i in range(3, 10):
        s = a.update(i / 2, i, people([0, 0, 20, 100], "b"))
    assert s["probes"][0]["state"] == "VERIFYING_VISIBILITY"


def test_same_track_sustained_upright_clears_only_suspicion():
    a = VisibilityProbe()
    start(a)
    for i in range(3, 8):
        s = a.update(i / 2, i, people([0, 0, 20, 100]))
    assert s["probes"][0]["state"] == "POSTURE_CHANGE_CLEARED"
    assert not s["probes"][0]["recovery_observed"]
    assert not s["request_more_evidence"]


@pytest.mark.parametrize("interruption", ["missing", "gap"])
def test_upright_dwell_cannot_cross_missing_evidence(interruption):
    a = VisibilityProbe()
    start(a)
    a.update(1.5, 3, people([0, 0, 20, 100]))
    a.update(2, 4, people([0, 0, 20, 100]))
    if interruption == "missing":
        a.update(2.5, 5, [])
    s = a.update(3.5, 7, people([0, 0, 20, 100]))
    assert s["probes"][0]["state"] == "VERIFYING_VISIBILITY"
    assert s["probes"][0]["upright_since"] == 3.5


def test_long_timestamp_jump_expires_without_refreshing_hypothesis():
    a = VisibilityProbe()
    start(a)
    s = a.update(20, 30, people([0, 0, 20, 100]))
    assert s["probes"][0]["state"] == "UNRESOLVED"


@pytest.mark.parametrize("kind", ["static", "translation", "upward", "wide"])
def test_no_probe_without_matching_posture_geometry(kind):
    a = VisibilityProbe()
    for i in range(8):
        t = i / 2
        box = {
            "static": [0, 0, 20, 100],
            "translation": [0, t * 20, 20, 100],
            "upward": [0, 50 - t * 10, 20, 50 + t * 10],
            "wide": [0, t * 10, 150, 100 - t * 10],
        }[kind]
        s = a.update(t, i, people(box))
        assert not s["request_more_evidence"]
        assert not s["started"]


def test_uncertain_trigger_requires_visible_low_dwell_for_candidate():
    e = OcclusionEvidence("a", 1, [0, 0, 20, 100], trigger="visibility_uncertainty")
    assert e.update(1.5)["state"] == "VERIFYING_VISIBILITY"
    assert e.update(2, fragment_box=[0, 100, 90, 15])["candidate_at"] is None
    for t in (2.5, 3, 3.5, 4):
        s = e.update(t, fragment_box=[0, 100, 90, 15])
    assert s["state"] == "OCCLUDED_INCIDENT_CANDIDATE"
    assert s["events"][-1]["reason"] == "uncertain_posture_then_continuous_low_fragment"
    assert not s["hidden_pose_observed"]
    assert not s["alerts_enabled"]


def test_uncertain_trigger_never_confirms_candidate_from_missing_pixels():
    e = OcclusionEvidence("a", 1, [0, 0, 20, 100], trigger="visibility_uncertainty")
    for t in (1.5, 2, 3, 4, 5, 6, 6.5):
        s = e.update(t)
    assert s["state"] == "NO_LOW_EVIDENCE"
    assert s["candidate_at"] is None


def test_bad_inputs_do_not_mutate_existing_probe():
    a = VisibilityProbe()
    start(a)
    with pytest.raises(ValueError):
        a.update(1.5, 3, people([0, math.nan, 20, 100]))
    assert a.probes[0]["last_seen"] == 1
    with pytest.raises(ValueError):
        VisibilityProbePolicy(probe_seconds=math.inf)
    with pytest.raises(ValueError):
        OcclusionEvidence("a", 1, [0, 0, 20, 100], trigger="unknown")


def test_end_of_evidence_is_unresolved():
    a = VisibilityProbe()
    start(a)
    s = a.finish()[0]
    assert s["state"] == "UNRESOLVED"
    assert s["reason"] == "end_of_saved_observations"


def test_opt_in_seeds_one_search_and_keeps_default_trigger_unchanged():
    # Synthetic blank pixels only; this checks orchestration, not camera vision.
    image = np.zeros((360, 640, 3), np.uint8)
    experimental = OcclusionFragments(visibility_probes=True)
    original = OcclusionFragments()
    for i, box in enumerate(([0, 0, 20, 100], [0, 4, 20, 96], [0, 10, 20, 90], [0, 20, 20, 80])):
        detections = people(box)
        detections[0]["bbox"] = box
        experimental.update(image, i / 2, i, detections)
        original.update(image, i / 2, i, detections)
        if i == 2:
            assert len(experimental.groups) == 1
            assert not original.groups
    assert len(experimental.groups) == 1
    assert len(original.groups) == 1
    assert experimental.groups[0]["ledger"].trigger == "visibility_uncertainty"
    assert original.groups[0]["ledger"].trigger == "descent"


def test_perspective_fragment_can_be_above_reference_feet_only_in_opt_in_mode():
    uncertain = OcclusionEvidence("a", 0, [0, 0, 20, 100], trigger="visibility_uncertainty")
    strict = OcclusionEvidence("a", 0, [0, 0, 20, 100])
    for i in range(6):
        t = i / 2
        a = uncertain.update(t, fragment_box=[0, 66, 90, 20])
        b = strict.update(t, fragment_box=[0, 66, 90, 20])
    assert a["candidate_at"] == 2
    assert b["candidate_at"] is None
    assert not a["hidden_pose_observed"]


def test_search_band_keeps_visible_pixels_above_foot_line():
    # Synthetic pixels: unit coverage of the new image-plane search band.
    tracker = OcclusionFragments(visibility_probes=True)
    blank = np.zeros((360, 640, 3), np.uint8)
    for i, box in enumerate(([200, 20, 30, 100], [200, 24, 30, 96])):
        ps = people(box)
        ps[0]["bbox"] = box
        tracker.update(blank, i / 2, i, ps)
    image = blank.copy()
    image[100:140, 200:300] = 255
    ps = people([200, 30, 30, 90])
    ps[0]["bbox"] = [200, 30, 30, 90]
    tracker.update(image, 1, 2, ps)
    for i in range(3, 7):
        state, _, _ = tracker.update(image, i / 2, i)
    assert state["episodes"][0]["candidate_at"] == 3
    assert state["episodes"][0]["fragment_box"][1] < 120


def regional_person(track, box, state="confirmed"):
    return {"track": track, "state": state, "bbox": box, "fraction": 1.0}


@pytest.mark.parametrize("repeated_partial", [True, False])
def test_partial_semantic_dropout_uses_continuous_region_before_upright_binding(
    repeated_partial,
):
    e = OcclusionEvidence("a", 0, [0, 0, 20, 100], trigger="visibility_uncertainty")
    fragment = [0, 66, 90, 20]
    for i in range(5):
        e.update(i / 2, fragment_box=fragment)
    partial = regional_person("partial", [0, 35, 30, 60], "tentative")
    e.update(2.5, fragment_box=fragment, matches=[partial], people=[partial])
    partial["state"] = "confirmed"
    e.update(
        3,
        fragment_box=fragment,
        matches=[partial] if repeated_partial else [],
        people=[partial] if repeated_partial else [],
    )
    assert e.bound_track is None
    for t in (3.5, 4):
        e.update(t, fragment_box=fragment, matches=[], people=[])
    full = regional_person("upright", [0, 10, 30, 90], "tentative")
    e.update(4.5, fragment_box=fragment, matches=[full], people=[full])
    full["state"] = "confirmed"
    s = e.update(5, fragment_box=fragment, matches=[full], people=[full])
    assert s["bound_track"] == "upright"
    for t in (5.5, 6, 6.5, 7):
        s = e.update(t, people=[full])
    assert (s["state"] == "RECOVERY_OBSERVED") == repeated_partial


def test_regional_semantics_never_bridge_a_missing_fragment():
    e = OcclusionEvidence("a", 0, [0, 0, 20, 100], trigger="visibility_uncertainty")
    fragment = [0, 66, 90, 20]
    for i in range(5):
        e.update(i / 2, fragment_box=fragment)
    partial = regional_person("partial", [0, 35, 30, 60])
    e.update(2.5, fragment_box=fragment, matches=[partial], people=[partial])
    e.update(3, fragment_box=fragment, matches=[partial], people=[partial])
    e.update(3.5)
    full = regional_person("upright", [0, 10, 30, 90])
    s = e.update(4, fragment_box=fragment, matches=[full], people=[full])
    assert s["state"] == "UNRESOLVED"
    assert not s["rise_seen"]
