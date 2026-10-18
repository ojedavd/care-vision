import importlib.util
import json
import sys
from pathlib import Path

import pytest
from test_incident_episode import opened, sample

from care_vision.cloud.controller import Controller, Policy
from care_vision.cloud.episode_store import EpisodeStore
from care_vision.cloud.reference_posture import bbox_block_reason, recovery_block_reason

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "artifacts/seated-pose-reserved-20261014/sit-remain-2"


@pytest.mark.parametrize("box", [
    [0, 40, 150, 500], [600, 0, 150, 500], [1130, 40, 150, 500],
    [600, 220, 150, 500], [-1, 40, 150, 500], [600, 40, 0, 500],
    [600, 40, 150, float("nan")], [600, 40, 150, float("inf")], None, [600],
])
def test_invalid_or_boundary_box_is_ineligible(box):
    assert bbox_block_reason(box, 1280, 720) is not None


@pytest.mark.parametrize("interruption", ["tracked_boundary", "raw_boundary", "weak"])
def test_current_insufficiency_restarts_dwell(identity, interruption):
    c = opened(identity)
    c.observe(sample(3, box=[600, 40, 150, 500]))
    item = sample(3.5, box=[600, 40, 150, 500])
    person = item["persons"][0]
    if interruption == "tracked_boundary":
        person["tracked_bbox"] = [600, 0, 150, 500]
    elif interruption == "raw_boundary":
        person["bbox"] = [600, 0, 150, 500]
    else:
        person["association_only"] = True
    c.observe(item)
    assert c.episode["upright_since_ns"] is None
    assert c.pop_episode_events()[-1]["reason"] == "reference_posture_evidence_insufficient"
    for t in (4, 4.5, 5, 5.5):
        c.observe(sample(t, box=[600, 40, 150, 500]))
        assert c.episode is not None
    c.observe(sample(6, box=[600, 40, 150, 500]))
    assert c.episode is None


def test_invalid_reference_never_reanchors_from_later_person(identity):
    c = Controller(identity, Policy(experimental_s_only_5fps=True))
    c._open_episode(sample(1), "a", [600, 0, 150, 500])
    c.state = "UNRESOLVED"
    reference = list(c.episode["reference_bbox"])
    for i in range(30):
        c.observe(sample(2 + i * .5, box=[600, 40, 150, 500]))
    assert c.episode["reference_bbox"] == reference
    assert c.episode["reference_recovery_eligible"] is False
    assert c.episode["upright_since_ns"] is None
    assert c.finish_episode()["episode_open"]
    assert sum(e["reason"] == "reference_posture_evidence_insufficient"
               for e in c.pop_episode_events()) == 1


def test_invalid_reference_retains_alert_and_bounded_followup(identity):
    c = Controller(identity, Policy(
        experimental_s_only_5fps=True, experimental_episode_continuity=True,
        experimental_alert_after_seconds=2,
    ))
    c._open_episode(sample(1), "a", [600, 0, 150, 500])
    c.state = "UNRESOLVED"
    for t in (2, 2.5, 3, 3.5, 4):
        c.observe(sample(t, box=[550, 410, 330, 110]))
    assert c.state == "ASSISTANCE_REQUIRED"
    assert c.episode["test_alert_emitted"]
    command = c.observe(sample(31, box=[600, 40, 150, 500]))
    assert command["transmit_fps"] == 5
    assert c.episode["evidence_unavailable"]
    assert c.episode["followup_requests"] == 1
    assert c.episode["reference_recovery_eligible"] is False


def test_reference_guard_survives_private_restart(tmp_path, identity):
    policy = Policy(experimental_s_only_5fps=True, experimental_episode_continuity=True)
    c = Controller(identity, policy)
    c._open_episode(sample(1), "a", [600, 0, 150, 500])
    store = EpisodeStore(tmp_path / "episode.json")
    store.save(c)
    new = Controller({**identity, "session_id": "new"}, policy)
    assert EpisodeStore(store.path).restore(new)
    assert new.episode["reference_recovery_eligible"] is False
    assert new.episode["reference_recovery_block_reason"] == "reference_bbox_touches_frame_boundary"
    assert new.episode["identity_link_status"] == "unresolved_after_reconnect"


def test_real_seated_closure_is_blocked_by_new_guard(identity):
    spec = importlib.util.spec_from_file_location(
        "baseline_controller", ROOT / "artifacts/reference-recovery-guard-20261014/"
        "baseline/src/care_vision/cloud/controller.py",
    )
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    rows = [json.loads(x) for x in (CASE / "cloud.jsonl").read_text().splitlines()]
    start = next(r for r in rows if r["event"] == "incident_episode"
                 and r["reason"] == "episode_opened")
    closing = next(r for r in rows if r["event"] == "incident_episode"
                   and r["reason"] == "same_track_sustained_reference_posture")
    opening_observation = next(r for r in rows if r["event"] == "observation"
                               and r["observation_id"] == start["observation_id"])
    controllers = [old.Controller(identity, old.Policy(experimental_s_only_5fps=True)),
                   Controller(identity, Policy(experimental_s_only_5fps=True))]
    for c in controllers:
        c._open_episode(opening_observation, start["track_id"], [
            v * (1280 if i % 2 == 0 else 720) for i, v in enumerate(start["reference_bbox"])
        ])
        c.state = "UNRESOLVED"
    dwell_start = closing["upright_since_ns"]
    for row in rows:
        if row["event"] == "observation" and dwell_start <= row["captured_mono_ns"] <= closing[
            "captured_mono_ns"
        ]:
            for c in controllers:
                c.expected_command = row["applied_command_id"]
                c.observe(row)
    assert controllers[0].episode is None  # Recorded old bbox rule closes on seated evidence.
    assert controllers[1].episode is not None
    assert controllers[1].episode["recovery_block_reason"] == (
        "reference_bbox_touches_frame_boundary"
    )
    assert controllers[1].episode["upright_since_ns"] is None


def test_guard_is_not_a_general_seated_classifier():
    # An interior bbox alone still cannot prove upright. Preserve this limitation.
    observation = sample(1, box=[453, 30, 208, 394])
    reference = [563 / 1280, 1 / 720, 157 / 1280, 438 / 720]
    assert recovery_block_reason(reference, observation, observation["persons"][0]) is None


def test_invalid_current_box_with_invalid_reference_cannot_crash_or_count(identity):
    c = Controller(identity, Policy(experimental_s_only_5fps=True))
    c._open_episode(sample(1), "a", [600, 0, 150, 500])
    c.state = "UNRESOLVED"
    c.observe(sample(2, box=[600]))
    assert c.episode["recovery_block_reason"] == "current_invalid_bbox"
    assert c.episode["horizontal_evidence_seconds"] == 0
    assert c.episode["last_seen_ns"] is None


def test_horizontal_reference_cannot_authorize_return(identity):
    c = Controller(identity, Policy(experimental_s_only_5fps=True))
    c._open_episode(sample(1), "a", [550, 410, 330, 110])
    c.state = "UNRESOLVED"
    for t in (2, 2.5, 3, 3.5, 4):
        c.observe(sample(t, box=[600, 40, 150, 500]))
    assert c.episode["reference_recovery_block_reason"] == "reference_aspect_incompatible"
    assert c.episode["upright_since_ns"] is None
