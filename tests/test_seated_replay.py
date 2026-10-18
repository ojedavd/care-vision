"""Regression on retained development logs, not independent validation."""

import importlib.util
import json
from pathlib import Path

import pytest

from care_vision.cloud.seated_candidate import SeatContext

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "replay_seated_candidates", ROOT / "tools/replay_seated_candidates.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_recorded_seated_episodes_have_candidates_and_descent_episodes_do_not():
    source = ROOT / "artifacts/tapo-weak-live-20261005"
    if not source.exists():
        pytest.skip("retained development logs unavailable")
    context = SeatContext(**json.loads(
        (ROOT / "artifacts/seated-shadow-20261014/seat-context.json").read_text()
    ))
    sitting = MODULE.replay(source / "sitting-1/cloud.jsonl", context)
    assert sitting["episodes"][0]["first_candidate_seconds"] < 25
    assert sitting["episodes"][0]["recorded_closure_seconds"] > 45
    recovery = MODULE.replay(source / "recovery-1/cloud.jsonl", context)
    assert recovery["episodes"][0]["first_candidate_seconds"] is None
    assert recovery["episodes"][1]["first_candidate_seconds"] is not None
    assert recovery["episodes"][1]["recorded_closure_seconds"] is None
    for name in ("recovery-2", "no-recovery-1"):
        result = MODULE.replay(source / name / "cloud.jsonl", context)
        assert all(e["first_candidate_seconds"] is None for e in result["episodes"])
    assert all(not r["would_close_episode"] for r in sitting["timeline"])


@pytest.mark.parametrize("name", ["crouch-near-seat-1", "kneel-near-seat-1"])
def test_real_nonseated_candidates_remain_shadow_only(name):
    """Retained failure evidence: these candidates must not become safe closures.

    The corresponding recorded capture was reviewed as crouching/kneeling beside
    the chair. This tests the v1 ambiguity boundary, not successful recognition.
    """
    source = ROOT / "artifacts/seated-confusers-live-20261014" / name / "cloud.jsonl"
    if not source.exists():
        pytest.skip("retained physical trial unavailable")
    context = SeatContext(**json.loads(
        (ROOT / "artifacts/seated-shadow-20261014/seat-context.json").read_text()
    ))
    result = MODULE.replay(source, context)
    candidates = [r for r in result["timeline"] if r["seated_like_candidate"]]
    assert candidates, "v1 counterexample no longer reproduced; audit rule changes explicitly"
    assert all(not r["would_close_episode"] for r in candidates)
