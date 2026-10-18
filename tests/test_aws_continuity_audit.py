import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_aws_episode_continuity", ROOT / "tools/audit_aws_episode_continuity.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
EVIDENCE = ROOT / "artifacts/episode-continuity-aws-20261014"


@pytest.fixture
def evidence():
    if not (EVIDENCE / "state-after-restart.json").exists():
        pytest.skip("retained Graviton experiment unavailable")
    return EVIDENCE


def test_retained_milestone_does_not_hide_incomplete_transport(evidence):
    result = MODULE.audit(evidence)
    assert result["milestone_passed"]
    assert not result["transport_complete"] and not result["full_demo_passed"]
    assert len(result["runs"][0]["unprocessed_received_frames"]) == 1
    assert len(result["followups"]) == 3


@pytest.mark.parametrize("failure", ["ack", "new_perception", "false_closure"])
def test_audit_rejects_missing_causal_evidence_and_false_closure(evidence, monkeypatch, failure):
    original_rows = MODULE.rows

    def altered_rows(path):
        data = deepcopy(original_rows(path))
        if path.parent.name != "restart-1" or path.name != "cloud.jsonl":
            return data
        if failure == "ack":
            return [r for r in data if r["event"] != "ack_received"]
        if failure == "new_perception":
            for r in data:
                if r["event"] == "observation":
                    r["applied_command_id"] = None
        else:
            episode = next(r["episode_id"] for r in data if r["event"] == "incident_episode")
            data.append({"event": "incident_episode", "episode_id": episode,
                         "incident_status": "recovery_observed"})
        return data

    monkeypatch.setattr(MODULE, "rows", altered_rows)
    assert not MODULE.audit(evidence)["milestone_passed"]
