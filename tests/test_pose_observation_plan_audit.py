import copy
import importlib.util
from pathlib import Path

from test_pose_observation_plan import inputs

spec = importlib.util.spec_from_file_location(
    "pose_plan_audit", Path(__file__).resolve().parents[1] / "tools/audit_pose_observation_plan.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_late_stream_result_cannot_request_evidence_for_closed_episode():
    planner, obs, diagnostic, ep = inputs()
    obs.update(event="observation", local_mono_ns=10_000_000_000)
    diagnostic.update(event="pose_shadow", local_mono_ns=10_020_000_000)
    opened = {**ep, "event": "incident_episode", "incident_status": "unresolved"}
    closed = {**opened, "incident_status": "posture_change_cleared"}
    results = module.audit([obs, opened, closed, diagnostic], None, planner.geometry, planner.hips)
    assert results[0]["reason"] == "no_active_episode"


def test_stream_pending_command_is_not_ignored_and_replay_clocks_are_not_subtracted():
    planner, obs, diagnostic, ep = inputs()
    obs.update(event="observation", local_mono_ns=10_000_000_000)
    diagnostic.update(event="pose_shadow", local_mono_ns=10_020_000_000,
                      processing_ms=10, queue_wait_ms=1)
    opened = {**ep, "event": "incident_episode", "incident_status": "unresolved"}
    command = {"event": "agent_action", "command_id": "c"}
    results = module.audit([obs, opened, command, diagnostic], None,
                           planner.geometry, planner.hips)
    assert results[0]["reason"] == "command_pending"
    replay = copy.deepcopy(diagnostic)
    replay["local_mono_ns"] += 100_000_000_000
    results = module.audit([obs, opened], [replay], planner.geometry, planner.hips)
    assert results[0]["plan"]
    assert results[0]["age_basis"] == "replay_service_time_only"
    assert results[0]["checked_age_seconds"] == .011
