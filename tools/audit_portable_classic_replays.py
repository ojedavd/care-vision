"""Audit portable classic replays with external archived assets; no clinical accuracy claim."""

import argparse
import hashlib
import json
from pathlib import Path

from audit_mvp_sensing_comparison import distribution, resource_window, rows
from run_pose_observation_loop import save


def prepared_runtime_matches(preflight, plan, workspace):
    return (
        preflight.get('cool', {}).get('verified') is True
        and preflight.get('classic_available') is True
        and preflight.get('machine') in {'aarch64', 'arm64'}
        and preflight.get('sdk_installed_by_tool') is False
        and preflight.get('uv_version') == plan['portable_runtime']['uv_version']
        and preflight.get('python_requested') == plan['portable_runtime']['python']
        and preflight.get('python_environment') == str(Path(workspace) / '.venv/bin/python')
    )


def causal(event, observations, opened):
    obs = observations.get(event.get("observation_id"))
    origin = opened.get(event["episode_id"])
    if not obs or not origin:
        return False
    people = obs["persons"]
    return bool(
        obs["valid"]
        and obs.get("evidence_kind") == "live"
        and obs.get("view_kind", "full_frame") == "full_frame"
        and event["frame_id"] == obs["frame_id"]
        and len(people) == 1
        and people[0]["track_state"] == "confirmed"
        and origin["track_id"] == event["track_id"] == people[0]["track_id"]
    )


def audit(root, asset_root):
    project = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "plan.json").read_text())
    results = []
    for case in plan["conditions"]:
        source, mode = case["source"], case["mode"]
        folder = root / (source + "-" + mode)
        cloud, edge = rows(folder / "cloud.jsonl"), rows(folder / "edge.jsonl")
        original = (
            asset_root
            / "artifacts"
            / ("pose-observation-live-20261014" if mode == "pose" else "tapo-weak-live-20261005")
            / source
        )
        captures = rows(original / "demo.captures.jsonl")
        stamps = {r["capture_seq"]: r["captured_mono_ns"] for r in captures}
        groups = [
            [r for r in records if r["event"] == event]
            for records, event in (
                (edge, "frame_sent"),
                (cloud, "frame_received"),
                (cloud, "observation"),
                (cloud, "frame_dropped"),
            )
        ]
        sent, received, observed, dropped = groups
        ids = [{r["frame_id"] for r in group} for group in groups]
        observations = {r["observation_id"]: r for r in observed}
        events = [r for r in cloud if r["event"] == "incident_episode"]
        opened = {r["episode_id"]: r for r in events if r["reason"] == "episode_opened"}
        horizontal = [r for r in events if r["reason"] == "same_track_horizontal_observed"]
        closures = [
            r
            for r in events
            if r["incident_status"] in {"recovery_observed", "posture_change_cleared"}
        ]
        alerts = [r for r in cloud if r["event"] == "test_alert"]
        pose = [r for r in cloud if r["event"] == "pose_shadow"]
        actions = [r for r in cloud if r["event"] == "agent_action"]
        cloud_acks = {r["command_id"]: r for r in cloud if r["event"] == "ack_received"}
        edge_acks = {r["command_id"]: r for r in edge if r["event"] == "command_ack"}
        action_checks = []
        for action in actions:
            cid = action["command_id"]
            ack, edge_ack = cloud_acks.get(cid), edge_acks.get(cid)
            cause = observations.get(action["caused_by_observation_id"])
            field = (
                "applied_command_id"
                if action["type"] == "SET_TRANSMIT_FPS"
                else "applied_visual_command_id"
            )
            fresh = [
                r
                for r in observed
                if r.get(field) == cid
                and edge_ack
                and r["capture_seq"] >= edge_ack.get("effective_from_capture_seq", 0)
            ]
            action_checks.append(
                {
                    "command_id": cid,
                    "fps": action.get("transmit_fps"),
                    "new_observations": len(fresh),
                    "new_valid_observations": sum(r["valid"] for r in fresh),
                    "passed": bool(
                        cause
                        and cause["frame_id"] == action["based_on_frame_id"]
                        and ack
                        and edge_ack
                        and ack["result"] == edge_ack["result"] == "applied"
                        and ack["session_id"] == edge_ack["session_id"] == action["session_id"]
                        and any(r["valid"] for r in fresh)
                    ),
                }
            )
        mapping = rows(folder / "replay-captures.jsonl")
        stop = next(r for r in edge if r["event"] == "source_stopped")
        runtime = json.loads((folder / "runtime.json").read_text())
        transitions = [r for r in cloud if r["event"] == "agent_state_transition"]
        final_state = transitions[-1]["to_state"] if transitions else None
        expected_state = (
            "NORMAL"
            if source in {"recovery-2", "sit-return-live-2"}
            else "ASSISTANCE_REQUIRED"
            if mode == "test20"
            else "UNRESOLVED"
        )
        checks = {
            "all_frames_processed": ids[0] == ids[1] == ids[2] and not dropped,
            "unique_records": all(len(g) == len(i) for g, i in zip(groups, ids)),
            "capture_complete": stop["captured"] == len(captures) and not stop["source_error"],
            "original_timestamps_preserved": len(mapping) == len(captures)
            and all(r["original_captured_mono_ns"] == stamps[r["capture_seq"]] for r in mapping)
            and len(
                {r["replay_captured_mono_ns"] - r["original_captured_mono_ns"] for r in mapping}
            )
            == 1,
            "drained": any(r["event"] == "drain_confirmed" for r in cloud),
            "single_session": sum(r["event"] == "session_started" for r in cloud) == 1,
            "no_errors_or_stale": not any(
                r["event"] in {"session_error", "pose_shadow_error"}
                or r.get("invalid_reason") == "processing_stale"
                for r in cloud
            ),
            "graviton_cool_8threads": runtime["cool_verified"]
            and runtime["runtime"]["ec2"]["architecture"] == "arm64"
            and runtime["opencv_num_threads"] == 8,
            "frozen_runtime_profile": runtime["experimental_episode_continuity"]
            and runtime["experimental_s_only_5fps"]
            and runtime["pose_observation_enabled"]
            and runtime["association_confidence"] == 0.20
            and runtime["new_track_confidence"] == 0.35
            and runtime["experimental_alert_after_seconds"] == (20 if mode == "test20" else None),
            "horizontal_observed": bool(horizontal),
            "causal_episode_events": bool(horizontal)
            and all(causal(r, observations, opened) for r in horizontal + closures + alerts),
            "expected_closure": (
                len(closures) == 1
                and closures[0]["incident_status"] == "recovery_observed"
                and any(r["episode_id"] == closures[0]["episode_id"] for r in horizontal)
                if source == "recovery-2"
                else not closures
                and bool(events)
                and events[-1]["reason"] == "evidence_ended_episode_open"
                and events[-1]["episode_open"]
            ),
            "alert_policy": (
                len(alerts) == 1
                and alerts[0]["horizontal_evidence_seconds"] >= 20
                and alerts[0]["test_only"]
                and not alerts[0]["notification_sent"]
                and events[-1]["episode_id"] == alerts[0]["episode_id"]
                if mode == "test20"
                else not alerts
            ),
            "final_state": final_state == expected_state,
            "closure_dwell_and_guard": all(
                r.get("recovery_block_reason") is None
                and r.get("upright_since_ns") is not None
                and r["captured_mono_ns"] - r["upright_since_ns"] >= 2e9
                for r in closures
            ),
            "unknown_context_blocks_pose": bool(pose)
            and all(
                r["context_status"] == "unknown"
                and r["pose_status"] == "not_requested"
                and not r["would_close_episode"]
                for r in pose
            ),
            "no_roi_commands": all(r["type"] == "SET_TRANSMIT_FPS" for r in actions),
            "actions_causal_acked_fresh": bool(actions) and all(r["passed"] for r in action_checks),
            "final_baseline": sent[-1]["target_transmit_fps"] == 2,
            "worker_stopped": json.loads((folder / "cleanup.json").read_text())[
                "confirmed_stopped"
            ],
        }
        checks["explicit_classic_observed"] = (
            bool(observed)
            and all(r["person_detector_backend"].endswith(":engine-classic") for r in observed)
            and any(
                r["event"] == "person_detector_configuration"
                and r["dnn_engine_requested"] == "classic"
                for r in cloud
            )
        )
        if mode == "pose":
            for key in (
                "horizontal_observed",
                "causal_episode_events",
                "expected_closure",
                "unknown_context_blocks_pose",
                "no_roi_commands",
            ):
                del checks[key]
            cycles = [r for r in cloud if r["event"] == "pose_observation_cycle"]
            checks["one_roi_cycle_complete"] = (
                sum(r["stage"] == "cycle_complete" for r in cycles) == 1
                and sum(r["type"] == "REQUEST_ROI" for r in actions) == 1
                and sum(r["type"] == "REQUEST_FULL_FRAME" for r in actions) == 1
            )
            checks["pose_inference_and_current_context"] = any(
                r["pose_status"] == "ok" and r["context_status"] == "current" for r in pose
            )
            checks["no_pose_closure"] = all(not r["would_close_episode"] for r in pose)
            checks["expected_outcome"] = (
                len(closures) == 1
                and closures[0]["incident_status"] == "posture_change_cleared"
                and causal(closures[0], observations, opened)
                if source == "sit-return-live-2"
                else not closures
                and events[-1]["reason"] == "evidence_ended_episode_open"
                and events[-1]["episode_open"]
            )
            review = json.loads((original / "activity-review.json").read_text())
            checks["closure_in_reviewed_standing"] = all(
                any(
                    p["activity"] == "standing"
                    and p["from"]
                    <= (
                        stamps[observations[c["observation_id"]]["capture_seq"]]
                        - captures[0]["captured_mono_ns"]
                    )
                    / 1e9
                    <= p["to"]
                    for p in review["intervals_seconds_conservative"]
                )
                for c in closures
            )
        resources = resource_window(
            [r for r in cloud if r["event"] == "process_resources"],
            received[0]["local_mono_ns"],
            max(r["local_mono_ns"] for r in cloud if r["event"] in {"observation", "pose_shadow"}),
        )
        checks["resource_window_covered"] = abs(
            resources["sample_coverage_seconds"] - resources["window_seconds"]
        ) < .1
        timeline = []
        for event in events:
            obs = observations.get(event.get("observation_id"))
            timeline.append(
                {
                    k: event.get(k)
                    for k in (
                        "reason",
                        "episode_id",
                        "track_id",
                        "incident_status",
                        "horizontal_evidence_seconds",
                        "recovery_block_reason",
                    )
                }
            )
            timeline[-1]["source_seconds"] = (
                (stamps[obs["capture_seq"]] - captures[0]["captured_mono_ns"]) / 1e9
                if obs
                else None
            )
        results.append(
            {
                **case,
                "checks": checks,
                "passed": all(checks.values()),
                "sent_received_processed_dropped": [len(g) for g in groups],
                "network_limited_frames": sum(bool(r.get("network_limited")) for r in sent),
                "final_state": final_state,
                "alerts": len(alerts),
                "processing_ms": distribution([r["processing_ms"] for r in observed]),
                "resources": resources,
                "pose_inferences": sum(r["pose_status"] == "ok" for r in pose),
                "closures": len(closures),
                "actions": action_checks,
                "events": timeline,
            }
        )
    deployment = json.loads((root / "deployment.json").read_text())
    integrity = all(
        hashlib.sha256(((project if key == 'code_sha256' else asset_root) / p).read_bytes())
        .hexdigest() == h
        for key in ("code_sha256", "input_sha256")
        for p, h in plan[key].items()
    )
    cleanup = deployment.get("complete") and deployment.get("existing_states_unchanged")
    cleanup = bool(cleanup and deployment["cleanup"]["workspace_removed"])
    preflight = json.loads((root / 'remote-preflight.json').read_text())
    runtime_ok = prepared_runtime_matches(preflight, plan, deployment['workspace'])
    frozen = json.loads((root / 'local-preflight.json').read_text())['files_sha256']
    required = {p for p in plan['code_sha256'] if p.startswith('src/')}
    required.update(plan['frozen_asset_paths'])
    integrity = integrity and bool(frozen) and required.issubset(frozen) and all(
        hashlib.sha256(((project if p.startswith('src/') else asset_root) / p).read_bytes())
        .hexdigest() == h for p, h in frozen.items()
    )
    result = {
        "passed": all(r["passed"] for r in results) and integrity and cleanup and runtime_ok,
        "prepared_runtime_verified": runtime_ok,
        "frozen_code_and_inputs_unchanged": integrity,
        "cleanup_verified": cleanup,
        "runs": results,
        "scope": "Classic YOLOX opt-in episodes and pose/ROI on separate known videos; "
        "not fall accuracy or active ROI interaction with risky episodes",
    }
    save(root / "assessment.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder, args.asset_root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
