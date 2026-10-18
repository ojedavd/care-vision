"""Replay frozen seated-pattern policy on cloud detections and cloud pose caches.

Uses synthetic comparison windows, not actual incident openings or live outcomes.
Unavailable pose remains unavailable; annotations are scored only at exact frames.
"""

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from care_vision.cloud.seated_candidate import SeatContext, SeatedCandidate
from care_vision.cloud.seated_pose_candidate import (
    SeatedPoseCandidate,
    SeatedPosePolicy,
    SeatPoseContext,
)
from care_vision.cloud.tracking import PersonTracker


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(plan_path, cache):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(plan_path.read_text())
    if plan["pose_policy"] != asdict(SeatedPosePolicy()):
        raise ValueError("policy differs from frozen plan")
    context_path = root / plan["geometry_context_path"]
    pose_path = root / plan["source_pose"]
    review_path = root / plan["manual_review"]
    for path, digest in (
        (context_path, plan["geometry_context_sha256"]),
        (pose_path, plan["pose_sha256"]),
        (review_path, plan["manual_review_sha256"]),
    ):
        if sha(path) != digest:
            raise ValueError(f"frozen source digest mismatch: {path.name}")
    context = SeatContext(**json.loads(context_path.read_text()))
    hip_context = SeatPoseContext(**plan["hip_context"])
    runtime = json.loads((cache / "cache-runtime.json").read_text())
    cache_plan = json.loads((cache / "plan.json").read_text())
    if sha(cache / "detections.jsonl") != runtime["cache_sha256"]:
        raise ValueError("detection cache digest mismatch")
    if sha(cache / "plan.json") != runtime["plan_sha256"]:
        raise ValueError("cache plan digest mismatch")
    pose_plan = json.loads(pose_path.with_name("pose-plan.json").read_text())
    if pose_plan["cache_sha256"] != runtime["cache_sha256"]:
        raise ValueError("pose cache belongs to different detector inputs")
    raw = {
        r["frame_id"]: r
        for r in map(json.loads, (cache / "detections.jsonl").read_text().splitlines())
    }
    poses = {r["frame_id"]: r for r in map(json.loads, pose_path.read_text().splitlines())}
    reviewed = json.loads(review_path.read_text())
    conditions = []
    for run in cache_plan["runs"]:
        for schedule in run["schedules"]:
            fps = schedule["requested_sample_fps"]
            if fps not in plan["sample_fps"]:
                continue
            namespace = f"synthetic:{run['name']}:{fps}fps"
            tracker, evaluators, anchor = PersonTracker(association_confidence=0.2), {}, None
            timelines = {name: [] for name in plan["models"]}
            for frame in schedule["frames"]:
                frame_id = f"{run['name']}:{frame['capture_seq']}"
                row = raw[frame_id]
                for key in ("frame_index", "capture_seq", "captured_mono_ns"):
                    if row[key] != frame[key]:
                        raise ValueError("schedule metadata mismatch")
                tracked = tracker.update(
                    copy.deepcopy(row["observation"]), row["captured_mono_ns"], namespace
                )
                observation = {
                    **tracked,
                    "observation_id": frame_id,
                    "captured_mono_ns": row["captured_mono_ns"],
                    "capture_seq": row["capture_seq"],
                    "edge_id": context.edge_id,
                    "boot_id": "synthetic-recorded-comparison",
                    "session_id": namespace,
                    "width": context.source_width,
                    "height": context.source_height,
                    "source_width": context.source_width,
                    "source_height": context.source_height,
                    "view_kind": "full_frame",
                    "evidence_kind": "live",
                    "valid": True,
                    "input_scope": "synthetic eligible observation from recorded JPEG; not live",
                }
                people = tracked["persons"]
                if anchor is None:
                    if len(people) != 1 or people[0]["track_state"] != "confirmed":
                        continue
                    person = people[0]
                    x, y, w, h = person["tracked_bbox"]
                    if person.get("association_only") or w / h > 0.8:
                        continue
                    anchor = {
                        "episode_id": namespace + ":comparison-window",
                        "track_id": person["track_id"],
                        "started_ns": row["captured_mono_ns"],
                        "reference_bbox": [x / 1280, y / 720, w / 1280, h / 720],
                        "horizontal_observed": False,
                        "test_alert_emitted": False,
                    }
                    evaluators = {
                        name: SeatedPoseCandidate(
                            SeatedCandidate(anchor, observation, context),
                            hip_context,
                            identity_link_status="session_track_only",
                        )
                        for name in plan["models"]
                    }
                pose = poses.get(frame_id)
                if pose is not None:
                    if pose["jpeg_sha256"] != row["jpeg_sha256"]:
                        raise ValueError("pose/image digest mismatch")
                    if pose["bbox"] != row["observation"]["persons"][0]["bbox"]:
                        raise ValueError("pose belongs to different person region")
                for name, evaluator in evaluators.items():
                    prediction = pose["predictions"][name] if pose else None
                    points = (
                        prediction["points"]
                        if prediction and prediction["status"] == "ok"
                        else None
                    )
                    result = evaluator.update(observation, points)
                    timelines[name].append(
                        {
                            **result,
                            "frame_id": frame_id,
                            "capture_seq": row["capture_seq"],
                            "seconds_from_first_observation": row["seconds_from_first_observation"],
                            "pose_status": prediction["status"]
                            if prediction
                            else "not_in_pose_cache",
                        }
                    )
            for name, timeline in timelines.items():
                conditions.append(
                    {
                        "condition": f"{run['name']}:{fps}fps:{name}",
                        "run": run["name"],
                        "sample_fps": fps,
                        "model": name,
                        "synthetic_anchor": anchor,
                        "summary": {
                            "observed_samples": len(timeline),
                            "geometric_candidates": sum(r["geometric_candidate"] for r in timeline),
                            "pose_candidates": sum(r["seated_pose_candidate"] for r in timeline),
                            "first_pose_candidate_seconds": next(
                                (
                                    r["seconds_from_first_observation"]
                                    for r in timeline
                                    if r["seated_pose_candidate"]
                                ),
                                None,
                            ),
                            "reason_counts": dict(Counter(r["reason"] for r in timeline)),
                        },
                        "timeline": timeline,
                    }
                )
    snapshots = []
    for annotation in reviewed["frames"]:
        for condition in conditions:
            if condition["run"] != annotation["run"]:
                continue
            row = next(
                (r for r in condition["timeline"] if r["frame_id"] == annotation["frame_id"]), None
            )
            snapshots.append(
                {
                    "condition": condition["condition"],
                    "frame_id": annotation["frame_id"],
                    "posture_visual": annotation["posture_visual"],
                    "observed": row is not None,
                    "geometric_candidate": row["geometric_candidate"] if row else None,
                    "pose_candidate": row["seated_pose_candidate"] if row else None,
                    "evidence_status": row["evidence_status"] if row else "NOT_OBSERVED",
                    "reason": row["reason"] if row else "annotation_not_in_schedule",
                }
            )
    return {
        "schema_version": 1,
        "experiment": "seated-pose-temporal-shadow-v1",
        "plan_sha256": sha(plan_path),
        "detector_cache_sha256": runtime["cache_sha256"],
        "pose_cache_sha256": plan["pose_sha256"],
        "manual_review_sha256": plan["manual_review_sha256"],
        "policy": asdict(SeatedPosePolicy()),
        "hip_context": asdict(hip_context),
        "conditions": conditions,
        "reviewed_snapshots": snapshots,
        "scope": (
            "Synthetic comparison windows and track streams over recorded cloud caches. "
            "valid/live fields indicate eligibility only; "
            "not measured live observation validity. "
            "No actual episode opening, alert history, closure or transport simulation. "
            "Eight provisional posture points; "
            "no reviewed full-video intervals or accuracy metrics."
        ),
        "seat_support_verified": False,
        "would_close_episode": False,
        "live_activation_supported": False,
        "new_image_inferences": 0,
    }


def replay_recorded(plan_path, pose_cache):
    """Keep original episode references, track IDs, validity and capture sampling."""
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(plan_path.read_text())
    if plan["pose_policy"] != asdict(SeatedPosePolicy()):
        raise ValueError("policy differs from frozen plan")
    context_path = root / plan["geometry_context_path"]
    if sha(context_path) != plan["geometry_context_sha256"]:
        raise ValueError("context digest mismatch")
    context = SeatContext(**json.loads(context_path.read_text()))
    hip_context = SeatPoseContext(**plan["hip_context"])
    runtime = json.loads((pose_cache / "pose-runtime.json").read_text())
    if sha(pose_cache / "poses.jsonl") != runtime["poses_sha256"]:
        raise ValueError("pose cache digest mismatch")
    pose_plan = json.loads((pose_cache / "pose-plan.json").read_text())
    if not pose_plan["common_roi"].startswith("recorded cloud tracked_bbox"):
        raise ValueError("recorded replay requires logged track regions")
    if sha(pose_cache / "pose-plan.json") != runtime["plan_sha256"]:
        raise ValueError("pose plan digest mismatch")
    lookup = {
        r["frame_id"]: r
        for r in map(json.loads, (pose_cache / "poses.jsonl").read_text().splitlines())
    }
    conditions = []
    for source in plan["source_logs"]:
        path = root / source["path"]
        if sha(path) != source["sha256"]:
            raise ValueError("recorded log digest mismatch")
        metadata = next(r for r in pose_plan["runs"] if r["name"] == source["run"])
        videos = [
            digest
            for filename, digest in metadata["source_files"].items()
            if filename.endswith("demo.avi")
        ]
        if len(videos) != 1:
            raise ValueError("recorded replay requires exactly one source video hash")
        rows = list(map(json.loads, path.read_text().splitlines()))
        observations = {r["observation_id"]: r for r in rows if r["event"] == "observation"}
        origin = next(iter(observations.values()))["captured_mono_ns"]
        for model in plan["models"]:
            active, episodes, timeline = None, [], []

            def inspect(observation):
                frame_id = f"{source['run']}:{observation['capture_seq']}"
                pose = lookup.get(frame_id)
                prediction = pose["predictions"][model] if pose else None
                if pose:
                    if pose["captured_mono_ns"] != observation["captured_mono_ns"]:
                        raise ValueError("recorded pose timestamp mismatch")
                    if pose["bbox"] != observation["persons"][0]["tracked_bbox"]:
                        raise ValueError("pose belongs to different recorded track region")
                points = (
                    prediction["points"] if prediction and prediction["status"] == "ok" else None
                )
                result = active.update(observation, points)
                result.update(
                    frame_id=frame_id,
                    capture_seq=observation["capture_seq"],
                    seconds_from_first_observation=(observation["captured_mono_ns"] - origin) / 1e9,
                    pose_status=prediction["status"] if prediction else "not_in_pose_cache",
                )
                timeline.append(result)
                current = episodes[-1]
                for field, candidate in (
                    ("first_geometric_candidate_seconds", result["geometric_candidate"]),
                    ("first_pose_candidate_seconds", result["seated_pose_candidate"]),
                ):
                    if candidate and current[field] is None:
                        current[field] = result["seconds_from_first_observation"]

            for row in rows:
                if row["event"] == "observation" and active is not None:
                    inspect(row)
                elif row["event"] == "incident_episode":
                    if row["reason"] == "episode_opened":
                        start = observations[row["observation_id"]]
                        active = SeatedPoseCandidate(
                            SeatedCandidate(row, start, context),
                            hip_context,
                            identity_link_status=row.get(
                                "identity_link_status", "session_track_only"
                            ),
                        )
                        episodes.append(
                            {
                                "episode_id": row["episode_id"],
                                "track_id": row["track_id"],
                                "recorded_reference_bbox": row["reference_bbox"],
                                "first_geometric_candidate_seconds": None,
                                "first_pose_candidate_seconds": None,
                                "recorded_closure_seconds": None,
                            }
                        )
                        inspect(start)
                    elif active is not None and row["episode_id"] == active.geometry.episode_id:
                        active.geometry.blocked |= (
                            row["horizontal_observed"] or row["test_alert_emitted"]
                        )
                        if row["incident_status"] in {
                            "recovery_observed",
                            "posture_change_cleared",
                        }:
                            episodes[-1]["recorded_closure_seconds"] = (
                                row["captured_mono_ns"] - origin
                            ) / 1e9
                            active = None
            conditions.append(
                {
                    "run": source["run"],
                    "model": model,
                    "source": str(path),
                    "source_sha256": source["sha256"],
                    "source_video_sha256": videos[0],
                    "episodes": episodes,
                    "summary": {
                        "observed_samples": len(timeline),
                        "geometric_candidates": sum(r["geometric_candidate"] for r in timeline),
                        "pose_candidates": sum(r["seated_pose_candidate"] for r in timeline),
                        "reason_counts": dict(Counter(r["reason"] for r in timeline)),
                    },
                    "timeline": timeline,
                }
            )
    return {
        "schema_version": 1,
        "experiment": "seated-pose-recorded-episodes-shadow-v1",
        "plan_sha256": sha(plan_path),
        "pose_cache_sha256": runtime["poses_sha256"],
        "conditions": conditions,
        "policy": asdict(SeatedPosePolicy()),
        "scope": (
            "Conditional on original recorded episode references, identity, validity and sampling. "
            "New pose on reencoded AVI JPEGs and logged track boxes; "
            "no original payload parity. "
            "No new openings, suppression, transport or actual closures simulated."
        ),
        "seat_support_verified": False,
        "would_close_episode": False,
        "live_activation_supported": False,
        "new_inferences_in_replay": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--recorded-poses", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    if bool(args.cache) == bool(args.recorded_poses):
        parser.error("select either synthetic --cache or original --recorded-poses")
    result = (
        replay(args.plan, args.cache)
        if args.cache
        else replay_recorded(args.plan, args.recorded_poses)
    )
    root = Path(__file__).resolve().parents[1]
    result["code_sha256"] = {
        name: sha(root / name)
        for name in (
            "tools/replay_seated_pose_candidates.py",
            "src/care_vision/cloud/seated_pose_candidate.py",
            "src/care_vision/cloud/seated_candidate.py",
            "src/care_vision/cloud/tracking.py",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for condition in result["conditions"]:
        print(
            condition.get("condition", f"{condition['run']}:{condition['model']}"),
            condition["summary"],
        )


if __name__ == "__main__":
    main()
