"""Replay stored AWS vision evidence without opening cameras or invoking detectors.

The manifest fixes source hashes before evaluation. Annotations are used only
after replay, never by IncidentEvidence. Use one evaluator per isolated stream.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from care_vision.cloud.incident_evidence import IncidentEvidence
from care_vision.cloud.temporal_candidate import (
    SequenceObservation,
    SequencePolicy,
    TemporalCandidate,
)
from care_vision.cloud.trajectory import PostureTrajectoryAnalyzer
from care_vision.sequence_evaluation import score_episodes, validate_observations


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(rows, *, all_tracks=False):
    evaluator = IncidentEvidence()
    trajectory = PostureTrajectoryAnalyzer()
    baseline = TemporalCandidate(retain_pending_descent=True) if not all_tracks else None
    timeline = []
    for row in rows:
        t = row["source_time_seconds"]
        frame = row.get("frame_index", row.get("capture_seq", 0) - 1)
        people = []
        if all_tracks:
            header = {
                "width": 1280, "height": 720, "source_width": 1280, "source_height": 720,
                "view_kind": "full_frame", "captured_mono_ns": row["captured_mono_ns"],
            }
            for person in row["persons"]:
                if person["track_state"] != "confirmed":
                    continue
                track = person["track_id"]
                signals = trajectory.update({
                    "primary_track_id": track, "primary_tracked_bbox": person["tracked_bbox"],
                }, header, "offline-full-frame")
                people.append(SequenceObservation(
                    t, frame, track, signals["primary_orientation"],
                    signals["primary_motion_direction"], None,
                ))
        elif row["primary_track_id"] is not None:
            people.append(SequenceObservation(
                t, frame, row["primary_track_id"], row["primary_orientation"],
                row["primary_motion_direction"], None,
            ))
        if baseline is not None:
            baseline.update(people[0] if people else SequenceObservation(
                t, frame, None, None, "unknown", None,
            ))
        timeline.append(evaluator.update(t, frame, people))
    episodes = evaluator.finish()
    return {
        "samples": len(rows), "timeline": timeline, "episodes": episodes,
        "previous_candidate_count": len(baseline.episodes) if baseline is not None else None,
        "summary": {
            "descent_episodes": len(episodes),
            "post_descent_candidates": sum(e["candidate_frame"] is not None for e in episodes),
            "recovery_proxies": sum(e["recovery_frame"] is not None for e in episodes),
            "unresolved": sum(e["outcome"] == "UNRESOLVED" for e in episodes),
            "episodes_with_evidence_loss": sum(bool(e["losses"]) for e in episodes),
            "frames_without_available_person": sum(
                not any(t["visible"] for t in r["tracks"]) for r in timeline
            ),
            "track_state_counts": dict(Counter(
                t["state"] for r in timeline for t in r["tracks"]
            )),
        },
    }


def evaluate(root, manifest):
    for name, expected in manifest["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"source hash mismatch: {name}")
    runs = []
    for source in manifest["chair_reports"]:
        report = json.loads((root / source).read_text())
        for stream in report["streams"]:
            for all_tracks in (False, True):
                result = replay(stream["rows"], all_tracks=all_tracks)
                result.update(
                    source=source, video_id="chair-20260926", model=report["profile"]["label"],
                    schedule=stream["name"],
                    scope="all_confirmed_tracks" if all_tracks else "primary_only",
                )
                result["event_window"] = {
                    "window_seconds": [22, 47],
                    "descent_episodes_started": sum(
                        22 <= e["descent_time"] < 47 for e in result["episodes"]
                    ),
                    "candidates_started": sum(
                        e["candidate_time"] is not None and 22 <= e["candidate_time"] < 47
                        for e in result["episodes"]
                    ),
                }
                runs.append(result)
    dataset = json.loads((root / manifest["dataset"]).read_text())
    legacy = json.loads((root / manifest["legacy_report"]).read_text())
    validate_observations(
        legacy, dataset, manifest["files"][manifest["dataset"]],
        manifest["files"][manifest["legacy_evidence_manifest"]],
    )
    sources = {v["video_id"]: v for v in dataset["videos"]}
    for run in legacy["runs"]:
        for video in run["videos"]:
            result = replay(video["samples"])
            source = sources[video["video_id"]]
            result.update(
                source=manifest["legacy_report"], video_id=video["video_id"],
                model=legacy["backend"], schedule=f'{run["sample_fps"]}fps',
                scope="primary_only", annotated_events=len(source["events"]),
            )
            # Existing scorer expects cue/end fields. Never invent missing cues.
            candidates = [
                {**e, "recovery_cue_frame": None,
                 "recovery_cue_sources": [], "end_frame": e["recovery_frame"]}
                for e in result["episodes"] if e["candidate_frame"] is not None
            ]
            result["candidate_scoring"] = score_episodes(source, candidates)
            # The old scorer's onset cue is not this evaluator's completion proxy.
            result["candidate_scoring"].pop("recovery_cue_counts")
            for event in result["candidate_scoring"]["events"]:
                event.pop("recovery_cue")
                matched = next((e for e in result["episodes"]
                                if e["episode_id"] == event["matched_episode_id"]), None)
                annotation = next(e for e in source["events"]
                                  if e["event_id"] == event["event_id"])
                frame = matched["recovery_frame"] if matched else None
                lo, hi = annotation["upright_again_frames"]
                event["recovery_proxy_delay_from_upright_seconds"] = (
                    [(frame - hi) / source["fps"], (frame - lo) / source["fps"]]
                    if frame is not None else None
                )
            runs.append(result)
    return {
        "experiment": "offline-incident-evidence-v1", "policy": asdict(SequencePolicy()),
        "thresholds_calibrated": False, "alerts_enabled": False, "new_cv_inferences": 0,
        "camera_opened": False, "input_manifest": manifest, "runs": runs,
        "limitations": [
            "Stored AWS Graviton + COOL detections; this run evaluates temporal logic only.",
            "Chair all-track signals reuse existing bbox trajectory rules; no new detector.",
            "Legacy negatives expose only primary-person signals, not every visible person.",
            "Track IDs may fragment or swap; no identity reassociation is attempted.",
            "Horizontal boxes do not establish floor contact; "
            "recovery proxies do not prove safety.",
            "Development videos and provisional labels, not independent fall validation.",
            "REQUEST_CONTEXT is an offline suggestion, not a command sent to the Edge.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.root, json.loads(args.manifest.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for run in result["runs"]:
        print(run["video_id"], run["schedule"], run["scope"], run["summary"])


if __name__ == "__main__":
    main()
