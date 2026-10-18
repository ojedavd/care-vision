"""Compare a frozen cumulative bbox hypothesis with the existing offline ledger."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from care_vision.cloud.accumulated_descent import AccumulatedDescent, DescentPolicy
from care_vision.cloud.incident_evidence import IncidentEvidence
from care_vision.cloud.temporal_candidate import SequenceObservation
from care_vision.cloud.trajectory import PostureTrajectoryAnalyzer
from care_vision.sequence_evaluation import validate_observations


def replay(rows, *, all_tracks):
    analyzers = {name: IncidentEvidence() for name in ("baseline", "accumulated")}
    timelines = {name: [] for name in analyzers}
    trajectory = PostureTrajectoryAnalyzer()
    accumulated = AccumulatedDescent()
    features = []
    for row in rows:
        timestamp = row["source_time_seconds"]
        frame = row.get("frame_index", row.get("capture_seq", 0) - 1)
        people = []
        if all_tracks:
            header = {
                "width": 1280, "height": 720, "captured_mono_ns": row["captured_mono_ns"],
            }
            for person in row["persons"]:
                if person["track_state"] != "confirmed":
                    continue
                track = person["track_id"]
                signals = trajectory.update({
                    "primary_track_id": track, "primary_tracked_bbox": person["tracked_bbox"],
                }, header, "offline-full-frame")
                people.append({
                    "track_id": track, "bbox": signals["primary_source_bbox"],
                    "orientation": signals["primary_orientation"],
                    "motion": signals["primary_motion_direction"],
                })
        elif row["primary_track_id"] is not None and row["primary_source_bbox"] is not None:
            people.append({
                "track_id": row["primary_track_id"], "bbox": row["primary_source_bbox"],
                "orientation": row["primary_orientation"],
                "motion": row["primary_motion_direction"],
            })
        extra = accumulated.update(timestamp, frame, {p["track_id"]: p["bbox"] for p in people})
        by_track = {s["track_id"]: s for s in extra}
        for person in people:
            signal = by_track[person["track_id"]]
            signal["original_motion"] = person["motion"]
            signal["orientation"] = person["orientation"]
            signal["added_downward"] = (
                signal["descent_signal"] and person["motion"] in ("stable", "unknown")
            )
            signal["direction_conflict"] = (
                signal["descent_signal"] and person["motion"] == "upward"
            )
        features.extend(extra)
        for name, analyzer in analyzers.items():
            observations = []
            for person in people:
                motion = person["motion"]
                if name == "accumulated" and by_track[person["track_id"]]["added_downward"]:
                    motion = "downward"
                observations.append(SequenceObservation(
                    timestamp, frame, person["track_id"], person["orientation"], motion, None,
                ))
            timelines[name].append(analyzer.update(timestamp, frame, observations))
    variants = {}
    for name, analyzer in analyzers.items():
        episodes = analyzer.finish()
        variants[name] = {
            "episodes": episodes, "timeline": timelines[name],
            "summary": {
                "suspicions": len(episodes),
                "candidates": sum(e["candidate_frame"] is not None for e in episodes),
                "recovery_proxies": sum(e["recovery_frame"] is not None for e in episodes),
                "unresolved": sum(e["outcome"] == "UNRESOLVED" for e in episodes),
            },
        }
    return {
        "samples": len(rows), "scope": "all_confirmed_tracks" if all_tracks else "primary_only",
        "features": features, "variants": variants,
        "added_downward_frames": sum(s["added_downward"] for s in features),
        "direction_conflict_frames": sum(s["direction_conflict"] for s in features),
    }


def evaluate(root, manifest):
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"hash mismatch: {name}")
    if manifest["policy"] != asdict(DescentPolicy()):
        raise ValueError("policy differs from frozen experiment")
    runs = []
    for source in manifest["detector_reports"]:
        report = json.loads((root / source["path"]).read_text())
        for stream in report["streams"]:
            result = replay(stream["rows"], all_tracks=True)
            result.update(
                video_id=stream.get("video_id", "chair-20260926"),
                model=report["profile"]["label"], schedule=stream["name"], source=source["path"],
            )
            runs.append(result)
    legacy = json.loads((root / manifest["legacy_report"]).read_text())
    dataset = json.loads((root / manifest["dataset"]).read_text())
    validate_observations(
        legacy, dataset, manifest["files"][manifest["dataset"]],
        manifest["files"][manifest["legacy_evidence_manifest"]],
    )
    for run in legacy["runs"]:
        for video in run["videos"]:
            result = replay(video["samples"], all_tracks=False)
            result.update(video_id=video["video_id"], model=legacy["backend"],
                          schedule=f'{run["sample_fps"]}fps', source=manifest["legacy_report"])
            runs.append(result)
    # All annotation-dependent interpretation is downstream of replay.
    annotations = {v["video_id"]: v for v in dataset["videos"]}
    for run in runs:
        if run["video_id"] == "chair-20260926":
            onset, contact, rise = [22, 24], [26, 27], [41, 43]
        else:
            video = annotations[run["video_id"]]
            if not video["events"]:
                run["annotation"] = {"kind": "development_negative"}
                continue
            event = video["events"][0]
            onset = [f / video["fps"] for f in event["descent_onset_frames"]]
            contact = [f / video["fps"] for f in event["floor_contact_frames"]]
            rise = [f / video["fps"] for f in event["recovery_onset_frames"]]
        run["annotation"] = {
            "kind": "provisional_controlled_descent", "onset_seconds": onset,
            "contact_seconds": contact, "recovery_onset_seconds": rise,
            "identity_attribution": "geometric track only; inspect images separately",
        }
        for variant in run["variants"].values():
            variant["descent_starts_in_event"] = [
                e["descent_time"] for e in variant["episodes"]
                if onset[0] <= e["descent_time"] < contact[1]
            ]
            variant["candidates_in_floor_window"] = [
                e["candidate_time"] for e in variant["episodes"]
                if e["candidate_time"] is not None and contact[1] <= e["candidate_time"] < rise[0]
            ]
    return {
        "experiment": "accumulated-descent-v1", "manifest": manifest, "runs": runs,
        "alerts_enabled": False, "camera_opened": False,
        "scope": "Offline bbox signal and temporal replay; no live controller changes.",
        "limitations": [
            "Hypothesis designed after inspecting development chair geometry; "
            "not held-out validation.",
            "Sitting, bending, occlusion and detector deformation can match bbox descent geometry.",
            "An added descent is a posture-change suspicion, not a confirmed physical fall.",
            "All-track chair and primary-only legacy results "
            "have different detector/scope coverage.",
            "No cross-ID reassociation or recovery under occlusion is inferred.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.root, json.loads(args.manifest.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for run in report["runs"]:
        print(run["video_id"], run["model"], run["schedule"],
              "added", run["added_downward_frames"],
              {k: v["summary"] for k, v in run["variants"].items()})


if __name__ == "__main__":
    main()
