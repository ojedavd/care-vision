"""Build the three-trial panel from measured logs; never recompute controller decisions."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/tapo-weak-live-20261005"
OUT = BASE / "dashboard"
SCENARIOS = [
    ("recovery-2", "Recovery", "Same episode → NORMAL", "recovery", 32.6),
    (
        "no-recovery-1",
        "No recovery",
        "Open episode + test alert",
        "open-alert",
        40.1,
    ),
    ("sitting-1", "Sitting", "No alert → NORMAL", "negative", 49.1),
]
LABELS = {
    "episode_opened": "Episode opens",
    "same_track_horizontal_observed": "Horizontal posture · same ID",
    "same_track_sustained_reference_posture": "Recovery · same episode",
    "TRIGGER_ALERT_TEST": "Experimental alert · no notification",
    "evidence_ended_episode_open": "End of capture · episode open",
}


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    trials = []
    for name, title, subtitle, outcome, focus in SCENARIOS:
        folder = BASE / name
        cloud, edge = records(folder / "cloud.jsonl"), records(folder / "edge.jsonl")
        captures = records(folder / "demo.captures.jsonl")
        observed = [row for row in cloud if row["event"] == "observation"]
        sent = [row for row in edge if row["event"] == "frame_sent"]
        ready = next(row for row in cloud if row["event"] == "server_ready")
        report = json.loads((folder / "report.json").read_text())
        audit = json.loads((folder / "episode-audit-v2.json").read_text())
        assert report["passed"] and audit["passed"] and audit["expected_outcome"] == outcome
        assert ready["experimental_weak_association"] and ready["cool_verified"]
        assert [row["frame_id"] for row in sent] == [row["frame_id"] for row in observed]
        assert len({row["primary_track_id"] for row in observed if row["primary_track_id"]}) == 1
        origin = captures[0]["captured_mono_ns"]

        def seconds(ns):
            return round((ns - origin) / 1e9, 6)

        frames = []
        total_bytes = 0
        for row, frame in zip(observed, sent, strict=True):
            total_bytes += frame["payload_bytes"]
            frames.append(
                {
                    "t": seconds(row["captured_mono_ns"]),
                    "seq": row["capture_seq"],
                    "fps": row["target_transmit_fps"],
                    "track": row["primary_track_id"],
                    "bbox": row["primary_tracked_bbox"],
                    "confirmed": row["confirmed_person_detected"],
                    "confidence": row["primary_person_confidence"],
                    "orientation": row["primary_orientation"],
                    "weak": any(p.get("association_only", False) for p in row["persons"]),
                    "processing": round(row["processing_ms"], 3),
                    "valid": row["valid"],
                    "bytes": total_bytes,
                }
            )
        states = [{"t": 0, "state": "NORMAL"}] + [
            {"t": seconds(row["captured_mono_ns"]), "state": row["to_state"]}
            for row in cloud
            if row["event"] == "agent_state_transition"
        ]
        episodes, events = [], []
        for row in cloud:
            if row["event"] == "incident_episode":
                ended = row["reason"] == "evidence_ended_episode_open"
                t = frames[-1]["t"] if ended else seconds(row["captured_mono_ns"])
                closed = row["incident_status"] in {"recovery_observed", "posture_change_cleared"}
                episodes.append(
                    {
                        "t": t,
                        "id": row["episode_id"],
                        "status": row["incident_status"],
                        "open": not closed,
                        "horizontal": round(row["horizontal_evidence_seconds"], 3),
                        "alert": row["test_alert_emitted"],
                    }
                )
                label = LABELS[row["reason"]]
                if row["incident_status"] == "posture_change_cleared":
                    label = "Posture change cleared · same ID"
                events.append({"t": t, "kind": "episode", "label": label})
            elif row["event"] == "agent_action" and row["type"] == "SET_TRANSMIT_FPS":
                observation = next(
                    r for r in observed if r["observation_id"] == row["caused_by_observation_id"]
                )
                ack = next(
                    r
                    for r in edge
                    if r["event"] == "command_ack" and r["command_id"] == row["command_id"]
                )
                assert ack["result"] == "applied"
                events.append(
                    {
                        "t": seconds(observation["captured_mono_ns"]),
                        "kind": "command",
                        "label": f"Agent requests {row['transmit_fps']} FPS",
                        "applied": seconds(ack["applied_at_edge_mono_ns"]),
                        "command": row["command_id"],
                    }
                )
        events.sort(key=lambda row: row["t"])
        target = OUT / "videos" / f"{name}.mp4"
        if not target.exists():
            target.symlink_to(Path("../..") / name / "review.mp4")
        trials.append(
            {
                "key": name,
                "title": title,
                "subtitle": subtitle,
                "focus": focus,
                "video": f"videos/{name}.mp4",
                "duration": seconds(captures[-1]["captured_mono_ns"]),
                "captureTimes": [seconds(row["captured_mono_ns"]) for row in captures],
                "frames": frames,
                "states": states,
                "episodes": episodes,
                "events": events,
                "captured": len(captures),
                "sent": len(sent),
                "bytes": total_bytes,
                "alerts": audit["alerts"],
                "alertThreshold": audit["threshold_seconds"],
                "final": audit["final_state"],
                "weak": sum(frame["weak"] for frame in frames),
                "captureMedian": report["capture_fps_median"],
                "processingMedian": report["processing_ms_median"],
                "episodeId": episodes[0]["id"],
            }
        )
    payload = {"trials": trials, "date": "October 5, 2026", "instance": "m8g.4xlarge"}
    (OUT / "data.js").write_text(
        "window.CARE_TRIALS = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n"
    )
    sources = {
        name: {
            file: digest(BASE / name / file)
            for file in (
                "cloud.jsonl",
                "edge.jsonl",
                "demo.captures.jsonl",
                "report.json",
                "episode-audit-v2.json",
                "review.mp4",
            )
        }
        for name, *_ in SCENARIOS
    }
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "source": (
                    "Measured Edge/cloud logs; prerecorded local video proxy; no controller replay"
                ),
                "timeline": (
                    "Capture monotonic timestamps, origin = first local capture. "
                    "Video frame index maps to captureTimes."
                ),
                "bbox": "Last cloud tracked bounding box; never inference on video playback",
                "episode_evidence": (
                    "Logged milestone snapshots only; not interpolated accumulation"
                ),
                "passed_trials": [trial["key"] for trial in trials],
                "sources_sha256": sources,
                "excluded_failed_trial": "recovery-1, preserved in parent review",
            },
            indent=2,
        )
        + "\n"
    )
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
