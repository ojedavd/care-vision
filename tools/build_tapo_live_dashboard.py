"""Build an offline, telemetry-only dashboard for the two live Tapo trials."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/tapo-live-loss-20261004"
OUTPUT = SOURCE / "dashboard"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def seconds(captured_ns, origin):
    return round((captured_ns - origin) / 1e9, 3)


def build_trial(name):
    folder = SOURCE / name
    cloud = read_jsonl(folder / "cloud.jsonl")
    edge = read_jsonl(folder / "edge.jsonl")
    audit = json.loads((folder / "loss-loop-audit.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    spoken = json.loads((folder / "spoken-cues.json").read_text())
    origin = spoken["capture_start_mono_ns"]
    sent = {row["frame_id"]: row for row in edge if row["event"] == "frame_sent"}
    observations = [row for row in cloud if row["event"] == "observation"]
    assert len(sent) == len(observations) == audit["processed"]
    rows = []
    total_bytes = 0
    for row in observations:
        frame = sent[row["frame_id"]]
        total_bytes += frame["payload_bytes"]
        rows.append(
            {
                "t": seconds(row["captured_mono_ns"], origin),
                "seq": row["capture_seq"],
                "fps": row["target_transmit_fps"],
                "width": row["width"],
                "height": row["height"],
                "bytes": frame["payload_bytes"],
                "cumulativeBytes": total_bytes,
                "detected": row["person_detected"],
                "confirmed": row["confirmed_person_detected"],
                "personCount": row["person_count"],
                "track": row["primary_track_id"].split(":")[-1]
                if row["primary_track_id"]
                else None,
                "bbox": row["primary_person_bbox"],
                "orientation": row["primary_orientation"],
                "nearFloor": row["primary_near_floor"],
                "motion": row["primary_motion_direction"],
                "confidence": row["primary_person_confidence"],
                "processingMs": round(row["processing_ms"], 2),
                "queueMs": round(row["cloud_queue_wait_ms"], 2),
                "backend": "YOLOX-S" if row["target_transmit_fps"] == 5 else "YOLOX-Nano",
            }
        )
    by_id = {row["frame_id"]: row for row in observations}
    events = []
    for cue in spoken["delivered_cues"]:
        events.append(
            {"t": cue["actual_seconds"], "kind": "cue", "label": cue["text"]}
        )
    for action in (row for row in cloud if row["event"] == "agent_action"):
        cause = by_id[action["based_on_frame_id"]]
        events.append(
            {
                "t": seconds(cause["captured_mono_ns"], origin),
                "kind": "command",
                "label": (
                    f"SET_TRANSMIT_FPS({action['transmit_fps']})"
                    if action["type"] == "SET_TRANSMIT_FPS"
                    else "REQUEST_FULL_FRAME"
                ),
                "reason": action["reason"],
                "seq": action["command_seq"],
                "ack": True,
            }
        )
    for outcome in (row for row in cloud if row["event"] == "loss_verification_outcome"):
        events.append(
            {
                "t": seconds(outcome["captured_mono_ns"], origin),
                "kind": "outcome",
                "label": "Verificación sin resolver",
                "frames": outcome["s_frames"],
                "personFrames": outcome["s_person_frames"],
            }
        )
    events.sort(key=lambda row: (row["t"], {"cue": 0, "command": 1, "outcome": 2}[row["kind"]]))
    return {
        "name": name,
        "rows": rows,
        "events": events,
        "captured": report["frames_captured"],
        "sent": audit["sent"],
        "received": audit["received"],
        "processed": audit["processed"],
        "dropped": audit["dropped"],
        "commands": audit["commands"],
        "acks": audit["acks"],
        "completedLoops": audit["completed_loops"],
        "verificationOpen": audit["verification_open_at_capture_end"],
        "totalBytes": total_bytes,
        "sMedianMs": audit["s_processing_median_ms"],
        "coolVerified": audit["cool_verified"],
        "duration": 60,
    }


def main():
    OUTPUT.mkdir(exist_ok=True)
    trials = [build_trial("trial-1"), build_trial("trial-2")]
    assert trials[0]["completedLoops"] == 0
    assert trials[1]["completedLoops"] == 3
    assert trials[1]["commands"] == trials[1]["acks"] == 14
    payload = json.dumps(trials, ensure_ascii=False, separators=(",", ":"))
    (OUTPUT / "data.js").write_text(
        "window.CARE_LIVE_DATA = " + payload + ";\n", encoding="utf-8"
    )
    (OUTPUT / "qa.json").write_text(
        json.dumps(
            {
                "source": "live RTSP WebSocket logs; no source video or image pixels",
                "trials": [
                    {
                        "name": trial["name"],
                        "rows": len(trial["rows"]),
                        "commands": trial["commands"],
                        "completed_loops": trial["completedLoops"],
                    }
                    for trial in trials
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT / "index.html")


if __name__ == "__main__":
    main()
