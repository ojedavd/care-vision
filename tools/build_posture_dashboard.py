"""Build the local video dashboard from the saved WebSocket run."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/posture-websocket-20261004/lateral"
OUT = ROOT / "artifacts/posture-websocket-20261004/dashboard"
VIDEO_FPS = 25


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def video_time(frame_id):
    return (int(frame_id.rsplit(":", 1)[-1]) - 1) / VIDEO_FPS


def main():
    cloud = records(RUN / "cloud.jsonl")
    observations = [row for row in cloud if row.get("event") == "observation"]
    actions = [row for row in cloud if row.get("event") == "agent_action"]
    outcomes = [row for row in cloud if row.get("event") == "loss_verification_outcome"]
    audit = json.loads((RUN / "loss-loop-audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] and audit["processed"] == len(observations)
    ack_ids = {
        row["command_id"]
        for row in cloud
        if row.get("event") == "ack_received" and row.get("result") in {"applied", "duplicate"}
    }
    assert len(ack_ids) == len(actions) == audit["commands"]
    rows = []
    for row in observations:
        rows.append(
            {
                "t": round(video_time(row["frame_id"]), 3),
                "seq": row["capture_seq"],
                "fps": row["target_transmit_fps"],
                "width": row["width"],
                "height": row["height"],
                "detector": "YOLOX-S" if row["target_transmit_fps"] == 5 else "YOLOX-Nano",
                "bbox": row.get("primary_person_bbox"),
                "confirmed": row.get("confirmed_person_detected", False),
                "track": row.get("primary_track_id", "").split(":")[-1]
                if row.get("primary_track_id")
                else None,
                "orientation": row.get("primary_orientation"),
                "processingMs": round(row.get("processing_ms") or 0, 1),
                "queueMs": round(row.get("cloud_queue_wait_ms") or 0, 2),
            }
        )
    events = []
    for row in actions:
        events.append(
            {
                "t": round(video_time(row["based_on_frame_id"]), 3),
                "kind": "action",
                "type": row["type"],
                "fps": row.get("transmit_fps"),
                "reason": row.get("reason"),
                "ack": row["command_id"] in ack_ids,
            }
        )
    for row in outcomes:
        events.append(
            {
                "t": round(video_time(row["frame_id"]), 3),
                "kind": "outcome",
                "type": row["outcome"],
                "incidentStatus": row["incident_status"],
                "horizontalFrames": row["horizontal_frames"],
                "verticalFrames": row["vertical_frames"],
            }
        )
    events.sort(key=lambda item: item["t"])
    data = {"duration": 2994 / VIDEO_FPS, "stats": audit, "rows": rows, "events": events}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.js").write_text(
        "window.CARE_DASHBOARD = "
        + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256()
    with (RUN / "demo.avi").open("rb") as video:
        for chunk in iter(lambda: video.read(1024 * 1024), b""):
            digest.update(chunk)
    provenance = {
        "video_source": str((RUN / "demo.avi").relative_to(ROOT)),
        "video_source_sha256": digest.hexdigest(),
        "video_output": "video.mp4",
        "video_fps": VIDEO_FPS,
        "video_frames": 2994,
        "overlay_source": str((RUN / "cloud.jsonl").relative_to(ROOT)),
        "overlay_rows": len(rows),
        "actions": len(actions),
        "outcomes": len(outcomes),
        "alignment": "capture_seq maps to AVI frame index; video t=(capture_seq-1)/25",
        "warning": "Scene video is the original capture; unsent frames were not seen by AWS.",
    }
    (OUT / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"dashboard data: {len(rows)} observations, {len(events)} events")


if __name__ == "__main__":
    main()
