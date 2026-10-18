"""Build an offline V4-style dashboard from the measured S-only WebSocket replay."""

import hashlib
import json
import re
import shutil
from pathlib import Path

import cv2

from care_vision.edge.buffer import encode

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/s-only-websocket-20261004/lateral-retry"
OUT = ROOT / "artifacts/s-only-websocket-20261004/dashboard"
REFERENCE = ROOT / "artifacts/visibility-recovery-v4-dashboard-20260927/index.html"
REFERENCE_VIDEO = ROOT / "artifacts/posture-websocket-20261004/dashboard/video.mp4"
REFERENCE_PROVENANCE = ROOT / "artifacts/posture-websocket-20261004/dashboard/provenance.json"
FPS = 25


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    audit = json.loads((RUN / "s-only-audit.json").read_text())
    assert audit["passed"] and audit["sent"] == audit["processed"]
    cloud, edge = records(RUN / "cloud.jsonl"), records(RUN / "edge.jsonl")
    sent = [r for r in edge if r["event"] == "frame_sent"]
    obs = [r for r in cloud if r["event"] == "observation"]
    actions = [r for r in cloud if r["event"] == "agent_action"]
    outcomes = [r for r in cloud if r["event"] == "loss_verification_outcome"]
    transitions = [r for r in cloud if r["event"] == "agent_state_transition"]
    assert [r["frame_id"] for r in sent] == [r["frame_id"] for r in obs]
    assert len(actions) == len(outcomes) * 2 == 2
    assert all((r["width"], r["height"]) == (1280, 720) for r in obs)
    assert all("yolox-s" in r["person_detector_backend"] for r in obs)
    assert all(r["view_kind"] == "full_frame" for r in obs)
    acked = {
        r["command_id"] for r in cloud if r["event"] == "ack_received" and r["result"] == "applied"
    }
    assert len(acked) == 2 and all(r["command_id"] in acked for r in actions)

    manifest = json.loads(
        (ROOT / "artifacts/tapo-heldout-20261004/controlled-lateral-1/manifest.json").read_text()
    )
    video = RUN / "demo.avi"
    source_sha = sha(video)
    reference = json.loads(REFERENCE_PROVENANCE.read_text())
    assert source_sha == reference["video_source_sha256"]
    assert source_sha == manifest["video_sha256"]
    OUT.mkdir(parents=True, exist_ok=True)
    received_dir = OUT / "received"
    received_dir.mkdir(exist_ok=True)
    shutil.copyfile(REFERENCE_VIDEO, OUT / "video.mp4")

    selected = {row["capture_seq"] - 1: (row, observation) for row, observation in zip(sent, obs)}
    assert len(selected) == len(sent)
    cap = cv2.VideoCapture(str(video))
    saved = {}
    for index in range(manifest["decoded_frames"]):
        ok, frame = cap.read()
        if not ok or frame.shape[:2] != (1080, 1920):
            raise ValueError(f"source decode failed at frame {index}")
        if index not in selected:
            continue
        edge_row, _ = selected[index]
        jpeg = encode(frame, 1280, 720)
        if len(jpeg) != edge_row["payload_bytes"]:
            raise ValueError(f"JPEG byte length mismatch at capture {index + 1}")
        image_name = f"{index + 1:06d}.jpg"
        (received_dir / image_name).write_bytes(jpeg)
        saved[index + 1] = (image_name, hashlib.sha256(jpeg).hexdigest())
    cap.release()
    if len(saved) != len(sent):
        raise ValueError("not all transmitted frames were reconstructed")

    cumulative = 0
    rows = []
    for edge_row, observation in zip(sent, obs):
        seq = observation["capture_seq"]
        cumulative += edge_row["payload_bytes"]
        image_name, image_sha = saved[seq]
        rows.append(
            {
                "t": round((seq - 1) / FPS, 3),
                "seq": seq,
                "image": image_name,
                "sha256": image_sha,
                "fps": observation["target_transmit_fps"],
                "bytes": edge_row["payload_bytes"],
                "total": cumulative,
                "bbox": observation.get("primary_person_bbox"),
                "detected": observation.get("person_detected", False),
                "confirmed": observation.get("confirmed_person_detected", False),
                "track": observation.get("primary_track_id", "").split(":")[-1]
                if observation.get("primary_track_id")
                else None,
                "orientation": observation.get("primary_orientation"),
                "people": [
                    {
                        "bbox": p["bbox"],
                        "confidence": round(p["confidence"], 3),
                        "track": p.get("track_id", "").split(":")[-1],
                        "state": p.get("track_state"),
                    }
                    for p in observation.get("persons", [])
                ],
                "processingMs": round(observation["processing_ms"], 2),
                "queueMs": round(observation["cloud_queue_wait_ms"], 3),
                "valid": observation["valid"],
            }
        )

    events = []
    for action in actions:
        events.append(
            {
                "t": round((int(action["based_on_frame_id"].rsplit(":", 1)[-1]) - 1) / FPS, 3),
                "kind": "command",
                "seq": action["command_seq"],
                "fps": action["transmit_fps"],
                "ack": True,
                "reason": action["reason"],
            }
        )
    for outcome in outcomes:
        events.append(
            {
                "t": round((int(outcome["frame_id"].rsplit(":", 1)[-1]) - 1) / FPS, 3),
                "kind": "outcome",
                "outcome": outcome["outcome"],
                "incident": outcome["incident_status"],
                "horizontalFrames": outcome["horizontal_frames"],
                "verticalFrames": outcome["vertical_frames"],
            }
        )
    for transition in transitions:
        events.append(
            {
                "t": round((int(transition["frame_id"].rsplit(":", 1)[-1]) - 1) / FPS, 3),
                "kind": "state",
                "from": transition["from_state"],
                "to": transition["to_state"],
                "reason": transition["reason"],
            }
        )
    events.sort(key=lambda row: (row["t"], {"state": 0, "command": 1, "outcome": 2}[row["kind"]]))
    duration = (manifest["decoded_frames"] - 1) / FPS
    data = {
        "rows": rows,
        "events": events,
        "duration": duration,
        "stats": audit,
        "source_sha256": source_sha,
    }
    (OUT / "data.js").write_text(
        "window.EVIDENCE = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    )
    css = re.search(r"<style>(.*?)</style>", REFERENCE.read_text(), re.S).group(1)
    css += "\n" + (OUT / "extra.css").read_text()
    (OUT / "style.css").write_text(css)
    provenance = {
        "source_video": str(video.relative_to(ROOT)),
        "source_sha256": source_sha,
        "video_mp4_sha256": sha(OUT / "video.mp4"),
        "source_fps": FPS,
        "decoded_frames": manifest["decoded_frames"],
        "observation_rows": len(rows),
        "commands_acknowledged": len(actions),
        "reconstructed_jpeg_lengths_matching_edge": len(saved),
        "reconstructed_jpeg_sha256": {str(k): v[1] for k, v in saved.items()},
        "alignment": "frame_id suffix = capture_seq; video time = (capture_seq - 1)/25",
        "limits": [
            "Full source video on the left includes frames never transmitted to AWS",
            "Right panel uses latest actually transmitted capture; no interpolated inference",
            "JPEG length matches Edge telemetry; original JPEG hashes were not logged",
            "One development replay; horizontal posture is not a fall or alert decision",
        ],
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"{len(rows)} reconstructed frames, {len(events)} events, {cumulative} JPEG bytes")


if __name__ == "__main__":
    main()
