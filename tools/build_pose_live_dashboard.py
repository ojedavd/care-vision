"""Present saved live-loop evidence; extract review images without running inference."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_trial(source, output, key):
    audit = json.loads((source / "assessment.json").read_text())
    strict = json.loads((source / "strict-report.json").read_text())
    scenario_passed = (audit.get("scoped_sit_return_scenario_passed", False)
                       or audit.get("scoped_seated_hold_scenario_passed", False))
    if not (strict["passed"] and audit["causal_loop_passed"] and scenario_passed):
        raise ValueError("Requires accepted trial; preserve failed trials separately")
    cloud, edge = records(source / "cloud.jsonl"), records(source / "edge.jsonl")
    captures = records(source / "demo.captures.jsonl")
    origin = captures[0]["captured_mono_ns"]

    def seconds(ns):
        return (ns - origin) / 1e9

    observations = [r for r in cloud if r["event"] == "observation"]
    sent = [r for r in edge if r["event"] == "frame_sent"]
    received = [r for r in cloud if r["event"] == "frame_received"]
    if not ([r["frame_id"] for r in sent] == [r["frame_id"] for r in received]
            == [r["frame_id"] for r in observations]):
        raise ValueError("Unmatched transmitted/received/processed frames")
    capture_by_seq = {r["capture_seq"]: r for r in captures}
    for row in observations:
        if capture_by_seq[row["capture_seq"]]["captured_mono_ns"] != row["captured_mono_ns"]:
            raise ValueError("Capture timestamps differ")
    (output / key).mkdir(mode=0o700)
    wanted = {r["capture_seq"] for r in observations}
    reader = cv2.VideoCapture(str(source / "demo.avi"))
    images = {}
    seq = 0
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            seq += 1
            if seq in wanted:
                name = f"{key}/{seq:06d}.jpg"
                if not cv2.imwrite(str(output / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    raise RuntimeError("Review image write failed")
                images[seq] = name
    finally:
        reader.release()
    if len(images) != len(wanted) or seq != len(captures):
        raise ValueError("Video decode/count mismatch")
    shadows = {r["observation_id"]: r for r in cloud if r["event"] == "pose_shadow"}
    states = [r for r in cloud if r["event"] == "agent_state_transition"]
    episodes = [r for r in cloud if r["event"] == "incident_episode"]
    statuses = [r for r in edge if r["event"] == "edge_status"]
    rows, total_bytes = [], 0
    for row, frame in zip(observations, sent, strict=True):
        total_bytes += frame["payload_bytes"]
        shadow = shadows[row["observation_id"]]
        state = [s for s in states if s["captured_mono_ns"] <= row["captured_mono_ns"]]
        episode = [e for e in episodes if e.get("captured_mono_ns", float("inf"))
                   <= row["captured_mono_ns"]]
        status = [s for s in statuses if s["local_mono_ns"] <= row["captured_mono_ns"]]
        latest_status = status[-1] if status else {}
        rows.append({
            "t": seconds(row["captured_mono_ns"]), "seq": row["capture_seq"],
            "image": images[row["capture_seq"]], "id": row["observation_id"],
            "state": state[-1]["to_state"] if state else "NORMAL",
            "episode": episode[-1]["incident_status"] if episode else "not_opened",
            "fps": row["target_transmit_fps"], "view": row["view_kind"],
            "width": row["width"], "height": row["height"], "roi": row.get("source_roi"),
            "bbox": row.get("primary_tracked_source_bbox"),
            "track": row.get("primary_track_id"), "valid": row["valid"],
            "invalidReason": row["invalid_reason"], "confidence": row["primary_person_confidence"],
            "processing": row["processing_ms"], "bytes": total_bytes,
            "poseStatus": shadow["pose_status"], "poseReason": shadow["reason"],
            "posePattern": shadow.get("seated_pose_candidate", False),
            "points": shadow.get("points") if shadow["pose_status"] == "ok" else None,
            "network": latest_status.get("network_quality", "UNKNOWN"),
            "rtt": latest_status.get("network_rtt_ms"),
        })
    by_id = {r["observation_id"]: r for r in observations}
    commands = []
    for action in (r for r in cloud if r["event"] == "agent_action"):
        ack = next(r for r in edge if r["event"] == "command_ack"
                   and r["command_id"] == action["command_id"])
        if ack["result"] != "applied":
            raise ValueError("Non-applied command")
        cause = by_id[action["caused_by_observation_id"]]
        commands.append({"type": action["type"], "fps": action.get("transmit_fps"),
                         "causeSeq": cause["capture_seq"],
                         "causeT": seconds(cause["captured_mono_ns"]),
                         "edgeAppliedT": seconds(ack["applied_at_edge_mono_ns"]),
                         "effectiveSeq": ack["effective_from_capture_seq"], "ack": ack["result"]})
    cycles = [r for r in cloud if r["event"] == "pose_observation_cycle"]
    roi = next(r for r in observations if r["view_kind"] == "roi")
    completed = next(r for r in cycles if r["stage"] == "cycle_complete")
    opened = next(r for r in episodes if r["reason"] == "episode_opened")
    closed = next((r for r in episodes
                   if r["reason"] == "same_track_sustained_reference_posture"), None)
    if closed is None and not audit.get("episode_open_at_end"):
        raise ValueError("Trial has neither verified closure nor an unresolved end")
    def seq_from_id(row):
        return int(row["frame_id"].rsplit(":", 1)[1])
    moments = [
        {"seq": rows[0]["seq"], "label": "Inicio"},
        {"seq": seq_from_id(opened), "label": "Episodio abierto"},
        {"seq": next(c["causeSeq"] for c in commands if c["type"] == "REQUEST_ROI"),
         "label": "Pose solicita ROI"},
        {"seq": roi["capture_seq"], "label": "Primera ROI nueva"},
        {"seq": next(c["causeSeq"] for c in commands if c["type"] == "REQUEST_FULL_FRAME"),
         "label": "Solicita retorno"},
        {"seq": completed["capture_seq"], "label": "Full-frame nuevo · ciclo completo"},
        {"seq": seq_from_id(closed) if closed else rows[-1]["seq"],
         "label": ("De pie · episodio cerrado" if closed
                   else "Última observación · episodio pendiente")},
    ]
    previous = source.parent / "sit-return-live-1"
    old = json.loads((previous / "strict-report.json").read_text())
    payload = {"key": key, "source": source.name,
               "title": "Volver de pie" if closed else "Permanecer sentado",
               "outcome": "Cierra tras volver de pie" if closed else "Conserva episodio pendiente",
               "rows": rows, "commands": commands, "moments": moments, "audit": {
        "counts": audit["sent_received_processed"], "captured": len(captures),
        "captureFps": strict["report"]["capture_fps_median"],
        "roiCount": audit["roi_observations"], "invalid": sum(not r["valid"] for r in rows),
        "poseCount": audit["pose_inferences_ok"],
        "closedT": seconds(closed["captured_mono_ns"]) if closed else None,
        "roiReturnDelay": audit["roi_ack_to_return_request_seconds"],
        "restoredDelay": audit["return_ack_to_valid_full_frame_seconds"],
        "previousFailure": old.get("reason"), "strictPassed": strict["passed"],
    }}
    sources = ["cloud.jsonl", "edge.jsonl", "demo.avi", "demo.captures.jsonl", "assessment.json",
               "strict-report.json", "activity-review.json", "verification.json"]
    provenance = {
        "scope": "Saved physical trial; no inference, camera, AWS or recomputed decisions",
        "images": "Full-frame review images decoded from recoded AVI, not received AWS JPEG bytes",
        "overlay": "Recorded source-coordinate box/ROI/pose for this observation; not ground truth",
        "clock": "Playback uses capture timestamps. Decisions associated with causal capture; "
                 "ACK application uses Edge clock. Cloud delays come from same-clock audit only.",
        "source_sha256": {name: digest(source / name) for name in sources},
        "images_sha256": {name: digest(output / name) for name in images.values()},
        "previous_failure_sha256": digest(previous / "strict-report.json"),
        "frame_count": len(rows), "commands": len(commands), "decoded_frames": seq,
    }
    return payload, provenance


def build(source, output, hold_source=None):
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    trials, provenance = [], {}
    inputs = [(source, "return")]
    if hold_source:
        inputs.append((hold_source, "hold"))
    for folder, key in inputs:
        trial, trace = build_trial(folder, output, key)
        trials.append(trial)
        provenance[key] = trace
    (output / "data.js").write_text("window.POSE_LIVE=" + json.dumps(
        {"trials": trials}, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).replace("<", "\\u003c") + ";\n")
    (output / "index.html").write_bytes(Path(__file__).with_name("pose_live_dashboard.html")
                                        .read_bytes())
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(output), "trials": len(trials),
                      "observations": sum(len(t["rows"]) for t in trials)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hold-source", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    build(args.source.resolve(), args.output.resolve(),
          args.hold_source.resolve() if args.hold_source else None)
