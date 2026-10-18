"""Build a portable viewer of frozen CLASSIC evidence, with no new CV inference."""

import argparse
import json
import shutil
from pathlib import Path

import cv2
from check_mvp_demo_profile import DEFAULT, ROOT, check, digest

BASE = ROOT / "artifacts/yolox-classic-integration-20261015/run-1"
TITLES = {
    "sit-return-live-2-pose": "ROI y retorno de pie",
    "sit-hold-live-1-pose": "ROI y permanencia sentado",
    "recovery-2-baseline": "Suelo y recuperación",
    "no-recovery-1-test20": "Permanencia y alerta de prueba",
    "no-recovery-1-baseline": "Permanencia sin alerta",
}


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest(records, field, time):
    eligible = [row for row in records if row.get(field) is not None and row[field] <= time]
    return eligible[-1] if eligible else {}


def build_case(case, assessment, output):
    label = case["label"]
    folder, source = BASE / label, ROOT / case["source_folder"]
    cloud, edge = records(folder / "cloud.jsonl"), records(folder / "edge.jsonl")
    captures = records(source / "demo.captures.jsonl")
    stamps = {r["capture_seq"]: r["captured_mono_ns"] for r in captures}
    mapping = {r["capture_seq"]: r for r in records(folder / "replay-captures.jsonl")}
    observed = [r for r in cloud if r["event"] == "observation"]
    received = [r for r in cloud if r["event"] == "frame_received"]
    sent = [r for r in edge if r["event"] == "frame_sent"]
    assert [r["frame_id"] for r in observed] == [r["frame_id"] for r in received]
    assert [r["frame_id"] for r in observed] == [r["frame_id"] for r in sent]
    assert len({r["frame_id"] for r in observed}) == len(observed)
    replay_start = mapping[captures[0]["capture_seq"]]["replay_captured_mono_ns"]
    original_start = captures[0]["captured_mono_ns"]
    for obs in observed:
        seq = obs["capture_seq"]
        assert obs["captured_mono_ns"] == mapping[seq]["replay_captured_mono_ns"]
        assert stamps[seq] == mapping[seq]["original_captured_mono_ns"]
        assert obs["captured_mono_ns"] - replay_start == stamps[seq] - original_start
        assert obs["person_detector_backend"].endswith(":engine-classic")

    def seconds(seq):
        return (stamps[seq] - original_start) / 1e9

    shadows = {r["observation_id"]: r for r in cloud if r["event"] == "pose_shadow"}
    states = [r for r in cloud if r["event"] == "agent_state_transition"]
    episodes = [r for r in cloud if r["event"] == "incident_episode"]
    statuses = [r for r in edge if r["event"] == "edge_status"]
    resources = [r for r in cloud if r["event"] == "process_resources"]
    rows, total_bytes = [], 0
    for obs, frame in zip(observed, sent, strict=True):
        shadow = shadows[obs["observation_id"]]
        state = latest(states, "captured_mono_ns", obs["captured_mono_ns"])
        episode = latest(episodes, "captured_mono_ns", obs["captured_mono_ns"])
        status = latest(statuses, "local_mono_ns", obs["captured_mono_ns"])
        resource = latest(resources, "local_mono_ns", obs["local_mono_ns"])
        total_bytes += frame["payload_bytes"]
        seq = obs["capture_seq"]
        rows.append({
            "seq": seq, "t": seconds(seq), "image": f"{label}/{seq:06d}.jpg",
            "id": obs["observation_id"], "state": state.get("to_state", "NORMAL"),
            "episode": episode.get("incident_status", "not_opened"),
            "episodeReason": episode.get("reason", "no_episode"),
            "fps": obs["target_transmit_fps"], "view": obs["view_kind"],
            "width": obs["width"], "height": obs["height"], "roi": obs.get("source_roi"),
            "bbox": obs.get("primary_tracked_source_bbox"), "valid": obs["valid"],
            "invalidReason": obs["invalid_reason"], "track": obs["primary_track_id"],
            "confidence": obs["primary_person_confidence"], "processing": obs["processing_ms"],
            "bytes": total_bytes, "poseStatus": shadow["pose_status"],
            "poseReason": shadow["reason"], "posePattern": shadow.get("seated_pose_candidate"),
            "points": shadow.get("points") if shadow["pose_status"] == "ok" else None,
            "network": status.get("network_quality", "Sin dato"),
            "pipeline": status.get("pipeline_quality", "Sin dato"),
            "rtt": status.get("network_rtt_ms"), "measuredFps": status.get("transmit_fps"),
            "statusAge": ((obs["captured_mono_ns"] - status["local_mono_ns"]) / 1e9
                          if status else None),
            "statusWindow": status.get("window_seconds"),
            "rss": resource.get("rss_bytes"),
            "resourceAge": ((obs["local_mono_ns"] - resource["local_mono_ns"]) / 1e9
                            if resource else None),
        })
    output.joinpath(label).mkdir()
    wanted = {r["seq"] for r in rows}
    reader = cv2.VideoCapture(str(source / "demo.avi"))
    decoded, images = 0, 0
    try:
        while True:
            ok, image = reader.read()
            if not ok:
                break
            decoded += 1
            if decoded in wanted:
                assert cv2.imwrite(str(output / label / f"{decoded:06d}.jpg"), image,
                                   [cv2.IMWRITE_JPEG_QUALITY, 90])
                images += 1
    finally:
        reader.release()
    assert decoded == len(captures) and images == len(rows)
    by_id = {r["observation_id"]: r for r in observed}
    edge_acks = {r["command_id"]: r for r in edge if r["event"] == "command_ack"}
    cloud_acks = {r["command_id"]: r for r in cloud if r["event"] == "ack_received"}
    commands = []
    for action in (r for r in cloud if r["event"] == "agent_action"):
        cid = action["command_id"]
        ack, cloud_ack = edge_acks[cid], cloud_acks[cid]
        assert ack["result"] == cloud_ack["result"] == "applied"
        cause = by_id[action["caused_by_observation_id"]]
        field = ("applied_command_id" if action["type"] == "SET_TRANSMIT_FPS"
                 else "applied_visual_command_id")
        fresh = next(r for r in observed if r.get(field) == cid and r["valid"]
                     and r["capture_seq"] >= ack["effective_from_capture_seq"])
        assert fresh["captured_mono_ns"] >= ack["applied_at_edge_mono_ns"]
        commands.append({
            "id": cid, "type": action["type"], "fps": action.get("transmit_fps"),
            "reason": action["reason"], "causeSeq": cause["capture_seq"],
            "causeT": seconds(cause["capture_seq"]),
            "appliedT": (ack["applied_at_edge_mono_ns"] - replay_start) / 1e9,
            "effectiveSeq": ack["effective_from_capture_seq"], "freshSeq": fresh["capture_seq"],
            "freshT": seconds(fresh["capture_seq"]), "ack": ack["result"],
            "ackToPerceptionMs": (fresh["local_mono_ns"] - cloud_ack["local_mono_ns"]) / 1e6,
        })
    moments = [{"seq": rows[0]["seq"], "label": "Inicio"}]
    for command in commands:
        if command["type"] in {"REQUEST_ROI", "REQUEST_FULL_FRAME"}:
            title = "Solicita ROI" if command["type"] == "REQUEST_ROI" else "Solicita retorno"
            fresh_title = ("ROI nueva válida" if command["type"] == "REQUEST_ROI"
                           else "Full-frame nuevo válido")
            moments.extend([{"seq": command["causeSeq"], "label": title},
                            {"seq": command["freshSeq"], "label": fresh_title}])
    names = {"episode_opened": "Abre episodio",
             "same_track_horizontal_observed": "Evidencia horizontal",
             "same_track_sustained_reference_posture": "Retorno de pie · cierre",
             "TRIGGER_ALERT_TEST": "Alerta de prueba"}
    for event in episodes:
        if event["reason"] in names and event.get("observation_id") in by_id:
            moments.append({"seq": by_id[event["observation_id"]]["capture_seq"],
                            "label": names[event["reason"]]})
    moments.append({"seq": rows[-1]["seq"], "label": "Final"})
    moments.sort(key=lambda r: r["seq"])
    return {
        "key": label, "title": TITLES[label], "rows": rows, "commands": commands,
        "moments": moments, "audit": assessment, "captured": decoded,
        "captureFps": (len(captures) - 1) * 1e9 / (captures[-1]["captured_mono_ns"]
                                                   - original_start),
        "source": case["source_folder"],
        "context": "Vista del 14 revisada" if case["context"] == "pose-current.json"
        else "Contexto de silla desconocido; pose/ROI bloqueadas",
    }


def build(output):
    assert check(DEFAULT)["passed"], "Frozen profile changed; review before presenting"
    profile = json.loads(DEFAULT.read_text())
    assessment = json.loads((BASE / "assessment.json").read_text())
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    cases = {c["label"]: c for c in profile["cases"]}
    audits = {r["source"] + "-" + r["mode"]: r for r in assessment["runs"]}
    trials = [build_case(cases[label], audits[label], output) for label in TITLES]
    (output / "data.js").write_text("window.CLASSIC_DEMO=" + json.dumps(
        {"trials": trials}, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).replace("<", "\\u003c") + ";\n")
    for source, name in [(Path(__file__).with_name("mvp_classic_dashboard.html"), "index.html"),
                         (BASE / "assessment.json", "assessment.json"),
                         (DEFAULT, "profile.json")]:
        shutil.copyfile(source, output / name)
    provenance = {
        "scope": "Presentation of five October15 replays; no camera/AWS/inference",
        "images": "Full source frames decoded from recoded AVI, not transmitted JPEG bytes",
        "clock": "Original capture_seq maps to rebased replay timestamps; offsets verified. "
                 "Results associated with causal capture, not decision emission time. "
                 "ACK-to-perception intervals use cloud clock only.",
        "observations": sum(len(t["rows"]) for t in trials),
        "builder_sha256": digest(Path(__file__)),
        "profile_sha256": digest(DEFAULT),
        "files_sha256": {str(p.relative_to(output)): digest(p)
                         for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(output), "observations": provenance["observations"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output.resolve())
