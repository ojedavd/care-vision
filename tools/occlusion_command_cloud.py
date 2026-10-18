"""Bounded experimental command-loop worker over SSH; no production service/alerts."""

import hashlib
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

from care_vision.cloud.nanodet import NanoDetPersonDetector
from care_vision.cloud.occlusion_fragments import OcclusionFragments
from care_vision.cloud.occlusion_roi import incident_roi
from care_vision.cloud.person import observe_people
from care_vision.cloud.tracking import PersonTracker
from care_vision.cloud.yolox import YoloXPersonDetector
from care_vision.common.contracts import envelope, uid, unpack_frame, utc, validate
from care_vision.common.environment import runtime_info

base = Path(sys.argv[1])
plan = json.loads((base / "plan.json").read_text())
cv2.setNumThreads(1)
assert hashlib.sha256(Path(plan["model"]).read_bytes()).hexdigest() == plan["model_sha256"]
detector = NanoDetPersonDetector(plan["model"])
verifier = None
if plan.get("recovery_verifier"):
    config = plan["recovery_verifier"]
    assert hashlib.sha256(Path(config["model"]).read_bytes()).hexdigest() == config["sha256"]
    verifier = YoloXPersonDetector(
        config["model"], input_size=416, score_threshold=0.4, nms_threshold=0.3
    )
persons = PersonTracker()
vision = OcclusionFragments(visibility_probes=plan.get("visibility_probes", False))
rows = []
commands = []
acks = []
pending = {}
seq = 0
rate = 2
roi = None
semantic_bucket = None
expected_rate_command = expected_visual_command = None
runtime = runtime_info(3)
inferences = []


def reply(data):
    print(json.dumps(data), flush=True)


while True:
    prefix = sys.stdin.buffer.read(4)
    if not prefix:
        break
    size = struct.unpack("!I", prefix)[0]
    if not 0 < size < 4 * 1024 * 1024:
        raise ValueError("invalid transport size")
    payload = sys.stdin.buffer.read(size)
    if payload.startswith(b"{"):
        message = json.loads(payload)
        if message.get("control") == "finish":
            break
        validate(message)
        command = pending.pop(message["command_id"])
        assert message["result"] in {"applied", "duplicate"}
        if command["type"] == "REQUEST_ROI":
            assert message["active_context_fps"] == 2
        if command["type"] == "SET_TRANSMIT_FPS":
            assert message["active_transmit_fps"] == command["transmit_fps"]
            expected_rate_command = command["command_id"]
        else:
            expected_visual_command = command["command_id"]
        acks.append(message)
        reply({"acknowledged": message["command_id"]})
        continue
    header, jpeg = unpack_frame(payload)
    assert not pending, "new evidence arrived before command acknowledgment"
    assert header["applied_command_id"] == expected_rate_command
    assert header["applied_visual_command_id"] == expected_visual_command
    identity = {k: header[k] for k in ("edge_id", "boot_id", "session_id")}
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (header["height"], header["width"])
    visible = None
    if header["view_kind"] == "roi":
        x, y, w, h = [header["roi_" + k] for k in ("x", "y", "width", "height")]
        scale = min(header["width"] / w, header["height"] / h)
        rw, rh = round(w * scale), round(h * scale)
        left = (header["width"] - rw) // 2
        top = (header["height"] - rh) // 2
        crop = cv2.resize(image[top : top + rh, left : left + rw], (w, h))
        canvas = np.zeros((header["source_height"], header["source_width"], 3), np.uint8)
        canvas[y : y + h, x : x + w] = crop
        visible = [x, y, w, h]
    else:
        canvas = image
    t = header["captured_mono_ns"] / 1e9
    bucket = int(t * 2)
    people = None
    semantic_backend = None
    if header["view_kind"] == "full_frame" and bucket != semantic_bucket:
        semantic_bucket = bucket
        recovery_active = any(
            not g["ledger"].terminal and g["ledger"].bound_track is not None
            for g in vision.groups
        )
        selected = verifier if verifier is not None and recovery_active else detector
        detection = observe_people(selected, canvas, header["frame_id"])
        assert detection["person_detection_status"] == "ok"
        semantic_backend = detection["person_detector_backend"]
        inferences.append(
            {
                "frame_id": header["frame_id"],
                "time": t,
                "backend": semantic_backend,
                "milliseconds": detection["person_detection_ms"],
                "recovery_verification_active": recovery_active,
            }
        )
        people = persons.update(detection, header["captured_mono_ns"], "network-replay")["persons"]
    state, _, _ = vision.update(canvas, t, header["capture_seq"] - 1, people, visible_roi=visible)
    state.update(header=header, people=people, semantic_backend=semantic_backend)
    rows.append(state)
    active = [g for g in vision.groups if not g["ledger"].terminal]
    unresolved = any(g["ledger"].state in {"UNRESOLVED", "NO_LOW_EVIDENCE"} for g in vision.groups)
    desired_rate = 20 if active or unresolved else 2
    candidates = [g for g in active if g["ledger"].bound_track is None]
    desired_roi = (
        incident_roi(np.array(candidates[0]["ledger"].reference_bbox) * 2, canvas.shape)
        if candidates
        else None
    )
    outgoing = []

    def command(kind, reason, **fields):
        global seq
        seq += 1
        c = envelope(
            kind,
            identity,
            command_id=uid(),
            command_seq=seq,
            issued_at_utc=utc(),
            caused_by_observation_id=uid(),
            based_on_frame_id=header["frame_id"],
            valid_until_edge_mono_ns=header["captured_mono_ns"] + 5_000_000_000,
            reason=reason,
            **fields,
        )
        validate(c)
        pending[c["command_id"]] = c
        commands.append(c)
        outgoing.append(c)

    if desired_rate != rate:
        command("SET_TRANSMIT_FPS", "incident_evidence_policy", transmit_fps=desired_rate)
        rate = desired_rate
    if desired_roi != roi:
        if desired_roi:
            x, y, w, h = desired_roi
            command(
                "REQUEST_ROI",
                "occlusion_with_periodic_context",
                source_width=1280,
                source_height=720,
                roi_x=x,
                roi_y=y,
                roi_width=w,
                roi_height=h,
                output_width=1280,
                output_height=720,
                context_fps=2,
            )
        else:
            command(
                "REQUEST_FULL_FRAME",
                "verify_recovery_or_missing_evidence",
                output_width=1280,
                output_height=720,
            )
        roi = desired_roi
    reply({"state": state, "commands": outgoing})
report = {
    "runtime": runtime,
    "rows": rows,
    "commands": commands,
    "acks": acks,
    "episodes": [g["ledger"].finish() for g in vision.groups],
    "visibility_probes": vision.visibility.finish() if vision.visibility else [],
    "inferences": inferences,
}
(base / "result.json").write_text(json.dumps(report) + "\n")
reply({"finished": True, "episodes": report["episodes"], "frames": len(rows)})
