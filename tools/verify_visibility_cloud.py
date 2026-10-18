"""Verify recorded transport and report development outcome without assuming success."""

import argparse
import hashlib
import json
from pathlib import Path

from care_vision.common.contracts import validate

root = Path(__file__).resolve().parents[1]
capture = root / "artifacts/new-chair-validation-20260927/trial-1"
parser = argparse.ArgumentParser()
parser.add_argument("--evaluation", default="visibility-evaluation")
parser.add_argument("--controls", default="aws-controls")
args = parser.parse_args()
base = capture / args.evaluation
r = json.loads((base / "result.json").read_text())
e = json.loads((base / "edge.json").read_text())
plan = json.loads((base / "plan.json").read_text())
assert r["runtime"]["machine"] == "aarch64"
assert r["runtime"]["opencv_version"].startswith("5.")
assert r["runtime"]["cool"]["verified"]
assert len(r["rows"]) == len(e["rows"])
assert len(r["commands"]) == len(r["acks"]) == len(e["acks"])
assert sum(x["header"]["payload_bytes"] for x in r["rows"]) == e["jpeg_bytes"]
headers = {x["header"]["frame_id"]: x["header"] for x in r["rows"]}
acks = {a["command_id"]: a for a in r["acks"]}
for c in r["commands"]:
    validate(c)
    a = acks[c["command_id"]]
    validate(a)
    assert a["result"] == "applied"
    assert a["command_type"] == c["type"]
    assert c["based_on_frame_id"] in headers
    assert a["applied_at_edge_mono_ns"] == headers[c["based_on_frame_id"]]["captured_mono_ns"]
for x, y in zip(r["rows"], e["rows"]):
    assert x["header"] == y["header"]
    validate(x["header"])
assert all(not x["hidden_pose_observed"] and not x["alerts_enabled"] for x in r["episodes"])
# These brackets are source-only approximate review, not precise latency labels.
candidates = [x for x in r["episodes"] if x["candidate_at"] is not None]
normal_candidates = [x for x in candidates if x["candidate_at"] < 80]
fall_candidates = [x for x in candidates if 80 <= x["candidate_at"] <= 104]
recoveries = [
    event for x in fall_candidates for event in x["events"] if event["state"] == "RECOVERY_OBSERVED"
]
recovery_matches = [x for x in recoveries if x["time"] >= 104]
inference_summary = {}
if plan.get("recovery_verifier"):
    inferences = {x["frame_id"]: x for x in r["inferences"]}
    previous_episodes = []
    terminal = {"RECOVERY_OBSERVED", "UNRESOLVED", "NO_LOW_EVIDENCE", "POSTURE_CHANGE_CLEARED"}
    for row in r["rows"]:
        call = inferences.get(row["header"]["frame_id"])
        if call:
            active_bound = any(
                ep["bound_track"] and ep["state"] not in terminal for ep in previous_episodes
            )
            assert call["recovery_verification_active"] == bool(active_bound)
            assert row["header"]["view_kind"] == "full_frame"
            assert row["semantic_backend"] == call["backend"]
            if active_bound:
                assert call["backend"].startswith(
                    "opencv-dnn-yolox-416x416:" + plan["recovery_verifier"]["sha256"][:12]
                )
            else:
                assert call["backend"].startswith("opencv-dnn-nanodet-")
            summary = inference_summary.setdefault(call["backend"], {"calls": 0, "total_ms": 0})
            summary["calls"] += 1
            summary["total_ms"] += call["milliseconds"]
        else:
            assert row["people"] is None
        previous_episodes = row["episodes"]
    for ep in fall_candidates:
        if ep["state"] == "RECOVERY_OBSERVED":
            recovery = next(ev["time"] for ev in ep["events"] if ev["state"] == ep["state"])
            assert recovery - ep["upright_since"] + 1e-9 >= 2
            observed = [
                row for row in r["rows"]
                if ep["upright_since"] <= row["time"] <= recovery and row["people"] is not None
            ]
            assert len(observed) >= 3
            for row in observed:
                person = next(p for p in row["people"] if p["track_id"] == ep["bound_track"])
                reference = next(
                    e["reference_bbox"] for e in row["episodes"]
                    if e["source_track"] == ep["source_track"]
                )
                assert person["track_state"] == "confirmed"
                assert person["bbox"][2] / person["bbox"][3] <= 0.8
                assert person["bbox"][3] >= 0.85 * reference[3] * 2
            assert all(b["time"] - a["time"] <= 0.8 for a, b in zip(observed, observed[1:]))
status = (
    "CASE_PASSED_DEVELOPMENT"
    if fall_candidates and recovery_matches and not normal_candidates
    else "CASE_NOT_RESOLVED"
)
controls = json.loads(
    (root / "artifacts/visibility-activation-20260927" / args.controls / "result.json").read_text()
)
control_summary = [
    {
        "id": j["id"],
        "frames": len(j["rows"]),
        "candidates": sum(x["candidate_at"] is not None for x in j["episodes"]),
        "recoveries": sum(x["state"] == "RECOVERY_OBSERVED" for x in j["episodes"]),
        "terminal_states": [x["state"] for x in j["episodes"]],
    }
    for j in controls["jobs"]
]
report = {
    "status": status,
    "transport_verified": True,
    "frames": len(r["rows"]),
    "roi_frames": sum(x["header"]["view_kind"] == "roi" for x in r["rows"]),
    "commands_applied": len(r["acks"]),
    "jpeg_bytes": e["jpeg_bytes"],
    "application_edge_to_cloud_bytes": e["framed_edge_to_cloud_bytes"],
    "wall_seconds": e["wall_seconds"],
    "episodes": r["episodes"],
    "normal_activity_candidates": len(normal_candidates),
    "review_bracket_candidates": len(fall_candidates),
    "recovery_events": recoveries,
    "inference_summary": inference_summary,
    "controls": control_summary,
    "code_unchanged_since_deployment": all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
        for name, sha in plan["files"].items()
    ),
    "limits": [
        "Development/regression case, not held-out validation",
        "Sequential replay, not live measured capture-to-alert latency",
        "Supplementary controls use full frames and cached AWS semantic warmup, not ROI loop",
        "Controlled lowering, not a real fall; hidden pose unobserved",
    ],
}
(base / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ["episodes", "limits"]}, indent=2))
