"""Replay agent arithmetic on saved AWS observations, with no image inference."""

import hashlib
import json
from pathlib import Path

from care_vision.cloud.visibility_probe import VisibilityProbe

root = Path(__file__).resolve().parents[1]
out = root / "artifacts/visibility-activation-20260927"
out.mkdir(exist_ok=True)
new = root / "artifacts/new-chair-validation-20260927/trial-1/aws-evaluation/result.json"
old = root / "artifacts/occlusion-command-loop-20260927/trial-2/result.json"
plan = root / "artifacts/occlusion-continuity-20260927/verified-v5/plan.json"
sources = [new, old, plan]
jobs = []
for name, path in [("new-chair", new), ("previous-command-loop", old)]:
    r = json.loads(path.read_text())
    jobs.append(
        (
            name,
            [
                {"t": x["time"], "index": x["frame_index"], "people": x["people"]}
                for x in r["rows"]
                if x["people"] is not None
            ],
        )
    )
for j in json.loads(plan.read_text())["jobs"]:
    if j["id"].startswith("chair-"):
        continue
    jobs.append(
        (
            j["id"],
            [
                {"t": x["source_time_seconds"], "index": x["frame_index"], "people": x["persons"]}
                for x in j["detections"]
            ],
        )
    )
results = []
for name, rows in jobs:
    a = VisibilityProbe()
    history = [a.update(x["t"], x["index"], x["people"]) for x in rows]
    a.finish()
    results.append(
        {"id": name, "semantic_observations": len(rows), "probes": a.probes, "rows": history}
    )
    print(name, [{k: p[k] for k in ("start", "state", "ended_at")} for p in a.probes])
newprobes = results[0]["probes"]
assert len(newprobes) == 2
assert 50 <= newprobes[0]["start"] < 55
assert newprobes[0]["state"] == "POSTURE_CHANGE_CLEARED"
assert 80 <= newprobes[1]["start"] < 82
assert newprobes[1]["state"] == "UNRESOLVED"
assert all(
    not p["incident_confirmed"] and not p["recovery_observed"] for j in results for p in j["probes"]
)
report = {
    "method": "Arithmetic replay of saved AWS observations; no camera, inference or network",
    "scope": "Activation only. Earlier probe can change later transmitted pixels and detections; "
    "this replay does not simulate those new observations or confirm a fall.",
    "development_case": True,
    "requires_new_graviton_image_evaluation": True,
    "sources": {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    },
    "cloud_source_hashes": {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.glob("src/care_vision/cloud/*.py"))
    },
    "jobs": results,
}
(out / "activation-replay.json").write_text(json.dumps(report, indent=2) + "\n")
