"""Build a local evidence viewer from observer results; never runs image inference."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def build(source, output):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((source / "replay-plan.json").read_text())
    config = json.loads((source / "replay-config.json").read_text())
    report = json.loads((source / "observer-replay/report.json").read_text())
    trials = []
    for run in plan["runs"]:
        folder = root / run["input_path"]
        frames = {r["capture_seq"]: r for r in json.loads((folder / "plan.json").read_text())[
            "frames"
        ] if r["run"] == run["run"]}
        log_path = source / "observer-replay" / (run["run"] + ".jsonl")
        expected = next(r for r in report["runs"] if r["run"] == run["run"])
        if hashlib.sha256(log_path.read_bytes()).hexdigest() != expected["log_sha256"]:
            raise ValueError("observer evidence changed")
        cloud = [json.loads(line) for line in (root / run["cloud_path"]).read_text().splitlines()]
        observations = {r["observation_id"]: r for r in cloud if r["event"] == "observation"}
        origin = next(iter(observations.values()))["captured_mono_ns"]
        rows = []
        for diagnostic in map(json.loads, log_path.read_text().splitlines()):
            if diagnostic["event"] != "pose_shadow":
                continue
            obs = observations[diagnostic["observation_id"]]
            frame = frames[obs["capture_seq"]]
            image = folder / frame["filename"]
            if hashlib.sha256(image.read_bytes()).hexdigest() != frame["jpeg_sha256"]:
                raise ValueError("dashboard JPEG differs")
            rows.append({
                **diagnostic,
                "seconds": (obs["captured_mono_ns"] - origin) / 1e9,
                "image": os.path.relpath(image, output),
                "bbox": obs["persons"][0]["tracked_bbox"] if len(obs["persons"]) == 1 else None,
                "fps": obs.get("target_transmit_fps"),
            })
        trials.append({"name": run["run"], "summary": expected, "rows": rows})
    output.mkdir(parents=True, exist_ok=False)
    data = json.dumps({"trials": trials, "hips": config["hips"]["hip_proximity_region"]},
                      separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    (output / "data.js").write_text("window.EVIDENCE=" + data + ";\n")
    template = Path(__file__).with_name("pose_shadow_dashboard.html")
    (output / "index.html").write_bytes(template.read_bytes())
    (output / "provenance.json").write_text(json.dumps({
        "scope": "Recorded frames, AWS observer replay; no live camera or inference here",
        "report_sha256": hashlib.sha256(
            (source / "observer-replay/report.json").read_bytes()
        ).hexdigest(),
        "trials": [r["name"] for r in trials], "images": sum(len(r["rows"]) for r in trials),
    }, indent=2) + "\n")
    print(f"Built {len(trials)} trials / {sum(len(r['rows']) for r in trials)} frames")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    build(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
