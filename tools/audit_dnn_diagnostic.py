"""Audit diagnostic timings, output agreement and observed CPU thread activity."""

import argparse
import json
import statistics
from pathlib import Path

from audit_cv_distributions import iou
from benchmark_cv_distributions import save, sha
from diagnose_dnn_distribution import CASES


def compare(reference, candidate):
    differences, overlaps, deltas = [], [], []
    if [r["input"] for r in reference] != [r["input"] for r in candidate]:
        return {"passed": False, "reason": "different_inputs"}
    for a, b in zip(reference, candidate, strict=True):
        remaining = list(b["detections"])
        faults = []
        for detection in a["detections"]:
            if not remaining:
                faults.append("missing")
                continue
            match = max(remaining, key=lambda p: iou(p["bbox"], detection["bbox"]))
            remaining.remove(match)
            overlap = iou(match["bbox"], detection["bbox"])
            delta = abs(match["confidence"] - detection["confidence"])
            overlaps.append(overlap)
            deltas.append(delta)
            if overlap < 0.99 or delta > 0.001:
                faults.append("outside_tolerance")
        if remaining:
            faults.append("extra")
        if faults:
            differences.append({"input": a["input"], "faults": faults})
    return {
        "passed": not differences,
        "different_frames": differences,
        "min_iou": min(overlaps, default=None),
        "max_confidence_delta": max(deltas, default=None),
    }


def audit(folder, prior):
    plan = json.loads((folder / "plan.json").read_text())
    prior_plan = json.loads((prior / "plan.json").read_text())
    runs = [json.loads((folder / (label + ".json")).read_text()) for label, *_ in CASES]
    summaries = []
    for r in runs:
        stages = {
            key: statistics.median(row[key] for row in r["rows"])
            for key in (
                "decode_ms",
                "preprocess_ms",
                "set_input_ms",
                "forward_ms",
                "post_and_other_ms",
                "detect_ms",
            )
        }
        cpu = sorted(r["task_cpu_seconds"].values(), reverse=True)
        summaries.append(
            {
                "label": r["label"],
                "frames": len(r["rows"]),
                "median_ms": stages,
                "wall_seconds": r["wall_seconds"],
                "cpu_seconds": r["cpu_seconds"],
                "average_cpu_cores": r["cpu_seconds"] / r["wall_seconds"],
                "threads_with_over_100ms_cpu": sum(t > 0.1 for t in cpu),
                "task_cpu_seconds_descending": cpu,
                "libraries": r["libraries_before"],
                "profile_error": r["profile"].get("error"),
                "profile_mode_before": r["profile"].get("mode_before"),
            }
        )
    parity = [{"label": r["label"], **compare(runs[0]["rows"], r["rows"])} for r in runs[1:]]
    source = Path(__file__).resolve().parents[1]
    checks = {
        "requested_library_caps_verified": all(
            lib["reported_threads"] and all(v == 1 for v in lib["reported_threads"].values())
            for r in runs if r["blas_cap_requested"]
            for field in ("libraries_before", "libraries_after") for lib in r[field]
        ),
        "all_inputs_processed": all(
            [row["input"] for row in r["rows"]] == [f["path"] for f in prior_plan["frames"]]
            for r in runs
        ),
        "same_python_numpy": all(
            all(
                r["runtime"][k] == runs[0]["runtime"][k]
                for k in ("python_sha256", "numpy_native_sha256")
            )
            for r in runs
        ),
        "same_instance": all(r["runtime"]["ec2"] == runs[0]["runtime"]["ec2"] for r in runs),
        "cv_threads8": all(r["runtime"]["opencv_threads"] == 8 for r in runs),
        "cool_verified": all(r["runtime"]["cool"]["verified"] for r in runs if r["mode"] == "cool"),
        "same_cool_binary": len(
            {r["runtime"]["cool"]["native_sha256"] for r in runs if r["mode"] == "cool"}
        )
        == 1,
        "first_preprocessing_blob_equal": len({r["preprocess_blob_sha256"] for r in runs}) == 1,
        "wrapping_preserves_first_frame": all(r["wrapper_parity_first_frame"] for r in runs),
        "profile_disabled_while_timing_new_engine": all(
            r["profile"].get("mode_before") == 0
            for r in runs
            if r["mode"] == "cool" and r["engine_requested"] != "classic"
        ),
        "frozen_code_unchanged": all(
            sha(source / p) == h for p, h in prior_plan["code_sha256"].items()
        ),
        "diagnostic_code_unchanged": sha(source / "tools/diagnose_dnn_distribution.py")
        == plan["worker_sha256"],
        "prior_package_unchanged": sha(prior / "package.tar.gz") == plan["prior_package_sha256"],
    }
    deployment = json.loads((folder / "deployment.json").read_text())
    checks["cleanup"] = all(
        deployment[k] for k in ("complete", "workspace_removed", "existing_states_unchanged")
    )
    result = {
        "measurement_checks_passed": all(checks.values()),
        "checks": checks,
        "output_tolerance_passed": all(r["passed"] for r in parity),
        "runs": summaries,
        "output_comparisons": parity,
        "scope": "Detector-only stage diagnosis; one run per condition. "
        "No production changes, tracking/controller validation or per-kernel proof.",
    }
    save(folder / "assessment.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prior", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.folder, args.prior), indent=2))
