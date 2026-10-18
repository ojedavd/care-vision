"""Explain rate windows and missing motion context from saved logs, without inference."""

import argparse
import hashlib
import json
from collections import Counter, deque
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def windows(rows, target, duration=5):
    """Same boundaries/tolerance as rate_windows, reporting every window failure."""
    if len(rows) < 2:
        return {"windows": [], "enough_data": False, "times_increase": True}
    times = [r["local_mono_ns"] for r in rows]
    start = times[0] / 1e9 + 0.25
    end = times[-1] / 1e9
    result = []
    while start + duration <= end:
        members = [r for r in rows if start <= r["local_mono_ns"] / 1e9 < start + duration]
        rate = len(members) / duration
        result.append({
            "start_seconds_since_first_event": start - times[0] / 1e9,
            "frames": len(members), "fps": rate,
            "within_original_tolerance": abs(rate - target) <= target * 0.1 + 1e-6,
            "capture_seqs": [r["capture_seq"] for r in members],
        })
        start += duration
    max_burst = max(sum(t <= later < t + 1_000_000_000 for later in times[index:])
                    for index, t in enumerate(times))
    return {"windows": result, "enough_data": bool(result),
            "maximum_events_in_one_second": max_burst,
            "burst_guard_within_original_limit": max_burst <= target + 2,
            "times_increase": all(b > a for a, b in zip(times, times[1:]))}


def temporal_context(observations):
    """Reconstruct only the 400–800 ms reference rule, without model/image replay."""
    history = deque(maxlen=64)
    identity = None
    epoch_start = None
    invalid = []
    mismatches = []
    epoch_index = -1
    for row in observations:
        current = (
            row["view_kind"], row.get("applied_visual_command_id"),
            row["width"], row["height"], row["source_width"], row["source_height"],
            tuple(row.get("source_roi") or []),
        )
        captured = row["captured_mono_ns"]
        if current != identity:
            history.clear()
            identity = current
            epoch_start = captured
            epoch_index += 1
        while history and captured - history[0][0] > 1_000_000_000:
            history.popleft()
        candidates = [item for item in history
                      if 400_000_000 <= captured - item[0] <= 800_000_000]
        chosen = min(candidates, key=lambda item: abs(captured - item[0] - 500_000_000)) \
            if candidates else None
        expected = chosen[1] if chosen else None
        if row.get("reference_frame_id") != expected:
            mismatches.append({"capture_seq": row["capture_seq"], "expected": expected,
                               "actual": row.get("reference_frame_id")})
        if row.get("invalid_reason") == "insufficient_temporal_context":
            if captured - epoch_start < 400_000_000:
                category = "startup_context" if epoch_index == 0 else "view_change_context"
            else:
                category = "same_view_no_reference_in_400_800ms"
            invalid.append({
                "capture_seq": row["capture_seq"], "view_kind": row["view_kind"],
                "category": category, "candidate_count": len(candidates),
                "preceding_same_view_intervals_ms": [
                    (captured - item[0]) / 1e6 for item in history
                ],
            })
        history.append((captured, row["frame_id"]))
    return {"missing_context": invalid,
            "categories": dict(Counter(r["category"] for r in invalid)),
            "reference_mismatches": mismatches,
            "all_missing_context_have_zero_candidates": all(
                r["candidate_count"] == 0 for r in invalid),
            "scope": "Timestamp/view metadata only; no pixels, motion scores or inference replay"}


def diagnose(folder):
    edge = read_rows(folder / "edge.jsonl")
    cloud = read_rows(folder / "cloud.jsonl")
    sent = [r for r in edge if r["event"] == "frame_sent"
            and r.get("evidence_kind", "live") == "live"]
    received = [r for r in cloud if r["event"] == "frame_received"
                and r.get("evidence_kind", "live") == "live"]
    observations = [r for r in cloud if r["event"] == "observation"
                    and r.get("evidence_kind", "live") == "live"]
    policies = []
    for policy in sorted({r["policy_seq"] for r in sent}):
        outbound = [r for r in sent if r["policy_seq"] == policy]
        inbound = [r for r in received if r["policy_seq"] == policy]
        target = outbound[0]["target_transmit_fps"]
        policies.append({"policy_seq": policy, "target_fps": target,
                         "send": windows(outbound, target),
                         "receive": windows(inbound, target)})
    by_id = {r["frame_id"]: r for r in received}
    linked = [r for r in sent if r["frame_id"] in by_id]
    first = linked[0]
    first_receive = by_id[first["frame_id"]]["local_mono_ns"]
    relative = [{
        "capture_seq": r["capture_seq"],
        "send_elapsed_seconds": (r["local_mono_ns"] - first["local_mono_ns"]) / 1e9,
        "receive_elapsed_seconds": (by_id[r["frame_id"]]["local_mono_ns"]
                                    - first_receive) / 1e9,
        "capture_to_send_ms": (r["local_mono_ns"] - r["captured_mono_ns"]) / 1e6,
    } for r in linked]
    return {
        "scope": "Read-only diagnosis of saved live evidence; no new camera or AWS work",
        "input_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                         for name in ("edge.jsonl", "cloud.jsonl")},
        "policies": policies, "temporal_context": temporal_context(observations),
        "relative_send_receive_intervals": relative,
        "clock_scope": "Elapsed intervals on each endpoint's own monotonic clock; "
                       "never absolute one-way delay or synchronized clocks",
        "transport_cause": "Not isolated by existing logs: socket completion and cloud "
                           "callback receipt do not identify the delayed component",
        "full_demo_passed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.folder)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output),
                      "missing_context_categories": report["temporal_context"]["categories"],
                      "reference_mismatches": len(report["temporal_context"]
                                                  ["reference_mismatches"])}))


if __name__ == "__main__":
    main()
