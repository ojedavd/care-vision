"""Audit sustained horizontal posture in saved 2→5→2 verification windows."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from care_vision.cloud.posture_evidence import PostureEvidence  # noqa: E402

SOURCES = {
    "live-tapo-2": "artifacts/tapo-live-loss-20261004/trial-2/cloud.jsonl",
    "websocket-lateral": "artifacts/loss-service-websocket-20261004/lateral/cloud.jsonl",
    "websocket-chair": "artifacts/loss-service-websocket-20261004/controlled1/cloud.jsonl",
    "websocket-natural": "artifacts/loss-service-websocket-20261004/natural/cloud.jsonl",
    "websocket-phase100": "artifacts/loss-phase-service-20261004/early-100/cloud.jsonl",
}
FIXED_S_CONTROLS = {
    "fixed-s-natural": "artifacts/s-baseline-2fps-20261004/natural-1280/fps-2/cloud.jsonl",
    "fixed-s-lateral": "artifacts/s-baseline-2fps-20261004/lateral-1280/fps-2/cloud.jsonl",
    "fixed-s-chair": "artifacts/s-baseline-2fps-20261004/controlled1-1280/fps-2/cloud.jsonl",
}
OUTPUT = ROOT / "artifacts/loss-posture-evidence-20261004"
MAX_HORIZONTAL_GAP_SECONDS = 0.6
SHADOW_MINIMUM_SECONDS = 1.5
SHADOW_MINIMUM_FRAMES = 8


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def longest_horizontal_run(observations):
    longest = {"seconds": 0.0, "frames": 0, "track": None, "first_ns": None}
    run = []

    def finish():
        nonlocal longest, run
        if run:
            span = (run[-1]["captured_mono_ns"] - run[0]["captured_mono_ns"]) / 1e9
            if span > longest["seconds"]:
                longest = {
                    "seconds": round(span, 3),
                    "frames": len(run),
                    "track": run[0]["primary_track_id"].split(":")[-1],
                    "first_ns": run[0]["captured_mono_ns"],
                }
        run = []

    for observation in [*observations, None]:
        horizontal = (
            observation is not None
            and observation.get("confirmed_person_detected")
            and observation.get("primary_orientation") == "horizontal"
            and observation.get("primary_track_id")
        )
        continuous = (
            horizontal
            and run
            and observation["primary_track_id"] == run[-1]["primary_track_id"]
            and (observation["captured_mono_ns"] - run[-1]["captured_mono_ns"]) / 1e9
            <= MAX_HORIZONTAL_GAP_SECONDS
        )
        if continuous:
            run.append(observation)
        else:
            finish()
            if horizontal:
                run = [observation]
    return longest


def analyze(name, path):
    stream = records(ROOT / path)
    observations = [row for row in stream if row["event"] == "observation"]
    by_frame = {row["frame_id"]: row for row in observations}
    rates = [
        row
        for row in stream
        if row["event"] == "agent_action" and row["type"] == "SET_TRANSMIT_FPS"
    ]
    assert [row["transmit_fps"] for row in rates] == [
        5 if index % 2 == 0 else 2 for index in range(len(rates))
    ]
    origin = observations[0]["captured_mono_ns"]
    episodes = []
    for index, up in enumerate(rates[::2]):
        start = by_frame[up["based_on_frame_id"]]["captured_mono_ns"]
        closed = 2 * index + 1 < len(rates)
        end = (
            by_frame[rates[2 * index + 1]["based_on_frame_id"]]["captured_mono_ns"]
            if closed
            else observations[-1]["captured_mono_ns"]
        )
        window = [
            row
            for row in observations
            if row["target_transmit_fps"] == 5 and start < row["captured_mono_ns"] <= end
        ]
        run = longest_horizontal_run(window)
        evidence = PostureEvidence()
        for row in window:
            evidence.observe(row)
        end_evidence = evidence.classify(end)
        episodes.append(
            {
                "start_seconds": round((start - origin) / 1e9, 3),
                "end_seconds": round((end - origin) / 1e9, 3),
                "closed": closed,
                "s_frames": len(window),
                "horizontal_frames": sum(
                    row.get("primary_orientation") == "horizontal" for row in window
                ),
                "longest_horizontal": run,
                "end_window_evidence": end_evidence if closed else None,
                "shadow_horizontal_posture": (
                    run["seconds"] >= SHADOW_MINIMUM_SECONDS
                    and run["frames"] >= SHADOW_MINIMUM_FRAMES
                ),
            }
        )
    return {"scene": name, "observations": len(observations), "episodes": episodes}


def analyze_fixed(name, path):
    observations = [row for row in records(ROOT / path) if row["event"] == "observation"]
    assert all("yolox-s" in row["person_detector_backend"] for row in observations)
    run = longest_horizontal_run(observations)
    return {
        "scene": name,
        "observations": len(observations),
        "sampling_fps": 2,
        "horizontal_frames": sum(
            row.get("primary_orientation") == "horizontal" for row in observations
        ),
        "longest_horizontal": run,
    }


def main():
    OUTPUT.mkdir(exist_ok=True)
    result = {
        "scope": "shadow classification of posture only; no incident or alert decision",
        "maximum_gap_seconds": MAX_HORIZONTAL_GAP_SECONDS,
        "minimum_span_seconds": SHADOW_MINIMUM_SECONDS,
        "minimum_frames": SHADOW_MINIMUM_FRAMES,
        "scenes": [analyze(name, path) for name, path in SOURCES.items()],
        "fixed_s_controls": [analyze_fixed(name, path) for name, path in FIXED_S_CONTROLS.items()],
    }
    (OUTPUT / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
