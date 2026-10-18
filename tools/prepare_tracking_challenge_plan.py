"""Pin existing multi-person/partial-person sources and real media PTS schedules."""

import argparse
import bisect
import hashlib
import json
import math
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule(timestamps, fps):
    if not math.isfinite(fps) or fps <= 0 or any(not math.isfinite(t) for t in timestamps):
        raise ValueError("media timestamps and sampling rate must be finite; rate positive")
    if not timestamps or any(a >= b for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("media timestamps must strictly increase")
    relative = [t - timestamps[0] for t in timestamps]
    indices = []
    step = 0
    while step / fps <= relative[-1]:
        target = step / fps
        right = bisect.bisect_left(relative, target)
        candidates = [i for i in (right - 1, right) if 0 <= i < len(relative)]
        index = min(candidates, key=lambda i: (abs(relative[i] - target), i))
        if not indices or indices[-1] != index:
            indices.append(index)
        step += 1
    return [
        {
            "capture_seq": i + 1,
            "frame_index": i,
            "captured_mono_ns": 1_000_000_000 + round(relative[i] * 1_000_000_000),
            "seconds_from_first_observation": relative[i],
        }
        for i in indices
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resize-mode", choices=("stretch", "letterbox"), default="letterbox")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    prior_profile = root / "evaluation/visual-comparison-20261014/profile.json"
    previous = json.loads(prior_profile.read_text())
    profile = {
        k: previous[k]
        for k in (
            "schema_version",
            "instance_type",
            "architecture",
            "opencv_version",
            "opencv_num_threads",
            "person_model_sha256",
            "person_input_size",
            "representation",
            "strong_confidence",
            "weak_association_floor",
        )
    }
    profile.update(
        scope="offline recorded media tracking challenge; no camera capture target",
        baseline_profile_sha256=sha(prior_profile),
        requested_sample_fps=[2, 5],
        resize_mode=args.resize_mode,
    )
    (args.output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    dataset = json.loads(args.dataset.read_text())
    manifest = {
        "schema_version": 1,
        "status": "planned before inference",
        "scope": "existing development sources; fixed 2/5 FPS; sparse manual labels only",
        "annotation_path": str(args.dataset.resolve()),
        "annotation_sha256": sha(args.dataset),
        "time_basis": (
            "ffprobe best_effort_timestamp_time, relative video PTS; "
            "captured_mono_ns is synthetic 1s+PTS, not measured camera clock"
        ),
        "resize": args.resize_mode + " to1280x720; transform annotated boxes identically",
        "new_ground_truth": False,
        "runs": [],
    }
    for video in dataset["videos"]:
        source = (args.video_dir / video["source_filename"]).resolve()
        if sha(source) != video["sha256"]:
            raise ValueError("video source digest mismatch")
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(source),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        times = [float(f["best_effort_timestamp_time"]) for f in json.loads(probe.stdout)["frames"]]
        if len(times) != video["frame_count"]:
            raise ValueError("frame count mismatch")
        (args.output / f"{video['video_id']}-pts.json").write_text(probe.stdout)
        schedules = [
            {"requested_sample_fps": fps, "frames": schedule(times, fps)} for fps in (2, 5)
        ]
        width, height = video["width"], video["height"]
        if args.resize_mode == "letterbox":
            scale = min(1280 / width, 720 / height)
            resized_width, resized_height = round(width * scale), round(height * scale)
            transform = [
                resized_width / width,
                resized_height / height,
                (1280 - resized_width) // 2,
                (720 - resized_height) // 2,
            ]
        else:
            transform = [1280 / width, 720 / height, 0, 0]
        manifest["runs"].append(
            {
                "name": video["video_id"],
                "dataset_split": "development",
                "video_path": str(source),
                "source_files": {str(source): video["sha256"]},
                "source_dimensions": [video["width"], video["height"]],
                "resize_to": [1280, 720],
                "resize_mode": args.resize_mode,
                "box_transform": transform,
                "schedules": schedules,
                "annotation_frame_indices": [s["frame_index"] for s in video["samples"]],
                "annotation_review": dataset["review"],
            }
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "plan ready",
        [(r["name"], [len(s["frames"]) for s in r["schedules"]]) for r in manifest["runs"]],
    )


if __name__ == "__main__":
    main()
