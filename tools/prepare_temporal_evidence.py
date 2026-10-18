"""Prepare a portable fixed full-frame JPEG replay using the existing Edge encoder."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from care_vision.edge.buffer import encode
from care_vision.temporal_evaluation import file_sha256, sample_indices, validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, nargs="+", default=[2, 20])
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text())
    validate_dataset(dataset)
    schedules = {
        v["video_id"]: set().union(
            *[set(sample_indices(v["frame_count"], v["fps"], r)) for r in args.sample_fps]
        )
        for v in dataset["videos"]
    }
    args.output.mkdir(parents=True, exist_ok=False)
    frames = []
    for number, video in enumerate(dataset["videos"]):
        source = args.video_dir / video["source_filename"]
        if file_sha256(source) != video["sha256"]:
            raise ValueError("source hash mismatch")
        capture = cv2.VideoCapture(str(source))
        try:
            if (
                not capture.isOpened()
                or round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != video["frame_count"]
                or abs(capture.get(cv2.CAP_PROP_FPS) - video["fps"]) > 0.001
            ):
                raise ValueError("source timeline mismatch")
            for index in range(video["frame_count"]):
                ok, image = capture.read()
                if not ok or image.shape[:2] != (video["height"], video["width"]):
                    raise ValueError("source decode or orientation mismatch")
                if index not in schedules[video["video_id"]]:
                    continue
                jpeg = encode(image, 1280, 720)
                name = f"video-{number}-frame-{index:06d}.jpg"
                (args.output / name).write_bytes(jpeg)
                frames.append(
                    {
                        "video_id": video["video_id"],
                        "frame_index": index,
                        "filename": name,
                        "sha256": hashlib.sha256(jpeg).hexdigest(),
                    }
                )
        finally:
            capture.release()
    manifest = {
        "schema_version": 1,
        "dataset_sha256": file_sha256(args.dataset),
        "source_hashes": {v["video_id"]: v["sha256"] for v in dataset["videos"]},
        "opencv_version": cv2.__version__,
        "width": 1280,
        "height": 720,
        "encoding": "care_vision.edge.buffer.encode; JPEG quality 80",
        "scope": "Fixed full-frame representation; source timestamps, no paced capture or ROI",
        "frames": frames,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(frames)} JPEG frames in {args.output}")


if __name__ == "__main__":
    main()
