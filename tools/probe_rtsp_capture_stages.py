"""Bounded local acquisition timing; no inference, AWS or agent commands."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import cv2

from care_vision.edge.source import RTSPSource


class TimedReader:
    def __init__(self, reader, owner):
        self.reader, self.owner = reader, owner

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def read(self):
        started = time.monotonic_ns()
        ok, image = self.reader.read()
        finished = time.monotonic_ns()
        self.owner.last_read = (started, finished)
        if ok:
            self.owner.native_shape = list(image.shape)
        return ok, image


class TimedSource(RTSPSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timings = []
        self.native_shape = None
        self.last_read = None
        self.reported_stream_fps = None

    def _open_rtsp(self):
        reader = super()._open_rtsp()
        self.reported_stream_fps = reader.get(cv2.CAP_PROP_FPS)
        return TimedReader(reader, self)

    def _record(self, image, capture):
        before = time.monotonic_ns()
        super()._record(image, capture)
        after = time.monotonic_ns()
        self.timings.append({
            "capture_seq": capture.seq, "captured_mono_ns": capture.mono_ns,
            "read_started_ns": self.last_read[0], "read_finished_ns": self.last_read[1],
            "read_ms": (self.last_read[1] - self.last_read[0]) / 1e6,
            "encode_and_capture_object_ms": (before - capture.mono_ns) / 1e6,
            "record_ms": (after - before) / 1e6,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--seconds", type=int, choices=range(1, 31), default=15)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    plan = {"scope": __doc__, "duration_seconds_requested": args.seconds,
            "capture_fps_requested": 25, "capture_resolution": [1280, 720],
            "record_video": args.record, "no_audio_guidance": True,
            "measured_stages": ["RTSP read including decode", "resize/JPEG/capture object",
                                "AVI/timestamp recording hook"],
            "limits": ["No pure camera FPS or optical latency measurement",
                       "Read includes network/decoder effects; no attribution to one",
                       "CAP_PROP_FPS is metadata, not measured capture rate",
                       "Instrumentation overhead present; no CPU benchmark"],
            "camera_started": False, "aws_contacted": False, "inference_calls": 0}
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    # Use the existing private account file; never print or put credentials in argv.
    credentials = json.loads((Path.home() / ".config/care-vision/rtsp-credentials.json")
                             .read_text())
    source = TimedSource(args.ip, credentials["username"], credentials["password"],
                         seconds=args.seconds, fps=25, width=1280, height=720,
                         record=args.output / "demo.avi" if args.record else None)
    source.start()
    try:
        source.thread.join(timeout=args.seconds + 20)
    finally:
        source.close()
        with (args.output / "stages.jsonl").open("x") as stream:
            for row in source.timings:
                stream.write(json.dumps(row) + "\n")
    _, count, done, error = source.snapshot()
    intervals = [(b["captured_mono_ns"] - a["captured_mono_ns"]) / 1e6
                 for a, b in zip(source.timings, source.timings[1:])]
    report = {**plan, "camera_started": True, "frames_acquired": count,
              "completed": done, "error": error,
              "capture_thread_stopped": not source.thread.is_alive(),
              "native_shape": source.native_shape,
              "reported_stream_fps_not_measured": source.reported_stream_fps,
              "capture_interval_median_ms": statistics.median(intervals) if intervals else None,
              "capture_max_gap_ms": max(intervals) if intervals else None,
              "stage_median_ms": {key: statistics.median(r[key] for r in source.timings)
                                  for key in (
                                      "read_ms", "encode_and_capture_object_ms", "record_ms")}
              if source.timings else {}}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if error or not done or source.thread.is_alive() or count < 2:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
