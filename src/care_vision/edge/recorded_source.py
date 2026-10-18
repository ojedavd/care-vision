"""Paced dense-file acquisition preserving recorded intervals, without inference."""

import json
import time
from pathlib import Path

import cv2

from care_vision.common.contracts import utc
from care_vision.edge.buffer import Capture, encode
from care_vision.edge.source import VideoSource


class RecordedSource(VideoSource):
    def __init__(self, video, captures, fps=25, width=1280, height=720, mapping=None):
        super().__init__(str(video), fps, width, height)
        self.rows = [json.loads(line) for line in Path(captures).read_text().splitlines()]
        if len(self.rows) < 2:
            raise ValueError("recorded replay requires at least two captures")
        for index, row in enumerate(self.rows):
            stamp = row.get("captured_mono_ns")
            if (
                row.get("capture_seq") != index + 1
                or type(stamp) is not int
                or stamp < 0
                or (index and stamp <= self.rows[index - 1]["captured_mono_ns"])
            ):
                raise ValueError("recorded capture identity/time must increase")
        self.duration_ns = self.rows[-1]["captured_mono_ns"] - self.rows[0]["captured_mono_ns"]
        if self.duration_ns > 120_000_000_000:
            raise ValueError("recorded replay is limited to 120 seconds")
        self.mapping = Path(mapping) if mapping else None
        self.replay_origin_ns = None
        if self.mapping and self.mapping.exists():
            raise ValueError("replay mapping must be new")

    def _capture(self):
        video = log = None
        try:
            video = cv2.VideoCapture(self.path)
            if not video.isOpened() or round(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(self.rows):
                raise ValueError("recorded video/capture count mismatch")
            if self.mapping:
                log = self.mapping.open("x")
            origin = time.monotonic_ns() + 500_000_000
            self.replay_origin_ns = origin
            original = self.rows[0]["captured_mono_ns"]
            native_shape = None
            for row in self.rows:
                if self.stop_event.is_set():
                    break
                ok, image = video.read()
                if not ok:
                    raise ValueError("recorded video ended before capture timeline")
                if native_shape is None:
                    native_shape = image.shape[:2]
                elif image.shape[:2] != native_shape:
                    raise ValueError("recorded source dimensions changed")
                jpeg = encode(image, self.width, self.height)
                due = origin + row["captured_mono_ns"] - original
                if self.stop_event.wait(max(0, (due - time.monotonic_ns()) / 1e9)):
                    break
                available = time.monotonic_ns()
                capture = Capture(row["capture_seq"], due, utc(), jpeg)
                with self.lock:
                    self.buffer.append(capture)
                    self.latest = capture
                    self.count = capture.seq
                if log:
                    log.write(
                        json.dumps(
                            {
                                "capture_seq": capture.seq,
                                "original_captured_mono_ns": row["captured_mono_ns"],
                                "replay_captured_mono_ns": due,
                                "available_mono_ns": available,
                                "lateness_ms": (available - due) / 1e6,
                            }
                        )
                        + "\n"
                    )
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            if video is not None:
                video.release()
            if log:
                log.close()
            with self.lock:
                self.done = True
