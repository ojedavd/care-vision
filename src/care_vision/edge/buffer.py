"""Bounded encoded capture history. Caller holds the source lock."""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from care_vision.common.image_geometry import letterbox_geometry
from care_vision.edge.sampler import Sampler

BUFFER_SECONDS = 10
BUFFER_FRAMES = 300
BUFFER_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class Capture:
    seq: int
    mono_ns: int
    at_utc: str
    jpeg: bytes


def fit_to_canvas(image, width, height):
    """Resize without distortion and center the image on a black canvas."""
    source_height, source_width = image.shape[:2]
    rw, rh, left, top = letterbox_geometry(source_width, source_height, width, height)
    scale = min(width / source_width, height / source_height)
    resized = cv2.resize(
        image,
        (rw, rh),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def encode(image, width=640, height=360):
    if image.shape[:2] != (height, width):
        image = fit_to_canvas(image, width, height)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


class RingBuffer:
    def __init__(self, seconds=BUFFER_SECONDS, max_bytes=BUFFER_BYTES, max_frames=BUFFER_FRAMES):
        if not 0 < seconds <= BUFFER_SECONDS or max_bytes <= 0 or max_frames <= 0:
            raise ValueError("invalid ring buffer limits")
        self.duration_ns = int(seconds * 1e9)
        self.max_bytes, self.max_frames = max_bytes, max_frames
        self.frames = deque()
        self.bytes = 0
        self.evicted = 0
        self.last_seq = self.last_ns = 0

    def append(self, capture):
        if capture.seq <= self.last_seq or capture.mono_ns <= self.last_ns:
            raise ValueError("capture identity and time must increase")
        self.last_seq, self.last_ns = capture.seq, capture.mono_ns
        self.frames.append(capture)
        self.bytes += len(capture.jpeg)
        while self.frames and (
            self.frames[0].mono_ns <= capture.mono_ns - self.duration_ns
            or self.bytes > self.max_bytes
            or len(self.frames) > self.max_frames
        ):
            self.bytes -= len(self.frames.popleft().jpeg)
            self.evicted += 1

    def select(self, start_ns, end_ns, fps):
        sampler = Sampler(fps)
        sampler.set_rate(fps, start_ns)
        return tuple(
            frame
            for frame in self.frames
            if start_ns <= frame.mono_ns < end_ns and sampler.select(frame.seq, frame.mono_ns)
        )

    def status(self):
        return {
            "buffer_frames": len(self.frames),
            "buffer_bytes": self.bytes,
            "buffer_evicted": self.evicted,
            "buffer_oldest_mono_ns": self.frames[0].mono_ns if self.frames else None,
            "buffer_newest_mono_ns": self.frames[-1].mono_ns if self.frames else None,
        }
