"""Reproducible video pixels only. The controller never sees the fixture schedule."""

from pathlib import Path

import cv2
import numpy as np


def make_video(path: str | Path, static=False, seconds=34, fps=30):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 360))
    if not writer.isOpened():
        raise RuntimeError("MJPEG VideoWriter unavailable")
    try:
        for index in range(round(seconds * fps)):
            t = index / fps
            x = 40
            if not static and 7 <= t < 18:
                phase = ((t - 7) * 140) % 720
                x += int(phase if phase < 360 else 720 - phase)
            image = np.full((360, 640, 3), 25, dtype=np.uint8)
            cv2.rectangle(image, (x, 90), (x + 180, 270), (220, 220, 220), -1)
            writer.write(image)
    finally:
        writer.release()
    return path
