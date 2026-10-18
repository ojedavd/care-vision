"""Sampling captured frames without changing capture FPS or duplicating images."""

import math


class Sampler:
    def __init__(self, fps=2):
        self.last_seq = 0
        self.set_rate(fps, 0)

    def set_rate(self, fps: float, now_ns: int):
        if not math.isfinite(fps) or not 0 < fps <= 30:
            raise ValueError("FPS must be in (0, 30]")
        self.fps = fps
        self.period = 1_000_000_000 / fps
        self.next_due = float(now_ns)

    def select(self, seq: int, captured_ns: int) -> bool:
        if seq <= self.last_seq:
            return False
        self.last_seq = seq
        if captured_ns < round(self.next_due):
            return False
        # Advance a fixed time grid, rather than rounding 30/20 to every second frame.
        skipped = math.floor((captured_ns - self.next_due + 0.5) / self.period)
        self.next_due += (skipped + 1) * self.period
        return True
