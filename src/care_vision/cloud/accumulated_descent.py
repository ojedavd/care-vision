"""Experimental, causal bbox posture-change signal for offline evaluation only.

This is not a fall classifier. Sitting, bending and detector box deformation can
produce the same geometry. Use one instance per camera/view/independent stream.
"""

import math
from collections import deque
from dataclasses import dataclass

from care_vision.cloud.temporal_candidate import SequenceObservation


@dataclass(frozen=True)
class DescentPolicy:
    window_seconds: float = 2.0
    minimum_span_seconds: float = 1.0
    maximum_gap_seconds: float = 0.8
    minimum_top_drop: float = 0.15
    minimum_center_drop: float = 0.05
    minimum_height_contraction: float = 0.15
    maximum_bottom_shift: float = 0.20
    maximum_reference_aspect: float = 0.8
    # Opt-in experiment; default preserves immediate reset on missing evidence.
    maximum_missing_gap_seconds: float = 0.0

    def __post_init__(self):
        if any(
            not math.isfinite(v) or v <= 0
            for k, v in vars(self).items()
            if k != "maximum_missing_gap_seconds"
        ):
            raise ValueError("parameters must be finite and positive")
        if (
            not math.isfinite(self.maximum_missing_gap_seconds)
            or self.maximum_missing_gap_seconds < 0
            or self.maximum_missing_gap_seconds > self.window_seconds
        ):
            raise ValueError("missing gap must be finite and within the evidence window")
        if self.minimum_span_seconds > self.window_seconds:
            raise ValueError("minimum span exceeds window")


class AccumulatedDescent:
    """Compare current box to oldest contiguous box within a finite time window.

    Ratios are normalized to reference box height, not image dimensions. Every
    missing frame clears that person's history by default. An opt-in experiment
    retains it for a bounded time without generating boxes during absence.
    Unknown identity and view changes
    must not be spatially stitched together. Parameters are development hypotheses.
    """

    def __init__(self, *, policy=DescentPolicy()):
        self.policy = policy
        self.histories = {}
        self.last_time = self.last_frame = None

    def update(self, timestamp_seconds, frame_index, boxes):
        SequenceObservation(timestamp_seconds, frame_index, None, None, "unknown", None)
        if self.last_time is not None and (
            timestamp_seconds <= self.last_time or frame_index <= self.last_frame
        ):
            raise ValueError("frames and timestamps must strictly increase")
        clean = {}
        for track, box in boxes.items():
            if not isinstance(track, str) or not track or len(box) != 4:
                raise ValueError("invalid track or bbox")
            if any(not math.isfinite(v) for v in box) or box[2] <= 0 or box[3] <= 0:
                raise ValueError("bbox must be finite with positive dimensions")
            clean[track] = tuple(box)
        # Retained boxes are references only; absent tracks produce no signal.
        self.histories = {
            k: h
            for k, h in self.histories.items()
            if k in clean
            or (
                self.policy.maximum_missing_gap_seconds > 0
                and h
                and timestamp_seconds - h[-1][0] <= self.policy.maximum_missing_gap_seconds
            )
        }
        signals = []
        for track, box in sorted(clean.items()):
            history = self.histories.setdefault(track, deque())
            if history and timestamp_seconds - history[-1][0] > max(
                self.policy.maximum_gap_seconds, self.policy.maximum_missing_gap_seconds
            ):
                history.clear()
            while history and timestamp_seconds - history[0][0] > self.policy.window_seconds:
                history.popleft()
            signal = {
                "track_id": track,
                "frame_index": frame_index,
                "source_time_seconds": timestamp_seconds,
                "bbox": list(box),
                "status": "insufficient_contiguous_history",
                "descent_signal": False,
                "reference_frame": None,
                "reference_time": None,
                "reference_bbox": None,
                "span_seconds": None,
                "top_drop": None,
                "center_drop": None,
                "height_contraction": None,
                "bottom_shift": None,
            }
            if history and timestamp_seconds - history[0][0] >= self.policy.minimum_span_seconds:
                ref_time, ref_frame, reference = history[0]
                _, y0, w0, h0 = reference
                _, y, _, height = box
                top_drop = (y - y0) / h0
                center_drop = (y + height / 2 - y0 - h0 / 2) / h0
                contraction = (h0 - height) / h0
                bottom_shift = (y + height - y0 - h0) / h0
                descent = (
                    w0 / h0 <= self.policy.maximum_reference_aspect
                    and top_drop >= self.policy.minimum_top_drop
                    and center_drop >= self.policy.minimum_center_drop
                    and contraction >= self.policy.minimum_height_contraction
                    and abs(bottom_shift) <= self.policy.maximum_bottom_shift
                )
                signal.update(
                    status="available",
                    descent_signal=descent,
                    reference_frame=ref_frame,
                    reference_time=ref_time,
                    reference_bbox=list(reference),
                    span_seconds=timestamp_seconds - ref_time,
                    top_drop=top_drop,
                    center_drop=center_drop,
                    height_contraction=contraction,
                    bottom_shift=bottom_shift,
                )
            signals.append(signal)
            history.append((timestamp_seconds, frame_index, box))
        self.last_time, self.last_frame = timestamp_seconds, frame_index
        return signals
