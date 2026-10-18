"""Causal, offline-only sequence hypothesis. No commands or incident classification.

These development thresholds are not calibrated. A horizontal box is not proof
of floor contact, and a recovery cue never establishes that a person is safe.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SequencePolicy:
    descent_memory_seconds: float = 5.0
    horizontal_seconds: float = 1.0
    upright_seconds: float = 2.0
    maximum_gap_seconds: float = 0.8

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("sequence durations must be finite and positive")


@dataclass(frozen=True)
class SequenceObservation:
    """Allowlist: the sequence analyzer cannot access annotations or future frames."""

    timestamp_seconds: float
    frame_index: int
    track_id: str | None
    orientation: str | None
    motion: str
    pose_straightening: bool | None

    def __post_init__(self):
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("invalid timestamp")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("invalid frame index")
        if self.track_id is not None and (not isinstance(self.track_id, str) or not self.track_id):
            raise ValueError("invalid track ID")
        if self.orientation not in (None, "horizontal", "vertical", "intermediate"):
            raise ValueError("invalid orientation")
        if self.motion not in ("unknown", "upward", "downward", "stable"):
            raise ValueError("invalid motion")
        if self.pose_straightening is not None and type(self.pose_straightening) is not bool:
            raise ValueError("invalid pose cue")


class TemporalCandidate:
    def __init__(self, *, use_pose=False, retain_pending_descent=False, policy=SequencePolicy()):
        self.use_pose = use_pose
        self.retain_pending_descent = retain_pending_descent
        self.policy = policy
        self.last = None
        self.last_visible = None
        self.track_id = None
        self.descent = None
        self.horizontal = None
        self.upright = None
        self.active = None
        self.episodes = []

    def _clear(self, observation, reason):
        if self.active is not None:
            self.active["end_frame"] = observation.frame_index
            self.active["end_reason"] = reason
        self.active = self.descent = self.horizontal = self.upright = None

    def update(self, observation):
        o = observation
        if self.last is not None and (
            o.timestamp_seconds <= self.last.timestamp_seconds
            or o.frame_index <= self.last.frame_index
        ):
            raise ValueError("observations must increase in time and frame index")
        reset = None
        if self.last is not None:
            if o.timestamp_seconds - self.last.timestamp_seconds > self.policy.maximum_gap_seconds:
                reset = "observation_gap"
            elif o.track_id != self.track_id:
                reset = "track_changed"
        if o.track_id is None or o.orientation is None:
            reset = "evidence_unavailable"
        if reset:
            # Optional development ablation: retain only an unconfirmed descent,
            # bounded by the last VISIBLE observation (not successive missing ticks).
            # Horizontal dwell always resets; active episodes always abort on loss.
            retained = (
                self.descent
                if self.retain_pending_descent
                and self.active is None
                and self.last_visible is not None
                and o.timestamp_seconds - self.last_visible.timestamp_seconds
                <= self.policy.maximum_gap_seconds
                and o.track_id in (None, self.last_visible.track_id)
                and reset != "observation_gap"
                else None
            )
            self._clear(o, reset)
            self.descent = retained
        self.last, self.track_id = o, o.track_id
        available = o.track_id is not None and o.orientation is not None
        if available:
            self.last_visible = o
        if (
            self.descent is not None
            and o.timestamp_seconds - self.descent.timestamp_seconds
            > self.policy.descent_memory_seconds
        ):
            self.descent = self.horizontal = None
        result = {
            "frame_index": o.frame_index,
            "source_time_seconds": o.timestamp_seconds,
            "state": ("DESCENT_SEEN" if self.descent is not None else "OBSERVING")
            if available
            else "UNKNOWN",
            "bbox_available": available,
            "motion_available": available and o.motion != "unknown",
            "pose_cue_available": available and o.pose_straightening is not None,
            "reset_reason": reset,
            "pending_descent_retained": bool(reset and self.descent is not None),
            "events": [],
        }
        if not available or reset:
            return result

        if self.active is not None:
            # First cue only: never replace an early/incorrect cue with a later match.
            cues = []
            if o.motion == "upward":
                cues.append("bbox_upward")
            if self.use_pose and o.pose_straightening is True:
                cues.append("pose_straightening")
            if self.active["recovery_cue_frame"] is None and cues:
                self.active["recovery_cue_frame"] = o.frame_index
                self.active["recovery_cue_sources"] = cues
                result["events"].append("recovery_cue")
            if o.orientation == "vertical":
                if self.upright is None:
                    self.upright = o
                if (
                    o.timestamp_seconds - self.upright.timestamp_seconds
                    >= self.policy.upright_seconds
                ):
                    self._clear(o, "sustained_vertical_proxy")
                    result["events"].append("upright_proxy")
            else:
                self.upright = None
        else:
            # Descent must be observed BEFORE the horizontal run, not inferred
            # retrospectively from a horizontal box or pose on the same frame.
            if o.orientation != "horizontal":
                self.horizontal = None
                if o.motion == "downward":
                    self.descent = o
            if o.orientation == "horizontal" and self.descent is not None:
                if self.horizontal is None:
                    self.horizontal = o
                if (
                    o.timestamp_seconds - self.horizontal.timestamp_seconds
                    >= self.policy.horizontal_seconds
                ):
                    self.active = {
                        "episode_id": len(self.episodes) + 1,
                        "track_id": o.track_id,
                        "descent_frame": self.descent.frame_index,
                        "horizontal_start_frame": self.horizontal.frame_index,
                        "candidate_frame": o.frame_index,
                        "recovery_cue_frame": None,
                        "recovery_cue_sources": [],
                        "end_frame": None,
                        "end_reason": None,
                    }
                    self.episodes.append(self.active)
                    result["events"].append("candidate")
        if self.active is not None:
            result["state"] = (
                "RECOVERY_CUE"
                if self.active["recovery_cue_frame"] is not None
                else "POST_DESCENT_HORIZONTAL"
            )
        elif self.descent is not None:
            result["state"] = "DESCENT_SEEN"
        else:
            result["state"] = "OBSERVING"
        return result
