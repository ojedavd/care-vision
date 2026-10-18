"""Offline temporal seated-pattern diagnostic, never a benign episode closure.

Projected hip proximity and pose do not establish physical seat support.
This extends the geometric shadow evaluator; it has no controller or Edge hook.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from care_vision.cloud.pose_evidence import assess_pose_support
from care_vision.cloud.seated_candidate import SeatedCandidate, checked_box


@dataclass(frozen=True)
class SeatPoseContext:
    hip_proximity_region: tuple[float, float, float, float]
    provenance: str

    def __post_init__(self):
        object.__setattr__(self, "hip_proximity_region", checked_box(self.hip_proximity_region))
        if not self.provenance:
            raise ValueError("hip region requires independent scene provenance")


@dataclass(frozen=True)
class SeatedPosePolicy:
    minimum_response: float = 0.3
    maximum_torso_angle_degrees: float = 30.0
    maximum_thigh_vertical_fraction: float = 0.5
    minimum_shank_downward_fraction: float = 0.7
    knee_above_hip_tolerance: float = 0.04
    maximum_joint_variation: float = 0.08

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("pose thresholds must be positive and finite")
        if not 0 < self.maximum_torso_angle_degrees < 90:
            raise ValueError("invalid torso angle")
        if not 0 < self.maximum_thigh_vertical_fraction < 1:
            raise ValueError("invalid thigh fraction")
        if not 0 < self.minimum_shank_downward_fraction < 1:
            raise ValueError("invalid shank fraction")


class SeatedPoseCandidate:
    def __init__(
        self,
        geometry: SeatedCandidate,
        context: SeatPoseContext,
        *,
        policy=SeatedPosePolicy(),
        identity_link_status="unverified",
    ):
        self.geometry, self.context, self.policy = geometry, context, policy
        self.identity_link_status = identity_link_status
        self.samples = deque()
        self.last_ns = None

    def update(self, observation, points):
        base = self.geometry.update(observation)
        now = observation["captured_mono_ns"]
        gap = self.last_ns is not None and (
            now - self.last_ns > self.geometry.policy.maximum_gap_seconds * 1e9
        )
        self.last_ns = now
        result = {
            **base,
            "geometric_candidate": base["seated_like_candidate"],
            "seated_pose_candidate": False,
            "pose_stable_seconds": 0.0,
            "seat_support_verified": False,
            "would_close_episode": False,
            "evidence_status": "INSUFFICIENT_EVIDENCE",
            "pose_checks": {},
            "suggested_observation": None,
        }

        def unresolved(reason, *, incident=False):
            self.samples.clear()
            result["suggested_observation"] = self._suggest_observation(observation, reason)
            return {
                **result,
                "reason": reason,
                "evidence_status": "INCIDENT_FOLLOWUP_REQUIRED"
                if incident
                else "INSUFFICIENT_EVIDENCE",
            }

        if self.geometry.blocked:
            return unresolved("prior_horizontal_or_alert_requires_incident_followup", incident=True)
        if self.identity_link_status != "session_track_only":
            return unresolved("episode_identity_unlinked")
        if base["reason"] not in {"awaiting_stable_evidence", "seated_like_requires_review"}:
            return unresolved(base["reason"])
        if points is None:
            return unresolved("pose_unavailable")
        values = np.asarray(points, dtype=np.float64)
        if values.shape != (17, 3) or not np.all(np.isfinite(values)):
            return unresolved("pose_invalid")
        person = observation["persons"][0]
        width, height = self.geometry.context.source_width, self.geometry.context.source_height
        support = assess_pose_support(
            values, person["tracked_bbox"], (height, width), threshold=self.policy.minimum_response
        )
        if not support["body_support_usable"]:
            return unresolved("pose_support_insufficient")
        hips = values[[11, 12], :2].mean(axis=0)
        shoulders = values[[5, 6], :2].mean(axis=0)
        torso = shoulders - hips
        torso_angle = float(np.degrees(np.arctan2(abs(torso[0]), abs(torso[1]))))
        x, y, w, h = self.context.hip_proximity_region
        hip_near_seat = x <= hips[0] / width <= x + w and y <= hips[1] / height <= y + h
        checks = {
            "hip_proximity": bool(hip_near_seat),
            "torso_angle_degrees": torso_angle,
            "legs": [],
        }
        reference_height = self.geometry.reference[3] * height
        valid = hip_near_seat and torso_angle <= self.policy.maximum_torso_angle_degrees
        for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
            thigh = values[knee, :2] - values[hip, :2]
            shank = values[ankle, :2] - values[knee, :2]
            thigh_length, shank_length = np.linalg.norm(thigh), np.linalg.norm(shank)
            if min(thigh_length, shank_length) <= 0:
                return unresolved("degenerate_leg")
            thigh_fraction = float(abs(thigh[1]) / thigh_length)
            shank_fraction = float(shank[1] / shank_length)
            knee_not_high = bool(
                thigh[1] >= -self.policy.knee_above_hip_tolerance * reference_height
            )
            leg_valid = (
                knee_not_high
                and thigh_fraction <= self.policy.maximum_thigh_vertical_fraction
                and shank_fraction >= self.policy.minimum_shank_downward_fraction
            )
            checks["legs"].append(
                {
                    "thigh_vertical_fraction": thigh_fraction,
                    "shank_downward_fraction": shank_fraction,
                    "knee_not_above_hip": knee_not_high,
                    "consistent": bool(leg_valid),
                }
            )
            valid = valid and leg_valid
        result["pose_checks"] = checks
        if not valid:
            return unresolved("pose_not_consistent_with_seated_pattern")
        if gap:
            self.samples.clear()
        joints = values[[11, 12, 13, 14, 15, 16], :2].copy()
        self.samples.append((now, joints))
        cutoff = now - self.geometry.policy.stable_seconds * 1e9
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        variation = float(
            np.ptp(np.stack([p for _, p in self.samples]), axis=0).max() / reference_height
        )
        if variation > self.policy.maximum_joint_variation:
            self.samples.clear()
            self.samples.append((now, joints))
        seconds = (now - self.samples[0][0]) / 1e9
        candidate = base["seated_like_candidate"] and seconds >= self.geometry.policy.stable_seconds
        if candidate:
            # Diagnostic suggestion only. No protocol command is dispatched.
            sx, sy, sw, sh = self.context.hip_proximity_region
            result["suggested_observation"] = {
                "action": "REQUEST_ROI",
                "x": math.floor(sx * width),
                "y": math.floor(sy * height),
                "width": math.ceil(sw * width),
                "height": math.ceil(sh * height),
                "purpose": "review pelvis/seat relation; support may remain unobservable",
                "dispatched": False,
            }
        return {
            **result,
            "seated_pose_candidate": bool(candidate),
            "pose_stable_seconds": seconds,
            "joint_variation": variation,
            "evidence_status": "SEATED_PATTERN_REQUIRES_SUPPORT"
            if candidate
            else "INSUFFICIENT_EVIDENCE",
            "reason": "seat_support_unverified" if candidate else "awaiting_stable_pose",
        }

    def _suggest_observation(self, observation, reason):
        """Diagnostic proposals only; uncertainty never dispatches or closes here."""
        if reason == "prior_horizontal_or_alert_requires_incident_followup":
            return {
                "action": "KEEP_OBSERVING",
                "dispatched": False,
                "purpose": "continue incident followup; sitting pattern cannot clear this episode",
            }
        if reason in {"pose_unavailable", "pose_invalid", "pose_support_insufficient"}:
            people = observation.get("persons", [])
            if len(people) == 1 and people[0].get("track_id") == self.geometry.track_id:
                width = self.geometry.context.source_width
                height = self.geometry.context.source_height
                x, y, w, h = people[0]["tracked_bbox"]
                sx, sy, sw, sh = self.context.hip_proximity_region
                left = max(0, math.floor(min(x, sx * width) - 0.05 * width))
                top = max(0, math.floor(min(y, sy * height) - 0.05 * height))
                right = min(width, math.ceil(max(x + w, (sx + sw) * width) + 0.05 * width))
                bottom = min(height, math.ceil(max(y + h, (sy + sh) * height) + 0.05 * height))
                return {
                    "action": "REQUEST_ROI",
                    "x": left,
                    "y": top,
                    "width": right - left,
                    "height": bottom - top,
                    "dispatched": False,
                    "purpose": "review full body and seat context; hidden joints remain unknown",
                }
        if reason in {
            "context_or_stream_mismatch",
            "missing_or_ambiguous_person",
            "episode_identity_unlinked",
            "track_continuity_unavailable",
            "evidence_unavailable",
            "invalid_bbox",
        }:
            return {
                "action": "REQUEST_FULL_FRAME",
                "dispatched": False,
                "purpose": "obtain fresh scene context; does not establish identity or recovery",
            }
        return {
            "action": "KEEP_OBSERVING",
            "dispatched": False,
            "purpose": "seated pattern remains unconfirmed; continue temporal observation",
        }
