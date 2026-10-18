"""Bounded, non-executable observation plans from diagnostic pose evidence.

No controller/socket hook: testing a plan does not claim its requested evidence
exists. The production observer and incident decisions remain independent.
"""

import math
from dataclasses import dataclass

from care_vision.cloud.reference_posture import bbox_block_reason


@dataclass(frozen=True)
class PoseObservationPolicy:
    maximum_result_age_seconds: float = 1.0
    roi_window_seconds: float = 2.0
    maximum_roi_frames: int = 4
    padding_fraction: float = 0.05
    maximum_episode_budgets: int = 256

    def __post_init__(self):
        for value in (self.maximum_result_age_seconds, self.roi_window_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("observation times must be finite and positive")
        if not 0 <= self.padding_fraction <= 0.25:
            raise ValueError("invalid ROI padding")
        if any(type(v) is not int or v < 1 for v in (
            self.maximum_roi_frames, self.maximum_episode_budgets,
        )):
            raise ValueError("observation budgets must be positive integers")


class PoseObservationPlanner:
    """One proposed detail window per logical episode, including after reconnect.

Use the same planner for that episode's lifetime. Capacity exhaustion suppresses
new plans instead of evicting budgets and silently permitting repeated requests.
"""

    def __init__(self, geometry, hips, *, policy=PoseObservationPolicy()):
        self.geometry, self.hips, self.policy = geometry, hips, policy
        self.spent = set()

    def consider(self, observation, shadow, episode, *, result_age_seconds, commands_pending=False):
        result = {
            "mode": "observation_plan_only", "dispatched": False,
            "would_close_episode": False, "seat_support_verified": False,
            "observation_id": observation["observation_id"],
            "episode_id": episode["episode_id"] if episode else None,
            "plan": None,
        }

        def skip(reason):
            return {**result, "reason": reason}

        if episode is None:
            return skip("no_active_episode")
        if commands_pending:
            return skip("command_pending")
        if not math.isfinite(result_age_seconds) or not (
            0 <= result_age_seconds <= self.policy.maximum_result_age_seconds
        ):
            return skip("result_stale_or_clock_invalid")
        if any(shadow.get(k) != observation.get(k) for k in (
            "observation_id", "frame_id", "capture_seq", "captured_mono_ns",
        )) or shadow.get("episode_id") != episode["episode_id"]:
            return skip("result_observation_or_episode_mismatch")
        if shadow.get("context_status") != "current" or not shadow.get("context_provenance"):
            return skip("context_not_current")
        if episode.get("identity_link_status", "session_track_only") != "session_track_only":
            return skip("episode_identity_unlinked")
        if episode["horizontal_observed"] or episode["test_alert_emitted"]:
            return skip("incident_followup_required")
        c = self.geometry
        if (
            observation.get("valid") is not True
            or observation.get("evidence_kind") != "live"
            or observation.get("view_kind") != "full_frame"
            or observation.get("edge_id") != c.edge_id
            or (observation.get("width"), observation.get("height"))
            != (c.source_width, c.source_height)
            or (observation.get("source_width"), observation.get("source_height"))
            != (c.source_width, c.source_height)
            or observation.get("person_detection_status") != "ok"
            or observation.get("person_tracking_status") != "ok"
        ):
            return skip("fresh_full_scene_required")
        people = observation.get("persons", [])
        if len(people) != 1 or (
            people[0].get("track_id") != episode["track_id"]
            or people[0].get("track_state") != "confirmed"
            or people[0].get("association_only", False)
        ):
            return skip("unique_confirmed_episode_track_required")
        person = people[0]
        for key in ("tracked_bbox", "bbox"):
            if key == "bbox" and key not in person:
                continue
            if bbox_block_reason(person.get(key), c.source_width, c.source_height):
                return skip("body_extent_insufficient")
        stable = (
            shadow.get("pose_status") == "ok"
            and shadow.get("seated_pose_candidate") is True
            and shadow.get("reason") == "seat_support_unverified"
            and shadow.get("evidence_status") == "SEATED_PATTERN_REQUIRES_SUPPORT"
        )
        weak = (
            shadow.get("pose_status") == "ok"
            and shadow.get("reason") == "pose_support_insufficient"
        )
        if not (stable or weak):
            return skip("no_actionable_pose_uncertainty")
        if episode.get("pose_observation_requested") or episode["episode_id"] in self.spent:
            return skip("episode_observation_budget_spent")
        if len(self.spent) >= self.policy.maximum_episode_budgets:
            return skip("planner_capacity_exhausted")
        # Full detected extent plus manually projected chair regions. These are
        # image-space context proxies, not segmented furniture or visible joints.
        boxes = [person["tracked_bbox"]]
        if "bbox" in person:
            boxes.append(person["bbox"])
        boxes.extend([
            [v * (c.source_width if i % 2 == 0 else c.source_height)
             for i, v in enumerate(region)]
            for region in (c.occupancy_region, self.hips.hip_proximity_region)
        ])
        px = self.policy.padding_fraction * c.source_width
        py = self.policy.padding_fraction * c.source_height
        left = max(0, math.floor(min(b[0] for b in boxes) - px))
        top = max(0, math.floor(min(b[1] for b in boxes) - py))
        right = min(c.source_width, math.ceil(max(b[0] + b[2] for b in boxes) + px))
        bottom = min(c.source_height, math.ceil(max(b[1] + b[3] for b in boxes) + py))
        self.spent.add(episode["episode_id"])
        return {
            **result, "reason": "bounded_body_and_chair_detail",
            "plan": {
                "request": {
                    "action": "REQUEST_ROI", "source_width": c.source_width,
                    "source_height": c.source_height,
                    "roi": [left, top, right - left, bottom - top],
                    "output_width": 640, "output_height": 360,
                    "transmit_fps": "keep_current",
                    "purpose": "inspect body and chair context; physical occlusion may persist",
                },
                "return": {
                    "action": "REQUEST_FULL_FRAME", "output_width": observation["width"],
                    "output_height": observation["height"],
                    "after_applied_ack_seconds": self.policy.roi_window_seconds,
                    "or_after_roi_frames": self.policy.maximum_roi_frames,
                    "requires_applied_ack_and_new_full_frame": True,
                    "on_missing_evidence": "request full frame; keep episode unresolved",
                },
                "critical_decisions": "ROI cannot establish episode identity or recovery",
                "physical_visibility_status": "unknown",
            },
        }
