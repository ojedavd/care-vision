"""Deterministic experimental observation policy. No incident/medical decision."""

import math
from copy import deepcopy
from dataclasses import dataclass, replace

from care_vision.cloud.accumulated_descent import AccumulatedDescent
from care_vision.cloud.loss_probe import LossProbe
from care_vision.cloud.posture_evidence import PostureEvidence
from care_vision.cloud.reference_posture import (
    RECOVERY_RULE_VERSION,
    recovery_block_reason,
    reference_block_reason,
)
from care_vision.cloud.visibility_probe import VisibilityProbe
from care_vision.common.contracts import envelope, uid, utc


@dataclass(frozen=True)
class Policy:
    high: float = 0.035
    low: float = 0.01
    suspicious_seconds: float = 0.5
    quiet_seconds: float = 3
    hold_seconds: float = 5
    minimum_new_observations: int = 10
    maximum_gap_seconds: float = 0.8
    recovery_seconds: float = 2
    experimental_loss_5fps: bool = False
    experimental_s_only_5fps: bool = False
    loss_verification_seconds: float = 6
    loss_max_verification_seconds: float = 10
    loss_downward_extension_pixels: int = 150
    experimental_alert_after_seconds: float | None = None
    experimental_episode_continuity: bool = False
    episode_followup_seconds: float = 30
    episode_followup_limit: int = 3

    def __post_init__(self):
        value = self.experimental_alert_after_seconds
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("experimental alert time must be finite and positive")
        if value is not None and not self.experimental_s_only_5fps:
            raise ValueError("experimental alerts require the S-only mode")
        if self.experimental_episode_continuity and not self.experimental_s_only_5fps:
            raise ValueError("episode continuity currently requires S-only mode")
        if not math.isfinite(self.episode_followup_seconds) or self.episode_followup_seconds <= 0:
            raise ValueError("episode follow-up interval must be finite and positive")
        if type(self.episode_followup_limit) is not int or self.episode_followup_limit < 1:
            raise ValueError("episode follow-up limit must be positive")


class Controller:
    def __init__(self, identity: dict, policy: Policy | None = None):
        self.identity = identity
        self.policy = policy or Policy()
        self.state = "NORMAL"
        self.command_seq = 0
        self.pending = None
        self.visual_pending = None
        self.expected_command = None
        self.expected_visual_command = None
        self.active_view_kind = "full_frame"
        self.baseline_resolution = (640, 360)
        self.high_since = self.quiet_since = self.escalated_since = None
        self.recovery_since = None
        self.last_time = None
        self.new_observations = 0
        self.transitions = []
        self.loss_probe = LossProbe() if self.policy.experimental_loss_5fps else None
        self.s_probe = VisibilityProbe() if self.policy.experimental_s_only_5fps else None
        if self.s_probe is not None:
            self.s_probe.change = AccumulatedDescent(
                policy=replace(self.s_probe.change.policy, maximum_missing_gap_seconds=1.1)
            )
        self.loss_started_ns = None
        self.loss_first_s_top = self.loss_max_s_top = None
        self.loss_s_frames = self.loss_s_person_frames = 0
        self.loss_outcomes = []
        self.episode = None
        self.episode_events = []
        self.posture_evidence = PostureEvidence() if self.loss_probe or self.s_probe else None

    def acknowledge(self, ack: dict) -> bool:
        if not self.pending or ack["command_id"] != self.pending["command_id"]:
            return False
        if (
            ack["result"] not in {"applied", "duplicate"}
            or ack["active_policy_seq"] != self.pending["command_seq"]
            or ack["active_transmit_fps"] != self.pending["transmit_fps"]
        ):
            raise ValueError("edge did not apply the requested command")
        self.expected_command = self.pending["command_id"]
        self.pending = None
        return True

    def acknowledge_visual(self, ack: dict) -> bool:
        if not self.visual_pending or ack["command_id"] != self.visual_pending["command_id"]:
            return False
        expected_view = "roi" if self.visual_pending["type"] == "REQUEST_ROI" else "full_frame"
        if (
            ack["result"] not in {"applied", "duplicate"}
            or ack["active_visual_policy_seq"] != self.visual_pending["command_seq"]
            or ack["active_view_kind"] != expected_view
        ):
            raise ValueError("edge did not apply the requested visual command")
        self.expected_visual_command = self.visual_pending["command_id"]
        self.active_view_kind = expected_view
        self.visual_pending = None
        return True

    def observe(self, observation: dict) -> dict | None:
        if observation.get("evidence_kind", "live") != "live":
            return None
        now = observation["captured_mono_ns"] / 1e9
        gap = self.last_time is not None and now - self.last_time > self.policy.maximum_gap_seconds
        self.last_time = now
        if gap or not observation["valid"]:
            if self.episode is not None:
                self.episode["upright_since_ns"] = None
                self.episode["previous_horizontal_ns"] = None
                if self.policy.experimental_episode_continuity:
                    self.episode["evidence_unavailable"] = True
            self.high_since = self.quiet_since = None
            if self.state == "SUSPICIOUS":
                self._transition(
                    "NORMAL",
                    observation,
                    "observation_gap" if gap else "invalid_observation",
                )
            if self.state == "RECOVERY":
                self.recovery_since = None
        if not observation["valid"] or self.pending or self.visual_pending:
            if self.episode is not None:
                self.episode["upright_since_ns"] = None
                self.episode["previous_horizontal_ns"] = None
            return None
        if observation["applied_command_id"] != self.expected_command:
            if self.episode is not None:
                self.episode["upright_since_ns"] = None
                self.episode["previous_horizontal_ns"] = None
            return None
        if observation.get("applied_visual_command_id") != self.expected_visual_command:
            if self.episode is not None:
                self.episode["upright_since_ns"] = None
                self.episode["previous_horizontal_ns"] = None
            return None
        if self.episode is not None:
            self._observe_episode(observation)
        if self.policy.experimental_episode_continuity and self.episode is not None:
            followup = self._episode_followup(observation)
            if followup:
                return followup
        if self.loss_probe is not None:
            return self._observe_loss_verification(observation)
        if self.s_probe is not None:
            return self._observe_s_only_verification(observation)
        score = observation["motion_score"]
        if self.state == "NORMAL":
            if score >= self.policy.high:
                self.high_since = now
                self._transition("SUSPICIOUS", observation, "motion_above_high_threshold")
        elif self.state == "SUSPICIOUS":
            if score < self.policy.high:
                self.high_since = None
                self._transition("NORMAL", observation, "motion_not_sustained")
            elif now - self.high_since >= self.policy.suspicious_seconds:
                return self._command(observation, 20, "sustained_motion_requires_more_evidence")
        elif self.state == "VERIFYING":
            if observation.get("view_kind", "full_frame") == "roi" and not observation.get(
                "confirmed_person_detected"
            ):
                # A quiet crop without a confirmed person is missing context,
                # not evidence that activity in the full scene has stopped.
                return self.request_full_frame(observation, reason="roi_person_unconfirmed")
            if self.escalated_since is None:
                self.escalated_since = now
            self.new_observations += 1
            if score <= self.policy.low:
                if self.quiet_since is None:
                    self.quiet_since = now
                if (
                    now - self.quiet_since >= self.policy.quiet_seconds
                    and now - self.escalated_since >= self.policy.hold_seconds
                    and self.new_observations >= self.policy.minimum_new_observations
                ):
                    if observation.get("view_kind", "full_frame") == "roi":
                        return self.request_full_frame(
                            observation, reason="verify_scene_before_return_to_baseline"
                        )
                    return self._command(observation, 2, "new_evidence_is_quiet_return_to_baseline")
            else:
                self.quiet_since = None
        elif self.state == "RECOVERY":
            if score > self.policy.low:
                self.recovery_since = None
                self.high_since = now
                self._transition("SUSPICIOUS", observation, "motion_resumed_during_recovery")
            else:
                if self.recovery_since is None:
                    self.recovery_since = now
                elif now - self.recovery_since >= self.policy.recovery_seconds:
                    self.recovery_since = None
                    self._transition("NORMAL", observation, "quiet_recovery_confirmed")
        return None

    def _observe_loss_verification(self, observation):
        now_ns = observation["captured_mono_ns"]
        if self.state == "NORMAL" and observation.get("target_transmit_fps") == 2:
            lost = self.loss_probe.observe(now_ns, observation)
            if lost:
                self._open_episode(observation, lost["track_id"], lost["bbox"])
                self.baseline_resolution = (observation["width"], observation["height"])
                self.loss_started_ns = now_ns
                self.loss_first_s_top = self.loss_max_s_top = None
                self.loss_s_frames = self.loss_s_person_frames = 0
                self.posture_evidence = PostureEvidence()
                return self._command(observation, 5, "confirmed_person_missing_verify")
        elif self.state == "VERIFYING" and self.loss_started_ns is not None:
            if observation.get("target_transmit_fps") != 5 or (
                observation["width"],
                observation["height"],
            ) != (1280, 720):
                return None
            self.loss_s_frames += 1
            self.posture_evidence.observe(observation)
            box = observation.get("primary_person_bbox")
            if box is not None:
                self.loss_s_person_frames += 1
                if self.loss_first_s_top is None:
                    self.loss_first_s_top = box[1]
                self.loss_max_s_top = max(
                    self.loss_max_s_top if self.loss_max_s_top is not None else box[1],
                    box[1],
                )
            elapsed = (now_ns - self.loss_started_ns) / 1e9
            extending = (
                self.loss_first_s_top is not None
                and self.loss_max_s_top - self.loss_first_s_top
                >= self.policy.loss_downward_extension_pixels
            )
            limit = (
                self.policy.loss_max_verification_seconds
                if extending
                else self.policy.loss_verification_seconds
            )
            if elapsed >= limit:
                posture = self.posture_evidence.classify(now_ns)
                self.loss_outcomes.append(
                    {
                        **posture,
                        "reason": "posture_evidence_at_verification_limit",
                        "incident_status": "unresolved",
                        "observation_id": observation["observation_id"],
                        "frame_id": observation["frame_id"],
                        "captured_mono_ns": now_ns,
                        "elapsed_seconds": round(elapsed, 3),
                        "s_frames": self.loss_s_frames,
                        "s_person_frames": self.loss_s_person_frames,
                        "first_s_top": self.loss_first_s_top,
                        "max_s_top": self.loss_max_s_top,
                        "downward_extension_used": extending,
                    }
                )
                self.loss_started_ns = None
                return self._command(
                    observation,
                    2,
                    "loss_verification_posture_assessed_return_to_baseline",
                    next_state="UNRESOLVED",
                )
        elif self.state == "UNRESOLVED" and observation.get("target_transmit_fps") == 2:
            # Restoring the low-rate stream does not resolve the episode.
            return None
        return None

    def _observe_s_only_verification(self, observation):
        now_ns = observation["captured_mono_ns"]
        rate = observation.get("target_transmit_fps")
        if (
            observation.get("view_kind", "full_frame") != "full_frame"
            or (observation["width"], observation["height"]) != (1280, 720)
        ):
            return None
        if self.state == "NORMAL" and rate == 2:
            change = self.s_probe.update(
                now_ns / 1e9,
                observation["capture_seq"],
                [p for p in observation["persons"] if not p.get("association_only", False)],
            )
            if change["started"]:
                reference = change["started"][0]
                self._open_episode(
                    observation, reference["source_track"], reference["reference_bbox"]
                )
                self.loss_started_ns = now_ns
                self.loss_first_s_top = self.loss_max_s_top = None
                self.loss_s_frames = self.loss_s_person_frames = 0
                self.posture_evidence = PostureEvidence()
                return self._command(observation, 5, "posture_change_requires_more_evidence")
        elif self.state == "VERIFYING" and self.loss_started_ns is not None and rate == 5:
            self.loss_s_frames += 1
            self.posture_evidence.observe(observation)
            box = observation.get("primary_person_bbox")
            if box is not None:
                self.loss_s_person_frames += 1
            elapsed = (now_ns - self.loss_started_ns) / 1e9
            if elapsed >= self.policy.loss_verification_seconds:
                posture = self.posture_evidence.classify(now_ns)
                self.loss_outcomes.append(
                    {
                        **posture,
                        "reason": "s_only_posture_evidence_at_verification_limit",
                        "incident_status": "unresolved",
                        "observation_id": observation["observation_id"],
                        "frame_id": observation["frame_id"],
                        "captured_mono_ns": now_ns,
                        "elapsed_seconds": round(elapsed, 3),
                        "s_frames": self.loss_s_frames,
                        "s_person_frames": self.loss_s_person_frames,
                    }
                )
                self.loss_started_ns = None
                return self._command(
                    observation,
                    2,
                    "s_only_posture_assessed_return_to_baseline",
                    next_state=(
                        "ASSISTANCE_REQUIRED"
                        if self.episode and self.episode["test_alert_emitted"] else "UNRESOLVED"
                    ),
                )
        elif self.state == "UNRESOLVED" and rate == 2:
            # Continue observing the same episode at baseline rate.
            return None
        return None

    def _open_episode(self, observation, track_id, bbox):
        self.episode = {
            "episode_id": observation["observation_id"],
            "track_id": track_id,
            "started_ns": observation["captured_mono_ns"],
            "reference_bbox": [
                value / observation["width" if i % 2 == 0 else "height"]
                for i, value in enumerate(bbox)
            ],
            "horizontal_observed": False,
            "upright_since_ns": None,
            "last_seen_ns": None,
            "incident_status": "unresolved",
            "horizontal_evidence_seconds": 0.0,
            "previous_horizontal_ns": None,
            "test_alert_emitted": False,
            "alert_after_seconds": self.policy.experimental_alert_after_seconds,
        }
        self._set_reference_eligibility(observation)
        if self.policy.experimental_episode_continuity:
            self.episode.update(
                origin_identity=dict(self.identity), identity_link_status="session_track_only",
                evidence_unavailable=False, followup_requests=0,
                followup_last_ns=observation["captured_mono_ns"], followup_budget_reported=False,
            )
        self._episode_event(observation, "episode_opened")

    def _set_reference_eligibility(self, observation):
        reason = reference_block_reason(
            self.episode["reference_bbox"], observation["width"], observation["height"]
        )
        self.episode.update(
            recovery_rule_version=RECOVERY_RULE_VERSION,
            reference_recovery_eligible=reason is None,
            reference_recovery_block_reason=reason,
        )

    def restore_episode(self, episode):
        if not self.policy.experimental_episode_continuity:
            raise ValueError("restoring an episode requires continuity mode")
        self.episode = deepcopy(episode)
        self.episode.update(
            upright_since_ns=None, previous_horizontal_ns=None, last_seen_ns=None,
            evidence_unavailable=True, identity_link_status="unresolved_after_reconnect",
            followup_last_ns=None, followup_budget_reported=False,
        )
        self.state = "ASSISTANCE_REQUIRED" if episode["test_alert_emitted"] else "UNRESOLVED"

    def _episode_followup(self, observation):
        episode = self.episode
        if self.state not in {"UNRESOLVED", "ASSISTANCE_REQUIRED"}:
            return None
        # A confirmed alert alone does not require repetitive high-rate sensing.
        if self.state == "ASSISTANCE_REQUIRED" and not episode["evidence_unavailable"]:
            return None
        now = observation["captured_mono_ns"]
        previous = episode["followup_last_ns"]
        if previous is not None and now - previous < self.policy.episode_followup_seconds * 1e9:
            return None
        if episode["followup_requests"] >= self.policy.episode_followup_limit:
            if not episode["followup_budget_reported"]:
                episode["followup_budget_reported"] = True
                self._episode_event(observation, "episode_observation_budget_exhausted")
            return None
        if observation.get("view_kind", "full_frame") != "full_frame" or (
            observation["width"], observation["height"]
        ) != (1280, 720):
            return self.request_full_frame(
                observation, 1280, 720, reason="unresolved_episode_requires_scene_context"
            )
        episode["followup_requests"] += 1
        episode["followup_last_ns"] = now
        self.loss_started_ns = now
        self.loss_s_frames = self.loss_s_person_frames = 0
        self.posture_evidence = PostureEvidence()
        self._episode_event(observation, "episode_requires_more_observation")
        return self._command(observation, 5, "unresolved_episode_requires_more_observation")

    def _episode_event(self, observation, reason):
        self.episode_events.append(
            {
                **self.episode,
                "reference_bbox": list(self.episode["reference_bbox"]),
                "reason": reason,
                "observation_id": observation["observation_id"],
                "frame_id": observation["frame_id"],
                "captured_mono_ns": observation["captured_mono_ns"],
            }
        )

    def _observe_episode(self, observation):
        episode = self.episode
        now = observation["captured_mono_ns"]
        if episode.get("identity_link_status") == "unresolved_after_reconnect":
            # A fresh session tracker cannot prove continuity of the physical person.
            episode["upright_since_ns"] = episode["previous_horizontal_ns"] = None
            return
        if observation.get("view_kind", "full_frame") != "full_frame":
            episode["upright_since_ns"] = None
            episode["previous_horizontal_ns"] = None
            return
        people = observation.get("persons", [])
        person = next(
            (
                p
                for p in people
                if p.get("track_state") == "confirmed" and p.get("track_id") == episode["track_id"]
            ),
            None,
        )
        if (
            person is None
            or len(people) != 1
            or (
                observation.get("person_detection_status") != "ok"
                or observation.get("person_tracking_status") != "ok"
            )
        ):
            episode["upright_since_ns"] = None
            episode["previous_horizontal_ns"] = None
            if self.policy.experimental_episode_continuity:
                episode["evidence_unavailable"] = True
            return
        if self.policy.experimental_episode_continuity:
            episode["evidence_unavailable"] = False
        self._set_reference_eligibility(observation)
        blocked = recovery_block_reason(episode["reference_bbox"], observation, person)
        if episode.get("recovery_block_reason") != blocked:
            episode["recovery_block_reason"] = blocked
            if blocked:
                self._episode_event(observation, "reference_posture_evidence_insufficient")
        if blocked:
            episode["upright_since_ns"] = None
            if self.policy.experimental_episode_continuity:
                episode["evidence_unavailable"] = True
            if blocked.startswith("current_"):
                episode["previous_horizontal_ns"] = None
                episode["last_seen_ns"] = None
                return
        _, y, w, h = [
            value / observation["width" if i % 2 == 0 else "height"]
            for i, value in enumerate(person["tracked_bbox"])
        ]
        if h <= 0:
            episode["upright_since_ns"] = None
            episode["previous_horizontal_ns"] = None
            return
        # Geometric evidence only; these development thresholds do not classify falls.
        aspect = w * observation["width"] / (h * observation["height"])
        previous_horizontal = episode["previous_horizontal_ns"]
        horizontal = aspect >= 1.2
        if horizontal and previous_horizontal is not None:
            delta = (now - previous_horizontal) / 1e9
            if 0 < delta <= self.policy.maximum_gap_seconds:
                episode["horizontal_evidence_seconds"] += delta
        episode["previous_horizontal_ns"] = now if horizontal else None
        if horizontal and not episode["horizontal_observed"]:
            episode["horizontal_observed"] = True
            self._episode_event(observation, "same_track_horizontal_observed")
        _, reference_y, _, reference_h = episode["reference_bbox"]
        upright = (
            blocked is None
            and aspect <= 0.8
            and h >= 0.85 * reference_h
            and y <= reference_y + 0.15 * reference_h
        )
        if episode["last_seen_ns"] is None or (
            now - episode["last_seen_ns"] > self.policy.maximum_gap_seconds * 1e9
        ):
            episode["upright_since_ns"] = None
        episode["last_seen_ns"] = now
        if not upright:
            episode["upright_since_ns"] = None
        elif episode["upright_since_ns"] is None:
            episode["upright_since_ns"] = now
        elif (
            self.state in {"UNRESOLVED", "ASSISTANCE_REQUIRED"}
            and now - episode["upright_since_ns"] >= self.policy.recovery_seconds * 1e9
        ):
            episode["incident_status"] = (
                "recovery_observed" if episode["horizontal_observed"] else "posture_change_cleared"
            )
            self._episode_event(observation, "same_track_sustained_reference_posture")
            self._transition("RECOVERY", observation, episode["incident_status"])
            self.episode = None
            if self.s_probe is not None:
                self.s_probe = VisibilityProbe()
                self.s_probe.change = AccumulatedDescent(
                    policy=replace(self.s_probe.change.policy, maximum_missing_gap_seconds=1.1)
                )
            if self.loss_probe is not None:
                self.loss_probe = LossProbe()
            self._transition("NORMAL", observation, "episode_resolved")
            return
        threshold = self.policy.experimental_alert_after_seconds
        if (
            threshold is not None
            and not episode["test_alert_emitted"]
            and horizontal
            and self.state == "UNRESOLVED"
            and episode["horizontal_evidence_seconds"] >= threshold
        ):
            episode["test_alert_emitted"] = True
            episode["incident_status"] = "assistance_required_test"
            self._transition("ASSISTANCE_REQUIRED", observation, "horizontal_without_recovery_test")
            self._episode_event(observation, "TRIGGER_ALERT_TEST")

    def finish_episode(self):
        """Report unresolved evidence at session end without inventing a recovery."""
        if self.episode is None:
            return None
        self.episode["upright_since_ns"] = None
        if self.policy.experimental_episode_continuity:
            self.episode.update(
                evidence_unavailable=True, previous_horizontal_ns=None, last_seen_ns=None
            )
        return {
            **self.episode,
            "reason": "evidence_ended_episode_open",
            "episode_open": True,
        }

    def pop_episode_events(self):
        events, self.episode_events = self.episode_events, []
        return events

    def pop_transitions(self):
        transitions, self.transitions = self.transitions, []
        return transitions

    def pop_loss_outcomes(self):
        outcomes, self.loss_outcomes = self.loss_outcomes, []
        return outcomes

    def request_recent_buffer(self, observation):
        self.command_seq += 1
        end = observation["captured_mono_ns"]
        return envelope(
            "REQUEST_RECENT_BUFFER",
            self.identity,
            command_id=uid(),
            command_seq=self.command_seq,
            issued_at_utc=utc(),
            caused_by_observation_id=observation["observation_id"],
            based_on_frame_id=observation["frame_id"],
            from_edge_mono_ns=max(0, end - 5_000_000_000),
            to_edge_mono_ns=end,
            history_fps=20,
            reason="recover_pre_event_evidence",
            valid_until_edge_mono_ns=end + 5_000_000_000,
        )

    def request_offline_recovery(self, observation, start, end, *, archived=False):
        self.command_seq += 1
        return envelope(
            "REQUEST_RECENT_BUFFER",
            self.identity,
            command_id=uid(),
            command_seq=self.command_seq,
            issued_at_utc=utc(),
            caused_by_observation_id=observation["observation_id"],
            based_on_frame_id=observation["frame_id"],
            from_edge_mono_ns=start,
            to_edge_mono_ns=end,
            history_fps=5,
            reason="recover_archived_offline_evidence" if archived else "recover_offline_evidence",
            valid_until_edge_mono_ns=observation["captured_mono_ns"] + 60_000_000_000,
        )

    def request_roi(
        self, observation, bbox, output_width=640, output_height=360, *,
        padding_fraction=0.15, reason="inspect_confirmed_person_detail",
    ):
        if observation.get("view_kind", "full_frame") != "full_frame":
            raise ValueError("ROI must be based on a full-frame observation")
        source_width = observation["source_width"]
        source_height = observation["source_height"]
        x, y, width, height = bbox
        if not math.isfinite(padding_fraction) or not 0 <= padding_fraction <= 1:
            raise ValueError("invalid ROI padding fraction")
        pad_x, pad_y = round(width * padding_fraction), round(height * padding_fraction)
        scale_x = source_width / observation["width"]
        scale_y = source_height / observation["height"]
        x1 = max(0, math.floor((x - pad_x) * scale_x))
        y1 = max(0, math.floor((y - pad_y) * scale_y))
        x2 = min(source_width, math.ceil((x + width + pad_x) * scale_x))
        y2 = min(source_height, math.ceil((y + height + pad_y) * scale_y))
        self.baseline_resolution = (observation["width"], observation["height"])
        self.command_seq += 1
        self.visual_pending = envelope(
            "REQUEST_ROI",
            self.identity,
            command_id=uid(),
            command_seq=self.command_seq,
            issued_at_utc=utc(),
            caused_by_observation_id=observation["observation_id"],
            based_on_frame_id=observation["frame_id"],
            source_width=source_width,
            source_height=source_height,
            roi_x=x1,
            roi_y=y1,
            roi_width=x2 - x1,
            roi_height=y2 - y1,
            output_width=output_width,
            output_height=output_height,
            reason=reason,
            valid_until_edge_mono_ns=observation["captured_mono_ns"] + 5_000_000_000,
        )
        return self.visual_pending

    def request_full_frame(
        self, observation, output_width=None, output_height=None, *, reason="return_to_full_frame"
    ):
        if output_width is None or output_height is None:
            output_width, output_height = self.baseline_resolution
        if self.state == "VERIFYING":
            # Verify fresh full-scene evidence at the existing escalated FPS.
            self.quiet_since = self.escalated_since = None
            self.new_observations = 0
        self.command_seq += 1
        self.visual_pending = envelope(
            "REQUEST_FULL_FRAME",
            self.identity,
            command_id=uid(),
            command_seq=self.command_seq,
            issued_at_utc=utc(),
            caused_by_observation_id=observation["observation_id"],
            based_on_frame_id=observation["frame_id"],
            output_width=output_width,
            output_height=output_height,
            reason=reason,
            valid_until_edge_mono_ns=observation["captured_mono_ns"] + 5_000_000_000,
        )
        return self.visual_pending

    def _command(self, observation, fps, reason, *, next_state=None):
        self.command_seq += 1
        self._transition(
            next_state or ("VERIFYING" if fps != 2 else "RECOVERY"), observation, reason
        )
        self.high_since = self.quiet_since = self.escalated_since = None
        self.recovery_since = None
        self.new_observations = 0
        self.pending = envelope(
            "SET_TRANSMIT_FPS",
            self.identity,
            command_id=uid(),
            command_seq=self.command_seq,
            issued_at_utc=utc(),
            caused_by_observation_id=observation["observation_id"],
            based_on_frame_id=observation["frame_id"],
            transmit_fps=fps,
            reason=reason,
            valid_until_edge_mono_ns=observation["captured_mono_ns"] + 5_000_000_000,
        )
        return self.pending

    def _transition(self, new_state, observation, reason):
        if new_state == self.state:
            return
        previous = self.state
        self.state = new_state
        self.transitions.append(
            {
                "from_state": previous,
                "to_state": new_state,
                "reason": reason,
                "observation_id": observation["observation_id"],
                "frame_id": observation["frame_id"],
                "captured_mono_ns": observation["captured_mono_ns"],
            }
        )
