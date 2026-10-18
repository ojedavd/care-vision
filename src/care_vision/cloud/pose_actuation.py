"""ACK/evidence state for one optional sensing cycle, never an incident verdict."""

from care_vision.common.contracts import ProtocolError


class PoseObservationCycle:
    def __init__(self, proposal, observation, command, now):
        self.proposal, self.origin = proposal, observation
        self.episode_id = proposal["episode_id"]
        self.roi_command = command
        self.return_command = None
        self.phase = "await_roi_ack"
        self.phase_since = now
        self.roi_ack_seq = self.return_ack_seq = None
        self.roi_frames = 0
        self.last_roi_seq = observation["capture_seq"]
        self.return_reason = None

    @property
    def active(self):
        return self.phase != "complete"

    def expect_return(self, command, now, reason):
        self.return_command = command
        self.phase, self.phase_since = "await_full_frame_ack", now
        self.return_reason = reason

    def acknowledge(self, ack, now):
        command = self.roi_command if self.phase == "await_roi_ack" else self.return_command
        if command is None or ack["command_id"] != command["command_id"]:
            return False
        if self.phase not in {"await_roi_ack", "await_full_frame_ack"}:
            return False
        if (ack.get("active_output_width"), ack.get("active_output_height")) != (
            command["output_width"], command["output_height"],
        ) or type(ack.get("effective_from_capture_seq")) is not int:
            raise ProtocolError("pose sensing ACK has incompatible output or capture boundary")
        boundary = (
            self.origin["capture_seq"] if self.phase == "await_roi_ack" else self.last_roi_seq
        )
        if ack["effective_from_capture_seq"] <= boundary:
            raise ProtocolError("pose sensing ACK predates its evidence boundary")
        if self.phase == "await_roi_ack":
            self.roi_ack_seq = ack["effective_from_capture_seq"]
            self.phase = "observing_roi"
        else:
            self.return_ack_seq = ack["effective_from_capture_seq"]
            self.phase = "await_full_frame_evidence"
        self.phase_since = now
        return True

    def observe(self, observation, now):
        if observation.get("evidence_kind") != "live":
            return None
        if self.phase == "observing_roi":
            command = self.roi_command
            matches = (
                observation.get("view_kind") == "roi"
                and observation.get("applied_visual_command_id") == command["command_id"]
                and observation.get("visual_policy_seq") == command["command_seq"]
                and observation.get("source_roi") == self.proposal["plan"]["request"]["roi"]
                and (observation.get("source_width"), observation.get("source_height"))
                == (command["source_width"], command["source_height"])
                and observation["capture_seq"] >= self.roi_ack_seq
                and observation["capture_seq"] > self.last_roi_seq
                and observation["captured_mono_ns"] > self.origin["captured_mono_ns"]
                and (observation["width"], observation["height"])
                == (command["output_width"], command["output_height"])
                and observation.get("person_detection_status") == "ok"
                and observation.get("person_tracking_status") == "ok"
            )
            if matches:
                self.last_roi_seq = observation["capture_seq"]
                self.roi_frames += 1
                return "roi_evidence_processed"
        elif self.phase == "await_full_frame_evidence":
            command = self.return_command
            if (
                observation.get("valid") is True
                and observation.get("view_kind") == "full_frame"
                and (observation.get("source_width"), observation.get("source_height"))
                == (self.origin["source_width"], self.origin["source_height"])
                and observation.get("applied_visual_command_id") == command["command_id"]
                and observation.get("visual_policy_seq") == command["command_seq"]
                and observation["capture_seq"] >= self.return_ack_seq
                and observation["capture_seq"] > self.last_roi_seq
                and observation["captured_mono_ns"] > self.origin["captured_mono_ns"]
                and (observation["width"], observation["height"])
                == (command["output_width"], command["output_height"])
                and observation.get("person_detection_status") == "ok"
                and observation.get("person_tracking_status") == "ok"
            ):
                self.phase, self.phase_since = "complete", now
                return "cycle_complete" if self.roi_frames else "restored_without_roi_evidence"
        return None

    def return_due(self, now, *, draining=False):
        if self.phase != "observing_roi":
            return None
        policy = self.proposal["plan"]["return"]
        if draining:
            return "drain_requires_full_scene"
        if self.roi_frames >= policy["or_after_roi_frames"]:
            return "pose_roi_frame_budget_complete"
        if now - self.phase_since >= policy["after_applied_ack_seconds"]:
            return "pose_roi_time_budget_complete"
        return None

    def check_timeout(self, now):
        limit = 3 if self.phase == "await_full_frame_evidence" else 5
        if self.phase in {"await_roi_ack", "await_full_frame_ack", "await_full_frame_evidence"}:
            if now - self.phase_since >= limit:
                raise ProtocolError(
                    "pose sensing " + self.phase + " timed out; evidence unresolved"
                )
