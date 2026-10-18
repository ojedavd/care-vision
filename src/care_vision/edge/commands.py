"""Apply a command once, in order, only to the current edge session."""

from collections import OrderedDict

from care_vision.common.contracts import RATES, envelope, validate
from care_vision.edge.representation import VisualPolicy
from care_vision.edge.sampler import Sampler


class CommandHandler:
    def __init__(
        self,
        identity: dict,
        sampler: Sampler,
        capture_fps: float,
        request_buffer=None,
        visual_policy=None,
    ):
        self.request_buffer = request_buffer
        self.identity = identity
        self.sampler = sampler
        self.capture_fps = capture_fps
        self.visual = visual_policy or VisualPolicy()
        self.command_id = None
        self.policy_seq = 0
        self.policy_effective_from_capture_seq = 1
        self.highest_seq = 0
        self.cache = OrderedDict()

    def apply(self, command: dict, now_ns: int, capture_seq: int) -> dict:
        validate(command)
        if command["type"] not in {
            "SET_TRANSMIT_FPS",
            "REQUEST_RECENT_BUFFER",
            "REQUEST_ROI",
            "REQUEST_FULL_FRAME",
        }:
            raise ValueError("expected command")
        result, reason = "rejected", "wrong_session"
        extra = {}
        same_session = all(command.get(k) == v for k, v in self.identity.items())
        if same_session:
            cached = self.cache.get(command["command_id"])
            if cached:
                original, ack = cached
                if command != original:
                    reason = "command_id_reused_with_different_payload"
                else:
                    return {
                        **ack,
                        "result": "duplicate" if ack["result"] == "applied" else ack["result"],
                    }
            elif command["command_seq"] <= self.highest_seq:
                result, reason = "stale", "older_sequence"
            else:
                self.highest_seq = command["command_seq"]
                if now_ns > command["valid_until_edge_mono_ns"]:
                    result, reason = "expired", "delivery_deadline_passed"
                elif command["type"] == "REQUEST_RECENT_BUFFER":
                    if command["to_edge_mono_ns"] > now_ns:
                        reason = "future_buffer_window"
                    elif command["history_fps"] > self.capture_fps:
                        reason = "rate_exceeds_capture"
                    elif self.request_buffer is None:
                        reason = "buffer_unavailable"
                    else:
                        extra = self.request_buffer(command)
                        if extra is None:
                            reason = "buffer_busy"
                        else:
                            result, reason = "applied", "buffer_snapshot_queued"
                            extra.update(
                                applied_at_edge_mono_ns=now_ns,
                                effective_from_capture_seq=capture_seq + 1,
                            )
                elif command["type"] in {"REQUEST_ROI", "REQUEST_FULL_FRAME"}:
                    if (
                        command.get("source_width", self.visual.source_width)
                        != self.visual.source_width
                    ):
                        reason = "source_dimensions_mismatch"
                    elif (
                        command.get("source_height", self.visual.source_height)
                        != self.visual.source_height
                    ):
                        reason = "source_dimensions_mismatch"
                    else:
                        self.visual.apply(command, capture_seq + 1)
                        result, reason = "applied", "visual_policy_updated"
                        extra = {
                            "applied_at_edge_mono_ns": now_ns,
                            "effective_from_capture_seq": capture_seq + 1,
                        }
                elif command["transmit_fps"] not in RATES:
                    reason = "unsupported_rate"
                elif command["transmit_fps"] > self.capture_fps:
                    reason = "rate_exceeds_capture"
                else:
                    self.sampler.set_rate(command["transmit_fps"], now_ns)
                    self.command_id = command["command_id"]
                    self.policy_seq = command["command_seq"]
                    self.policy_effective_from_capture_seq = capture_seq + 1
                    result, reason = "applied", "sampler_updated"
                    extra = {
                        "applied_at_edge_mono_ns": now_ns,
                        "effective_from_capture_seq": capture_seq + 1,
                    }
        ack = envelope(
            "COMMAND_ACK",
            self.identity,
            command_type=command["type"],
            command_id=command["command_id"],
            command_seq=command["command_seq"],
            result=result,
            reason=reason,
            active_transmit_fps=self.sampler.fps,
            active_policy_seq=self.policy_seq,
            active_view_kind=self.visual.view_kind,
            active_context_fps=self.visual.context_fps,
            active_visual_policy_seq=self.visual.policy_seq,
            active_output_width=self.visual.output_width,
            active_output_height=self.visual.output_height,
            applied_visual_command_id=self.visual.command_id,
            **(extra or {}),
        )
        if same_session and command["command_id"] not in self.cache:
            self.cache[command["command_id"]] = (dict(command), ack)
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        return ack
