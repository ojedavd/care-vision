from conftest import command, roi_command

from care_vision.edge.commands import CommandHandler
from care_vision.edge.sampler import Sampler


def test_duplicate_does_not_reset_sampling_deadline(identity):
    sampler = Sampler()
    handler = CommandHandler(identity, sampler, 30)
    message = command(identity)
    ack = handler.apply(message, 100, 3)
    deadline = sampler.next_due
    duplicate = handler.apply(message, 200, 5)
    assert ack["result"] == "applied"
    assert duplicate["result"] == "duplicate"
    assert duplicate["effective_from_capture_seq"] == 4
    assert sampler.next_due == deadline
    changed = {**message, "transmit_fps": 2}
    assert handler.apply(changed, 300, 6)["result"] == "rejected"
    assert sampler.fps == 20


def test_expiry_stale_and_session_fencing(identity):
    handler = CommandHandler(identity, Sampler(), 30)
    assert handler.apply(command(identity, deadline=10), 11, 1)["result"] == "expired"
    newer = command(identity, seq=3, command_id="C3")
    assert handler.apply(newer, 20, 2)["result"] == "applied"
    old = command(identity, seq=2, command_id="C2", fps=2)
    assert handler.apply(old, 30, 3)["result"] == "stale"
    other = command({**identity, "session_id": "old-session"}, seq=4, command_id="C4", fps=2)
    assert handler.apply(other, 40, 4)["result"] == "rejected"
    assert handler.sampler.fps == 20


def test_unsupported_rate_and_insufficient_capture(identity):
    handler = CommandHandler(identity, Sampler(), 10)
    assert handler.apply(command(identity, fps=10), 1, 1)["reason"] == "unsupported_rate"
    high = command(identity, seq=2, command_id="C2")
    assert handler.apply(high, 2, 2)["reason"] == "rate_exceeds_capture"


def test_five_fps_can_be_applied_and_returns_to_baseline(identity):
    handler = CommandHandler(identity, Sampler(), 25)
    up = handler.apply(command(identity, fps=5), 1, 1)
    assert up["result"] == "applied"
    assert up["active_transmit_fps"] == 5
    assert handler.sampler.period == 200_000_000
    down = handler.apply(command(identity, seq=2, command_id="C2", fps=2), 2, 2)
    assert down["result"] == "applied"
    assert down["active_transmit_fps"] == 2


def test_roi_policy_is_separate_from_transmit_policy(identity):
    sampler = Sampler()
    handler = CommandHandler(identity, sampler, 30)
    fps = command(identity)
    assert handler.apply(fps, 1, 1)["result"] == "applied"
    roi = roi_command(identity, seq=2)
    ack = handler.apply(roi, 2, 2)

    assert ack["result"] == "applied"
    assert ack["active_transmit_fps"] == 20
    assert ack["active_policy_seq"] == 1
    assert ack["active_view_kind"] == "roi"
    assert ack["active_visual_policy_seq"] == 2
    assert handler.command_id == fps["command_id"]
    assert handler.visual.command_id == roi["command_id"]
    assert handler.policy_effective_from_capture_seq == 2
    assert handler.visual.effective_from_capture_seq == 3


def test_roi_rejects_wrong_source_dimensions(identity):
    handler = CommandHandler(identity, Sampler(), 30)
    ack = handler.apply({**roi_command(identity), "source_width": 1280}, 1, 1)
    assert ack["result"] == "rejected"
    assert ack["reason"] == "source_dimensions_mismatch"


def test_mixed_roi_ack_and_duplicate_preserve_context_schedule(identity):
    handler = CommandHandler(identity, Sampler(), 30)
    message = {**roi_command(identity), "context_fps": 2}
    ack = handler.apply(message, 1, 1)
    handler.visual.last_context_bucket = 4
    duplicate = handler.apply(message, 2, 2)
    assert ack["active_context_fps"] == duplicate["active_context_fps"] == 2
    assert duplicate["result"] == "duplicate"
    assert handler.visual.last_context_bucket == 4
