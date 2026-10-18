import pytest

from care_vision.cloud.controller import Controller


def observation(t, score=0.1, command_id=None, valid=True):
    return {
        "captured_mono_ns": int(t * 1e9),
        "motion_score": score,
        "valid": valid,
        "applied_command_id": command_id,
        "observation_id": f"O{t}",
        "frame_id": f"F{t}",
        "view_kind": "full_frame",
        "source_width": 640,
        "source_height": 360,
        "width": 640,
        "height": 360,
    }


def escalate(controller):
    assert controller.observe(observation(1)) is None
    assert controller.state == "SUSPICIOUS"
    command = controller.observe(observation(1.5))
    assert command["transmit_fps"] == 20
    assert controller.state == "VERIFYING"
    controller.acknowledge(
        {
            "command_id": command["command_id"],
            "result": "applied",
            "active_policy_seq": 1,
            "active_transmit_fps": 20,
        }
    )
    return command["command_id"]


def test_full_policy_needs_ack_and_new_evidence(identity):
    controller = Controller(identity)
    controller.observe(observation(1))
    command = controller.observe(observation(1.5))
    assert command["transmit_fps"] == 20
    assert controller.observe(observation(2, 0)) is None  # No ACK yet.
    controller.acknowledge(
        {
            "command_id": command["command_id"],
            "result": "applied",
            "active_policy_seq": 1,
            "active_transmit_fps": 20,
        }
    )
    assert controller.observe(observation(2.5, 0)) is None  # Old policy evidence.
    result = None
    for i in range(11):
        result = controller.observe(observation(3 + i * 0.5, 0, command["command_id"]))
        if i < 10:
            assert result is None
    assert result["transmit_fps"] == 2
    assert result["caused_by_observation_id"] == "O8.0"
    assert controller.state == "RECOVERY"
    assert controller.observe(observation(8.5, 0, result["command_id"])) is None
    controller.acknowledge(
        {
            "command_id": result["command_id"],
            "result": "applied",
            "active_policy_seq": result["command_seq"],
            "active_transmit_fps": 2,
        }
    )
    for t in (9, 9.5, 10, 10.5, 11):
        assert controller.observe(observation(t, 0, result["command_id"])) is None
    assert controller.state == "NORMAL"


def test_missing_evidence_never_means_quiet(identity):
    controller = Controller(identity)
    cid = escalate(controller)
    for i in range(10):
        assert controller.observe(observation(2 + i * 0.5, 0.1, cid)) is None
    for t in (7, 7.5, 8, 8.5):
        assert controller.observe(observation(t, 0, cid)) is None
    assert controller.observe(observation(9, None, cid, valid=False)) is None
    for t in (9.5, 10, 10.5, 11, 11.5, 12):
        assert controller.observe(observation(t, 0, cid)) is None
    assert controller.observe(observation(12.5, 0, cid))["transmit_fps"] == 2


def test_gap_resets_suspicion(identity):
    controller = Controller(identity)
    assert controller.observe(observation(1)) is None
    assert controller.observe(observation(10)) is None
    assert controller.observe(observation(10.5))["transmit_fps"] == 20


def test_static_observations_never_escalate(identity):
    controller = Controller(identity)
    assert all(controller.observe(observation(i * 0.5, 0)) is None for i in range(100))


def test_state_transitions_are_explicit_and_causal(identity):
    controller = Controller(identity)
    command_id = escalate(controller)
    transitions = controller.pop_transitions()
    assert [(item["from_state"], item["to_state"]) for item in transitions] == [
        ("NORMAL", "SUSPICIOUS"),
        ("SUSPICIOUS", "VERIFYING"),
    ]
    assert transitions[0]["observation_id"] == "O1"
    assert transitions[1]["observation_id"] == "O1.5"
    for t in (2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8):
        command = controller.observe(observation(t, 0, command_id))
        if command:
            break
    assert command["transmit_fps"] == 2
    assert controller.pop_transitions()[0]["to_state"] == "RECOVERY"


def test_roi_command_is_padded_clamped_and_acknowledged_separately(identity):
    controller = Controller(identity)
    request = controller.request_roi(observation(2), [0, 30, 200, 300])

    assert request["type"] == "REQUEST_ROI"
    assert request["roi_x"] == 0
    assert request["roi_y"] == 0
    assert request["roi_width"] == 230
    assert request["roi_height"] == 360
    assert controller.acknowledge_visual(
        {
            "command_id": request["command_id"],
            "result": "applied",
            "active_visual_policy_seq": request["command_seq"],
            "active_view_kind": "roi",
        }
    )
    assert controller.active_view_kind == "roi"


def test_roi_cannot_be_based_on_another_roi(identity):
    controller = Controller(identity)
    with pytest.raises(ValueError, match="full-frame"):
        controller.request_roi({**observation(2), "view_kind": "roi"}, [10, 10, 20, 20])


def test_roi_maps_reduced_observation_to_high_resolution_source(identity):
    controller = Controller(identity)
    reduced = {
        **observation(2),
        "source_width": 1280,
        "source_height": 720,
        "width": 320,
        "height": 180,
    }
    request = controller.request_roi(reduced, [100, 20, 80, 120])
    assert [request[key] for key in ("roi_x", "roi_y", "roi_width", "roi_height")] == [
        352,
        8,
        416,
        624,
    ]
    restored = controller.request_full_frame({**reduced, "view_kind": "roi"})
    assert (restored["output_width"], restored["output_height"]) == (320, 180)


def verifying_roi(controller):
    cid = escalate(controller)
    roi = controller.request_roi(observation(1.5), [100, 20, 80, 120])
    controller.acknowledge_visual({
        "command_id": roi["command_id"], "result": "applied",
        "active_visual_policy_seq": roi["command_seq"], "active_view_kind": "roi",
    })
    return cid, roi["command_id"]


@pytest.mark.parametrize("confirmed", [False, True])
def test_roi_quiet_requires_full_frame_before_lowering_fps(identity, confirmed):
    controller = Controller(identity)
    cid, rid = verifying_roi(controller)
    command = None
    for i in range(12):
        command = controller.observe({
            **observation(2 + i * 0.5, 0, cid), "view_kind": "roi",
            "applied_visual_command_id": rid, "confirmed_person_detected": confirmed,
        })
        if command:
            break
    assert command["type"] == "REQUEST_FULL_FRAME"
    assert controller.state == "VERIFYING"
    assert controller.pending is None  # The acknowledged 20 FPS policy stays active.
    assert command["reason"] == (
        "verify_scene_before_return_to_baseline" if confirmed else "roi_person_unconfirmed"
    )


def test_full_frame_ack_and_fresh_quiet_required_after_losing_roi_person(identity):
    controller = Controller(identity)
    cid, rid = verifying_roi(controller)
    roi = {**observation(2, 0, cid), "view_kind": "roi",
           "applied_visual_command_id": rid, "confirmed_person_detected": False}
    request = controller.observe(roi)
    assert request and request["type"] == "REQUEST_FULL_FRAME"
    for i in range(1, 15):
        assert controller.observe({**roi, "captured_mono_ns": int((2 + i * .5) * 1e9)}) is None
    controller.acknowledge_visual({
        "command_id": request["command_id"], "result": "applied",
        "active_visual_policy_seq": request["command_seq"], "active_view_kind": "full_frame",
    })
    for i in range(15, 30):
        assert controller.observe({**roi, "captured_mono_ns": int((2 + i * .5) * 1e9)}) is None
    assert controller.state == "VERIFYING"
    # Restoring a view warms up motion context; neither stale ROI nor invalid
    # full-frame samples can count toward the quiet interval.
    def full(t, score=0, valid=True):
        return {**observation(t, score, cid, valid),
                "applied_visual_command_id": request["command_id"]}
    assert controller.observe(full(17, None, False)) is None
    assert controller.observe(full(17.5, .15)) is None  # Movement outside the old ROI.
    for i in range(10):
        result = controller.observe(full(18 + i * .5))
        if i < 9:
            assert result is None
    assert result["type"] == "SET_TRANSMIT_FPS" and result["transmit_fps"] == 2
    assert controller.state == "RECOVERY"
