import asyncio
import time

from care_vision.cloud.controller import Controller, Policy
from care_vision.cloud.server import Session
from care_vision.common.contracts import loads
from care_vision.common.metrics import EventLog


def observation(
    t, *, present=True, rate=2, width=320, height=180, fps_command=None, view_command=None, top=20
):
    person = {
        "track_state": "confirmed",
        "track_id": "track-a",
        "tracked_bbox": [40, top, 25, 80],
    }
    return {
        "captured_mono_ns": round(t * 1e9),
        "motion_score": 0,
        "valid": True,
        "applied_command_id": fps_command,
        "applied_visual_command_id": view_command,
        "observation_id": f"O-{t}",
        "frame_id": f"F-{t}",
        "capture_seq": round(t * 30),
        "view_kind": "full_frame",
        "source_width": 1920,
        "source_height": 1080,
        "width": width,
        "height": height,
        "target_transmit_fps": rate,
        "person_detection_status": "ok",
        "person_tracking_status": "ok",
        "confirmed_person_count": int(present),
        "persons": [person] if present else [],
        "primary_person_bbox": [160, top, 100, 320] if present else None,
    }


def test_s_only_requests_five_fps_on_same_track_descent_and_returns_to_two(identity):
    controller = Controller(identity, Policy(experimental_s_only_5fps=True))
    for t, top, height in [(1, 20, 80), (1.5, 40, 65)]:
        item = observation(t, width=1280, height=720)
        item["persons"][0]["tracked_bbox"] = [40, top, 25, height]
        assert controller.observe(item) is None
    item = observation(2, width=1280, height=720)
    item["persons"][0]["tracked_bbox"] = [40, 65, 40, 40]
    up = controller.observe(item)
    assert up["type"] == "SET_TRANSMIT_FPS"
    assert up["transmit_fps"] == 5
    assert controller.visual_pending is None
    acknowledge(controller, up, rate=5, view="full_frame", resolution=(1280, 720))
    for i in range(32):
        t = 2.2 + i * 0.2
        item = observation(t, rate=5, width=1280, height=720, fps_command=up["command_id"])
        item.update(
            primary_track_id="track-a",
            confirmed_person_detected=True,
            primary_orientation="horizontal",
        )
        down = controller.observe(item)
        if down is not None:
            break
    assert down["transmit_fps"] == 2
    assert controller.pop_loss_outcomes()[0]["outcome"] == "horizontal_posture_observed"
    acknowledge(controller, down, rate=2, view="full_frame", resolution=(1280, 720))
    item = observation(9, width=1280, height=720, fps_command=down["command_id"])
    controller.observe(item)
    assert controller.state == "UNRESOLVED"


def acknowledge(controller, command, *, rate, view, resolution):
    if command["type"] == "SET_TRANSMIT_FPS":
        assert controller.acknowledge(
            {
                "command_id": command["command_id"],
                "result": "applied",
                "active_policy_seq": command["command_seq"],
                "active_transmit_fps": rate,
            }
        )
    else:
        assert controller.acknowledge_visual(
            {
                "command_id": command["command_id"],
                "result": "applied",
                "active_visual_policy_seq": command["command_seq"],
                "active_view_kind": view,
                "active_output_width": resolution[0],
                "active_output_height": resolution[1],
            }
        )


def test_loss_mode_requires_fresh_s_evidence_and_returns_to_two_fps(identity):
    controller = Controller(identity, Policy(experimental_loss_5fps=True))
    controller.observe(observation(1))
    controller.observe(observation(1.5, present=False))
    up = controller.observe(observation(2, present=False))
    assert up["transmit_fps"] == 5
    assert controller.baseline_resolution == (320, 180)
    acknowledge(controller, up, rate=5, view="full_frame", resolution=(320, 180))
    visual_up = controller.request_full_frame(observation(2), 1280, 720)
    acknowledge(controller, visual_up, rate=5, view="full_frame", resolution=(1280, 720))
    assert (
        controller.observe(observation(2.2, present=False, rate=5, fps_command=up["command_id"]))
        is None
    )
    for t, top in ((2.4, 80), (4.0, 100), (7.8, 250)):
        assert (
            controller.observe(
                observation(
                    t,
                    rate=5,
                    width=1280,
                    height=720,
                    fps_command=up["command_id"],
                    view_command=visual_up["command_id"],
                    top=top,
                )
            )
            is None
        )
    down = controller.observe(
        observation(
            8.1,
            rate=5,
            width=1280,
            height=720,
            fps_command=up["command_id"],
            view_command=visual_up["command_id"],
            top=255,
        )
    )
    assert down is None  # Downward evidence extends the window to 10 seconds.
    down = controller.observe(
        observation(
            12.1,
            rate=5,
            width=1280,
            height=720,
            fps_command=up["command_id"],
            view_command=visual_up["command_id"],
            top=255,
        )
    )
    assert down["transmit_fps"] == 2
    assert controller.state == "UNRESOLVED"
    assert down["reason"] == "loss_verification_posture_assessed_return_to_baseline"
    outcomes = controller.pop_loss_outcomes()
    assert len(outcomes) == 1
    assert outcomes[0]["outcome"] == "unresolved"
    assert outcomes[0]["incident_status"] == "unresolved"
    assert outcomes[0]["s_frames"] == 5
    assert outcomes[0]["s_person_frames"] == 5
    assert outcomes[0]["downward_extension_used"] is True
    assert controller.pop_loss_outcomes() == []
    acknowledge(controller, down, rate=2, view="full_frame", resolution=(1280, 720))
    visual_down = controller.request_full_frame(observation(12.1), 320, 180)
    acknowledge(controller, visual_down, rate=2, view="full_frame", resolution=(320, 180))
    controller.observe(
        observation(
            12.6,
            present=False,
            fps_command=down["command_id"],
            view_command=visual_down["command_id"],
        )
    )
    assert controller.state == "UNRESOLVED"


def test_loss_timeout_without_downward_evidence_is_unresolved(identity):
    controller = Controller(identity, Policy(experimental_loss_5fps=True))
    controller.observe(observation(1))
    controller.observe(observation(1.5, present=False))
    up = controller.observe(observation(2, present=False))
    acknowledge(controller, up, rate=5, view="full_frame", resolution=(320, 180))
    visual_up = controller.request_full_frame(observation(2), 1280, 720)
    acknowledge(controller, visual_up, rate=5, view="full_frame", resolution=(1280, 720))
    down = controller.observe(
        observation(
            8.1,
            present=False,
            rate=5,
            width=1280,
            height=720,
            fps_command=up["command_id"],
            view_command=visual_up["command_id"],
        )
    )
    assert down["transmit_fps"] == 2
    assert controller.state == "UNRESOLVED"
    assert controller.pop_loss_outcomes() == [
        {
            "outcome": "unresolved",
            "track_id": None,
            "horizontal_frames": 0,
            "vertical_frames": 0,
            "window_seconds": 2,
            "reason": "posture_evidence_at_verification_limit",
            "incident_status": "unresolved",
            "observation_id": "O-8.1",
            "frame_id": "F-8.1",
            "captured_mono_ns": 8_100_000_000,
            "elapsed_seconds": 6.1,
            "s_frames": 1,
            "s_person_frames": 0,
            "first_s_top": None,
            "max_s_top": None,
            "downward_extension_used": False,
        }
    ]


def test_session_dispatch_sends_five_fps_and_full_scene_without_alert(tmp_path, identity):
    class Socket:
        def __init__(self):
            self.sent = []

        async def send(self, data):
            self.sent.append(loads(data))

    async def scenario():
        socket = Socket()
        log = EventLog("cloud", tmp_path / "loss-dispatch.jsonl")
        try:
            session = Session(
                socket,
                identity,
                log,
                verification_detector_factory=lambda: object(),
                experimental_loss_5fps=True,
            )
            event = observation(time.monotonic(), present=False)
            command = session.controller._command(event, 5, "confirmed_person_missing_verify")
            await session.dispatch(command, event)
            assert [item["type"] for item in socket.sent] == [
                "SET_TRANSMIT_FPS",
                "REQUEST_FULL_FRAME",
            ]
            assert socket.sent[0]["transmit_fps"] == 5
            assert (socket.sent[1]["output_width"], socket.sent[1]["output_height"]) == (1280, 720)
            assert session.controller.pending == command
            assert session.controller.visual_pending == socket.sent[1]
        finally:
            log.close()

    asyncio.run(scenario())
