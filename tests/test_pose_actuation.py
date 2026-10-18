import asyncio
import time

import pytest
from test_pose_observation_plan import inputs
from test_pose_shadow import processor

from care_vision.cloud.controller import Controller, Policy
from care_vision.cloud.episode_store import EpisodeStore
from care_vision.cloud.pose_observation_plan import PoseObservationPlanner
from care_vision.cloud.pose_shadow import PoseShadowObserver
from care_vision.cloud.server import CloudServer, Session
from care_vision.common.contracts import ProtocolError, loads
from care_vision.common.metrics import EventLog
from care_vision.edge.commands import CommandHandler
from care_vision.edge.representation import VisualPolicy
from care_vision.edge.sampler import Sampler


class Socket:
    def __init__(self):
        self.sent = []

    async def send(self, raw):
        self.sent.append(loads(raw))


def session(tmp_path, identity):
    _, obs, diagnostic, ep = inputs()
    obs.update(applied_command_id=None, applied_visual_command_id=None,
               visual_policy_seq=0, target_transmit_fps=2)
    log = EventLog("cloud", tmp_path / "cloud.jsonl")
    shadow = PoseShadowObserver(log, processor())
    s = Session(Socket(), {**identity, "edge_id": "edge"}, log,
                experimental_s_only_5fps=True, pose_shadow_factory=lambda _: shadow,
                experimental_pose_observation=True)
    s.controller.episode, s.controller.state = ep, "UNRESOLVED"
    s.pose_latest_observation = obs
    shadow.latest_result = (obs, diagnostic, time.monotonic())
    edge = CommandHandler(s.identity, Sampler(2), capture_fps=25,
                          visual_policy=VisualPolicy(1280, 720, 1280, 720))
    return s, obs, diagnostic, edge, log


def apply(s, edge, command, obs):
    ack = edge.apply(command, obs["captured_mono_ns"], obs["capture_seq"])
    assert s.controller.acknowledge_visual(ack)
    assert s.pose_cycle.acknowledge(ack, time.monotonic())
    return ack


def roi_observation(obs, command, offset=1):
    return {
        **obs, "capture_seq": obs["capture_seq"] + offset,
        "captured_mono_ns": obs["captured_mono_ns"] + offset * 200_000_000,
        "view_kind": "roi", "width": 640, "height": 360,
        "applied_visual_command_id": command["command_id"],
        "visual_policy_seq": command["command_seq"],
        "source_roi": [command[k] for k in ("roi_x", "roi_y", "roi_width", "roi_height")],
    }


def test_real_command_handler_roi_ack_four_new_frames_return_ack_then_fresh_full_frame(
    tmp_path, identity,
):
    async def scenario():
        s, obs, diagnostic, edge, log = session(tmp_path, identity)
        try:
            await s.pose_observation_tick()
            command = s.websocket.sent[-1]
            assert command["type"] == "REQUEST_ROI"
            assert s.controller.episode["pose_observation_requested"]
            apply(s, edge, command, obs)
            for i in range(1, 5):
                row = roi_observation(obs, command, i)
                assert s.pose_cycle.observe(row, time.monotonic()) == "roi_evidence_processed"
                s.pose_latest_observation = row
            await s.pose_observation_tick()
            restore = s.websocket.sent[-1]
            assert restore["type"] == "REQUEST_FULL_FRAME"
            apply(s, edge, restore, row)
            assert s.pose_cycle.active  # ACK alone is not new evidence.
            full = {**obs, "capture_seq": row["capture_seq"] + 1,
                    "captured_mono_ns": row["captured_mono_ns"] + 500_000_000,
                    "visual_policy_seq": restore["command_seq"],
                    "applied_visual_command_id": restore["command_id"]}
            assert s.pose_cycle.observe({**full, "valid": False}, time.monotonic()) is None
            assert s.pose_cycle.observe(full, time.monotonic()) == "cycle_complete"
            assert s.controller.episode is not None and s.controller.state == "UNRESOLVED"
            s.pose_latest_observation = full
            diagnostic = {**diagnostic, **{k: full[k] for k in (
                "observation_id", "frame_id", "capture_seq", "captured_mono_ns",
            )}}
            s.pose_shadow.latest_result = (full, diagnostic, time.monotonic())
            await s.pose_observation_tick()
            assert len(s.websocket.sent) == 2
        finally:
            log.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["old_capture", "command", "crop", "source", "historical"])
def test_unrelated_evidence_does_not_spend_roi_frame_budget(tmp_path, identity, fault):
    async def scenario():
        s, obs, _, edge, log = session(tmp_path, identity)
        try:
            await s.pose_observation_tick()
            command = s.websocket.sent[-1]
            apply(s, edge, command, obs)
            row = roi_observation(obs, command)
            if fault == "old_capture":
                row["capture_seq"] = obs["capture_seq"]
            elif fault == "command":
                row["applied_visual_command_id"] = "old"
            elif fault == "crop":
                row["source_roi"] = [0, 0, 20, 20]
            elif fault == "source":
                row["source_width"] = 1920
            else:
                row["evidence_kind"] = "historical"
            assert s.pose_cycle.observe(row, time.monotonic()) is None
            assert s.pose_cycle.roi_frames == 0
        finally:
            log.close()

    asyncio.run(scenario())


def test_missing_evidence_restores_on_time_budget_and_missing_ack_times_out(tmp_path, identity):
    async def scenario():
        s, obs, _, edge, log = session(tmp_path, identity)
        try:
            await s.pose_observation_tick()
            command = s.websocket.sent[-1]
            with pytest.raises(ProtocolError):
                s.pose_cycle.check_timeout(s.pose_cycle.phase_since + 5)
            apply(s, edge, command, obs)
            since = s.pose_cycle.phase_since
            assert s.pose_cycle.return_due(since + 2) == "pose_roi_time_budget_complete"
            assert s.pose_cycle.return_due(since, draining=True) == "drain_requires_full_scene"
            ack = {"command_id": command["command_id"]}
            assert not s.pose_cycle.acknowledge(ack, since + 1)  # Duplicate cannot reset clock.
            assert s.pose_cycle.phase_since == since
        finally:
            log.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["old_result", "closed_episode", "pending", "unlinked"])
def test_result_cannot_trigger_after_state_or_age_changes(tmp_path, identity, fault):
    async def scenario():
        s, obs, diagnostic, _, log = session(tmp_path, identity)
        try:
            if fault == "old_result":
                s.pose_shadow.latest_result = (obs, diagnostic, time.monotonic() - 2)
            elif fault == "closed_episode":
                s.controller.episode = None
            elif fault == "pending":
                s.controller.pending = {"command_id": "busy"}
            else:
                s.controller.episode["identity_link_status"] = "unresolved_after_reconnect"
            await s.pose_observation_tick()
            assert not s.websocket.sent
        finally:
            log.close()

    asyncio.run(scenario())


def test_option_requires_shadow_and_tested_profile(tmp_path):
    log = EventLog("cloud", tmp_path / "log")
    try:
        with pytest.raises(ValueError):
            CloudServer(log, experimental_pose_observation=True)
        assert not CloudServer(log).experimental_pose_observation
    finally:
        log.close()


@pytest.mark.parametrize("bad_ack", ["dimensions", "capture_boundary"])
def test_ack_must_match_output_and_postdate_origin(tmp_path, identity, bad_ack):
    async def scenario():
        s, obs, _, edge, log = session(tmp_path, identity)
        try:
            await s.pose_observation_tick()
            command = s.websocket.sent[-1]
            ack = edge.apply(command, obs["captured_mono_ns"], obs["capture_seq"])
            key = "active_output_width" if bad_ack == "dimensions" else "effective_from_capture_seq"
            ack[key] = 0
            with pytest.raises(ProtocolError):
                s.pose_cycle.acknowledge(ack, time.monotonic())
            assert s.pose_cycle.phase == "await_roi_ack"
        finally:
            log.close()

    asyncio.run(scenario())


def test_return_without_roi_is_distinguished_and_full_evidence_timeout_is_bounded(
    tmp_path, identity,
):
    async def scenario():
        s, obs, _, edge, log = session(tmp_path, identity)
        try:
            await s.pose_observation_tick()
            apply(s, edge, s.websocket.sent[-1], obs)
            restore = s.controller.request_full_frame(obs)
            await s.dispatch(restore, obs)  # Existing controller can restore early.
            assert s.pose_cycle.return_command == restore
            apply(s, edge, restore, obs)
            with pytest.raises(ProtocolError):
                s.pose_cycle.check_timeout(s.pose_cycle.phase_since + 3)
            full = {**obs, "capture_seq": obs["capture_seq"] + 1,
                    "captured_mono_ns": obs["captured_mono_ns"] + 500_000_000,
                    "applied_visual_command_id": restore["command_id"],
                    "visual_policy_seq": restore["command_seq"]}
            assert s.pose_cycle.observe(full, time.monotonic()) == "restored_without_roi_evidence"
        finally:
            log.close()

    asyncio.run(scenario())


def test_episode_budget_survives_private_restart(tmp_path, identity):
    planner, obs, diagnostic, ep = inputs()
    policy = Policy(experimental_s_only_5fps=True, experimental_episode_continuity=True)
    controller = Controller(identity, policy)
    controller._open_episode(obs, ep["track_id"], [400, 40, 160, 500])
    controller.episode["pose_observation_requested"] = True
    store = EpisodeStore(tmp_path / "episode.json")
    store.save(controller)
    restored = Controller({**identity, "session_id": "new"}, policy)
    assert EpisodeStore(store.path).restore(restored)
    assert restored.episode["pose_observation_requested"] is True
    # Even a future separately verified identity link cannot replenish the budget.
    restored.episode["identity_link_status"] = "session_track_only"
    diagnostic["episode_id"] = restored.episode["episode_id"]
    fresh_planner = PoseObservationPlanner(planner.geometry, planner.hips)
    assert fresh_planner.consider(obs, diagnostic, restored.episode, result_age_seconds=0)[
        "reason"
    ] == "episode_observation_budget_spent"


def test_async_observer_mailbox_reaches_session_without_model_access_to_commands(
    tmp_path, identity,
):
    async def scenario():
        s, obs, _, _, log = session(tmp_path, identity)
        try:
            for t in (1, 1.5, 2, 2.5, 3):
                row = {**obs, "observation_id": f"o-{t}", "frame_id": f"f-{t}",
                       "captured_mono_ns": round(t * 1e9), "capture_seq": round(t * 25)}
                s.pose_latest_observation = row
                s.pose_shadow.submit(row, b"synthetic", s.controller.episode)
                await s.pose_shadow.queue.join()
            assert s.pose_shadow.latest_result[0]["capture_seq"] == 75
            await s.pose_observation_tick()
            assert s.websocket.sent[-1]["type"] == "REQUEST_ROI"
            assert s.pose_shadow.latest_result is None
            assert not hasattr(s.pose_shadow.processor, "controller")
        finally:
            await s.pose_shadow.close()
            log.close()

    asyncio.run(scenario())
