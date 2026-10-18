"""Combined real local WebSocket/Edge commands with scripted vision and pose.

These tests validate lifecycle and evidence contracts, not CV accuracy or WAN
resilience. A fresh CloudServer object exercises disk restore, not an OS restart.
"""

import asyncio
import copy
import hashlib
import json
import stat
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import status
from test_episode_continuity import observation
from test_pose_shadow import processor
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

import care_vision.cloud.server as server_module
from care_vision.check_loop import read_log
from care_vision.cloud.controller import Controller, Policy
from care_vision.cloud.episode_store import EpisodeStore
from care_vision.cloud.pose_shadow import PoseShadowObserver
from care_vision.cloud.server import CloudServer
from care_vision.common.contracts import dumps, envelope, loads, pack_frame, utc
from care_vision.common.metrics import EventLog
from care_vision.edge.commands import CommandHandler
from care_vision.edge.representation import VisualPolicy
from care_vision.edge.sampler import Sampler

STAGES = (
    "await_roi_ack", "roi_applied_ack_lost", "observing_roi", "roi_evidence",
    "await_full_frame_ack", "return_applied_ack_lost", "await_full_frame_evidence", "complete",
)


async def eventually(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.002)


def edge_for(identity):
    return CommandHandler(identity, Sampler(2), capture_fps=25,
                          visual_policy=VisualPolicy(1280, 720, 1280, 720))


class ScriptedVision:
    def __init__(self, *args, **kwargs):
        pass

    def observe(self, header, jpeg):
        # No image inference: a known standing -> contracted sequence creates
        # an episode. Every reconnected-session person is standing and unlinked.
        t = header["captured_mono_ns"] / 1e9
        box = (400, 200, 160, 300) if header["session_id"] == "first" and t >= 2 else (
            400, 40, 160, 500
        )
        result = observation(t, track="track:" + header["session_id"], box=box)
        result.update(header, observation_id=f"{header['session_id']}:{header['capture_seq']}",
                      valid=header.get("test_valid", True), evidence_kind="live")
        if header["view_kind"] == "roi":
            result["source_roi"] = [header[k] for k in
                                    ("roi_x", "roi_y", "roi_width", "roi_height")]
        return result


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("restart", ["socket", "service_object", "edge_reboot"])
def test_pose_cycle_disconnect_preserves_episode_budget_and_requires_new_evidence(
    tmp_path, monkeypatch, stage, restart,
):
    monkeypatch.setattr(server_module, "MotionVision", ScriptedVision)
    sessions = []
    real_session = server_module.Session

    class RecordingSession(real_session):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sessions.append(self)

    monkeypatch.setattr(server_module, "Session", RecordingSession)

    async def scenario():
        identity = {"edge_id": "edge", "boot_id": "boot", "session_id": "first"}
        log_path, snapshot_path = tmp_path / "cloud.jsonl", tmp_path / "episode.json"
        log = EventLog("cloud", log_path)
        _, encoded = cv2.imencode(".jpg", np.zeros((720, 1280, 3), dtype=np.uint8))
        source_jpeg = encoded.tobytes()

        def make_server():
            return CloudServer(
                log, experimental_s_only_5fps=True, experimental_episode_continuity=True,
                episode_state_file=snapshot_path,
                pose_shadow_factory=lambda stream: PoseShadowObserver(stream, processor()),
                experimental_pose_observation=True,
            )

        def records(event):
            return [r for r in read_log(log_path) if r["event"] == event]

        async def send_sample(socket, edge, t, seq, *, valid=True):
            now = round(t * 1e9)
            jpeg, metadata = edge.visual.render(source_jpeg, now)
            header = envelope(
                "FRAME", edge.identity, frame_id=f"{edge.identity['boot_id']}:{seq}",
                capture_seq=seq, captured_mono_ns=now, sampled_mono_ns=now,
                captured_at_utc=utc(), encoding="jpeg", payload_bytes=len(jpeg),
                width=edge.visual.output_width, height=edge.visual.output_height,
                applied_command_id=edge.command_id, policy_seq=edge.policy_seq,
                target_transmit_fps=edge.sampler.fps, test_valid=valid, **metadata,
            )
            await socket.send(pack_frame(header, jpeg))
            expected = f"{edge.identity['session_id']}:{seq}"
            await eventually(lambda: any(r["observation_id"] == expected
                                         for r in records("pose_shadow")))

        async def command(socket, kind):
            result = loads(await asyncio.wait_for(socket.recv(), 3))
            assert result["type"] == kind, result
            return result

        async def acknowledge(socket, edge, item, t, seq):
            ack = edge.apply(item, round(t * 1e9), seq)
            assert ack["result"] == "applied", ack
            await socket.send(dumps(ack))
            await eventually(lambda: any(r["command_id"] == item["command_id"]
                                         for r in records("ack_received")))
            return ack

        cloud = make_server()
        listener = await serve(cloud.handler, "127.0.0.1", 0)
        try:
            url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
            old_edge = edge_for(identity)
            async with connect(url, proxy=None) as first:
                await first.send(dumps(status(identity)))
                for seq in range(1, 5):
                    await send_sample(first, old_edge, seq * .5, seq)
                escalation = await command(first, "SET_TRANSMIT_FPS")
                assert escalation["transmit_fps"] == 5
                await acknowledge(first, old_edge, escalation, 2, 4)
                for seq in range(5, 9):
                    await send_sample(first, old_edge, seq * .5, seq)
                roi = await command(first, "REQUEST_ROI")
                await eventually(lambda: snapshot_path.exists())
                committed = json.loads(snapshot_path.read_text())["snapshot"]["episode"]
                assert committed["pose_observation_requested"] is True
                assert stat.S_IMODE(snapshot_path.stat().st_mode) == 0o600
                # The budget is durable before the Edge acknowledges the ROI.
                original = copy.deepcopy(committed)
                old_session = sessions[-1]
                seq, t = 8, 4.0
                if stage == "roi_applied_ack_lost":
                    lost_ack = old_edge.apply(roi, round(t * 1e9), seq)
                    assert lost_ack["result"] == "applied"
                    assert old_edge.visual.view_kind == "roi"
                elif stage != "await_roi_ack":
                    await acknowledge(first, old_edge, roi, t, seq)
                if stage in {"roi_evidence", "await_full_frame_ack",
                             "return_applied_ack_lost", "await_full_frame_evidence", "complete"}:
                    count = 1 if stage == "roi_evidence" else 4
                    for _ in range(count):
                        seq, t = seq + 1, t + .2
                        await send_sample(first, old_edge, t, seq)
                if stage in {"await_full_frame_ack", "return_applied_ack_lost",
                             "await_full_frame_evidence", "complete"}:
                    restore = await command(first, "REQUEST_FULL_FRAME")
                    if stage == "return_applied_ack_lost":
                        lost_ack = old_edge.apply(restore, round(t * 1e9), seq)
                        assert lost_ack["result"] == "applied"
                        assert old_edge.visual.view_kind == "full_frame"
                    elif stage != "await_full_frame_ack":
                        await acknowledge(first, old_edge, restore, t, seq)
                if stage == "complete":
                    # ACK and an invalid new frame must not complete the cycle.
                    seq, t = seq + 1, t + .2
                    await send_sample(first, old_edge, t, seq, valid=False)
                    assert old_session.pose_cycle.active
                    seq, t = seq + 1, t + .2
                    await send_sample(first, old_edge, t, seq)
                    assert old_session.pose_cycle.phase == "complete"
                else:
                    expected = {
                        "roi_evidence": "observing_roi",
                        "roi_applied_ack_lost": "await_roi_ack",
                        "return_applied_ack_lost": "await_full_frame_ack",
                    }.get(stage, stage)
                    assert old_session.pose_cycle.phase == expected
                assert old_session.controller.episode["episode_id"] == original["episode_id"]

            await eventually(lambda: cloud.active is None)
            assert old_session.pose_shadow.closed
            if restart == "service_object":
                listener.close()
                await listener.wait_closed()
                cloud = make_server()  # Reload committed bytes; discard old in-memory store.
                listener = await serve(cloud.handler, "127.0.0.1", 0)
                url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"

            new_identity = {**identity, "session_id": "second"}
            if restart == "edge_reboot":
                new_identity["boot_id"] = "new-boot"
            new_edge = edge_for(new_identity)
            t0, seq0 = (.5, 1) if restart == "edge_reboot" else (10, seq + 1)
            # Old commands cannot keep the new session in ROI.
            rejected = new_edge.apply(roi, round(t0 * 1e9), seq0)
            assert rejected["result"] == "rejected" and rejected["reason"] == "wrong_session"
            assert new_edge.visual.view_kind == "full_frame"
            async with connect(url, proxy=None) as second:
                await second.send(dumps(status(new_identity)))
                await send_sample(second, new_edge, t0, seq0)
                fresh = await command(second, "SET_TRANSMIT_FPS")
                assert fresh["transmit_fps"] == 5
                assert fresh["reason"] == "unresolved_episode_requires_more_observation"
                assert fresh["command_id"] not in {roi["command_id"], escalation["command_id"]}
                restored = sessions[-1]
                ep = restored.controller.episode
                for key in ("episode_id", "reference_bbox", "origin_identity", "track_id",
                            "horizontal_observed", "horizontal_evidence_seconds",
                            "test_alert_emitted", "pose_observation_requested"):
                    assert ep[key] == original[key]
                assert ep["identity_link_status"] == "unresolved_after_reconnect"
                assert ep["followup_requests"] == original["followup_requests"] + 1
                assert restored.pose_cycle is None
                assert restored.controller.visual_pending is None
                await acknowledge(second, new_edge, fresh, t0, seq0)
                for i in range(1, 13):
                    await send_sample(second, new_edge, t0 + i * .5, seq0 + i)
                baseline = await command(second, "SET_TRANSMIT_FPS")
                assert baseline["transmit_fps"] == 2
                await acknowledge(second, new_edge, baseline, t0 + 6, seq0 + 12)
                assert restored.controller.state == "UNRESOLVED"
                assert restored.controller.episode is not None
                await second.send(dumps(envelope(
                    "DRAIN_REQUEST", new_identity, drain_id="end", live_frames_sent=13,
                )))
                drained = await command(second, "DRAINED")
                assert drained["live_received"] == drained["live_processed"] == 13
            await eventually(lambda: cloud.active is None)
            final = json.loads(snapshot_path.read_text())["snapshot"]["episode"]
            assert final["episode_id"] == original["episode_id"]
            assert final["pose_observation_requested"] is True
            assert final["incident_status"] == "unresolved"
            assert final["identity_link_status"] == "unresolved_after_reconnect"
            assert final["followup_requests"] == original["followup_requests"] + 1
            actions = records("agent_action")
            assert sum(r["type"] == "REQUEST_ROI" for r in actions) == 1
            cycles = records("pose_observation_cycle")
            assert sum(r["stage"] == "session_ended_cycle_incomplete" for r in cycles) == (
                0 if stage == "complete" else 1
            )
            assert sum(r["stage"] == "cycle_complete" for r in cycles) == (
                1 if stage == "complete" else 0
            )
            assert not records("session_error")
            assert not records("test_alert")
            assert not any(r.get("reason") == "episode_resolved" for r in read_log(log_path))
            assert not any(r["would_close_episode"] for r in records("pose_shadow"))
            (tmp_path / "assessment.json").write_text(json.dumps({
                "stage": stage, "restart": restart, "episode_id": final["episode_id"],
                "roi_requests": 1, "restored_fresh_observations": 13,
                "episode_open": True, "identity_link_status": final["identity_link_status"],
                "pose_budget_spent": True, "followup_requests": final["followup_requests"],
                "scope": "local WebSocket and real CommandHandler; synthetic images/vision/pose",
            }, indent=2) + "\n")
        finally:
            listener.close()
            await listener.wait_closed()
            log.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("run", ["sit-return-live-2", "sit-hold-live-1"])
def test_recorded_observations_cannot_close_restored_unlinked_episode(tmp_path, run):
    """Counterfactual controller-only replay, not a measured camera reconnect.

    Keep saved semantic fields, times and track IDs. Simulate new transport IDs
    and immediate ACKs because those recordings did not enable continuity.
    """
    source = Path(__file__).parents[1] / "artifacts/pose-observation-live-20261014" / run
    rows = read_log(source / "cloud.jsonl")
    observations = [r for r in rows if r["event"] == "observation"]
    opened = next(r for r in rows if r["event"] == "incident_episode"
                  and r["reason"] == "episode_opened")
    anchor = next(r for r in observations if r["observation_id"] == opened["observation_id"])
    roi = next(r for r in rows if r["event"] == "agent_action" and r["type"] == "REQUEST_ROI")
    roi_obs = next(r for r in observations if r["frame_id"] == roi["based_on_frame_id"])
    identity = {key: anchor[key] for key in ("edge_id", "boot_id", "session_id")}
    policy = Policy(experimental_s_only_5fps=True, experimental_episode_continuity=True)
    prior = Controller(identity, policy)
    prior._open_episode(anchor, opened["track_id"], [
        v * anchor["width" if i % 2 == 0 else "height"]
        for i, v in enumerate(opened["reference_bbox"])
    ])
    prior.episode["pose_observation_requested"] = True
    prior.state = "UNRESOLVED"
    store = EpisodeStore(tmp_path / "episode.json")
    store.save(prior)
    restored = Controller({**identity, "session_id": "counterfactual-restored"}, policy)
    assert EpisodeStore(store.path).restore(restored)
    edge = edge_for(restored.identity)
    processed = 0
    commands = []
    # Recorded ROI geometry is not usable as a newly requested full-frame view.
    # Keep only full-frame samples after the hypothetical cut; preserve gaps.
    for original in observations:
        if original["captured_mono_ns"] <= roi_obs["captured_mono_ns"] or (
            original["view_kind"] != "full_frame"
        ):
            continue
        item = copy.deepcopy(original)
        item.update(restored.identity, applied_command_id=edge.command_id,
                    policy_seq=edge.policy_seq, target_transmit_fps=edge.sampler.fps,
                    applied_visual_command_id=None, visual_policy_seq=0)
        request = restored.observe(item)
        if request:
            commands.append(request)
            ack = edge.apply(request, item["captured_mono_ns"], item["capture_seq"])
            assert ack["result"] == "applied"
            restored.acknowledge(ack)
        assert restored.episode is not None
        processed += 1
    final = restored.finish_episode()
    assert final["episode_id"] == opened["episode_id"]
    assert final["pose_observation_requested"] is True
    assert final["identity_link_status"] == "unresolved_after_reconnect"
    assert final["horizontal_evidence_seconds"] == 0
    assert final["upright_since_ns"] is None
    assert processed > 50
    assert commands[0]["transmit_fps"] == 5
    assert all(c["type"] == "SET_TRANSMIT_FPS" for c in commands)
    assert not any(e["incident_status"] in {"posture_change_cleared", "recovery_observed"}
                   for e in restored.pop_episode_events())
    (tmp_path / "assessment.json").write_text(json.dumps({
        "run": run, "cloud_log_sha256": hashlib.sha256((source / "cloud.jsonl").read_bytes())
        .hexdigest(), "observations_replayed": processed,
        "requested_rates": [c["transmit_fps"] for c in commands],
        "episode_id": final["episode_id"], "episode_open": final["episode_open"],
        "identity_link_status": final["identity_link_status"],
        "scope": "counterfactual controller-only replay; original semantic observations, "
                 "new session and immediate synthetic ACKs; no inference or adaptive resampling",
    }, indent=2) + "\n")
