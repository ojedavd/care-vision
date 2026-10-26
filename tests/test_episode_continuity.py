import asyncio
import json
import stat
from dataclasses import replace

import cv2
import numpy as np
import pytest
from conftest import frame, status
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

import care_vision.cloud.episode_store as store_module
import care_vision.cloud.server as server_module
from care_vision.check_loop import read_log
from care_vision.cloud.controller import Controller, Policy
from care_vision.cloud.episode_store import EpisodeStore
from care_vision.cloud.server import CloudServer
from care_vision.common.contracts import dumps, envelope, loads, pack_frame
from care_vision.common.metrics import EventLog

POLICY = Policy(experimental_s_only_5fps=True, experimental_episode_continuity=True,
                experimental_alert_after_seconds=20)


def observation(t, *, track="original", command=None, fps=2, box=(600, 40, 150, 500)):
    return {
        "observation_id": f"o-{t}", "frame_id": f"f-{t}", "capture_seq": round(t * 25),
        "captured_mono_ns": round(t * 1e9), "valid": True, "motion_score": 0,
        "applied_command_id": command, "width": 1280, "height": 720,
        "target_transmit_fps": fps, "person_detection_status": "ok",
        "person_tracking_status": "ok", "view_kind": "full_frame",
        "persons": [{"track_id": track, "track_state": "confirmed", "tracked_bbox": list(box)}],
    }


def opened(identity, policy=POLICY):
    c = Controller(identity, policy)
    c._open_episode(observation(1), "original", [600, 40, 150, 500])
    c.state = "UNRESOLVED"
    c.observe(observation(2, box=(550, 410, 330, 110)))
    c.observe(observation(2.5, box=(550, 410, 330, 110)))
    return c


def acknowledge(c, command):
    c.acknowledge({"command_id": command["command_id"], "result": "applied",
                   "active_policy_seq": command["command_seq"],
                   "active_transmit_fps": command["transmit_fps"]})


def wire_frame(identity, seq, t, *, command=None):
    header, jpeg = frame(identity, seq, at_ns=round(t * 1e9),
                         policy_seq=command["command_seq"] if command else 0,
                         command_id=command["command_id"] if command else None)
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    ok, encoded = cv2.imencode(".jpg", cv2.resize(image, (1280, 720)))
    assert ok
    jpeg = encoded.tobytes()
    header.update(width=1280, height=720, payload_bytes=len(jpeg),
                  target_transmit_fps=command["transmit_fps"] if command else 2)
    return pack_frame(header, jpeg)


def wire_ack(identity, command, seq, t):
    return dumps(envelope(
        "COMMAND_ACK", identity, command_id=command["command_id"],
        command_seq=command["command_seq"], command_type="SET_TRANSMIT_FPS", result="applied",
        active_policy_seq=command["command_seq"], active_transmit_fps=command["transmit_fps"],
        applied_at_edge_mono_ns=round(t * 1e9), effective_from_capture_seq=seq,
    ))


def test_private_snapshot_survives_restart_without_commands_or_pixels(tmp_path, identity):
    path = tmp_path / "private/episode.json"
    old = opened(identity)
    old.finish_episode()
    store = EpisodeStore(path)
    store.save(old)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    new = Controller({**identity, "session_id": "new-session"}, POLICY)
    assert EpisodeStore(path).restore(new)
    assert new.episode["episode_id"] == old.episode["episode_id"]
    assert new.episode["horizontal_evidence_seconds"] == 0.5
    assert new.episode["reference_bbox"] == old.episode["reference_bbox"]
    assert new.episode["identity_link_status"] == "unresolved_after_reconnect"
    assert new.episode["review_required"] is True
    assert new.episode["review_reason"] == "identity_unresolved_after_reconnect"
    assert new.pending is None and new.command_seq == 0
    assert not any(k in path.read_text() for k in ["jpeg", "keypoints", "password"])
    # New-session standing person cannot close the old logical episode.
    request = new.observe(observation(3, track="new-track"))
    assert request["type"] == "SET_TRANSMIT_FPS" and request["transmit_fps"] == 5
    acknowledge(new, request)
    for t in (3.5, 4, 4.5, 5, 5.5, 6):
        new.observe(observation(t, track="new-track", command=request["command_id"], fps=5))
    assert new.episode is not None
    assert new.episode["horizontal_evidence_seconds"] == 0.5
    assert new.episode["review_required"] is True
    store.save(new)
    again = Controller({**identity, "session_id": "third-session"}, POLICY)
    assert EpisodeStore(path).restore(again)
    assert again.episode["review_required"] is True


def test_alert_and_evidence_survive_reconnect_without_duplicate_alert(identity):
    old = opened(identity)
    old.episode.update(test_alert_emitted=True, horizontal_evidence_seconds=21,
                       incident_status="assistance_required_test")
    store = EpisodeStore()
    store.save(old)
    new = Controller({**identity, "boot_id": "reboot", "session_id": "new"}, POLICY)
    store.restore(new)
    request = new.observe(observation(0.5, track="other"))
    acknowledge(new, request)
    baseline = new.observe(observation(6.5, track="other", fps=5, command=request["command_id"]))
    assert baseline["transmit_fps"] == 2
    assert new.state == "ASSISTANCE_REQUIRED"
    assert new.episode["horizontal_evidence_seconds"] == 21
    assert not any(e["reason"] == "TRIGGER_ALERT_TEST" for e in new.pop_episode_events())


def test_restored_review_is_durable_without_any_new_frame(tmp_path, identity):
    store = EpisodeStore(tmp_path / "episode.json")
    store.save(opened(identity))
    log = EventLog("cloud", tmp_path / "cloud.jsonl")
    try:
        server_module.Session(
            None, {**identity, "session_id": "new"}, log,
            experimental_s_only_5fps=True, experimental_episode_continuity=True,
            experimental_alert_after_seconds=20, episode_store=store,
        )
    finally:
        log.close()
    assert EpisodeStore(store.path).snapshot["episode"]["review_required"] is True
    records = read_log(tmp_path / "cloud.jsonl")
    assert any(r["event"] == "episode_review_required" for r in records)
    assert not any(r["event"] in {"observation", "test_alert"} for r in records)


def test_followup_budget_and_cooldown_leave_episode_open(identity):
    old = opened(identity)
    new = Controller({**identity, "session_id": "new"}, POLICY)
    new.restore_episode(old.episode)
    applied = None
    for t in (3, 33, 63):
        request = new.observe(observation(t, track="new", command=applied))
        assert request["transmit_fps"] == 5
        acknowledge(new, request)
        baseline = new.observe(observation(t + 6, track="new", fps=5,
                                           command=request["command_id"]))
        acknowledge(new, baseline)
        applied = baseline["command_id"]
        assert new.observe(observation(t + 7, track="new", command=applied)) is None
    assert new.observe(observation(93, track="new", command=applied)) is None
    assert new.episode is not None and new.state == "UNRESOLVED"
    assert any(e["reason"] == "episode_observation_budget_exhausted"
               for e in new.pop_episode_events())


def test_closure_removes_only_pending_metadata(tmp_path, identity):
    store = EpisodeStore(tmp_path / "episode.json")
    c = opened(identity)
    store.save(c)
    for t in (3, 3.5, 4, 4.5, 5):
        c.observe(observation(t))
    assert c.episode is None
    store.save(c)
    assert store.snapshot is None and not store.path.exists()


def test_other_edge_policy_change_corruption_and_public_file_do_not_reset_episode(
    tmp_path, identity
):
    path = tmp_path / "episode.json"
    store = EpisodeStore(path)
    store.save(opened(identity))
    for new in [Controller({**identity, "edge_id": "other"}, POLICY),
                Controller(identity, replace(POLICY, episode_followup_limit=4))]:
        with pytest.raises(ValueError):
            store.restore(new)
        with pytest.raises(ValueError):
            store.save(new)
    data = json.loads(path.read_text())
    data["snapshot"]["episode"]["horizontal_evidence_seconds"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest"):
        EpisodeStore(path)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        EpisodeStore(path)


def test_failed_atomic_write_preserves_last_committed_evidence(tmp_path, identity, monkeypatch):
    store = EpisodeStore(tmp_path / "episode.json")
    c = opened(identity)
    store.save(c)
    before = store.path.read_bytes()
    c.episode["horizontal_evidence_seconds"] = 3

    def failed_replace(*args):
        raise OSError("simulated write failure")

    monkeypatch.setattr(store_module.os, "replace", failed_replace)
    with pytest.raises(OSError):
        store.save(c)
    assert store.path.read_bytes() == before
    assert store.snapshot["episode"]["horizontal_evidence_seconds"] == .5
    assert not list(tmp_path.glob(".episode-*"))


def test_superseding_socket_waits_for_old_episode_checkpoint(tmp_path, identity, monkeypatch):
    async def scenario():
        old_ready = asyncio.Event()

        class DelayedSession:
            def __init__(self, websocket, ident, log, *args):
                self.websocket, self.ident = websocket, ident
                self.store = args[-1]
                self.controller = Controller(ident, POLICY)
                if ident["session_id"] != identity["session_id"]:
                    assert self.store.restore(self.controller)
                    assert self.controller.episode["horizontal_evidence_seconds"] == .5

            async def run(self):
                if self.ident["session_id"] == identity["session_id"]:
                    self.controller = opened(self.ident)
                    old_ready.set()
                    await self.websocket.wait_closed()
                    await asyncio.sleep(.05)  # Old finalizer has not committed yet.
                    self.store.save(self.controller)
                else:
                    await self.websocket.send("restored after checkpoint")
                    await self.websocket.wait_closed()

        monkeypatch.setattr(server_module, "Session", DelayedSession)
        log = EventLog("cloud", tmp_path / "handover.jsonl")
        cloud = CloudServer(log, experimental_s_only_5fps=True,
                            experimental_episode_continuity=True)
        try:
            async with serve(cloud.handler, "127.0.0.1", 0) as server:
                url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
                async with connect(url) as old:
                    await old.send(dumps(status(identity)))
                    await asyncio.wait_for(old_ready.wait(), 2)
                    async with connect(url) as new:
                        await new.send(dumps(status({**identity, "session_id": "new"})))
                        assert await asyncio.wait_for(new.recv(), 2) == "restored after checkpoint"
        finally:
            log.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("reboot", [False, True])
def test_websocket_disconnect_restores_episode_and_requests_fresh_evidence(
    tmp_path, identity, monkeypatch, reboot
):
    class ScriptedVision:
        """Controlled semantic observations; this test does not validate inference."""
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, header, jpeg):
            box = (600, 40, 150, 500)
            if header["session_id"] == identity["session_id"] and header["capture_seq"] >= 4:
                box = (600, 200, 160, 320)
            o = observation(header["captured_mono_ns"] / 1e9,
                            track=f"live:{header['session_id']}:track:1", box=box)
            o.update(header, observation_id=f"{header['session_id']}:{header['capture_seq']}")
            return o

    monkeypatch.setattr(server_module, "MotionVision", ScriptedVision)

    async def scenario():
        log = EventLog("cloud", tmp_path / "cloud.jsonl")
        cloud = CloudServer(log, experimental_s_only_5fps=True,
                            experimental_episode_continuity=True,
                            episode_state_file=tmp_path / "episode.json")
        try:
            async with serve(cloud.handler, "127.0.0.1", 0) as server:
                url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
                async with connect(url) as first:
                    await first.send(dumps(status(identity)))
                    for seq in range(1, 5):
                        await first.send(wire_frame(identity, seq, seq * .5))
                        async with asyncio.timeout(2):
                            while len([r for r in read_log(tmp_path / "cloud.jsonl")
                                       if r["event"] == "observation"]) < seq:
                                await asyncio.sleep(.005)
                    escalation = loads(await asyncio.wait_for(first.recv(), 2))
                    assert escalation["transmit_fps"] == 5
                    episode_id = cloud.episode_store.snapshot["episode"]["episode_id"]
                    old_reference = cloud.episode_store.snapshot["episode"]["reference_bbox"]
                    # Disconnect with an unacknowledged command; it must not replay.
                new_identity = {**identity, "session_id": "new-session"}
                if reboot:
                    new_identity["boot_id"] = "reboot"
                async with connect(url) as second:
                    await second.send(dumps(status(new_identity)))
                    t = .5 if reboot else 10
                    initial_seq = 1 if reboot else 5
                    await second.send(wire_frame(new_identity, initial_seq, t))
                    fresh = loads(await asyncio.wait_for(second.recv(), 2))
                    assert fresh["command_id"] != escalation["command_id"]
                    assert fresh["session_id"] == "new-session" and fresh["transmit_fps"] == 5
                    assert fresh["reason"] == "unresolved_episode_requires_more_observation"
                    snapshot = cloud.episode_store.snapshot["episode"]
                    assert snapshot["episode_id"] == episode_id
                    assert snapshot["reference_bbox"] == old_reference
                    assert snapshot["identity_link_status"] == "unresolved_after_reconnect"
                    assert snapshot["upright_since_ns"] is None
                    await second.send(wire_ack(new_identity, fresh, initial_seq + 1, t))
                    for index in range(2, 14):
                        await second.send(wire_frame(
                            new_identity, initial_seq + index - 1, t + (index - 1) * .5,
                            command=fresh,
                        ))
                        async with asyncio.timeout(2):
                            while len([r for r in read_log(tmp_path / "cloud.jsonl")
                                       if r["event"] == "observation"
                                       and r["session_id"] == "new-session"]) < index:
                                await asyncio.sleep(.005)
                    baseline = loads(await asyncio.wait_for(second.recv(), 2))
                    assert baseline["transmit_fps"] == 2
                    await second.send(wire_ack(new_identity, baseline, initial_seq + 13, t + 6))
                    await second.send(dumps(envelope(
                        "DRAIN_REQUEST", new_identity, drain_id="end", live_frames_sent=13
                    )))
                    drained = loads(await asyncio.wait_for(second.recv(), 2))
                    assert drained["type"] == "DRAINED" and drained["live_processed"] == 13
                    snapshot = cloud.episode_store.snapshot["episode"]
                    assert snapshot["episode_id"] == episode_id
                    assert snapshot["identity_link_status"] == "unresolved_after_reconnect"
        finally:
            log.close()
        records = read_log(tmp_path / "cloud.jsonl")
        assert any(r["event"] == "episode_restored" for r in records)
        review = [r for r in records if r["event"] == "episode_review_required"]
        assert len(review) == 1
        assert review[0]["review_required"] is True
        assert review[0]["incident_confirmed"] is False
        assert review[0]["notification_sent"] is False
        assert cloud.episode_store.snapshot["episode"]["review_required"] is True
        assert not any(r.get("reason") == "episode_resolved" for r in records)
        assert len([r for r in records if r["event"] == "ack_received"]) == 2

    asyncio.run(scenario())
