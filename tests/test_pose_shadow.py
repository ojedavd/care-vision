import asyncio
import copy
import threading
from types import SimpleNamespace

import pytest
from conftest import frame, status
from test_seated_pose_candidate import observation, pose
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from care_vision.cloud import server as server_module
from care_vision.cloud.pose_shadow import PoseShadowObserver, PoseShadowProcessor
from care_vision.cloud.seated_candidate import SeatContext
from care_vision.cloud.seated_pose_candidate import SeatPoseContext
from care_vision.cloud.server import CloudServer, Session
from care_vision.common.contracts import dumps, pack_frame
from care_vision.common.metrics import EventLog


def episode():
    return {
        "episode_id": "e", "track_id": "a", "started_ns": 1_000_000_000,
        "reference_bbox": [400 / 1280, 40 / 720, 160 / 1280, 500 / 720],
        "horizontal_observed": False, "test_alert_emitted": False,
    }


def processor(infer=None, status="current"):
    return PoseShadowProcessor(
        SeatContext("edge", 1280, 720, (0.30, 0.4, 0.15, 0.15), "synthetic seat"),
        SeatPoseContext((0.34, 0.4, 0.12, 0.12), "synthetic hips"),
        infer or (lambda *_: {"status": "ok", "points": pose().tolist()}),
        context_status=status, context_provenance="synthetic context, not a live calibration",
    )


def test_shadow_pattern_has_no_access_to_controller_and_does_not_mutate_inputs():
    p = processor()
    ep = episode()
    before = copy.deepcopy(ep)
    for t in [1, 1.5, 2, 2.5, 3]:
        obs = observation(t)
        original = copy.deepcopy(obs)
        result = p.process(obs, b"unused", ep)
        assert obs == original and ep == before
    assert result["seated_pose_candidate"]
    assert result["physical_visibility_status"] == "unknown"
    assert not result["seat_support_verified"] and not result["would_close_episode"]
    assert result["suggested_observation"]["dispatched"] is False
    assert p.process(observation(3.5), b"", None)["reason"] == "no_episode"


@pytest.mark.parametrize("failure", ["context", "inference", "weak", "reconnect", "horizontal"])
def test_failure_and_uncertainty_cannot_accumulate_benign_stability(failure):
    p = processor()
    ep = episode()
    for t in [1, 1.5, 2, 2.5]:
        p.process(observation(t), b"", ep)
    obs = observation(3)
    if failure == "context":
        p.context_status = "unknown"
    elif failure == "inference":
        def broken(*_):
            raise RuntimeError("synthetic inference failure")
        p.infer = broken
    elif failure == "weak":
        obs["persons"][0]["association_only"] = True
    elif failure == "reconnect":
        ep["identity_link_status"] = "unresolved_after_reconnect"
    else:
        ep["horizontal_observed"] = True
    result = p.process(obs, b"", ep)
    assert not result["seated_pose_candidate"]
    assert not result["would_close_episode"]
    p.context_status = "current"
    p.infer = lambda *_: {"status": "ok", "points": pose().tolist()}
    assert not p.process(observation(3.5), b"", ep)["seated_pose_candidate"]


def test_bounded_shadow_queue_preserves_snapshots_and_reports_replacements(tmp_path):
    entered, release = threading.Event(), threading.Event()
    seen = []

    def slow(obs, jpeg, ep):
        seen.append((obs, ep))
        entered.set()
        assert release.wait(2)
        return {"observation_id": obs["observation_id"]}

    async def scenario():
        log = EventLog("cloud", tmp_path / "shadow.jsonl")
        observer = PoseShadowObserver(log, SimpleNamespace(process=slow))
        obs, ep = observation(1), episode()
        observer.submit(obs, b"jpeg", ep)
        await asyncio.to_thread(entered.wait, 1)
        assert entered.is_set()
        obs["persons"].clear()
        ep["horizontal_observed"] = True
        observer.submit(observation(2), b"jpeg", episode())
        observer.submit(observation(3), b"jpeg", episode())
        assert observer.queue.qsize() == 1 and observer.dropped == 1
        release.set()
        await observer.close()
        log.close()
        assert observer.completed == 2

    asyncio.run(scenario())
    assert [x[0]["capture_seq"] for x in seen] == [25, 75]
    assert len(seen[0][0]["persons"]) == 1 and not seen[0][1]["horizontal_observed"]


def test_session_dispatch_precedes_shadow_and_shadow_failure_is_nonfatal(tmp_path, identity):
    async def scenario():
        order = []
        log = EventLog("cloud", tmp_path / "session.jsonl")

        class Shadow:
            def submit(self, *_):
                order.append("shadow")
                raise RuntimeError("synthetic failure")

        session = Session(object(), identity, log, pose_shadow_factory=lambda _: Shadow())
        session.vision = SimpleNamespace(observe=lambda *_: {
            **observation(1), "valid": True,
        })
        session.controller = SimpleNamespace(
            observe=lambda _: {"action": "test"}, pop_episode_events=lambda: [],
            pop_loss_outcomes=lambda: [], pop_transitions=lambda: [], episode=None,
        )

        async def dispatch(*_):
            order.append("dispatch")

        session.dispatch = dispatch
        session.frames.put_nowait(({}, b"jpeg", __import__("time").monotonic()))
        task = asyncio.create_task(session.perceive())
        for _ in range(100):
            if order == ["dispatch", "shadow"]:
                break
            await asyncio.sleep(.001)
        assert order == ["dispatch", "shadow"]
        assert not task.done()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        log.close()

    asyncio.run(scenario())


def test_optional_factory_reaches_real_socket_session_and_drains(tmp_path, identity):
    async def scenario():
        created = []
        log = EventLog("cloud", tmp_path / "socket.jsonl")

        def create(stream):
            observer = PoseShadowObserver(stream, processor())
            created.append(observer)
            return observer

        cloud = CloudServer(log, pose_shadow_factory=create)
        try:
            async with serve(cloud.handler, "127.0.0.1", 0) as server:
                url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
                async with connect(url, proxy=None) as socket:
                    await socket.send(dumps(status(identity)))
                    header, jpeg = frame(identity)
                    await socket.send(pack_frame(header, jpeg))
                    for _ in range(100):
                        if created and created[0].completed:
                            break
                        await asyncio.sleep(.01)
                    assert len(created) == 1 and created[0].completed == 1
        finally:
            log.close()
        assert created[0].closed

    asyncio.run(scenario())


def test_episode_checkpoint_precedes_shadow_shutdown(tmp_path, identity, monkeypatch):
    async def scenario():
        order = []

        async def finish_immediately(*tasks):
            for task in tasks:
                task.close()

        monkeypatch.setattr(server_module, "run_until_first", finish_immediately)

        class Shadow:
            async def close(self):
                order.append("shadow_close")

        log = EventLog("cloud", tmp_path / "checkpoint.jsonl")
        session = Session(object(), identity, log, pose_shadow_factory=lambda _: Shadow())
        session.episode_store = SimpleNamespace(save=lambda _: order.append("checkpoint"))
        await session.run()
        log.close()
        assert order == ["checkpoint", "shadow_close"]

    asyncio.run(scenario())
