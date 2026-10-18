"""Optional bounded diagnostic observer. No controller, socket or actuation access."""

import asyncio
import copy
import hashlib
import json
import time
from pathlib import Path

from care_vision.cloud.seated_candidate import SeatContext, SeatedCandidate
from care_vision.cloud.seated_pose_candidate import SeatedPoseCandidate, SeatPoseContext


class RTMShadowInference:
    """Lazy cloud-only adapter; responses never establish physical visibility."""

    def __init__(self, model_path, model_sha256):
        self.path, self.digest = Path(model_path), model_sha256
        self.estimator = None
        self.load_failed = False

    def __call__(self, observation, jpeg):
        import cv2
        import numpy as np

        from care_vision.cloud.pose_evidence import assess_pose_support
        from care_vision.cloud.rtmpose import RTMPoseEstimator
        from care_vision.common.environment import runtime_info

        if self.estimator is None:
            if self.load_failed:
                raise RuntimeError("shadow model unavailable; restart observer to retry")
            try:
                environment = runtime_info(3)
                if not environment["cool"]["verified"] or (
                    environment["ec2"].get("architecture") != "arm64"
                ):
                    raise RuntimeError("shadow inference requires Graviton and verified COOL")
                if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.digest:
                    raise ValueError("shadow model digest mismatch")
                self.estimator = RTMPoseEstimator(self.path)
            except Exception:
                self.load_failed = True
                raise
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (observation["height"], observation["width"]):
            raise ValueError("shadow JPEG/observation dimensions differ")
        box = observation["persons"][0]["tracked_bbox"]
        points = self.estimator.estimate(image, box)
        return {
            "status": "ok",
            "points": points.tolist(),
            "response_support": assess_pose_support(points, box, image.shape),
        }


class PoseShadowProcessor:
    """One independent temporal evaluator per session, using episode snapshots."""

    def __init__(self, geometry, hips, infer, *, context_status, context_provenance):
        if context_status not in {"current", "unknown", "changed"} or not context_provenance:
            raise ValueError("shadow requires explicit manual context status and provenance")
        self.geometry, self.hips, self.infer = geometry, hips, infer
        self.context_status, self.context_provenance = context_status, context_provenance
        self.evaluator = None

    def process(self, observation, jpeg, episode):
        started = time.monotonic()
        result = {
            "mode": "stream_shadow",
            "observation_id": observation["observation_id"],
            "frame_id": observation.get("frame_id"),
            "capture_seq": observation["capture_seq"],
            "captured_mono_ns": observation["captured_mono_ns"],
            "episode_id": episode["episode_id"] if episode else None,
            "seated_pose_candidate": False,
            "seat_support_verified": False,
            "would_close_episode": False,
            "physical_visibility_status": "unknown",
            "context_status": self.context_status,
            "context_provenance": self.context_provenance,
            "context_validation": "manual; identifiers/dimensions do not detect camera movement",
            "model": "RTMPose-t-body7",
            "pose_status": "not_requested",
            "points": None,
            "suggested_observation": None,
            "evidence_status": "INSUFFICIENT_EVIDENCE",
        }
        if episode is None:
            self.evaluator = None
            return {**result, "evidence_status": "NO_ACTIVE_EPISODE", "reason": "no_episode"}
        if self.context_status != "current":
            self.evaluator = None
            return {**result, "reason": "manual_context_" + self.context_status}
        try:
            if (self.evaluator is None
                    or self.evaluator.geometry.episode_id != episode["episode_id"]):
                self.evaluator = SeatedPoseCandidate(
                    SeatedCandidate(episode, observation, self.geometry),
                    self.hips,
                    identity_link_status=episode.get("identity_link_status", "session_track_only"),
                )
            evaluator = self.evaluator
            evaluator.geometry.blocked |= (
                episode["horizontal_observed"] or episode["test_alert_emitted"]
            )
            evaluator.identity_link_status = episode.get(
                "identity_link_status", "session_track_only"
            )
            people = observation.get("persons", [])
            eligible = (
                not evaluator.geometry.blocked
                and evaluator.identity_link_status == "session_track_only"
                and observation.get("valid") is True
                and observation.get("evidence_kind") == "live"
                and observation.get("view_kind") == "full_frame"
                and observation.get("edge_id") == self.geometry.edge_id
                and (observation.get("width"), observation.get("height"))
                == (self.geometry.source_width, self.geometry.source_height)
                and len(people) == 1
                and people[0].get("track_id") == episode["track_id"]
                and people[0].get("track_state") == "confirmed"
                and not people[0].get("association_only", False)
            )
            prediction = {"status": "not_requested", "points": None}
            if eligible:
                try:
                    prediction = self.infer(observation, jpeg)
                except Exception as error:
                    prediction = {
                        "status": "inference_error", "points": None,
                        "error_type": type(error).__name__,
                    }
            points = prediction.get("points") if prediction["status"] == "ok" else None
            diagnostic = evaluator.update(observation, points)
            result.update(diagnostic)
            result.update(
                mode="stream_shadow", pose_status=prediction["status"], points=points,
                response_support=prediction.get("response_support"),
                inference_error_type=prediction.get("error_type"),
                processing_ms=(time.monotonic() - started) * 1000,
            )
            assert not result["would_close_episode"] and not result["seat_support_verified"]
            suggestion = result["suggested_observation"]
            assert suggestion is None or suggestion["dispatched"] is False
            return result
        except Exception as error:
            self.evaluator = None
            return {
                **result, "seated_pose_candidate": False, "seat_support_verified": False,
                "would_close_episode": False, "suggested_observation": None,
                "reason": "shadow_input_or_evaluator_error", "error_type": type(error).__name__,
            }


class PoseShadowObserver:
    """Best-effort latest pending sample; slow/errors never gate controller dispatch."""

    def __init__(self, log, processor):
        self.log, self.processor = log, processor
        self.queue = asyncio.Queue(maxsize=1)
        self.task = None
        self.submitted = self.completed = self.dropped = 0
        self.closed = False
        self.latest_result = None

    def take_result(self):
        result, self.latest_result = self.latest_result, None
        return result

    def submit(self, observation, jpeg, episode):
        if self.closed:
            return
        if self.task is None:
            self.task = asyncio.create_task(self.run())
        if self.queue.full():
            old = self.queue.get_nowait()
            self.queue.task_done()
            self.dropped += 1
            self.log.emit(
                "pose_shadow_dropped", observation_id=old[0]["observation_id"],
                captured_mono_ns=old[0]["captured_mono_ns"], reason="shadow_queue_replaced",
            )
        self.queue.put_nowait((copy.deepcopy(observation), jpeg, copy.deepcopy(episode),
                               time.monotonic()))
        self.submitted += 1

    async def run(self):
        while True:
            observation, jpeg, episode, queued = await self.queue.get()
            try:
                wait_ms = (time.monotonic() - queued) * 1000
                if wait_ms > 1000:
                    observation.update(valid=False, invalid_reason="shadow_queue_stale")
                result = await asyncio.to_thread(self.processor.process, observation, jpeg, episode)
                if time.monotonic() - queued > 1:
                    self.processor.evaluator = None
                    result.update(
                        seated_pose_candidate=False, suggested_observation=None,
                        evidence_status="STALE_SHADOW_RESULT", reason="shadow_processing_stale",
                    )
                self.log.emit("pose_shadow", **result, queue_wait_ms=wait_ms)
                self.latest_result = (observation, result, queued)
                self.completed += 1
            except Exception as error:
                self.log.emit("pose_shadow_error", error_type=type(error).__name__)
            finally:
                self.queue.task_done()

    async def close(self):
        self.closed = True
        drained = True
        if self.task is not None:
            try:
                await asyncio.wait_for(self.queue.join(), timeout=2)
            except TimeoutError:
                drained = False
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self.log.emit(
            "pose_shadow_finished", submitted=self.submitted, completed=self.completed,
            pending_replaced=self.dropped, drained=drained,
            unfinished=self.submitted - self.completed - self.dropped,
            in_progress_inference_may_finish=not drained,
        )


def shadow_factory(config_path):
    """Read configuration explicitly; default service never imports/loads a model."""
    path = Path(config_path)
    config = json.loads(path.read_text())
    geometry = SeatContext(**config["geometry"])
    hips = SeatPoseContext(**config["hips"])
    status, provenance = config["context_status"], config["context_provenance"]
    if status not in {"current", "unknown", "changed"} or not provenance:
        raise ValueError("shadow context must declare status and provenance")
    model = path.parent / config["model_path"]
    digest = config["model_sha256"]
    if hashlib.sha256(model.read_bytes()).hexdigest() != digest:
        raise ValueError("shadow model digest mismatch")

    def create(log):
        return PoseShadowObserver(log, PoseShadowProcessor(
            geometry, hips, RTMShadowInference(model, digest),
            context_status=status, context_provenance=provenance,
        ))

    return create
