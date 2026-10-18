"""One cloud-side session, with perception separate from socket receive and commands."""

import asyncio
import math
import time
from collections import OrderedDict

import cv2
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from care_vision.cloud.controller import Controller, Policy
from care_vision.cloud.episode_store import EpisodeStore
from care_vision.cloud.history import HistoryTransfer
from care_vision.cloud.pose_actuation import PoseObservationCycle
from care_vision.cloud.pose_observation_plan import PoseObservationPlanner
from care_vision.cloud.vision import MotionVision
from care_vision.common.contracts import (
    MAX_MESSAGE,
    ProtocolError,
    dumps,
    envelope,
    loads,
    unpack_frame,
)
from care_vision.common.environment import runtime_info
from care_vision.common.metrics import EventLog
from care_vision.common.transport import run_until_first

LIVE_QUEUE_CAPACITY = 4
RECOVERY_WINDOW_NS = 10_000_000_000
MAX_ARCHIVE_WINDOWS = 6


def recovery_windows(status):
    """Plan bounded, chronological archive windows followed by the live ring tail."""
    if not status:
        return []
    windows = []
    ring_start = status.get("recovery_from_edge_mono_ns")
    spool_start = status.get("spool_oldest_mono_ns")
    spool_end = status.get("spool_newest_mono_ns")
    if spool_start is not None and spool_end is not None and status.get("spool_frames", 0):
        end = min(spool_end + 1, ring_start) if ring_start is not None else spool_end + 1
        start = max(spool_start, end - MAX_ARCHIVE_WINDOWS * RECOVERY_WINDOW_NS)
        while start < end:
            boundary = min(start + RECOVERY_WINDOW_NS, end)
            windows.append((start, boundary, "archive"))
            start = boundary
    if ring_start is not None:
        windows.append((ring_start, status["recovery_to_edge_mono_ns"], "ring"))
    return windows


class Session:
    def __init__(
        self,
        websocket,
        identity,
        log,
        backend=None,
        runtime_id=None,
        person_detector_factory=None,
        initial_status=None,
        verification_detector_factory=None,
        experimental_loss_5fps=False,
        experimental_s_only_5fps=False,
        experimental_alert_after_seconds=None,
        experimental_weak_association=False,
        experimental_episode_continuity=False,
        diagnostic_disconnect_check=None,
        episode_store=None,
        pose_shadow_factory=None,
        experimental_pose_observation=False,
    ):
        self.websocket, self.identity, self.log = websocket, identity, log
        self.pose_shadow = pose_shadow_factory(log) if pose_shadow_factory else None
        if experimental_pose_observation and (
            self.pose_shadow is None or not experimental_s_only_5fps
        ):
            raise ValueError("pose observation requires shadow config and S-only mode")
        self.pose_planner = (
            PoseObservationPlanner(
                self.pose_shadow.processor.geometry, self.pose_shadow.processor.hips
            )
            if experimental_pose_observation else None
        )
        self.pose_cycle = None
        self.pose_latest_observation = None
        self.experimental_loss_5fps = experimental_loss_5fps
        self.experimental_s_only_5fps = experimental_s_only_5fps
        self.experimental_alert_after_seconds = experimental_alert_after_seconds
        self.experimental_weak_association = experimental_weak_association
        self.controller = Controller(
            identity,
            Policy(
                experimental_loss_5fps=experimental_loss_5fps,
                experimental_s_only_5fps=experimental_s_only_5fps,
                experimental_alert_after_seconds=experimental_alert_after_seconds,
                experimental_episode_continuity=experimental_episode_continuity,
            ),
        )
        self.episode_store = episode_store
        self.diagnostic_disconnect_check = diagnostic_disconnect_check
        if episode_store and episode_store.restore(self.controller):
            self.log.emit(
                "episode_restored", **identity,
                episode_id=self.controller.episode["episode_id"],
                origin_identity=self.controller.episode["origin_identity"],
                identity_link_status="unresolved_after_reconnect",
                incident_status=self.controller.episode["incident_status"],
            )
        person_detector = person_detector_factory() if person_detector_factory else None
        self.vision = MotionVision(
            backend,
            runtime_id,
            person_detector,
            experimental_weak_association=experimental_weak_association,
        )
        self.verification_vision = (
            MotionVision(
                backend,
                runtime_id,
                verification_detector_factory(),
            )
            if verification_detector_factory
            else None
        )
        if experimental_loss_5fps and self.verification_vision is None:
            raise ValueError("experimental loss verification requires YOLOX-S detector")
        # Absorb the bounded transport bunching measured in Phase 4 while keeping
        # latest-frame backpressure when vision falls materially behind.
        self.frames = asyncio.Queue(maxsize=LIVE_QUEUE_CAPACITY)
        self.history = HistoryTransfer(
            log,
            backend,
            runtime_id,
            person_detector_factory,
            defer_processing=lambda: not self.frames.empty(),
        )
        self.seen = OrderedDict()
        self.last_seq = self.last_capture_ns = 0
        self.pending_since = 0
        self.control_last_sent = {}
        self.drain_request = None
        self.drain_ack_sent = False
        self.actions_inflight = False
        self.received = self.processed = self.dropped = 0
        self.processing_frame = None
        self.shutdown_dropped = 0
        self.stale = self.status_seq = 0
        self.cloud_status_enabled = (initial_status or {}).get("cloud_status_version") == 1
        self.status_lock = asyncio.Lock()
        self.recovery_status = initial_status or {}
        self.recovery = recovery_windows(initial_status)
        self.recovery_anchor = None
        if self.recovery:
            self.log.emit(
                "offline_recovery_planned",
                windows=len(self.recovery),
                archive_windows=sum(kind == "archive" for _, _, kind in self.recovery),
                archive_frames=self.recovery_status.get("spool_frames", 0),
                archive_prefix_truncated=(
                    self.recovery[0][2] == "archive"
                    and self.recovery_status.get("spool_oldest_mono_ns") < self.recovery[0][0]
                ),
            )

    def check_identity(self, message):
        if any(message.get(k) != v for k, v in self.identity.items()):
            raise ProtocolError("message belongs to another session")

    async def receive(self):
        async for raw in self.websocket:
            if isinstance(raw, bytes):
                header, jpeg = unpack_frame(raw)
                self.check_identity(header)
                historical = header.get("evidence_kind", "live") == "historical"
                if header["frame_id"] in self.seen or (
                    not historical and header["capture_seq"] <= self.last_seq
                ):
                    self.log.emit("duplicate_frame_rejected", frame_id=header["frame_id"])
                    continue
                self.seen[header["frame_id"]] = None
                if len(self.seen) > 4096:
                    self.seen.popitem(last=False)
                if historical:
                    self.history.receive(header, jpeg)
                    self.log.emit("frame_received", **header)
                    continue
                if header["captured_mono_ns"] <= self.last_capture_ns:
                    raise ProtocolError("capture time is not increasing")
                self.last_seq, self.last_capture_ns = (
                    header["capture_seq"],
                    header["captured_mono_ns"],
                )
                self.received += 1
                arrival = time.monotonic()
                self.log.emit(
                    "frame_received",
                    **header,
                    live_queue_depth_before=self.frames.qsize(),
                    live_queue_capacity=self.frames.maxsize,
                )
                if self.frames.full():
                    old, _, old_arrival = self.frames.get_nowait()
                    self.dropped += 1
                    self.log.emit(
                        "frame_dropped",
                        frame_id=old["frame_id"],
                        capture_seq=old["capture_seq"],
                        replaced_by_frame_id=header["frame_id"],
                        replaced_by_capture_seq=header["capture_seq"],
                        reason="vision_busy",
                        drop_policy="latest_live_frame_wins",
                        intentional_backpressure=True,
                        queue_wait_ms=(arrival - old_arrival) * 1000,
                        live_queue_capacity=self.frames.maxsize,
                    )
                self.frames.put_nowait((header, jpeg, arrival))
                # recv() may return an entire buffered burst without suspending.
                # Let a ready perceiver take work before filling the bounded queue.
                await asyncio.sleep(0)
            else:
                message = loads(raw)
                self.check_identity(message)
                if message["type"] == "COMMAND_ACK":
                    self.log.emit("ack_received", **message)
                    if message.get("command_type") == "REQUEST_RECENT_BUFFER":
                        self.history.acknowledge(message)
                    elif message.get("command_type") in {"REQUEST_ROI", "REQUEST_FULL_FRAME"}:
                        accepted = self.controller.acknowledge_visual(message)
                        if accepted and self.pose_cycle and self.pose_cycle.acknowledge(
                            message, time.monotonic()
                        ):
                            self.log_pose_cycle("visual_ack", command_id=message["command_id"])
                    else:
                        self.controller.acknowledge(message)
                    self.control_last_sent.pop(message["command_id"], None)
                elif message["type"] == "BUFFER_COMPLETE":
                    self.history.finish_receiving(message)
                    self.log.emit("buffer_received", **message)
                elif message["type"] == "EDGE_STATUS":
                    self.log.emit(
                        "status_received",
                        **message,
                        frames_received=self.received,
                        frames_processed=self.processed,
                        frames_dropped=self.dropped,
                    )
                elif message["type"] == "DRAIN_REQUEST":
                    if self.drain_request is not None and message != self.drain_request:
                        raise ProtocolError("conflicting drain request")
                    self.drain_request = message
                    self.log.emit("drain_requested", **message)
                else:
                    raise ProtocolError("unexpected edge message")

    async def perceive(self):
        while True:
            header, jpeg, arrival = await self.frames.get()
            # A cancelled to_thread await does not stop native inference. Keep
            # its ID until an observation is logged, so teardown can account for
            # a result that the session will never consume.
            self.processing_frame = header
            started = time.monotonic()
            vision = (
                self.verification_vision
                if self.verification_vision is not None
                and (header["width"], header["height"]) == (1280, 720)
                else self.vision
            )
            observation = await asyncio.to_thread(vision.observe, header, jpeg)
            # Reject delayed processing as evidence for either escalation or recovery.
            if time.monotonic() - arrival > 1:
                observation.update(valid=False, invalid_reason="processing_stale")
                self.stale += 1
            observation.update(
                cloud_queue_wait_ms=(started - arrival) * 1000,
                live_queue_depth_after_dequeue=self.frames.qsize(),
                live_queue_capacity=self.frames.maxsize,
            )
            self.processed += 1
            self.log.emit("observation", **observation)
            self.processing_frame = None
            if self.pose_planner is not None:
                self.pose_latest_observation = observation
                if self.pose_cycle:
                    result = self.pose_cycle.observe(observation, time.monotonic())
                    if result:
                        self.log_pose_cycle(result, observation_id=observation["observation_id"],
                                            capture_seq=observation["capture_seq"])
            if observation["valid"]:
                self.recovery_anchor = observation
            command = self.controller.observe(observation) if self.drain_request is None else None
            self.checkpoint_episode()
            for event in self.controller.pop_episode_events():
                self.log.emit("incident_episode", **event)
                if event["reason"] == "TRIGGER_ALERT_TEST":
                    self.log.emit(
                        "test_alert",
                        **event,
                        test_only=True,
                        action="TRIGGER_ALERT",
                        notification_sent=False,
                    )
            for outcome in self.controller.pop_loss_outcomes():
                self.log.emit("loss_verification_outcome", **outcome)
            for transition in self.controller.pop_transitions():
                self.log.emit("agent_state_transition", **transition)
            if command:
                self.actions_inflight = True
                try:
                    await self.dispatch(command, observation)
                finally:
                    self.actions_inflight = False
            if self.diagnostic_disconnect_check and self.diagnostic_disconnect_check(
                self.controller, observation
            ):
                self.log.emit(
                    "diagnostic_disconnect", **self.identity,
                    episode_id=self.controller.episode["episode_id"],
                    captured_mono_ns=observation["captured_mono_ns"],
                    test_only=True, scope="server-induced WebSocket close; not WAN packet loss",
                )
                await self.websocket.close(code=1012, reason="controlled continuity test")
                return
            if self.pose_shadow is not None:
                # Submit after controller dispatch. The observer gets snapshots only;
                # it cannot send commands, change the episode or delay this decision.
                try:
                    self.pose_shadow.submit(observation, jpeg, self.controller.episode)
                except Exception as error:
                    self.log.emit("pose_shadow_error", error_type=type(error).__name__)

    def log_pose_cycle(self, stage, **fields):
        self.log.emit(
            "pose_observation_cycle", stage=stage, phase=self.pose_cycle.phase,
            episode_id=self.pose_cycle.episode_id, roi_frames=self.pose_cycle.roi_frames,
            roi_command_id=self.pose_cycle.roi_command["command_id"],
            return_command_id=(self.pose_cycle.return_command or {}).get("command_id"),
            seat_support_verified=False, episode_closed_by_pose=False, **fields,
        )

    async def pose_observation_tick(self):
        now = time.monotonic()
        cycle = self.pose_cycle
        if cycle and cycle.active:
            cycle.check_timeout(now)
            reason = cycle.return_due(now, draining=self.drain_request is not None)
            if reason and not self.controller.visual_pending:
                obs = self.pose_latest_observation
                command = self.controller.request_full_frame(obs, reason=reason)
                cycle.expect_return(command, now, reason)
                self.log_pose_cycle("return_requested", reason=reason)
                await self.dispatch(command, obs)
            return
        sample = self.pose_shadow.take_result()
        if sample is None or self.drain_request is not None or self.actions_inflight:
            return
        obs, diagnostic, queued = sample
        latest = self.pose_latest_observation
        if latest is None or latest["capture_seq"] != obs["capture_seq"]:
            return
        proposal = self.pose_planner.consider(
            obs, diagnostic, self.controller.episode, result_age_seconds=now - queued,
            commands_pending=bool(self.controller.pending or self.controller.visual_pending),
        )
        self.log.emit("pose_observation_plan", **proposal)
        if proposal["plan"] is None:
            return
        request = proposal["plan"]["request"]
        command = self.controller.request_roi(
            obs, request["roi"], request["output_width"], request["output_height"],
            padding_fraction=0, reason="pose_uncertainty_requires_body_chair_detail",
        )
        self.controller.episode["pose_observation_requested"] = True
        self.checkpoint_episode()
        self.pose_cycle = PoseObservationCycle(proposal, obs, command, now)
        self.log_pose_cycle("roi_requested", proposal=proposal)
        await self.dispatch(command, obs)

    async def control_pose_observations(self):
        if self.pose_planner is None:
            await asyncio.Future()
        while True:
            await asyncio.sleep(.02)
            if self.pose_planner is not None:
                await self.pose_observation_tick()

    async def recover_offline(self):
        while True:
            await asyncio.sleep(0.01)
            if (
                not self.recovery
                or self.recovery_anchor is None
                or self.actions_inflight
                or not self.history.finished
            ):
                continue
            start, end, kind = self.recovery.pop(0)
            request = self.controller.request_offline_recovery(
                self.recovery_anchor, start, end, archived=kind == "archive"
            )
            self.history.start(request)
            self.log.emit(
                "offline_recovery_requested",
                buffer_request_id=request["command_id"],
                window_kind=kind,
                remaining_windows=len(self.recovery),
                available_frames=self.recovery_status.get("recovery_available_frames", 0)
                if kind == "ring"
                else self.recovery_status.get("spool_frames", 0),
                truncated=self.recovery_status.get("recovery_truncated", False),
                includes_unconfirmed_context=self.recovery_status.get(
                    "recovery_includes_unconfirmed_context", False
                ),
                from_edge_mono_ns=start,
                to_edge_mono_ns=end,
            )
            self.log.emit("agent_action", **request, agent_state=self.controller.state)
            await asyncio.wait_for(self.websocket.send(dumps(request)), timeout=2)

    async def dispatch(self, command, observation):
        if (
            command["type"] == "REQUEST_FULL_FRAME" and self.pose_cycle
            and self.pose_cycle.active and self.pose_cycle.return_command is None
        ):
            # An existing controller followup can also restore the scene.
            self.pose_cycle.expect_return(command, time.monotonic(), command["reason"])
            self.log_pose_cycle("return_requested_by_controller", reason=command["reason"])
        self.pending_since = time.monotonic()
        self.log.emit("agent_action", **command, agent_state=self.controller.state)
        self.control_last_sent[command["command_id"]] = time.monotonic()
        await asyncio.wait_for(self.websocket.send(dumps(command)), timeout=2)
        if command["type"] == "REQUEST_FULL_FRAME":
            # Standalone context recovery keeps the acknowledged FPS unchanged.
            return
        if self.experimental_s_only_5fps:
            return
        if self.experimental_loss_5fps:
            width, height = (
                (1280, 720) if command["transmit_fps"] == 5 else self.controller.baseline_resolution
            )
            visual = self.controller.request_full_frame(
                observation,
                width,
                height,
                reason=(
                    "loss_verify_full_scene"
                    if command["transmit_fps"] == 5
                    else "loss_verify_return_to_baseline_view"
                ),
            )
            self.log.emit("agent_action", **visual, agent_state=self.controller.state)
            self.control_last_sent[visual["command_id"]] = time.monotonic()
            await asyncio.wait_for(self.websocket.send(dumps(visual)), timeout=2)
            return
        if command["transmit_fps"] == 20:
            if self.history.finished:
                request = self.controller.request_recent_buffer(observation)
                self.history.start(request)
                self.log.emit("agent_action", **request, agent_state=self.controller.state)
                await asyncio.wait_for(self.websocket.send(dumps(request)), timeout=2)
            else:
                self.log.emit(
                    "buffer_request_skipped",
                    reason="previous_transfer_still_active",
                    buffer_request_id=self.history.request["command_id"],
                )
            if (
                observation.get("confirmed_person_detected")
                and observation.get("primary_tracked_bbox")
                and observation.get("view_kind", "full_frame") == "full_frame"
            ):
                visual = self.controller.request_roi(
                    observation, observation["primary_tracked_bbox"]
                )
                self.log.emit("agent_action", **visual, agent_state=self.controller.state)
                self.control_last_sent[visual["command_id"]] = time.monotonic()
                await asyncio.wait_for(self.websocket.send(dumps(visual)), timeout=2)
        elif self.controller.active_view_kind == "roi":
            visual = self.controller.request_full_frame(observation)
            self.log.emit("agent_action", **visual, agent_state=self.controller.state)
            self.control_last_sent[visual["command_id"]] = time.monotonic()
            await asyncio.wait_for(self.websocket.send(dumps(visual)), timeout=2)

    async def send_cloud_status(self):
        if not self.cloud_status_enabled:
            return
        async with self.status_lock:
            self.status_seq += 1
            message = envelope(
                "CLOUD_STATUS",
                self.identity,
                status_seq=self.status_seq,
                live_received=self.received,
                live_processed=self.processed,
                live_dropped=self.dropped,
                live_stale=self.stale,
                queue_depth=self.frames.qsize(),
                queue_capacity=self.frames.maxsize,
            )
            await asyncio.wait_for(self.websocket.send(dumps(message)), timeout=2)
            self.log.emit("cloud_status_sent", **message)

    async def report_health(self):
        while True:
            await asyncio.sleep(1)
            if not self.drain_ack_sent:
                await self.send_cloud_status()

    async def confirm_drain(self):
        while True:
            await asyncio.sleep(0.01)
            if self.drain_request is None or self.drain_ack_sent:
                continue
            if self.received > self.drain_request["live_frames_sent"]:
                raise ProtocolError("drain live count is smaller than received evidence")
            if (
                self.recovery
                and self.recovery_anchor is None
                and self.received == self.drain_request["live_frames_sent"]
                and self.processed + self.dropped == self.received
                and self.frames.empty()
            ):
                self.log.emit(
                    "offline_recovery_skipped",
                    reason="no_valid_live_observation_before_drain",
                    remaining_windows=len(self.recovery),
                )
                self.recovery.clear()
            if (
                self.actions_inflight
                or self.controller.pending
                or self.controller.visual_pending
                or (self.pose_cycle is not None and self.pose_cycle.active)
                or self.recovery
                or not self.history.finished
                or not self.frames.empty()
                or self.received != self.drain_request["live_frames_sent"]
                or self.processed + self.dropped != self.received
            ):
                continue
            ack = envelope(
                "DRAINED",
                self.identity,
                drain_id=self.drain_request["drain_id"],
                live_received=self.received,
                live_processed=self.processed,
                live_dropped=self.dropped,
                historical_received=self.history.total_received,
                historical_processed=self.history.total_processed,
            )
            await self.send_cloud_status()
            await asyncio.wait_for(self.websocket.send(dumps(ack)), timeout=2)
            self.drain_ack_sent = True
            self.log.emit("drain_confirmed", **ack)

    async def retry_control(self, command, kind):
        now = time.monotonic()
        if now - self.pending_since >= 5:
            raise ProtocolError(f"{kind} unconfirmed; new session required")
        last_sent = self.control_last_sent.get(command["command_id"], self.pending_since)
        if now - last_sent < 1:
            return
        self.log.emit("command_retry", command_id=command["command_id"])
        self.control_last_sent[command["command_id"]] = now
        await asyncio.wait_for(self.websocket.send(dumps(command)), timeout=2)

    async def retry_commands(self):
        while True:
            await asyncio.sleep(1)
            history = self.history
            history.check_timeout(time.monotonic())
            if not history.finished and not history.complete:
                if not history.ack:
                    self.log.emit("command_retry", command_id=history.request["command_id"])
                    await asyncio.wait_for(self.websocket.send(dumps(history.request)), timeout=2)
            command = self.controller.pending
            if command:
                await self.retry_control(command, "command")
            visual = self.controller.visual_pending
            if visual:
                await self.retry_control(visual, "visual command")

    def discard_live_on_shutdown(self):
        """Account for accepted evidence after all session workers have stopped."""
        pending = []
        if self.processing_frame is not None:
            pending.append((self.processing_frame, "inference"))
            self.processing_frame = None
        while not self.frames.empty():
            header, _, _ = self.frames.get_nowait()
            pending.append((header, "queued"))
        for header, stage in pending:
            self.dropped += 1
            self.shutdown_dropped += 1
            self.log.emit(
                "frame_dropped", **self.identity,
                frame_id=header["frame_id"], capture_seq=header["capture_seq"],
                evidence_kind="live", reason="session_ended_before_observation",
                processing_stage=stage, intentional_backpressure=False,
                inference_result_discarded=stage == "inference",
            )

    async def run(self):
        try:
            await run_until_first(
                self.receive(),
                self.perceive(),
                self.recover_offline(),
                self.history.perceive(),
                self.retry_commands(),
                self.confirm_drain(),
                self.report_health(),
                self.control_pose_observations(),
            )
        finally:
            # run_until_first has cancelled and joined every session coroutine.
            # Do not run new perception or incident decisions on a closed link.
            self.discard_live_on_shutdown()
            if self.pose_cycle and self.pose_cycle.active:
                self.log_pose_cycle("session_ended_cycle_incomplete")
            episode = self.controller.finish_episode()
            if episode is not None:
                self.log.emit("incident_episode", **episode)
            self.checkpoint_episode()
            if self.pose_shadow is not None:
                try:
                    await self.pose_shadow.close()
                except Exception as error:
                    self.log.emit("pose_shadow_error", error_type=type(error).__name__)
            if self.recovery:
                self.log.emit(
                    "offline_recovery_skipped",
                    reason="session_ended_before_recovery_completed",
                    remaining_windows=len(self.recovery),
                    truncated=self.recovery_status.get("recovery_truncated", False),
                )
            if self.history.request and not self.history.finished:
                self.log.emit(
                    "buffer_incomplete",
                    **self.identity,
                    buffer_request_id=self.history.request["command_id"],
                    frames_received=self.history.received,
                    frames_processed=self.history.processed,
                )
            self.log.emit(
                "session_finished",
                **self.identity,
                frames_received=self.received,
                frames_processed=self.processed,
                frames_dropped=self.dropped,
                shutdown_frames_dropped=self.shutdown_dropped,
                live_accounting_complete=self.received == self.processed + self.dropped,
                historical_received=self.history.total_received,
                historical_processed=self.history.total_processed,
                historical_complete=self.history.finished,
            )

    def checkpoint_episode(self):
        if self.episode_store is None:
            return
        started = time.monotonic()
        changed = self.episode_store.save(self.controller)
        if changed:
            self.log.emit(
                "episode_checkpoint", **self.identity,
                episode_id=(self.controller.episode or {}).get("episode_id"),
                checkpoint_ms=(time.monotonic() - started) * 1000,
                durable=self.episode_store.path is not None,
            )


class CloudServer:
    def __init__(
        self,
        log: EventLog,
        person_detector_factory=None,
        verification_detector_factory=None,
        experimental_loss_5fps=False,
        experimental_s_only_5fps=False,
        experimental_alert_after_seconds=None,
        experimental_weak_association=False,
        experimental_episode_continuity=False,
        episode_state_file=None,
        diagnostic_disconnect_after_episode_seconds=None,
        pose_shadow_factory=None,
        experimental_pose_observation=False,
    ):
        self.log = log
        self.pose_shadow_factory = pose_shadow_factory
        self.experimental_pose_observation = experimental_pose_observation
        if experimental_pose_observation and (
            pose_shadow_factory is None or not experimental_s_only_5fps
        ):
            raise ValueError("pose observation requires shadow config and S-only mode")
        self.person_detector_factory = person_detector_factory
        self.verification_detector_factory = verification_detector_factory
        self.experimental_loss_5fps = experimental_loss_5fps
        self.experimental_s_only_5fps = experimental_s_only_5fps
        self.experimental_alert_after_seconds = experimental_alert_after_seconds
        self.experimental_weak_association = experimental_weak_association
        if experimental_episode_continuity and not experimental_s_only_5fps:
            raise ValueError("episode continuity requires S-only mode")
        if episode_state_file and not experimental_episode_continuity:
            raise ValueError("episode state file requires continuity mode")
        self.experimental_episode_continuity = experimental_episode_continuity
        value = diagnostic_disconnect_after_episode_seconds
        if value is not None and (
            not experimental_episode_continuity or not math.isfinite(value) or value <= 0
        ):
            raise ValueError("diagnostic disconnect requires continuity and a positive time")
        self.diagnostic_disconnect_after_episode_seconds = value
        self.diagnostic_disconnect_done = False
        self.episode_store = (
            EpisodeStore(episode_state_file) if experimental_episode_continuity else None
        )
        self.active = None
        self.active_done = None
        self.lock = asyncio.Lock()
        self.backend = self.runtime_id = None

    async def handler(self, websocket):
        done = asyncio.Event()
        try:
            initial = await asyncio.wait_for(websocket.recv(), timeout=3)
            if not isinstance(initial, str):
                raise ProtocolError("first message must be EDGE_STATUS")
            status = loads(initial)
            if status["type"] != "EDGE_STATUS":
                raise ProtocolError("first message must be EDGE_STATUS")
            identity = {k: status[k] for k in ("edge_id", "boot_id", "session_id")}
            async with self.lock:
                if self.active:
                    old, old_identity = self.active
                    if old_identity["edge_id"] != identity["edge_id"]:
                        raise ProtocolError("phase 1 supports one edge")
                    old_done = self.active_done
                    await old.close(code=1008, reason="superseded session")
                    if self.experimental_episode_continuity:
                        # Wait for old workers/checkpoint before restoring the new
                        # controller; otherwise the old finalizer can overwrite it.
                        await asyncio.wait_for(old_done.wait(), timeout=3)
                self.active = websocket, identity
                self.active_done = done
            self.log.emit("session_started", **identity)
            self.log.emit(
                "status_received",
                **status,
                frames_received=0,
                frames_processed=0,
                frames_dropped=0,
                initial=True,
            )
            if status.get("offline_frames_lost_at_reconnect", 0):
                self.log.emit(
                    "offline_evidence_gap",
                    **identity,
                    captured_while_offline=status["captured_while_offline"],
                    offline_frames_lost_at_reconnect=status["offline_frames_lost_at_reconnect"],
                    offline_frames_retained_at_reconnect=status[
                        "offline_frames_retained_at_reconnect"
                    ],
                    encrypted_archive_samples_available=status.get("spool_frames", 0),
                    reason="ring_buffer_evicted",
                )
            shadow_options = (
                {"pose_shadow_factory": self.pose_shadow_factory}
                if self.pose_shadow_factory is not None else {}
            )
            if self.experimental_pose_observation:
                shadow_options["experimental_pose_observation"] = True
            await Session(
                websocket,
                identity,
                self.log,
                self.backend,
                self.runtime_id,
                self.person_detector_factory,
                status,
                self.verification_detector_factory,
                self.experimental_loss_5fps,
                self.experimental_s_only_5fps,
                self.experimental_alert_after_seconds,
                self.experimental_weak_association,
                self.experimental_episode_continuity,
                self.diagnostic_disconnect_due,
                self.episode_store,
                **shadow_options,
            ).run()
        except (ProtocolError, ValueError, OSError, TimeoutError, ConnectionClosed) as exc:
            self.log.emit(
                "session_error",
                reason=str(exc) or repr(exc),
                error_type=type(exc).__name__,
                failed_worker=getattr(exc, "failed_worker", None),
            )
            await websocket.close(code=1008, reason="session ended; consult server log")
        finally:
            if self.active and self.active[0] is websocket:
                self.active = None
                self.active_done = None
            done.set()

    def diagnostic_disconnect_due(self, controller, observation):
        value = self.diagnostic_disconnect_after_episode_seconds
        episode = controller.episode
        if value is None or self.diagnostic_disconnect_done or episode is None:
            return False
        if (
            not observation["valid"]
            or episode.get("identity_link_status") != "session_track_only"
            or (observation["captured_mono_ns"] - episode["started_ns"]) / 1e9 < value
        ):
            return False
        self.diagnostic_disconnect_done = True
        return True

    async def run(self, host="127.0.0.1", port=8765, phase=1):
        if self.experimental_pose_observation and phase != 3:
            raise ValueError("pose observation requires Graviton + COOL phase 3")
        cv2.setNumThreads(8 if self.experimental_loss_5fps or self.experimental_s_only_5fps else 1)
        cv2.ocl.setUseOpenCL(False)
        environment = await asyncio.to_thread(runtime_info, phase)
        cool = environment.get("cool", {})
        if cool.get("verified"):
            self.backend = f"opencv-{cv2.__version__}-COOL-motion-probe"
            self.runtime_id = cool["runtime_id"]
        async with serve(
            self.handler,
            host,
            port,
            max_size=MAX_MESSAGE,
            max_queue=2,
            compression=None,
            ping_interval=2,
            ping_timeout=5,
            close_timeout=1,
        ) as server:
            self.log.emit(
                "server_ready",
                host=host,
                port=server.sockets[0].getsockname()[1],
                opencv_version=cv2.__version__,
                cool_verified=cool.get("verified", False),
                temporal_buffer_enabled=True,
                experimental_loss_5fps=self.experimental_loss_5fps,
                experimental_s_only_5fps=self.experimental_s_only_5fps,
                experimental_alert_after_seconds=self.experimental_alert_after_seconds,
                experimental_weak_association=self.experimental_weak_association,
                experimental_episode_continuity=self.experimental_episode_continuity,
                pose_shadow_enabled=self.pose_shadow_factory is not None,
                pose_observation_enabled=self.experimental_pose_observation,
                episode_state_durable=bool(self.episode_store and self.episode_store.path),
                diagnostic_disconnect_after_episode_seconds=(
                    self.diagnostic_disconnect_after_episode_seconds
                ),
                association_confidence=0.2 if self.experimental_weak_association else None,
                new_track_confidence=0.35 if self.experimental_weak_association else None,
                opencv_num_threads=cv2.getNumThreads(),
                phase=phase,
                runtime=environment,
            )
            await asyncio.Future()
