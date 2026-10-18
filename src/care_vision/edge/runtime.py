"""Local gateway: capture, sample, encode, transmit, and execute cloud commands."""

import asyncio
import time
from collections import deque

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from care_vision.common.contracts import (
    MAX_MESSAGE,
    ProtocolError,
    dumps,
    envelope,
    loads,
    pack_frame,
    uid,
)
from care_vision.common.history_budget import (
    MAX_HISTORY_SECONDS,
    MIN_HISTORY_BYTES_PER_SECOND,
    MIN_HISTORY_FPS,
)
from care_vision.common.metrics import EventLog
from care_vision.common.transport import Outbox, run_until_first
from care_vision.edge.cloud_feedback import CloudFeedback
from care_vision.edge.commands import CommandHandler
from care_vision.edge.network import NetworkMonitor
from care_vision.edge.recorded_source import RecordedSource
from care_vision.edge.representation import VisualPolicy
from care_vision.edge.sampler import Sampler
from care_vision.edge.source import VideoSource

DRAIN_TIMEOUT_SECONDS = MAX_HISTORY_SECONDS + 10


class Edge:
    def __init__(
        self,
        source: VideoSource,
        url: str,
        log: EventLog,
        edge_id="edge-1",
        output_width=None,
        output_height=None,
        initial_transmit_fps=2,
        replay_sampling_phase_ms=None,
    ):
        self.source, self.url, self.log = source, url, log
        self.output_width = output_width or source.width
        self.output_height = output_height or source.height
        if (self.output_width, self.output_height) not in {(320, 180), (640, 360), (1280, 720)}:
            raise ValueError("unsupported baseline resolution")
        if initial_transmit_fps not in (2, 20):
            raise ValueError("initial transmit FPS must be 2 or 20")
        self.initial_transmit_fps = initial_transmit_fps
        if replay_sampling_phase_ms is not None and (
            not isinstance(source, RecordedSource)
            or initial_transmit_fps != 2
            or type(replay_sampling_phase_ms) is not int
            or not 0 <= replay_sampling_phase_ms < 500
        ):
            raise ValueError("sampling phase requires recorded 2 FPS input and 0-499 ms")
        self.replay_sampling_phase_ms = replay_sampling_phase_ms
        self.identity = {
            "edge_id": edge_id,
            "boot_id": source.boot_id or uid(),
            "session_id": uid(),
        }
        self.sampler = Sampler(initial_transmit_fps)
        self.network_sampler = Sampler(5)
        self.visual = VisualPolicy(
            source.width, source.height, self.output_width, self.output_height
        )
        self.commands = CommandHandler(
            self.identity, self.sampler, source.fps, self.request_buffer, self.visual
        )
        self.outbox: Outbox | None = None
        self.sent = self.selected = self.dropped = self.bytes_sent = 0
        self.network_limited_frames = 0
        self.sent_seqs = deque(maxlen=4096)
        self.historical_sent = 0
        self.last_status = (time.monotonic(), 0, 0)
        self.source_ended = False
        self.sending_header = None
        self.session_sent = self.session_historical_sent = 0
        self.drain_request = None
        self.drain_done = asyncio.Event()
        self.network = NetworkMonitor()
        self.cloud_feedback = CloudFeedback(self.identity)
        self.recovery = {}
        self.latest_history_request_start_ns = None
        self.pending_recovery_context_ns = None

    def status(self):
        now = time.monotonic()
        _, captured, done, error = self.source.snapshot()
        before, before_capture, before_sent = self.last_status
        duration = max(now - before, 1e-9)
        self.last_status = (now, captured, self.sent)
        return envelope(
            "EDGE_STATUS",
            self.identity,
            capture_fps=(captured - before_capture) / duration,
            transmit_fps=(self.sent - before_sent) / duration,
            window_seconds=duration,
            target_transmit_fps=self.sampler.fps,
            effective_target_transmit_fps=min(
                self.sampler.fps, self.network.transmission_cap_fps or 30
            ),
            network_cap_fps=self.network.transmission_cap_fps,
            configured_capture_fps=self.source.fps,
            captured=captured,
            selected=self.selected,
            sent=self.sent,
            dropped=self.dropped + (self.outbox.dropped if self.outbox else 0),
            network_limited_frames=self.network_limited_frames,
            bytes_sent=self.bytes_sent,
            policy_seq=self.commands.policy_seq,
            applied_command_id=self.commands.command_id,
            connection="connected" if self.outbox else "offline",
            network_quality=self.network.quality,
            network_rtt_ms=self.network.rtt_ms,
            network_recovery_threshold_ms=self.network.recovery_threshold_ms,
            cloud_analysis_available=self.outbox is not None,
            cloud_status_version=1,
            **self.cloud_feedback.fields(self.network.quality, now),
            **self.network.outage(captured, now),
            **self.recovery,
            source_state="error" if error else "ended" if done else "running",
            historical_sent=self.historical_sent,
            **self.source.buffer_status(),
        )

    async def report(self, websocket):
        while True:
            await asyncio.sleep(1)
            pong = await websocket.ping()
            try:
                # Keep external session cancellation even if pong completes in
                # the same loop turn (wait_for can swallow it on Python 3.11).
                async with asyncio.timeout(3):
                    rtt_seconds = await pong
            except TimeoutError:
                raise TimeoutError("WebSocket pong not received within 3 seconds") from None
            self.observe_network_rtt(rtt_seconds, time.monotonic_ns())
            status = self.status()
            self.log.emit("edge_status", **status)
            if self.outbox:
                self.outbox.put_control((dumps(status), None))

    def observe_network_rtt(self, rtt_seconds, now_ns):
        prior_quality = self.network.quality
        prior_cap = self.network.transmission_cap_fps
        prior_history_limits = self.history_limits()
        quality = self.network.observed_rtt(rtt_seconds)
        cap = self.network.transmission_cap_fps
        if cap != prior_cap:
            if cap is not None:
                self.network_sampler.set_rate(cap, now_ns)
            self.log.emit(
                "network_transmission_cap_changed",
                cap_fps=cap,
                requested_fps=self.sampler.fps,
                effective_target_fps=min(self.sampler.fps, cap or 30),
                network_rtt_ms=self.network.rtt_ms,
                recovery_threshold_ms=self.network.recovery_threshold_ms,
            )
        if quality != prior_quality:
            self.log.emit(
                "network_transition",
                network_quality=quality,
                network_rtt_ms=self.network.rtt_ms,
                cloud_analysis_available=True,
            )
        if self.history_limits() != prior_history_limits:
            fps, bytes_per_second = self.history_limits()
            self.log.emit(
                "history_transmission_budget_changed",
                wire_max_fps=fps,
                wire_bytes_per_second=bytes_per_second,
                pending_records=len(self.outbox.history) if self.outbox else 0,
                network_quality=quality,
            )
            if self.outbox:
                self.outbox.ready.set()

    async def sample(self):
        while True:
            capture, _, done, error = self.source.snapshot()
            if error:
                raise RuntimeError(error)
            effective_from = max(
                self.commands.policy_effective_from_capture_seq,
                self.visual.effective_from_capture_seq,
            )
            if (
                capture
                and capture.seq >= effective_from
                and self.select_live_capture(capture.seq, capture.mono_ns)
            ):
                sampled = time.monotonic_ns()
                policy_seq = self.commands.policy_seq
                command_id = self.commands.command_id
                target = self.sampler.fps
                self.selected += 1
                jpeg, visual = self.visual.render(capture.jpeg, capture.mono_ns)
                header = envelope(
                    "FRAME",
                    self.identity,
                    evidence_kind="live",
                    frame_id=f"{self.identity['boot_id']}:{capture.seq}",
                    capture_seq=capture.seq,
                    captured_mono_ns=capture.mono_ns,
                    captured_at_utc=capture.at_utc,
                    sampled_mono_ns=sampled,
                    encoding="jpeg",
                    width=self.visual.output_width,
                    height=self.visual.output_height,
                    payload_bytes=len(jpeg),
                    applied_command_id=command_id,
                    policy_seq=policy_seq,
                    target_transmit_fps=target,
                    effective_target_transmit_fps=min(
                        target, self.network.transmission_cap_fps or 30
                    ),
                    network_limited=self.network.transmission_cap_fps is not None
                    and target > self.network.transmission_cap_fps,
                    **visual,
                )
                self.outbox.put_frame((pack_frame(header, jpeg), header))
            if done:
                await self.drain()
                return
            await asyncio.sleep(0.002)

    def select_live_capture(self, seq, captured_ns):
        if not self.sampler.select(seq, captured_ns):
            return False
        cap = self.network.transmission_cap_fps
        if cap is not None and self.sampler.fps > cap:
            if not self.network_sampler.select(seq, captured_ns):
                self.network_limited_frames += 1
                return False
        return True

    async def send(self, websocket):
        while True:
            data, header = await self.outbox.get()
            if (
                header
                and header["evidence_kind"] == "live"
                and header["policy_seq"] != self.commands.policy_seq
            ):
                self.dropped += 1
                continue
            send_started = time.monotonic_ns()
            self.sending_header = header
            async with asyncio.timeout(2):
                await websocket.send(data)
            send_completed = time.monotonic_ns()
            if header:
                if header["evidence_kind"] == "live":
                    self.sent += 1
                    self.session_sent += 1
                else:
                    self.historical_sent += 1
                    self.session_historical_sent += 1
                self.sent_seqs.append(header["capture_seq"])
                self.bytes_sent += len(data)
                self.log.emit(
                    "frame_sent",
                    **header,
                    send_started_mono_ns=send_started,
                    send_completed_mono_ns=send_completed,
                    send_duration_ms=(send_completed - send_started) / 1_000_000,
                )
            self.sending_header = None

    def history_limits(self):
        # This is wire pacing, not the source sampling FPS requested by the agent.
        if self.network.transmission_cap_fps is not None:
            return MIN_HISTORY_FPS, MIN_HISTORY_BYTES_PER_SECOND
        if self.sampler.fps > 2 and not self.source_ended:
            return 5, 256 * 1024
        return 20, 512 * 1024

    async def drain(self):
        self.source_ended = True
        self.outbox.ready.set()  # Recalculate history pacing at EOF.
        self.log.emit("drain_started", timeout_seconds=DRAIN_TIMEOUT_SECONDS)
        try:
            async with asyncio.timeout(DRAIN_TIMEOUT_SECONDS):
                # Finish the last live send before publishing its immutable count.
                while self.outbox.frame is not None or (
                    self.sending_header is not None
                    and self.sending_header.get("evidence_kind", "live") == "live"
                ):
                    await asyncio.sleep(0.01)
                self.drain_request = envelope(
                    "DRAIN_REQUEST",
                    self.identity,
                    drain_id=uid(),
                    live_frames_sent=self.session_sent,
                )
                self.outbox.put_control((dumps(self.drain_request), None))
                await self.drain_done.wait()
        except TimeoutError:
            self.log.emit("drain_timeout", pending_history_records=len(self.outbox.history))
            raise RuntimeError("cloud drain not confirmed before EOF deadline") from None

    def acknowledge_drain(self, message):
        if any(message[k] != self.identity[k] for k in self.identity):
            raise ProtocolError("drain ACK belongs to another session")
        if (
            not self.drain_request
            or message["drain_id"] != self.drain_request["drain_id"]
            or message["live_received"] != self.session_sent
            or message["historical_received"] != self.session_historical_sent
            or self.outbox.history
            or self.outbox.frame is not None
            or self.sending_header is not None
        ):
            raise ProtocolError("drain ACK does not match sent evidence")
        self.log.emit("drain_complete", **message)
        discarded = self.source.acknowledge_captures(self.sent_seqs)
        if discarded:
            self.log.emit("spool_frames_confirmed", **self.identity, frames_removed=discarded)
        self.pending_recovery_context_ns = None
        self.drain_done.set()

    def request_buffer(self, command):
        if self.outbox.history:
            return None
        start, end = command["from_edge_mono_ns"], command["to_edge_mono_ns"]
        captures, status = self.source.recent(start, end, command["history_fps"])
        coverage = "empty"
        if captures:
            oldest = status["buffer_oldest_mono_ns"]
            newest = status["buffer_newest_mono_ns"]
            coverage = (
                "full"
                if not status.get("archive_sparse")
                and oldest is not None
                and newest is not None
                and oldest <= start
                and newest >= end - 1
                else "partial"
            )
        sent = set(self.sent_seqs)
        selected = [capture for capture in captures if capture.seq not in sent]
        records = []
        for index, capture in enumerate(selected):
            header = envelope(
                "FRAME",
                self.identity,
                frame_id=f"{self.identity['boot_id']}:{capture.seq}",
                capture_seq=capture.seq,
                captured_mono_ns=capture.mono_ns,
                captured_at_utc=capture.at_utc,
                sampled_mono_ns=time.monotonic_ns(),
                evidence_kind="historical",
                buffer_request_id=command["command_id"],
                history_index=index,
                history_fps=command["history_fps"],
                encoding="jpeg",
                width=self.source.width,
                height=self.source.height,
                payload_bytes=len(capture.jpeg),
                applied_command_id=None,
                policy_seq=0,
                target_transmit_fps=command["history_fps"],
            )
            records.append((pack_frame(header, capture.jpeg), header))
        complete = envelope(
            "BUFFER_COMPLETE",
            self.identity,
            buffer_request_id=command["command_id"],
            frames_sent=len(selected),
        )
        records.append((dumps(complete), None))
        self.outbox.put_history(records, command["history_fps"])
        start = command["from_edge_mono_ns"]
        self.latest_history_request_start_ns = (
            start
            if self.latest_history_request_start_ns is None
            else min(self.latest_history_request_start_ns, start)
        )
        self.log.emit(
            "history_transmission_budget",
            buffer_request_id=command["command_id"],
            wire_max_fps=self.history_limits()[0],
            wire_bytes_per_second=self.history_limits()[1],
            history_sample_fps=command["history_fps"],
        )
        return {
            "selected_frames": len(selected),
            "selected_bytes": sum(len(c.jpeg) for c in selected),
            "deduplicated_frames": len(captures) - len(selected),
            "coverage": coverage,
            **status,
        }

    async def receive(self, websocket):
        async for raw in websocket:
            if not isinstance(raw, str):
                raise ProtocolError("cloud must send JSON commands")
            command = loads(raw)
            if command["type"] == "CLOUD_STATUS":
                self.cloud_feedback.accept(command)
                self.log.emit(
                    "cloud_status_received",
                    **command,
                    **self.cloud_feedback.fields(self.network.quality),
                )
                continue
            if command["type"] == "DRAINED":
                self.acknowledge_drain(command)
                continue
            if command["type"] not in {
                "SET_TRANSMIT_FPS",
                "REQUEST_RECENT_BUFFER",
                "REQUEST_ROI",
                "REQUEST_FULL_FRAME",
            }:
                raise ProtocolError("expected command")
            _, seq, _, _ = self.source.snapshot()
            ack = self.commands.apply(command, time.monotonic_ns(), seq)
            self.log.emit("command_ack", **ack)
            self.outbox.put_control((dumps(ack), None))
            if (
                command["type"] == "SET_TRANSMIT_FPS"
                and ack["result"] in {"applied", "duplicate"}
                and ack["active_transmit_fps"] == 2
                and self.outbox.history
            ):
                self.log.emit(
                    "history_transmission_resumed",
                    pending_records=len(self.outbox.history),
                    reason="live_returned_to_baseline",
                )

    async def run(self):
        self.source.start()
        backoff = 0.25
        try:
            while True:
                _, _, done, error = self.source.snapshot()
                if error:
                    raise RuntimeError(error)
                if done:
                    break
                try:
                    async with connect(
                        self.url,
                        open_timeout=2,
                        close_timeout=1,
                        ping_interval=2,
                        ping_timeout=5,
                        max_size=MAX_MESSAGE,
                        max_queue=2,
                        compression=None,
                        proxy=None,
                    ) as websocket:
                        self.identity = {**self.identity, "session_id": uid()}
                        self.cloud_feedback = CloudFeedback(self.identity)
                        self.sampler = Sampler()
                        if self.replay_sampling_phase_ms is None:
                            sampler_origin = time.monotonic_ns()
                        else:
                            for _ in range(100):
                                if self.source.replay_origin_ns is not None:
                                    break
                                await asyncio.sleep(0.01)
                            if self.source.replay_origin_ns is None:
                                raise RuntimeError("recorded source did not publish replay origin")
                            sampler_origin = (
                                self.source.replay_origin_ns
                                + self.replay_sampling_phase_ms * 1_000_000
                            )
                        self.sampler.set_rate(self.initial_transmit_fps, sampler_origin)
                        self.network_sampler = Sampler(5)
                        self.visual = VisualPolicy(
                            self.source.width,
                            self.source.height,
                            self.output_width,
                            self.output_height,
                        )
                        self.commands = CommandHandler(
                            self.identity,
                            self.sampler,
                            self.source.fps,
                            self.request_buffer,
                            self.visual,
                        )
                        self.sent_seqs.clear()
                        self.source_ended = False
                        self.sending_header = None
                        self.session_sent = self.session_historical_sent = 0
                        self.drain_request = None
                        self.drain_done = asyncio.Event()
                        self.outbox = Outbox(history_limits=self.history_limits)
                        self.source.set_offline(False)
                        _, captured, _, _ = self.source.snapshot()
                        self.recovery = (
                            self.source.recovery_window(
                                self.network.offline_capture_count + 1,
                                captured,
                                self.pending_recovery_context_ns,
                            )
                            if captured > self.network.offline_capture_count
                            or self.pending_recovery_context_ns is not None
                            else {}
                        )
                        self.latest_history_request_start_ns = None
                        outage = self.network.connected(captured)
                        self.log.emit(
                            "connected",
                            **self.identity,
                            baseline_fps=self.initial_transmit_fps,
                            **outage,
                        )
                        self.log.emit(
                            "network_transition",
                            network_quality="DEGRADED",
                            cloud_analysis_available=True,
                            **outage,
                        )
                        await websocket.send(dumps(self.status()))
                        backoff = 0.25
                        await run_until_first(
                            self.sample(),
                            self.send(websocket),
                            self.receive(websocket),
                            self.report(websocket),
                        )
                except (OSError, ConnectionClosed, TimeoutError, ProtocolError) as exc:
                    self.log.emit(
                        "connection_error",
                        reason=str(exc) or repr(exc),
                        error_type=type(exc).__name__,
                        failed_worker=getattr(exc, "failed_worker", None),
                    )
                finally:
                    if (
                        not self.drain_done.is_set()
                        and self.latest_history_request_start_ns is not None
                    ):
                        start = self.latest_history_request_start_ns
                        self.pending_recovery_context_ns = (
                            start
                            if self.pending_recovery_context_ns is None
                            else min(self.pending_recovery_context_ns, start)
                        )
                        self.log.emit(
                            "unconfirmed_history_context_retained",
                            from_edge_mono_ns=self.pending_recovery_context_ns,
                        )
                        self.latest_history_request_start_ns = None
                    if self.outbox:
                        if self.outbox.history:
                            self.log.emit(
                                "buffer_transfer_aborted", pending_records=len(self.outbox.history)
                            )
                        self.dropped += self.outbox.dropped + int(self.outbox.frame is not None)
                    self.outbox = None
                    self.source.set_offline(True)
                    self.recovery = {}
                    _, captured, _, _ = self.source.snapshot()
                    self.network.disconnected(captured)
                    self.log.emit("offline", **self.identity, cloud_analysis_available=False)
                    self.log.emit(
                        "network_transition",
                        network_quality="OFFLINE",
                        cloud_analysis_available=False,
                    )
                if not self.source.snapshot()[2]:
                    self.log.emit("edge_status", **self.status())
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 2)
            self.log.emit("edge_status", **self.status())
            if not self.drain_done.is_set():
                self.log.emit(
                    "cloud_processing_unconfirmed_at_eof",
                    captured=self.source.count,
                    sent=self.sent,
                    network_quality=self.network.quality,
                )
                raise RuntimeError("source ended without cloud processing confirmation")
        finally:
            await asyncio.to_thread(self.source.close)
            self.log.emit(
                "source_stopped",
                captured=self.source.count,
                sent=self.sent,
                source_error=self.source.error,
            )
