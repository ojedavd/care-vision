"""One bounded historical transfer, with a separate visual timeline from live input."""

import asyncio
import time

from care_vision.cloud.vision import MotionVision
from care_vision.common.contracts import ProtocolError
from care_vision.common.history_budget import HISTORY_IDLE_SECONDS, history_timeout_seconds


class HistoryTransfer:
    def __init__(
        self,
        log,
        backend=None,
        runtime_id=None,
        person_detector_factory=None,
        defer_processing=None,
    ):
        self.log, self.backend, self.runtime_id = log, backend, runtime_id
        self.person_detector_factory = person_detector_factory
        self.defer_processing = defer_processing or (lambda: False)
        self.queue = asyncio.Queue(maxsize=301)
        self.request = self.ack = None
        self.finished = True
        self.complete = False
        self.received = self.processed = self.bytes = 0
        self.total_received = self.total_processed = 0
        self.last_seq = self.last_ns = 0
        self.started = 0.0
        self.last_progress = 0.0
        self.timeout_seconds = 30
        self.vision = None

    def start(self, request):
        if not self.finished:
            raise ProtocolError("historical transfer still active")
        self.request, self.ack = request, None
        self.finished = self.complete = False
        self.received = self.processed = self.bytes = 0
        self.last_seq = self.last_ns = 0
        self.started = time.monotonic()
        self.last_progress = self.started
        self.timeout_seconds = 30
        person_detector = self.person_detector_factory() if self.person_detector_factory else None
        self.vision = MotionVision(
            self.backend, self.runtime_id, person_detector, timeline="historical"
        )

    def acknowledge(self, ack):
        if not self.request or ack["command_id"] != self.request["command_id"]:
            raise ProtocolError("unsolicited buffer ACK")
        if (
            ack.get("command_type") != "REQUEST_RECENT_BUFFER"
            or ack["command_seq"] != self.request["command_seq"]
            or ack["result"] not in {"applied", "duplicate"}
        ):
            raise ProtocolError("buffer request was not accepted")
        if self.ack and any(
            ack[k] != self.ack[k]
            for k in ("selected_frames", "selected_bytes", "coverage", "deduplicated_frames")
        ):
            raise ProtocolError("inconsistent buffer ACK")
        if self.ack is None:
            self.last_progress = time.monotonic()
            self.timeout_seconds = history_timeout_seconds(
                ack["selected_frames"], ack["selected_bytes"], self.request["history_fps"]
            )
            self.log.emit(
                "history_deadline",
                buffer_request_id=self.request["command_id"],
                timeout_seconds=self.timeout_seconds,
                idle_timeout_seconds=HISTORY_IDLE_SECONDS,
                selected_frames=ack["selected_frames"],
                selected_bytes=ack["selected_bytes"],
            )
        self.ack = ack

    def check_timeout(self, now):
        if self.finished:
            return
        if not self.ack and now - self.started >= 5:
            raise ProtocolError("historical ACK timeout")
        if now - self.started >= self.timeout_seconds:
            raise ProtocolError("historical transfer total deadline exceeded")
        if self.ack and now - self.last_progress >= HISTORY_IDLE_SECONDS:
            raise ProtocolError("historical transfer made no progress before idle deadline")

    def check_request(self, message):
        if (
            not self.request
            or self.complete
            or message["buffer_request_id"] != self.request["command_id"]
            or not self.ack
        ):
            raise ProtocolError("unsolicited or completed historical transfer")

    def receive(self, header, jpeg):
        self.check_request(header)
        if (
            not self.request["from_edge_mono_ns"]
            <= header["captured_mono_ns"]
            < self.request["to_edge_mono_ns"]
        ):
            raise ProtocolError("historical frame outside requested window")
        if (
            header["capture_seq"] <= self.last_seq
            or header["captured_mono_ns"] <= self.last_ns
            or header["history_index"] != self.received
            or header["history_fps"] != self.request["history_fps"]
        ):
            raise ProtocolError("historical order or sampling mismatch")
        if (
            self.received >= self.ack["selected_frames"]
            or self.bytes + len(jpeg) > self.ack["selected_bytes"]
        ):
            raise ProtocolError("historical transfer exceeds acknowledged budget")
        self.last_seq, self.last_ns = header["capture_seq"], header["captured_mono_ns"]
        self.received += 1
        self.total_received += 1
        self.bytes += len(jpeg)
        self.last_progress = time.monotonic()
        self.queue.put_nowait((header, jpeg))

    def finish_receiving(self, message):
        self.check_request(message)
        if (
            message["frames_sent"] != self.received
            or self.received != self.ack["selected_frames"]
            or self.bytes != self.ack["selected_bytes"]
        ):
            raise ProtocolError("incomplete historical transfer")
        self.complete = True
        self.queue.put_nowait((message, None))

    async def perceive(self):
        while True:
            header, jpeg = await self.queue.get()
            if jpeg is None:
                self.log.emit(
                    "buffer_processed",
                    **header,
                    frames_processed=self.processed,
                    coverage=self.ack["coverage"],
                )
                self.finished = True
            else:
                deferred = False
                while self.defer_processing():
                    if not deferred:
                        self.log.emit(
                            "buffer_processing_deferred",
                            buffer_request_id=self.request["command_id"],
                            reason="pending_live_frames_have_priority",
                        )
                        deferred = True
                    await asyncio.sleep(0.01)
                if deferred:
                    self.log.emit(
                        "buffer_processing_resumed",
                        buffer_request_id=self.request["command_id"],
                    )
                observation = await asyncio.to_thread(self.vision.observe, header, jpeg)
                # Age is intentional for historical evidence. Never feed the live controller.
                self.processed += 1
                self.total_processed += 1
                self.last_progress = time.monotonic()
                self.log.emit("observation", **observation)
