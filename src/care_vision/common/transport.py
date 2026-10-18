"""One pending live frame; bounded control queue with priority over that frame."""

import asyncio
import time
from collections import deque


class Outbox:
    def __init__(self, defer_history=None, history_limits=None):
        self.controls = deque()
        self.frame = None
        self.history = deque()
        self.history_due = 0.0
        self.history_period = 0.05
        self.ready = asyncio.Event()
        self.dropped = 0
        self.defer_history = defer_history or (lambda: False)
        self.history_limits = history_limits
        self.history_last_sent = None
        self.history_last_bytes = 0

    def put_control(self, data):
        if len(self.controls) >= 16:
            raise RuntimeError("control queue full; connection is unresponsive")
        self.controls.append(data)
        self.ready.set()

    def put_frame(self, data):
        if self.frame is not None:
            self.dropped += 1
        self.frame = data
        self.ready.set()

    def put_history(self, records, fps):
        if self.history:
            raise RuntimeError("historical transfer already queued")
        if len(records) > 301 or sum(len(data) for data, _ in records) > 35 * 1024 * 1024:
            raise RuntimeError("historical transfer exceeds queue budget")
        self.history.extend(records)
        self.history_period = 1 / fps
        self.history_due = 0.0
        self.ready.set()

    async def get(self):
        while True:
            if self.controls:
                return self.controls.popleft()
            if self.frame is not None:
                value, self.frame = self.frame, None
                return value
            history_allowed = self.history and not self.defer_history()
            due = self.history_due
            if self.history_limits and self.history_last_sent is not None:
                fps_limit, bytes_per_second = self.history_limits()
                due = self.history_last_sent + max(
                    self.history_period,
                    1 / fps_limit,
                    self.history_last_bytes / bytes_per_second,
                )
            delay = max(0, due - time.monotonic())
            if history_allowed and delay == 0:
                value = self.history.popleft()
                self.history_last_sent = time.monotonic()
                self.history_last_bytes = len(value[0])
                self.history_due = self.history_last_sent + self.history_period
                return value
            self.ready.clear()
            if history_allowed:
                try:
                    async with asyncio.timeout(delay):
                        await self.ready.wait()
                except TimeoutError:
                    pass
            else:
                await self.ready.wait()


async def run_until_first(*coroutines):
    """A disconnect, EOF, or worker failure tears down the entire session."""
    tasks = [asyncio.create_task(coro, name=coro.__qualname__) for coro in coroutines]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                task.result()
            except Exception as exc:
                exc.failed_worker = task.get_name()
                raise
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
