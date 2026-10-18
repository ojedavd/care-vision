"""Local monotonic timings; UTC timestamps are diagnostic, not synchronized latency."""

import json
import time
from pathlib import Path

from care_vision.common.contracts import utc


class EventLog:
    def __init__(self, component: str, path: str | Path | None = None):
        self.component = component
        self.file = None
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("w", encoding="utf-8")

    def emit(self, event: str, **fields):
        record = {
            "event": event,
            "component": self.component,
            "at_utc": utc(),
            "local_mono_ns": time.monotonic_ns(),
            **fields,
        }
        line = json.dumps(record, allow_nan=False)
        if self.file:
            self.file.write(line + "\n")
            self.file.flush()
        else:
            print(line, flush=True)

    def close(self):
        if self.file:
            self.file.close()
