"""Bounded projections of private logs. No raw records or free-form text leave here."""

import json
import math
import re
import time
from collections import deque
from pathlib import Path

EVENTS = {
    "server_ready",
    "session_started",
    "session_ended",
    "agent_action",
    "ack_received",
    "agent_state_transition",
    "observation",
    "test_alert",
    "frame_received",
    "frame_dropped",
    "process_resources",
    "server_stopped",
}
ACTIONS = {
    "SET_TRANSMIT_FPS",
    "SET_RESOLUTION",
    "REQUEST_ROI",
    "REQUEST_FULL_FRAME",
    "REQUEST_RECENT_BUFFER",
    "RETURN_TO_BASELINE",
    "WAIT",
    "KEEP_OBSERVING",
}
STATES = {
    "NORMAL",
    "SUSPICIOUS",
    "POSSIBLE_FALL",
    "VERIFYING",
    "INCIDENT",
    "RECOVERY",
    "ASSISTANCE_REQUIRED",
    "UNRESOLVED",
    "UNKNOWN",
}


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


class Projection:
    def __init__(self):
        self.counts = {
            "received": 0,
            "processed": 0,
            "dropped": 0,
            "live_processed": 0,
            "historical_processed": 0,
        }
        self.timeline = deque(maxlen=60)
        self.commands = {}
        self.observations = {}
        self.metrics = {}
        self.runtime = {"cool_verified": False, "opencv": None, "phase": None}
        self.agent_state = "UNKNOWN"
        self.last_observation = None
        self.last_event = None
        self.edge_connected = False
        self.cloud_port = None
        self.malformed = 0
        self.fps_times = deque(maxlen=120)

    def accept(self, row):
        event = row.get("event")
        if event not in EVENTS:
            return
        self.last_event = time.monotonic()
        mono = number(row.get("local_mono_ns"))
        if event == "server_ready":
            phase = row.get("phase")
            self.runtime = {
                "cool_verified": row.get("cool_verified") is True and phase == 3,
                "opencv": str(row.get("opencv_version", ""))[:24]
                if re.fullmatch(r"[0-9][a-zA-Z0-9.+_-]{0,40}", str(row.get("opencv_version", "")))
                else None,
                "phase": phase if phase in (1, 2, 3) else None,
            }
            self.cloud_port = row.get("port") if type(row.get("port")) is int else None
        elif event == "session_started":
            self.edge_connected = True
        elif event in {"session_ended", "server_stopped"}:
            self.edge_connected = False
        elif event == "frame_received":
            self.counts["received"] += 1
        elif event == "frame_dropped":
            self.counts["dropped"] += 1
        elif event == "observation":
            self.counts["processed"] += 1
            live = row.get("evidence_kind", "live") == "live"
            self.counts["live_processed" if live else "historical_processed"] += 1
            if live:
                self.last_observation = time.monotonic()
                if mono is not None:
                    self.fps_times.append(mono)
            for key in ("processing_ms", "target_transmit_fps", "cloud_queue_wait_ms"):
                self.metrics[key] = number(row.get(key))
            self.metrics["valid"] = row.get("valid") is True
            self.metrics["roi"] = row.get("view_kind") == "roi"
            observation_id = row.get("observation_id")
            if isinstance(observation_id, str) and len(observation_id) <= 200:
                self.observations[observation_id] = mono
                if len(self.observations) > 400:
                    del self.observations[next(iter(self.observations))]
            # Coordinates, tracks, images, host paths and identifiers are not exposed.
            for command_id in (row.get("applied_command_id"), row.get("applied_visual_command_id")):
                command = self.commands.get(command_id) if isinstance(command_id, str) else None
                if (
                    command
                    and command["caused_by_seen_observation"]
                    and command["ack"]
                    and live
                    and row.get("valid") is True
                    and mono is not None
                    and mono > command["ack_ns"]
                ):
                    if not command["perceived"]:
                        command["perceived"] = True
                        self.timeline.append(
                            {
                                "stage": "perceive_again",
                                "action": command["action"],
                                "index": command["index"],
                            }
                        )
        elif event == "agent_action":
            key = row.get("command_id")
            if not isinstance(key, str) or len(key) > 200:
                return
            action = row.get("type")
            if action not in ACTIONS:
                return
            if key not in self.commands:
                index = self.counts.get("commands", 0) + 1
                self.counts["commands"] = index
                self.commands[key] = {
                    "action": action,
                    "index": index,
                    "ack": False,
                    "ack_ns": 0,
                    "issued_ns": mono,
                    "perceived": False,
                    "caused_by_seen_observation": (
                        isinstance(row.get("caused_by_observation_id"), str)
                        and self.observations.get(row["caused_by_observation_id"]) is not None
                        and mono is not None
                        and self.observations[row["caused_by_observation_id"]] < mono
                    ),
                }
                if len(self.commands) > 200:
                    del self.commands[next(iter(self.commands))]
                self.timeline.append({"stage": "decision_action", "action": action, "index": index})
            if row.get("agent_state") in STATES:
                self.agent_state = row["agent_state"]
        elif event == "ack_received":
            key = row.get("command_id")
            command = self.commands.get(key) if isinstance(key, str) else None
            if (
                command
                and row.get("result") in {"applied", "duplicate"}
                and mono is not None
                and command["issued_ns"] is not None
                and mono >= command["issued_ns"]
            ):
                if not command["ack"]:
                    self.timeline.append(
                        {
                            "stage": "edge_ack",
                            "action": command["action"],
                            "index": command["index"],
                        }
                    )
                command.update(ack=True, ack_ns=mono)
        elif event == "agent_state_transition":
            state = row.get("to_state", row.get("to"))
            if state in STATES:
                self.agent_state = state
        elif event == "test_alert":
            self.timeline.append({"stage": "test_alert", "action": "NO_EXTERNAL_NOTIFICATION"})
        elif event == "process_resources":
            for key in ("cpu_percent_one_core", "rss_bytes"):
                self.metrics[key] = number(row.get(key))

    def snapshot(self):
        age = None if self.last_observation is None else time.monotonic() - self.last_observation
        fps = None
        if len(self.fps_times) > 1 and self.fps_times[-1] > self.fps_times[0]:
            fps = (len(self.fps_times) - 1) * 1e9 / (self.fps_times[-1] - self.fps_times[0])
        return {
            "runtime": self.runtime,
            "counts": dict(self.counts),
            "metrics": dict(self.metrics),
            "agent_state": self.agent_state,
            "timeline": list(self.timeline),
            "edge_connected": self.edge_connected,
            "observation_age_seconds": age,
            "vision_live": self.edge_connected and age is not None and age < 5,
            "processed_fps": fps,
            "malformed_records": self.malformed,
            "causal_cycles": sum(c["perceived"] for c in self.commands.values()),
        }


class LogTail:
    def __init__(self, path, projection):
        self.path, self.projection = Path(path), projection
        self.offset = 0
        self.identity = None

    def poll(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return
        identity = (stat.st_dev, stat.st_ino)
        if self.identity is not None and (identity != self.identity or stat.st_size < self.offset):
            # Never combine a replacement/truncated run with the previous session.
            self.projection.__init__()
            self.offset = 0
        self.identity = identity
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            for _ in range(2000):
                line = stream.readline(262145)
                if not line or not line.endswith(b"\n"):
                    if len(line) > 262144:
                        self.projection.malformed += 1
                        self.offset = stat.st_size
                    break
                self.offset = stream.tell()
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError()
                    self.projection.accept(row)
                except (ValueError, TypeError, KeyError, OverflowError):
                    self.projection.malformed += 1
