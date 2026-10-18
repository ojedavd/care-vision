"""One pending logical episode, without pixels or session commands.

Memory survives socket reconnects. Optional private atomic JSON survives a service
restart. Pending metadata is retained until observed closure or explicit operator
deletion; it never expires into NORMAL. One edge is supported, without ReID.
"""

import hashlib
import json
import os
import stat
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


class EpisodeStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.snapshot = None
        if self.path and self.path.exists():
            mode = self.path.lstat().st_mode
            if not stat.S_ISREG(mode) or mode & 0o077:
                raise ValueError("episode metadata must be a private regular file (0600)")
            data = json.loads(self.path.read_text())
            if data.get("schema_version") != 1 or data.get("sha256") != hashlib.sha256(
                canonical(data["snapshot"])
            ).hexdigest():
                raise ValueError("invalid episode metadata version or digest")
            self.snapshot = data["snapshot"]

    def restore(self, controller):
        if self.snapshot is None:
            return False
        if self.snapshot["edge_id"] != controller.identity["edge_id"]:
            raise ValueError("pending episode belongs to another edge")
        if self.snapshot["policy"] != asdict(controller.policy):
            raise ValueError("pending episode requires its original policy")
        controller.restore_episode(deepcopy(self.snapshot["episode"]))
        return True

    def save(self, controller):
        if self.snapshot is not None and (
            self.snapshot["edge_id"] != controller.identity["edge_id"]
            or self.snapshot["policy"] != asdict(controller.policy)
        ):
            raise ValueError("another edge or policy cannot replace pending episode metadata")
        snapshot = None if controller.episode is None else {
            "edge_id": controller.identity["edge_id"],
            "policy": asdict(controller.policy),
            "episode": deepcopy(controller.episode),
        }
        if snapshot == self.snapshot:
            return False
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if snapshot is None:
                self.path.unlink(missing_ok=True)
            else:
                data = {"schema_version": 1, "snapshot": snapshot,
                        "sha256": hashlib.sha256(canonical(snapshot)).hexdigest()}
                fd, name = tempfile.mkstemp(prefix=".episode-", dir=self.path.parent)
                try:
                    with os.fdopen(fd, "wb") as output:
                        output.write(canonical(data))
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(name, self.path)
                finally:
                    Path(name).unlink(missing_ok=True)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        self.snapshot = snapshot
        return True
