"""Own only the explicitly launched child process, with bounded lifetime."""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

from care_vision.judge.state import LogTail, Projection


class Sessions:
    def __init__(
        self,
        root,
        *,
        synthetic=False,
        cloud_python=None,
        cloud_source=None,
        cloud_args=(),
        seconds=180,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.synthetic = synthetic
        self.python = cloud_python or sys.executable
        self.source = Path(cloud_source).resolve() if cloud_source else None
        self.cloud_args = list(cloud_args)
        self.seconds = seconds
        self.lock = threading.RLock()
        self.process = None
        self.tail = None
        self.projection = Projection()
        self.started = None
        self.state = "IDLE"
        self.output = None

    def start(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                raise ValueError("session already running")
            run = self.root / uuid4().hex
            run.mkdir(mode=0o700)
            self.projection = Projection()
            self.tail = LogTail(run / "cloud.jsonl", self.projection)
            if self.synthetic:
                argv = [self.python, "-u", "-m", "care_vision.demo", "--output", str(run / "demo")]
                self.tail = LogTail(run / "demo/cloud.jsonl", self.projection)
                env = dict(os.environ)
            else:
                argv = [
                    self.python,
                    "-u",
                    "-m",
                    "care_vision.cloud",
                    "--phase",
                    "3",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "0",
                    "--log",
                    str(run / "cloud.jsonl"),
                    "--run-seconds",
                    str(self.seconds),
                    *self.cloud_args,
                ]
                env = {
                    **os.environ,
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": str(self.source / "src") + ":/opt/cool/python_3.11/site-packages",
                    "LD_LIBRARY_PATH": "/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib",
                }
                env.pop("OPENCV_FORCE_DNN_ENGINE", None)
            # Portal secrets never reach the inference child.
            for key in tuple(env):
                if key.startswith("CARE_JUDGE_"):
                    env.pop(key)
            self.output = (run / "worker.stderr").open("wb")
            try:
                self.process = subprocess.Popen(
                    argv,
                    cwd=self.source,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=self.output,
                    stderr=self.output,
                    start_new_session=True,
                )
            except Exception:
                self.output.close()
                self.state = "FAILED"
                raise
            self.started = time.monotonic()
            self.state = "STARTING"

    def stop(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=3)
            if self.output:
                self.output.close()
            self.state = "STOPPED"
            if self.tail:
                self.tail.poll()
            self.projection.edge_connected = False

    def snapshot(self, operator=False):
        with self.lock:
            if self.tail:
                self.tail.poll()
            running = self.process is not None and self.process.poll() is None
            if running and time.monotonic() - self.started > self.seconds:
                self.stop()
                running = False
                self.state = "TIME_LIMIT"
            elif self.process and not running and self.state not in {"STOPPED", "TIME_LIMIT"}:
                self.state = "COMPLETED" if self.process.returncode == 0 else "FAILED"
                self.projection.edge_connected = False
                if self.output:
                    self.output.close()
            elif running:
                self.state = "RUNNING" if self.projection.runtime["phase"] else "STARTING"
            result = self.projection.snapshot()
            result.update(
                session_state=self.state,
                mode="SYNTHETIC_LOCAL_NOT_COOL" if self.synthetic else "GRAVITON_COOL",
                operator=operator,
                person_detector_configured="--person-model" in self.cloud_args,
                image_policy="NO_IMAGES_SERVED",
                test_alerts_only=True,
                session_limit_seconds=self.seconds,
            )
            if operator and running:
                result["cloud_port"] = self.projection.cloud_port
            return result
