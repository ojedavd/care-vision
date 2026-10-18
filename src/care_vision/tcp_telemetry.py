"""Read-only, bounded macOS telemetry for the owned SSH tunnel process."""

import json
import platform
import shutil
import subprocess
import time
from contextlib import contextmanager


@contextmanager
def mac_tunnel_telemetry(pid, folder):
    if type(pid) is not int or pid <= 0:
        raise ValueError("a positive owned tunnel PID is required")
    executable = shutil.which("nettop") if platform.system() == "Darwin" else None
    if not executable:
        raise RuntimeError("Mac tunnel telemetry requires macOS nettop")
    command = [executable, "-L", "300", "-s", "1", "-n", "-x", "-m", "tcp", "-p", str(pid)]
    metadata = {
        "pid": pid, "command": command, "started_mono_ns": time.monotonic_ns(),
        "scope": "Owned SSH process sockets; select external connection, not process totals",
        "timing": "Native CSV sample clock; lifecycle monotonic times are not sample times",
        "counters": "Cumulative native counters; blank fields are unavailable, not zero",
    }
    process = None
    try:
        with (folder / "mac-tcp.csv").open("x") as output, (
            folder / "mac-tcp.stderr"
        ).open("x") as error:
            process = subprocess.Popen(command, stdout=output, stderr=error)
            try:
                yield
            finally:
                metadata["exited_before_cleanup"] = process.poll() is not None
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                metadata["returncode"] = process.returncode
    finally:
        metadata["finished_mono_ns"] = time.monotonic_ns()
        (folder / "mac-tcp-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
