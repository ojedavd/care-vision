"""Two bounded recorded Edge replays against isolated Graviton workers; no camera."""

import argparse
import gzip
import json
import os
import select
import shlex
import socket
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

from run_pose_observation_loop import LAUNCH, READ_COMPRESSED, READY, SSH, STOP, save, sha

PREVIOUS_STATE = (
    "/home/ubuntu/care-vision-phase3/episode-state/continuity-20261014-a/episode.json"
)
PREVIOUS_SHA = "0b8a09695b9fbba16fd3c2f5fece5b668cee1567e973ed8e2152f1c40678d7c5"
INSPECT = """import hashlib,json,pathlib,platform,sys
p=pathlib.Path(sys.argv[1])
print(json.dumps({'machine':platform.machine(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                 'mode':oct(p.stat().st_mode&0o777)}))
"""
PROBE_LAUNCH = LAUNCH.replace(
    "'-m','care_vision.cloud'", "str(base/'tools/pose_continuity_cloud_probe.py')"
).replace(
    "'--experimental-pose-observation']",
    "'--experimental-pose-observation','--experimental-episode-continuity',"
    "'--episode-state-file',sys.argv[4]]",
)
PROBE_STOP = STOP.replace(
    "b'care_vision.cloud' not in cmd",
    "str(run.parent/'tools/pose_continuity_cloud_probe.py').encode() not in cmd",
)


def run(args):
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    ssh = SSH(args.host, args.identity)
    previous = json.loads(ssh.python(INSPECT, PREVIOUS_STATE))
    save(output / "previous-state-before.json", previous)
    assert previous == {"machine": "aarch64", "sha256": PREVIOUS_SHA, "mode": "0o600"}
    nonce = uuid.uuid4().hex[:10]
    remote = "/home/ubuntu/care-vision-pose-continuity-" + nonce
    state = (
        "/home/ubuntu/care-vision-phase3/episode-state/pose-continuity-" + nonce + "/episode.json"
    )
    source = root / "artifacts/pose-observation-live-20261014/sit-return-live-2"
    config_source = root / (
        "artifacts/pose-observation-live-20261014/preparation-hold-1/pose-config.json"
    )
    config = json.loads(config_source.read_text())
    config["context_provenance"] += " Replay only: same reviewed sit-return-live-2 recording."
    save(output / "pose-config.json", config)
    files = sorted((root / "src").rglob("*.py")) + [
        root / "tools/pose_continuity_cloud_probe.py", Path(__file__).resolve(),
        root / "tools/run_pose_observation_loop.py",
    ]
    models = {
        "models/person.onnx": root / (
            "artifacts/phase5c-dataset-review-20260920/models/object_detection_yolox_2022nov.onnx"
        ),
        "models/pose.onnx": root / "artifacts/phase7d-pose-20260921/models/end2end.onnx",
    }
    plan = {
        "scope": "Real Edge timestamp-paced archived video; new CV on AWS; controlled WS close",
        "source": str(source.relative_to(root)), "new_camera_capture": False,
        "source_role": "previously reviewed development recording, original ~25 FPS timestamps",
        "runs": ["cut-roi", "restart-from-disk"], "state_file": state,
        "cut": "first fresh ROI observation matched by PoseObservationCycle after ACK",
        "cut_scope": "temporal valid may be false after view change; detector/tracker must be ok",
        "restart": "stop first owned cloud process, start new process using same snapshot",
        "code_sha256": {str(p.relative_to(root)): sha(p) for p in files},
        "models": {n: sha(p) for n, p in models.items()},
        "inputs": {str(p.relative_to(root)): sha(p) for p in (
            source / "demo.avi", source / "demo.captures.jsonl", config_source,
        )},
        "config_sha256": sha(output / "pose-config.json"),
        "acceptance": [
            "COOL verified on Graviton; first cycle interrupted after ROI evidence",
            "same episode, origin, reference and spent ROI budget survive reconnect and restart",
            "new session has full frame and new 5 FPS command, ACK and fresh evidence then 2 FPS",
            "one ROI total; no episode closure by unlinked identity; no alert",
            "all sent/received/processed/dropped frame differences explicitly accounted",
            "previous snapshot unchanged; own workers stopped; experimental switches remain off",
        ],
    }
    save(output / "plan.json", plan)
    with tarfile.open(output / "package.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        for name, path in models.items():
            archive.add(path, arcname=name, recursive=False)
        archive.add(output / "pose-config.json", arcname="pose-config.json", recursive=False)
    deployment = {"workspace": remote, "state_file": state, "instance_left_running": True,
                  "package_sha256": sha(output / "package.tar.gz"), "complete": False}
    save(output / "deployment.json", deployment)
    setup = """import io,os,pathlib,sys,tarfile
os.umask(0o077);p=pathlib.Path(sys.argv[1]);p.mkdir(exist_ok=False)
state=pathlib.Path(sys.argv[2]);state.parent.mkdir(exist_ok=False)
tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz').extractall(p,filter='data')
"""
    ssh.python(setup, remote, state, payload=(output / "package.tar.gz").read_bytes(), timeout=90)
    for name in plan["runs"]:
        destination = output / name
        destination.mkdir(mode=0o700)
        remote_run = remote + "/" + name
        tunnel = None
        launched = False
        print("Starting " + name, flush=True)
        try:
            result = json.loads(ssh.python(PROBE_LAUNCH, remote, name, "pose-config.json", state))
            launched = True
            save(destination / "deployment.json", result)
            ready = json.loads(ssh.python(READY, remote_run))
            save(destination / "runtime.json", ready)
            assert ready["cool_verified"] and ready["phase"] == 3
            assert ready["pose_observation_enabled"] and ready["episode_state_durable"]
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            marker = "import time;print('TUNNEL_READY',flush=True);time.sleep(120)"
            with (destination / "ssh.stderr").open("w") as error:
                tunnel = subprocess.Popen([
                    *ssh.args, "-o", "ExitOnForwardFailure=yes", "-L",
                    f"127.0.0.1:{port}:127.0.0.1:{ready['port']}", ssh.host,
                    shlex.join(["python3", "-c", marker]),
                ], stdout=subprocess.PIPE, stderr=error)
                readable, _, _ = select.select([tunnel.stdout], [], [], 20)
                if not readable or tunnel.stdout.readline().strip() != b"TUNNEL_READY":
                    raise RuntimeError("probe tunnel unavailable")
                with (destination / "edge.stderr").open("w") as edge_log:
                    subprocess.run([
                        sys.executable, "-m", "care_vision.edge", "--video",
                        str(source / "demo.avi"),
                        "--replay-captures", str(source / "demo.captures.jsonl"),
                        "--replay-sampling-phase-ms", "0", "--capture-fps", "25",
                        "--capture-resolution", "1280x720", "--baseline-resolution", "1280x720",
                        "--url", f"ws://127.0.0.1:{port}", "--log", str(destination / "edge.jsonl"),
                    ], stdout=edge_log, stderr=subprocess.STDOUT, check=True, timeout=90)
        finally:
            if tunnel is not None:
                tunnel.terminate()
                try:
                    tunnel.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    tunnel.kill()
                    tunnel.wait()
            if launched:
                stopped = ssh.python(PROBE_STOP, remote_run).decode()
                save(destination / "cleanup.json", {"confirmed_stopped": True, "message": stopped})
                for filename in ("cloud.jsonl", "cloud.stderr"):
                    (destination / filename).write_bytes(gzip.decompress(ssh.python(
                        READ_COMPRESSED, remote_run + "/" + filename, timeout=55,
                    )))
        snapshot = gzip.decompress(ssh.python(READ_COMPRESSED, state))
        (destination / "episode.json").write_bytes(snapshot)
        save(destination / "state-metadata.json", json.loads(ssh.python(INSPECT, state)))
        assert json.loads(snapshot)["snapshot"]["episode"]["pose_observation_requested"]
        if name == "cut-roi":
            events = [json.loads(line) for line in
                      (destination / "cloud.jsonl").read_text().splitlines()]
            assert sum(r["event"] == "pose_continuity_probe_trigger" for r in events) == 1
            assert any(r["event"] == "episode_restored" for r in events)
        print("Finished " + name, flush=True)
    after = json.loads(ssh.python(INSPECT, PREVIOUS_STATE))
    save(output / "previous-state-after.json", after)
    assert after == previous
    cleanup = """import json,pathlib,shutil,sys
p=pathlib.Path(sys.argv[1]);assert p.parent==pathlib.Path('/home/ubuntu')
assert p.name.startswith('care-vision-pose-continuity-')
shutil.rmtree(p);print(json.dumps({'workspace_removed':not p.exists()}))
"""
    deployment.update(complete=True, cleanup=json.loads(ssh.python(cleanup, remote)),
                      new_private_state_retained=True, previous_state_unchanged=True)
    save(output / "deployment.json", deployment)
    print("Two recorded continuity replays finished; audit still required", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    os.umask(0o077)
    run(parser.parse_args())
