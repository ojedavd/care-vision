"""Five opt-in classic YOLOX replays on existing recordings; no camera."""

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

STATE_HASHES = """import hashlib,json,pathlib
base=pathlib.Path('/home/ubuntu/care-vision-phase3/episode-state')
print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(base.glob('*/episode.json'))}))
"""
START = LAUNCH.replace(
    "'--experimental-pose-observation']",
    "'--experimental-pose-observation']\n"
    "cmd += ['--experimental-episode-continuity','--episode-state-file',str(run/'episode.json'),\n"
    "        '--experimental-yolox-classic-engine']\n"
    "if sys.argv[4]=='test20':cmd += ['--experimental-alert-after-seconds','20']",
)


def run(args):
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    ssh = SSH(args.host, args.identity)
    before = json.loads(ssh.python(STATE_HASHES))
    save(output / "existing-state-before.json", before)
    config_source = root / (
        "artifacts/pose-observation-live-20261014/preparation-hold-1/pose-config.json"
    )
    config = json.loads(config_source.read_text())
    config["context_provenance"] += " Existing same-view October14 recordings only."
    save(output / "pose-current.json", config)
    config["context_status"] = "unknown"
    config["context_provenance"] = (
        "October5 view differs from October14 calibration; no seat inference."
    )
    save(output / "pose-unknown.json", config)
    cases = [
        ("recovery-2", "baseline"),
        ("no-recovery-1", "baseline"),
        ("no-recovery-1", "test20"),
        ("sit-return-live-2", "pose"),
        ("sit-hold-live-1", "pose"),
    ]
    sources = {
        name: root
        / "artifacts"
        / ("pose-observation-live-20261014" if mode == "pose" else "tapo-weak-live-20261005")
        / name
        for name, mode in cases
    }
    files = sorted((root / "src").rglob("*.py")) + [
        Path(__file__).resolve(),
        root / "tools/run_pose_observation_loop.py",
    ]
    models = {
        "models/person.onnx": root
        / ("artifacts/phase5c-dataset-review-20260920/models/object_detection_yolox_2022nov.onnx"),
        "models/pose.onnx": root / "artifacts/phase7d-pose-20260921/models/end2end.onnx",
    }
    plan = {
        "scope": "Opt-in classic YOLOX: recorded end-to-end episode and pose/ROI regression",
        "conditions": [{"source": name, "mode": mode} for name, mode in cases],
        "profile": "YOLOX-S640, weak .20/strong .35, NMS .30, 720p, 8 threads, COOL; 2/5 FPS",
        "baseline": "Current opt-in continuity/pose observer; alert threshold remains None",
        "test20": "Separate existing diagnostic opt-in T20; test-only, no external notification",
        "limits": [
            "Known development recordings; not accidental falls or held-out evaluation",
            "Sparse visual review, not dense frame ground truth",
            "October5 context unknown; October14 reviewed context current for archived sources",
            "One run per condition; no thresholds changed on failure",
        ],
        "code_sha256": {str(p.relative_to(root)): sha(p) for p in files},
        "model_sha256": {name: sha(p) for name, p in models.items()},
        "input_sha256": {
            str(p.relative_to(root)): sha(p)
            for source in sources.values()
            for p in (source / "demo.avi", source / "demo.captures.jsonl")
        },
        "config_sha256": {p.name: sha(p) for p in output.glob("pose-*.json")},
        "new_camera": False,
        "engine": "classic",
        "pose_engine": "unchanged",
        "acceptance": [
            "All sent/received/processed IDs accounted; report loss/caps",
            "Both baseline videos produce same-track horizontal evidence",
            "Recovery closes only on sustained same-track reference posture",
            "No-recovery remains open at end with no false closure",
            "Baseline: zero alerts; test20: one test-only alert after >=20s",
            "Unknown context blocks pose; current context completes ROI/return; final2FPS",
            "Pose-return closes upright; pose-hold stays open, no seated closure",
            "Every person observation records explicit classic; pose engine unchanged",
            "Owned workers stopped, snapshots saved, pre-existing state unchanged",
        ],
    }
    save(output / "plan.json", plan)
    with tarfile.open(output / "package.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        for name, path in models.items():
            archive.add(path, arcname=name, recursive=False)
        for config_path in output.glob("pose-*.json"):
            archive.add(config_path, arcname=config_path.name, recursive=False)
    remote = "/home/ubuntu/care-vision-classic-integration-" + uuid.uuid4().hex[:10]
    deployment = {
        "workspace": remote,
        "complete": False,
        "package_sha256": sha(output / "package.tar.gz"),
        "instance_left_running": True,
    }
    save(output / "deployment.json", deployment)
    setup = """import io,os,pathlib,sys,tarfile
os.umask(0o077);p=pathlib.Path(sys.argv[1]);p.mkdir(exist_ok=False)
tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz').extractall(p,filter='data')
"""
    ssh.python(setup, remote, payload=(output / "package.tar.gz").read_bytes(), timeout=90)
    for name, mode in cases:
        label = name + "-" + mode
        destination, remote_run = output / label, remote + "/" + label
        destination.mkdir(mode=0o700)
        source = sources[name]
        tunnel, launched = None, False
        print("Starting " + label, flush=True)
        try:
            config_name = "pose-current.json" if mode == "pose" else "pose-unknown.json"
            result = json.loads(ssh.python(START, remote, label, config_name, mode))
            launched = True
            save(destination / "deployment.json", result)
            ready = json.loads(ssh.python(READY, remote_run))
            save(destination / "runtime.json", ready)
            assert ready["cool_verified"] and ready["opencv_num_threads"] == 8
            assert ready["pose_observation_enabled"] is True
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            marker = "import time;print('TUNNEL_READY',flush=True);time.sleep(120)"
            with (destination / "ssh.stderr").open("w") as error:
                tunnel = subprocess.Popen(
                    [
                        *ssh.args,
                        "-o",
                        "ExitOnForwardFailure=yes",
                        "-L",
                        f"127.0.0.1:{port}:127.0.0.1:{ready['port']}",
                        ssh.host,
                        shlex.join(["python3", "-c", marker]),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=error,
                )
                readable, _, _ = select.select([tunnel.stdout], [], [], 20)
                if not readable or tunnel.stdout.readline().strip() != b"TUNNEL_READY":
                    raise RuntimeError("comparison tunnel unavailable")
                edge_command = [
                    sys.executable,
                    "-m",
                    "care_vision.edge",
                    "--video",
                    str(source / "demo.avi"),
                    "--replay-captures",
                    str(source / "demo.captures.jsonl"),
                    "--capture-fps",
                    "25",
                    "--capture-resolution",
                    "1280x720",
                    "--baseline-resolution",
                    "1280x720",
                    "--url",
                    f"ws://127.0.0.1:{port}",
                    "--log",
                    str(destination / "edge.jsonl"),
                ]
                edge_command += ["--replay-sampling-phase-ms", "0"]
                with (destination / "edge.stderr").open("w") as edge_log:
                    subprocess.run(
                        edge_command,
                        stdout=edge_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=90,
                    )
        finally:
            if tunnel is not None:
                tunnel.terminate()
                try:
                    tunnel.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    tunnel.kill()
                    tunnel.wait()
            if launched:
                stopped = ssh.python(STOP, remote_run).decode()
                save(destination / "cleanup.json", {"confirmed_stopped": True, "message": stopped})
                for filename in ("cloud.jsonl", "cloud.stderr"):
                    (destination / filename).write_bytes(
                        gzip.decompress(
                            ssh.python(
                                READ_COMPRESSED,
                                remote_run + "/" + filename,
                                timeout=55,
                            )
                        )
                    )
                if launched:
                    state = ssh.python(
                        "import pathlib,sys;p=pathlib.Path(sys.argv[1]);"
                        "sys.stdout.buffer.write(p.read_bytes() if p.exists() else b'null')",
                        remote_run + "/episode.json",
                    )
                    (destination / "episode.json").write_bytes(state)
        print("Finished " + label, flush=True)
    after = json.loads(ssh.python(STATE_HASHES))
    save(output / "existing-state-after.json", after)
    assert before == after
    cleanup = """import json,pathlib,shutil,sys
p=pathlib.Path(sys.argv[1]);assert p.parent==pathlib.Path('/home/ubuntu')
assert p.name.startswith('care-vision-classic-integration-')
shutil.rmtree(p);print(json.dumps({'workspace_removed':not p.exists()}))
"""
    deployment.update(
        complete=True,
        existing_states_unchanged=True,
        cleanup=json.loads(ssh.python(cleanup, remote)),
    )
    save(output / "deployment.json", deployment)
    print("Five runs complete; audit required", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    os.umask(0o077)
    run(parser.parse_args())
