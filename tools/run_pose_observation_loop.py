"""Real Edge recorded replay against opt-in cloud CLI; never starts a camera."""

import argparse
import gzip
import hashlib
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

from care_vision.aws_demo import READY, SSH


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


LAUNCH = """import json,os,pathlib,subprocess,sys
base=pathlib.Path(sys.argv[1]);run=base/sys.argv[2];run.mkdir(exist_ok=False)
env={**os.environ,'PYTHONNOUSERSITE':'1','PYTHONPATH':str(base/'src')+':/opt/cool/python_3.11/site-packages',
     'LD_LIBRARY_PATH':'/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib'}
python='/home/ubuntu/care-vision-phase3/releases/20261014-115106-d0341185/.venv/bin/python'
cmd=[python,'-u','-m','care_vision.cloud','--phase','3','--port','0',
     '--log',str(run/'cloud.jsonl'),'--run-seconds','100',
     '--person-model','models/person.onnx','--person-detector','yolox','--person-input-size','640',
     '--experimental-s-only-5fps','--experimental-weak-association',
     '--pose-shadow-config',str(base/sys.argv[3]),'--experimental-pose-observation']
with (run/'cloud.stderr').open('w') as log:
 p=subprocess.Popen(cmd,cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                    start_new_session=True)
(run/'pid').write_text(str(p.pid));print(json.dumps({'pid':p.pid,'argv':cmd}))
"""

STOP = """import os,pathlib,signal,sys,time
run=pathlib.Path(sys.argv[1]);pid=int((run/'pid').read_text());proc=pathlib.Path('/proc')/str(pid)/'cmdline'
if proc.exists() and proc.read_bytes():
 cmd=proc.read_bytes().split(b'\\0')
 if b'care_vision.cloud' not in cmd or str(run/'cloud.jsonl').encode() not in cmd:
  raise SystemExit('refusing to stop unowned PID')
 os.killpg(pid,signal.SIGINT)
 for _ in range(50):
  if not proc.exists() or not proc.read_bytes():break
  time.sleep(.1)
 else:raise SystemExit('owned worker not yet stopped; bounded lifetime remains')
print('owned worker stopped')
"""

READ_COMPRESSED = (
    "import gzip,pathlib,sys; "
    "sys.stdout.buffer.write(gzip.compress(pathlib.Path(sys.argv[1]).read_bytes()))"
)


def run(args):
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    sources = [
        ("sit-return-1", root / "artifacts/reference-recovery-live-20261014/sit-return-1"),
        ("sit-hold-1", root / (
            "artifacts/seated-pose-new-view-reserved-20261014/sit-hold-new-view-1"
        )),
    ]
    configs = []
    for name, _ in sources:
        config = json.loads((root / "artifacts/pose-shadow-integration-20261014/replay-config.json")
                            .read_text())
        config.update(model_path="models/pose.onnx", context_provenance=(
            "Manual context valid only for reviewed recorded " + name + "; no fresh camera view"
        ))
        path = output / ("config-" + name + ".json")
        save(path, config)
        configs.append(path)
    files = list((root / "src").rglob("*.py")) + [Path(__file__).resolve()]
    models = {
        "models/person.onnx": root / (
            "artifacts/phase5c-dataset-review-20260920/models/object_detection_yolox_2022nov.onnx"
        ),
        "models/pose.onnx": root / "artifacts/phase7d-pose-20260921/models/end2end.onnx",
    }
    plan = {
        "scope": "Two real Edge timestamp-paced recorded replays; new detector/pose on Graviton",
        "sources": [{"name": name, "folder": str(folder.relative_to(root))}
                    for name, folder in sources],
        "code_sha256": {str(p.relative_to(root)): sha(p) for p in files},
        "models": {n: sha(p) for n, p in models.items()},
        "source_sha256": {str(p.relative_to(root)): sha(p) for _, folder in sources
                          for p in (folder / "demo.avi", folder / "demo.captures.jsonl")},
        "configs": {p.name: sha(p) for p in configs},
        "new_camera_capture": False, "source_role": "previously analyzed development sources",
        "capture_scope": "original timestamps ~15 FPS; nominal config25 is not measured FPS",
        "acceptance": ["ROI command with matching applied ACK", "new ROI detector observations",
                       "bounded return command with applied ACK", "new valid full-frame evidence",
                       "no closure seated; return case closes only after reference posture",
                       "hold case ends unresolved", "no alert/loss/stale/session error"],
    }
    save(output / "plan.json", plan)
    with tarfile.open(output / "package.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        for name, path in models.items():
            archive.add(path, arcname=name, recursive=False)
        for path in configs:
            archive.add(path, arcname=path.name, recursive=False)
    remote = "/home/ubuntu/care-vision-pose-loop-" + uuid.uuid4().hex[:10]
    ssh = SSH(args.host, args.identity)
    deployment = {"workspace": remote, "package_sha256": sha(output / "package.tar.gz"),
                  "instance_left_running": True, "complete": False}
    save(output / "deployment.json", deployment)
    setup = """import io,os,pathlib,sys,tarfile
os.umask(0o077);p=pathlib.Path(sys.argv[1]);p.mkdir(exist_ok=False)
tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz').extractall(p,filter='data')
"""
    ssh.python(setup, remote, payload=(output / "package.tar.gz").read_bytes(), timeout=90)
    for (name, source), config in zip(sources, configs, strict=True):
        destination = output / name
        destination.mkdir(mode=0o700)
        remote_run = remote + "/" + name
        tunnel = None
        print("Starting " + name, flush=True)
        try:
            launched = json.loads(ssh.python(LAUNCH, remote, name, config.name))
            save(destination / "deployment.json", launched)
            ready = json.loads(ssh.python(READY, remote_run))
            save(destination / "runtime.json", ready)
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
                    raise RuntimeError("loop tunnel unavailable")
                with (destination / "edge.stderr").open("w") as edge_log:
                    subprocess.run([
                        sys.executable, "-m", "care_vision.edge", "--video",
                        str(source / "demo.avi"),
                        "--replay-captures", str(source / "demo.captures.jsonl"),
                        "--replay-sampling-phase-ms", "0", "--capture-fps", "25",
                        "--capture-resolution", "1280x720", "--baseline-resolution", "1280x720",
                        "--url", f"ws://127.0.0.1:{port}",
                        "--log", str(destination / "edge.jsonl"),
                    ], stdout=edge_log, stderr=subprocess.STDOUT, check=True, timeout=90)
        finally:
            if tunnel is not None:
                tunnel.terminate()
                try:
                    tunnel.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    tunnel.kill()
                    tunnel.wait()
            stopped = ssh.python(STOP, remote_run).decode()
            save(destination / "cleanup.json", {"confirmed_stopped": True, "message": stopped})
            for filename in ("cloud.jsonl", "cloud.stderr"):
                (destination / filename).write_bytes(
                    gzip.decompress(ssh.python(
                        READ_COMPRESSED, remote_run + "/" + filename, timeout=55
                    ))
                )
        print("Finished " + name, flush=True)
    cleanup = """import hashlib,json,pathlib,shutil,sys
p=pathlib.Path(sys.argv[1]);assert p.parent==pathlib.Path('/home/ubuntu')
assert p.name.startswith('care-vision-pose-loop-')
state=pathlib.Path('/home/ubuntu/care-vision-phase3/episode-state/continuity-20261014-a/episode.json')
result={'previous_episode_sha256':hashlib.sha256(state.read_bytes()).hexdigest(),
        'previous_episode_mode':oct(state.stat().st_mode&0o777)}
shutil.rmtree(p);result['workspace_removed']=not p.exists();print(json.dumps(result))
"""
    deployment.update(complete=True, cleanup=json.loads(ssh.python(cleanup, remote)))
    save(output / "deployment.json", deployment)
    print("Two recorded pose observation loops complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    run(args)
