"""Controlled paced socket comparison, followed by one real Edge recorded replay.

No new camera capture. Fixed-input trials acknowledge commands but deliberately
override frame selection; only the final Edge trial exercises adaptive sampling.
"""

import argparse
import asyncio
import hashlib
import json
import os
import select
import shlex
import socket
import subprocess
import sys
import tarfile
import time
import uuid
from functools import partial
from pathlib import Path

from care_vision.aws_demo import READ_FILE, READY, SSH
from care_vision.common.contracts import dumps, envelope, loads, pack_frame, uid, utc


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def cloud_worker(run, config, pose_enabled):
    from care_vision.cloud.pose_shadow import shadow_factory
    from care_vision.cloud.server import CloudServer
    from care_vision.cloud.yolox import YoloXPersonDetector
    from care_vision.common.metrics import EventLog
    from care_vision.common.resources import monitor_resources, snapshot

    class TrialLog(EventLog):
        start = None

        def emit(self, event, **fields):
            super().emit(event, **fields)
            if event == "frame_received" and self.start is None:
                self.start = snapshot()
                super().emit("measurement_started", **self.start)
            elif event == "session_finished" and self.start is not None:
                end = snapshot()
                seconds = end["mono"] - self.start["mono"]
                cpu = end["cpu_seconds"] - self.start["cpu_seconds"]
                super().emit(
                    "measurement_finished", wall_seconds=seconds, cpu_seconds=cpu,
                    cpu_percent_one_core=100 * cpu / seconds,
                    rss_bytes=end["rss_bytes"], peak_rss_bytes=end["peak_rss_bytes"],
                    scope="first frame_received through session finalization and shadow drain",
                )

    async def serve():
        service = CloudServer(
            log, person_detector_factory=partial(
                YoloXPersonDetector, "models/person.onnx", input_size=640,
                score_threshold=.2, nms_threshold=.3,
            ), experimental_s_only_5fps=True, experimental_weak_association=True,
            pose_shadow_factory=shadow_factory(config) if pose_enabled else None,
        )
        async with monitor_resources(log):
            await asyncio.wait_for(service.run(port=0, phase=3), 100)

    log = TrialLog("cloud", run / "cloud.jsonl")
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
    finally:
        log.close()


async def fixed_client(url, source, output):
    from websockets.asyncio.client import connect

    from care_vision.edge.commands import CommandHandler
    from care_vision.edge.representation import VisualPolicy
    from care_vision.edge.sampler import Sampler

    input_plan = json.loads((source / "plan.json").read_text())
    frames = input_plan["frames"]
    payloads = []
    for frame in frames:
        path = source / frame["filename"]
        if sha(path) != frame["jpeg_sha256"]:
            raise ValueError("fixed JPEG changed")
        payloads.append(path.read_bytes())
    identity = {"edge_id": "edge-1", "boot_id": uid(), "session_id": uid()}
    commands = CommandHandler(identity, Sampler(2), 25, visual_policy=VisualPolicy(
        1280, 720, 1280, 720,
    ))
    drained = asyncio.Event()
    records = []
    latest_seq, sent, payload_bytes = 0, 0, 0

    async def receive(ws):
        async for raw in ws:
            message = loads(raw)
            records.append({"event": "received", "edge_mono_ns": time.monotonic_ns(),
                            "message": message})
            if message["type"] in {"SET_TRANSMIT_FPS", "REQUEST_FULL_FRAME", "REQUEST_ROI"}:
                ack = commands.apply(message, time.monotonic_ns(), latest_seq)
                await ws.send(dumps(ack))
                records.append({"event": "ack_sent", "message": ack})
            elif message["type"] == "DRAINED":
                drained.set()

    async with connect(url, proxy=None, max_size=1_048_576, compression=None) as ws:
        await ws.send(dumps(envelope(
            "EDGE_STATUS", identity, capture_fps=14.653, transmit_fps=2,
            window_seconds=1, target_transmit_fps=2, captured=0, selected=0,
            sent=0, dropped=0, bytes_sent=0, policy_seq=0, connection="connected",
            cloud_status_version=1,
        )))
        receiver = asyncio.create_task(receive(ws))
        origin = time.monotonic_ns() + 1_000_000_000
        original = frames[0]["captured_mono_ns"]
        try:
            for frame, jpeg in zip(frames, payloads, strict=True):
                due = origin + frame["captured_mono_ns"] - original
                await asyncio.sleep(max(0, (due - time.monotonic_ns()) / 1e9))
                latest_seq = frame["capture_seq"]
                sampled = time.monotonic_ns()
                header = envelope(
                    "FRAME", identity, frame_id=f"{identity['boot_id']}:{latest_seq}",
                    capture_seq=latest_seq, captured_mono_ns=due, sampled_mono_ns=sampled,
                    captured_at_utc=utc(), encoding="jpeg", width=1280, height=720,
                    source_width=1280, source_height=720, view_kind="full_frame",
                    payload_bytes=len(jpeg), policy_seq=commands.policy_seq,
                    applied_command_id=commands.command_id,
                    target_transmit_fps=commands.sampler.fps, evidence_kind="live",
                )
                await ws.send(pack_frame(header, jpeg))
                sent += 1
                payload_bytes += len(jpeg)
                records.append({
                    "event": "frame_sent", "capture_seq": latest_seq,
                    "jpeg_sha256": frame["jpeg_sha256"], "payload_bytes": len(jpeg),
                    "captured_mono_ns": due, "send_lateness_ms": (sampled - due) / 1e6,
                    "policy_seq": commands.policy_seq,
                })
            await ws.send(dumps(envelope("DRAIN_REQUEST", identity, drain_id=uid(),
                                         live_frames_sent=sent)))
            await asyncio.wait_for(drained.wait(), 12)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            (output / "client.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    save(output / "client-report.json", {
        "frames_sent": sent, "jpeg_payload_bytes": payload_bytes, "drained": drained.is_set(),
        "scope": "Fixed JPEG schedule; real CommandHandler ACK, sampler deliberately bypassed",
    })


LAUNCH = """
import os,pathlib,subprocess,sys,json
base=pathlib.Path(sys.argv[1]);run=base/sys.argv[2];run.mkdir(exist_ok=False)
env={**os.environ,'PYTHONNOUSERSITE':'1','PYTHONPATH':str(base/'src')+':/opt/cool/python_3.11/site-packages',
     'LD_LIBRARY_PATH':'/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib'}
python='/home/ubuntu/care-vision-phase3/releases/20261014-115106-d0341185/.venv/bin/python'
cmd=[python,'-u',str(base/'tools/measure_pose_shadow_stream.py'),'--cloud-worker','--run',str(run),
     '--config',str(base/'config.json')]
if sys.argv[3]=='1':cmd.append('--pose')
with (run/'cloud.stderr').open('w') as f:
 p=subprocess.Popen(cmd,cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
(run/'pid').write_text(str(p.pid));print(json.dumps({'run':str(run),'pid':p.pid}))
"""

STOP = """
import os,pathlib,sys,signal,time
run=pathlib.Path(sys.argv[1]);pid=int((run/'pid').read_text());proc=pathlib.Path('/proc')/str(pid)/'cmdline'
if proc.exists() and proc.read_bytes():
 cmd=proc.read_bytes().split(b'\\0')
 if str(run).encode() not in cmd or b'--cloud-worker' not in cmd:
  raise SystemExit('PID no longer belongs to measurement worker')
 os.killpg(pid,signal.SIGINT)
 for _ in range(50):
  if not proc.exists() or not proc.read_bytes():break
  time.sleep(.1)
 else:raise SystemExit('measurement worker still running; bounded lifetime remains')
print('measurement worker stopped')
"""


def measure(args):
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    source = root / "artifacts/seated-pose-new-view-20261014/inputs"
    dense = root / "artifacts/reference-recovery-live-20261014/sit-return-1"
    config = json.loads((root / "artifacts/pose-shadow-integration-20261014/replay-config.json")
                        .read_text())
    config.update(model_path="models/pose.onnx", context_provenance=(
        "Manual context current only for recorded sit-return-1; no new live camera view"
    ))
    schedule = [("off-1", False, False), ("on-1", True, False),
                ("on-2", True, False), ("off-2", False, False), ("edge-on-1", True, True)]
    files = list((root / "src/care_vision").rglob("*.py")) + [Path(__file__).resolve()]
    model_paths = {"models/person.onnx": root / (
        "artifacts/phase5c-dataset-review-20260920/models/object_detection_yolox_2022nov.onnx"
    ), "models/pose.onnx": root / "artifacts/phase7d-pose-20260921/models/end2end.onnx"}
    plan = {
        "scope": "ABBA fixed input socket comparison and separate adaptive Edge recorded pilot",
        "trials": [{"name": n, "pose": p, "real_edge": e} for n, p, e in schedule],
        "code_sha256": {str(p.relative_to(root)): sha(p) for p in files},
        "models": {n: sha(p) for n, p in model_paths.items()},
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in [
            source / "plan.json", dense / "demo.avi", dense / "demo.captures.jsonl",
        ]}, "threads": 8, "detector": "YOLOX-S640, strong0.35/association0.2/NMS0.3",
        "capture_scope": "recorded ~15 FPS; timestamp-paced, no new camera capture",
        "thresholds_changed": False, "source_role": "previously analyzed development source",
    }
    save(output / "plan.json", plan)
    save(output / "config.json", config)
    with tarfile.open(output / "package.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        for name, path in model_paths.items():
            archive.add(path, arcname=name, recursive=False)
        archive.add(output / "config.json", arcname="config.json", recursive=False)
    remote = "/home/ubuntu/care-vision-pose-stream-" + uuid.uuid4().hex[:10]
    ssh = SSH(args.host, args.identity)
    deployment = {"workspace": remote, "package_sha256": sha(output / "package.tar.gz"),
                  "complete": False, "instance_left_running": True}
    save(output / "deployment.json", deployment)
    setup = """import os,pathlib,sys,tarfile,io
os.umask(0o077);p=pathlib.Path(sys.argv[1]);p.mkdir(exist_ok=False)
tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz').extractall(p,filter='data')
"""
    ssh.python(setup, remote, payload=(output / "package.tar.gz").read_bytes(), timeout=90)
    all_stopped = True
    for name, enabled, real_edge in schedule:
        destination = output / name
        destination.mkdir(mode=0o700)
        remote_run = remote + "/" + name
        tunnel = None
        print("Starting " + name, flush=True)
        try:
            launched = json.loads(ssh.python(LAUNCH, remote, name, "1" if enabled else "0"))
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
                    raise RuntimeError("measurement tunnel unavailable")
                url = f"ws://127.0.0.1:{port}"
                if real_edge:
                    with (destination / "edge.stderr").open("w") as edge_error:
                        subprocess.run([
                            sys.executable, "-m", "care_vision.edge", "--video",
                            str(dense / "demo.avi"),
                            "--replay-captures", str(dense / "demo.captures.jsonl"),
                            "--replay-sampling-phase-ms", "0", "--capture-fps", "25",
                            "--capture-resolution", "1280x720", "--baseline-resolution", "1280x720",
                            "--url", url, "--log", str(destination / "edge.jsonl"),
                        ], stdout=edge_error, stderr=subprocess.STDOUT, check=True, timeout=90)
                else:
                    asyncio.run(fixed_client(url, source, destination))
        finally:
            if tunnel is not None:
                tunnel.terminate()
                try:
                    tunnel.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    tunnel.kill()
                    tunnel.wait()
            try:
                stopped = ssh.python(STOP, remote_run).decode()
                save(destination / "cleanup.json", {"confirmed_stopped": True, "message": stopped})
            except Exception as error:
                all_stopped = False
                save(destination / "cleanup.json", {"confirmed_stopped": False,
                                                     "error_type": type(error).__name__})
                raise
            for filename in ["cloud.jsonl", "cloud.stderr"]:
                (destination / filename).write_bytes(
                    ssh.python(READ_FILE, remote_run + "/" + filename)
                )
        print("Finished " + name, flush=True)
        save(output / "progress.json", {"last_completed": name})
    if all_stopped:
        cleanup = """import pathlib,shutil,sys
p=pathlib.Path(sys.argv[1])
assert p.parent==pathlib.Path('/home/ubuntu') and p.name.startswith('care-vision-pose-stream-')
shutil.rmtree(p);assert not p.exists()
"""
        ssh.python(cleanup, remote)
        deployment.update(complete=True, workspace_removed=True)
        save(output / "deployment.json", deployment)
    print("All five recorded streaming trials complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-worker", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--host")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.cloud_worker:
        cloud_worker(args.run, args.config, args.pose)
    else:
        if not all([args.host, args.identity, args.output]):
            parser.error("host, identity and new output are required")
        measure(args)


if __name__ == "__main__":
    main()
