"""Deploy an isolated phase-2/3 release and verify the real Mac-to-Graviton loop.

Requires an existing instance and already verified SSH host key. Never creates
AWS resources, uploads private keys, or changes the system OpenCV installation.
"""

import argparse
import getpass
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import tarfile
import time
from contextlib import nullcontext
from pathlib import Path

from care_vision.check_loop import CheckFailed, check_records, read_log
from care_vision.common.contracts import uid
from care_vision.fixture import make_video
from care_vision.netem_probe import validate_loss
from care_vision.tcp_telemetry import mac_tunnel_telemetry
from care_vision.transport_probe import TransportProbe, probe_process

UV_VERSION = "0.11.32"
READ_FILE = "import pathlib,sys; sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())"


def source_archive(root: Path, extra_files=()) -> bytes:
    files = [root / name for name in ("pyproject.toml", "uv.lock", ".python-version")]
    files += sorted((root / "src" / "care_vision").rglob("*.py"))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in files:
            if path.is_symlink():
                raise ValueError(f"refusing to deploy symlink: {path}")
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        for path, archive_name in extra_files:
            path = Path(path)
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"invalid extra deployment file: {path}")
            archive.add(path, arcname=archive_name, recursive=False)
    return buffer.getvalue()


def prepare_video(destination: Path, source_video: Path | None = None) -> Path:
    if source_video is None:
        return make_video(destination)
    if not source_video.is_file():
        raise ValueError("source video does not exist")
    shutil.copyfile(source_video, destination)
    return destination


def complete_five_second_rates(records):
    """Measured event rates after the first second, without partial windows."""
    times = [row["local_mono_ns"] / 1e9 for row in records]
    if not times:
        return []
    start = times[0] + 1
    rates = []
    while start + 5 <= times[-1]:
        rates.append(round(sum(start <= at < start + 5 for at in times) / 5, 2))
        start += 5
    return rates


class SSH:
    def __init__(self, host: str, identity: Path):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host):
            raise ValueError("expected an SSH alias or user@hostname")
        if not identity.is_file():
            raise ValueError("SSH private key file does not exist")
        self.host = host
        self.args = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
            "-i",
            str(identity.resolve()),
        ]

    def python(self, script: str, *args, payload=None, timeout=30):
        command = shlex.join(["python3", "-c", script, *map(str, args)])
        return subprocess.run(
            [*self.args, self.host, command],
            input=payload,
            stdout=subprocess.PIPE,
            check=True,
            timeout=timeout,
        ).stdout


PREPARE = """
import pathlib, platform, subprocess, sys
if platform.system() != 'Linux' or platform.machine() not in {'aarch64', 'arm64'}:
    raise SystemExit('Remote phases require Linux ARM64')
phase = int(sys.argv[2])
if phase not in {2, 3}:
    raise SystemExit('unsupported remote phase')
if phase == 3 and not pathlib.Path('/opt/cool/python_3.11/site-packages/cv2').is_dir():
    raise SystemExit('Phase 3 requires the preinstalled COOL SDK for Python 3.11')
base = pathlib.Path.home() / ('care-vision-phase' + str(phase))
bootstrap = base / 'bootstrap'
base.mkdir(exist_ok=True)
if not (bootstrap / 'bin/python').exists():
    subprocess.run([sys.executable, '-m', 'venv', str(bootstrap)], check=True,
                   stdout=sys.stderr)
subprocess.run([str(bootstrap / 'bin/python'), '-m', 'pip', 'install',
                'uv==' + sys.argv[1]], check=True, stdout=sys.stderr)
print(base)
"""

INSTALL = """
import io, pathlib, subprocess, sys, tarfile
base, name, phase = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
release = base / 'releases' / name
release.mkdir(parents=True, exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()), mode='r:gz') as archive:
    for member in archive.getmembers():
        path = pathlib.PurePosixPath(member.name)
        if not member.isfile() or path.is_absolute() or '..' in path.parts:
            raise SystemExit('invalid source archive')
        target = release / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.extractfile(member).read())
groups = ['--no-default-groups'] if phase == 3 else ['--no-dev']
subprocess.run([str(base / 'bootstrap/bin/uv'), 'sync', '--locked', *groups,
                '--python', '/usr/bin/python3.11'], cwd=release, check=True, stdout=sys.stderr)
print(release)
"""

START = """
import json, os, pathlib, subprocess, sys
release = pathlib.Path(sys.argv[1])
phase = int(sys.argv[2])
person_model = sys.argv[3]
person_detector = sys.argv[4]
person_threshold = sys.argv[5]
person_nms_threshold = sys.argv[6]
person_input_size = sys.argv[7]
verification_model = sys.argv[8]
experimental_loss_5fps = sys.argv[9] == '1'
experimental_s_only_5fps = sys.argv[10] == '1'
experimental_alert_after_seconds = sys.argv[11] if len(sys.argv) > 11 else ''
experimental_weak_association = len(sys.argv) > 12 and sys.argv[12] == '1'
episode_continuity = len(sys.argv) > 13 and sys.argv[13] == '1'
episode_state_file = sys.argv[14] if len(sys.argv) > 14 else ''
disconnect_seconds = sys.argv[15] if len(sys.argv) > 15 else ''
environment = os.environ.copy()
environment['PYTHONNOUSERSITE'] = '1'
if phase == 3:
    environment['PYTHONPATH'] = '/opt/cool/python_3.11/site-packages'
    environment['LD_LIBRARY_PATH'] = '/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib'
run = release / 'run'
run.mkdir(exist_ok=False)
command = [
        str(release / '.venv/bin/python'), '-m', 'care_vision.cloud',
        '--port', '0', '--phase', str(phase), '--run-seconds', '300',
        '--log', str(run / 'cloud.jsonl')]
if person_model:
    command.extend([
        '--person-model', person_model,
        '--person-detector', person_detector,
        '--person-threshold', person_threshold,
        '--person-nms-threshold', person_nms_threshold,
    ])
    if person_input_size:
        command.extend(['--person-input-size', person_input_size])
if experimental_loss_5fps:
    command.extend(['--experimental-loss-5fps', '--verification-model', verification_model])
if experimental_s_only_5fps:
    command.append('--experimental-s-only-5fps')
if experimental_weak_association:
    command.append('--experimental-weak-association')
if experimental_alert_after_seconds:
    command.extend(['--experimental-alert-after-seconds', experimental_alert_after_seconds])
if episode_continuity:
    command.append('--experimental-episode-continuity')
if episode_state_file:
    command.extend(['--episode-state-file', episode_state_file])
if disconnect_seconds:
    command.extend(['--diagnostic-disconnect-after-episode-seconds', disconnect_seconds])
with (run / 'cloud.stderr').open('w') as error:
    process = subprocess.Popen(command, cwd=release,
        stdout=error, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True, env=environment)
(run / 'pid').write_text(str(process.pid))
print(json.dumps({'pid': process.pid, 'run': str(run)}))
"""

READY = """
import json, pathlib, sys, time
run = pathlib.Path(sys.argv[1])
for _ in range(100):
    log = run / 'cloud.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('event') == 'server_ready':
                print(json.dumps(event))
                raise SystemExit(0)
    time.sleep(0.1)
print((run / 'cloud.stderr').read_text(), file=sys.stderr)
raise SystemExit('cloud readiness timeout')
"""

TUNNEL_PROBE = """
import json, os, pathlib, subprocess, sys, time
run = pathlib.Path(sys.argv[1])
client, client_port, server, server_port = os.environ['SSH_CONNECTION'].split()
selector = (f'( src = {server} and dst = {client} '
            f'and sport = :{server_port} and dport = :{client_port} )')
deadline = time.monotonic() + 300
with (run / 'tcp-socket.jsonl').open('w') as log:
    print('TUNNEL_READY', flush=True)
    while time.monotonic() < deadline and not (run / 'tcp-probe.stop').exists():
        started = time.monotonic_ns()
        try:
            result = subprocess.run(['ss', '-tinH', selector], capture_output=True,
                                    text=True, timeout=2)
            record = {'event': 'tcp_sample', 'local_mono_ns': started,
                      'socket': result.stdout, 'stderr': result.stderr,
                      'returncode': result.returncode}
        except (OSError, subprocess.TimeoutExpired) as exc:
            record = {'event': 'tcp_sample_error', 'local_mono_ns': started,
                      'error_type': type(exc).__name__, 'reason': str(exc)}
        record['sample_duration_ms'] = (time.monotonic_ns() - started) / 1e6
        log.write(json.dumps(record) + '\\n')
        log.flush()
        time.sleep(.25)
"""

STOP = """
import os, pathlib, signal, sys, time
run = pathlib.Path(sys.argv[1])
(run / 'tcp-probe.stop').touch()
pid = int((run / 'pid').read_text())
proc = pathlib.Path('/proc') / str(pid) / 'cmdline'
if proc.exists():
    command = proc.read_bytes().split(b'\\0')
    if str(run / 'cloud.jsonl').encode() not in command or b'care_vision.cloud' not in command:
        raise SystemExit('PID no longer belongs to this demonstration')
    os.killpg(pid, signal.SIGINT)
    for _ in range(30):
        if not proc.exists() or not proc.read_bytes():
            break
        time.sleep(0.1)
    else:
        raise SystemExit('cloud did not stop; bounded lifetime remains active')
print('cloud stopped')
"""

START_NETEM = """
import json, pathlib, subprocess, sys, time
run = pathlib.Path(sys.argv[1])
with (run / 'netem.stderr').open('w') as error:
    process = subprocess.Popen(['sudo', '-n', 'python3',
        str(run.parent / 'src/care_vision/netem_probe.py'), 'supervise',
        str(run), sys.argv[2], sys.argv[3]], stdin=subprocess.DEVNULL,
        stdout=error, stderr=error, start_new_session=True)
(run / 'netem.pid').write_text(str(process.pid))
for _ in range(150):
    ready = run / 'netem-ready.json'
    if ready.exists():
        print(ready.read_text())
        raise SystemExit(0)
    if process.poll() is not None:
        break
    time.sleep(.1)
raise SystemExit((run / 'netem.stderr').read_text() or 'netem readiness timeout')
"""

STOP_NETEM = """
import pathlib, sys, time
run = pathlib.Path(sys.argv[1])
(run / 'netem.stop').touch()
for _ in range(100):
    if (run / 'netem-done').exists():
        print('netem cleanup completed')
        raise SystemExit(0)
    time.sleep(.1)
raise SystemExit('netem cleanup not confirmed; inspect retained logs')
"""


def run_aws_demo(
    host: str,
    identity: Path,
    output: Path,
    root: Path,
    phase=2,
    person_model: Path | None = None,
    person_detector="nanodet",
    person_threshold=0.35,
    person_nms_threshold=0.3,
    person_input_size: int | None = None,
    source_video: Path | None = None,
    capture_resolution="640x360",
    baseline_resolution="640x360",
    transport_pause=None,
    packet_loss=None,
    camera_index=None,
    camera_seconds=45,
    tcp_telemetry=False,
    rtsp_ip=None,
    rtsp_stream=1,
    rtsp_username=None,
    rtsp_password=None,
    rtsp_ffmpeg_nobuffer=False,
    record_rtsp=False,
    capture_fps=30,
    initial_transmit_fps=2,
    source_captures: Path | None = None,
    replay_sampling_phase_ms: int | None = None,
    verification_model: Path | None = None,
    experimental_loss_5fps=False,
    experimental_s_only_5fps=False,
    experimental_alert_after_seconds=None,
    experimental_weak_association=False,
    experimental_episode_continuity=False,
    episode_state_key=None,
    diagnostic_disconnect_after_episode_seconds=None,
):
    if experimental_weak_association and (not experimental_s_only_5fps or person_threshold != 0.35):
        raise ValueError("weak association requires S-only mode and strong threshold 0.35")
    if experimental_alert_after_seconds is not None or experimental_episode_continuity:
        from care_vision.cloud.controller import Policy

        Policy(
            experimental_s_only_5fps=experimental_s_only_5fps,
            experimental_alert_after_seconds=experimental_alert_after_seconds,
            experimental_episode_continuity=experimental_episode_continuity,
        )
    if episode_state_key is not None and (
        not experimental_episode_continuity
        or not re.fullmatch(r"[a-z][a-z0-9-]{1,63}", episode_state_key)
    ):
        raise ValueError("episode state key requires continuity and a lowercase experiment name")
    if diagnostic_disconnect_after_episode_seconds is not None:
        value = diagnostic_disconnect_after_episode_seconds
        if not experimental_episode_continuity or not math.isfinite(value) or value <= 0:
            raise ValueError("diagnostic disconnect requires continuity and a positive time")
    if experimental_loss_5fps and experimental_s_only_5fps:
        raise ValueError("experimental detector modes are mutually exclusive")
    if experimental_loss_5fps:
        if (
            person_model is None
            or person_detector != "yolox"
            or person_input_size != 416
            or verification_model is None
        ):
            raise ValueError("loss 5 FPS mode requires Nano416 and a YOLOX-S verification model")
        if capture_resolution != "1280x720" or baseline_resolution != "320x180":
            raise ValueError("loss 5 FPS mode requires 1280x720 capture and 320x180 baseline")
    elif verification_model is not None:
        raise ValueError("verification model requires experimental loss 5 FPS mode")
    if experimental_s_only_5fps:
        if person_model is None or person_detector != "yolox" or person_input_size != 640:
            raise ValueError("S-only mode requires a YOLOX-S person model at input size 640")
        if capture_resolution != "1280x720" or baseline_resolution != "1280x720":
            raise ValueError("S-only mode requires 1280x720 capture and baseline")
        if initial_transmit_fps != 2:
            raise ValueError("S-only mode requires 2 FPS initial transmission")
    if source_captures is not None:
        if source_video is None or camera_index is not None or rtsp_ip is not None:
            raise ValueError("source captures require only source-video input")
        if not source_captures.is_file():
            raise ValueError("source capture timestamps do not exist")
    if replay_sampling_phase_ms is not None and (
        source_captures is None
        or initial_transmit_fps != 2
        or type(replay_sampling_phase_ms) is not int
        or not 0 <= replay_sampling_phase_ms < 500
    ):
        raise ValueError("sampling phase requires recorded 2 FPS input and 0-499 ms")
    if initial_transmit_fps not in (2, 20):
        raise ValueError("initial transmit FPS must be 2 or 20")
    if initial_transmit_fps != 2 and rtsp_ip is None:
        raise ValueError("fixed 20 FPS diagnostic currently requires RTSP input")
    if record_rtsp and rtsp_ip is None:
        raise ValueError("record-rtsp requires RTSP input")
    if rtsp_ffmpeg_nobuffer and rtsp_ip is None:
        raise ValueError("RTSP FFmpeg no-buffer option requires RTSP input")
    if rtsp_ip is not None:
        if source_video is not None or camera_index is not None:
            raise ValueError("RTSP, camera, and source video are mutually exclusive")
        if not rtsp_username or not rtsp_password:
            raise ValueError("RTSP camera account credentials are required")
        if rtsp_stream not in (1, 2):
            raise ValueError("RTSP stream must be 1 or 2")
    if camera_index is not None:
        if source_video is not None:
            raise ValueError("camera and source video are mutually exclusive")
        if type(camera_index) is not int or not 0 <= camera_index <= 16:
            raise ValueError("invalid camera index")
        if not 1 <= camera_seconds <= 120:
            raise ValueError("camera duration must be between 1 and 120 seconds")
    if rtsp_ip is not None and not 1 <= camera_seconds <= 120:
        raise ValueError("RTSP duration must be between 1 and 120 seconds")
    if not 0 < capture_fps <= 30:
        raise ValueError("capture FPS must be in (0, 30]")
    if packet_loss is not None:
        validate_loss(packet_loss)
        if phase != 3:
            raise ValueError("isolated packet-loss diagnostic requires phase 3")
    if transport_pause is not None:
        TransportProbe(1, None, *transport_pause)  # Validate before deployment.
    output.mkdir(parents=True, exist_ok=False)
    ssh = SSH(host, identity)
    video = (
        None
        if rtsp_ip is not None and not record_rtsp
        else output / "demo.avi"
        if camera_index is not None or rtsp_ip is not None
        else prepare_video(output / "demo.avi", source_video)
    )
    print("Preparando entorno aislado en Graviton…", flush=True)
    base = ssh.python(PREPARE, UV_VERSION, phase, timeout=120).decode().strip()
    episode_state_remote = (
        str(Path(base) / "episode-state" / episode_state_key / "episode.json")
        if episode_state_key else ""
    )
    release_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uid()[:8]
    model_hash = None
    extra_files = []
    remote_person_model = ""
    remote_verification_model = ""
    verification_hash = None
    if person_model:
        if not person_model.is_file():
            raise ValueError("person model does not exist")
        if person_detector not in {"nanodet", "yolox"}:
            raise ValueError("unsupported person detector")
        if not 0 < person_threshold <= 1 or not 0 < person_nms_threshold <= 1:
            raise ValueError("person thresholds must be in (0, 1]")
        if person_input_size is not None and person_detector != "yolox":
            raise ValueError("person input size is only supported by yolox")
        if person_input_size is not None and (
            type(person_input_size) is not int or person_input_size <= 0 or person_input_size % 32
        ):
            raise ValueError("person input size must be a positive multiple of 32")
        model_hash = hashlib.sha256(person_model.read_bytes()).hexdigest()
        remote_person_model = "models/person-detector.onnx"
        extra_files.append((person_model, remote_person_model))
    if verification_model is not None:
        if not verification_model.is_file():
            raise ValueError("verification model does not exist")
        verification_hash = hashlib.sha256(verification_model.read_bytes()).hexdigest()
        remote_verification_model = "models/verification-detector.onnx"
        extra_files.append((verification_model, remote_verification_model))
    archive = source_archive(root, extra_files)
    (output / "source.tar.gz").write_bytes(archive)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
        manifest = {
            member.name: hashlib.sha256(package.extractfile(member).read()).hexdigest()
            for member in package.getmembers()
        }
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    release = (
        ssh.python(
            INSTALL,
            base,
            release_id,
            phase,
            payload=archive,
            timeout=180,
        )
        .decode()
        .strip()
    )
    (output / "deployment.json").write_text(
        json.dumps(
            {
                "host": host,
                "release": release,
                "uv_version": UV_VERSION,
                "transport": "SSH local forward",
                "packet_loss_percent": packet_loss,
                "transport_probe": (
                    {
                        "pause_at_seconds": transport_pause[0],
                        "pause_seconds": transport_pause[1],
                        "direction": "Edge to AWS",
                        "scope": "Loopback TCP relay; not IP packet loss",
                    }
                    if transport_pause is not None
                    else None
                ),
                "requested_phase": phase,
                "source_archive_sha256": hashlib.sha256(archive).hexdigest(),
                "source_manifest_sha256": hashlib.sha256(
                    (output / "source-manifest.json").read_bytes()
                ).hexdigest(),
                "source_kind": (
                    "rtsp_camera"
                    if rtsp_ip is not None
                    else "physical_camera"
                    if camera_index is not None
                    else "video_file"
                ),
                "tcp_telemetry": tcp_telemetry,
                "camera_index": camera_index,
                "rtsp_ip": rtsp_ip,
                "rtsp_stream": rtsp_stream if rtsp_ip is not None else None,
                "rtsp_ffmpeg_nobuffer": rtsp_ffmpeg_nobuffer if rtsp_ip is not None else None,
                "record_rtsp": record_rtsp,
                "camera_seconds": (
                    camera_seconds if camera_index is not None or rtsp_ip is not None else None
                ),
                "capture_fps_target": capture_fps,
                "initial_transmit_fps": initial_transmit_fps,
                "recorded_timestamp_replay": source_captures is not None,
                "replay_sampling_phase_ms": replay_sampling_phase_ms,
                "source_captures_sha256": (
                    hashlib.sha256(source_captures.read_bytes()).hexdigest()
                    if source_captures
                    else None
                ),
                "input_video_sha256": (
                    hashlib.sha256(video.read_bytes()).hexdigest()
                    if video and video.exists()
                    else None
                ),
                "capture_resolution": capture_resolution,
                "baseline_resolution": baseline_resolution,
                "person_model_sha256": model_hash,
                "verification_model_sha256": verification_hash,
                "experimental_loss_5fps": experimental_loss_5fps,
                "experimental_s_only_5fps": experimental_s_only_5fps,
                "experimental_weak_association": experimental_weak_association,
                "experimental_alert_after_seconds": experimental_alert_after_seconds,
                "experimental_episode_continuity": experimental_episode_continuity,
                "episode_state_remote": episode_state_remote or None,
                "diagnostic_disconnect_after_episode_seconds": (
                    diagnostic_disconnect_after_episode_seconds
                ),
                "person_detector": person_detector if person_model else None,
                "person_threshold": person_threshold if person_model else None,
                "person_nms_threshold": person_nms_threshold if person_model else None,
                "person_input_size": person_input_size if person_model else None,
                "verification": "pending; see runtime evidence and final report",
            },
            indent=2,
        )
        + "\n"
    )
    started = json.loads(
        ssh.python(
            START,
            release,
            phase,
            remote_person_model,
            person_detector,
            person_threshold,
            person_nms_threshold,
            person_input_size or "",
            remote_verification_model,
            "1" if experimental_loss_5fps else "0",
            "1" if experimental_s_only_5fps else "0",
            str(experimental_alert_after_seconds)
            if experimental_alert_after_seconds is not None
            else "",
            "1" if experimental_weak_association else "0",
            "1" if experimental_episode_continuity else "0",
            episode_state_remote,
            str(diagnostic_disconnect_after_episode_seconds)
            if diagnostic_disconnect_after_episode_seconds is not None else "",
        )
    )
    remote_run = started["run"]
    tunnel = None
    netem_started = False
    tunnel_log = (output / "ssh.stderr").open("w")
    try:
        ready = json.loads(ssh.python(READY, remote_run))
        (output / "cloud-runtime.json").write_text(json.dumps(ready, indent=2) + "\n")
        endpoint = {"host": "127.0.0.1", "port": ready["port"]}
        if packet_loss is not None:
            netem_started = True
            endpoint = json.loads(ssh.python(START_NETEM, remote_run, ready["port"], packet_loss))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        # The remote marker is emitted after SSH has established local forwarding.
        # No TCP probe is sent to the application (it would look like a bad session).
        marker = "import time; print('TUNNEL_READY', flush=True); time.sleep(300)"
        marker_command = ["python3", "-c", marker]
        if transport_pause is not None or tcp_telemetry:
            marker_command = ["python3", "-c", TUNNEL_PROBE, remote_run]
        tunnel = subprocess.Popen(
            [
                *ssh.args,
                "-o",
                "ExitOnForwardFailure=yes",
                "-L",
                f"127.0.0.1:{port}:{endpoint['host']}:{endpoint['port']}",
                ssh.host,
                shlex.join(marker_command),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=tunnel_log,
        )
        import select

        readable, _, _ = select.select([tunnel.stdout], [], [], 20)
        if not readable or tunnel.stdout.readline().strip() != b"TUNNEL_READY":
            raise RuntimeError("SSH tunnel did not become ready; inspect ssh.stderr")
        source_seconds = camera_seconds if camera_index is not None or rtsp_ip is not None else 34
        if source_captures:
            rows = read_log(source_captures)
            source_seconds = (rows[-1]["captured_mono_ns"] - rows[0]["captured_mono_ns"]) / 1e9
        print(
            f"Cloud en {ready['runtime']['ec2']['instanceType']}; "
            f"iniciando edge local ({source_seconds} s)…",
            flush=True,
        )
        if camera_index is not None or rtsp_ip is not None:
            print("CAMERA_START_IN_5_SECONDS", flush=True)
            time.sleep(5)
        probe = (
            probe_process(port, output, *transport_pause)
            if transport_pause is not None
            else nullcontext(port)
        )
        telemetry = mac_tunnel_telemetry(tunnel.pid, output) if tcp_telemetry else nullcontext()
        with telemetry, probe as edge_port, (output / "edge.stderr").open("w") as error:
            if camera_index is not None:
                source_args = [
                    "--camera",
                    str(camera_index),
                    "--camera-seconds",
                    str(camera_seconds),
                    "--record-camera",
                    str(video),
                ]
            elif rtsp_ip is not None:
                source_args = [
                    "--rtsp-ip",
                    rtsp_ip,
                    "--rtsp-stream",
                    str(rtsp_stream),
                    "--camera-seconds",
                    str(camera_seconds),
                ]
                if record_rtsp:
                    source_args.extend(["--record-camera", str(video)])
                if rtsp_ffmpeg_nobuffer:
                    source_args.append("--rtsp-ffmpeg-nobuffer")
            else:
                source_args = ["--video", str(video)]
                if source_captures:
                    source_args.extend(["--replay-captures", str(source_captures.resolve())])
                    if replay_sampling_phase_ms is not None:
                        source_args.extend(
                            ["--replay-sampling-phase-ms", str(replay_sampling_phase_ms)]
                        )
            edge_env = os.environ.copy()
            if rtsp_ip is not None:
                edge_env["CARE_VISION_RTSP_USERNAME"] = rtsp_username
                edge_env["CARE_VISION_RTSP_PASSWORD"] = rtsp_password
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "care_vision.edge",
                    *source_args,
                    "--url",
                    f"ws://127.0.0.1:{edge_port}",
                    "--log",
                    str(output / "edge.jsonl"),
                    "--capture-resolution",
                    capture_resolution,
                    "--capture-fps",
                    str(capture_fps),
                    "--initial-transmit-fps",
                    str(initial_transmit_fps),
                    "--baseline-resolution",
                    baseline_resolution,
                ],
                env=edge_env,
                stdout=error,
                stderr=subprocess.STDOUT,
                timeout=240,
                check=True,
            )
        # Stop only this run's process, allowing it to flush the final session record.
        ssh.python(STOP, remote_run)
        for name in ("cloud.jsonl", "cloud.stderr"):
            data = ssh.python(
                READ_FILE,
                remote_run + "/" + name,
            )
            (output / name).write_bytes(data)
        cloud_records = read_log(output / "cloud.jsonl")
        if transport_pause is not None or tcp_telemetry:
            (output / "tcp-socket.jsonl").write_bytes(
                ssh.python(READ_FILE, remote_run + "/tcp-socket.jsonl")
            )
        edge_records = read_log(output / "edge.jsonl")
        try:
            report = check_records(
                edge_records,
                cloud_records,
                escalation_fps=5 if experimental_s_only_5fps else 20,
                capture_fps=capture_fps,
            )
        except CheckFailed as exc:
            if rtsp_ip is None and source_captures is None:
                raise
            statuses = [
                row["capture_fps"]
                for row in edge_records
                if row["event"] == "edge_status" and row.get("capture_fps", 0) > 0
            ]
            ready_records = [row for row in cloud_records if row["event"] == "server_ready"]
            report = {
                "passed": False,
                "source_kind": "recorded_timestamp_replay" if source_captures else "rtsp_camera",
                "strict_demo_verifier_failure": str(exc),
                "capture_fps_median": (round(statistics.median(statuses), 3) if statuses else None),
                "frames_captured": max(
                    (
                        row.get("captured", 0)
                        for row in edge_records
                        if row["event"] == "edge_status"
                    ),
                    default=0,
                ),
                "frames_sent": sum(row["event"] == "frame_sent" for row in edge_records),
                "frames_received": sum(row["event"] == "frame_received" for row in cloud_records),
                "observations": sum(row["event"] == "observation" for row in cloud_records),
                "agent_actions": [
                    {"type": row.get("type"), "transmit_fps": row.get("transmit_fps")}
                    for row in cloud_records
                    if row["event"] == "agent_action"
                ],
                "cool_verified_at_startup": bool(
                    ready_records and ready_records[0].get("cool_verified")
                ),
                "scope": (
                    "Experimental episode continuity with new sessions/observations; strict "
                    "single-loop scripted verifier is not the experiment acceptance criterion"
                    if experimental_episode_continuity
                    else "Adaptive recorded replay with experimental loss 5 FPS controller; "
                    "strict scripted 2-20-2 verifier is not applicable"
                    if experimental_loss_5fps and source_captures
                    else "Adaptive recorded replay with unchanged cloud controller; strict "
                    "scripted verifier is not a natural-scene acceptance criterion"
                    if source_captures
                    else "Fixed 20 FPS RTSP transport/load diagnostic; strict adaptive "
                    "demo verifier is not applicable"
                    if initial_transmit_fps == 20
                    else "Live RTSP adaptive diagnostic; strict 30 FPS scripted demo was not passed"
                ),
            }
        if person_model:
            observations = [row for row in cloud_records if row["event"] == "observation"]
            statuses = {row.get("person_detection_status") for row in observations}
            backends = {row.get("person_detector_backend") for row in observations}
            tracking_statuses = {row.get("person_tracking_status") for row in observations}
            tracking_fields = {
                "tracks",
                "active_track_count",
                "confirmed_person_detected",
                "confirmed_person_count",
                "primary_track_id",
                "primary_tracked_bbox",
            }
            missing_tracking = sum(
                not tracking_fields.issubset(observation) for observation in observations
            )
            if (
                statuses != {"ok"}
                or tracking_statuses != {"ok"}
                or None in backends
                or missing_tracking
            ):
                raise ValueError(
                    "person detector/tracker did not process every observation: "
                    f"{statuses}, {tracking_statuses}, {backends}, missing={missing_tracking}"
                )
            report["person_detection"] = {
                "verified": True,
                "tracking_contract_verified": True,
                "model_sha256": model_hash,
                "detector": person_detector,
                "score_threshold": person_threshold,
                "nms_threshold": person_nms_threshold,
                "input_size": person_input_size,
                "backends": sorted(backends),
                "observations": len(observations),
                "detections": sum(row["person_count"] for row in observations),
                "confirmed_detections": sum(row["confirmed_person_count"] for row in observations),
            }
        if initial_transmit_fps == 20:
            sent_live = [
                row
                for row in edge_records
                if row["event"] == "frame_sent" and row.get("evidence_kind") == "live"
            ]
            received_live = [
                row
                for row in cloud_records
                if row["event"] == "frame_received" and row.get("evidence_kind") == "live"
            ]
            processed_live = [
                row
                for row in cloud_records
                if row["event"] == "observation" and row.get("evidence_kind") == "live"
            ]
            report["fixed_20_fps_diagnostic"] = {
                "scope": "Fixed sensing transport/load test; no agent command is claimed",
                "target_transmit_fps": 20,
                "live_frames_sent": len(sent_live),
                "live_frames_received": len(received_live),
                "live_frames_processed": len(processed_live),
                "live_frames_dropped": sum(
                    row["event"] == "frame_dropped" for row in cloud_records
                ),
                "edge_send_fps_complete_5s_windows": complete_five_second_rates(sent_live),
                "cloud_receive_fps_complete_5s_windows": complete_five_second_rates(received_live),
                "processing_ms_median": (
                    round(statistics.median(row["processing_ms"] for row in processed_live), 3)
                    if processed_live
                    else None
                ),
            }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        if netem_started:
            try:
                ssh.python(STOP_NETEM, remote_run)
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"Netem cleanup requires inspection: {exc}", file=sys.stderr)
            for name in ("netem.jsonl", "netem.stderr"):
                try:
                    (output / name).write_bytes(ssh.python(READ_FILE, remote_run + "/" + name))
                except (OSError, subprocess.SubprocessError):
                    pass
        if tunnel:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=3)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait()
        tunnel_log.close()
        # Best effort cleanup even when the edge or verification fails. The remote
        # 300-second bound also stops the service if SSH connectivity is lost.
        try:
            ssh.python(STOP, remote_run)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"No se pudo confirmar limpieza remota: {exc}", file=sys.stderr)
        for name in ("cloud.jsonl", "cloud.stderr"):
            if not (output / name).exists():
                try:
                    data = ssh.python(
                        READ_FILE,
                        remote_run + "/" + name,
                    )
                    (output / name).write_bytes(data)
                except (OSError, subprocess.SubprocessError):
                    pass
        if (transport_pause is not None or tcp_telemetry) and not (
            output / "tcp-socket.jsonl"
        ).exists():
            try:
                (output / "tcp-socket.jsonl").write_bytes(
                    ssh.python(READ_FILE, remote_run + "/tcp-socket.jsonl")
                )
            except (OSError, subprocess.SubprocessError):
                pass
        if (camera_index is not None or record_rtsp) and video and video.exists():
            deployment_path = output / "deployment.json"
            deployment = json.loads(deployment_path.read_text())
            deployment["input_video_sha256"] = hashlib.sha256(video.read_bytes()).hexdigest()
            deployment["recording_scope"] = (
                "Local camera recording; frame_sent identifies transmitted captures"
            )
            deployment_path.write_text(json.dumps(deployment, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="CARE-Vision phases 2/3: edge local, cloud en Graviton"
    )
    parser.add_argument("--host", required=True, help="user@hostname or existing SSH alias")
    parser.add_argument("--phase", type=int, choices=[2, 3], default=2)
    parser.add_argument(
        "--identity", type=Path, required=True, help="local SSH key (never uploaded)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts") / ("aws-" + time.strftime("%Y%m%d-%H%M%S"))
    )
    parser.add_argument("--person-model", type=Path, help="optional person detector ONNX model")
    parser.add_argument("--person-detector", choices=["nanodet", "yolox"], default="nanodet")
    parser.add_argument("--person-threshold", type=float, default=0.35)
    parser.add_argument("--person-nms-threshold", type=float, default=0.3)
    parser.add_argument("--person-input-size", type=int)
    parser.add_argument("--experimental-loss-5fps", action="store_true")
    parser.add_argument("--experimental-s-only-5fps", action="store_true")
    parser.add_argument("--experimental-weak-association", action="store_true")
    parser.add_argument("--experimental-alert-after-seconds", type=float)
    parser.add_argument("--experimental-episode-continuity", action="store_true")
    parser.add_argument(
        "--episode-state-key", help="Isolated stable pending-state namespace on EC2"
    )
    parser.add_argument("--diagnostic-disconnect-after-episode-seconds", type=float)
    parser.add_argument("--verification-model", type=Path)
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--source-captures", type=Path)
    parser.add_argument("--replay-sampling-phase-ms", type=int)
    parser.add_argument("--camera", type=int)
    parser.add_argument("--rtsp-ip", help="local IPv4 address of an IP camera")
    parser.add_argument("--rtsp-stream", type=int, choices=[1, 2], default=1)
    parser.add_argument("--rtsp-ffmpeg-nobuffer", action="store_true")
    parser.add_argument("--record-rtsp", action="store_true")
    parser.add_argument("--camera-seconds", type=float, default=45)
    parser.add_argument("--capture-fps", type=float, default=30)
    parser.add_argument("--initial-transmit-fps", type=int, choices=[2, 20], default=2)
    parser.add_argument(
        "--tcp-telemetry",
        action="store_true",
        help="read-only Mac nettop and AWS ss telemetry; no relay or loss injection",
    )
    parser.add_argument("--capture-resolution", choices=["640x360", "1280x720"], default="640x360")
    parser.add_argument(
        "--baseline-resolution", choices=["320x180", "640x360", "1280x720"], default="640x360"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if not (root / "uv.lock").is_file():
        parser.error("run from an editable checkout containing uv.lock")
    if args.output.exists():
        parser.error("output directory must be new")
    rtsp_username = rtsp_password = None
    if args.rtsp_ip is not None:
        rtsp_username = os.environ.get("CARE_VISION_RTSP_USERNAME") or input(
            "Camera account username: "
        )
        rtsp_password = os.environ.get("CARE_VISION_RTSP_PASSWORD") or getpass.getpass(
            "Camera account password: "
        )
    try:
        report = run_aws_demo(
            args.host,
            args.identity,
            args.output.resolve(),
            root,
            args.phase,
            args.person_model,
            args.person_detector,
            args.person_threshold,
            args.person_nms_threshold,
            args.person_input_size,
            args.source_video,
            args.capture_resolution,
            args.baseline_resolution,
            camera_index=args.camera,
            camera_seconds=args.camera_seconds,
            tcp_telemetry=args.tcp_telemetry,
            rtsp_ip=args.rtsp_ip,
            rtsp_stream=args.rtsp_stream,
            rtsp_username=rtsp_username,
            rtsp_password=rtsp_password,
            rtsp_ffmpeg_nobuffer=args.rtsp_ffmpeg_nobuffer,
            record_rtsp=args.record_rtsp,
            capture_fps=args.capture_fps,
            initial_transmit_fps=args.initial_transmit_fps,
            source_captures=args.source_captures,
            replay_sampling_phase_ms=args.replay_sampling_phase_ms,
            verification_model=args.verification_model,
            experimental_loss_5fps=args.experimental_loss_5fps,
            experimental_s_only_5fps=args.experimental_s_only_5fps,
            experimental_weak_association=args.experimental_weak_association,
            experimental_alert_after_seconds=args.experimental_alert_after_seconds,
            experimental_episode_continuity=args.experimental_episode_continuity,
            episode_state_key=args.episode_state_key,
            diagnostic_disconnect_after_episode_seconds=(
                args.diagnostic_disconnect_after_episode_seconds
            ),
        )
    except Exception as exc:
        parser.exit(1, f"FAIL: {exc}\nEvidence: {args.output.resolve()}\n")
    print(json.dumps(report, indent=2))
    if report.get("passed"):
        print(f"PASS: Fase {args.phase} verificada. Evidencia: {args.output.resolve()}")
    else:
        print(f"DIAGNOSTIC: evidencia conservada en {args.output.resolve()}")


if __name__ == "__main__":
    main()
