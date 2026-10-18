"""Five recorded classic replays using a fresh locked cloud environment; SDK external."""

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

from portable_cloud_runtime import CLEANUP, PREPARE_RUNTIME, START, STATE_HASHES
from run_pose_observation_loop import READ_COMPRESSED, READY, SSH, STOP, save, sha

from care_vision.aws_demo import UV_VERSION


def remote_path(value):
    if not value.startswith('/') or '..' in Path(value).parts or '\n' in value:
        raise argparse.ArgumentTypeError('expected an absolute remote path without traversal')
    return value.rstrip('/') or '/'


def edge_environment(root):
    """Load Edge from the verified source tree, ahead of an editable install."""
    environment = dict(os.environ)
    paths = [str(root / "src")]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def verify_inputs(root, assets, profile_path, asset_paths):
    expected = json.loads(profile_path.read_text())['files_sha256']
    checked = {}
    expected_source = {p for p in expected if p.startswith('src/') and p.endswith('.py')}
    actual_source = {str(p.relative_to(root)) for p in (root / 'src').rglob('*.py')}
    if not actual_source or actual_source != expected_source:
        raise ValueError('frozen pipeline mismatch: source file set changed')
    for path in sorted((root / 'src').rglob('*.py')):
        if path.is_symlink():
            raise ValueError('source symlink rejected')
        name = str(path.relative_to(root))
        if expected.get(name) != sha(path):
            raise ValueError(f'frozen pipeline mismatch: {name}')
        checked[name] = sha(path)
    for path in sorted(set(asset_paths)):
        if path.is_symlink() or not path.resolve().is_relative_to(assets):
            raise ValueError('asset must be a regular file inside asset root')
        name = str(path.relative_to(assets))
        if expected.get(name) != sha(path):
            raise ValueError(f'frozen asset mismatch: {name}')
        checked[name] = sha(path)
    return checked


def run(args):
    if os.environ.get("CARE_DIAGNOSTIC_BYTE_BUDGET") != "114688":
        raise ValueError("This experimental runner requires CARE_DIAGNOSTIC_BYTE_BUDGET=114688")
    root = Path(__file__).resolve().parents[1]
    assets = args.asset_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    config_source = assets / (
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
    cases = [('recovery-2', 'baseline'), ('no-recovery-1', 'baseline'), ('no-recovery-1', 'test20'), ('sit-return-live-2', 'pose'), ('sit-hold-live-1', 'pose')]
    sources = {
        name: assets
        / "artifacts"
        / ("pose-observation-live-20261014" if mode == "pose" else "tapo-weak-live-20261005")
        / name
        for name, mode in cases
    }
    files = sorted((root / "src").rglob("*.py")) + [
        Path(__file__).resolve(),
        root / "tools/run_pose_observation_loop.py",
        root / "tools/portable_cloud_runtime.py",
        root / "pyproject.toml", root / "uv.lock", root / ".python-version",
    ]
    models = {
        "models/person.onnx": assets
        / ("artifacts/phase5c-dataset-review-20260920/models/object_detection_yolox_2022nov.onnx"),
        "models/pose.onnx": assets / "artifacts/phase7d-pose-20260921/models/end2end.onnx",
    }
    required_assets = [config_source, *models.values(), *(
        p for source in sources.values()
        for p in (source / 'demo.avi', source / 'demo.captures.jsonl')
    )]
    frozen = verify_inputs(root, assets, args.profile, required_assets)
    save(output / 'local-preflight.json', {'files_sha256': frozen,
                                         'profile_sha256': sha(args.profile)})
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
            str(p.relative_to(assets)): sha(p)
            for source in sources.values()
            for p in (source / "demo.avi", source / "demo.captures.jsonl",
                      *([source / 'activity-review.json']
                        if source.name.startswith('sit-') else []))
        },
        "config_sha256": {p.name: sha(p) for p in output.glob("pose-*.json")},
        "new_camera": False,
        "engine": "classic",
        "pose_engine": "unchanged",
        "portable_runtime": {"python": args.remote_python, "uv": args.remote_uv,
                             "uv_version": UV_VERSION, "sdk": "external /opt/cool"},
        "asset_root": str(assets),
        "frozen_asset_paths": sorted(str(p.relative_to(assets)) for p in set(required_assets)),
        "prepare_only": args.prepare_only,
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
    if args.prepare_only:
        print(json.dumps({'prepared': True, 'remote_contacted': False,
                          'package_sha256': sha(output / 'package.tar.gz')}))
        return
    ssh = SSH(args.host, args.identity)
    before = json.loads(ssh.python(STATE_HASHES, args.state_root))
    save(output / "existing-state-before.json", before)
    remote = args.remote_root + "/care-vision-portable-classic-" + uuid.uuid4().hex[:10]
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
    try:
        preflight = json.loads(ssh.python(
            PREPARE_RUNTIME, remote, args.remote_python, args.remote_uv, UV_VERSION, timeout=90,
        ))
    except Exception:
        save(output / 'runtime-preparation-failed.json', {
            'workspace': remote, 'worker_started': False,
            'workspace_retained_for_diagnosis': True,
        })
        raise
    save(output / 'remote-preflight.json', preflight)
    for name, mode in cases:
        label = name + "-" + mode
        destination, remote_run = output / label, remote + "/" + label
        destination.mkdir(mode=0o700)
        source = sources[name]
        tunnel, launched, network_proxy = None, False, None
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
                network_proxy = subprocess.Popen(
                    [sys.executable, str(root / "tools/diagnostic_proxy.py"),
                     str(port), str(destination / "proxy.json")],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                readable, _, _ = select.select([network_proxy.stdout], [], [], 10)
                if not readable:
                    raise RuntimeError("diagnostic proxy did not become ready")
                proxy_port = json.loads(network_proxy.stdout.readline())["port"]
                edge_command = [
                    sys.executable,
                    str(root / "tools/diagnostic_edge.py"),
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
                    f"ws://127.0.0.1:{proxy_port}",
                    "--log",
                    str(destination / "edge.jsonl"),
                ]
                edge_command += ["--replay-sampling-phase-ms", "0"]
                with (destination / "edge.stderr").open("w") as edge_log:
                    subprocess.run(
                        edge_command,
                        env=edge_environment(root),
                        stdout=edge_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=90,
                    )
        finally:
            if network_proxy is not None:
                network_proxy.terminate()
                try: network_proxy.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    network_proxy.kill(); network_proxy.wait()
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
    after = json.loads(ssh.python(STATE_HASHES, args.state_root))
    save(output / "existing-state-after.json", after)
    assert before == after
    deployment.update(
        complete=True,
        existing_states_unchanged=True,
        cleanup=json.loads(ssh.python(CLEANUP, remote, args.remote_root)),
    )
    save(output / "deployment.json", deployment)
    print("Five runs complete; audit required", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True,
                        help="Root of the externally supplied archived artifacts tree")
    parser.add_argument("--profile", type=Path, required=True,
                        help="Original frozen profile.json; hashes checked before SSH")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Build and hash the deployment payload locally; never contact SSH")
    parser.add_argument("--remote-root", type=remote_path)
    parser.add_argument("--state-root", type=remote_path)
    parser.add_argument("--remote-python", type=remote_path)
    parser.add_argument("--remote-uv", type=remote_path)
    os.umask(0o077)
    args = parser.parse_args()
    if not args.prepare_only and any(getattr(args, field) is None for field in (
        'host', 'identity', 'remote_root', 'state_root', 'remote_python', 'remote_uv',
    )):
        parser.error('remote execution requires host, identity, roots, Python and uv paths')
    run(args)
