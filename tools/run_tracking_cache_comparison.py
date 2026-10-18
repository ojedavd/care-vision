"""Run the pinned offline trackers in the cache's temporary Graviton workspace."""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import tarfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if (args.base / "tracking-comparison.json").exists():
        raise ValueError("refusing to overwrite comparison")
    deployment = json.loads((args.base / "deployment.json").read_text())
    remote = deployment["remote"]
    if not re.fullmatch(r"/tmp/care-vision-tracking-cache-[a-f0-9]{32}", remote):
        raise ValueError("unexpected remote workspace")
    provenance = json.loads((args.upstream / "provenance.json").read_text())
    for filename, digest in provenance["files"].items():
        if sha(args.upstream / filename) != digest:
            raise ValueError("upstream source digest mismatch")
    worker = Path(__file__).with_name("compare_tracking_cache.py")
    package = args.base / "tracker-package.tar.gz"
    with tarfile.open(package, "x:gz") as archive:
        archive.add(args.upstream, arcname="bytetrack-upstream")
        archive.add(worker, arcname=worker.name)
    (args.base / "tracker-deployment.json").write_text(
        json.dumps(
            {
                "upstream_commit": provenance["commit"],
                "package_sha256": sha(package),
                "worker_sha256": sha(worker),
                "runner_sha256": sha(Path(__file__)),
                "remote": remote,
            },
            indent=2,
        )
        + "\n"
    )
    options = [
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=10",
        "-i",
        str(args.identity),
    ]
    ssh = ["ssh", *options, args.host]
    subprocess.run(["scp", *options, str(package), f"{args.host}:{remote}/"], check=True)
    uv = "/home/ubuntu/care-vision-phase3/bootstrap/bin/uv"
    command = (
        f"tar -xzf {remote}/tracker-package.tar.gz -C {remote} && "
        f"{uv} venv --python {shlex.quote(deployment['remote_python'])} {remote}/track-env && "
        f"{uv} pip install --python {remote}/track-env/bin/python "
        "numpy==2.3.4 scipy==1.16.3 lap==0.5.12 cython-bbox==0.1.5 cython==3.3.0"
    )
    with (args.base / "tracker-install.log").open("x") as log:
        subprocess.run([*ssh, command], stdout=log, stderr=subprocess.STDOUT, check=True)
    prefix = (
        f"PYTHONNOUSERSITE=1 PYTHONPATH={remote}:/opt/cool/python_3.11/site-packages "
        "LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib "
    )
    command = (
        prefix + f"timeout 120 {remote}/track-env/bin/python -u "
        f"{remote}/{worker.name} --base {remote}"
    )
    with (args.base / "tracker-execution.log").open("x") as log:
        subprocess.run([*ssh, command], stdout=log, stderr=subprocess.STDOUT, check=True)
    for filename in ("tracking-comparison.json", "tracking-timelines.jsonl"):
        subprocess.run(
            ["scp", *options, f"{args.host}:{remote}/{filename}", str(args.base / filename)],
            check=True,
        )
    result = json.loads((args.base / "tracking-comparison.json").read_text())
    if sha(args.base / "tracking-timelines.jsonl") != result["timeline_sha256"]:
        raise ValueError("downloaded timeline digest mismatch")
    probe = (
        "import numpy,json; print(json.dumps({'numpy_version':numpy.__version__,"
        "'numpy_module':numpy.__file__}))"
    )
    completed = subprocess.run(
        [
            *ssh,
            prefix + f"{remote}/track-env/bin/python -c " + shlex.quote(probe),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    (args.base / "dependency-probe.json").write_text(
        json.dumps(json.loads(completed.stdout), indent=2) + "\n"
    )
    subprocess.run(
        [*ssh, f"tar -czf {remote}/bytetrack-adapted.tar.gz -C {remote} bytetrack-numpy"],
        check=True,
    )
    subprocess.run(
        [
            "scp",
            *options,
            f"{args.host}:{remote}/bytetrack-adapted.tar.gz",
            str(args.base / "bytetrack-adapted.tar.gz"),
        ],
        check=True,
    )
    # Only this validated temporary workspace is removed, after result verification.
    subprocess.run([*ssh, f"rm -rf {remote} && test ! -e {remote}"], check=True)
    (args.base / "cleanup.json").write_text(
        json.dumps(
            {
                "remote": remote,
                "temporary_workdir_removed": True,
                "offline_workers_exited": True,
                "instance_left_running": True,
                "pending_episode_snapshot_untouched": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("comparison ready; temporary AWS evidence removed")


if __name__ == "__main__":
    main()
