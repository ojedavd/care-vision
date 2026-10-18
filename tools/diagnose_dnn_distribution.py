"""Bounded stage/engine/thread diagnostic over the frozen distribution benchmark inputs."""

import argparse
import ctypes
import gzip
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

from benchmark_cv_distributions import PYTHON, save, sha

CASES = [
    ("standard-auto", "standard", "auto", False),
    ("cool-auto", "cool", "auto", False),
    ("cool-classic", "cool", "classic", False),
    ("cool-new", "cool", "new", False),
    ("cool-auto-blas1", "cool", "auto", True),
    ("standard-auto-blas1", "standard", "auto", True),
]


class TimedNet:
    def __init__(self, net, timings):
        self.net, self.timings = net, timings

    def __getattr__(self, name):
        return getattr(self.net, name)

    def setInput(self, *args):
        start = time.perf_counter()
        result = self.net.setInput(*args)
        self.timings["set_input_ms"] = (time.perf_counter() - start) * 1000
        return result

    def forward(self, *args):
        start = time.perf_counter()
        result = self.net.forward(*args)
        self.timings["forward_ms"] = (time.perf_counter() - start) * 1000
        return result


def task_times():
    result = {}
    for p in Path("/proc/self/task").glob("*/stat"):
        try:
            fields = p.read_text().rsplit(")", 1)[1].split()
            result[p.parent.name] = int(fields[11]) + int(fields[12])
        except FileNotFoundError:
            pass
    return result


def libraries():
    import subprocess

    paths = sorted(
        {
            r.split()[-1]
            for r in Path("/proc/self/maps").read_text().splitlines()
            if "/" in r and any(s in r for s in ("openblas", "libarmpl"))
        }
    )
    result = []
    for path in paths:
        symbols = subprocess.run(["nm", "-D", path], capture_output=True, text=True, check=True)
        names = [line.split()[-1] for line in symbols.stdout.splitlines() if line.split()]
        getters = [
            n
            for n in names
            if n.startswith(
                (
                    "openblas_get_num_threads",
                    "scipy_openblas_get_num_threads",
                    "armpl_omp_get_max_threads",
                )
            )
        ]
        counts = {}
        lib = ctypes.CDLL(path)
        for name in getters:
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = [], ctypes.c_int
            counts[name] = fn()
        result.append({"path": path, "sha256": sha(Path(path)), "reported_threads": counts})
    return result


def worker(base, label):
    import cv2
    import numpy as np

    from care_vision.cloud.yolox import YoloXPersonDetector
    from care_vision.common.environment import runtime_info

    _, mode, engine, cap = next(c for c in CASES if c[0] == label)
    cv2.setNumThreads(8)
    cv2.ocl.setUseOpenCL(False)
    plan = json.loads((base / "plan.json").read_text())
    assert sha(base / "model.onnx") == plan["model_sha256"]
    for p, h in plan["code_sha256"].items():
        assert sha(base / p) == h
    assert (
        sha(Path(__file__))
        == json.loads((base / "diagnostic-plan.json").read_text())["worker_sha256"]
    )
    detector = YoloXPersonDetector(
        base / "model.onnx", score_threshold=0.20, nms_threshold=0.30, input_size=640
    )
    if engine != "auto":
        enum = getattr(cv2.dnn, "ENGINE_" + engine.upper())
        detector.net = cv2.dnn.readNet(str(base / "model.onnx"), "", "", enum)
        detector.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        detector.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    original_net = detector.net
    payloads = []
    for item in plan["frames"]:
        path = base / item["path"]
        assert sha(path) == item["sha256"]
        payloads.append(path.read_bytes())
    first = cv2.imdecode(np.frombuffer(payloads[0], np.uint8), cv2.IMREAD_COLOR)
    for _ in range(5):
        detector.detect(first)
    reference = detector.detect(first)
    timings = {}
    detector.net = TimedNet(original_net, timings)
    original_preprocess = detector._preprocess

    def preprocess(image):
        start = time.perf_counter()
        value = original_preprocess(image)
        timings["preprocess_ms"] = (time.perf_counter() - start) * 1000
        return value

    detector._preprocess = preprocess
    assert detector.detect(first) == reference
    runtime = runtime_info(3 if mode == "cool" else 2)
    before_libraries = libraries()
    before = task_times()
    rows = []
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    for item, payload in zip(plan["frames"], payloads, strict=True):
        start = time.perf_counter()
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        decode = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        detections = detector.detect(image)
        total = (time.perf_counter() - start) * 1000
        rows.append(
            {
                "input": item["path"],
                "decode_ms": decode,
                "detect_ms": total,
                **timings,
                "post_and_other_ms": total - sum(timings.values()),
                "detections": [{"bbox": d.bbox, "confidence": d.confidence} for d in detections],
            }
        )
    wall, cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
    after = task_times()
    hz = os.sysconf("SC_CLK_TCK")
    task_cpu = {tid: (ticks - before.get(tid, 0)) / hz for tid, ticks in after.items()}
    # Profiles are taken outside the timing window; new-engine profiling stays off while timed.
    profile = {}
    try:
        if engine != "classic" and mode == "cool":
            profile["mode_before"] = original_net.getProfilingMode()
            original_net.setProfilingMode(cv2.dnn.DNN_PROFILE_DETAILED)
            detector.detect(first)
        data = original_net.getPerfProfile()
        profile["raw"] = [v.tolist() if hasattr(v, "tolist") else v for v in data]
        profile["tick_frequency"] = cv2.getTickFrequency()
        try:
            profile["layer_names"] = list(original_net.getLayerNames())
        except Exception as error:
            profile["layer_names_error"] = str(error)
    except Exception as error:
        profile["error"] = str(error)
    try:
        (base / (label + ".graph.txt")).write_text(original_net.dump())
    except Exception as error:
        (base / (label + ".graph.txt")).write_text(str(error))
    runtime.update(
        numpy_native_sha256=sha(Path(np._core._multiarray_umath.__file__)),
        python_sha256=sha(Path(sys.executable)),
        opencv_threads=cv2.getNumThreads(),
        engine_enums={k: getattr(cv2.dnn, k) for k in dir(cv2.dnn) if k.startswith("ENGINE_")},
        thread_environment={
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "ARMPL_NUM_THREADS",
                "OPENCV_FORCE_DNN_ENGINE",
            )
        },
    )
    save(
        base / (label + ".json"),
        {
            "label": label,
            "mode": mode,
            "engine_requested": engine,
            "blas_cap_requested": cap,
            "runtime": runtime,
            "rows": rows,
            "libraries_before": before_libraries,
            "libraries_after": libraries(),
            "task_cpu_seconds": task_cpu,
            "wall_seconds": wall,
            "cpu_seconds": cpu,
            "preprocess_blob_sha256": hashlib.sha256(
                original_preprocess(first)[0].tobytes()
            ).hexdigest(),
            "profile": profile,
            "wrapper_parity_first_frame": True,
        },
    )


def run(args):
    from run_mvp_sensing_comparison import STATE_HASHES
    from run_pose_observation_loop import READ_COMPRESSED, SSH

    prior, output = args.prior.resolve(), args.output.resolve()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    plan = {
        "prior_package_sha256": sha(prior / "package.tar.gz"),
        "cases": CASES,
        "worker_sha256": sha(Path(__file__)),
        "input": "All 240 frozen JPEG; no new video",
        "scope": "Locate detector cost, compare classic/new inside same COOL build, "
        "test BLAS/OMP environment caps with CV threads kept at8",
        "timing": "decode, original preprocess, setInput, forward, post residual; "
        "five warmups, parity check; detailed profiling outside measured loop",
        "thread_caveat": "Getter values are pool configuration; task CPU deltas measure activity",
        "output_tolerance": "Same counts, IoU>=.99, confidence delta<=.001; report every failure",
        "new_camera": False,
        "production_changes": False,
    }
    save(output / "plan.json", plan)
    ssh = SSH(args.host, args.identity)
    before = json.loads(ssh.python(STATE_HASHES))
    save(output / "existing-state-before.json", before)
    remote = "/home/ubuntu/care-vision-dnn-diagnostic-" + uuid.uuid4().hex[:10]
    deployment = {"workspace": remote, "complete": False, "instance_left_running": True}
    save(output / "deployment.json", deployment)
    setup = """import io,os,pathlib,sys,tarfile
os.umask(0o077);p=pathlib.Path(sys.argv[1]);p.mkdir(exist_ok=False)
tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz').extractall(p,filter='data')
(p/'cool-site').mkdir();(p/'cool-site/cv2').symlink_to('/opt/cool/python_3.11/site-packages/cv2')
"""
    ssh.python(setup, remote, payload=(prior / "package.tar.gz").read_bytes(), timeout=120)
    write = "import pathlib,sys;pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())"
    ssh.python(
        write, remote + "/tools/diagnose_dnn_distribution.py", payload=Path(__file__).read_bytes()
    )
    ssh.python(write, remote + "/diagnostic-plan.json", payload=(output / "plan.json").read_bytes())
    launch = """import os,pathlib,subprocess,sys
base=pathlib.Path(sys.argv[1]);mode=sys.argv[2];label=sys.argv[3]
env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','LD_LIBRARY_PATH')}
assert 'OPENCV_FORCE_DNN_ENGINE' not in env
env.update(PYTHONNOUSERSITE='1',PYTHONPATH=str(base/'src'))
if mode=='cool':
 env['PYTHONPATH']+=':'+str(base/'cool-site')
 env['LD_LIBRARY_PATH']='/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib'
if sys.argv[5]=='1':
 env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',ARMPL_NUM_THREADS='1')
with (base/(label+'.stderr')).open('w') as error:
 subprocess.run([sys.argv[4],str(base/'tools/diagnose_dnn_distribution.py'),
 '--worker-base',str(base),'--label',label],env=env,stdout=error,stderr=error,
 check=True,timeout=180)
"""
    for label, mode, _, cap in CASES:
        print("Starting " + label, flush=True)
        try:
            ssh.python(launch, remote, mode, label, PYTHON, "1" if cap else "0", timeout=200)
        finally:
            (output / (label + ".stderr")).write_bytes(
                gzip.decompress(ssh.python(READ_COMPRESSED, remote + "/" + label + ".stderr"))
            )
        for suffix in (".json", ".graph.txt"):
            (output / (label + suffix)).write_bytes(
                gzip.decompress(
                    ssh.python(READ_COMPRESSED, remote + "/" + label + suffix, timeout=55)
                )
            )
        print("Finished " + label, flush=True)
    after = json.loads(ssh.python(STATE_HASHES))
    save(output / "existing-state-after.json", after)
    assert before == after
    ssh.python(
        """import pathlib,shutil,sys
p=pathlib.Path(sys.argv[1]);assert p.parent==pathlib.Path('/home/ubuntu')
assert p.name.startswith('care-vision-dnn-diagnostic-');shutil.rmtree(p);assert not p.exists()
""",
        remote,
    )
    deployment.update(complete=True, existing_states_unchanged=True, workspace_removed=True)
    save(output / "deployment.json", deployment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-base", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--identity", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.worker_base:
        worker(args.worker_base, args.label)
    else:
        run(args)
