"""Paired fixed-input cloud workload: installed standard OpenCV versus COOL distribution.

No camera, controller, package installation, SDK mutation or external alert.
Different OpenCV versions prevent attributing the difference solely to COOL.
"""

import argparse
import bisect
import hashlib
import json
import os
import sys
import tarfile
import time
import uuid
from pathlib import Path

PYTHON = "/home/ubuntu/care-vision-phase2/releases/20260919-093333-2385cfc5/.venv/bin/python"
ORDER = ["standard", "cool", "cool", "standard"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sample_indices(stamps, interval_ns=500_000_000):
    """Nearest original timestamp to a fixed grid; never choose based on detections."""
    result = []
    for target in range(stamps[0], stamps[-1] + 1, interval_ns):
        right = bisect.bisect_left(stamps, target)
        options = [i for i in (right - 1, right) if 0 <= i < len(stamps)]
        index = min(options, key=lambda i: (abs(stamps[i] - target), i))
        if index not in result:
            result.append(index)
    return result


def worker(base, mode, label):
    import cv2
    import numpy as np

    from care_vision.cloud.vision import MotionVision
    from care_vision.cloud.yolox import YoloXPersonDetector
    from care_vision.common.environment import runtime_info

    cv2.setNumThreads(8)
    cv2.ocl.setUseOpenCL(False)
    assert cv2.getNumThreads() == 8 and cv2.useOptimized()
    plan = json.loads((base / "plan.json").read_text())
    assert sha(base / "model.onnx") == plan["model_sha256"]
    for path, digest in plan["code_sha256"].items():
        assert sha(base / path) == digest
    payloads = []
    decoded_hashes = []
    for item in plan["frames"]:
        path = base / item["path"]
        assert sha(path) == item["sha256"]
        payload = path.read_bytes()
        payloads.append(payload)
        decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (720, 1280, 3)
        decoded_hashes.append(hashlib.sha256(decoded.tobytes()).hexdigest())
    del decoded
    detector = YoloXPersonDetector(base / "model.onnx", score_threshold=.20,
                                   nms_threshold=.30, input_size=640)
    for payload in payloads[:5]:
        detector.detect(cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR))
    runtime = runtime_info(3 if mode == "cool" else 2)
    native = Path(getattr(cv2, "_native", cv2).__file__).resolve()
    mapped = sorted({r.split()[-1] for r in Path('/proc/self/maps').read_text().splitlines()
                     if '/' in r and any(s in r for s in ('libopencv_', '/cv2', 'openblas',
                                                         'libarmpl', 'kleidicv'))})
    if mode == "standard":
        assert not any('/opt/cool/' in p for p in mapped)
        assert cv2.__version__ == "4.14.0"
    else:
        assert runtime["cool"]["verified"]
    runtime.update(numpy_version=np.__version__, numpy_path=np.__file__,
                   numpy_native_sha256=sha(Path(np._core._multiarray_umath.__file__)),
                   python_executable=sys.executable, python_sha256=sha(Path(sys.executable)),
                   native_path=str(native), native_sha256=sha(native),
                   mapped_libraries={p: sha(Path(p)) for p in mapped},
                   build=cv2.getBuildInformation(), opencv_threads=cv2.getNumThreads(),
                   cpu_affinity=sorted(os.sched_getaffinity(0)), opencl=cv2.ocl.useOpenCL())
    load_before = Path('/proc/loadavg').read_text().strip()
    visions = {name: MotionVision(person_detector=detector, experimental_weak_association=True)
               for name in plan["sources"]}
    observations = []
    begin_cpu, begin_wall = time.process_time(), time.perf_counter()
    for item, payload in zip(plan["frames"], payloads, strict=True):
        started = time.perf_counter()
        result = visions[item["source"]].observe(item["header"], payload)
        seconds = time.perf_counter() - started
        assert result["person_detection_status"] == result["person_tracking_status"] == "ok"
        observations.append({"input": item["path"], "seconds": seconds, "observation": result})
    wall, cpu = time.perf_counter() - begin_wall, time.process_time() - begin_cpu
    rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    save(base / (label + ".json"), {
        "mode": mode, "label": label, "runtime": runtime, "observations": observations,
        "decoded_sha256": decoded_hashes, "cpu_seconds": cpu, "wall_seconds": wall,
        "rss_bytes_end": rss, "load_before": load_before,
        "load_after": Path('/proc/loadavg').read_text().strip(),
    })
    print(label + " complete", flush=True)


def prepare(root, output):
    import cv2

    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (output / "images").mkdir(mode=0o700)
    frames, inputs = [], {}
    sources = ["recovery-2", "no-recovery-1"]
    for name in sources:
        folder = root / "artifacts/tapo-weak-live-20261005" / name
        rows = [json.loads(line) for line in
                (folder / "demo.captures.jsonl").read_text().splitlines()]
        indices = set(sample_indices([r["captured_mono_ns"] for r in rows]))
        cap = cv2.VideoCapture(str(folder / "demo.avi"))
        try:
            for index, row in enumerate(rows):
                ok, frame = cap.read()
                assert ok and frame.shape == (720, 1280, 3)
                if index not in indices:
                    continue
                path = f"images/{name}-{row['capture_seq']:06d}.jpg"
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                assert ok
                (output / path).write_bytes(encoded.tobytes())
                header = {"edge_id": "recorded-evaluation", "boot_id": name, "session_id": name,
                          "frame_id": f"{name}:{row['capture_seq']}",
                          "capture_seq": row["capture_seq"],
                          "captured_mono_ns": row["captured_mono_ns"],
                          "captured_at_utc": row.get("captured_at_utc", "recorded"),
                          "applied_command_id": None, "policy_seq": 0,
                          "width": 1280, "height": 720, "evidence_kind": "live",
                          "view_kind": "full_frame", "target_transmit_fps": 2}
                frames.append({"source": name, "path": path, "sha256": sha(output / path),
                               "header": header})
        finally:
            cap.release()
        for filename in ('demo.avi', 'demo.captures.jsonl'):
            p = folder / filename
            inputs[str(p.relative_to(root))] = sha(p)
    files = sorted((root / 'src').rglob('*.py')) + [Path(__file__).resolve()]
    model = root / (
        'artifacts/phase5c-dataset-review-20260920/models/object_detection_yolox_2022nov.onnx'
    )
    plan = {
        "scope": "Distribution comparison, same Graviton; not isolated COOL or Arm speedup",
        "order": ORDER, "sources": sources, "frames": frames, "input_sha256": inputs,
        "code_sha256": {str(p.relative_to(root)): sha(p) for p in files},
        "model_sha256": sha(model), "numpy_python": "Same phase2 Python and NumPy for both arms",
        "profile": "YOLOX-S640 threshold .20, NMS .30; tracker strong .35/weak .20; 8 CV threads",
        "workload": "JPEG decode + detector + tracking + trajectory + motion; 2Hz source grid",
        "measurement": "Unpaced sequential cloud capacity, preloaded JPEG bytes; five warmups",
        "excludes": ["disk/network", "model startup", "pose", "controller", "alerts", "billing"],
        "output_comparison": "Counts/track states exact; boxes IoU>=.99, confidence delta<=.001; "
                             "report all differences and never equate parity with ground truth",
        "limits": ["OpenCV4.14 wheel versus OpenCV5.1-dev COOL; versions/builds differ",
                   "Two runs per arm, ABBA; development data, no confidence intervals",
                   "Same ARM machine does not measure ARM versus x86",
                   "Grid is input selection; measured throughput is not real-time transmission"],
        "new_camera": False, "encoder_opencv": cv2.__version__,
    }
    save(output / 'plan.json', plan)
    with tarfile.open(output / 'package.tar.gz', 'w:gz') as archive:
        for p in files:
            archive.add(p, arcname=str(p.relative_to(root)), recursive=False)
        for item in frames:
            archive.add(output / item['path'], arcname=item['path'], recursive=False)
        archive.add(model, arcname='model.onnx', recursive=False)
        archive.add(output / 'plan.json', arcname='plan.json', recursive=False)


def run(args):
    from run_mvp_sensing_comparison import STATE_HASHES
    from run_pose_observation_loop import READ_COMPRESSED, SSH

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    prepare(root, output)
    ssh = SSH(args.host, args.identity)
    before = json.loads(ssh.python(STATE_HASHES))
    save(output / 'existing-state-before.json', before)
    remote = '/home/ubuntu/care-vision-cv-distributions-' + uuid.uuid4().hex[:10]
    deployment = {"workspace": remote, "complete": False, "instance_left_running": True,
                  "package_sha256": sha(output / 'package.tar.gz')}
    save(output / 'deployment.json', deployment)
    setup = '''import io,os,pathlib,sys,tarfile
os.umask(0o077);p=pathlib.Path(sys.argv[1]);p.mkdir(exist_ok=False)
tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz').extractall(p,filter='data')
(p/'cool-site').mkdir();(p/'cool-site/cv2').symlink_to('/opt/cool/python_3.11/site-packages/cv2')
'''
    ssh.python(setup, remote, payload=(output / 'package.tar.gz').read_bytes(), timeout=120)
    launch = '''import os,pathlib,subprocess,sys
base=pathlib.Path(sys.argv[1]);mode=sys.argv[2];label=sys.argv[3]
env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','LD_LIBRARY_PATH')}
env.update(PYTHONNOUSERSITE='1',PYTHONPATH=str(base/'src'))
if mode=='cool':
 env['PYTHONPATH']+=':'+str(base/'cool-site')
 env['LD_LIBRARY_PATH']='/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib'
with (base/(label+'.stderr')).open('w') as error:
 subprocess.run([sys.argv[4],str(base/'tools/benchmark_cv_distributions.py'),
  '--worker-base',str(base),'--mode',mode,'--label',label],env=env,
  stdout=error,stderr=error,check=True,timeout=300)
'''
    import gzip

    for i, mode in enumerate(ORDER):
        label = f'{i+1}-{mode}'
        print('Starting ' + label, flush=True)
        try:
            ssh.python(launch, remote, mode, label, PYTHON, timeout=320)
        finally:
            (output / (label + '.stderr')).write_bytes(gzip.decompress(ssh.python(
                READ_COMPRESSED, remote + '/' + label + '.stderr')))
        (output / (label + '.json')).write_bytes(gzip.decompress(ssh.python(
            READ_COMPRESSED, remote + '/' + label + '.json', timeout=55)))
        print('Finished ' + label, flush=True)
    after = json.loads(ssh.python(STATE_HASHES))
    save(output / 'existing-state-after.json', after)
    assert before == after
    cleanup = '''import pathlib,shutil,sys
p=pathlib.Path(sys.argv[1]);assert p.parent==pathlib.Path('/home/ubuntu')
assert p.name.startswith('care-vision-cv-distributions-')
shutil.rmtree(p);assert not p.exists()
'''
    ssh.python(cleanup, remote)
    deployment.update(complete=True, existing_states_unchanged=True, workspace_removed=True)
    save(output / 'deployment.json', deployment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-base', type=Path)
    parser.add_argument('--mode', choices=['standard', 'cool'])
    parser.add_argument('--label')
    parser.add_argument('--host')
    parser.add_argument('--identity', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.worker_base:
        worker(args.worker_base, args.mode, args.label)
    else:
        run(args)
