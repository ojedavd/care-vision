"""Build a verified local replay and a silent MP4 backup from an AWS demo bundle."""

import argparse
import bisect
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from care_vision.aws_demo import source_archive
from care_vision.checkin import validate_bundle
from care_vision.edge.buffer import fit_to_canvas


def make_replay(run, output, rows):
    observations = [r for r in rows if r["event"] == "observation"
                    and r.get("evidence_kind", "live") == "live"]
    times = [r["t"] for r in observations]
    needed = {r["capture_seq"] for r in observations}
    video = cv2.VideoCapture(str(run / "demo.avi"))
    images = {}
    index = 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            index += 1
            if index in needed:
                images[index] = fit_to_canvas(frame, 1280, 720)
    finally:
        video.release()
    if set(images) != needed:
        raise ValueError("source video does not contain every logged capture_seq")
    duration = max(r["t"] for r in rows) + 1
    encoder = subprocess.Popen([
        shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", "1280x820", "-r", "10", "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output / "replay.mp4"),
    ], stdin=subprocess.PIPE)
    state = "NORMAL"
    event_index = 0
    action = "KEEP_OBSERVING"
    try:
        for i in range(math.ceil(duration * 10)):
            t = i / 10
            while event_index < len(rows) and rows[event_index]["t"] <= t:
                event = rows[event_index]
                if event["event"] == "agent_state_transition":
                    state = event["to_state"]
                if event["event"] == "agent_action":
                    action = event["type"]
                    if event.get("transmit_fps") is not None:
                        action += f'({event["transmit_fps"]})'
                event_index += 1
            n = bisect.bisect_right(times, t) - 1
            canvas = np.full((820, 1280, 3), (24, 21, 18), dtype=np.uint8)
            if n >= 0:
                obs = observations[n]
                canvas[:720] = images[obs["capture_seq"]]
                for key, color in (("primary_source_bbox", (90, 230, 130)),
                                   ("source_roi", (230, 190, 75))):
                    if obs.get(key):
                        x, y, w, h = map(round, obs[key])
                        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3)
                detail = f'Frame {obs["capture_seq"]} | {obs["width"]}x{obs["height"]}'
            else:
                detail = "Esperando primera observacion"
            cv2.putText(canvas, f'{t:05.1f}s AWS | {state} | {action}', (22, 754),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)
            cv2.putText(canvas, f'{detail} | Replay de logs; imagen fuente reconstruida',
                        (22, 785), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)
            cv2.putText(canvas, 'Demo de arquitectura. No clasifica caidas ni activa alertas.',
                        (22, 809), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 200, 230), 1)
            encoder.stdin.write(canvas.tobytes())
    finally:
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f"ffmpeg failed: {code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, help="optional JSON record of all demo attempts")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary, rows = validate_bundle(args.run)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["passed"]:
        parser.exit(1, f"Verification failed; see {args.output / 'summary.json'}\n")
    make_replay(args.run, args.output, rows)
    evidence = {"summary": summary, "rows": rows}
    if args.attempts:
        evidence["attempts"] = json.loads(args.attempts.read_text())
        shutil.copyfile(args.attempts, args.output / "attempts.json")
    data = json.dumps(evidence).replace("<", "\\u003c")
    template = Path(__file__).with_name("checkin_demo.html").read_text()
    (args.output / "index.html").write_text(template.replace("__EVIDENCE_JSON__", data))
    (args.output / "replay.json").write_text(json.dumps(rows, indent=2) + "\n")
    # Preserve the verifier/presentation code too: it may postdate the measured run.
    root = Path(__file__).resolve().parents[1]
    (args.output / "replay-tooling.tar.gz").write_bytes(source_archive(root, [
        (Path(__file__), "tools/build_checkin_demo.py"),
        (Path(__file__).with_name("checkin_demo.html"), "tools/checkin_demo.html"),
    ]))
    files = [p for p in args.run.iterdir() if p.is_file()] + list(args.output.iterdir())
    (args.output / "evidence-sha256.json").write_text(json.dumps({
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }, indent=2) + "\n")
    print(json.dumps({"passed": True, "output": str(args.output.resolve()),
                      "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
