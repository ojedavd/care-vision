"""Measure screen-to-Edge RTSP delay with a large visible black/white marker.

Point the IP camera at this Mac's screen before running. This is a local
diagnostic: it does not contact AWS or persist frames.
"""

import argparse
import getpass
import json
import os
import statistics
import sys
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np

from care_vision.edge.source import RTSP_FFMPEG_NOBUFFER, RTSPSource

FLASH_SECONDS = (8, 16, 24, 32)
TOTAL_SECONDS = 41


def analyze(samples, flashes):
    """Find the first persistent brightness change after each screen flash."""
    results = []
    for index, flash in enumerate(flashes):
        at = flash["mono_ns"]
        next_at = (
            flashes[index + 1]["mono_ns"] if index + 1 < len(flashes) else (at + 8_000_000_000)
        )
        before = [value for stamp, value in samples if at - 2_000_000_000 <= stamp < at]
        after = [value for stamp, value in samples if next_at - 2_000_000_000 <= stamp < next_at]
        result = {"color": flash["color"], "screen_change_mono_ns": at}
        if len(before) < 10 or len(after) < 10:
            result["status"] = "insufficient_samples"
            results.append(result)
            continue
        before_level = statistics.median(before)
        after_level = statistics.median(after)
        change = after_level - before_level
        result["brightness_change"] = round(change, 2)
        result["response_polarity"] = (
            "expected" if (flash["color"] == "white") == (change > 0) else "inverted"
        )
        if abs(change) < 15:
            result["status"] = "insufficient_visual_contrast"
            results.append(result)
            continue
        threshold = (before_level + after_level) / 2
        window = [(stamp, value) for stamp, value in samples if at <= stamp < next_at]

        def crossing(value):
            return value >= threshold if change > 0 else value <= threshold

        detected = next(
            (
                stamp
                for (stamp, value), (_, next_value) in zip(window, window[1:])
                if crossing(value) and crossing(next_value)
            ),
            None,
        )
        if detected is None:
            result["status"] = "transition_not_detected"
        else:
            result["status"] = "measured"
            result["screen_to_edge_ms"] = round((detected - at) / 1_000_000, 1)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True, help="local IPv4 address of the Tapo")
    parser.add_argument("--stream", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--ffmpeg-nobuffer",
        action="store_true",
        help="experimental FFmpeg no-buffer option; keeps RTSP over TCP",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("the screen-flash probe requires macOS")
    if args.output.exists():
        parser.error("output file must be new")
    trace_path = args.output.with_suffix(".trace.jsonl")
    if trace_path.exists():
        parser.error("brightness trace file must be new")
    username = os.environ.get("CARE_VISION_RTSP_USERNAME") or input("Camera account username: ")
    password = os.environ.get("CARE_VISION_RTSP_PASSWORD") or getpass.getpass(
        "Camera account password: "
    )
    print(
        "Point the Tapo at the Mac display so it fills most of the camera view. "
        "Press Enter when ready. The screen will flash four times; Esc cancels."
    )
    input()
    source = RTSPSource(
        args.ip,
        username,
        password,
        stream=args.stream,
        seconds=55,
        fps=25,
        width=1280,
        height=720,
        ffmpeg_nobuffer=args.ffmpeg_nobuffer,
    )
    del password
    root = tk.Tk()
    root.title("CARE-Vision camera latency probe")
    root.attributes("-fullscreen", True)
    root.configure(bg="black")
    root.lift()
    root.focus_force()
    label = tk.Label(
        root, text="Waiting for camera…", bg="black", fg="gray", font=("Helvetica", 20)
    )
    label.pack(anchor="sw", side="bottom", padx=24, pady=24)
    samples = []
    flashes = []
    seen_seq = 0
    first_frame_ns = None
    cancelled = False
    error = None

    def change(color):
        root.configure(bg=color)
        label.configure(bg=color, fg="gray" if color == "black" else "#aaaaaa")
        root.update_idletasks()
        flashes.append({"color": color, "mono_ns": time.monotonic_ns()})

    def finish():
        root.quit()

    def cancel(_event=None):
        nonlocal cancelled
        cancelled = True
        root.quit()

    root.bind("<Escape>", cancel)

    def poll():
        nonlocal seen_seq, first_frame_ns, error
        capture, seq, done, error = source.snapshot()
        if error or done and not capture:
            root.quit()
            return
        if capture is not None and seq != seen_seq:
            seen_seq = seq
            image = cv2.imdecode(np.frombuffer(capture.jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                height, width = image.shape
                center = image[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
                whole_level = float(cv2.mean(cv2.resize(image, (80, 45)))[0])
                center_level = float(cv2.mean(cv2.resize(center, (80, 45)))[0])
                samples.append((capture.mono_ns, whole_level, center_level))
                if first_frame_ns is None:
                    first_frame_ns = capture.mono_ns
                    label.configure(text="Camera connected. Keep the camera and Mac still.")
                    for at, color in zip(FLASH_SECONDS, ("white", "black", "white", "black")):
                        root.after(at * 1000, lambda color=color: change(color))
                    root.after(TOTAL_SECONDS * 1000, finish)
        root.after(10, poll)

    try:
        source.start()
        root.after(10, poll)
        root.mainloop()
    finally:
        source.close()
        root.destroy()
    if error:
        raise RuntimeError(error)
    if cancelled:
        print("Probe cancelled; no result written.")
        return
    if len(flashes) != 4:
        raise RuntimeError("screen-flash sequence was incomplete")
    whole_transitions = analyze([(at, whole) for at, whole, _ in samples], flashes)
    center_transitions = analyze([(at, center) for at, _, center in samples], flashes)
    center_count = sum(r["status"] == "measured" for r in center_transitions)
    selected_signal = "center" if center_count >= 3 else "whole_frame"
    transitions = center_transitions if selected_signal == "center" else whole_transitions
    measured = [r["screen_to_edge_ms"] for r in transitions if r["status"] == "measured"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("x") as stream:
        for at, whole, center in samples:
            stream.write(
                json.dumps(
                    {
                        "edge_read_mono_ns": at,
                        "whole_frame_brightness": round(whole, 3),
                        "center_brightness": round(center, 3),
                    }
                )
                + "\n"
            )
    report = {
        "source": "Tapo RTSP viewed by CARE-Vision Edge on this Mac",
        "method": (
            "four full-screen black/white transitions; timestamps at Tk UI update "
            "and after decoded RTSP read"
        ),
        "includes_aws": False,
        "includes_display_latency": True,
        "includes_human_reaction": False,
        "ffmpeg_capture_options": (
            RTSP_FFMPEG_NOBUFFER if args.ffmpeg_nobuffer else
            os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        ),
        "sample_count": len(samples),
        "selected_signal": selected_signal,
        "transitions": transitions,
        "whole_frame_transitions": whole_transitions,
        "center_transitions": center_transitions,
        "brightness_trace": str(trace_path.resolve()),
        "screen_to_edge_ms_median": round(statistics.median(measured), 1) if measured else None,
        "result": "measured" if len(measured) >= 3 else "inconclusive",
        "limitations": (
            "Screen presentation time is approximated by the UI update callback; "
            "this is not a hardware-synchronized camera exposure timestamp. "
            "A sufficiently large display region must be visible to the camera. "
            "Automatic exposure may invert the whole-frame brightness response."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
