"""Summarize delivered evidence without inventing incident labels or decisions."""

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def analyze(directory):
    summary = json.loads((directory / "summary.json").read_text())
    observations = [
        json.loads(line)
        for line in (directory / "cloud.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "observation"
    ]
    sent = [json.loads(line) for line in (directory / "sent.jsonl").read_text().splitlines()]
    if len(observations) != summary["processed"] or len(sent) != summary["selected"]:
        raise ValueError("summary/log count mismatch")
    if [r["frame_id"] for r in sent] != [r["frame_id"] for r in observations]:
        raise ValueError("frame identity mismatch")
    if [r["sha256"] for r in sent] != [r["jpeg_sha256"] for r in observations]:
        raise ValueError("evidence integrity mismatch")
    first = observations[0]
    source_start = first["captured_mono_ns"]
    receipt_start = first["received_cloud_mono_ns"]
    receipt_drift = [
        (r["received_cloud_mono_ns"] - receipt_start - r["captured_mono_ns"] + source_start) / 1e6
        for r in observations
    ]
    elapsed = (observations[-1]["captured_mono_ns"] - source_start) / 1e9
    primary = [r.get("primary_track_id") for r in observations]
    tracks = set(t for t in primary if t)
    gaps = []
    gap_start = None
    for r in observations:
        if not r.get("confirmed_person_detected") and gap_start is None:
            gap_start = r["captured_mono_ns"]
        elif r.get("confirmed_person_detected") and gap_start is not None:
            gaps.append((r["captured_mono_ns"] - gap_start) / 1e9)
            gap_start = None
    if gap_start is not None:
        gaps.append((observations[-1]["captured_mono_ns"] - gap_start) / 1e9)
    phases = []
    protocol = json.loads((directory.parent / "protocol.json").read_text())
    scenario = protocol["capture_manifest"].get("scenario", "natural")
    intervals = [
        (0, 15, "initial_quiet"),
        (15, 30, "walk"),
        (30, 45, "sit"),
        (45, 55, "stand"),
        (55, 65, "walk_again"),
        (65, 90, "final_quiet"),
    ]
    if scenario == "controlled_episode":
        intervals = [
            (0, 15, "initial_quiet"),
            (15, 30, "walk"),
            (30, 40, "prepare"),
            (40, 60, "controlled_descent"),
            (60, 85, "floor_rest"),
            (85, 105, "rise"),
            (105, 120, "final_quiet"),
        ]
    for a, b, label in intervals:
        rows = [r for r in observations if a <= (r["captured_mono_ns"] - source_start) / 1e9 < b]
        scores = [r["motion_score"] for r in rows if r.get("motion_score") is not None]
        phases.append(
            {
                "planned_cue_interval": label,
                "from_seconds": a,
                "to_seconds": b,
                "observations": len(rows),
                "person_detected": sum(r.get("person_detected", False) for r in rows),
                "confirmed_person": sum(r.get("confirmed_person_detected", False) for r in rows),
                "motion_score_max": max(scores, default=None),
                "motion_score_median": statistics.median(scores) if scores else None,
            }
        )
    return {
        "fps": summary["rate"],
        "scenario": scenario,
        "opencv_num_threads": summary["runtime"]["runtime"].get("opencv_num_threads"),
        "selected": len(sent),
        "processed": len(observations),
        "integrity_verified": True,
        "person_detection_fraction": sum(r.get("person_detected", False) for r in observations)
        / len(observations),
        "confirmed_person_fraction": sum(
            r.get("confirmed_person_detected", False) for r in observations
        )
        / len(observations),
        "primary_track_ids": sorted(tracks),
        "primary_track_count": len(tracks),
        "unconfirmed_gap_seconds_max": max(gaps, default=0),
        "jpeg_payload_bytes": summary["jpeg_payload_bytes"],
        "processing_ms_median": summary["processing_ms_median"],
        "processing_ms_p95": percentile([r["processing_ms"] for r in observations], 0.95),
        "source_selected_fps": (len(observations) - 1) / elapsed if elapsed else None,
        "cloud_receipt_drift_ms_max_relative_to_first": max(receipt_drift),
        "cloud_receipt_drift_ms_p95_relative_to_first": percentile(receipt_drift, 0.95),
        "send_lateness_ms_max": summary["send_lateness_ms_max"],
        "write_block_ms_max": summary["write_block_ms_max"],
        "phases_by_planned_cues": phases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    summaries = json.loads((args.replay / "summary.json").read_text())
    protocol = json.loads((args.replay / "protocol.json").read_text())
    width, height = protocol["resolution"]
    metrics = [analyze(args.replay / f"fps-{s['rate']}") for s in summaries]
    report = {
        "conditions": metrics,
        "limitations": [
            "One scene, one replay per rate, source already reviewed for development",
            "Detection fractions are sample coverage, not precision or recall",
            "Track IDs and gaps require visual review to interpret fragmentation",
            "Intervals are spoken-cue schedules, not independent action annotations",
            "Receipt drift compares relative clocks; excludes absolute WAN and optical latency",
            "SSH/base64 transport and MJPEG recording differ from the production live loop",
            "No adaptive agent, ROI, history or incident decision in this fixed-rate experiment",
        ],
    }
    if any(r["opencv_num_threads"] != 1 for r in metrics):
        report["limitations"].append(
            "Worker did not fix/log one OpenCV thread; production server sets one thread; "
            "processing times are not directly comparable to production live runs"
        )
    (args.replay / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Tapo comparación de evidencia a tasas fijas",
        "",
        f"Misma fuente densa y JPEG full-frame {width}×{height}, YOLOX 416/0,4 en Graviton con "
        "OpenCV 5 + COOL. No hay decisiones del agente ni evaluación de incidentes.",
        "",
        "| FPS | Procesados / seleccionados | Detección por muestra | Confirmación por muestra "
        "| IDs primarios | JPEG MiB | Proceso mediana / p95 ms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in metrics:
        lines.append(
            f"| {r['fps']} | {r['processed']}/{r['selected']} | "
            f"{100 * r['person_detection_fraction']:.1f}% | "
            f"{100 * r['confirmed_person_fraction']:.1f}% | {r['primary_track_count']} | "
            f"{r['jpeg_payload_bytes'] / 1048576:.2f} | "
            f"{r['processing_ms_median']:.2f}/{r['processing_ms_p95']:.2f} |"
        )
    lines += [
        "",
        "## Alcance",
        "",
        *[f"- {s}" for s in report["limitations"]],
        "",
        "Los logs preservan timestamps de fuente, envío y recepción/procesamiento cloud. "
        "Mayor número de observaciones no demuestra mejor detección de incidentes. "
        "Los límites de continuidad y drift están en `analysis.json`; entrega completa "
        "no demuestra ausencia de acumulación ni identifica todas las acciones.",
        "",
    ]
    (args.replay / "comparison.md").write_text("\n".join(lines))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
