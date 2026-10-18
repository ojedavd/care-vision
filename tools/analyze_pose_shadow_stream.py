"""Audit matched JPEG inputs and summarize scoped process cost, preserving failures."""

import argparse
import hashlib
import html
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def stats(values):
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values),
            "p95": float(np.percentile(values, 95)), "max": max(values)}


def behavior_signature(rows):
    """Compare evidence-based decisions, omitting generated IDs and clock origins."""
    observed = [r for r in rows if r["event"] == "observation"]
    origin = observed[0]["captured_mono_ns"] if observed else None
    return {
        "actions": [
            [r["type"], r.get("transmit_fps"), r["reason"],
             int(r["based_on_frame_id"].rsplit(":", 1)[1])]
            for r in rows if r["event"] == "agent_action"
        ],
        "episodes": [
            [r["reason"], r["incident_status"],
             (r["captured_mono_ns"] - origin) if r.get("captured_mono_ns") else None,
             r["horizontal_observed"], r["test_alert_emitted"]]
            for r in rows if r["event"] == "incident_episode"
        ],
    }


def summarize(folder, fixed):
    rows = records(folder / "cloud.jsonl")
    observations = [r for r in rows if r["event"] == "observation"]
    received = [r for r in rows if r["event"] == "frame_received"]
    finished = [r for r in rows if r["event"] == "measurement_finished"]
    if len(finished) != 1:
        raise ValueError("measurement requires exactly one complete session")
    window = finished[0]
    shadow = [r for r in rows if r["event"] == "pose_shadow"]
    ending = [r for r in rows if r["event"] == "pose_shadow_finished"]
    if any(r["would_close_episode"] or r["seat_support_verified"] for r in shadow):
        raise ValueError("shadow support/closure invariant violated")
    if any(r["suggested_observation"] and r["suggested_observation"]["dispatched"] for r in shadow):
        raise ValueError("shadow suggestion was dispatched")
    first = observations[0]["captured_mono_ns"]
    vision = [{
        "capture_seq": r["capture_seq"], "valid": r["valid"],
        "motion_score": r["motion_score"],
        "persons": [{k: p.get(k) for k in [
            "confidence", "bbox", "tracked_bbox", "track_state", "track_hits", "association_only",
        ]} for p in r["persons"]],
    } for r in observations]
    ready = next(r for r in rows if r["event"] == "server_ready")
    result = {
        "name": folder.name, "fixed_inputs": fixed,
        "received": len(received), "processed": len(observations),
        "dropped": sum(r["event"] == "frame_dropped" for r in rows),
        "processing_stale": sum(r.get("invalid_reason") == "processing_stale"
                                for r in observations),
        "processing_ms": stats([r["processing_ms"] for r in observations]),
        "queue_wait_ms": stats([r["cloud_queue_wait_ms"] for r in observations]),
        "cpu_seconds": window["cpu_seconds"], "window_seconds": window["wall_seconds"],
        "cpu_percent_one_core": window["cpu_percent_one_core"],
        "end_rss_mib": window["rss_bytes"] / 2**20,
        "peak_rss_mib": window["peak_rss_bytes"] / 2**20,
        "cpu_seconds_per_processed_frame": window["cpu_seconds"] / len(observations),
        "shadow": {
            "events": len(shadow), "candidates": sum(r["seated_pose_candidate"] for r in shadow),
            "pose_status": dict(Counter(r["pose_status"] for r in shadow)),
            "reason_counts": dict(Counter(r["reason"] for r in shadow)),
            "processing_ms": stats([r["processing_ms"] for r in shadow if "processing_ms" in r]),
            "queue_wait_ms": stats([r["queue_wait_ms"] for r in shadow]),
            "errors": sum(r["event"] == "pose_shadow_error" for r in rows),
            "pending_replaced": sum(r["event"] == "pose_shadow_dropped" for r in rows),
            "unfinished": sum(r["unfinished"] for r in ending),
            "finished": ending,
        },
        "behavior": behavior_signature(rows),
        "open_seconds": [(r["captured_mono_ns"] - first) / 1e9 for r in rows
                         if r["event"] == "incident_episode" and r["reason"] == "episode_opened"],
        "closure_seconds": [(r["captured_mono_ns"] - first) / 1e9 for r in rows
                            if r["event"] == "incident_episode" and r["incident_status"]
                            in {"recovery_observed", "posture_change_cleared"}],
        "alerts": sum(r["event"] == "test_alert" for r in rows),
        "runtime_threads": next(
            r["opencv_num_threads"] for r in rows if r["event"] == "server_ready"
        ),
        "session_errors": [r for r in rows if r["event"] == "session_error"],
        "cloud_log_sha256": hashlib.sha256((folder / "cloud.jsonl").read_bytes()).hexdigest(),
        "vision_signature_sha256": hashlib.sha256(
            json.dumps(vision, sort_keys=True).encode()
        ).hexdigest(),
        "runtime_signature": {
            "instance": ready["runtime"]["ec2"], "threads": ready["opencv_num_threads"],
            "cool_runtime_id": ready["runtime"]["cool"]["runtime_id"],
        },
    }
    if fixed:
        client = records(folder / "client.jsonl")
        sent = [r for r in client if r["event"] == "frame_sent"]
        result.update(
            sent_fingerprint=[[r["capture_seq"], r["jpeg_sha256"]] for r in sent],
            receive_sequences=[r["capture_seq"] for r in received],
            processed_sequences=[r["capture_seq"] for r in observations],
            jpeg_bytes=sum(r["payload_bytes"] for r in sent),
            send_lateness_ms=stats([r["send_lateness_ms"] for r in sent]),
        )
    else:
        edge = records(folder / "edge.jsonl")
        sent = [r for r in edge if r["event"] == "frame_sent"]
        policies = []
        for policy in sorted({r["policy_seq"] for r in sent}):
            frames = [r for r in sent if r["policy_seq"] == policy]
            elapsed = (frames[-1]["captured_mono_ns"] - frames[0]["captured_mono_ns"]) / 1e9
            policies.append({
                "policy_seq": policy, "requested_fps": frames[0]["target_transmit_fps"],
                "frames": len(frames), "measured_fps_between_samples":
                (len(frames) - 1) / elapsed if elapsed else None,
                "applied_command_id": frames[0]["applied_command_id"],
            })
        result["edge"] = {
            "frames_sent": len(sent), "policies": policies,
            "commands_issued_by_cloud": [r for r in rows if r["event"] == "agent_action"],
            "acks": [r for r in edge if r["event"] == "command_ack"],
            "drained": any(r["event"] == "drain_complete" for r in edge),
            "source_stopped": [r for r in edge if r["event"] == "source_stopped"],
            "event_counts": dict(Counter(r["event"] for r in edge)),
        }
    return result


def matched_inputs(results):
    first = results[0]
    return all(
        r["sent_fingerprint"] == first["sent_fingerprint"]
        and r["receive_sequences"] == first["receive_sequences"]
        and r["processed_sequences"] == first["processed_sequences"]
        and r["receive_sequences"] == r["processed_sequences"]
        and r["processed_sequences"] == [s[0] for s in r["sent_fingerprint"]]
        for r in results
    )


def analyze(source):
    runs = [summarize(source / name, True) for name in ["off-1", "on-1", "on-2", "off-2"]]
    pilot = summarize(source / "edge-on-1", False)
    identical = matched_inputs(runs)
    behavior_equal = all(r["behavior"] == runs[0]["behavior"] for r in runs)
    vision_equal = all(r["vision_signature_sha256"] == runs[0]["vision_signature_sha256"]
                       for r in runs)
    runtime_equal = all(r["runtime_signature"] == runs[0]["runtime_signature"] for r in runs)
    pairs = []
    for off, on in [(runs[0], runs[1]), (runs[3], runs[2])]:
        delta = on["cpu_seconds"] - off["cpu_seconds"]
        pairs.append({
            "off": off["name"], "on": on["name"], "cpu_seconds_delta": delta,
            "cpu_percent_relative_delta": 100 * delta / off["cpu_seconds"],
            "processing_p95_ms_delta": on["processing_ms"]["p95"] - off["processing_ms"]["p95"],
            "end_rss_mib_delta": on["end_rss_mib"] - off["end_rss_mib"],
        })
    result = {
        "scope": "Exploratory ABBA with matched transmitted JPEGs; separate adaptive Edge pilot",
        "runs": runs, "paired_differences": pairs, "pilot": pilot,
        "identical_sent_received_processed_inputs": identical,
        "identical_fixed_run_controller_behavior": behavior_equal,
        "identical_fixed_run_vision_signatures": vision_equal,
        "identical_runtime_and_threads": runtime_equal,
        "cost_difference_attribution_supported": identical and runtime_equal
        and vision_equal and behavior_equal and all(
            not r["session_errors"] and not r["processing_stale"] for r in runs
        ),
        "limits": [
            "Two runs/condition, same person/room; no general accuracy or confidence interval",
            "Cloud CPU after first receive; excludes detector/Python startup before first frame",
            "CPU100% means one core, not the entire 16-core instance",
            "Fixed client applies ACKs but bypasses sampler; not adaptive actuation validation",
            "Pilot uses real RecordedSource/Edge and commands; inputs differ from fixed comparison",
            "Recorded timestamps reflect roughly15FPS, not nominal25FPS or new camera capture",
            "No fall, immobility, physical visibility, seat support or benign closure validation",
        ],
    }
    save_path = source / "assessment.json"
    if save_path.exists():
        raise ValueError("refusing to overwrite assessment")
    save_path.write_text(json.dumps(result, indent=2) + "\n")
    table = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in [
            r["name"], r["processed"], f"{r['cpu_seconds']:.3f}",
            f"{r['processing_ms']['p95']:.2f}", f"{r['end_rss_mib']:.1f}",
            r["shadow"]["pending_replaced"],
        ]) + "</tr>" for r in runs
    )
    page = f"""<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CARE-Vision · Costo de pose</title>
<style>body{{background:#10191d;color:#e4eceb;font:16px/1.6 system-ui;max-width:1100px;
margin:40px auto;padding:20px}}h1{{font-size:32px}}table{{border-collapse:collapse;width:100%}}
td,th{{text-align:left;border-bottom:1px solid #456;padding:12px}}a{{color:#8ce4cf}}
.note{{color:#b4c5c4}}.box{{background:#18252a;padding:22px;border-radius:12px}}</style>
<p>CARE-VISION / MEDICIÓN DEL 14 OCT 2026</p><h1>Cuánto añade el observador de pose</h1>
<p class="note">Mismos JPEG, mismos intervalos y mismo Graviton.
Dos ejecuciones sin pose y dos con pose,
en orden OFF–ON–ON–OFF. Fuente grabada; no hubo una toma nueva de cámara.</p>
<div class="box"><table><tr><th>Ensayo</th><th>Procesados</th><th>CPU · segundos</th>
<th>Visión p95 · ms</th><th>RAM final · MiB</th>
<th>Muestras de pose reemplazadas</th></tr>{table}</table></div>
<p>Entradas enviadas/recibidas/procesadas iguales: <b>{'sí' if identical else 'no'}</b>.
Decisiones del controlador iguales: <b>{'sí' if behavior_equal else 'no'}</b>.</p>
<p>El piloto adicional usó el Edge real reproduciendo el video y aplicando comandos:
{pilot['processed']} cuadros procesados;
{len(pilot['closure_seconds'])} cierres del controlador original.
Pose conservó apoyo sin verificar y no ejecutó propuestas.</p>
<p class="note">CPU medida para el proceso cloud desde el primer cuadro hasta finalizar la sesión,
incluido el drenaje de pose. Se excluye el arranque previo. Los ensayos fijos no cambian el muestreo
con los comandos; el piloto sí.
Dos repeticiones por condición no validan otros FPS, escenas o equipos.</p>
<p><a href="assessment.json">Medición y límites completos</a> ·
<a href="../../pose-shadow-integration-20261014/dashboard/">Panel de posturas</a></p></html>"""
    (source / "report.html").write_text(page)
    print(json.dumps({"matched_inputs": identical, "behavior_equal": behavior_equal,
                      "paired_differences": pairs}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    analyze(parser.parse_args().source)
