"""Link actual commands, ACKs and newly processed views in recorded Edge trials."""

import argparse
import hashlib
import html
import json
from collections import Counter
from pathlib import Path


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(folder, source, name):
    cloud, edge = records(folder / "cloud.jsonl"), records(folder / "edge.jsonl")
    captures = records(source / "demo.captures.jsonl")
    times = {r["capture_seq"]: r["captured_mono_ns"] for r in captures}
    origin = captures[0]["captured_mono_ns"]
    observations = [r for r in cloud if r["event"] == "observation"
                    and r.get("evidence_kind", "live") == "live"]
    def seconds(row):
        seq = row.get("capture_seq")
        if seq is None:
            seq = int(row["frame_id"].rsplit(":", 1)[1])
        return (times[seq] - origin) / 1e9
    actions = [r for r in cloud if r["event"] == "agent_action"]
    roi_requests = [r for r in actions if r["type"] == "REQUEST_ROI"]
    full_requests = [r for r in actions if r["type"] == "REQUEST_FULL_FRAME"]
    cycles = [r for r in cloud if r["event"] == "pose_observation_cycle"]
    completed = [r for r in cycles if r["stage"] == "cycle_complete"]
    opens = [r for r in cloud if r["event"] == "incident_episode"
             and r["reason"] == "episode_opened"]
    closes = [r for r in cloud if r["event"] == "incident_episode"
              and r["reason"] == "same_track_sustained_reference_posture"]
    unresolved = [r for r in cloud if r["event"] == "incident_episode"
                  and r["reason"] == "evidence_ended_episode_open"]
    sent = [r for r in edge if r["event"] == "frame_sent"
            and r.get("evidence_kind", "live") == "live"]
    received = [r for r in cloud if r["event"] == "frame_received"
                and r.get("evidence_kind", "live") == "live"]

    def signature(row):
        roi = None
        if row.get("view_kind") == "roi":
            roi = row.get("source_roi")
            if roi is None:
                roi = [row.get(k) for k in ("roi_x", "roi_y", "roi_width", "roi_height")]
        return [row["frame_id"], row["capture_seq"], row["captured_mono_ns"],
                row["width"], row["height"], row.get("view_kind"),
                row.get("applied_visual_command_id"), row.get("visual_policy_seq"),
                row.get("source_width"), row.get("source_height"), roi]
    roi = roi_requests[0] if len(roi_requests) == 1 else None
    full = full_requests[0] if len(full_requests) == 1 else None
    def ack(command):
        return next((r for r in cloud if r["event"] == "ack_received"
                     and r["command_id"] == command["command_id"]), None) if command else None
    roi_ack, full_ack = ack(roi), ack(full)
    roi_evidence = [r for r in observations if roi
                    and r.get("applied_visual_command_id") == roi["command_id"]
                    and r["view_kind"] == "roi"]
    restored = next((r for r in observations if completed and r["observation_id"]
                     == completed[0]["observation_id"]), None)
    track = opens[0]["track_id"] if len(opens) == 1 else None
    shadow = [r for r in cloud if r["event"] == "pose_shadow"]
    offset_count = len({r["captured_mono_ns"] - times[r["capture_seq"]] for r in sent})
    checks = {
        "same_sent_received_processed_headers": [signature(r) for r in sent]
        == [signature(r) for r in received] == [signature(r) for r in observations],
        "recorded_timestamp_intervals_preserved": offset_count == 1,
        "one_roi_and_one_return": roi is not None and full is not None,
        "both_visual_acks_applied": all(a and a["result"] == "applied"
                                        for a in (roi_ack, full_ack)),
        "cycle_complete_with_roi_evidence": len(completed) == 1
        and completed[0]["roi_frames"] >= 1 and len(roi_evidence) >= completed[0]["roi_frames"],
        "restored_valid_new_full_frame": bool(restored and full_ack and restored["valid"]
            and restored["view_kind"] == "full_frame"
            and restored["applied_visual_command_id"] == full["command_id"]
            and restored["capture_seq"] >= full_ack["effective_from_capture_seq"]),
        "episode_track_retained_in_all_roi_images": bool(track and roi_evidence) and all(
            len(r["persons"]) == 1 and r["persons"][0]["track_id"] == track for r in roi_evidence),
        "episode_track_retained_on_scene_return": bool(track and restored and
            len(restored["persons"]) == 1 and restored["persons"][0]["track_id"] == track),
        "no_pose_support_or_closure": all(not r["seat_support_verified"]
                                          and not r["would_close_episode"] for r in shadow),
        "no_loss_stale_alert_or_session_error": not any(
            r["event"] in {"frame_dropped", "test_alert", "session_error", "pose_shadow_error"}
            or r.get("invalid_reason") == "processing_stale" for r in cloud),
        "edge_and_cloud_drained": any(r["event"] == "drain_complete" for r in edge)
        and any(r["event"] == "drain_confirmed" for r in cloud),
    }
    # Both sources have manually reviewed sitting from17s. The return source
    # transitions out at46.4s and is reviewed standing from48s.
    checks["no_closure_in_reviewed_seated_interval"] = not any(
        17 <= seconds(r) <= (46.4 if name == "sit-return-1" else 60) for r in closes
    )
    checks["expected_episode_outcome"] = (
        len(closes) == 1 and seconds(closes[0]) >= 48 and not unresolved
        if name == "sit-return-1" else not closes and len(unresolved) == 1
    )
    def cloud_duration(later, earlier):
        return ((later["local_mono_ns"] - earlier["local_mono_ns"]) / 1e9
                if later and earlier else None)
    duration = cloud_duration(full, roi_ack)
    trigger = next((r for r in cycles if r["stage"] == "return_requested"), None)
    checks["bounded_roi_return_requested"] = bool(trigger and duration is not None
        and duration <= 2.1 and trigger["reason"] in {
            "pose_roi_frame_budget_complete", "pose_roi_time_budget_complete",
        })
    return {
        "name": name, "checks": checks, "scoped_loop_passed": all(checks.values()),
        "sent_received_processed": [len(sent), len(received), len(observations)],
        "historical_sent_received_processed": [
            sum(r["event"] == event and r.get("evidence_kind") == "historical" for r in rows)
            for rows, event in ((edge, "frame_sent"), (cloud, "frame_received"),
                                (cloud, "observation"))
        ],
        "roi_observations": len(roi_evidence),
        "counted_roi_evidence_before_return_request": (
            completed[0]["roi_frames"] if completed else 0
        ),
        "roi_ack_to_return_request_seconds": duration,
        "return_ack_to_valid_full_frame_seconds": cloud_duration(restored, full_ack),
        "roi_requested_at_source_seconds": next((seconds(r) for r in observations if roi
            and r["observation_id"] == roi["caused_by_observation_id"]), None),
        "cycle_completed_at_source_seconds": seconds(restored) if restored else None,
        "episode_open_seconds": [seconds(r) for r in opens],
        "episode_close_seconds": [seconds(r) for r in closes],
        "episode_unresolved_at_end": bool(unresolved),
        "actions": [{k: r.get(k) for k in ("type", "command_id", "reason", "transmit_fps")}
                    for r in actions],
        "roi": [roi[k] for k in ("roi_x", "roi_y", "roi_width", "roi_height")] if roi else None,
        "pose_status": dict(Counter(r["pose_status"] for r in shadow)),
        "shadow_queue_replaced": sum(r["event"] == "pose_shadow_dropped" for r in cloud),
        "jpeg_payload_bytes": sum(r["payload_bytes"] for r in sent),
        "command_retries": sum(r["event"] == "command_retry" for r in cloud),
        "runtime": json.loads((folder / "runtime.json").read_text()),
        "limitations": ("Known development videos, geometric track identity only; "
                        "ROI runs YOLOX, pose remains full-frame only"),
    }


def audit(folder):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((folder / "plan.json").read_text())
    def frozen_hash(p):
        if p == "tools/run_pose_observation_loop.py" and (folder / "original-runner.py").exists():
            return sha(folder / "original-runner.py")
        return sha(root / p)

    unchanged = all(frozen_hash(p) == h for p, h in {
        **plan["code_sha256"], **plan["source_sha256"],
    }.items())
    trials = [summarize(folder / s["name"], root / s["folder"], s["name"])
              for s in plan["sources"]]
    report = {"scope": plan["scope"], "frozen_code_and_source_unchanged": unchanged,
              "trials": trials, "both_scoped_loops_passed": unchanged
              and all(t["scoped_loop_passed"] for t in trials), "full_demo_passed": False,
              "normal_activity_or_fall_detection_validated": False, "new_camera_capture": False}
    (folder / "assessment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    table = "".join("<tr><td>" + html.escape(t["name"]) + "</td><td>"
                    + str(t["sent_received_processed"]) + "</td><td>" + str(t["roi_observations"])
                    + "</td><td>" + str(t["episode_close_seconds"] or "Abierto al final")
                    + "</td><td>" + str(t["scoped_loop_passed"]) + "</td></tr>" for t in trials)
    document = """<!doctype html><html lang="es"><meta charset="utf-8">
<title>CARE-Vision · Ciclo de observación por pose</title><style>
body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#142236}
td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}
.scope{padding:20px;background:#edf4ff;border-radius:12px}</style>
<h1>Pose solicita detalle y recupera la escena</h1><p class="scope">Edge real reproduciendo
dos grabaciones existentes y procesando evidencia nueva en AWS Graviton + OpenCV 5 / COOL.
Integración optativa, desactivada por defecto. Sin captura nueva de cámara.</p>
<p>Pose → solicitud de ROI → ACK → imágenes nuevas procesadas por YOLOX → solicitud de retorno
→ ACK → cuadro completo nuevo y válido. Pose continúa ejecutándose sólo en cuadro completo.
El recorte no verifica apoyo ni cierra el episodio.</p>
<table><tr><th>Fuente</th><th>Enviados / recibidos / procesados</th><th>ROI procesadas</th>
<th>Cierre, segundos fuente</th><th>Ciclo acotado aprobado</th></tr>TABLE</table>
<p>Ver detalle para tiempos, ACK y chequeos. La tabla conserva fallos; no declara aprobada la
demo completa, reconocimiento de actividad normal, caídas, identidad física o red degradada.
Las fuentes ya habían sido analizadas. Volver a baja tasa no equivale a cerrar el episodio.</p>
<p><a href="assessment.json">Detalle de la auditoría</a></p></html>"""
    previews = []
    for trial in trials:
        provenance = folder / trial["name"] / "preview-provenance.json"
        if not provenance.exists():
            continue
        images = "".join(
            '<figure style="margin:0"><img style="width:100%" src="'
            + html.escape(trial["name"] + "/" + e["file"]) + '"><figcaption>'
            + html.escape(e["stage"]) + " · cuadro " + str(e["capture_seq"])
            + "</figcaption></figure>" for e in json.loads(provenance.read_text())
        )
        previews.append("<h3>" + html.escape(trial["name"]) + "</h3>"
                        + '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">'
                        + images + "</div>")
    if previews:
        document = document.replace("</html>",
            "<h2>Vistas reconstruidas con el video y render del Edge</h2>"
            "<p>Coincide el tamaño del payload registrado. No se guardó un hash del JPEG recibido; "
            "son reconstrucciones, sin inferencia local.</p>" + "".join(previews) + "</html>")
    (folder / "report.html").write_text(document.replace("TABLE", table))
    print(json.dumps({"both_scoped_loops_passed": report["both_scoped_loops_passed"],
                      "checks": {t["name"]: t["checks"] for t in trials}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    audit(parser.parse_args().folder.resolve())
