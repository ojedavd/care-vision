"""Audit non-executable sensing plans on hash-frozen, existing AWS evidence."""

import argparse
import copy
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

from care_vision.cloud.pose_observation_plan import PoseObservationPlanner
from care_vision.cloud.seated_candidate import SeatContext
from care_vision.cloud.seated_pose_candidate import SeatPoseContext


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(cloud, diagnostics, geometry, hips):
    """Stream logs use actual completion order; sequential replays use snapshots.

    Replay clocks are unrelated to source-cloud clocks. Their processing+queue
    duration is only a replay service-time check, never a measured online age.
    """
    planner = PoseObservationPlanner(geometry, hips)
    separate = diagnostics is not None
    by_id = {r["observation_id"]: r for r in diagnostics or [] if r["event"] == "pose_shadow"}
    observations, active, pending, results = {}, None, set(), []
    waiting = None

    def consider(shadow, obs, ep, busy):
        if separate:
            age = (shadow.get("processing_ms", 0) + shadow["queue_wait_ms"]) / 1000
        else:
            age = (shadow["local_mono_ns"] - obs["local_mono_ns"]) / 1e9
        item = planner.consider(obs, shadow, ep, result_age_seconds=age, commands_pending=busy)
        results.append({
            **item, "capture_seq": obs["capture_seq"],
            "captured_mono_ns": obs["captured_mono_ns"],
            "pose_reason": shadow["reason"], "checked_age_seconds": age,
            "age_basis": "replay_service_time_only" if separate else "logged_completion_delay",
            "pending_commands_basis": "assumed_none_offline" if separate else "logged_send_and_ack",
        })

    def flush():
        if waiting is not None:
            consider(by_id[waiting["observation_id"]], waiting, copy.deepcopy(active), False)

    for row in cloud:
        event = row["event"]
        if event == "observation":
            if separate:
                flush()
            waiting = row
            observations[row["observation_id"]] = row
        elif event == "incident_episode":
            active = None if row["incident_status"] in {
                "recovery_observed", "posture_change_cleared",
            } else row
        elif event == "agent_action":
            pending.add(row["command_id"])
        elif event == "ack_received":
            pending.discard(row["command_id"])
        elif event == "pose_shadow" and not separate:
            consider(row, observations[row["observation_id"]], active, bool(pending))
    if separate:
        flush()
        if len(results) != len(by_id) or len(results) != len(observations):
            raise ValueError("sequential replay must cover every source observation exactly once")
    return results


def run(plan_path, output):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(plan_path.read_text())
    for filename, expected in plan["frozen_files"].items():
        if digest(root / filename) != expected:
            raise ValueError(f"frozen input/code changed: {filename}")
    output.mkdir(parents=True, exist_ok=False)
    trials, table = [], []
    for source in plan["runs"]:
        config = json.loads((root / source["config"]).read_text())
        cloud = records(root / source["cloud"])
        shadow = records(root / source["shadow"]) if source.get("shadow") else None
        results = audit(cloud, shadow, SeatContext(**config["geometry"]),
                        SeatPoseContext(**config["hips"]))
        origin = next(r["captured_mono_ns"] for r in cloud if r["event"] == "observation")
        proposals = [r for r in results if r["plan"]]
        trial = {
            "name": source["name"], "rows": len(results), "proposals": proposals,
            "reason_counts": dict(Counter(r["reason"] for r in results)),
            "timing_scope": ("offline replay only" if shadow is not None
                             else "recorded stream order"),
            "proposed_at_seconds": [(r["captured_mono_ns"] - origin) / 1e9 for r in proposals],
        }
        trials.append(trial)
        (output / (source["name"] + ".jsonl")).write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in results)
        )
        table.append("<tr><td>" + html.escape(source["name"]) + "</td><td>" + str(len(results))
                     + "</td><td>" + str(len(proposals)) + "</td><td>"
                     + ", ".join(f"{t:.3f} s" for t in trial["proposed_at_seconds"])
                     + "</td><td>" + html.escape(trial["timing_scope"]) + "</td></tr>")
    report = {
        "scope": plan["scope"], "plan_sha256": digest(plan_path), "trials": trials,
        "rows": sum(t["rows"] for t in trials),
        "proposals": sum(len(t["proposals"]) for t in trials),
        "commands_dispatched": 0, "new_inferences": 0, "new_camera_capture": False,
        "new_roi_evidence_obtained": False, "full_demo_passed": False,
        "policy": {
            "one_roi_plan_per_episode": True, "roi_seconds_after_ack": 2,
            "or_roi_frames": 4, "fresh_full_scene_required_afterwards": True,
        },
        "limitations": [
            "Previously analyzed development sources; no held-out accuracy evaluation",
            "Offline replay checks service time; online age and pending commands unknown",
            "No counterfactual detector/identity/pose/bandwidth/closure outcome inferred",
            "ROI tracking uses image coordinates; integration must handle view changes",
            "Closure ignores ROI; full-frame tracking may still be affected by ROI updates",
            "Crop cannot reveal physically occluded joints or prove seat contact",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    document = """<!doctype html><html lang="es"><meta charset="utf-8">
<title>CARE-Vision · Plan de observación por pose</title><style>
body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#142236}
table{border-collapse:collapse;width:100%}
td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}
.scope{padding:20px;background:#edf4ff;border-radius:12px}a{color:#1261ad}</style>
<h1>Qué evidencia pediría pose</h1><p class="scope">Política probada sobre registros existentes.
<b>Cero comandos enviados; no se obtuvo evidencia ROI nueva.</b> Sin cámara ni inferencia nueva.</p>
<p>Un recorte de cuerpo y contexto de silla por episodio. Ventana propuesta: 2 segundos desde ACK
o cuatro cuadros ROI; después, solicitar cuadro completo y comprobar ACK + evidencia nueva.
El recorte no verifica apoyo ni cierra el episodio.</p>
<table><tr><th>Registro</th><th>Resultados revisados</th><th>Propuestas</th><th>Instante fuente</th>
<th>Alcance temporal</th></tr>TABLE</table>
<p>Los registros secuenciales permiten revisar la regla, pero no reconstruyen la antigüedad
online de pose ni los comandos pendientes en ese momento.
Todas las fuentes ya fueron analizadas.</p>
<p>Antes de activar: resolver coordenadas de tracking al cambiar de vista, conservar identidad
de cuadro completo y verificar retorno con Edge real. Una ampliación no revela articulaciones
tapadas por muebles.</p>
<p><a href="report.json">Detalle y coordenadas de las propuestas</a></p></html>"""
    (output / "report.html").write_text(document.replace("TABLE", "".join(table)))
    print(json.dumps({k: report[k] for k in ("rows", "proposals", "commands_dispatched")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.output)
