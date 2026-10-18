"""Build a dashboard from new measured WebSocket episode/alert trials."""

# Source HTML/JavaScript fragments retain their formatting.
# ruff: noqa: E501

import argparse
import json
import shutil
from pathlib import Path

import build_s_only_dashboard as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    out = run / "dashboard"
    out.mkdir(exist_ok=True)
    for name in ("index.html", "app.js", "extra.css"):
        shutil.copyfile(base.OUT / name, out / name)
    base.RUN, base.OUT = run, out
    base.main()
    raw = (out / "data.js").read_text()
    data = json.loads(raw[len("window.EVIDENCE = ") : -2])
    cloud = base.records(run / "cloud.jsonl")
    observations = {r["observation_id"]: r for r in cloud if r["event"] == "observation"}
    for event in (r for r in cloud if r["event"] == "incident_episode"):
        if "observation_id" not in event:
            continue
        row = observations[event["observation_id"]]
        data["events"].append(
            {
                "t": round((row["capture_seq"] - 1) / 25, 3),
                "kind": "episode",
                "reason": event["reason"],
                "incident": event["incident_status"],
                "track": event["track_id"].split(":")[-1],
                "episodeId": event["episode_id"],
                "evidenceSeconds": event["horizontal_evidence_seconds"],
            }
        )
    data["events"].sort(key=lambda e: e["t"])
    (out / "data.js").write_text(
        "window.EVIDENCE = " + json.dumps(data, ensure_ascii=False) + ";\n"
    )
    audit = json.loads((run / "episode-audit.json").read_text())
    threshold = audit["threshold_seconds"]
    html = (out / "index.html").read_text()
    html = html.replace(
        "Una observación más densa ante el descenso.", "Episodio, alerta de prueba y recuperación."
    )
    html = html.replace("Reproducción local · AWS apagado", "Ensayo nuevo · Edge ↔ Graviton + COOL")
    html = html.replace(
        '<p class="hero-note">',
        f'<p class="hero-note"><strong>Política experimental T={threshold:g} s de evidencia horizontal:</strong> ',
    )
    html = html.replace(
        "Postura horizontal</strong><span>incidente no resuelto · sin alerta",
        f"{audit['alerts']} alerta de prueba</strong><span>{audit['recoveries']} recuperación vinculada · sin notificación externa",
    )
    html = html.replace(
        '<div class="readout"><span>Tasa transmitida',
        '<div class="readout"><span>Episodio</span><b id="episode-status">—</b></div><div class="readout"><span>Alerta</span><b id="alert-status">—</b></div><div class="readout"><span>Tasa transmitida',
    )
    html = html.replace("<h2>Qué se demostró</h2>", "<h2>Qué se demostró en esta corrida</h2>")
    alert_event = next((e for e in data["events"] if e.get("reason") == "TRIGGER_ALERT_TEST"), None)
    recovery_event = next(e for e in data["events"] if e.get("incident") == "recovery_observed")
    moments = f'<button data-t="{recovery_event["t"] + 0.05}">Recuperación confirmada</button>'
    if alert_event:
        moments = f'<button data-t="{alert_event["t"] + 0.05}">Alerta de prueba</button>' + moments
    html = html.replace(
        '<button data-t="92">Incorporación</button>',
        '<button data-t="92">Incorporación</button>' + moments,
    )
    html = html.replace(
        '<div class="readout"><span>Episodio',
        f'<div class="readout"><span>Límite experimental</span><b>{threshold:g} s de evidencia</b></div><div class="readout"><span>Episodio',
    )
    html = html.replace(
        '<div class="links">',
        '<div class="links"><a href="../../recovery-before-limit/dashboard/index.html">Caso T60 · recuperación antes del límite</a><a href="../../alert-before-recovery-retry/dashboard/index.html">Caso T20 · alerta y recuperación posterior</a>',
    )

    html = html.replace(
        "El incidente permaneció <strong>sin resolver</strong>; no hubo alerta ni confirmación de caída.",
        "El episodio se conserva al volver a 2 FPS. Al superar T se registra una alerta de prueba y se sigue observando; la recuperación posterior se vincula al mismo episodio. No se notificó a un cuidador ni se confirmó una caída.",
    )
    html = html.replace('href="../lateral-retry/s-only-audit.json"', 'href="../episode-audit.json"')
    html = html.replace('href="../review.md"', 'href="../../review.md"')
    html = html.replace("video original · 119,76 s", "video original · 119,993 s capturados")
    app = (out / "app.js").read_text()
    app = app.replace(
        "UNRESOLVED:'Postura evaluada · incidente sin resolver'",
        "UNRESOLVED:'Episodio abierto · seguimiento a 2 FPS',ASSISTANCE_REQUIRED:'Asistencia requerida · alerta de prueba'",
    )
    app = app.replace(
        "UNRESOLVED:'Se observó postura horizontal, pero el sistema no determinó la causa ni emitió alerta.'",
        f"UNRESOLVED:'Episodio abierto. Se acumulan intervalos con evidencia horizontal del mismo track; T experimental={threshold:g} s. Ausencias y huecos no suman.',ASSISTANCE_REQUIRED:'Se superó el tiempo experimental de evidencia horizontal sin recuperación. Alerta registrada en AWS, sin notificación externa. El episodio sigue abierto.'",
    )
    app = app.replace(
        "function eventText(e){",
        "function eventText(e){if(e.kind==='episode')return ({episode_opened:'Episodio abierto',same_track_horizontal_observed:'Postura horizontal del mismo track',TRIGGER_ALERT_TEST:'TRIGGER_ALERT · solo prueba',same_track_sustained_reference_posture:'Recuperación sostenida · episodio cerrado'})[e.reason]||e.reason;if(e.kind==='state')return e.from+' → '+e.to;",
    )
    app = app.replace(
        "$('#explain').textContent=stateDetails[state]||'La evidencia no permite concluir un incidente.';",
        "const episodeEvents=events.filter(e=>e.kind==='episode'&&e.t<=t);const latest=episodeEvents.at(-1);const recovered=latest?.incident==='recovery_observed';const alerted=episodeEvents.some(e=>e.reason==='TRIGGER_ALERT_TEST');$('#episode-status').textContent=recovered?'CERRADO · recuperación':latest?'ABIERTO · #'+latest.track:'Sin episodio';$('#alert-status').textContent=alerted?'Emitida · solo prueba':'Sin alerta';$('#explain').textContent=recovered?'Recuperación del mismo track vinculada al episodio; se conserva la alerta previa si existió.':stateDetails[state]||'Episodio pendiente.';",
    )
    app = app.replace(
        "state==='UNRESOLVED'?'#ffd378'",
        "state==='ASSISTANCE_REQUIRED'?'#ff9286':state==='UNRESOLVED'?'#ffd378'",
    )
    app = app.replace(
        "seek(48);", "video.addEventListener('loadedmetadata',()=>seek(80),{once:true});seek(80);"
    )
    (out / "index.html").write_text(html)
    (out / "app.js").write_text(app)
    print(out / "index.html")


if __name__ == "__main__":
    main()
