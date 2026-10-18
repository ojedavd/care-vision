"""Overlay corrected controller replay on the existing measured video dashboard."""

# HTML and JavaScript fragments retain their source formatting.
# ruff: noqa: E501

import copy
import json
import shutil
from pathlib import Path

from care_vision.cloud.controller import Controller, Policy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/s-only-websocket-20261004/dashboard"
OUT = ROOT / "artifacts/episode-followup-20261004/dashboard"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("video.mp4", "received"):
        destination = OUT / name
        if not destination.exists():
            destination.symlink_to(SOURCE / name, target_is_directory=name == "received")
    for name in ("index.html", "app.js", "style.css"):
        shutil.copyfile(SOURCE / name, OUT / name)
    raw = (SOURCE / "data.js").read_text()
    data = json.loads(raw[len("window.EVIDENCE = ") : -2])
    observations = [
        r
        for r in map(
            json.loads,
            (ROOT / "artifacts/s-only-websocket-20261004/lateral-retry/cloud.jsonl")
            .read_text()
            .splitlines(),
        )
        if r["event"] == "observation"
    ]
    controller = Controller({}, Policy(experimental_s_only_5fps=True))
    events = [e for e in data["events"] if e["kind"] != "state"]
    for row, original in zip(data["rows"], observations, strict=True):
        observation = copy.deepcopy(original)
        observation["applied_command_id"] = controller.expected_command
        command = controller.observe(observation)
        for event in controller.pop_episode_events():
            events.append(
                {
                    "t": row["t"],
                    "kind": "episode",
                    "reason": event["reason"],
                    "incident": event["incident_status"],
                    "track": event["track_id"],
                    "episodeId": event["episode_id"],
                }
            )
        for event in controller.pop_transitions():
            events.append(
                {
                    "t": row["t"],
                    "kind": "state",
                    "from": event["from_state"],
                    "to": event["to_state"],
                    "reason": event["reason"],
                }
            )
        row["episode"] = copy.deepcopy(controller.episode)
        row["agentState"] = controller.state
        if command:
            controller.acknowledge(
                {
                    "command_id": command["command_id"],
                    "result": "applied",
                    "active_policy_seq": command["command_seq"],
                    "active_transmit_fps": command["transmit_fps"],
                }
            )
    data["events"] = sorted(events, key=lambda e: e["t"])
    window = [r for r in data["rows"] if 51 <= r["t"] <= 89]
    assert len(window) == 76 and sum(r["orientation"] == "horizontal" for r in window) == 74
    assert all(r["agentState"] == "UNRESOLVED" and r["episode"] for r in window)
    assert data["rows"][-1]["episode"] is None
    (OUT / "data.js").write_text(
        "window.EVIDENCE = " + json.dumps(data, ensure_ascii=False) + ";\n"
    )
    html = (OUT / "index.html").read_text()
    html = html.replace(
        "Una observación más densa ante el descenso.", "El episodio sigue abierto a 2 FPS."
    )
    html = html.replace(
        "Reproducción local · AWS apagado", "Controlador corregido · replay de observaciones"
    )
    html = html.replace(
        '<p class="hero-note">',
        '<p class="hero-note"><strong>Vista corregida:</strong> decisiones recalculadas sobre los resultados guardados del video completo; no es un nuevo ensayo WebSocket. ',
    )
    html = html.replace(
        "Postura horizontal</strong><span>incidente no resuelto · sin alerta",
        "Recuperación vinculada</strong><span>mismo track · sin alerta",
    )
    html = html.replace(
        '<div class="readout"><span>Tasa transmitida',
        '<div class="readout"><span>Episodio</span><b id="episode-status">—</b></div><div class="readout"><span>Tiempo abierto</span><b id="episode-age">—</b></div><div class="readout"><span>Track del episodio</span><b id="episode-track">—</b></div><div class="readout"><span>Horizontal 51–89 s</span><b>74 / 76 cuadros · 2 FPS</b></div><div class="readout"><span>Alerta</span><b>No habilitada</b></div><div class="readout"><span>Tasa transmitida',
    )
    html = html.replace(
        '<button data-t="92">Incorporación</button>',
        '<button data-t="80">Sigue pendiente · 80 s</button><button data-t="92">Incorporación</button><button data-t="96.12">Recuperación confirmada</button>',
    )
    html = html.replace(
        "El incidente permaneció <strong>sin resolver</strong>; no hubo alerta ni confirmación de caída.",
        "El episodio permanece <strong>abierto tras volver a 2 FPS</strong> y se cierra cuando el mismo track recupera la postura de referencia durante 2 s. Si falta esa evidencia, queda pendiente. No se definió aún cuándo alertar.",
    )
    html = html.replace(
        "<h2>Qué se demostró</h2>",
        "<h2>Qué cambia en esta vista</h2><p>Entre 51 y 89 s: 76 cuadros a 2 FPS, 74 horizontales y dos sin clasificación (56,08 y 88,60 s). El agente sigue en UNRESOLVED, con el episodio original abierto. La recuperación se vincula cerca de 96,04 s del video (96,308 s en el reloj de captura del informe).</p>",
    )
    html = html.replace(
        'href="../lateral-retry/s-only-audit.json"',
        'href="../../s-only-websocket-20261004/lateral-retry/s-only-audit.json"',
    )
    html = html.replace('href="../review.md"', 'href="review.md"').replace(
        'href="../../s-only-websocket-20261004/lateral-retry/s-only-audit.json"',
        'href="s-only-audit.json"',
    )
    shutil.copyfile(OUT.parent / "review.md", OUT / "review.md")
    shutil.copyfile(SOURCE.parent / "lateral-retry/s-only-audit.json", OUT / "s-only-audit.json")
    (OUT / "index.html").write_text(html)
    app = (OUT / "app.js").read_text()
    app = app.replace(
        "const {rows,events,duration}=window.EVIDENCE;",
        "const {rows,events,duration}=window.EVIDENCE;const recoveryTime=events.find(e=>e.kind==='episode'&&e.incident==='recovery_observed')?.t??Infinity;",
    )
    app = app.replace(
        "UNRESOLVED:'Postura evaluada · incidente sin resolver'",
        "UNRESOLVED:'Episodio abierto · seguimiento a 2 FPS'",
    )
    app = app.replace(
        "UNRESOLVED:'Se observó postura horizontal, pero el sistema no determinó la causa ni emitió alerta.'",
        "UNRESOLVED:'La verificación densa terminó; el episodio NO terminó. El agente sigue evaluando al mismo track a 2 FPS. La postura horizontal mantiene el episodio pendiente; todavía no hay una regla de alerta calibrada.'",
    )
    app = app.replace(
        "function eventText(e){",
        "function eventText(e){if(e.kind==='episode')return ({episode_opened:'Episodio abierto · conservar track',same_track_horizontal_observed:'Mismo track · postura horizontal observada',same_track_sustained_reference_posture:'Recuperación sostenida · episodio cerrado'})[e.reason]||e.reason;if(e.kind==='state')return e.from+' → '+e.to;",
    )
    app = app.replace("'UNRESOLVED → NORMAL'", "e.from+' → '+e.to")
    app = app.replace(
        "$('#explain').textContent=stateDetails[state]||'La evidencia no permite concluir un incidente.';",
        "$('#explain').textContent=r.episode?(stateDetails[state]||'Episodio abierto'):t>=recoveryTime?'Recuperación sostenida del mismo track vinculada al episodio. La transmisión permanece a 2 FPS.':stateDetails[state];$('#episode-status').textContent=r.episode?'ABIERTO · sin resolver':t>=recoveryTime?'CERRADO · recuperación':'Sin episodio';$('#episode-age').textContent=r.episode?(((r.episode.last_seen_ns||r.episode.started_ns)-r.episode.started_ns)/1e9).toFixed(1)+' s observados':'—';$('#episode-track').textContent=r.episode?'#'+r.episode.track_id.split(':').at(-1):t>=recoveryTime?'#1 · recuperado':'—';",
    )
    app = app.replace(
        "seek(48);", "video.addEventListener('loadedmetadata',()=>seek(60),{once:true});seek(60);"
    )
    (OUT / "app.js").write_text(app)
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "scope": "Corrected controller replay over 260 saved observations; original measured media reused",
                "source": str(SOURCE),
                "time": "Video=(capture_seq-1)/25; capture-clock times in review differ slightly",
                "validation": "51–89 s: 76 rows, 74 horizontal, all UNRESOLVED with open episode; final recovery",
            },
            indent=2,
        )
        + "\n"
    )
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
