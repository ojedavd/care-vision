"""Create a four-minute local narration over saved evidence; never opens camera/AWS."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from check_mvp_demo_profile import DEFAULT, ROOT, check, digest

SOURCE = ROOT / "artifacts/mvp-classic-dashboard-20261016"
SEGMENTS = [
    ("Arquitectura y alcance",
     "CARE Vision decide qué observar a continuación. La captura ocurre en el Edge, "
     "pero la visión principal y el agente se ejecutan en AWS Graviton, con Open CV cinco "
     "y COOL. Este ensayo muestra resultados guardados de cinco casos conocidos. "
     "No estamos conectados a la cámara ni ejecutando inferencias nuevas. "
     "La transmisión cambia entre dos y cinco cuadros por segundo."),
    ("La incertidumbre provoca una orden",
     "En este caso, la persona se sienta. El patrón de pose no alcanza para cerrar "
     "la sospecha. El agente solicita una región de interés. La orden tiene un cuadro "
     "causal, una confirmación de aplicación en el Edge y evidencia posterior. "
     "Al avanzar, el rectángulo ámbar delimita la región enviada. "
     "La imagen de fondo sigue siendo la escena completa reconstruida del video."),
    ("Nueva percepción y retorno",
     "Después de procesar la evidencia de la región, el agente solicita volver al "
     "cuadro completo. La confirmación de la orden por sí sola no basta: debe llegar "
     "una observación nueva y válida. Aquí el ciclo se completa a los veintidós segundos "
     "de la fuente. La pose continúa trabajando sobre cuadro completo. "
     "El ciclo conserva el episodio abierto; todavía no demuestra recuperación."),
    ("Volver de pie permite cerrar",
     "Más adelante, la persona vuelve de pie. El controlador observa un retorno "
     "sostenido a la referencia de la misma trayectoria y cierra el episodio. "
     "En este replay ocurre cerca de los cincuenta y seis segundos. "
     "Ese cierre se explica por la evidencia temporal de retorno, "
     "no por reconocer como benigno el período sentado. "
     "La transmisión ya podía haber bajado antes."),
    ("Permanecer sentado conserva la sospecha",
     "Ahora mostramos otra grabación. La persona permanece sentada hasta el final. "
     "El ciclo de región y retorno también se completó, pero el episodio sigue "
     "sin resolver, a dos cuadros por segundo. Esta distinción es importante: "
     "reducir transmisión no equivale a declarar normalidad. "
     "El cierre automático sentado queda fuera de este MVP, "
     "y no se modificaron umbrales para aprobar esta escena."),
    ("Evidencia horizontal y recuperación",
     "Pasamos a un caso distinto, con descenso controlado al suelo y recuperación. "
     "El sistema conserva evidencia horizontal de la trayectoria. Después observa "
     "el retorno sostenido a la referencia y registra recuperación, cerca de los "
     "treinta y tres segundos. Esta grabación no prueba detección de caídas "
     "accidentales. Tampoco contiene el ciclo de región mostrado antes: "
     "son pruebas separadas y se presentan como tales."),
    ("Alerta exclusivamente de prueba",
     "En la grabación sin recuperación, el episodio permanece abierto. "
     "Esta variante habilita una alerta exclusivamente de prueba después de acumular "
     "veinte segundos de evidencia horizontal. Se registra una sola alerta cerca "
     "de los treinta y nueve segundos del video. No se envió una notificación. "
     "La variante con alertas desactivadas también conserva el episodio abierto, "
     "sin cerrarlo por falta de movimiento."),
    ("Mediciones y límites",
     "Los cinco replays procesaron los setecientos cincuenta y nueve cuadros enviados. "
     "El motor CLASSIC, dentro de Open CV cinco y COOL, logró medianas de unos "
     "setenta y cinco milisegundos, con mayor uso de memoria. "
     "Son resultados de estas grabaciones, no precisión general ni ahorro de factura. "
     "El MVP demuestra el circuito de percepción activa y conserva sus límites. "
     "Aquí termina el ensayo."),
]

# Panel cuts are selections of stored observations, not a new continuous trajectory.
CUES = [
    (0, "sit-return-live-2-pose", "Inicio", 0),
    (30, "sit-return-live-2-pose", "Solicita ROI", 1),
    (48, "sit-return-live-2-pose", "ROI nueva válida", 1),
    (60, "sit-return-live-2-pose", "Full-frame nuevo válido", 2),
    (90, "sit-return-live-2-pose", "Retorno de pie · cierre", 3),
    (120, "sit-hold-live-1-pose", "Final", 4),
    (150, "recovery-2-baseline", "Evidencia horizontal", 5),
    (167, "recovery-2-baseline", "Retorno de pie · cierre", 5),
    (180, "no-recovery-1-test20", "Alerta de prueba", 6),
    (210, "no-recovery-1-baseline", "Final", 7),
]

CONTROLS = """
<section id="rehearsal" class="panel" style="position:sticky;top:0;z-index:5">
<div class="controls" style="margin:0"><button id="narrate-play">Iniciar ensayo narrado</button>
<button id="narrate-reset">Reiniciar ensayo</button>
<strong id="narrate-time">0:00 / 4:00</strong><span id="narrate-status">Preparado</span></div>
<input id="narrate-seek" type="range" min="0" max="240" value="0" step="1"
 aria-label="Tiempo del ensayo narrado">
<div><strong id="narrate-title"></strong> · <span id="narrate-moment"></span></div>
<details><summary>Texto de este tramo</summary><p id="narrate-text" class="fine"></p></details>
<p class="fine" style="margin:4px 0 0">Voz sintética local. Saltos entre hitos y casos
 independientes; no es un video continuo. Pausar habilita la exploración manual.</p>
<audio id="narration" src="narration.m4a" preload="auto"></audio>
<p id="narrate-error" role="alert" class="error"></p></section>
"""


def build(output):
    assert check(DEFAULT)["passed"]
    source_provenance = json.loads((SOURCE / "provenance.json").read_text())
    for name, checksum in source_provenance["files_sha256"].items():
        assert digest(SOURCE / name) == checksum, name
    output.mkdir(parents=True, exist_ok=False)
    data = json.loads((SOURCE / "data.js").read_text()
                      .removeprefix("window.CLASSIC_DEMO=").removesuffix(";\n"))
    for trial in data["trials"]:
        shutil.copytree(SOURCE / trial["key"], output / trial["key"])
    for name in ("data.js", "assessment.json", "profile.json"):
        shutil.copyfile(SOURCE / name, output / name)
    cues = []
    for time, key, moment, segment in CUES:
        trial_index = next(i for i, t in enumerate(data["trials"]) if t["key"] == key)
        trial = data["trials"][trial_index]
        seq = next(m["seq"] for m in trial["moments"] if m["label"] == moment)
        row = next(r for r in trial["rows"] if r["seq"] == seq)
        cues.append({"time": time, "trial": trial_index, "seq": seq, "moment": moment,
                     "sourceTime": row["t"], "source": key,
                     "title": SEGMENTS[segment][0], "text": SEGMENTS[segment][1]})
    (output / "narration-cues.js").write_text(
        "window.NARRATION_CUES=" + json.dumps(cues, ensure_ascii=False) + ";\n")
    segments = []
    for i, (title, text) in enumerate(SEGMENTS):
        speech = output / f"segment-{i}.txt"
        speech.write_text(text + "\n")
        raw, padded = output / f"segment-{i}.aiff", output / f"segment-{i}.wav"
        subprocess.run(["/usr/bin/say", "-v", "Paulina", "-r", "155", "-f", str(speech),
                        "-o", str(raw)], check=True)
        duration = float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", str(raw)], text=True))
        if duration > 29.5:
            raise ValueError(f"Segment {i} too long: {duration}; refuse to cut speech")
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(raw), "-af",
                        "apad=whole_dur=30,atrim=duration=30", "-ar", "24000", "-ac", "1",
                        str(padded)], check=True)
        segments.append({"start": i * 30, "title": title, "text": text,
                         "speech_seconds": duration, "slot_seconds": 30})
    concat = output / "concat.txt"
    concat.write_text("".join(f"file 'segment-{i}.wav'\n" for i in range(8)))
    subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i",
                    str(concat), "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
                    str(output / "narration.m4a")], check=True)
    html = (SOURCE / "index.html").read_text().replace("</header>", "</header>" + CONTROLS)
    html = html.replace("</body>", '<script src="narration-cues.js"></script>'
                        '<script src="narration.js"></script></body>')
    (output / "index.html").write_text(html)
    shutil.copyfile(Path(__file__).with_name("mvp_narration.js"), output / "narration.js")
    (output / "script.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2) + "\n")
    manifest = {"scope": "Local synthetic narration and stored observation cuts; no inference",
                "voice": "Paulina es_MX, macOS say, rate155", "duration_seconds": 240,
                "cues": cues, "segments": segments,
                "source_panel_provenance_sha256": digest(SOURCE / "provenance.json"),
                "builder_sha256": digest(Path(__file__)),
                "files_sha256": {str(p.relative_to(output)): digest(p)
                                 for p in sorted(output.rglob("*")) if p.is_file()}}
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "speech_seconds":
                      [s["speech_seconds"] for s in segments], "duration": 240}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output.resolve())
