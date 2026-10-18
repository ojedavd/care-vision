"""Build local visual evidence from completed AWS output, without image inference."""

# ruff: noqa: E501 -- HTML and JS template strings
import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np

from care_vision.edge.representation import VisualPolicy

root = Path(__file__).resolve().parents[1]
capture = root / "artifacts/new-chair-validation-20260927/trial-1"
parser = argparse.ArgumentParser()
parser.add_argument("--evaluation", default="visibility-evaluation")
parser.add_argument("--output", default="visibility-cloud-dashboard-20260927")
args = parser.parse_args()
base = capture / args.evaluation
out = root / "artifacts" / args.output
out.mkdir(exist_ok=True)
old = root / "artifacts/occlusion-evidence-dashboard-20260927"
r = json.loads((base / "result.json").read_text())
v = json.loads((base / "verification.json").read_text())
manifest = json.loads((capture / "manifest.json").read_text())
images = {f["capture_seq"]: f for f in manifest["frames"]}
rows = []
assets = []
for d in ("scene", "received"):
    (out / d).mkdir(exist_ok=True)
total = 0
for row in r["rows"]:
    h = row["header"]
    f = images[h["capture_seq"]]
    original = (capture / "evidence" / f["filename"]).read_bytes()
    assert hashlib.sha256(original).hexdigest() == f["sha256"]
    roi = (
        [h["roi_" + k] for k in ("x", "y", "width", "height")] if h["view_kind"] == "roi" else None
    )
    policy = VisualPolicy(
        h["source_width"],
        h["source_height"],
        h["width"],
        h["height"],
        view_kind=h["view_kind"],
        roi=tuple(roi) if roi else None,
    )
    received, _ = policy.render(original)
    assert len(received) == h["payload_bytes"]
    name = f"{h['capture_seq']:06d}.jpg"
    (out / "received" / name).write_bytes(received)
    image = cv2.imdecode(np.frombuffer(original, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(
        str(out / "scene" / name), cv2.resize(image, (640, 360)), [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    total += len(received)
    rows.append(
        dict(
            t=row["time"],
            seq=h["capture_seq"],
            image=name,
            roi=roi,
            fps=h["target_transmit_fps"],
            bytes=len(received),
            total=total,
            visual=h["visual_policy_seq"],
            policy=h["policy_seq"],
            episodes=row["episodes"],
            people=row["people"],
            semantic_backend=row.get("semantic_backend"),
        )
    )
    assets.append(
        dict(
            image=name,
            source_sha256=f["sha256"],
            reconstructed_received_sha256=hashlib.sha256(received).hexdigest(),
        )
    )
assert total == v["jpeg_bytes"]
byframe = {x["header"]["frame_id"]: x for x in r["rows"]}
acks = {x["command_id"]: x for x in r["acks"]}
commands = [
    dict(
        t=byframe[c["based_on_frame_id"]]["time"],
        kind=c["type"],
        seq=c["command_seq"],
        fps=c.get("transmit_fps"),
        context=c.get("context_fps"),
        roi=[c["roi_" + k] for k in ("x", "y", "width", "height")]
        if c["type"] == "REQUEST_ROI"
        else None,
        ack=acks[c["command_id"]]["result"],
    )
    for c in r["commands"]
]
(out / "data.js").write_text(
    "window.EVIDENCE = "
    + json.dumps(dict(rows=rows, commands=commands), separators=(",", ":"))
    + ";\n"
)
(out / "provenance.json").write_text(
    json.dumps(
        dict(
            source_report=str((base / "result.json").relative_to(root)),
            source_report_sha256=hashlib.sha256((base / "result.json").read_bytes()).hexdigest(),
            method="Source thumbnails and received JPEG reconstruction with recorded ROI headers and "
            "the same Edge renderer; all payload lengths match. Sent JPEG hashes were not logged; "
            "byte identity is not independently proved. No image inference or network in builder.",
            assets=assets,
        ),
        indent=2,
    )
    + "\n"
)
app = (old / "app.js").read_text().replace("let cursor=28.2", "let cursor=85.5")
app = app.replace(
    "const labels={",
    "const labels={VERIFYING_VISIBILITY:'Solicitando evidencia por visibilidad incierta',",
)
app = app.replace(
    "const explanations={",
    "const explanations={VERIFYING_VISIBILITY:'El cambio leve de postura solicita ROI y más FPS. No afirma caída ni continuidad de identidad detrás de la silla.',",
)
app = app.replace(
    "const colors={",
    "const colors={VERIFYING_VISIBILITY:'#b5a0fa',UNRESOLVED:'#ffaaa4',NO_LOW_EVIDENCE:'#ffaaa4',",
)
app = app.replace(
    "La región visible de piernas y pies permanece baja después del descenso. "
    "El torso sigue oculto; el sistema no inventa su postura.",
    "Un fragmento visible permanece en la zona baja tras el cambio de postura. "
    "Es un candidato experimental: no confirma contacto con el suelo ni postura oculta.",
)
(out / "app.js").write_text(app)
if v.get("inference_summary"):
    app = app.replace(
        "r.people.length+' detecciones'",
        "r.people.length+' detecciones · '+(r.semantic_backend?.includes('yolox')?'YOLOX Nano':'NanoDet')",
    )
    (out / "app.js").write_text(app)
passed = v["status"] == "CASE_PASSED_DEVELOPMENT"
verdict = "Candidato y recuperación observados" if passed else "Episodio sin resolver"
heading = (
    "La nueva secuencia queda resuelta en este replay."
    if passed
    else "El ROI se activa; falta resolver el episodio."
)
if v["review_bracket_candidates"] and not passed:
    heading = "Candidato detectado; recuperación sin resolver."
html = (old / "index.html").read_text()
html = html.replace(
    "CARE-Vision · Evidencia de oclusión", "CARE-Vision · Resultado de activación temprana"
)
html = html.replace("Descenso, oclusión y recuperación.", heading)
html = re.sub(
    r'<p class="hero-note">.*?</p>',
    '<p class="hero-note">Evaluación nueva con transporte real Mac ↔ AWS Graviton / OpenCV 5 / COOL. Se reutilizó la grabación como caso de desarrollo y se activó la búsqueda de evidencia antes de perder la referencia. El resultado original fallido permanece guardado.</p>',
    html,
)
color = "var(--green)" if passed else "#ffaaa4"
stats = f"""<div class="stats"><div class="stat"><small>IMÁGENES RECIBIDAS</small><strong>{v["frames"]}</strong><span>{v["roi_frames"]} ROI · {v["frames"] - v["roi_frames"]} completas</span></div><div class="stat"><small>COMANDOS APLICADOS</small><strong>{v["commands_applied"]}</strong><span>ACK verificados</span></div><div class="stat"><small>JPEG ENVIADO</small><strong>{v["jpeg_bytes"] / 1e6:.2f} MB</strong><span>sin overhead de transporte</span></div><div class="stat"><small>RESULTADO DEL REPLAY</small><strong style="font-size:18px;color:{color}">{verdict}</strong></div></div>\n"""
a = html.index('<div class="stats">')
b = html.index('<div class="workspace">')
html = html[:a] + stats + html[b:]
html = (
    html.replace("00:00 / 01:30", "00:00 / 02:00")
    .replace('max="89.51"', 'max="120"')
    .replace('value="28.2"', 'value="85.5"')
)
html = re.sub(
    r'<div class="axis">.*?</div>',
    '<div class="axis"><span>0 s</span><span>20 s</span><span>40 s</span><span>60 s</span><span>80 s</span><span>100 s</span><span>120 s</span></div>',
    html,
)
buttons = [
    (23, "Sentarse"),
    (51.04, "Inclinación normal"),
    (54.1, "Volver a base"),
    (81.51, "Activación / ROI"),
    (85.5, "Oclusión"),
    (105, "Incorporación"),
    (110, "Erguido"),
]
buttons += [(ev["time"] + 0.01, ev["state"]) for ep in r["episodes"] for ev in ep["events"]]
html = re.sub(
    r'<div class="moments">.*?</div>',
    '<div class="moments">'
    + "".join(f'<button data-t="{t}">{s}</button>' for t, s in buttons)
    + "</div>",
    html,
)
events = "".join(
    f"<p class='fine'>{ep['source_track']}: "
    + ", ".join(f"{ev['state']} a {ev['time']:.2f} s" for ev in ep["events"])
    + "</p>"
    for ep in r["episodes"]
)
controls = "".join(
    f"<p class='fine'>{j['id']}: {j['candidates']} candidatos · {j['recoveries']} recuperaciones · "
    + ", ".join(j["terminal_states"])
    + "</p>"
    for j in v["controls"]
)
lower = f"""<div class="lower"><section class="panel"><h2>Estados registrados</h2>{events}<p class="fine">Las etiquetas son resultados experimentales, sin alertas. Se sigue el fragmento visible; el torso oculto no se reconstruye.</p></section><section class="panel"><h2>Comparación y límites</h2><p class="fine">Original: 241 frames, cero comandos, ningún episodio. Variante nueva: {v["frames"]} frames, {v["commands_applied"]} comandos y {v["normal_activity_candidates"]} candidatos en actividades normales anteriores a 80 s.</p><p class="fine">Replay secuencial: {v["wall_seconds"]:.2f} s para una grabación de 120 s. No demuestra rendimiento en vivo. Esta grabación es desarrollo/regresión; hace falta otra grabación reservada para validar la mejora.</p><h2>Controles complementarios en AWS</h2>{controls}<p class="fine">Usan cuadros completos y detecciones de persona previamente calculadas en AWS. No equivalen a otra prueba completa de comandos con ROI. IMG 1073 sigue sin candidato.</p></section></div>"""
if v.get("inference_summary"):
    before = json.loads((capture / "visibility-evaluation-v3/result.json").read_text())
    cards = []
    for report, title in ((before, "Antes: identidad perdida"), (r, "Ahora: recuperacion observada")):
        row = min(report["rows"], key=lambda x: abs(x["time"] - 112.005569667))
        image = cv2.imread(str(out / "received" / f"{row['header']['capture_seq']:06d}.jpg"))
        image = cv2.resize(image, (640, 360))
        for person in row["people"]:
            x, y, w, h = [round(value / 2) for value in person["bbox"]]
            cv2.rectangle(image, (x, y), (x + w, y + h), (181, 227, 131), 2)
            cv2.putText(image, "ID " + person["track_id"].split(":")[-1],
                        (x + 2, max(18, y + 18)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (181, 227, 131), 1)
        card = np.full((424, 640, 3), (27, 17, 11), np.uint8)
        card[64:] = image
        cv2.putText(card, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (252, 244, 237), 1)
        cv2.putText(card, f"{row['time']:.2f} s - {row['episodes'][-1]['state']}",
                    (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (239, 214, 102), 1)
        cards.append(card)
    cv2.imwrite(str(out / "recovery-comparison.png"), np.hstack(cards))
    lower = lower.replace(
        "Usan cuadros completos y detecciones de persona previamente calculadas en AWS.",
        "Usan cuadros completos, detecciones de AWS guardadas antes de la reasociación e inferencia nueva de YOLOX Nano durante la verificación de recuperación; NanoDet vuelve después. Los negativos no activaron el verificador.",
    )
    lower = lower.replace(
        "<h2>Comparación y límites</h2>",
        "<h2>Comparación y límites</h2><p class='fine'>YOLOX Nano se activa únicamente después de reasociar una persona erguida. Conserva la misma pista mediante el tracker existente y exige dos segundos erguido. No se unen IDs nuevos ni se baja el tiempo mínimo.</p>",
    )
    lower = "<section class='panel'><h2>La silla y el cuerpo durante la recuperación</h2><img src='recovery-comparison.png' style='width:100%;height:auto' alt='A los 112,01 segundos: antes la caja incluía la silla y cambiaba a ID 10; ahora sigue el cuerpo con ID 9 y registra recuperación.'><p class='fine'>Misma imagen de contexto, cajas y estados registrados en AWS: variante anterior a la izquierda y verificador de recuperación a la derecha.</p></section>" + lower
a = html.index('<div class="lower">')
b = html.index('<div class="links">')
html = html[:a] + lower + html[b:]
links = """<div class="links"><a href="../new-chair-validation-20260927/trial-1/dashboard/index.html">Prueba original fallida</a><a href="../new-chair-validation-20260927/trial-1/visibility-evaluation/verification.json">Resultados verificados</a><a href="provenance.json">Procedencia de imágenes</a><a href="../../docs/visibility-cloud-20260927.md">Informe técnico</a></div>"""
html = re.sub(r'<div class="links">.*?</div>', links, html)
html = html.replace("/visibility-evaluation/verification.json",
                    "/" + args.evaluation + "/verification.json")
if v.get("inference_summary"):
    html = html.replace("../../docs/visibility-cloud-20260927.md",
                        "../../docs/visibility-recovery-verifier-20260927.md")
(out / "index.html").write_text(html)
print("Built", len(rows), "frame pairs with verified payload lengths.")
