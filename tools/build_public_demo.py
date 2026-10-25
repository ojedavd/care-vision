"""Build the data-free public presentation assets; never read private episode inputs."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src/care_vision/public_demo"


def build(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    site = destination / "site"
    site.mkdir()
    for name in ("index.html", "style.css"):
        (site / name).write_bytes((SOURCE / name).read_bytes())
    (site / "app.js").write_text(
        (SOURCE / "caregiver.js").read_text() + "\n" + (SOURCE / "walkthrough.js").read_text()
    )
    manifest = {"files": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                   "bytes": p.stat().st_size} for p in sorted(site.iterdir())}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path,
                        help="New output directory, outside the source tree")
    print(json.dumps(build(parser.parse_args().destination)))
