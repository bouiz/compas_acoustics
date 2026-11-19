"""Fingerprints of what each committed image shows, to tell when an image is out of date.

A component's fingerprint is a hash of its port names in order; an example's is a hash of its file. They are
recorded in docs/_images/fingerprints.json when the image is made, and compared with the sources at build time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .specs import REPO
from .specs import Component

IMAGES = REPO / "docs" / "_images"
FINGERPRINTS = IMAGES / "fingerprints.json"
KINDS = ("components", "py", "ghx")


@dataclass
class Stale:
    kind: str
    name: str
    reason: str


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _names(ports) -> list[str]:
    return [p.name for p in ports]


def port_fingerprint(component: Component) -> str:
    return _hash(json.dumps([_names(component.inputs), _names(component.outputs)]).encode("utf-8"))


def file_fingerprint(path: Path) -> str:
    """A hash of the file with its line endings normalised, so a CRLF checkout gives the same fingerprint."""
    return _hash(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def load(path: Path = FINGERPRINTS) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
    return {kind: data.get(kind, {}) for kind in KINDS}


def save(data: dict, path: Path = FINGERPRINTS) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record_component(data: dict, component: Component) -> None:
    data["components"][component.key] = {
        "hash": port_fingerprint(component),
        "inputs": _names(component.inputs),
        "outputs": _names(component.outputs),
    }


def record_file(data: dict, kind: str, path: Path) -> None:
    data[kind][Path(path).stem] = file_fingerprint(path)


def image_files(kind: str, name: str, image_dir: Path) -> list[Path]:
    """The files that must exist for an item's images to count as made."""
    if kind == "components":
        return [image_dir / "components" / (name + ".png")]
    if kind == "py":
        return [image_dir / "examples" / name / "stdout.txt"]
    return [image_dir / "examples" / name / "canvas.png", image_dir / "examples" / name / "viewport.png"]


def _port_changes(old: dict, component: Component) -> str:
    changes = []
    for side, ports in (("input", component.inputs), ("output", component.outputs)):
        before, now = old.get(side + "s", []), _names(ports)
        changes += ["removed {} '{}'".format(side, n) for n in before if n not in now]
        changes += ["added {} '{}'".format(side, n) for n in now if n not in before]
    return "ports changed ({})".format(", ".join(changes) if changes else "order")


def stale_components(components: list[Component], data: dict, image_dir: Path) -> list[Stale]:
    stale = []
    for c in components:
        recorded = data["components"].get(c.key)
        if recorded is None or not all(p.exists() for p in image_files("components", c.key, image_dir)):
            stale.append(Stale("components", c.key, "no image"))
        elif recorded["hash"] != port_fingerprint(c):
            stale.append(Stale("components", c.key, _port_changes(recorded, c)))
    return stale


def stale_files(kind: str, paths: list[Path], data: dict, image_dir: Path) -> list[Stale]:
    stale = []
    for path in paths:
        name = Path(path).stem
        recorded = data[kind].get(name)
        if recorded is None or not all(p.exists() for p in image_files(kind, name, image_dir)):
            stale.append(Stale(kind, name, "no image" if kind == "ghx" else "no output"))
        elif recorded != file_fingerprint(path):
            stale.append(Stale(kind, name, "file changed"))
    return stale
