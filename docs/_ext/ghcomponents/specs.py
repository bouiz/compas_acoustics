"""The Grasshopper components as the plugin declares them: their specs, the ribbon order and their GUIDs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PLUGIN = REPO / "src" / "compas_acoustics_ghcsharp" / "CompasAcoustics.Grasshopper"
SPECS_DIR = PLUGIN / "Components" / "Specs"
COMPONENT_BASE_CS = PLUGIN / "Components" / "ComponentBase.cs"
CSHARP_DIR = PLUGIN

ACCESS = {0: "item", 1: "list", 2: "tree"}


@dataclass
class Port:
    name: str
    description: str
    type_hint: str | None = None
    access: str = "item"
    optional: bool = False
    default: object = None
    choices: list[str] = field(default_factory=list)


@dataclass
class Component:
    key: str
    name: str
    nickname: str
    subcategory: str
    description: str
    inputs: list[Port]
    outputs: list[Port]
    icon: Path


def _port(data: dict) -> Port:
    return Port(
        name=data["name"],
        description=data.get("description", ""),
        type_hint=data.get("typeHintID"),
        access=ACCESS.get(data.get("scriptParamAccess", 0), "item"),
        optional=bool(data.get("optional", False)),
        default=data.get("default"),
        choices=[str(value[0]) for value in data.get("namedValues", [])],
    )


def load_components(specs_dir: Path) -> list[Component]:
    """Every component in ``specs_dir``, one folder each with metadata.json and icon.png, sorted by folder name."""
    components = []
    for metadata in sorted(Path(specs_dir).glob("*/metadata.json")):
        data = json.loads(metadata.read_text(encoding="utf-8"))
        ports = data.get("ports", {})
        components.append(
            Component(
                key=metadata.parent.name,
                name=data["name"],
                nickname=data.get("nickname", ""),
                subcategory=data["subcategory"],
                description=data.get("description", ""),
                inputs=[_port(p) for p in ports.get("inputParameters", [])],
                outputs=[_port(p) for p in ports.get("outputParameters", [])],
                icon=metadata.parent / "icon.png",
            )
        )
    return components


def read_subcategory_order(component_base_cs: Path) -> list[str]:
    """The ribbon order: the names in ``ComponentBase.Subcategories``."""
    text = Path(component_base_cs).read_text(encoding="utf-8")
    match = re.search(r"Subcategories\s*=\s*new\[\]\s*\{([^}]*)\}", text)
    if not match:
        raise ValueError("No Subcategories list in {}".format(component_base_cs))
    return re.findall(r'"([^"]+)"', match.group(1))


def read_component_guids(components_dir: Path) -> dict[str, str]:
    """Each component's GUID (lowercase), keyed by spec folder, from the C# classes that declare both."""
    guids = {}
    for source in sorted(Path(components_dir).rglob("*.cs")):
        text = source.read_text(encoding="utf-8")
        for block in re.split(r"\bclass\s+\w+", text)[1:]:
            key = re.search(r'SpecKey\s*=>\s*"(CA_\w+)"', block)
            guid = re.search(r'ComponentGuid\s*=>\s*new Guid\("([0-9a-fA-F-]+)"\)', block)
            if key and guid:
                guids[key.group(1)] = guid.group(1).lower()
    return guids
