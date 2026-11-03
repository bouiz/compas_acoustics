"""The examples in examples/: Python scripts and Grasshopper definitions, paired by name."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Example:
    name: str
    py: Path | None
    ghx: Path | None


def discover(examples_dir: Path) -> list[Example]:
    names = sorted({p.stem for p in Path(examples_dir).glob("*.py")} | {p.stem for p in Path(examples_dir).glob("*.ghx")})
    examples = []
    for name in names:
        py, ghx = Path(examples_dir) / (name + ".py"), Path(examples_dir) / (name + ".ghx")
        examples.append(Example(name, py if py.exists() else None, ghx if ghx.exists() else None))
    return examples


def _title_and_body(text: str) -> tuple[str, str]:
    first, _, rest = text.strip().partition("\n")
    return first.strip().rstrip("."), rest.strip()


def docstring(py: Path) -> tuple[str, str] | None:
    """A script's docstring as (title, body): its first line, without a final period, and the rest."""
    text = ast.get_docstring(ast.parse(Path(py).read_text(encoding="utf-8")))
    return _title_and_body(text) if text else None


def walkthrough(path: Path) -> tuple[str, str]:
    """A walkthrough file as (title, body): its first line and the reStructuredText after it."""
    return _title_and_body(Path(path).read_text(encoding="utf-8"))


def components_in(ghx: Path, guids: dict[str, str]) -> set[str]:
    """The components whose GUID occurs in a .ghx file."""
    text = Path(ghx).read_text(encoding="utf-8").lower()
    return {key for key, guid in guids.items() if guid in text}
