"""Generated pages of the documentation: the Grasshopper component reference and the examples.

``generate()`` writes them into docs/grasshopper/components/, docs/grasshopper/examples/ and docs/python/examples/
(all gitignored) and checks that the committed images in docs/_images/ match their sources. ``invoke docs``,
``invoke docs-components`` and Sphinx (``builder-inited``) all call it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import checks
from . import render
from .examples import components_in
from .examples import discover
from .examples import docstring
from .examples import walkthrough
from .fingerprints import load
from .fingerprints import stale_components
from .fingerprints import stale_files
from .specs import COMPONENT_BASE_CS
from .specs import CSHARP_DIR
from .specs import REPO
from .specs import SPECS_DIR
from .specs import load_components
from .specs import read_component_guids
from .specs import read_subcategory_order


class DocsCheckError(Exception):
    """The documentation's images or sources are out of step; the message says what and how to fix it."""


@dataclass
class Paths:
    specs: Path
    component_base_cs: Path
    csharp: Path
    examples: Path
    docs: Path
    images: Path
    equivalents: Path
    notes: Path
    walkthroughs: Path

    @classmethod
    def default(cls) -> "Paths":
        docs = REPO / "docs"
        return cls(
            specs=SPECS_DIR,
            component_base_cs=COMPONENT_BASE_CS,
            csharp=CSHARP_DIR,
            examples=REPO / "examples",
            docs=docs,
            images=docs / "_images",
            equivalents=docs / "grasshopper" / "python_equivalents.json",
            notes=docs / "grasshopper" / "notes",
            walkthroughs=docs / "grasshopper" / "walkthroughs",
        )


class _Writer:
    """Writes files only when their content changed, and remembers what it wrote."""

    def __init__(self):
        self.written = []

    def bytes(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
        self.written.append(path)

    def text(self, path: Path, text: str) -> None:
        self.bytes(path, text.encode("utf-8"))


def _site_path(paths: Paths, path: Path) -> str:
    """A path as Sphinx reads it from any page: relative to docs/, with a leading slash."""
    return "/" + Path(path).relative_to(paths.docs).as_posix()


def _rel(path: Path) -> str:
    try:
        return Path(path).relative_to(REPO).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _numbered(folder: Path, prefix: str) -> list[Path]:
    """``<prefix>_<n>.png`` in ``folder``, by number."""
    return sorted(folder.glob(prefix + "_*.png"), key=lambda f: int(f.stem[len(prefix) + 1 :]))


def generate(paths: Paths | None = None, *, check: bool = True) -> list[Path]:
    """Write the generated pages; with ``check``, raise DocsCheckError listing every problem found."""
    paths = paths or Paths.default()
    components = load_components(paths.specs)
    order = read_subcategory_order(paths.component_base_cs)
    guids = read_component_guids(paths.csharp)
    examples = discover(paths.examples)
    equivalents = json.loads(paths.equivalents.read_text(encoding="utf-8")) if paths.equivalents.exists() else {}
    fingerprints = load(paths.images / "fingerprints.json")

    gh_components = paths.docs / "grasshopper" / "components"
    gh_examples = paths.docs / "grasshopper" / "examples"
    py_examples = paths.docs / "python" / "examples"
    out = _Writer()

    used_in = {}
    for ex in examples:
        if ex.ghx is not None:
            for key in components_in(ex.ghx, guids):
                used_in.setdefault(key, []).append(ex.name)

    for c in components:
        large = paths.images / "icons" / (c.key + ".png")  # tools/icons.py draws it at four times the size
        out.bytes(gh_components / "_icons" / (c.key + ".png"), (large if large.exists() else c.icon).read_bytes())
        notes = paths.notes / (c.key + ".rst")
        out.text(
            gh_components / (c.key + ".rst"),
            render.component_page(
                c,
                image=_site_path(paths, paths.images / "components" / (c.key + ".png")),
                icon="_icons/{}.png".format(c.key),
                notes=Path(_site_path(paths, notes)) if notes.exists() else None,
                used_in=sorted(used_in.get(c.key, [])),
                python=equivalents.get(c.key),
            ),
        )
    out.text(gh_components / "index.rst", render.overview_page(components, order, lambda c: "_icons/{}.png".format(c.key)))

    missing_docstrings, missing_walkthroughs = [], []
    gh_index, py_index = [], []
    for ex in examples:
        images = paths.images / "examples" / ex.name
        if ex.ghx is not None:
            out.bytes(gh_examples / "_downloads" / (ex.name + ".ghx"), ex.ghx.read_bytes())
            notes = paths.walkthroughs / (ex.name + ".rst")
            if notes.exists():
                out.text(
                    gh_examples / (ex.name + ".rst"),
                    render.ghx_example_page(ex, notes, "_downloads/{}.ghx".format(ex.name), _site_path(paths, images / "canvas.png"), _site_path(paths, images / "viewport.png")),
                )
                gh_index.append((ex.name, walkthrough(notes)[0]))
            else:
                missing_walkthroughs.append(ex.name)
        if ex.py is not None:
            doc = docstring(ex.py)
            if doc is None:
                missing_docstrings.append(ex.name)
                continue
            figures = [_site_path(paths, f) for f in _numbered(images, "figure") + _numbered(images, "viewer")]
            stdout = (images / "stdout.txt").read_text(encoding="utf-8") if (images / "stdout.txt").exists() else None
            out.text(py_examples / (ex.name + ".rst"), render.python_example_page(ex, doc[0], doc[1], figures, stdout))
            py_index.append((ex.name, doc[0]))
    out.text(gh_examples / "index.rst", render.examples_index("Grasshopper examples", gh_index))
    out.text(py_examples / "index.rst", render.examples_index("Python examples", py_index))

    for folder in (gh_components, gh_examples, py_examples):
        for path in folder.rglob("*"):
            if path.is_file() and path not in out.written:
                path.unlink()

    if check:
        report = checks.run(
            components=stale_components(components, fingerprints, paths.images),
            ghx=stale_files("ghx", [e.ghx for e in examples if e.ghx], fingerprints, paths.images),
            py=stale_files("py", [e.py for e in examples if e.py], fingerprints, paths.images),
            unknown_subcategories=[(c.key, c.subcategory) for c in components if c.subcategory not in order],
            missing_equivalents=[c.key for c in components if c.key not in equivalents],
            missing_docstrings=missing_docstrings,
            missing_walkthroughs=missing_walkthroughs,
            walkthroughs_dir=_rel(paths.walkthroughs),
        )
        if report:
            raise DocsCheckError("The documentation is out of step with its sources.\n\n" + "\n\n".join(report))
    return out.written


def _on_builder_inited(app) -> None:
    from sphinx.errors import ExtensionError

    try:
        generate()
    except DocsCheckError as error:
        raise ExtensionError(str(error)) from None


def setup(app):
    app.connect("builder-inited", _on_builder_inited)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
