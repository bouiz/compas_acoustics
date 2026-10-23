"""Reads hybrid-rir at the pinned revision and rewrites it for compas_acoustics.

Used by the porting steps of the hybrid ambisonic RIR plan and kept so that an
upstream sync can be replayed: write the new revision to UPSTREAM_REVISION,
rerun the steps, and diff. Definitions are located by name through ``ast``,
never by line number, and every lookup fails loudly when upstream changed
shape. The upstream working tree is never read.
"""

import ast
import os
import pathlib
import re
import subprocess

# A local clone of gpapadimitri/hybrid-rir: $HYBRID_RIR_UPSTREAM, else the first that exists.
_CANDIDATES = [
    os.environ.get("HYBRID_RIR_UPSTREAM", ""),
    "/Users/gpapadim/Documents/Projects/2026_hybridrir/hybrid-rir",
    str(pathlib.Path(__file__).resolve().parents[3] / "hybrid-rir"),
]
UPSTREAM = next((c for c in _CANDIDATES if c and pathlib.Path(c, ".git").exists()), _CANDIDATES[1])
REVISION = pathlib.Path(__file__).with_name("UPSTREAM_REVISION").read_text().strip()


def upstream(path):
    """The text of ``path`` at the pinned revision."""
    return subprocess.run(
        ["git", "-C", UPSTREAM, "show", f"{REVISION}:{path}"],
        capture_output=True, text=True, check=True,
    ).stdout


def provenance(path):
    return f"Ported from hybrid-rir (gpapadimitri/hybrid-rir @ {REVISION}, {path})."


def once(text, old, new, count=1):
    """``text`` with ``old`` replaced, which must occur exactly ``count`` times."""
    found = text.count(old)
    if found != count:
        raise AssertionError(f"expected {count} of {old!r}, found {found}")
    return text.replace(old, new)


def _names(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = set()
        for target in targets:
            elements = target.elts if isinstance(target, ast.Tuple) else [target]
            names |= {e.id for e in elements if isinstance(e, ast.Name)}
        return names
    return set()


def _find(body, name):
    for node in body:
        if name in _names(node):
            return node
    raise AssertionError(f"{name} not found upstream")


def _body(text, cls):
    tree = ast.parse(text)
    return tree.body if cls is None else _find(tree.body, cls).body


def _span(text, node):
    """1-based (first, last) line of a node with its decorators and the comment lines right above it."""
    lines = text.splitlines(keepends=True)
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    while first > 1 and lines[first - 2].lstrip().startswith("#"):
        first -= 1
    return first, node.end_lineno


def node(text, name, cls=None):
    """The source of a top-level definition, or of a method of class ``cls``."""
    first, last = _span(text, _find(_body(text, cls), name))
    return "".join(text.splitlines(keepends=True)[first - 1:last])


def span(text, first_name, last_name):
    """The top-level source from definition ``first_name`` through ``last_name``."""
    body = _body(text, None)
    first, _ = _span(text, _find(body, first_name))
    _, last = _span(text, _find(body, last_name))
    return "".join(text.splitlines(keepends=True)[first - 1:last])


def replace_node(text, name, new, cls=None):
    """``text`` with a definition replaced by ``new``; ``""`` removes it."""
    text = once(text, node(text, name, cls), new)
    return re.sub(r"\n{4,}", "\n\n\n", text)


def replace_relative_import(text, module, new):
    """``text`` with its one ``from .module import ...`` statement replaced by ``new``."""
    hits = [
        n for n in ast.parse(text).body
        if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module == module
    ]
    if len(hits) != 1:
        raise AssertionError(f"expected one 'from .{module} import', found {len(hits)}")
    lines = text.splitlines(keepends=True)
    return "".join(lines[:hits[0].lineno - 1]) + new + "".join(lines[hits[0].end_lineno:])


def with_provenance(text, path, note=""):
    """``text`` with the provenance line, and ``note``, appended to its module docstring."""
    doc = ast.parse(text).body[0]
    if not (isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant)):
        raise AssertionError(f"{path} has no module docstring")
    lines = text.splitlines(keepends=True)
    addition = provenance(path) + ("\n" + note if note else "")
    if doc.lineno == doc.end_lineno:
        lines[doc.lineno - 1] = lines[doc.lineno - 1].rstrip()[:-3] + f'\n\n{addition}\n"""\n'
    else:
        lines[doc.end_lineno - 1:doc.end_lineno - 1] = ["\n", addition + "\n"]
    return "".join(lines)


def write(path, text):
    pathlib.Path(path).write_text(text)


# ------------------------------------------------------------------ tests --
# Where a name that upstream's tests import from hybrid_rir lives here. Names
# from a submodule other than room map to the same submodule of
# compas_acoustics.hybrid; _compat stands in for upstream's room types and
# its image source search. Upstream's synthesize has no stand-in: the ported
# tests call prepare and render_ambisonics.
COMPAT = {"Material", "MeshGeometry", "RoomSpec", "ShoeboxGeometry", "image_sources"}
TOP_LEVEL = {
    "BANDS": "compas_acoustics.hybrid.bands",
    "BAND_EDGES": "compas_acoustics.hybrid.bands",
    "N_BANDS": "compas_acoustics.hybrid.bands",
    "BANDWIDTHS": "compas_acoustics.hybrid.bands",
    "DiffuseFieldWarning": "compas_acoustics.room.diffuse",
    "mean_absorption": "compas_acoustics.room.diffuse",
    "check_bands": "compas_acoustics.hybrid.bands",
    "RoomTarget": "compas_acoustics.hybrid.target",
    "ImageSourceSet": "compas_acoustics.hybrid.early",
    "encode": "compas_acoustics.hybrid.early",
}

MULTISLOPE_MARK = (
    "import importlib.util\n\n"
    "needs_multislope = pytest.mark.skipif(\n"
    "    importlib.util.find_spec(\"multislope\") is None, reason=\"multislope is not installed\"\n"
    ")\n"
)


def _target(module, name):
    if name in COMPAT:
        return "_compat"
    if module in ("hybrid_rir", "hybrid_rir.room"):
        if name not in TOP_LEVEL:
            raise AssertionError(f"no home for {module}.{name}; extend TOP_LEVEL or drop the tests using it")
        return TOP_LEVEL[name]
    return "compas_acoustics.hybrid." + module.split(".", 1)[1]


def _rewrite_imports(text, drop_names):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    edits = []
    # Every import, also those inside test functions, keeping its indentation
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.startswith("hybrid_rir") for a in node.names):
            raise AssertionError("plain 'import hybrid_rir' is not supported; extend _rewrite_imports")
        if not (isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("hybrid_rir")):
            continue
        grouped = {}
        for alias in node.names:
            if alias.name in drop_names:
                continue
            spelled = alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
            grouped.setdefault(_target(node.module, alias.name), []).append(spelled)
        indent = " " * node.col_offset
        lines_out = []
        for module, names in grouped.items():
            line = f"{indent}from {module} import {', '.join(names)}\n"
            if len(line) > 100:
                line = f"{indent}from {module} import (\n" + "".join(f"{indent}    {n},\n" for n in names) + f"{indent})\n"
            lines_out.append(line)
        new = "".join(lines_out)
        if not new and indent:
            new = f"{indent}pass\n"
        edits.append((node.lineno, node.end_lineno, new))
    for first, last, new in sorted(edits, reverse=True):
        lines[first - 1:last] = [new]
    return "".join(lines)


def port(name, drop=(), drop_names=(), needs_multislope=(), skip=None):
    """Write upstream ``tests/<name>`` to ``tests/hybrid/<name>``.

    ``drop`` removes top-level functions (tests of features not ported);
    ``drop_names`` removes imported names only those used; ``needs_multislope``
    marks tests to skip without multislope; ``skip`` maps test names to a
    reason they are skipped here (a compatibility limit, never a port bug).
    """
    text = upstream(f"tests/{name}")
    for function in drop:
        text = replace_node(text, function, "")
    text = _rewrite_imports(text, set(drop_names))
    if needs_multislope:
        text = once(text, "\nFS = 48000.0\n", "\n" + MULTISLOPE_MARK + "\nFS = 48000.0\n")
        for test in needs_multislope:
            text = once(text, f"\ndef {test}(", f"\n@needs_multislope\ndef {test}(")
    for test, reason in (skip or {}).items():
        text = once(text, f"\ndef {test}(", f"\n@pytest.mark.skip(reason={reason!r})\ndef {test}(")
    if re.search(r"^\s*(from|import) hybrid_rir", text, re.M):
        raise AssertionError(f"{name} still imports hybrid_rir")
    first = ast.parse(text).body[0] if ast.parse(text).body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        text = with_provenance(text, "tests/" + name)
    else:
        text = f'"""{provenance("tests/" + name)}"""\n\n' + text
    write(f"tests/hybrid/{name}", text)


def port_data(name):
    """Copy upstream ``tests/data/<name>`` (binary) to ``tests/hybrid/data/<name>``."""
    data = subprocess.run(
        ["git", "-C", UPSTREAM, "show", f"{REVISION}:tests/data/{name}"],
        capture_output=True, check=True,
    ).stdout
    target = pathlib.Path("tests/hybrid/data", name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
