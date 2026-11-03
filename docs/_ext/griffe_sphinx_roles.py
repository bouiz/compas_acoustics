"""A griffe extension that turns the Sphinx roles of the docstrings into mkdocstrings cross-references.

``:class:`~compas_acoustics.Scene``` becomes ``[`Scene`][compas_acoustics.Scene]``; the docstrings keep the Sphinx
syntax that editors and ``help()`` show. A role whose target is not on the API page (a private or undocumented
member, a module function) becomes plain code, as Sphinx showed it, instead of a broken link.
"""

import re
from pathlib import Path

import griffe

ROLE = re.compile(r":(?:py:)?(?:class|meth|func|obj|attr|mod|data|exc):`(~?)([\w.]+)(?:\(\))?`")
API_PAGE = Path(__file__).parents[1] / "api" / "compas_acoustics.md"


class SphinxRoles(griffe.Extension):
    def on_package(self, *, pkg, loader, **kwargs):
        if pkg.name != "compas_acoustics":
            return
        roots = [pkg[name].final_target.path for name in re.findall(r"^::: compas_acoustics\.(\w+)$", API_PAGE.read_text(encoding="utf-8"), re.M)]
        for obj in _walk(pkg):
            if obj.docstring is not None:
                obj.docstring.value = ROLE.sub(lambda m: _reference(m, obj, loader.modules_collection, roots), obj.docstring.value)


def _walk(obj):
    yield obj
    for member in obj.members.values():
        if not member.is_alias:
            yield from _walk(member)


def _reference(match, obj, collection, roots):
    short, target = match.groups()
    text = target.rsplit(".", 1)[-1] if short else target
    if not target.startswith("compas_acoustics"):
        first, _, rest = target.partition(".")
        try:
            target = ".".join(filter(None, (obj.resolve(first), rest)))
        except (griffe.NameResolutionError, KeyError, ValueError):
            pass
    if not target.startswith("compas_acoustics"):
        return "[`{}`][{}]".format(text, target)  # other packages: the inventories of mkdocs.yml
    try:
        found = collection[target]
        path = found.canonical_path
    except (KeyError, griffe.AliasResolutionError, griffe.CyclicAliasError):
        return "`{}`".format(text)
    for root in roots:
        rest = path[len(root) + 1 :].split(".") if path.startswith(root + ".") else [] if path == root else None
        if rest is not None and not any(part.startswith("_") for part in rest) and (found.docstring or not rest):
            return "[`{}`][{}]".format(text, path)
    return "`{}`".format(text)
