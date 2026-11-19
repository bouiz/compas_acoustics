"""The build checks: each failing check is one section of the report, with what is wrong and how to fix it."""

from __future__ import annotations

from .fingerprints import Stale

GUIDE = "docs/contributing/docs.md"

CAPTURE = [
    "Install the current plugin in Rhino 8 (invoke yakerize, then install the .yak of dist/yak_package).",
    "Open Rhino 8 and Grasshopper.",
    "Open docs/_tools/capture_grasshopper.py in Rhino's ScriptEditor and run it.",
    "Commit docs/_images/ (the PNGs and fingerprints.json).",
]


def section(heading: str, items: list[tuple[str, str]], fix: list[str], anchor: str) -> str:
    width = max(len(name) for name, _ in items)
    lines = [heading]
    lines += ["  {}   {}".format(name.ljust(width), reason) for name, reason in items]
    lines += ["", "To fix:"]
    if len(fix) == 1:
        lines.append("  " + fix[0])
    else:
        lines += ["  {}. {}".format(i, step) for i, step in enumerate(fix, 1)]
    lines.append("See {}#{}.".format(GUIDE, anchor))
    return "\n".join(lines)


def run(
    *,
    components: list[Stale],
    ghx: list[Stale],
    py: list[Stale],
    unknown_subcategories: list[tuple[str, str]],
    missing_equivalents: list[str],
    missing_docstrings: list[str],
    missing_walkthroughs: list[str],
    walkthroughs_dir: str,
) -> list[str]:
    report = []
    if components:
        items = [(s.name, "no screenshot" if s.reason == "no image" else s.reason) for s in components]
        report.append(section("Grasshopper component screenshots are out of date:", items, CAPTURE, "screenshots"))
    if ghx:
        report.append(section("Grasshopper example images are out of date:", [(s.name, s.reason) for s in ghx], CAPTURE, "example-images"))
    if py:
        report.append(
            section("Python example outputs are out of date:", [(s.name, s.reason) for s in py], ["Run: invoke docs-images", "Commit docs/_images/."], "example-outputs")
        )
    if unknown_subcategories:
        report.append(
            section(
                "Components are in a ribbon panel that ComponentBase.Subcategories does not list:",
                unknown_subcategories,
                ["Add it to ComponentBase.Subcategories in src/compas_acoustics_ghcsharp/CompasAcoustics.Grasshopper/Components/ComponentBase.cs."],
                "subcategories",
            )
        )
    if missing_equivalents:
        report.append(
            section(
                "Components have no entry in docs/grasshopper/python_equivalents.json:",
                [(key, "no entry") for key in missing_equivalents],
                ["Add an entry to docs/grasshopper/python_equivalents.json (a dotted path, or null for Grasshopper only)."],
                "python-equivalents",
            )
        )
    if missing_docstrings:
        report.append(
            section(
                "Python examples have no docstring:",
                [(name, "no docstring") for name in missing_docstrings],
                ["Start the script with a docstring: a one-line title, a blank line, then a paragraph on what it shows."],
                "example-docstrings",
            )
        )
    if missing_walkthroughs:
        report.append(
            section(
                "Grasshopper examples have no walkthrough:",
                [(name, "no {}/{}.md".format(walkthroughs_dir, name)) for name in missing_walkthroughs],
                ["Write {}/<name>.md: a title line, a paragraph on what the definition does, then one numbered step per group.".format(walkthroughs_dir)],
                "walkthroughs",
            )
        )
    return report
