# Public Release Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Documentation for the first public release: two tracks (Python, Grasshopper), a Grasshopper component reference generated from the plugin's specs, paired examples, and build checks that fail with instructions when images go stale.

**Architecture:** A small package `docs/_ext/ghcomponents/` reads the component specs and the C# sources, renders `.rst` pages into gitignored folders, and checks committed images against fingerprints. `invoke docs-components`, `invoke docs` and Sphinx's `builder-inited` all call its `generate()`. Images are made offline (Rhino script, `invoke docs-images`) and committed.

**Tech Stack:** Sphinx 6.2 + sphinx_compas2_theme + sphinx_design, myst-parser (new), invoke / compas_invocations2, pytest, Rhino 8 Python (capture), lamcp (for building and capturing in Rhino during this work).

**Spec:** `docs/superpowers/specs/2026-11-01-public-release-docs-design.md`

## Global Constraints

- Sphinx with `sphinx_compas2_theme`; CI builds with `compas-dev/compas-actions.docs@v4` on Python `"3.12"`.
- Not on PyPI: no page says `pip install compas_acoustics` without the git URL; no PyPI icon link.
- CI has no Rhino: nothing in the build may need Rhino, .NET or a display.
- Unit tests only (`tests/test_release_metadata.py` guards this).
- `docs/_ext/ghcomponents/` must run on Python 3.9 (Rhino 8 imports it in the capture script): no `match`, no `str.removeprefix`; every module starts with `from __future__ import annotations` so `Path | None` annotations in dataclasses are not evaluated.
- Component source of truth: `src/compas_acoustics_ghcsharp/CompasAcoustics.Grasshopper/Components/Specs/CA_*/metadata.json` + `icon.png` (23 components).
- Section order: `ComponentBase.Subcategories` in `src/compas_acoustics_ghcsharp/CompasAcoustics.Grasshopper/Components/ComponentBase.cs`; subcategories with no components are skipped.
- All Grasshopper examples are `.ghx`. Python and Grasshopper example lists are independent; same-name pairs cross-link.
- First release: Python = every `examples/*.py`; Grasshopper = `simple_simulation_shoebox` only. `examples/shoebox.py` is renamed `examples/simple_simulation_shoebox.py`.
- Every build check is fatal and its message lists each item, why, the fix, and links `docs/contributing/docs.rst#<anchor>`.
- `docs/superpowers/` is excluded from the Sphinx build.

## Review Focus

- Windows checkouts with CRLF line endings must not make `.py`/`.ghx` fingerprints stale: file hashes normalise `\r\n` to `\n` (test in Task 2).
- Descriptions containing RST markup characters (`|`, `*`, backticks, trailing `_`) must render literally, not break the table or create links (test in Task 3).
- A component with no inputs or no outputs (e.g. `CA_Worker`) renders "None" instead of an empty or broken table (test in Task 3).
- A removed component or example must not leave its old generated page behind: `generate()` deletes generated files it did not write this run (test in Task 5).
- An example script with no module docstring, or a `.ghx` with no walkthrough notes file, fails the check with a fix, not a Python traceback (test in Task 5).

---

## File Structure

```
docs/_ext/ghcomponents/
  __init__.py        generate(), DocsCheckError, Sphinx setup()
  specs.py           read component specs, subcategory order, GUIDs from C#
  fingerprints.py    hashes, fingerprints.json load/save, stale detection
  render.py          rst for component pages, overview, example pages
  examples.py        discover .py/.ghx examples, docstrings, used-in
  checks.py          build checks and their messages
  __main__.py        CLI: python -m ghcomponents (used by invoke)
docs/_tools/
  capture_grasshopper.py   Rhino 8 script: component and .ghx images
  run_example.py           runs one example headless, saves figures + stdout
docs/_images/
  components/CA_<name>.png
  examples/<name>/{figure_N.png, stdout.txt, canvas.png, viewport.png}
  fingerprints.json
docs/grasshopper/python_equivalents.json
docs/grasshopper/notes/CA_<name>.rst          optional
docs/grasshopper/walkthroughs/<name>.rst      required per .ghx
generated (gitignored): docs/grasshopper/components/, docs/grasshopper/examples/, docs/python/examples/
tests/docs/conftest.py, tests/docs/test_*.py, tests/docs/fixtures/
```

`fingerprints.json` shape: `{"components": {"CA_X": "<hex12>"}, "py": {"<name>": "<hex12>"}, "ghx": {"<name>": "<hex12>"}}`.

---

### Task 1: Read component specs, subcategory order and GUIDs

**Files:**
- Create: `docs/_ext/ghcomponents/__init__.py` (empty for now), `docs/_ext/ghcomponents/specs.py`
- Create: `tests/docs/conftest.py`, `tests/docs/test_specs.py`, `tests/docs/fixtures/` (two spec folders, a mini `ComponentBase.cs`, a mini components `.cs`)

**Interfaces:**
- Produces:
  - `@dataclass Port(name: str, description: str, type_hint: str|None, access: str, optional: bool, default: object, choices: list[str])` — `access` is `"item"|"list"|"tree"` from `scriptParamAccess` 0/1/2 (missing → `"item"`); `choices` from `namedValues` first elements.
  - `@dataclass Component(key: str, name: str, nickname: str, subcategory: str, description: str, inputs: list[Port], outputs: list[Port], icon: Path)` — `key` is the folder name.
  - `load_components(specs_dir: Path) -> list[Component]` sorted by key.
  - `read_subcategory_order(component_base_cs: Path) -> list[str]`
  - `read_component_guids(components_dir: Path) -> dict[str, str]` key → lowercase GUID; scans `*.cs` recursively.
  - `REPO`, `SPECS_DIR`, `COMPONENT_BASE_CS`, `CSHARP_DIR` path constants (repo root = `Path(__file__).parents[3]`).

- [ ] **Step 1: Write failing tests** in `tests/docs/test_specs.py`. `conftest.py` inserts `docs/_ext` into `sys.path`.

```python
def test_ports_carry_access_defaults_and_choices(fixtures):
    (c,) = [c for c in load_components(fixtures / "Specs") if c.key == "CA_Fixture_A"]
    assert [p.access for p in c.inputs] == ["item", "list"]
    assert c.inputs[0].choices == ["0", "1", "2"] and c.inputs[1].default == 0.25
    assert c.inputs[1].optional is True and c.outputs[0].type_hint is None

def test_subcategory_order_is_read_from_the_csharp_list(fixtures):
    assert read_subcategory_order(fixtures / "ComponentBase.cs") == ["Elements", "Params", "Model"]

def test_guids_pair_spec_keys_with_component_guids_including_digits(fixtures):
    guids = read_component_guids(fixtures / "cs")
    assert guids == {"CA_Fixture_A": "410e7b44-f239-5e1e-ba74-5884e208b73e",
                     "CA_Plot_2D": "ddca21b5-bcc8-5806-85f8-fc0a45ced2e5"}  # a Params class with a GUID and no SpecKey is ignored

def test_the_real_specs_all_have_a_guid_and_a_known_subcategory():
    comps = load_components(SPECS_DIR)
    assert len(comps) == 23
    assert {c.key for c in comps} <= set(read_component_guids(CSHARP_DIR))
    assert {c.subcategory for c in comps} <= set(read_subcategory_order(COMPONENT_BASE_CS))
```

- [ ] **Step 2: Run** `pytest tests/docs/test_specs.py -v` — Expected: FAIL (ImportError).
- [ ] **Step 3: Implement `specs.py`.** Subcategories: regex `Subcategories\s*=\s*new\[\]\s*\{([^}]*)\}` then `"([^"]+)"`. GUIDs: split each file on `\bclass\s+\w+`; in each block take `SpecKey\s*=>\s*"(CA_\w+)"` and `ComponentGuid\s*=>\s*new Guid\("([0-9a-fA-F-]+)"\)`; keep blocks with both. Raise `ValueError` naming the file when the subcategory list is not found.
- [ ] **Step 4: Run** `pytest tests/docs/test_specs.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add docs/_ext tests/docs && git commit -m "Docs: read Grasshopper component specs, ribbon order and GUIDs"`

### Task 2: Fingerprints

**Files:**
- Create: `docs/_ext/ghcomponents/fingerprints.py`, `tests/docs/test_fingerprints.py`

**Interfaces:**
- Consumes: `Component` (Task 1).
- Produces:
  - `FINGERPRINTS = REPO / "docs/_images/fingerprints.json"`
  - `port_fingerprint(component: Component) -> str` — sha256 of `json.dumps([[p.name for p in inputs], [p.name for p in outputs]])`, first 12 hex.
  - `file_fingerprint(path: Path) -> str` — sha256 of bytes with `b"\r\n"` → `b"\n"`, first 12 hex.
  - `load(path: Path = FINGERPRINTS) -> dict` (missing file → the three empty sections), `save(data: dict, path: Path = FINGERPRINTS) -> None` (sorted keys, indent 2, trailing newline).
  - `@dataclass Stale(kind: str, name: str, reason: str)` — kind `"components"|"py"|"ghx"`; reason `"no image"`, `"ports changed (added input 'x', removed output 'y')"`, `"file changed"`.
  - `stale_components(components, data, image_dir) -> list[Stale]`; `stale_files(kind, paths: list[Path], data, image_dir) -> list[Stale]`.
  - For "ports changed" detail, `data["components"][key]` stores `{"hash": ..., "inputs": [...], "outputs": [...]}` so the reason can name added/removed ports.

- [ ] **Step 1: Write failing tests**

```python
def test_file_fingerprint_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_bytes(b"x = 1\ny = 2\n"); b.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert file_fingerprint(a) == file_fingerprint(b)

def test_a_component_with_a_matching_fingerprint_and_image_is_fresh(component, tmp_path): ...
def test_a_missing_image_is_stale_with_reason_no_image(component, tmp_path): ...
def test_changed_ports_name_what_was_added_and_removed(component, tmp_path):
    # stored inputs ["ir"], outputs ["parameters"]; current outputs ["parameters", "frequency"]
    assert stale_components([component], data, tmp_path)[0].reason == "ports changed (added output 'frequency')"

def test_a_changed_example_file_is_stale(tmp_path): ...
def test_load_of_a_missing_file_gives_empty_sections(tmp_path):
    assert load(tmp_path / "none.json") == {"components": {}, "py": {}, "ghx": {}}
```

Image presence: components → `image_dir/components/<key>.png`; `py` → `image_dir/examples/<name>/stdout.txt`; `ghx` → `image_dir/examples/<name>/canvas.png` and `viewport.png`.

- [ ] **Step 2: Run** `pytest tests/docs/test_fingerprints.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement `fingerprints.py`.**
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `"Docs: fingerprints for component and example images"`

### Task 3: Render component pages and the overview

**Files:**
- Create: `docs/_ext/ghcomponents/render.py`, `tests/docs/test_render_components.py`

**Interfaces:**
- Consumes: `Component`, `Port` (Task 1).
- Produces:
  - `escape(text: str) -> str` — backslash-escapes `\ * ` _ |` so text renders literally.
  - `component_page(c: Component, *, image: str, icon: str, notes: Path|None, used_in: list[str], python: str|None) -> str` — `image`/`icon` are paths relative to the page; `used_in` example names; `python` dotted path or `None` (Grasshopper only).
  - `overview_page(components: list[Component], order: list[str], icon_for: Callable[[Component], str]) -> str`

- [ ] **Step 1: Write failing tests**

```python
def test_page_has_header_description_image_and_tables(component):
    page = component_page(component, image="/_images/components/CA_Analyse_IR.png", icon="_icons/CA_Analyse_IR.png",
                          notes=None, used_in=["simple_simulation_shoebox"], python="compas_acoustics.Analyser")
    assert page.startswith(".. _gh-CA_Analyse_IR:\n")
    assert "Analyse Impulse Response" in page and "Analysis · AnalyseIR" in page
    assert ".. list-table::" in page and ":doc:`../examples/simple_simulation_shoebox`" in page
    assert ":obj:`compas_acoustics.Analyser`" in page

def test_markup_in_descriptions_renders_literally(component):
    component.description = "a|b *c* `d` ir_"
    assert r"a\|b \*c\* \`d\` ir\_" in component_page(component, image="i", icon="i", notes=None, used_in=[], python=None)

def test_no_ports_and_no_python_equivalent_say_so(worker_component):  # no inputs, no outputs
    page = component_page(worker_component, image="i", icon="i", notes=None, used_in=[], python=None)
    assert page.count("None.") == 2 and "No Python equivalent (Grasshopper only)." in page

def test_choices_optional_and_default_are_shown(component): ...  # "Choices: 0, 1, 2", "optional", "0.25"

def test_notes_are_included_when_given(component, tmp_path): ...  # ".. include:: <path>"

def test_overview_follows_the_order_and_skips_empty_subcategories(components):
    page = overview_page(components, ["Elements", "Params", "Analysis"], lambda c: f"_icons/{c.key}.png")
    assert page.index("Elements") < page.index("Analysis") and "Params" not in page
    assert ".. grid-item-card::" in page and ":link: CA_Analyse_IR" in page
```

Page layout (in order): label, title (display name), icon image (`:width: 48px`) + line `Subcategory · Nickname`, description, screenshot `.. image::`, "Inputs" `list-table` (Name, Type, Access, Default, Description; optional appended to Description as "*Optional.*", choices as "Choices: …"), "Outputs" `list-table` (Name, Description), "Notes" (include) if any, "Used in" if any, "Python equivalent". Overview: one section per non-empty subcategory, `.. grid:: 2 3 4 4` of `grid-item-card`s with the icon, name, nickname and first sentence of the description, plus a hidden `toctree` of all pages.

- [ ] **Step 2: Run** `pytest tests/docs/test_render_components.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** in `render.py`.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `"Docs: render Grasshopper component pages and overview"`

### Task 4: Discover and render examples

**Files:**
- Create: `docs/_ext/ghcomponents/examples.py`, `tests/docs/test_examples.py`, fixtures `tests/docs/fixtures/examples/{both.py,both.ghx,py_only.py,ghx_only.ghx}`, `tests/docs/fixtures/walkthroughs/{both.rst,ghx_only.rst}`
- Modify: `docs/_ext/ghcomponents/render.py` (add example page functions)

**Interfaces:**
- Consumes: `escape` (Task 3), `read_component_guids` output (Task 1).
- Produces:
  - `@dataclass Example(name: str, py: Path|None, ghx: Path|None)`; `discover(examples_dir: Path) -> list[Example]` sorted by name.
  - `docstring(py: Path) -> tuple[str, str]|None` — (title = first docstring line without trailing period, body = rest stripped) via `ast.get_docstring`; `None` when absent.
  - `components_in(ghx: Path, guids: dict[str, str]) -> set[str]` — component keys whose GUID occurs in the file text (case-insensitive).
  - `render.python_example_page(ex: Example, title: str, body: str, figures: list[str], stdout: str|None) -> str`
  - `render.ghx_example_page(ex: Example, walkthrough: Path, download: str, canvas: str, viewport: str) -> str`
  - `render.examples_index(title: str, names_titles: list[tuple[str, str]]) -> str`

- [ ] **Step 1: Write failing tests**

```python
def test_examples_pair_by_name_and_either_side_may_be_missing(fixtures):
    assert [(e.name, bool(e.py), bool(e.ghx)) for e in discover(fixtures / "examples")] == [
        ("both", True, True), ("ghx_only", False, True), ("py_only", True, False)]

def test_docstring_gives_title_and_body(fixtures): ...
def test_docstring_is_none_without_one(tmp_path): ...
def test_components_in_finds_guids_case_insensitively(fixtures): ...

def test_python_page_links_its_grasshopper_twin_only_when_there_is_one(fixtures):
    both, py_only = ...
    assert ":doc:`/grasshopper/examples/both`" in python_example_page(both, "T", "B", [], None)
    assert "grasshopper/examples" not in python_example_page(py_only, "T", "B", [], None)

def test_python_page_includes_script_figures_and_stdout(fixtures): ...  # literalinclude, .. image:: per figure, literal block of stdout
def test_ghx_page_has_download_images_walkthrough_and_python_link(fixtures): ...  # :download:, canvas, viewport, .. include::, :doc:`/python/examples/both`
```

- [ ] **Step 2: Run** `pytest tests/docs/test_examples.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `"Docs: discover and render Python and Grasshopper examples"`

### Task 5: Checks, `generate()` and the CLI

**Files:**
- Create: `docs/_ext/ghcomponents/checks.py`, `docs/_ext/ghcomponents/__main__.py`, `tests/docs/test_generate.py`
- Modify: `docs/_ext/ghcomponents/__init__.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces:
  - `class DocsCheckError(Exception)` — message is the full multi-section report.
  - `@dataclass Paths(specs, component_base_cs, csharp, examples, docs, images, equivalents, notes, walkthroughs)` with `Paths.default()` for the repo.
  - `generate(paths: Paths = None, *, check: bool = True) -> list[Path]` — writes all generated pages, returns the files written; deletes files under the three generated folders that this run did not write; writes a file only when its content changed (so a second call in one build does not force Sphinx to re-read); runs all checks, collects every problem, raises one `DocsCheckError` at the end if any.
  - `checks.run(...) -> list[str]` — one section per failing check, in the spec's order, each ending in `See docs/contributing/docs.rst#<anchor>.`. Anchors: `screenshots`, `example-images`, `example-outputs`, `subcategories`, `python-equivalents`, `example-docstrings`, `walkthroughs`.
  - CLI: `python docs/_ext/ghcomponents` → runs `generate()`, prints the written count, or the report to stderr with exit code 1.
  - `setup(app)` in `__init__.py`: connects `builder-inited` to a handler that calls `generate()` and re-raises `DocsCheckError` as `sphinx.errors.ExtensionError`; returns `{"parallel_read_safe": True}`.

Check messages (exact fix text):

| Check | Fix lines |
|---|---|
| components stale | `1. Install the current plugin in Rhino 8 (invoke installer-bundle, then run the installer).` `2. Open Rhino 8 and Grasshopper.` `3. Open docs/_tools/capture_grasshopper.py in Rhino's ScriptEditor and run it.` `4. Commit docs/_images/ (the PNGs and fingerprints.json).` |
| ghx stale | same four lines |
| py stale | `Run: invoke docs-images` / `Commit docs/_images/.` |
| subcategory unknown | `Add it to ComponentBase.Subcategories in src/compas_acoustics_ghcsharp/CompasAcoustics.Grasshopper/Components/ComponentBase.cs.` |
| equivalents missing (key absent; `null` is allowed) | `Add an entry to docs/grasshopper/python_equivalents.json (a dotted path, or null for Grasshopper only).` |
| example without docstring | `Start the script with a docstring: a one-line title, a blank line, then a paragraph on what it shows.` |
| `.ghx` without walkthrough | `Write docs/grasshopper/walkthroughs/<name>.rst: a paragraph on what the definition does, then one numbered step per group.` |

- [ ] **Step 1: Write failing tests** (all against a `tmp_path` copy of the fixtures through `Paths(...)`)

```python
def test_a_complete_fixture_tree_generates_without_errors(tree): ...
def test_every_problem_is_reported_in_one_error(tree):
    # remove one PNG, add an unknown subcategory, drop one equivalents entry
    with pytest.raises(DocsCheckError) as e:
        generate(tree)
    text = str(e.value)
    assert "CA_Fixture_A   no screenshot" in text and "#screenshots" in text
    assert "#subcategories" in text and "#python-equivalents" in text

def test_a_script_without_docstring_and_a_ghx_without_walkthrough_fail_with_a_fix(tree): ...
def test_pages_of_removed_components_are_deleted(tree):
    (tree.docs / "grasshopper/components/CA_Gone.rst").write_text("old")
    generate(tree)
    assert not (tree.docs / "grasshopper/components/CA_Gone.rst").exists()
def test_unchanged_pages_are_not_rewritten(tree): ...  # mtime stays after a second generate()
def test_check_false_writes_pages_and_raises_nothing(tree): ...
```

`generate()` also copies each `icon.png` to `docs/grasshopper/components/_icons/<key>.png` and each `.ghx` to `docs/grasshopper/examples/_downloads/<name>.ghx`, so no page references a path outside `docs/`.

- [ ] **Step 2: Run** `pytest tests/docs/test_generate.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest tests/docs -v` — Expected: all PASS.
- [ ] **Step 5: Commit** `"Docs: generate component and example pages, with build checks"`

### Task 6: Site skeleton and wiring

**Files:**
- Modify: `docs/conf.py`, `docs/index.rst`, `tasks.py`, `.gitignore`, `requirements-dev.txt`, `tests/test_release_metadata.py`, `CHANGELOG.md`
- Move: `docs/installation.rst` → `docs/installation/index.rst` (content split in Task 9), `docs/tutorial.rst` → `docs/python/tutorial.rst`
- Delete: `docs/examples.rst`, `docs/examples/`
- Create: `docs/concepts.rst`, `docs/installation/python.rst`, `docs/installation/grasshopper.rst`, `docs/grasshopper/getting_started.rst`, `docs/troubleshooting.rst`, `docs/contributing/docs.rst`, `docs/changelog.rst` (each a title and one sentence; filled in Tasks 9–11), `docs/grasshopper/python_equivalents.json`, `docs/grasshopper/notes/.gitkeep`, `docs/grasshopper/walkthroughs/.gitkeep`

**Interfaces:**
- Consumes: `ghcomponents.generate`, `ghcomponents.setup`, CLI (Task 5).
- Produces: `invoke docs-components`, `invoke docs` (same options as `compas_invocations2.docs.docs`: `doctest`, `rebuild`, `check_links`).

- [ ] **Step 1: Update `tests/test_release_metadata.py`** to the new layout, then run it and see it fail:
  - `EXAMPLES = ["simple_simulation_shoebox", "obj", "rays", "scattered_rays", "specular_reflections"]`; assert each `examples/<name>.py` exists and `examples/shoebox.py` does not.
  - tutorial path `docs/python/tutorial.rst`; installation checks read `docs/installation/grasshopper.rst` (`compas_acoustics_installer.zip`) and `docs/installation/python.rst` (no bare `pip install compas_acoustics\n`).
  - new: `assert "pypi.org" not in read("docs/conf.py")`; `assert "superpowers" in read("docs/conf.py")`.
  - Run `pytest tests/test_release_metadata.py -v` — Expected: FAIL.
- [ ] **Step 2: Rename** `git mv examples/shoebox.py examples/simple_simulation_shoebox.py`; update `README.md:114`; add under `Unreleased` in `CHANGELOG.md`: `* Renamed the example examples/shoebox.py to examples/simple_simulation_shoebox.py; its Grasshopper twin is examples/simple_simulation_shoebox.ghx.`
- [ ] **Step 3: `docs/conf.py`:** `sys.path.insert(0, os.path.abspath("_ext"))`; `extensions` gets `"myst_parser"` and `"ghcomponents"`; `exclude_patterns` gets `"superpowers/**"`; remove the PyPI entry from `icon_links`. `requirements-dev.txt` gets `myst-parser >=2.0, <3.0` (Sphinx 6.2).
- [ ] **Step 4: Toctree** in `docs/index.rst` (keep the lead paragraph verbatim): Getting started (`concepts`, `installation/index`), Python (`python/tutorial`, `python/examples/index`), Grasshopper (`grasshopper/getting_started`, `grasshopper/components/index`, `grasshopper/examples/index`), `api`, `troubleshooting`, `contributing/docs`, `changelog`, `license`. `docs/changelog.rst` holds `.. include:: ../CHANGELOG.md` with `:parser: myst_parser.sphinx_`.
- [ ] **Step 5: `.gitignore`:** `docs/grasshopper/components/`, `docs/grasshopper/examples/`, `docs/python/examples/`.
- [ ] **Step 6: `python_equivalents.json`:** one key per component (23). Value: the public `compas_acoustics` name the component's worker call uses (read the component's `CallWorker`/API function name in its C# class and the function in `src/compas_acoustics/worker/api.py`), or `null` for Grasshopper-only components. Keys sorted.
- [ ] **Step 7: `tasks.py`:** add `docs_components(ctx)` running `python docs/_ext/ghcomponents` in `ctx.base_folder`; add a `docs(ctx, doctest=False, rebuild=False, check_links=False)` task with `pre=[docs_components]` that calls `compas_invocations2.docs.docs(ctx, doctest=doctest, rebuild=rebuild, check_links=check_links)` (import the module as `from compas_invocations2 import docs as ci_docs` and rename uses in `ns`); register both in `ns` in place of `docs.docs`.
- [ ] **Step 8: Verify.** `pytest tests/test_release_metadata.py tests/docs -v` — Expected: PASS. `invoke docs-components` — Expected: exit 1 with the screenshots report listing all 23 components (no images yet), the `example-outputs` report for 5 scripts, and docstring failures for `obj` and `simple_simulation_shoebox`; pages still written. This failing state is expected until Tasks 7–8.
- [ ] **Step 9: Commit** `"Docs: site structure for two tracks; generator wired into invoke and Sphinx"`

### Task 7: Python example outputs (`invoke docs-images`)

**Files:**
- Create: `docs/_tools/run_example.py`, `tests/docs/test_run_example.py`, `docs/_images/examples/<name>/…`
- Modify: `tasks.py`, `examples/simple_simulation_shoebox.py`, `examples/obj.py` (add docstrings), `docs/_images/fingerprints.json`

**Interfaces:**
- Consumes: `fingerprints.file_fingerprint/load/save/stale_files`, `examples.discover` (Tasks 2, 4).
- Produces: `run_example(script: Path, out_dir: Path) -> None` — sets `MPLBACKEND=Agg`, `random.seed(0)`, `numpy.random.seed(0)`, replaces `matplotlib.pyplot.show` with a function that saves every open figure as `figure_<n>.png` (n counts across calls, from 1, `dpi=150`, `bbox_inches="tight"`) and closes them, replaces `compas_viewer.Viewer.show` with a no-op, runs the script with `runpy.run_path(..., run_name="__main__")`, writes captured stdout to `stdout.txt`, and empties `out_dir` first. `invoke docs-images [--all]` runs it in a subprocess per stale script (all with `--all`) and updates `py` fingerprints only for scripts that succeeded.

- [ ] **Step 1: Write failing test** `test_figures_and_stdout_are_saved`: a temp script that prints `"hi"`, plots two figures and calls `plt.show()` once → `figure_1.png`, `figure_2.png`, `stdout.txt == "hi\n"`.
- [ ] **Step 2: Run** `pytest tests/docs/test_run_example.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `run_example.py` (`python docs/_tools/run_example.py <script> <out_dir>`) and the `docs_images(ctx, all=False)` task.
- [ ] **Step 4: Run** the test — Expected: PASS.
- [ ] **Step 5: Docstrings.** `simple_simulation_shoebox.py`: title `A simple simulation in a shoebox room`, body: a shoebox built from scratch, one material per surface, simulated with image sources and ray tracing and analysed; it is the Python twin of `simple_simulation_shoebox.ghx`. `obj.py`: title `A room from OBJ files`, body from its current docs page.
- [ ] **Step 6: Run** `invoke docs-images --all` — Expected: `docs/_images/examples/<name>/` for all 5 scripts; `figure_1.png` for `simple_simulation_shoebox` and `obj` (T30 plot); `stdout.txt` for all. Look at each PNG.
- [ ] **Step 7: Run** `invoke docs-components` — Expected: only the component screenshot and `.ghx` reports remain.
- [ ] **Step 8: Commit** `"Docs: Python example outputs, made by invoke docs-images"`

### Task 8: Grasshopper capture and `simple_simulation_shoebox.ghx` (needs Rhino + lamcp)

**Ask the user to open Rhino 8 and Grasshopper, install the current plugin (`invoke installer-bundle`, run the installer, restart Rhino) and start the lamcp bridge before this task.** Check with `mcp__lamcp__bridge_health`. Placing components: see the team learning "Placing Grasshopper components through lamcp" (type GUID via `search_components`; keep clear of the LAMCP status panel).

**Files:**
- Create: `docs/_tools/capture_grasshopper.py`, `examples/simple_simulation_shoebox.ghx`, `docs/grasshopper/walkthroughs/simple_simulation_shoebox.rst`, `docs/_images/components/CA_*.png`, `docs/_images/examples/simple_simulation_shoebox/{canvas,viewport}.png`
- Modify: `docs/_images/fingerprints.json`

**Interfaces:**
- Consumes: `specs.load_components`, `specs.read_component_guids`, `fingerprints.*`, `examples.discover`.
- Produces: `capture_grasshopper.py`, a Rhino 8 (CPython 3.9) script. Top: `ALL = False`; repo root from `Path(__file__).resolve().parents[2]`, falling back to env `COMPAS_ACOUSTICS_REPO` when `__file__` is undefined (lamcp `run_python_script`). Adds `docs/_ext` to `sys.path` and imports `ghcomponents`. For each stale component (all if `ALL`): new empty GH document, place by GUID, solve, render that object's bounds plus a 20 px margin at 2× zoom to `docs/_images/components/<key>.png`, write its fingerprint. For each stale `.ghx`: open it, solve, zoom to the canvas extents and render `canvas.png` (2×), set the Perspective viewport to shaded, zoom extents, `CaptureToBitmap` 1600×1000 to `viewport.png`, write its fingerprint. Saves fingerprints after each item, so an interrupted run keeps its progress. Prints one line per capture.

- [ ] **Step 1: Write** `capture_grasshopper.py`; work out the canvas-render API live through lamcp `run_python_script` on one component (`CA_Source`) and check the PNG shows the component with its port labels.
- [ ] **Step 2: Run it** for all components. Expected: 23 PNGs; open a few (`CA_Analyse_IR`, `CA_Worker`, `CA_Simulate_Mono`) and check they show every port.
- [ ] **Step 3: Build `examples/simple_simulation_shoebox.ghx`** through lamcp to match the script: the same 5 × 4 × 3 m room from Rhino geometry with the same six materials and coefficients, the same two sources and two receivers, `Simulate Mono` with ray tracing, 10000 rays, air absorption, `Analyse IR` on receiver_2/source_2, `Plot 2D` of T30. Group and label each stage (Room, Materials, Sources & receivers, Simulate, Analyse). Save as `.ghx`.
- [ ] **Step 4: Check equivalence.** Read T30 and C80 per band from the `.ghx` (`inspect_component_data`) and from `python examples/simple_simulation_shoebox.py`. Expected: equal within 5 % per band (ray tracing is random). If not, find the difference in the definition, not in the tolerance.
- [ ] **Step 5: Write the walkthrough** `docs/grasshopper/walkthroughs/simple_simulation_shoebox.rst`: first line a title (`A simple simulation in a shoebox room`), a paragraph, then one numbered step per group, naming the components (as `:ref:` to `gh-CA_<key>`).
- [ ] **Step 6: Run the capture script again** — Expected: it captures only the `.ghx` images.
- [ ] **Step 7: Run** `invoke docs` — Expected: build succeeds; open `dist/docs/grasshopper/components/index.html`, one component page and both example pages; check the images, tables, cross-links and download.
- [ ] **Step 8: Commit** `"Docs: component screenshots, the capture script, and simple_simulation_shoebox.ghx"`

### Task 9: Installation pages and a shorter README

**Files:**
- Modify: `docs/installation/index.rst`, `docs/installation/python.rst`, `docs/installation/grasshopper.rst`, `README.md`

- [ ] **Step 1: Write the pages** from the README's "Installation", "Binaural and ambisonics methods" and "Install in Rhino 8 / Grasshopper" sections plus the old `docs/installation.rst`. `index.rst`: which route to pick, then a toctree of the two. `python.rst`: environment, pip from the git URL and from a tag, Python 3.11+ for Binaural/Ambisonics and the PyTorch size, development install. `grasshopper.rst`: download, the three steps, what the installer does, installer options (`source`, `venv`, `worker`), the worker (starts, port, token, `~/.compas_acoustics/worker.json`, `COMPAS_ACOUSTICS_WORKER_PYTHON`, `compas_acoustics.worker.client()`), updating (run the new installer), uninstalling (delete `Libraries/compas_acoustics/` and `~/.compas_acoustics/`; check the paths in `installer/installer.py`).
- [ ] **Step 2: Shorten the README:** keep the lead, features, a 5-line install summary for each route, and a link to `https://gramaziokohler.github.io/compas_acoustics/latest/installation/` (check the URL pattern against `docs.yml`/`gh-pages`), the Python quick example and the plugin development section. `tests/test_release_metadata.py::test_readme_and_docs_open_with_the_lead` must still pass.
- [ ] **Step 3: Verify** `pytest tests/test_release_metadata.py -v` and `invoke docs` — Expected: PASS, build succeeds.
- [ ] **Step 4: Commit** `"Docs: installation pages for Python and Grasshopper; README links to them"`

### Task 10: Home, Concepts, Grasshopper getting started, Troubleshooting, License & credits

**Files:**
- Modify: `docs/index.rst`, `docs/concepts.rst`, `docs/grasshopper/getting_started.rst`, `docs/troubleshooting.rst`, `docs/license.rst`
- Create: `docs/_images/home.png` (taken in Task 8's Rhino session or now: viewport of `simple_simulation_shoebox` with `Visualise Rays`)

- [ ] **Step 1: Home:** lead paragraph unchanged, hero image, `grid` of two cards linking to `grasshopper/getting_started` and `python/tutorial`, the README's feature list, "How to cite" (authors from `pyproject.toml`, title, year 2025, URL; BibTeX block).
- [ ] **Step 2: Concepts** per the spec's list; ISO 3382 table rows: EDT, T20, T30, C50, C80, D50, DRR, Ts, STI, G, SPL, LF, LFC — the set `CA_Analyse_IR` outputs — one line each on what it means for a listener, plus "needs" (G/SPL: simulated; LF/LFC: ambisonic).
- [ ] **Step 3: Grasshopper getting started** per the spec: data flow with `:ref:` links to component pages, the worker (when it starts, how long, `ComponentBase.OpenWorkerLog`'s menu entry, where the log is), messages users see (take the texts from the C# sources: grep for "installer again" and other user-facing strings in `ComponentBase.cs`/`WorkerComponent.cs`).
- [ ] **Step 4: Troubleshooting** per the spec's list, each as "Symptom / Cause / Fix". The stale Yak package case: see the team learning "Stale C# audio player in Grasshopper is the old Yak package".
- [ ] **Step 5: License & credits:** `literalinclude` of `LICENSE`, authors, bundled-data table (materials library, `KEMAR_FreeFieldComp_48kHz.sofa`, the four `.wav` files) with source and licence from each library's `README.md`; mark entries without a recorded source or licence as "Unresolved".
- [ ] **Step 6: Verify** `invoke docs` — Expected: build succeeds; read each page in `dist/docs`.
- [ ] **Step 7: Commit** `"Docs: home, concepts, Grasshopper getting started, troubleshooting, credits"`

### Task 11: Contributor instructions

**Files:**
- Modify: `docs/contributing/docs.rst`, `CONTRIBUTING.md`, `docs/python/tutorial.rst`

- [ ] **Step 1: `docs/contributing/docs.rst`**, sections with explicit labels matching the Task 5 anchors (`.. _screenshots:` etc.): building (`invoke docs`, `invoke docs-components`); what is generated and where its source is; making images (`capture_grasshopper.py` in Rhino with `ALL`, `invoke docs-images [--all]`); one section per check, with its message's fix; adding a Python example (docstring format, `invoke docs-images`); adding a Grasshopper example (`.ghx`, labelled groups, walkthrough file, capture, optional same-name `.py`); component notes and Python equivalents; release checklist (`invoke docs-images --all`, capture with `ALL = True`, `invoke linkcheck` — `releases/latest` 404s until the repo is public).
- [ ] **Step 2: Check that each anchor resolves:** the anchors in the check messages must be the HTML ids Sphinx generates for these labels. Run `invoke docs` and open `dist/docs/contributing/docs.html#screenshots` etc.; if Sphinx ids differ from the labels, change the anchors in `checks.py` (and its tests) to match.
- [ ] **Step 3: `CONTRIBUTING.md`:** a "Documentation" section after "Code contributions": `invoke docs` regenerates the component pages and fails when images are stale; how to remake images (two commands/steps); link to the docs page. In the task list, `invoke docs` reads "Generate the component pages and build the documentation."
- [ ] **Step 4: Tutorial check.** Extract the `.. code-block:: python` blocks of `docs/python/tutorial.rst` into one script in the scratchpad and run it. Fix the tutorial where it no longer matches the API.
- [ ] **Step 5: Commit** `"Docs: contributor instructions for the docs build and images"`

### Task 12: Final verification

- [ ] **Step 1:** `invoke lint && invoke test` — Expected: PASS.
- [ ] **Step 2:** `invoke docs --rebuild` — Expected: build succeeds with no new warnings from our pages.
- [ ] **Step 3:** Break it on purpose: add a dummy input name to one `metadata.json`, run `invoke docs-components`, check the message names the component and the added input and the fix; revert.
- [ ] **Step 4:** `invoke linkcheck` — Expected: only `releases/latest` and pages not yet published fail.
- [ ] **Step 5:** `trail check`; resolve or explain conflicts (release docs are known to match Coefficient-Lab learnings on keywords only).
- [ ] **Step 6:** Save a trail learning for the docs system (decided / ruled out / watch out) and commit it.
