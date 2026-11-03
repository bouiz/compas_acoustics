# Public release documentation: design

Date: 2026-11-01 · Branch: `release/public-prep` · Status: approved in brainstorming, awaiting spec review

## Goal

Documentation for the first public release of `compas_acoustics` that lets a newcomer install the package and run a
first simulation, in Python or in Grasshopper, from the docs alone. It describes the package's context, purpose and
intended use; how to install the Python package and the Grasshopper plugin; every Grasshopper component (image,
inputs, outputs, description); and Python and Grasshopper examples.

## Audience

Two equal tracks, "Use in Grasshopper" and "Use in Python", that share an introduction and a short acoustics primer.
Grasshopper readers (architects, designers) get acoustic terms explained once in Concepts; Python readers get the API.

## Constraints

- Keep Sphinx with `sphinx_compas2_theme`, built by `.github/workflows/docs.yml` (`compas-dev/compas-actions.docs@v4`,
  Python 3.12) and published to GitHub Pages.
- The package is not on PyPI. Nothing in the docs claims it is; the PyPI icon link in `docs/conf.py` is removed.
- CI has no Rhino. Anything that needs Rhino runs on a contributor's machine and its output is committed.
- The test suite stays unit-only (see `tests/test_release_metadata.py`).

## Site structure

```
Home                     lead, hero image, the two tracks, features, how to cite
Getting started
  Concepts               acoustics primer
  Installation
    Python
    Grasshopper
Python
  Tutorial               existing docs/tutorial.rst, refreshed
  Examples               one page per examples/*.py
Grasshopper
  Getting started        the canvas workflow and the worker
  Components             overview grid, then one generated page per component
  Examples               one page per examples/*.ghx
API reference            existing autosummary
Troubleshooting
Contributing
  Docs                   building the docs, images, failure messages, adding examples
Changelog                CHANGELOG.md included through MyST
License & credits
```

Proposed source layout (the plan may adjust file names, not the structure):

```
docs/
  index.rst
  concepts.rst
  installation/{index,python,grasshopper}.rst
  python/{tutorial,examples/}.rst
  grasshopper/
    getting_started.rst
    components/            generated, gitignored
    examples/              generated, gitignored
    notes/CA_<name>.rst    optional hand-written notes per component
    python_equivalents.json
  troubleshooting.rst
  contributing/docs.rst
  changelog.rst
  license.rst
  _ext/ghcomponents.py     Sphinx extension and generator
  _tools/capture_grasshopper.py
  _images/
    components/CA_<name>.png
    examples/<name>/...
    fingerprints.json
```

`docs/superpowers/` is added to `exclude_patterns` so specs and plans are not published.

## Written pages

- **Home**: the existing lead paragraph, a hero image (Rhino viewport with rays or an energy map), two cards into the
  tracks, a feature list, how to cite.
- **Concepts** (about one page): scene, acoustic objects, materials (absorption and scattering in 7 octave bands,
  125 Hz to 8 kHz), sources and receivers; the impulse response; Mono vs Binaural vs Ambisonics and what each needs
  (the hybrid methods need Python 3.11+); a table of the ISO 3382 parameters with one line each on what it means for
  a listener; auralisation. Other pages link here instead of repeating it.
- **Installation**: moved from `README.md` into the docs; the README keeps a short summary that links to them.
  Python: pip from a git tag, the 3.11+ hybrid dependencies and the PyTorch size, development install. Grasshopper:
  installer zip steps, installer options, the worker (what it is, where it lives, `COMPAS_ACOUSTICS_WORKER_PYTHON`),
  updating, uninstalling.
- **Grasshopper getting started**: how data flows (geometry → acoustic objects → scene → simulate →
  analyse / auralise / display), when the worker starts, how to read its log, the messages users will see (for
  example "run the installer again").
- **Python tutorial**: the existing page, checked against the current API. Its `code-block`s are extracted and run
  once as a script (`invoke testcodeblocks` only runs `>>>` doctests).
- **Troubleshooting**: worker does not start, version or schema mismatch, hybrid methods missing on Python 3.9/3.10,
  macOS Gatekeeper / quarantine on the `.gha`, a stale plugin from an old Yak package, where the logs are.
- **License & credits**: MIT, authors, a table of bundled data (materials, HRTFs, anechoic recordings) with source
  and licence. Entries the release audit left unresolved are shown as unresolved; resolving licences is out of scope.
- **Changelog**: `CHANGELOG.md` included through MyST.

## Grasshopper component reference

Source of truth: `src/compas_acoustics_ghcsharp/CompasAcoustics.Grasshopper/Components/Specs/CA_*/metadata.json`
and `icon.png` (23 components). Nothing is copied by hand.

**Ordering.** Sections follow `ComponentBase.Subcategories` in
`src/compas_acoustics_ghcsharp/CompasAcoustics.Grasshopper/Components/ComponentBase.cs`, which the generator reads
with one regex on that array literal. Subcategories without components (currently `Params`) are skipped, matching the
ribbon. Metadata holds plain subcategory names (commit `e33cc04`).

**Overview page** (`grasshopper/components/index`): one section per subcategory, each a grid of cards with icon,
display name, nickname and the first sentence of the description, linking to the component page.

**Component page**, one per component, file name from the folder (`CA_Analyse_IR`), title from `name`:

1. Header: icon, display name, subcategory, nickname.
2. Description.
3. Canvas screenshot with port labels (`docs/_images/components/CA_<name>.png`).
4. Inputs table: name, type (`typeHintID`), access (item / list / tree from `scriptParamAccess`), default, optional
   marker, description; `namedValues` listed as "Choices: …".
5. Outputs table: name, description.
6. Notes: included from `docs/grasshopper/notes/CA_<name>.rst` when that file exists.
7. Used in: Grasshopper examples whose `.ghx` contains the component's GUID (`.ghx` is XML; plain parse).
8. Python equivalent: link from `docs/grasshopper/python_equivalents.json`
   (`{"CA_Analyse_IR": "compas_acoustics.Analyser", ...}`), kept outside `metadata.json` so the plugin specs do not
   change for the docs.

Descriptions are shown as written; editing them is a separate content task.

## Examples

Two independent lists. Neither requires the other: a `.py` without a `.ghx`, or a `.ghx` without a `.py`, is fine.
When `examples/<name>.py` and `examples/<name>.ghx` share a name, their pages link to each other automatically.
All Grasshopper examples are `.ghx`.

**Python examples** (`docs/python/examples/`): every `examples/*.py`, one page each: title, a paragraph on what it
shows, `literalinclude` of the script, then its output: the matplotlib figures it opens, or its printed output for
scripts that only print (`rays.py`, `scattered_rays.py`, `specular_reflections.py`).

**Grasshopper examples** (`docs/grasshopper/examples/`): one page per `.ghx`: download link, canvas screenshot
(groups labelled in the `.ghx`), Rhino viewport screenshot, numbered walkthrough of the groups, link to the Python
page when one exists.

**First release content**: Python: all scripts in `examples/`. Grasshopper: `simple_simulation_shoebox` only.

- `examples/shoebox.py` is renamed `examples/simple_simulation_shoebox.py`. References updated: `README.md`, the
  example's docs page; a new `CHANGELOG.md` entry (the old entry stays as history).
- `examples/simple_simulation_shoebox.ghx` is built live through lamcp to match the script: same room, materials,
  source, receiver, Mono settings and analysis. Checked once by hand: both give the same T30 and C80 within
  tolerance.

## Tooling

One generator, `docs/_ext/ghcomponents.py: generate(...)`, writes the component and example pages into the
gitignored `docs/grasshopper/components/`, `docs/grasshopper/examples/` and `docs/python/examples/`. Three callers:

- `invoke docs-components`: standalone.
- `invoke docs`: a task in `tasks.py` that replaces `compas_invocations2.docs.docs` in the collection, keeps its
  options (`doctest`, `rebuild`, `check_links`), runs `docs-components` first and then the build.
- `docs/conf.py` calls it on Sphinx's `builder-inited`, so CI and a bare `sphinx-build` run it too.

Images (made offline, committed):

- `docs/_tools/capture_grasshopper.py`: a plain Rhino 8 Python script, run from Rhino's ScriptEditor or through
  lamcp's `run_python_script`, with the current plugin installed. No arguments: it reads `fingerprints.json`,
  captures only what is missing or stale (component PNGs; canvas and viewport PNGs of each `.ghx`) and rewrites the
  fingerprints. `ALL = False` at the top forces a full capture.
- `invoke docs-images`: runs each stale `examples/*.py` with `MPLBACKEND=Agg` and a fixed seed where the script
  allows, saves every figure and stdout to `docs/_images/examples/<name>/`, updates the fingerprints. `--all` re-runs
  every script.

`docs/_images/fingerprints.json` stores, per component, a hash of its port names in order; per `.ghx` and per `.py`,
a hash of the file.

## Build checks

All fatal, in every build (CI, `invoke docs`, `sphinx-build`). Each raises a Sphinx error listing every offending
item, why, and the fix, and links to the Contributing › Docs page.

| Check | Fix in the message |
|---|---|
| Component PNG missing, or its ports changed | Install the current plugin, open Rhino 8 and Grasshopper, run `docs/_tools/capture_grasshopper.py`, commit `docs/_images/` |
| `.ghx` example new or changed, its images stale | Same |
| `.py` example new or changed, its output stale | `invoke docs-images`, commit `docs/_images/` |
| A component's subcategory is not in `ComponentBase.Subcategories` | Add it in `ComponentBase.cs` |
| A component has no entry in `python_equivalents.json` | Add the entry |

Example message:

```
Grasshopper component screenshots are out of date:
  CA_Analyse_IR   ports changed (added output 'frequency')
  CA_Worker       no screenshot

To fix:
  1. Install the current plugin in Rhino 8 (invoke installer-bundle, then run the installer).
  2. Open Rhino 8 and Grasshopper.
  3. Open docs/_tools/capture_grasshopper.py in Rhino's ScriptEditor and run it.
  4. Commit docs/_images/ (the PNGs and fingerprints.json).
See docs/contributing/docs.rst#screenshots.
```

Known gap: Python output is fingerprinted on the script, not on library behaviour, so a library change that alters a
plot is not caught. `invoke docs-images --all` is added to the release checklist on the Contributing › Docs page.

## Contributor instructions

- **Contributing › Docs** (`docs/contributing/docs.rst`): build the docs (`invoke docs`); what each generated part
  is and where its source lives; making images (`capture_grasshopper.py` in Rhino, `invoke docs-images`); one
  section per build check with its fix, anchored so error messages can link to it; adding a Python example; adding a
  Grasshopper example (`.ghx`, labelled groups, capture, optional paired `.py`); adding component notes and Python
  equivalents; the release checklist (`invoke docs-images --all`, full capture, `invoke linkcheck`).
- **`CONTRIBUTING.md`**: a short "Documentation" section with the same steps in brief and a link to that page; the
  `invoke docs` line in the task list mentions that it regenerates the component pages.

## Testing

- `tests/docs/test_ghcomponents.py` (unit, fixture metadata): input and output tables, choices, optional and default
  values, "used in" from a fixture `.ghx`, reading the C# subcategory list, skipping empty subcategories, fingerprint
  comparison, the text of each failure message, `.py` / `.ghx` cross-links with and without a partner.
- The docs build in CI (`docs.yml`, already runs on PRs) is the integration check.
- Manual, once: `simple_simulation_shoebox.py` and `.ghx` agree on T30 and C80.
- `invoke linkcheck`; `releases/latest` is expected to return 404 until the repository is public and has a release.

## Out of scope

- More Grasshopper examples (Visualise sound paths, Energy map, Listen to the room, Custom materials, Design
  exploration); each comes later, with a Python version where it makes sense.
- Editing component descriptions in `metadata.json`.
- Resolving bundled-data licences and the git-history blockers from the release audit.
- Publishing to PyPI.
