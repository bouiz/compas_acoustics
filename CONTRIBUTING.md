# Contributing

Contributions are welcome and very much appreciated!

## Code contributions

We accept code contributions through pull requests.
In short, this is how that works.

1. Fork [the repository](https://github.com/gramaziokohler/compas_acoustics) and clone the fork.
2. Create a virtual environment using your tool of choice (e.g. `virtualenv`, `conda`, etc).
3. Install the package with its development dependencies:

   ```bash
   pip install -e ".[dev,hybrid]"
   ```

4. Make sure all tests pass:

   ```bash
   invoke test
   ```

5. Start making your changes on a branch off **main**.
6. Make sure lint and all tests still pass:

   ```bash
   invoke lint
   invoke test
   ```

   If you changed the Grasshopper plugin (`src/compas_acoustics_ghcsharp/`), also run its tests (they need the .NET 8 SDK):

   ```bash
   dotnet test src/compas_acoustics_ghcsharp/CompasAcoustics.sln
   ```

7. Add a line about your change to the `Unreleased` section of `CHANGELOG.md`.
8. Add yourself to the authors in `CITATION.cff`, the only list of authors edited by hand: `invoke sync-authors`
   copies it into `pyproject.toml` on release, and the docs read it when they are built.
9. Commit your changes and push your branch to GitHub.
10. Create a [pull request](https://help.github.com/articles/about-pull-requests/) through the GitHub website.

During development, use [pyinvoke](http://docs.pyinvoke.org/) tasks on the
command line to ease recurring operations:

* `invoke clean`: Clean all generated artifacts.
* `invoke check`: Run various code and documentation style checks.
* `invoke docs`: Generate the component pages and build the documentation (needs `pip install -e ".[docs]"`).
* `invoke test`: Run all tests and checks in one swift command.
* `invoke`: Show available tasks.

## Documentation

`invoke docs` builds the documentation into `dist/docs`. It generates the Grasshopper component pages (from the
plugin's `Components/Specs/`) and the example pages (from `examples/`), and fails when their committed images in
`docs/_images/` no longer match their sources. The message says what to do; in short:

* **Grasshopper images** (component screenshots, the canvas and viewport of each `.ghx` example): install the current
  plugin in Rhino 8, open Rhino and Grasshopper, open `docs/_tools/capture_grasshopper.py` in Rhino's ScriptEditor and
  run it.
* **Python example outputs**: run `invoke docs-images`.

Then commit `docs/_images/`. A new Python example needs a docstring (title line, blank line, paragraph); a new
Grasshopper example is saved as `.ghx` and needs a walkthrough in `docs/grasshopper/walkthroughs/`. The details are in
[Contributing to the documentation](https://gramaziokohler.github.io/compas_acoustics/latest/contributing/docs/)
(`docs/contributing/docs.md`).

## Bug reports

When [reporting a bug](https://github.com/gramaziokohler/compas_acoustics/issues) please include:

* Operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

## Feature requests

When [proposing a new feature](https://github.com/gramaziokohler/compas_acoustics/issues) please include:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
