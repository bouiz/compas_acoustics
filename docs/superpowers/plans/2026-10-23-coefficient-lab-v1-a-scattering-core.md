# Coefficient Lab V1 — A: Scattering Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python package `compas_acoustics.lab` that turns a 3D surface (COMPAS `Mesh`) into random-incidence scattering coefficients per band, with two interchangeable free-field solvers: `kirchhoff` (fast) and `bem` (mesh2scattering/NumCalc).

**Architecture:** `ScatteringSample` builds closed sample and reference bodies. Source and receiver grids and a frequency grid define the virtual free-field set-up. A solver returns complex pressures p₁ (sample) and p₀ (reference). The Mommertz correlation gives s(θ,φ) per frequency; band averaging and Paris weighting give random-incidence s in one-third octaves, then folded to octaves. `measure_scattering` ties this together and returns a serialisable `ScatteringResult`, which also holds the polar levels the Rhino previews (sub-project C) will draw. Plain numpy/scipy/compas, except the BEM solver, which imports mesh2scattering lazily.

**Tech Stack:** Python ≥ 3.9 (BEM: ≥ 3.11), numpy, scipy (`scipy.spatial.SphericalVoronoi`), compas 2 (`Mesh`, `earclip_polygon`, `Data`), sofar, pyfar (`io.convert_sofa`), mesh2scattering ≥ 1.0.5 + trimesh (BEM only), pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-virtual-coefficient-lab-design.md` (sections "What the documents dictate", "Scattering methods", "Architecture → A", "Version split → V1").

Sub-projects B (absorption chamber + turntable) and C (Grasshopper/compas_pb) get their own plans after this one. This plan covers neither the turntable nor anything V2/V3 (s_d, SOFA export, curves).

## Global Constraints

- `import compas_acoustics` and `import compas_acoustics.lab` must work on Python 3.9 with **no** mesh2scattering, trimesh or spharpy. Only `lab/solvers/bem.py` imports mesh2scattering and trimesh, and only inside functions.
- BEM needs Python ≥ 3.11 and `mesh2scattering >= 1.0.5`. Ship them as the optional extra `lab` (`requirements-lab.txt`).
- Code style: ruff line length 179, target py39, numpy docstrings that cite the standard or paper and equation, force-single-line imports. No `X | Y` annotations evaluated at runtime.
- Serialisable classes subclass `_PackageDtype` + `compas.data.Data` and are exported from `compas_acoustics/__init__.py` (dtype `compas_acoustics/<Class>`).
- Long steps report through `compas_acoustics.worker.jobs.progress(stage, done, of)` (the pattern in `energy_map.py:261`). It is a no-op outside a worker job and raises `jobs.Cancelled` on cancel.
- Bands:
  - One-third octaves with nominal centres **100 … 5000 Hz by default** (`ISO_BAND_RANGE = (100, 5000)`, 18 bands; ISO 354 §5, ISO 17497-1/-2 §5).
    The **optional extension** is `EXTENDED_BAND_RANGE = (100, 8000)` (20 bands), beyond the standards. Any `band_range` must lie within 100…8000.
    The exact centres are `1000·10**(n/10)` (n = −10…9 for 100…8000).
  - Each band is simulated at 3 frequencies, `fc·2**(-1/9)`, `fc`, `fc·2**(1/9)`, and band values are their arithmetic mean (Heimes eq. 11; ISO 17497-2 §7.4.4 asks for ≥ 3 points per band).
  - Octave values are the mean of the one-third-octave values inside each octave (ISO 354 §8.3 / ISO 11654). The default range gives complete octaves 125…4000.
    The extended range adds an 8 kHz octave from 6300 and 8000 only (10 kHz is not simulated).
  - The material input needs all 7 octaves (`Defaults._DEFAULT_FREQUENCY_BANDS`, 125…8000). An octave that was not simulated takes the nearest simulated octave's value (in the default range: 8 kHz ← 4 kHz). The result's warnings say so.
  - Warning texts (exact):
    - `"bands above 5000 Hz are beyond ISO 354 and ISO 17497 (100-5000 Hz)"` when the range goes above 5000
    - `"the {f} Hz octave was not simulated; the material takes the {g} Hz value"` per filled octave
- Default source preset for `measure_scattering` is `"standard"` (97 sources), not the spec's Heimes grid (`"fine"`, 864 sources): the fine grid makes BEM runs ~9× longer and the Kirchhoff method no longer fast. **Deviation from the spec; review it.**
- Correlation (Heimes eq. 3): `s = 1 − |Σ p₁ p₀* w|² / (Σ|p₁|² w · Σ|p₀|² w)`, summed over receivers.
- Random incidence (Paris): `s_rand = Σ wᵢ sᵢ`. The weights are `wᵢ = (sin²θ_hi − sin²θ_lo)/n_az(ring)` per colatitude ring, so they sum to 1.
- Receiver spacing: `min(5°, 50·λ_min/d)` degrees (ISO 17497-2 §6.2.2; Embrechts eq. 6, Heimes eq. 6), where d is the sample's footprint diameter.
- Sample geometry (Heimes §3.2): the lowest point of the relief is at z = 0 and the reference's top is at z = 0. Both bodies are closed and sit on a baseplate of thickness `baseplate` (default 0.01 m) below z = 0. ISO 17497-1 §6.3.2 depth limit: `h ≤ d/16`, warn if not.
- Direction convention: a source direction `û` is the unit vector **from the sample towards the source**. A plane wave travels along `−û`, and the specular direction is û mirrored in z.
- Commit messages follow the repo style (a sentence, no `feat:` prefix) and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Rhino meshes with quads and n-gons and unwelded vertices** (Rhino exports split vertices at seams) should become a valid sample, not open seams or holes. Pinned in Task 5 (`test_disc_from_unwelded_quad_tile_is_closed`).
2. **A tile that does not cover the disc, or a relief with holes** should give a clear `ValueError`, not a silently open body that NumCalc chokes on hours later. Pinned in Task 5 (`test_disc_from_tile_with_hole_raises`, `test_as_drawn_two_boundaries_raises`).
3. **Geometry in millimetres** (Rhino model units) should be caught before a BEM launch: 1000× the size means 10⁶× the elements. Pinned in Task 7 (`test_element_estimate_far_above_limit_raises`) and Task 5 (`test_sample_larger_than_20_m_warns`).
4. **An interrupted or changed BEM run** should resume when the geometry is unchanged and never reuse stale results when it changed. Pinned in Task 7 (`test_cache_key_changes_when_sample_moves_1_mm`).
5. **Degenerate faces and grazing incidence** (zero-area triangles from Rhino, incidence near 90°) should give finite pressures, not NaN coefficients. Pinned in Task 6 (`test_degenerate_triangle_is_finite`) and Task 8 (`test_result_has_no_nan_with_degenerate_faces`).

---

### Task 1: Spike — NumCalc on macOS and mesh2scattering's run layout (throwaway)

The output is a findings note, not code. Tasks 7 and 9 read it.

**Files:**
- Create: `docs/superpowers/plans/2026-10-23-coefficient-lab-v1-a-spike-findings.md`
- Scratch scripts go in the session scratchpad. **Do not commit them.**

- [ ] **Step 1: Install the BEM stack in the dev env**

Run: `python -m pip install "mesh2scattering>=1.0.5" trimesh && python -c "import mesh2scattering as m; print(m.__version__); print(m.numcalc.build_or_fetch_numcalc())"`
Expected: a version ≥ 1.0.5 and the path to a built `NumCalc` binary inside the installed package (`make` with the Xcode command-line tools). Record any build errors and fixes.

- [ ] **Step 2: Run one tiny project and record the layout**

Write a flat closed disc (d = 0.4 m, 10 mm baseplate, edge length ≈ 1 cm) with `trimesh`. Create a `SampleMesh(mesh, SurfaceDescription(surface_type=FLAT), sample_baseplate_hight=0.01, sample_diameter=0.4)`, a `SoundSource(pf.Coordinates(0, 0, -1), SoundSourceType.PLANE_WAVE)`, and `EvaluationGrid.from_spherical(<200 points on the upper hemisphere at r = 10 m>, "grid")`. Call `write_scattering_project_numcalc(project_path, "flat", np.array([500., 1000.]), ...)`, then `manage_numcalc(project_path, numcalc_path=...)`, then `output.write_pressure(project_path)`.

Record in the findings note:
- (a) the files and folders NumCalc writes per frequency step, and which file marks a finished step
- (b) where `write_pressure` writes its SOFA, and the shape and order of `pf.io.convert_sofa(...)[0].freq`
- (c) whether rerunning `manage_numcalc` on a half-finished project skips finished steps (kill it mid-run to check)
- (d) whether `EvaluationGrid.from_spherical` keeps the input point order
- (e) whether `SoundSource` needs one `SoundSource` per direction or takes a multi-point `Coordinates`

- [ ] **Step 3: Time it at 8 kHz**

Use the same disc at d = 0.8 m and time two frequencies, the default range's top (5400 Hz, edge `343/(5400·6)` m) and the extension's top (8600 Hz), each with one source and 3000 receivers. Record the element count, wall time and peak RAM (`/usr/bin/time -l`).
Expected: numbers to put in the findings note. Task 7's default `max_elements` and the "overnight" estimate come from them.

- [ ] **Step 4: Commit the findings note only**

```bash
git add docs/superpowers/plans/2026-10-23-coefficient-lab-v1-a-spike-findings.md
git commit -m "Lab spike: NumCalc on macOS, mesh2scattering project layout and timings"
```

---

### Task 2: Lab package, frequency bands, optional extra

**Files:**
- Create: `src/compas_acoustics/lab/__init__.py`, `src/compas_acoustics/lab/bands.py`, `requirements-lab.txt`
- Modify: `pyproject.toml` (the `[tool.setuptools.dynamic]` table, currently `optional-dependencies = { dev = { file = "requirements-dev.txt" } }`)
- Test: `tests/lab/test_bands.py`, `tests/test_python39.py`

**Interfaces:**
- Produces:
  - `THIRD_OCTAVE_CENTRES: tuple[int, ...]`: all supported nominal centres 100…8000 (20 values).
  - `ISO_BAND_RANGE = (100, 5000)`, `EXTENDED_BAND_RANGE = (100, 8000)`.
  - `third_octave_centres(band_range=ISO_BAND_RANGE) -> tuple[np.ndarray, np.ndarray]`: (nominal, exact) for the bands within the range; `ValueError` outside 100…8000.
  - `simulation_frequencies(band_range=ISO_BAND_RANGE) -> np.ndarray`: 3 per band, ascending, shape `(3·n_bands,)`.
  - `band_average(values: np.ndarray) -> np.ndarray`: the last axis has length 3·n_bands → n_bands (mean of each consecutive triple).
  - `fold_to_octaves(values: np.ndarray, nominal: np.ndarray) -> tuple[list[int], np.ndarray]`: last axis one-third octaves → (the octave centres with at least one band, their means).
  - `fill_octaves(octaves: list[int], values: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]`: the 7 material octaves, missing ones from the nearest simulated octave, plus the `(filled, source)` pairs.

- [ ] **Step 1: Write the failing tests**

```python
def test_default_is_the_iso_range_100_to_5000():
    nominal, exact = third_octave_centres()
    assert list(nominal) == [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000]
    assert np.allclose(exact, 1000 * 10 ** (np.arange(-10, 8) / 10))

def test_extended_range_reaches_8000():
    nominal, _ = third_octave_centres(EXTENDED_BAND_RANGE)
    assert list(nominal[-3:]) == [5000, 6300, 8000] and len(nominal) == 20

def test_range_outside_100_to_8000_raises():
    with pytest.raises(ValueError, match="100.*8000"):
        third_octave_centres((100, 10000))

def test_three_frequencies_per_band_inside_the_band():
    f = simulation_frequencies()
    _, exact = third_octave_centres()
    assert f.shape == (54,)
    assert np.allclose(f[1::3], exact)
    assert np.allclose(f[0::3], exact * 2 ** (-1 / 9)) and np.allclose(f[2::3], exact * 2 ** (1 / 9))

def test_band_average_is_mean_of_triples():
    v = np.arange(6.0)
    assert np.allclose(band_average(v), [1.0, 4.0])

def test_default_range_folds_to_complete_octaves_125_to_4000():
    nominal, _ = third_octave_centres()
    octaves, values = fold_to_octaves(np.arange(18.0), nominal)  # value i for band i
    assert octaves == [125, 250, 500, 1000, 2000, 4000]
    assert values[0] == pytest.approx((0 + 1 + 2) / 3)     # 125 octave: 100, 125, 160
    assert values[5] == pytest.approx((15 + 16 + 17) / 3)  # 4000 octave: 3150, 4000, 5000

def test_extended_range_adds_an_8000_octave_from_two_bands():
    nominal, _ = third_octave_centres(EXTENDED_BAND_RANGE)
    octaves, values = fold_to_octaves(np.arange(20.0), nominal)
    assert octaves[-1] == 8000 and values[-1] == pytest.approx((18 + 19) / 2)  # 6300, 8000 (10 kHz not simulated)

def test_fill_octaves_takes_the_nearest_simulated_octave():
    filled, pairs = fill_octaves([125, 250, 500, 1000, 2000, 4000], np.arange(6.0))
    assert np.allclose(filled, [0, 1, 2, 3, 4, 5, 5]) and pairs == [(8000, 4000)]
```

Add to `tests/test_python39.py`:

```python
def test_the_lab_imports_without_bem_dependencies():
    code = (
        "import sys\n"
        "for name in ('mesh2scattering', 'trimesh', 'spharpy'):\n"
        "    sys.modules[name] = None\n"
        "import compas_acoustics.lab\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
```

- [ ] **Step 2: Run them and see them fail**

Run: `pytest tests/lab/test_bands.py tests/test_python39.py -v`
Expected: FAIL (`ModuleNotFoundError: compas_acoustics.lab`).

- [ ] **Step 3: Implement `lab/bands.py` and the package**

- `lab/__init__.py` re-exports the public names of each module as later tasks add them. Start with the band functions.
- `requirements-lab.txt`:
  ```
  # The BEM scattering solver (compas_acoustics.lab, method="bem"): mesh2scattering builds and runs NumCalc
  mesh2scattering >=1.0.5; python_version >= "3.11"
  trimesh; python_version >= "3.11"
  ```
- `pyproject.toml`: `optional-dependencies = { dev = { file = "requirements-dev.txt" }, lab = { file = "requirements-lab.txt" } }`.

- [ ] **Step 4: Run them and see them pass**

Run: `pytest tests/lab/test_bands.py tests/test_python39.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/compas_acoustics/lab requirements-lab.txt pyproject.toml tests/lab/test_bands.py tests/test_python39.py
git commit -m "Lab: one-third-octave simulation grid, band averaging and octave folding; the lab extra for the BEM solver"
```

---

### Task 3: Source and receiver grids

**Files:**
- Create: `src/compas_acoustics/lab/grids.py`
- Test: `tests/lab/test_grids.py`

**Interfaces:**
- Produces:
  - `SourceGrid(colatitude: np.ndarray, azimuth: np.ndarray)`, both in degrees and one entry per source. Members:
    - `SourceGrid.preset(name: str, symmetry: str = "none") -> SourceGrid`: name in `{"survey", "standard", "fine"}`, symmetry in `{"none", "rotational"}`.
    - `.directions -> np.ndarray (n, 3)`: unit vectors û towards the sources.
    - `.paris_weights -> np.ndarray (n,)`: sums to 1.
    - `.n`
  - `ReceiverGrid(directions: np.ndarray (n, 3), weights: np.ndarray (n,), radius: float)`, with members:
    - `ReceiverGrid.hemisphere(spacing_deg: float, radius: float) -> ReceiverGrid`
    - `.points -> np.ndarray (n, 3)`: `directions * radius`
  - `receiver_spacing(diameter: float, f_max: float, c: float) -> float` (degrees).

The presets come from the spec:
- `survey`: ISO 17497-2 Table 1 (colatitude 0°, then 30° and 60° each at azimuths 0, 60, …, 300) = 13 sources.
- `standard`: colatitude 0, 10, …, 80 at azimuth step 30° (one source at 0°) = 97.
- `fine`: Heimes's colatitudes 1.26°…88.74° in steps of 2.52° (36) × azimuth step 15° (24) = 864.
- `rotational` keeps azimuth 0 only, giving 3, 9 and 36 sources.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("name, n_none, n_rot", [("survey", 13, 3), ("standard", 97, 9), ("fine", 864, 36)])
def test_preset_sizes(name, n_none, n_rot):
    assert SourceGrid.preset(name).n == n_none
    assert SourceGrid.preset(name, symmetry="rotational").n == n_rot

def test_paris_weights_sum_to_one_and_follow_sin2():
    g = SourceGrid.preset("survey")
    assert g.paris_weights.sum() == pytest.approx(1.0)
    # ring bounds at 0, 15, 45, 90 degrees -> sin^2 differences, split over each ring's azimuths
    assert g.paris_weights[g.colatitude == 0].sum() == pytest.approx(np.sin(np.radians(15)) ** 2)
    assert g.paris_weights[g.colatitude == 60].sum() == pytest.approx(1 - np.sin(np.radians(45)) ** 2)

def test_directions_point_up_towards_the_source():
    g = SourceGrid(np.array([0.0, 60.0]), np.array([0.0, 90.0]))
    assert np.allclose(g.directions, [[0, 0, 1], [0, np.sin(np.radians(60)), 0.5]])

def test_receiver_weights_cover_the_hemisphere():
    r = ReceiverGrid.hemisphere(spacing_deg=5.0, radius=10.0)
    assert (r.directions[:, 2] > 0).all()
    assert r.weights.sum() == pytest.approx(2 * np.pi, rel=1e-6)
    assert np.allclose(np.linalg.norm(r.points, axis=1), 10.0)

def test_receiver_spacing_matches_heimes_example():
    # Heimes sec. 3: d = 0.8 m, c = 346 m/s -> 5 degrees is valid up to 4.3 kHz
    assert receiver_spacing(0.8, 4300.0, 346.0) == pytest.approx(5.0, abs=0.05)
    assert receiver_spacing(0.8, 1000.0, 346.0) == 5.0  # never coarser than ISO 17497-2's 5 degrees
```

- [ ] **Step 2: Run them and see them fail**

Run: `pytest tests/lab/test_grids.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `lab/grids.py`**

- `ReceiverGrid.hemisphere`: a Fibonacci (golden-angle) spiral of `N = ceil(2π/Δ²)` points on the upper hemisphere, with Δ the spacing in radians.
- The Voronoi weights come from `scipy.spatial.SphericalVoronoi` of the points **together with their mirror images in z = 0**, so the sphere tiling is symmetric. Keep the areas of the upper points.
- Warn (`warnings.warn`) when N exceeds 50 000.

- [ ] **Step 4: Run them and see them pass**

Run: `pytest tests/lab/test_grids.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/compas_acoustics/lab/grids.py src/compas_acoustics/lab/__init__.py tests/lab/test_grids.py
git commit -m "Lab: source presets (ISO 17497-2 Table 1, standard, Heimes fine) with Paris weights; Voronoi-weighted receiver hemisphere"
```

---

### Task 4: Correlation scattering and random-incidence averaging

**Files:**
- Create: `src/compas_acoustics/lab/coefficients.py`
- Test: `tests/lab/test_coefficients.py`

**Interfaces:**
- Consumes: `SourceGrid.paris_weights` (Task 3).
- Produces:
  - `scattering_freefield(p_sample: np.ndarray, p_reference: np.ndarray, weights: np.ndarray, axis: int = 1) -> np.ndarray`: complex arrays with receivers along `axis`; returns s with that axis removed.
  - `random_incidence(s_angle: np.ndarray, source_grid: SourceGrid) -> np.ndarray`: sources on axis 0.

- [ ] **Step 1: Write the failing tests**

```python
rng = np.random.default_rng(0)

def test_identical_patterns_do_not_scatter():
    p = rng.normal(size=(2, 500, 3)) + 1j * rng.normal(size=(2, 500, 3))
    assert np.allclose(scattering_freefield(p, p, np.ones(500)), 0.0, atol=1e-12)

def test_a_common_phase_and_gain_do_not_scatter():
    p = rng.normal(size=(1, 500, 1)) + 1j * rng.normal(size=(1, 500, 1))
    assert np.allclose(scattering_freefield(3.0 * np.exp(1j * 0.7) * p, p, np.ones(500)), 0.0, atol=1e-12)

def test_uncorrelated_patterns_scatter_almost_everything():
    p0 = rng.normal(size=(1, 20000, 1)) + 1j * rng.normal(size=(1, 20000, 1))
    p1 = rng.normal(size=(1, 20000, 1)) + 1j * rng.normal(size=(1, 20000, 1))
    assert scattering_freefield(p1, p0, np.ones(20000))[0, 0] > 0.99

def test_random_incidence_of_constant_is_constant():
    g = SourceGrid.preset("standard")
    assert np.allclose(random_incidence(np.full((g.n, 4), 0.37), g), 0.37)

def test_random_incidence_of_sin2_is_one_half():
    # (1/pi) * integral of sin^2(theta) cos(theta) sin(theta) over the hemisphere = 1/2
    g = SourceGrid.preset("fine")
    s = np.sin(np.radians(g.colatitude))[:, None] ** 2
    assert random_incidence(s, g)[0] == pytest.approx(0.5, abs=1e-2)
```

- [ ] **Step 2: Run them and see them fail**

Run: `pytest tests/lab/test_coefficients.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement both functions in `lab/coefficients.py`**

Use the formulas in Global Constraints. The docstrings cite Heimes eq. 3 / Mommertz (2000) and Paris (Kuttruff). Do not import mesh2scattering: its `scattering_freefield` needs pyfar objects and an extra dependency, while this one works on plain arrays.

- [ ] **Step 4: Run them and see them pass**

Run: `pytest tests/lab/test_coefficients.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/compas_acoustics/lab/coefficients.py src/compas_acoustics/lab/__init__.py tests/lab/test_coefficients.py
git commit -m "Lab: Mommertz correlation scattering coefficient and Paris random-incidence average on plain arrays"
```

---

### Task 5: Scattering sample — tile + crop to a disc, or as drawn

**Files:**
- Create: `src/compas_acoustics/lab/sample.py`
- Test: `tests/lab/test_sample.py`

**Interfaces:**
- Produces: `ScatteringSample(_PackageDtype, Data)` with
  - attributes:
    - `sample: Mesh` (closed, triangulated) and `reference: Mesh` (closed, triangulated)
    - `mode: str` (`"disc"` | `"as_drawn"`)
    - `diameter: float`: the footprint diameter, i.e. the disc diameter or, as drawn, the largest xy distance between outline vertices
    - `period: tuple[float, float] | None`
    - `structural_depth: float`: max z of the relief, with min z = 0
    - `feature_size: float | None`
    - `symmetry: str`
    - `baseplate: float`
    - `warnings: list[str]`
  - constructors:
    - `ScatteringSample.disc(tile: Mesh, diameter: float, period=None, baseplate=0.01, segments=128, offset=(0.0, 0.0), symmetry="none", feature_size=None)`
    - `ScatteringSample.as_drawn(mesh: Mesh, baseplate=0.01, symmetry="none", feature_size=None)`
  - `__data__` / `__from_data__`.

The algorithm, where the signature leaves a choice:
- **Clean:** triangulate (`Mesh.quads_to_triangles` or a fan per face) and weld vertices within 1e-6 m.
- **Tile (disc mode):** take `period` from the arguments, or else the tile's xy bounding box. Repeat the tile over a grid that covers `[−d/2, d/2]²` shifted by `offset`, then weld again.
- **Clip:** clip every triangle against the `segments` vertical half-spaces of a regular polygon inscribed in the circle of radius d/2 (Sutherland–Hodgman in 3D, one plane at a time). Re-triangulate the clipped polygons by fan.
- **Close:** take the boundary edges (edges used by one face) and order them into loops. There must be **exactly one loop**; otherwise raise `ValueError("...one closed outline...")`. Shift z so that the relief minimum is 0. Add vertical side quads from the loop down to `z = −baseplate`, plus a bottom from `compas.geometry.earclip_polygon(Polygon(loop_xy))` (it takes a `Polygon`, not a point list), wound so the normals point outward.
- **Reference:** the same outline loop's xy at z = 0 (earclipped top), the same side walls and the same bottom.
- **As drawn:** a closed input mesh is used unchanged, after triangulating, welding and shifting min z to 0. Its reference is the earclipped xy outline of its lowest boundary, as a flat body. An open input is closed as above.
- **Warnings:**
  - `structural_depth > diameter/16` → "ISO 17497-1 6.3.2: structural depth … exceeds d/16"
  - `diameter > 20` → "the sample is … m across; is the model in millimetres?"

- [ ] **Step 1: Write the failing tests**

```python
def flat_tile(size=0.1):
    return Mesh.from_vertices_and_faces([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]], [[0, 1, 2, 3]])

def test_flat_disc_is_closed_and_has_the_disc_volume():
    s = ScatteringSample.disc(flat_tile(), diameter=0.8, baseplate=0.01, segments=256)
    assert s.sample.is_closed() and s.reference.is_closed()
    assert volume_polyhedron(s.sample.to_vertices_and_faces()) == pytest.approx(np.pi * 0.4 ** 2 * 0.01, rel=1e-3)
    assert s.structural_depth == pytest.approx(0.0, abs=1e-9)

def test_relief_minimum_and_reference_top_are_at_z0():
    s = ScatteringSample.disc(sine_tile(period=0.0708, depth=0.0204), diameter=0.8)
    zs = np.array([s.sample.vertex_coordinates(v)[2] for v in s.sample.vertices()])
    assert zs[zs > -1e-9].min() == pytest.approx(0.0, abs=1e-9)
    assert s.structural_depth == pytest.approx(0.0204, rel=1e-2)
    assert max(s.reference.vertex_coordinates(v)[2] for v in s.reference.vertices()) == pytest.approx(0.0)

def test_reference_has_the_sample_footprint():
    s = ScatteringSample.disc(sine_tile(period=0.0708, depth=0.0204), diameter=0.8)
    assert footprint_area(s.sample) == pytest.approx(footprint_area(s.reference), rel=1e-9)

def test_disc_from_unwelded_quad_tile_is_closed():
    # two quads sharing an edge, but with duplicated vertices along it (as Rhino exports split seams)
    tile = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [0.05, 0, 0.01], [0.05, 0.1, 0.01], [0, 0.1, 0],
         [0.05, 0, 0.01], [0.1, 0, 0], [0.1, 0.1, 0], [0.05, 0.1, 0.01]],
        [[0, 1, 2, 3], [4, 5, 6, 7]])
    assert ScatteringSample.disc(tile, diameter=0.5).sample.is_closed()

def test_disc_from_tile_with_hole_raises():
    with pytest.raises(ValueError, match="one closed outline"):
        ScatteringSample.disc(tile_with_hole(), diameter=0.5)

def test_as_drawn_open_patch_is_closed_with_matching_reference():
    s = ScatteringSample.as_drawn(open_l_shaped_relief())
    assert s.mode == "as_drawn" and s.sample.is_closed() and s.reference.is_closed()
    assert footprint_area(s.sample) == pytest.approx(footprint_area(s.reference), rel=1e-9)

def test_as_drawn_two_boundaries_raises():
    with pytest.raises(ValueError, match="one closed outline"):
        ScatteringSample.as_drawn(two_separate_patches())

def test_deep_relief_warns_iso_17497_1():
    s = ScatteringSample.disc(sine_tile(period=0.1, depth=0.1), diameter=0.8)
    assert any("d/16" in w for w in s.warnings)

def test_sample_larger_than_20_m_warns():
    s = ScatteringSample.disc(flat_tile(100.0), diameter=800.0)
    assert any("millimetres" in w for w in s.warnings)

def test_sample_round_trips_through_json():
    s = ScatteringSample.disc(flat_tile(), diameter=0.4)
    t = json_loads(json_dumps(s))
    assert isinstance(t, ScatteringSample) and t.diameter == 0.4 and t.sample.number_of_faces() == s.sample.number_of_faces()
```

Put the helpers in `tests/lab/conftest.py`:
- `sine_tile(period, depth, n=16)`: one period of `z = depth/2·(1 + sin(2πx/period))` as an n×1 quad strip, 0.05 m deep in y.
- `tile_with_hole()`: a 3×3 quad grid without its centre face.
- `open_l_shaped_relief()`: an L-shaped quad patch with a bump.
- `two_separate_patches()`: two quads that don't touch.
- `footprint_area(mesh)`: the summed xy-projected area of the faces whose normal has z > 0.

- [ ] **Step 2: Run them and see them fail**

Run: `pytest tests/lab/test_sample.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `lab/sample.py`**

Follow the algorithm above. Keep each phase a private function (`_clean`, `_tile`, `_clip_to_polygon`, `_boundary_loops`, `_close`, `_flat_body`) so the file reads top-down.

- [ ] **Step 4: Run them and see them pass**

Run: `pytest tests/lab/test_sample.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/compas_acoustics/lab/sample.py src/compas_acoustics/lab/__init__.py tests/lab/test_sample.py tests/lab/conftest.py
git commit -m "Lab: ScatteringSample, a tile cropped to a disc or a mesh as drawn, closed on a baseplate with its flat reference"
```

---

### Task 6: Solver interface and the Kirchhoff solver

**Files:**
- Create: `src/compas_acoustics/lab/solvers/__init__.py`, `src/compas_acoustics/lab/solvers/base.py`, `src/compas_acoustics/lab/solvers/kirchhoff.py`
- Test: `tests/lab/test_kirchhoff.py`

**Interfaces:**
- Consumes: `ScatteringSample` (Task 5), `SourceGrid` and `ReceiverGrid` (Task 3).
- Produces:
  - `Pressures(sample: np.ndarray, reference: np.ndarray, frequencies: np.ndarray, generation: dict)`, a dataclass. Both arrays are complex with shape `(n_sources, n_receivers, n_frequencies)`.
  - `ScatteringSolver`, the abstract base:
    - `.name: str`
    - `.solve(sample: ScatteringSample, sources: SourceGrid, receivers: ReceiverGrid, frequencies: np.ndarray, c: float) -> Pressures`
    - `.warnings(sample, nominal_bands, exact_bands, c) -> list[str]`
  - `triangle_phase_integral(triangles: np.ndarray (n_t, 3, 3), q: np.ndarray (..., 3)) -> np.ndarray (..., n_t)`: `∫_T exp(j q·x) dS`.
  - `KirchhoffSolver(shadowing: bool = False, chunk: int = 2048)`, with `name = "kirchhoff"`.

Pressure formula (rigid physical optics; the surface field is 2pᵢ and ∂p/∂n = 0), for source direction û, receiver direction r̂ and k = 2πf/c:

```
p(r̂) = (j k / 2π) · Σ_{faces with n̂·û > 0} (n̂·r̂) · I_T(q),   q = −k (û + r̂)
```

- This uses the rigid (Neumann) obliquity factor n̂·r̂ instead of the spec's Beckmann form n̂·(k̂ᵢ − k̂ₛ). Both are equal at the specular direction, and the correlation is insensitive to the common factor. The docstring says so.
- Faces of both closed bodies count, rim walls included, so the rim terms of sample and reference match.
- With `shadowing=True`, a face also needs its centroid's ray towards û to hit no other face (vectorised Möller–Trumbore).

`triangle_phase_integral`, from Hermite–Genocchi with α, β, γ = q·a, q·b, q·c and area A:

```
I = −2A · ( e^{jα}/((α−β)(α−γ)) + e^{jβ}/((β−α)(β−γ)) + e^{jγ}/((γ−α)(γ−β)) )
```

When phases coincide or nearly coincide, use confluent or series forms. The limit at α = β = γ is `A·e^{jα}`. The required accuracy is pinned by the test below. Faces with zero area contribute 0.

Validity warnings (`KirchhoffSolver.warnings`):
- For bands where `c/f > feature_size` (with `feature_size = sample.feature_size or min(sample.period)`; skipped when both are None): "kirchhoff: below {f} Hz the features ({x} m) are smaller than the wavelength; s is unreliable there".
- If more than 10 % of the relief's top area has faces tilted more than 45°: "kirchhoff: steep relief; multiple reflections are ignored".

- [ ] **Step 1: Write the failing tests**

```python
TRI = np.array([[[0.0, 0.0, 0.0], [0.3, 0.0, 0.05], [0.1, 0.2, -0.02]]])

def brute(tri, q, n=400):
    # midpoint rule on the n^2 congruent sub-triangles of a uniform barycentric subdivision
    # (written out in the test file; it is the reference, not the code under test)

@pytest.mark.parametrize("scale", [0.0, 1e-7, 1e-4, 1e-2, 1.0, 50.0])
def test_triangle_integral_matches_brute_force(scale):
    q = scale * np.array([[3.0, -1.0, 2.0]])
    assert np.allclose(triangle_phase_integral(TRI, q), brute(TRI[0], q[0]), rtol=1e-6, atol=1e-12)

def test_triangle_integral_two_equal_phases():
    tri = np.array([[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.1, 0.2, 0.0]]])
    q = np.array([[0.0, 40.0, 0.0]])  # vertices 0 and 1 share y = 0 -> two equal phases
    assert np.allclose(triangle_phase_integral(tri, q), brute(tri[0], q[0]), rtol=1e-6)

def test_degenerate_triangle_is_finite():
    tri = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], dtype=float)
    out = triangle_phase_integral(tri, np.array([[5.0, 1.0, 0.0]]))
    assert np.isfinite(out).all() and abs(out[0, 0]) < 1e-12

def test_flat_disc_against_its_own_reference_does_not_scatter():
    s = ScatteringSample.disc(flat_tile(), diameter=0.8)
    src = SourceGrid.preset("survey", symmetry="rotational")
    rec = ReceiverGrid.hemisphere(5.0, 10.0)
    p = KirchhoffSolver().solve(s, src, rec, np.array([500.0, 2000.0]), 343.0)
    assert np.allclose(scattering_freefield(p.sample, p.reference, rec.weights), 0.0, atol=1e-6)

def test_large_plate_reflects_specularly():
    s = ScatteringSample.disc(flat_tile(), diameter=4.0)
    src = SourceGrid(np.array([30.0]), np.array([0.0]))
    rec = ReceiverGrid.hemisphere(2.0, 10.0)
    p = KirchhoffSolver().solve(s, src, rec, np.array([4000.0]), 343.0)
    peak = rec.directions[np.argmax(np.abs(p.reference[0, :, 0]))]
    mirror = src.directions[0] * [-1, -1, 1]
    assert np.degrees(np.arccos(np.clip(peak @ mirror, -1, 1))) < 2.0

def test_sine_surface_scatters_above_the_first_grating_lobe():
    # Heimes: h = 20.4 mm, Lambda = 70.8 mm; at 31.5 deg the first lobe appears at Lambda/lambda = 1/(1 + sin 31.5) = 0.66
    s = ScatteringSample.disc(sine_tile(period=0.0708, depth=0.0204), diameter=0.8)
    src = SourceGrid(np.array([31.5]), np.array([0.0]))  # incidence plane across the grooves
    rec = ReceiverGrid.hemisphere(2.0, 10.0)
    c = 343.0
    f = np.array([0.4, 1.0]) * c / 0.0708
    p = KirchhoffSolver().solve(s, src, rec, f, c)
    s_low, s_high = scattering_freefield(p.sample, p.reference, rec.weights)[0]
    assert s_low < 0.2 and s_high > 0.4

def test_warns_below_the_feature_scale():
    s = ScatteringSample.disc(sine_tile(period=0.0708, depth=0.0204), diameter=0.8, period=(0.0708, 0.05))
    nominal, exact = third_octave_centres()
    msgs = KirchhoffSolver().warnings(s, nominal, exact, 343.0)
    assert any("kirchhoff: below" in m for m in msgs)
```

(The intent is pinned: integrals over exactly and nearly coincident phases agree with a brute-force reference to 1e-6 relative.)

- [ ] **Step 2: Run them and see them fail**

Run: `pytest tests/lab/test_kirchhoff.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `solvers/base.py` (Pressures, ScatteringSolver) and `solvers/kirchhoff.py`**

- Vectorise over faces × receivers, in chunks of `chunk` receivers.
- Report `jobs.progress("kirchhoff", done, of)` per (source, frequency) pair.
- `generation = {"method": "kirchhoff", "shadowing": ..., "faces": ...}`.
- `solvers/__init__.py` exports `Pressures`, `ScatteringSolver` and `KirchhoffSolver`, and **not** `BemSolver` (so 3.9 imports stay clean).

- [ ] **Step 4: Run them and see them pass**

Run: `pytest tests/lab/test_kirchhoff.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/compas_acoustics/lab/solvers tests/lab/test_kirchhoff.py
git commit -m "Lab: solver interface and the Kirchhoff (physical optics) free-field solver with closed-form triangle integrals"
```

---

### Task 7: BEM solver (mesh2scattering / NumCalc)

**Files:**
- Create: `src/compas_acoustics/lab/solvers/bem.py`, `src/compas_acoustics/lab/solvers/_numcalc.py` (a `python -m` entry point that runs `manage_numcalc` for one folder)
- Test: `tests/lab/test_bem.py`

**Interfaces:**
- Consumes: `Pressures` and `ScatteringSolver` (Task 6), `ScatteringSample` (Task 5), the grids (Task 3), and the spike findings (Task 1: step marker files, SOFA path and order, resume behaviour, source layout).
- Produces:
  - `BemSolver(cache_dir: str | None = None, numcalc_path: str | None = None, max_elements: int = <from spike, default 100_000>, elements_per_wavelength: int = 6, density: float = 1.1839)`, with `name = "bem"`.
    - `cache_dir` defaults to `~/.compas_acoustics/lab_cache`.
    - `numcalc_path` defaults to `$COMPAS_ACOUSTICS_NUMCALC`, or else `mesh2scattering.numcalc.build_or_fetch_numcalc()`.
  - `plan_splits(frequencies: np.ndarray, nominal_octaves=Defaults._DEFAULT_FREQUENCY_BANDS) -> list[np.ndarray]`: the simulation frequencies grouped by octave band, low to high.
  - `estimate_elements(mesh: Mesh, edge: float) -> int`: `ceil(area / (√3/4 · edge²))`.
  - `cache_key(sample, sources, receivers, frequencies, c, edge) -> str`: 12 hex characters of a sha1 over vertex coordinates rounded to 1e-6 m, faces, directions, frequencies, c and edge.

Behaviour:
- **Missing dependencies:** if mesh2scattering is absent or Python < 3.11, `solve` raises `ImportError` whose message names `pip install "compas_acoustics[lab]"` and Python ≥ 3.11.
- **Per split:** edge = `c / (f_max_split · elements_per_wavelength)`. Remesh both bodies with `trimesh.remesh.subdivide_to_size`.
  - If the estimate exceeds `max_elements`, warn and suggest a smaller diameter or the Kirchhoff method.
  - If it exceeds `10 × max_elements`, raise `ValueError` before writing anything.
- **Projects:** `<cache_dir>/<key>/split_<k>/{sample,reference}`. Skip writing when the project's `parameters.json` exists.
  - Source: plane waves travelling along `−û`.
  - Evaluation grid: `receivers.points` at radius `max(10·d, 2·d²/λ_min)`.
  - `SurfaceDescription(structural_wavelength_x/y = period or 0, structural_depth, surface_type = PERIODIC if period else STOCHASTIC, FLAT for the reference)`.
- **Run:** `subprocess.Popen([sys.executable, "-m", "compas_acoustics.lab.solvers._numcalc", folder, numcalc_path])`.
  - Every 2 s, count the finished steps (spike finding (a)) and call `jobs.progress("bem: split k/K", done, of)`.
  - On `jobs.Cancelled`, kill the process tree (`psutil`, a dependency of mesh2scattering) and re-raise. Finished steps stay on disk.
- **Read:** `output.write_pressure(project)`, then `sofar.read_sofa` + `pyfar.io.convert_sofa`. Reorder receivers to `receivers.points` by nearest neighbour, asserting a max distance < 1e-6·radius. Concatenate the splits along frequency.
- **Generation:** `generation = {"method": "bem", "mesh2scattering": version, "elements": [per split], "edge": [per split], "cache": path}`.

- [ ] **Step 1: Write the failing fast tests**

```python
def test_splits_group_simulation_frequencies_by_octave():
    f = simulation_frequencies()
    splits = plan_splits(f)
    assert len(splits) == 6 and all(len(s) == 9 for s in splits)  # default: complete octaves 125-4000
    ext = plan_splits(simulation_frequencies(EXTENDED_BAND_RANGE))
    assert len(ext) == 7 and len(ext[-1]) == 6                      # extended: an 8 kHz split from 6300 and 8000
    assert np.all(np.diff(np.concatenate(splits)) > 0)

def test_element_estimate_scales_with_area():
    plate = ScatteringSample.disc(flat_tile(), diameter=0.8).reference
    # top + bottom discs + rim wall of the 10 mm baseplate
    assert estimate_elements(plate, 0.01) == pytest.approx((2 * np.pi * 0.16 + np.pi * 0.8 * 0.01) / (np.sqrt(3) / 4 * 1e-4), rel=0.02)

def test_cache_key_changes_when_sample_moves_1_mm():
    a = ScatteringSample.disc(sine_tile(0.0708, 0.0204), diameter=0.4)
    b = ScatteringSample.disc(sine_tile(0.0708, 0.0204), diameter=0.4, offset=(0.001, 0.0))
    args = (SourceGrid.preset("survey"), ReceiverGrid.hemisphere(5.0, 10.0), simulation_frequencies(), 343.0, 0.007)
    assert cache_key(a, *args) == cache_key(ScatteringSample.disc(sine_tile(0.0708, 0.0204), diameter=0.4), *args)
    assert cache_key(a, *args) != cache_key(b, *args)

def test_element_estimate_far_above_limit_raises(tmp_path):
    big = ScatteringSample.disc(flat_tile(100.0), diameter=800.0)  # a model left in millimetres
    with pytest.raises(ValueError, match="elements"):
        BemSolver(cache_dir=str(tmp_path)).solve(big, SourceGrid.preset("survey", "rotational"), ReceiverGrid.hemisphere(5.0, 10.0), np.array([8000.0]), 343.0)

def test_missing_mesh2scattering_names_the_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mesh2scattering", None)
    with pytest.raises(ImportError, match=r"compas_acoustics\[lab\]"):
        BemSolver(cache_dir=str(tmp_path)).solve(
            ScatteringSample.disc(flat_tile(), diameter=0.3), SourceGrid.preset("survey", "rotational"),
            ReceiverGrid.hemisphere(5.0, 10.0), np.array([500.0]), 343.0)
```

The size check runs before importing mesh2scattering, so the millimetre test needs no NumCalc. Put the `ValueError` check first in `solve`.

- [ ] **Step 2: Write the failing slow test (skipped without mesh2scattering)**

```python
@pytest.mark.slow
def test_bem_sine_surface_scatters_above_the_first_grating_lobe(tmp_path):
    pytest.importorskip("mesh2scattering")
    s = ScatteringSample.disc(sine_tile(period=0.0708, depth=0.0204), diameter=0.4)
    src = SourceGrid(np.array([31.5]), np.array([0.0]))
    rec = ReceiverGrid.hemisphere(5.0, 10.0)
    f = np.array([0.4, 1.0]) * 343.0 / 0.0708
    p = BemSolver(cache_dir=str(tmp_path)).solve(s, src, rec, f, 343.0)
    s_low, s_high = scattering_freefield(p.sample, p.reference, rec.weights)[0]
    assert s_low < 0.2 and s_high > 0.4
    # a rerun reads the cache instead of solving again
    t0 = time.perf_counter(); BemSolver(cache_dir=str(tmp_path)).solve(s, src, rec, f, 343.0)
    assert time.perf_counter() - t0 < 30
```

- [ ] **Step 3: Run them and see them fail**

Run: `pytest tests/lab/test_bem.py -v -m "slow or not slow"`
Expected: FAIL (`ImportError: compas_acoustics.lab.solvers.bem`).

- [ ] **Step 4: Implement `solvers/bem.py` and `solvers/_numcalc.py`**

Follow the behaviour above. The mesh2scattering and trimesh imports go inside `solve` and the helpers that need them.

- [ ] **Step 5: Run them and see them pass**

Run: `pytest tests/lab/test_bem.py -v -m "slow or not slow"`
Expected: PASS (the slow test takes minutes; record its time in the commit message).

- [ ] **Step 6: Commit**

```bash
git add src/compas_acoustics/lab/solvers/bem.py src/compas_acoustics/lab/solvers/_numcalc.py tests/lab/test_bem.py
git commit -m "Lab: BEM solver over mesh2scattering/NumCalc, octave splits meshed at lambda/6, cached and resumable, cancellable"
```

---

### Task 8: `measure_scattering` and `ScatteringResult`

**Files:**
- Create: `src/compas_acoustics/lab/results.py`, `src/compas_acoustics/lab/scattering.py`
- Modify: `src/compas_acoustics/lab/__init__.py`, `src/compas_acoustics/__init__.py` (export `ScatteringSample`, `ScatteringResult`; add them to `__all__`)
- Test: `tests/lab/test_scattering.py`

**Interfaces:**
- Consumes: everything above. `BemSolver` is imported inside `measure_scattering` only when `method == "bem"`.
- Produces:
  - `ScatteringResult(_PackageDtype, Data)` with attributes:
    - `name`, `method`
    - `bands: list[int]` (nominal one-third octaves) and `s_bands: list[float]`
    - `octave_bands: list[int]` (the simulated octaves: 125…4000 by default, 125…8000 extended) and `s_octaves: list[float]`
    - `filled_octaves: list[int]` (material octaves taken from a neighbour)
    - `s_angle: np.ndarray (n_src, n_bands)`
    - `source_colatitude`, `source_azimuth`, `source_weights`
    - `receiver_directions: np.ndarray (n_rec, 3)`
    - `polar_sample_db` and `polar_reference_db: np.ndarray (n_src, len(octave_bands), n_rec)` (float32)
    - `warnings: list[str]`, `generation: dict`
  - Its property `scattering_coefficients -> list[float]` gives the 7 material octaves (125…8000, via `fill_octaves`), the input sub-project B's material step uses.
  - `measure_scattering(sample: ScatteringSample, method: str = "kirchhoff", preset: str = "standard", band_range=ISO_BAND_RANGE, air: AirConditions | None = None, name: str | None = None, **solver_options) -> ScatteringResult`.
    Pass `band_range=EXTENDED_BAND_RANGE` for the extension to 8 kHz.
    - Supported methods: `{"kirchhoff", "bem"}`, otherwise `ValueError` naming them (sub-project B adds `"turntable"`).
    - c comes from `(air or AirConditions()).speed_of_sound`.

Pipeline:
1. Frequencies come from `simulation_frequencies(band_range)`. Receivers are `ReceiverGrid.hemisphere(receiver_spacing(sample.diameter, f.max(), c), radius=10·sample.diameter)`. Sources are `SourceGrid.preset(preset, sample.symmetry)`.
2. `pressures = solver.solve(...)`. Non-finite values → `RuntimeError` naming the method.
3. `s_f = scattering_freefield(p.sample, p.reference, rec.weights)` gives shape (n_src, n_f). Then `s_angle = band_average(s_f)`, `s_bands = random_incidence(s_angle, src)` and `s_octaves = fold_to_octaves(s_bands, nominal)`.
4. Polar levels: per octave band, take the mean of |p|² over the simulation frequencies inside that octave. Convert to dB relative to the reference's maximum at the same source and band.
5. `warnings = sample.warnings + solver.warnings(...)`, plus the band warnings from Global Constraints (beyond ISO; filled octaves). Also re-emit each one through `warnings.warn` so the worker's `_messages` collects them.
6. `generation = {"method", "preset", "band_range", "c", "receivers": n, "sources": n, **pressures.generation}`.

- [ ] **Step 1: Write the failing tests**

```python
class FakeSolver(ScatteringSolver):
    name = "fake"
    def solve(self, sample, sources, receivers, frequencies, c):
        rng = np.random.default_rng(1)
        p0 = rng.normal(size=(sources.n, len(receivers.weights), len(frequencies))) + 0j
        return Pressures(p0.copy(), p0, frequencies, {"method": "fake"})
    def warnings(self, *args):
        return []

def test_shapes_and_bands(monkeypatch):
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: FakeSolver())
    r = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), method="kirchhoff", preset="survey")
    assert r.bands == list(THIRD_OCTAVE_CENTRES[:18]) and len(r.s_bands) == 18  # default: ISO 100-5000
    assert r.octave_bands == [125, 250, 500, 1000, 2000, 4000] and len(r.scattering_coefficients) == 7
    assert r.filled_octaves == [8000] and r.scattering_coefficients[6] == r.scattering_coefficients[5]
    assert any("8000 Hz octave was not simulated" in w for w in r.warnings)
    assert r.s_angle.shape == (13, 18) and r.polar_sample_db.shape[:2] == (13, 6)
    assert np.allclose(r.s_octaves, 0.0, atol=1e-9)  # identical pressures -> no scattering
    assert np.isclose(r.polar_reference_db.max(axis=2), 0.0).all()

def test_extended_range_simulates_8000_and_warns_beyond_iso(monkeypatch):
    monkeypatch.setattr(scattering, "_solver", lambda method, **kw: FakeSolver())
    r = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), preset="survey", band_range=EXTENDED_BAND_RANGE)
    assert r.bands[-1] == 8000 and r.octave_bands[-1] == 8000 and r.filled_octaves == []
    assert any("beyond ISO" in w for w in r.warnings)

def test_unknown_method_names_the_supported_ones():
    with pytest.raises(ValueError, match="kirchhoff.*bem"):
        measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), method="turntable")

def test_kirchhoff_end_to_end_sine_beats_flat():
    flat = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), preset="survey", band_range=(1000, 4000))
    sine = measure_scattering(ScatteringSample.disc(sine_tile(0.0708, 0.0204), diameter=0.4), preset="survey", band_range=(1000, 4000))
    assert max(flat.s_bands) < 0.05 and sine.s_bands[-1] > 0.3

def test_result_has_no_nan_with_degenerate_faces():
    tile = flat_tile_with_zero_area_face()
    r = measure_scattering(ScatteringSample.disc(tile, diameter=0.4), preset="survey", band_range=(500, 1000))
    assert np.isfinite(r.s_bands).all()

def test_result_round_trips_through_json():
    r = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.3), preset="survey", band_range=(500, 1000))
    t = json_loads(json_dumps(r))
    assert isinstance(t, ScatteringResult) and np.allclose(t.s_angle, r.s_angle) and t.generation["method"] == "kirchhoff"

def test_classes_are_exported_from_the_package():
    import compas_acoustics
    assert compas_acoustics.ScatteringResult is ScatteringResult and compas_acoustics.ScatteringSample is ScatteringSample
```

(`flat_tile_with_zero_area_face` goes in `tests/lab/conftest.py`: the flat tile plus a triangle with three collinear vertices. `band_range=(1000, 4000)` keeps the end-to-end tests fast; `band_range` slices the simulation grid.)

- [ ] **Step 2: Run them and see them fail**

Run: `pytest tests/lab/test_scattering.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `results.py`, `scattering.py` and the exports**

- `scattering._solver(method, **solver_options) -> ScatteringSolver` is the single place that maps method names to solvers. The tests patch it.
- `ScatteringResult.__data__` stores arrays as nested lists, the polar levels rounded to 0.1 dB (the ISO 17497-2 §8.5 presentation).

- [ ] **Step 4: Run them, plus the whole fast suite**

Run: `pytest tests/lab -v && pytest -m "not slow" -q`
Expected: PASS, with no regressions elsewhere (`tests/test_python39.py` included).

- [ ] **Step 5: Commit**

```bash
git add src/compas_acoustics/lab src/compas_acoustics/__init__.py tests/lab/test_scattering.py tests/lab/conftest.py
git commit -m "Lab: measure_scattering (kirchhoff or bem) returning a ScatteringResult with octave coefficients and polar levels"
```

---

### Task 9: Validation against Heimes & Vorländer, and Kirchhoff's measured accuracy

**Files:**
- Create: `tests/lab/test_validation_heimes.py`
- Modify: `src/compas_acoustics/lab/solvers/kirchhoff.py` (docstring: the measured accuracy)

**Interfaces:**
- Consumes: `measure_scattering`, `KirchhoffSolver`, `BemSolver`, the Task 1 spike timings.
- The helpers `heimes_scattering`, `heimes_scattering_random`, `bem_sine_s_angle`, `bem_sine_s_random` and `kirchhoff_vs_bem_gap` live in the test file. They read the SOFA files with `sofar` + `pyfar.io.convert_sofa`, convert frequency to Λ/λ with c = 346.18 m/s and Λ = 0.08 m, and average onto the same one-third-octave bands.

**Reference data:**
- Download `results_simulation_optimal.zip` from https://zenodo.org/records/15673498 (CC BY 4.0) into a directory outside the repo.
- Set `HEIMES_RESULTS=<extracted folder>`. The tests read `sine.scattering.sofa` (s per incidence angle, from `1.4_calc_scattering_from_s_d.py`) and `sine.scattering_rand.sofa`.
- The simulated sample: `sine` with Λ = 80 mm, n = 10, d = 0.8 m for Λ/λ ≤ 2 (script `0.1_simulate_optimal.py`), h = 0.051/2.5 m, c = 346.18 m/s.
- Run with `band_range=EXTENDED_BAND_RANGE`. Λ/λ = 0.5…2 is 2.2…8.7 kHz for this sample, and the default 5 kHz top stops at Λ/λ ≈ 1.2.

- [ ] **Step 1: Write the slow, opt-in tests**

```python
pytestmark = [pytest.mark.slow, pytest.mark.skipif("HEIMES_RESULTS" not in os.environ, reason="needs the Zenodo 15673498 results")]

@pytest.mark.parametrize("theta", [31.5, 61.74])
def test_bem_matches_heimes_per_angle(theta):
    ref = heimes_scattering(theta)                   # dict: Lambda/lambda -> s, one-third-octave averaged
    ours = bem_sine_s_angle(theta)                   # measure_scattering(method="bem") on the same sine sample, c = 346.18
    for ratio, s_ref in ref.items():
        if ratio >= 0.5:
            assert abs(ours[ratio] - s_ref) <= 0.1   # the paper's one-third-octave agreement

def test_bem_matches_heimes_random_incidence():
    ref = heimes_scattering_random()                 # dict: Lambda/lambda -> s_rand, from sine.scattering_rand.sofa
    ours = bem_sine_s_random()                       # measure_scattering(method="bem", preset="standard") s_bands on the same sine sample
    for ratio, s_ref in ref.items():
        if ratio >= 0.5:
            assert abs(ours[ratio] - s_ref) <= 0.1

def test_kirchhoff_gap_to_bem_is_recorded():
    gap = kirchhoff_vs_bem_gap()                     # max |s_kirchhoff - s_bem| per octave on the sine sample
    print(json.dumps(gap))                           # pasted into KirchhoffSolver's docstring (Step 3)
    assert all(np.isfinite(v) for v in gap.values())
```

- [ ] **Step 2: Run them once with the data**

Run: `HEIMES_RESULTS=<folder> pytest tests/lab/test_validation_heimes.py -v -s`
Expected: PASS (an overnight run, as the spike timings predict). Keep the printed gap table.

- [ ] **Step 3: Record Kirchhoff's accuracy in its docstring**

Add a "Known accuracy" section to `KirchhoffSolver`. It gives the measured max |Δs| per octave against BEM on Heimes's sine surface, the date, and the command that reproduces it.

- [ ] **Step 4: Commit**

```bash
git add tests/lab/test_validation_heimes.py src/compas_acoustics/lab/solvers/kirchhoff.py
git commit -m "Lab: validation against Heimes and Vorlaender's simulated sine surface; Kirchhoff's measured gap to BEM"
```

---

## After this plan

- Run `trail check` before pushing the branch.
- Next plans: **V1-B** (`ReverberationChamber`, `measure_absorption`, and the turntable method behind `measure_scattering(method="turntable")`), then **V1-C** (proto messages, conversions, worker functions, C# components and Polar Preview, `lab_material`).
