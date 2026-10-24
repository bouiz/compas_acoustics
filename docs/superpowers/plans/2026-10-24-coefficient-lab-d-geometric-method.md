# Coefficient lab D1+D2: proxy reference, diffusion coefficient and the geometric method

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** `measure_scattering(sample, method="geometric")` returns, in seconds, a heuristic per-octave s, the diffusion coefficient d and polar responses for a mounted 3D object with cavities, against an optional user proxy; `kirchhoff` and `bem` also return d.

**Architecture:** A numpy specular ray tracer over the rigid mesh (Embree through `trimesh`) bins exit directions on the existing `ReceiverGrid` hemisphere. s_max comes from the histogram overlap with the reference, per-band s from the `scattering_coefficients_from_depth` curve with s_max and h taken from the geometry. d is computed from intensities per receiver cell, for all methods. The geometric method does not fit `ScatteringSolver` (no complex pressures), so it is its own module called from `measure_scattering`.

**Tech Stack:** numpy, scipy, trimesh + embreex (lab extra, Python ≥ 3.11), pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-virtual-coefficient-lab-design.md`, section "Objects, the proxy reference and the geometric method". Out of scope here: SBR (D3), Grasshopper components and compas_pb messages (C), `AcousticMaterial` assembly (B).

## Global Constraints

- Branch `feat/coefficient-lab`; commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- `compas_acoustics.lab` must import on Python 3.9 (Rhino): no `match`, no `X | Y` annotations, and `trimesh`/`embreex` imported lazily inside the ray caster, never at module level.
- Style: ruff `line-length = 179`, `force-single-line` imports, numpy docstrings, `target-version = "py39"`. Run `ruff check src/compas_acoustics/lab tests/lab` before each commit.
- Bands: one-third octaves, `ISO_BAND_RANGE` (100-5000 Hz) default; octave values by arithmetic mean of 3; `ScatteringResult.scattering_coefficients` is 7 octaves.
- Default source preset `"standard"` (97 sources) for every method including `"geometric"`.
- Directions: a source direction û points from the sample towards the source; its rays travel along −û. Sample and reference are closed, outward-oriented COMPAS meshes with the relief's lowest point at z = 0 on a baseplate down to `−baseplate`.
- Messages go through `warnings.warn` and into `result.warnings`, like the existing methods.
- Tests live in `tests/lab/`, shapes in `tests/lab/lab_shapes.py`; run with `pytest tests/lab -q`. Ray-casting tests use `pytest.importorskip("embreex")`.

## Review Focus

- Deep narrow cavities: rays still bouncing after `max_bounces` are `lost`; the result warns above 1 % lost (Task 4, Task 5).
- Grazing incidence (colatitude 80°) on a relief with risers: rays hitting rim walls or leaving below the horizon are `lost`, and energy is conserved: binned + lost = 1 (Task 4).
- A proxy whose outline differs from the sample's raises a `ValueError` naming both bounding boxes (Task 3).
- A flat sample against its flat reference gives s = 0 and d_n = 0 exactly, with no NaN from empty bins (Task 4, Task 5).
- Same inputs and `seed` give identical results; a sample identical to its proxy gives s = 0 even when it is not flat (Task 4).

---

### Task 1: The ray caster

**Files:**
- Create: `src/compas_acoustics/lab/raycast.py`
- Modify: `requirements-lab.txt` (add `embreex; python_version >= "3.11"` with a comment that trimesh uses it for ray casting)
- Test: `tests/lab/test_raycast.py`

**Interfaces:**
- Consumes: a `compas.datastructures.Mesh` body (closed triangles, as `ScatteringSample.sample`).
- Produces: `class RayCaster` with `__init__(self, body: Mesh)` and `first_hits(self, origins: np.ndarray, directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]` returning `(triangle, point)`: `triangle` shape (n,) int, −1 where the ray hits nothing; `point` shape (n, 3). Also `normals: np.ndarray` (n_triangles, 3), outward unit normals.

- [x] **Step 1: Install and write the failing tests.** `pip install embreex` into the dev environment. In `test_raycast.py`: `test_a_ray_down_onto_a_box_hits_its_top_face` (a 1×1×1 closed box from `bumped_box(size=1.0, thickness=1.0, bump=0.0)`; origin (0.3, 0.4, 3), direction (0, 0, −1): triangle ≥ 0, `point[2] == 1.0` within 1e-9, `normals[triangle] ≈ (0, 0, 1)`); `test_a_ray_beside_the_box_misses` (origin (5, 5, 3): triangle == −1); `test_the_module_imports_without_trimesh` (import `compas_acoustics.lab.raycast` with `sys.modules["trimesh"] = None` set by monkeypatch, no error).
- [x] **Step 2: Run `pytest tests/lab/test_raycast.py -v`; expect FAIL (module missing).**
- [x] **Step 3: Implement `RayCaster`.** Build a `trimesh.Trimesh` from the body's vertices and triangulated faces inside `__init__` (lazy import; raise `ImportError` with an install hint for `pip install compas_acoustics[lab]` if trimesh or its ray backend is missing); `first_hits` uses `mesh.ray.intersects_id(..., multiple_hits=False, return_locations=True)` (or the Embree call trimesh exposes) and scatters the results back into arrays of length n. Rays are offset by 1e-9·(bounding size) from their origin to avoid self-hits at a bounce point.
- [x] **Step 4: Measure throughput.** In the test file add `test_throughput_is_recorded` marked `@pytest.mark.slow` (check `pyproject.toml` for the marker; register it only if it is missing) casting 200 000 rays at an icosphere of ~20 000 faces; print rays/s. Put the measured value in the module docstring. If below 1e6 rays/s with Embree, say so in the report.
- [x] **Step 5: Run `pytest tests/lab/test_raycast.py -v`; expect PASS. Commit** `Lab: RayCaster over trimesh/Embree, imported lazily`.

### Task 2: The diffusion coefficient

**Files:**
- Modify: `src/compas_acoustics/lab/coefficients.py`, `src/compas_acoustics/lab/__init__.py` (export both functions)
- Test: `tests/lab/test_coefficients.py`

**Interfaces:**
- Produces: `diffusion_coefficient(intensity: np.ndarray, weights: np.ndarray, axis: int = 1) -> np.ndarray` and `normalised_diffusion(d: np.ndarray, d_reference: np.ndarray) -> np.ndarray`.

- [x] **Step 1: Write the failing tests** in `test_coefficients.py`: `test_equal_intensity_everywhere_is_fully_diffuse` (intensity ones (1, 50), weights random positive: d == 1 within 1e-12); `test_one_receiver_carrying_everything_is_not_diffuse` (all zeros except one entry: d == 0); `test_weights_act_as_solid_angles` (an intensity that doubles on a receiver whose weight is halved and whose duplicate is added gives the same d as the unsplit grid, within 1e-12); `test_an_empty_pattern_gives_zero_not_nan`; `test_normalisation` (`normalised_diffusion(0.6, 0.2) == 0.5`; `d_reference == 1` returns 0 rather than NaN).
- [x] **Step 2: Run them; expect FAIL.**
- [x] **Step 3: Implement.** ISO 17497-2 eq. 6 with weights normalised to mean 1 (`w̃ = w·n/Σw`, n the number of receivers): `d = ((Σ w̃ I)² / Σ w̃ I² − 1) / (n − 1)`, receivers along `axis`; where Σ w̃ I² = 0 return 0. `normalised_diffusion` is `(d − d_ref)/(1 − d_ref)`, 0 where d_ref ≥ 1. Docstrings cite the equation and say the intensities are |p|² (far field) or the geometric method's energy per steradian.
- [x] **Step 4: Run `pytest tests/lab/test_coefficients.py -v`; expect PASS. Commit** `Lab: ISO 17497-2 diffusion coefficient on weighted receivers`.

### Task 3: The proxy reference

**Files:**
- Modify: `src/compas_acoustics/lab/sample.py` (`ScatteringSample.as_drawn`, `__init__`, `__data__`)
- Test: `tests/lab/test_sample.py`; add `v_groove_tile` and `sawtooth_tile` to `tests/lab/lab_shapes.py` here, used by Tasks 4-5.

**Interfaces:**
- Produces: `ScatteringSample.as_drawn(mesh, baseplate=0.01, symmetry="none", feature_size=None, reference=None)`; attribute `reference_kind: str` (`"flat"` or `"proxy"`), a new keyword `reference_kind="flat"` of `__init__`, stored in `__data__`. In `lab_shapes.py`: `v_groove_tile(period=0.04, width=0.05)`: one period of `z = |x − period/2|`, two quads, 90° groove, ridges at the tile edges; `sawtooth_tile(period=0.05, slope_deg=20.0, width=0.05)`: a quad rising `period·tan(slope)` over x then a vertical riser quad back down at x = period.

- [x] **Step 1: Write the failing tests:** `test_a_proxy_replaces_the_flat_reference` (as_drawn of a bumped box with `reference=` the same box without its bump: `reference_kind == "proxy"`, `reference` is closed (`ScatteringSample` construction already checks), its top at z = 0 away from the centre vertex like the sample's (the closed-mesh path moves the lowest point to −baseplate, so a 0.01 m box ends with its top at 0), a warning in `sample.warnings` containing "proxy"); `test_without_a_proxy_the_reference_is_flat` (`reference_kind == "flat"`, no such warning); `test_a_proxy_with_another_outline_is_refused` (proxy 2× larger: `ValueError` match `"outline"`); `test_an_open_proxy_is_closed_like_the_sample` (open L-shaped relief via `open_l_shaped_relief()` as sample and a flat open patch of the same outline from `_grid`-style construction as proxy: both closed bodies with the same baseplate); `test_round_trip` (`ScatteringSample.__from_data__(sample.__data__).reference_kind` equals the original; data without the key loads as `"flat"`).
- [x] **Step 2: Run them; expect FAIL.**
- [x] **Step 3: Implement.** The proxy goes through `_clean`, then the same z shift as the sample (the sample's shift decides, so both stay registered), then `_close`-style walling when open or as-is when closed, then `_to_body(..., "reference")`. Compare xy bounding boxes of the two bodies' vertices: more than 1e-3·diameter apart raises `ValueError("the proxy's outline (bounding box …) differs from the sample's (…)")`. The warning text: "the reference is a user proxy: s is the share of the sample's reflection the proxy does not reproduce, not the ISO 17497 coefficient".
- [x] **Step 4: Run `pytest tests/lab/test_sample.py -v`; expect PASS. Commit** `Lab: as_drawn takes an optional proxy as the reference`.

### Task 4: The geometric engine

**Files:**
- Create: `src/compas_acoustics/lab/geometric.py`
- Test: `tests/lab/test_geometric.py` (all tests `pytest.importorskip("embreex")`)

**Interfaces:**
- Consumes: `RayCaster` (Task 1), `ScatteringSample` with `reference_kind` (Task 3), `SourceGrid`, `ReceiverGrid.hemisphere(spacing_deg, radius)`.
- Produces:
  - `@dataclass GeometricResponse` with `directions (n_bins, 3)`, `weights (n_bins,)` (solid angles), `sample, reference (n_sources, n_bins)` energy fractions per cell, `lost_sample, lost_reference (n_sources,)` lost fractions, `depth: float`.
  - `geometric_response(sample, sources, bin_deg=10.0, n_rays=50000, max_bounces=20, seed=0) -> GeometricResponse`
  - `scattering_share(e_sample: np.ndarray, e_reference: np.ndarray) -> np.ndarray` (histogram overlap, bins on the last axis)
  - `heuristic_s(s_max: np.ndarray, depth: float, frequencies: np.ndarray, c: float) -> np.ndarray`, shape `s_max.shape + (len(frequencies),)`.

- [x] **Step 1: Write the failing tests.**
  - `test_a_flat_sample_reflects_everything_into_the_specular_cell`: `ScatteringSample.disc(flat_tile(), 0.4)`, `SourceGrid.preset("survey")`; for every source the cell holding the mirror of û has fraction 1.0 within 1e-9 for sample and reference, `scattering_share` == 0, `lost` == 0, `depth` == 0.
  - `test_energy_is_conserved`: a `sawtooth_tile` disc at "standard" (its 80° ring is the grazing case): `e.sum(axis=1) + lost == 1` within 1e-9 for sample and reference.
  - `test_a_tilted_facet_sends_normal_incidence_to_twice_its_slope`: `sawtooth_tile(slope_deg=20)`, source at colatitude 0, `n_rays=20000`: `scattering_share > 0.95`, and the energy-weighted mean colatitude of the exit directions is 40° ± 5°.
  - `test_a_right_angle_groove_retroreflects_with_enough_bounces`: `v_groove_tile` disc of diameter 0.4, colatitude 0: with `max_bounces=4` more than 0.9 of the energy is in the cell containing +z and `scattering_share < 0.1`; with `max_bounces=1` the lost fraction is above 0.8.
  - `test_a_sample_equal_to_its_proxy_scatters_nothing`: the sawtooth sample with a proxy built from the same mesh: `scattering_share == 0` for every source.
  - `test_the_same_seed_gives_the_same_response_and_another_seed_a_close_one`: `np.array_equal` for seed 0 twice; `scattering_share` of two seeds within 0.05.
  - `test_depth_is_twice_the_standard_deviation_of_the_height`: the sawtooth over a flat reference: `depth` within 10 % of `2·H/√12`, H = `period·tan(slope)`.
  - `test_the_depth_curve`: `heuristic_s(1.0, 0.01, [10, 1e5], 343)` → ≈ 0 at 10 Hz (< 1e-3) and ≈ 1 at 100 kHz (> 0.999); monotone in f; `heuristic_s(0.4, 0.0, …)` is 0.
- [x] **Step 2: Run them; expect FAIL.**
- [x] **Step 3: Implement `geometric_response`.** Launch points: uniform random in the triangles of the reference's top face (the reference triangles with normal +z at z ≈ 0, chosen by area), the same points for the sample and the reference, drawn once per call from `numpy.random.default_rng(seed)`. For each source û, each ray starts at the point plus `−û` times a distance above the tallest body, travels along −û, and is traced with `RayCaster.first_hits` for `max_bounces` iterations, reflecting `d − 2(d·n)n` at each hit with n the hit triangle's outward normal; a ray that misses leaves, and counts in a cell if its exit direction has z > 0 (cell = nearest receiver direction through `scipy.spatial.cKDTree`), otherwise in `lost`; a ray is reflected at most `max_bounces` times; if the cast after its last allowed reflection still hits a face it is `lost` (so `max_bounces=1` on a right-angle groove loses nearly everything). Each ray carries 1/n_rays. `depth` is 2·σ of the difference between the first-hit heights of straight-down rays on the sample and on the reference at the launch points. `scattering_share` is `1 − Σ min(a, b)` over the last axis. `heuristic_s` is `s_max[..., None]·(1 − exp(−(2πf·depth/c)²))`.
- [x] **Step 4: Run `pytest tests/lab/test_geometric.py -v`; expect PASS. Commit** `Lab: geometric engine, specular ray tracing binned on the receiver hemisphere`.

### Task 5: `measure_scattering(method="geometric")` and d for every method

**Files:**
- Modify: `src/compas_acoustics/lab/scattering.py` (`METHODS`, `DEFAULT_PRESETS`, `measure_scattering`, its docstring), `src/compas_acoustics/lab/results.py` (`ScatteringResult`), `src/compas_acoustics/lab/__init__.py`
- Test: `tests/lab/test_scattering.py`, `tests/lab/test_results.py` (create it if the result has no test file; check first)

**Interfaces:**
- Consumes: `geometric_response`, `scattering_share`, `heuristic_s` (Task 4); `diffusion_coefficient`, `normalised_diffusion` (Task 2).
- Produces: `ScatteringResult(..., d_angle=None, d_bands=None, d_octaves=None)`: the normalised diffusion, shapes (n_sources, n_bands), (n_bands,), (n_octaves,); in `__data__`, absent keys load as `None`. `measure_scattering(sample, method="geometric", ...)`, with `"geometric"` now the default method (the user's decision: the fast preview is the default, the others are options); existing tests that relied on the old `"kirchhoff"` default pass `method="kirchhoff"` explicitly; solver options `n_rays`, `bin_deg`, `max_bounces`, `seed` passed to `geometric_response`. For the correlation methods, d_n per source and frequency from `|p|²` of the sample and the reference over `receivers.weights`, band-averaged like s.

- [x] **Step 1: Write the failing tests.**
  - `test_geometric_flat_disc_scatters_nothing`: `measure_scattering(disc(flat_tile(), 0.4), method="geometric", n_rays=2000)`: every `s_octaves` == 0 and every `d_octaves` == 0 within 1e-9; `method == "geometric"`; `generation["heuristic"] is True`; a warning containing "heuristic".
  - `test_geometric_sawtooth_rises_with_frequency`: s_octaves non-decreasing, first < 0.01, last > 0.3 (h = 2σ ≈ 0.0105 m gives s ≈ 0.45 at 4 kHz); `len(scattering_coefficients) == 7`; `result.s_angle.shape == (97, 18)`.
  - `test_geometric_runs_in_seconds`: the sawtooth at "standard" with `n_rays=5000` takes under 20 s.
  - `test_geometric_deep_cavity_warns_about_lost_rays`: `max_bounces=1` on `v_groove_tile` gives a warning containing "lost".
  - `test_geometric_with_a_proxy_marks_the_result`: warnings contain "proxy"; `generation["reference"] == "proxy"`.
  - `test_the_correlation_methods_return_d`: with the existing `FakeSolver` (sample == reference) `d_octaves` are all 0 and `d_angle.shape == s_angle.shape`; Kirchhoff on a `sine_tile` disc (`preset="survey"`, small, as other Kirchhoff tests in the file do) gives `d_octaves` all in [0, 1] and at least one above 0.
  - `test_results_round_trip_with_d`: `json_loads(json_dumps(result))` equals on d fields; old-format data without d fields loads with `d_angle is None`.
- [x] **Step 2: Run them; expect FAIL.**
- [x] **Step 3: Implement.** Add `"geometric"` to `METHODS` and `DEFAULT_PRESETS["geometric"] = "standard"`. The geometric branch: `s_angle[i, band] = heuristic_s(scattering_share(...)[i], depth, exact_band_centres, c)`; `d_angle[i, band]` constant per source from `diffusion_coefficient(e/weights, weights)` for the sample and the reference, normalised; polar levels from the cells' energy per steradian in dB relative to the reference's maximum for the same source (−300 dB where empty), the same for every octave, for the preview sources, `receiver_directions` = the cell directions. Extract the shared tail (band average, random incidence, octave folding, filled octaves, warnings, result) of the existing function into one helper instead of duplicating it. Warnings: "heuristic" (text: s is frequency-modelled from the geometry; use kirchhoff, sbr or bem for the physics), the proxy warning from the sample, and "lost" (percent) when the lost fraction of any source exceeds 1 %. `generation` gets `heuristic`, `reference`, `n_rays`, `bin_deg`, `max_bounces`, `seed`, `lost` (max over sources).
- [x] **Step 4: Run `pytest tests/lab -q -m "not slow"`; expect PASS (existing tests unchanged). Run `ruff check src/compas_acoustics/lab tests/lab`. Commit** `Lab: the geometric method as the fast default, and d for every method`.

### Task 6: Validation record, spec and changelog

**Files:**
- Modify: `src/compas_acoustics/lab/geometric.py` (module docstring), `tests/lab/test_validation_heimes.py` (one new slow test), `CHANGELOG.md`, `docs/superpowers/plans/2026-10-24-coefficient-lab-d-geometric-method.md` (tick the steps)
- Test: `tests/lab/test_validation_heimes.py`

- [x] **Step 1: Add `test_geometric_against_kirchhoff_is_recorded`** (`@pytest.mark.slow`, same env-var gating as its neighbours): on the Heimes sine sample (the helper the file already uses), `measure_scattering` with `method="geometric"` and `"kirchhoff"` at the same `preset="survey"`; print per octave `s_geometric`, `s_kirchhoff` and |Δ|; assert both finite.
- [x] **Step 2: Run it once** (`pytest tests/lab/test_validation_heimes.py -k geometric -v -s`, with the env vars the file's docstring names, or without them if it falls back to the helper's own geometry). Paste the printed table and the date into the `geometric.py` docstring as the method's known accuracy against Kirchhoff, and state the validity limits there: no phase, no diffraction, no cavity resonances, heuristic frequency dependence.
- [x] **Step 3: Add a CHANGELOG entry** under Unreleased: the proxy reference, the diffusion coefficient output, `method="geometric"` (`measure_scattering` now defaults to `method="geometric"`; say so, since `"kirchhoff"` was the default before), `embreex` in the lab extra.
- [x] **Step 4: Run `pytest tests/lab -q -m "not slow"` and `ruff check`; expect PASS. Commit** `Lab: geometric method validation record, changelog`.
