# Coefficient Lab V1 — B: Absorption Chamber and Lab Material Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `compas_acoustics.lab` measures the random-incidence absorption coefficient of a 3D sample per octave band in a virtual ISO 354 reverberation chamber (ray traced), and combines it with a `ScatteringResult` into an `AcousticMaterial`.

**Architecture:** A `ReverberationChamber` is a shoebox built to the ISO 354 rules and **scaled to the sample** (decay times and areas scale with length, α does not, and in a full-size chamber a small sample changes the decay less than the ray tracer's noise). `measure_absorption` simulates the empty chamber (T₁) and the chamber with the sample (T₂) with the existing `Mono` ray tracer at paired seeds, takes T20 per octave from `AcousticAnalyser`, and applies the ISO 354 formulas (eq. 5–9) in pure-numpy `iso354.py`. `lab_material` turns an `AbsorptionResult` and an optional `ScatteringResult` into an `AcousticMaterial`.

**Tech Stack:** Python ≥ 3.9, numpy, compas 2, the existing `Scene`, `AcousticObjectsLibrary.shoebox_room`, `Mono`, `AcousticAnalyser`, `AirConditions`; pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-virtual-coefficient-lab-design.md` (sections "Architecture → B", "Turntable: dropped", "Version split → V1"). Builds on `docs/superpowers/plans/2026-10-23-coefficient-lab-v1-a-scattering-core.md` (done): `ScatteringSample`, `ScatteringResult`, `bands.py`.

The turntable method is **not** part of this plan (dropped 2026-10-24, see the spec). Nothing here is V2/V3 (no ±2σ output, no one-third-octave curves).

## Global Constraints

- `import compas_acoustics` and `import compas_acoustics.lab` must still work on Python 3.9 with no mesh2scattering, trimesh or spharpy. No `X | Y` annotations evaluated at runtime, single-line imports, ruff line length 179, numpy docstrings citing the standard and equation.
- Serialisable classes subclass `_PackageDtype` + `compas.data.Data` (as `ScatteringResult` in `lab/results.py`) and are exported from `lab/__init__.py` and `compas_acoustics/__init__.py` (+ `__all__`; the exported-class count in `tests/test_scene_serialisation.py` goes up by one per class).
- Bands: the 7 material octaves `Defaults._DEFAULT_FREQUENCY_BANDS` (125…8000). T20 per octave comes from `AcousticAnalyser(ir).reverberation_times()["T20"]` (regression over −5…−25 dB: ISO 354 §7.4.1 asks for an evaluation range of 20 dB starting 5 dB below the initial level). Octaves rather than ISO 354 §5's one-third octaves because the engine's materials are octave-wise, so third-octave decay times carry no extra information. The 8000 Hz octave is beyond ISO 354: every result warns `"the 8000 Hz octave is beyond ISO 354 (100-5000 Hz)"`.
- ISO 354 constants: `S_REFERENCE = 11.0` m² (§6.2.1.1: 10–12 m² for V = 200 m³); `REFERENCE_DIMENSIONS = (7.4, 5.8, 4.6)` m (V = 197.4 m³; l_max = 10.47 m < 1.9·V^⅓ = 11.06 m, §6.1.2; no two dimensions in a ratio of small integers); chamber scale `k = sqrt(S / S_REFERENCE)` with S the sample's footprint area; all lengths of the chamber scale with k.
- Chamber walls: absorption 0.025 and scattering 0.3 in every band (A₁ ≈ 5.2 m² against the Table 1 limit 6.5 m² at k = 1; scattering stands in for the diffusers of Annex A). Table 1 limits (A₁ at V = 200 m³, octave centre → m²): 125: 6.5, 250: 6.5, 500: 6.5, 1000: 7.0, 2000: 9.5, 4000: 13.0; none for 8000; multiplied by `(V/200)^(2/3)`.
- Measurement count (§7.1.4): 2 source positions × 6 microphone positions = 12 decays; source positions ≥ 3 m apart, microphones ≥ 1.5 m apart, ≥ 2 m from a source, all ≥ 1 m from the chamber boundaries (distances scale with k); microphone and source heights between 1 m and H − 1 m. Placement is deterministic: seeded rejection sampling (`numpy.random.default_rng(seed)`, seed 354) in the reference chamber, then scaled by k.
- Ray tracing: `Mono(ray_tracing=True, air_absorption=False, air=air, seed=354 + repeat)`, the same seed for T₁ and T₂ of one repeat (paired, so the noise of the two runs is correlated). T₁ and T₂ are the arithmetic mean over all decays and repeats (ISO 354 §8.1.1). Air absorption is off in the engine (the chamber is a scale model; the same air gives m₂ − m₁ = 0), but the formulas keep the m terms.
- Sampling rate of the chamber scene: 24 000 Hz (the 8000 Hz octave reaches 11.3 kHz).
- Long steps report through `compas_acoustics.worker.jobs.progress(stage, done, of)`. A run estimated longer than 10 minutes warns before it starts.
- Commit messages: a sentence in the repo style, ending with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A model left in millimetres** (footprint area ~10⁶ m²) should be refused with a clear error, not a chamber a kilometre wide. Pinned in Task 3 (`test_millimetre_model_raises`).
2. **A sample that does not fit the chamber floor** with the ISO clearance (a long strip, an L-shaped outline with a large bounding box) should raise, not poke through a wall and break the closed room. Pinned in Task 3 (`test_sample_that_does_not_fit_raises`).
3. **A sample that absorbs nothing, or a noisy band** gives α_s slightly below 0 or above 1 and decay times the ray tracer cannot evaluate; the raw result keeps the values, a material needs [0, 1] and no NaN, so `lab_material` clips with a warning and refuses NaN naming the band. Pinned in Task 5.
4. **The empty chamber must really be empty** (T₁ without the sample or its baseplate) and T₂ must contain it; the same seed for both. Pinned in Tasks 3 and 4 (`test_scene_without_sample_has_no_obstacle`, `test_t1_and_t2_use_paired_seeds`).
5. **NaN in a result** (a band whose decay could not be evaluated) must not break strict JSON that the Grasshopper side parses. Pinned in Task 4 (`test_nan_bands_serialise_as_none`).

---

### Task 1: Spike — ray-traced chamber: noise, cost and defaults (throwaway)

The output is a findings note, not code. Tasks 3–6 take their defaults from it.

**Files:**
- Create: `docs/superpowers/plans/2026-10-24-coefficient-lab-v1-b-spike-findings.md`
- Scratch scripts go in the session scratchpad. **Do not commit them.**

- [ ] **Step 1: Build the throwaway chamber and samples**

With the constants in Global Constraints, build the scaled chamber for a 0.8 m disc (`S = π·0.4²`, `k = sqrt(S/11) = 0.214`) with `AcousticObjectsLibrary.shoebox_room(...)`, `Source` and `Receiver` objects (`Receiver(name, x, y, z, radius=…)`), and two samples from `ScatteringSample.disc` placed on the floor (translate the body by `(cx, cy, baseplate + 0.002)`, `AcousticObject(mesh, material, name="sample", role="obstacle")`): a flat tile and `sine_tile(0.0708, 0.0204, n=8)` (`from lab_shapes import sine_tile`). The sample material is `AcousticMaterial("panel", [0.3] * 7, [0.1] * 7)`.

- [ ] **Step 2: Measure noise and cost**

For seeds 354…358, both samples and the empty chamber, with `Mono(ray_tracing=True, air_absorption=False, air=AirConditions(), seed=…)` and every combination of `n_rays ∈ {5 000, 20 000, 50 000}`, `image_source_order ∈ {0, 1, 2}` and `receiver_radius ∈ {0.25 k, 0.5 k, 1.0 k}` that the cost allows: wall time per `scene.simulate`, the T20 per octave of every (source, receiver) decay, α_s per octave from `lab.iso354`-style formulas written inline (`A = 55.3 V/(c T)`, `α = (A₂ − A₁)/S`), and its standard deviation over the seeds, with and without pairing the seeds of T₁ and T₂.

- [ ] **Step 3: Record the findings**

In the note record: (a) the A₁ of the empty chamber per octave against Table 1 (scaled); (b) for the flat sample, the mean error against 0.3 and the standard deviation over seeds per octave; (c) the cheapest `(n_rays, image_source_order, receiver_radius, n_repeats)` that gets the standard deviation of α_s ≤ 0.03 in the 250…4000 Hz octaves; (d) seconds per run for 2 444, 4 288 and 7 964-face samples, and a formula `seconds ≈ n_repeats · n_sources · (t_empty + t_face · faces · n_rays / 20 000)` with the fitted `t_empty` and `t_face`; (e) whether `Mono(air=...)` makes pyroomacoustics use `air.speed_of_sound`; (f) whether the sine sample's α_s is above the flat sample's (the relief has more area).

- [ ] **Step 4: Commit the findings note only**

```bash
git add docs/superpowers/plans/2026-10-24-coefficient-lab-v1-b-spike-findings.md
git commit -m "Lab spike: ray-traced absorption chamber, noise and cost of the ISO 354 measurement"
```

---

### Task 2: ISO 354 formulas

**Files:**
- Create: `src/compas_acoustics/lab/iso354.py`
- Modify: `src/compas_acoustics/lab/__init__.py`
- Test: `tests/lab/test_iso354.py`

**Interfaces:**
- Produces (all accept floats or numpy arrays and broadcast per band):
  - `equivalent_absorption_area(T, V, c, m) -> ndarray`: eq. 5 / 7, `55.3·V/(c·T) − 4·V·m` in m².
  - `sample_absorption_area(T1, T2, V, c1, c2, m1, m2) -> ndarray`: eq. 8, `A₂ − A₁`.
  - `absorption_coefficient(A_T, S) -> ndarray`: eq. 9, `A_T / S`.
  - `table1_limit(octave, V) -> float`: the Table 1 limit for the octave centre (125…4000) × `(V/200)^(2/3)`; `nan` for 8000 and any other octave.
  - `TABLE1_OCTAVE_LIMITS: dict[int, float]` with the values from Global Constraints.

- [ ] **Step 1: Write the failing tests**

```python
def test_equivalent_absorption_area_is_iso_354_eq_5():
    assert equivalent_absorption_area(5.0, 200.0, 343.0, 0.0) == pytest.approx(55.3 * 200 / (343 * 5.0))

def test_the_air_term_is_subtracted():
    assert equivalent_absorption_area(5.0, 200.0, 343.0, 0.001) == pytest.approx(55.3 * 200 / (343 * 5.0) - 4 * 200 * 0.001)

def test_sample_absorption_area_is_the_difference_eq_8():
    a = sample_absorption_area(5.0, 4.0, 200.0, 343.0, 343.0, 0.0, 0.0)
    assert a == pytest.approx(55.3 * 200 / 343 * (1 / 4.0 - 1 / 5.0))   # 1.612 m2

def test_air_changes_between_the_two_measurements_are_corrected():
    a = sample_absorption_area(5.0, 4.0, 200.0, 343.0, 343.0, 0.001, 0.0015)
    assert a == pytest.approx(55.3 * 200 / 343 * (1 / 4.0 - 1 / 5.0) - 4 * 200 * 0.0005)

def test_the_coefficient_is_the_area_over_the_sample_area_eq_9():
    assert absorption_coefficient(2.0, 10.0) == pytest.approx(0.2)

def test_per_band_arrays_broadcast():
    out = sample_absorption_area(np.array([5.0, 6.0]), np.array([4.0, 4.5]), 200.0, 343.0, 343.0, 0.0, 0.0)
    assert out.shape == (2,) and out[1] == pytest.approx(55.3 * 200 / 343 * (1 / 4.5 - 1 / 6.0))

def test_table1_limits_scale_with_the_volume_and_stop_at_4000():
    assert table1_limit(1000, 200.0) == 7.0 and table1_limit(4000, 200.0) == 13.0
    assert table1_limit(250, 8.0) == pytest.approx(6.5 * (8.0 / 200.0) ** (2 / 3))
    assert np.isnan(table1_limit(8000, 200.0))
```

- [ ] **Step 2: Run them and see them fail**

Run: `pytest tests/lab/test_iso354.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `lab/iso354.py` and export the names from `lab/__init__.py`**

Docstrings cite ISO 354:2003 eq. 5, 7, 8, 9 and Table 1.

- [ ] **Step 4: Run them and see them pass**

Run: `pytest tests/lab/test_iso354.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/compas_acoustics/lab/iso354.py src/compas_acoustics/lab/__init__.py tests/lab/test_iso354.py
git commit -m "Lab: ISO 354 equivalent absorption area and absorption coefficient formulas, Table 1 limits"
```

---

### Task 3: The reverberation chamber

**Files:**
- Create: `src/compas_acoustics/lab/chamber.py`
- Modify: `src/compas_acoustics/lab/__init__.py`, `src/compas_acoustics/__init__.py`, `tests/lab/lab_shapes.py` (add `long_strip_tile()`: a flat 5 m × 0.2 m tile), `tests/test_scene_serialisation.py` (class count +1)
- Test: `tests/lab/test_chamber.py`

**Interfaces:**
- Consumes: `ScatteringSample` (`.sample`, `.reference`, `.baseplate`).
- Produces:
  - `S_REFERENCE`, `REFERENCE_DIMENSIONS`, `LIFT = 0.002` (m, the gap between the floor and the sample body).
  - `footprint_area(sample: ScatteringSample) -> float`: the summed area of the faces of `sample.reference` whose normal points up, projected on xy.
  - `ReverberationChamber(_PackageDtype, Data)` with `__init__(self, scale=1.0, wall_absorption=0.025, wall_scattering=0.3, n_sources=2, n_receivers=6, seed=354)`, classmethod `for_area(area: float, **kwargs)` (`scale = sqrt(area / S_REFERENCE)`; `ValueError` containing "millimetres" when `scale > 10`), properties `dimensions` (tuple, scaled), `volume`, `max_length`, `surface_area`, `sources` and `receivers` (lists of `Source`/`Receiver` in chamber coordinates, the receivers with `radius = 0.5 * scale`), `receiver_radius`, `walls` (an `AcousticMaterial`), and:
    - `scene(sample=None, material=None, sampling_rate=24_000) -> Scene`: the six walls (`AcousticObjectsLibrary.shoebox_room`), the sources and receivers, and with a sample its body as `AcousticObject(mesh, material, name="sample", role="obstacle")`, translated so that the bounding-box centre of its footprint sits at `(3.9, 2.9) · scale`, its lowest point at z = `LIFT`. `material` is required when `sample` is given. `ValueError` when the footprint does not stay at least `1.0 · scale` m from every wall.
    - `warnings() -> list[str]`: the ISO 354 §6.1.2 shape rule and the §7.1.2–7.1.4 counts and distances, empty for a valid chamber.

- [ ] **Step 1: Write the failing tests**

```python
from lab_shapes import flat_tile, long_strip_tile, sine_tile

def test_the_reference_chamber_meets_the_iso_354_shape_rules():
    c = ReverberationChamber()
    assert 150.0 <= c.volume <= 250.0 and c.max_length < 1.9 * c.volume ** (1 / 3)
    assert c.warnings() == []

@pytest.mark.parametrize("scale", [0.2, 1.0, 2.5])
def test_placement_satisfies_the_iso_354_distances(scale):
    c = ReverberationChamber(scale=scale)
    s, m = [np.array(p.position) for p in c.sources], [np.array(r.position) for r in c.receivers]
    assert len(s) * len(m) >= 12 and len(m) >= 3 and len(s) >= 2
    dist = lambda a, b: np.linalg.norm(a - b)
    assert all(dist(a, b) >= 3.0 * scale for i, a in enumerate(s) for b in s[i + 1:])
    assert all(dist(a, b) >= 1.5 * scale for i, a in enumerate(m) for b in m[i + 1:])
    assert all(dist(a, b) >= 2.0 * scale for a in s for b in m)
    L, W, H = c.dimensions
    for p in s + m:
        assert (p >= scale - 1e-9).all() and (p <= np.array([L, W, H]) - scale + 1e-9).all()

def test_placement_is_deterministic_and_scales_with_the_chamber():
    a, b = ReverberationChamber(), ReverberationChamber(scale=0.3)
    assert np.allclose([0.3 * np.array(s.position) for s in a.sources], [s.position for s in b.sources])
    assert [s.position for s in ReverberationChamber().sources] == [s.position for s in a.sources]

def test_for_area_scales_by_the_square_root_of_the_area():
    assert ReverberationChamber.for_area(11.0).scale == pytest.approx(1.0)
    assert ReverberationChamber.for_area(np.pi * 0.4 ** 2).scale == pytest.approx(np.sqrt(np.pi * 0.16 / 11.0))

def test_footprint_area_of_a_disc():
    assert footprint_area(ScatteringSample.disc(flat_tile(), diameter=0.8, segments=256)) == pytest.approx(np.pi * 0.16, rel=1e-3)

def test_scene_without_sample_has_no_obstacle():
    scene = ReverberationChamber().scene()
    assert scene.number_of_acoustic_objects == 6 and scene.number_of_sources == 2 and scene.number_of_receivers == 6

def test_scene_with_sample_puts_the_body_on_the_floor_at_the_chamber_position():
    sample = ScatteringSample.disc(sine_tile(0.0708, 0.0204), diameter=0.8)
    chamber = ReverberationChamber.for_area(footprint_area(sample))
    scene = chamber.scene(sample, AcousticMaterial("panel", [0.3] * 7))
    body = [o for o in scene.acoustic_objects if o.name == "sample"][0]
    assert body.role == "obstacle" and scene.number_of_acoustic_objects == 7
    xyz = np.array([body.acoustic_geometry.vertex_coordinates(v) for v in body.acoustic_geometry.vertices()])
    assert xyz[:, 2].min() == pytest.approx(LIFT)
    assert (xyz[:, :2].min(axis=0) + xyz[:, :2].max(axis=0)) / 2 == pytest.approx(np.array([3.9, 2.9]) * chamber.scale, abs=1e-6)

def test_the_room_builds_and_excludes_the_sample_body():
    sample = ScatteringSample.disc(flat_tile(), diameter=0.8)
    chamber = ReverberationChamber.for_area(footprint_area(sample))
    geometry = RoomGeometry.from_scene(chamber.scene(sample, AcousticMaterial("panel", [0.3] * 7)))
    centre = np.array([3.9, 2.9]) * chamber.scale
    assert all(geometry.is_inside(p.position) for p in chamber.sources + chamber.receivers)
    assert not geometry.is_inside([centre[0], centre[1], 0.006])   # inside the baseplate body: the room has a hole there
    assert geometry.is_inside([centre[0], centre[1], 0.05])

def test_millimetre_model_raises():
    with pytest.raises(ValueError, match="millimetres"):
        ReverberationChamber.for_area(np.pi * 400.0 ** 2)

def test_sample_that_does_not_fit_raises():
    strip = ScatteringSample.as_drawn(long_strip_tile())
    chamber = ReverberationChamber.for_area(footprint_area(strip))
    with pytest.raises(ValueError, match="wall"):
        chamber.scene(strip, AcousticMaterial("panel", [0.3] * 7))

def test_chamber_round_trips_through_json():
    c = ReverberationChamber(scale=0.3, wall_absorption=0.03)
    d = json_loads(json_dumps(c))
    assert isinstance(d, ReverberationChamber) and d.scale == 0.3 and d.wall_absorption == 0.03
```

(`RoomGeometry` is `compas_acoustics.RoomGeometry`.)

- [ ] **Step 2: Run them and see them fail**

Run: `pytest tests/lab/test_chamber.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `lab/chamber.py`**

Placement: draw candidate points uniformly in the reference box shrunk by 1 m (z from 1 to `H − 1`), accept a candidate when it meets the pairwise distances against the points accepted so far, sources first, then receivers. `warnings()` re-checks the same rules on the final positions plus the shape rule (§6.1.2, no two dimensions within 2 % of a ratio p:q with p, q ≤ 3). Class docstring cites ISO 354 §6.1, §6.2.1, §7.1 and Annex A, and says why the chamber is scaled.

- [ ] **Step 4: Run them and see them pass**

Run: `pytest tests/lab/test_chamber.py tests/test_scene_serialisation.py tests/test_python39.py -v`
Expected: PASS (except the known pre-existing pickle test).

- [ ] **Step 5: Commit**

```bash
git add src/compas_acoustics/lab src/compas_acoustics/__init__.py tests/lab tests/test_scene_serialisation.py
git commit -m "Lab: ReverberationChamber, an ISO 354 shoebox scaled to the sample with deterministic source and microphone placement"
```

---

### Task 4: `AbsorptionResult` and `measure_absorption`

**Files:**
- Create: `src/compas_acoustics/lab/absorption.py`
- Modify: `src/compas_acoustics/lab/results.py` (add `AbsorptionResult`), `src/compas_acoustics/lab/__init__.py`, `src/compas_acoustics/__init__.py`, `tests/test_scene_serialisation.py` (class count +1)
- Test: `tests/lab/test_absorption.py`

**Interfaces:**
- Consumes: Task 2 (`iso354`), Task 3 (`ReverberationChamber`, `footprint_area`), `ScatteringSample`, `Mono`, `AcousticAnalyser`, `AirConditions`; the spike findings for the defaults and the run-time constants.
- Produces:
  - `AbsorptionResult(_PackageDtype, Data)` with `__init__(self, bands, alpha_s, absorption_area, T1, T2, A1, area, volume, chamber_scale, material, n_decays, n_repeats, warnings=None, generation=None, name=None)`: `bands` the 7 octave centres, `alpha_s` the raw α_s per band (unclipped), `absorption_area` A_T per band (m², at chamber scale), `T1`/`T2` the mean decay times per band (s), `A1` the empty chamber's equivalent absorption area per band (m²), `area` the sample's footprint S (m²), `volume` V (m³), `chamber_scale`, `material` the name of the sample's material, `n_decays` the decays per measurement (sources × receivers × repeats), property `absorption_coefficients` (the 7 values of `alpha_s`), `__data__` / `__from_data__` that store NaN as `None` and restore it as NaN.
  - `measure_absorption(sample, material, chamber=None, air=None, n_repeats=<spike>, n_rays=<spike>, image_source_order=<spike>, name=None) -> AbsorptionResult`; `chamber` defaults to `ReverberationChamber.for_area(footprint_area(sample))`, `air` to `AirConditions()`. `ValueError` when `n_repeats < 1`.
  - `estimate_seconds(faces: int, n_sources: int, n_repeats: int, n_rays: int) -> float` with the spike's formula.
  - `_method(seed, n_rays, image_source_order, receiver_radius, air) -> SimulationMethod`: the one place that builds the `Mono` (tests patch it).

Pipeline per repeat `r`: `method = _method(354 + r, ...)`; `rirs1 = empty_scene.simulate(method)`; `rirs2 = sample_scene.simulate(method)`; T20 per octave of every response via `AcousticAnalyser(ir).reverberation_times()["T20"]`; stack to `(n_decays, 7)`. `T1`, `T2` are `np.nanmean` over decays and repeats; `V = chamber.volume`, `c = air.speed_of_sound`, `m` = `air.air_attenuation(bands)` for both (air absorption is off in the engine, so `m₂ − m₁ = 0`); `A1 = equivalent_absorption_area(T1, V, c, m)`, `A_T = sample_absorption_area(...)`, `alpha_s = absorption_coefficient(A_T, area)`. Warnings (exact texts): `"the 8000 Hz octave is beyond ISO 354 (100-5000 Hz)"` always; `f"empty chamber: A1 = {a:.1f} m2 exceeds the ISO 354 limit {l:.1f} m2 in the {f} Hz octave"` per octave over `table1_limit`; `f"the decay in the {f} Hz octave could not be evaluated"` per NaN band; `f"estimated run time {minutes:.0f} min"` + `" (sample with {faces} faces)"` when `estimate_seconds` exceeds 600 s, before the first simulation. Progress: `jobs.progress(f"absorption: repeat {r + 1}/{n_repeats}", done, of)` per simulation. `generation = {"method": "ray tracing", "n_rays", "image_source_order", "n_repeats", "seeds", "air": air.to_dict()}`.

- [ ] **Step 1: Write the failing tests** (fast, with a fake `Mono` whose responses are synthetic exponential noise decays: T60 = `T_EMPTY = 1.2` s in the empty scene and `T_SAMPLE = 0.8` s when the scene holds the sample object)

```python
class FakeMono(SimulationMethod):
    name, output = "mono", "mono"
    runs = []   # (seed, scene holds the sample) per simulation
    def __init__(self, seed): self.seed = seed
    def compute(self, context):
        has_sample = any(o.name == "sample" for o in context.scene.acoustic_objects)
        FakeMono.runs.append((self.seed, has_sample))
        t60 = 0.8 if has_sample else 1.2
        rng = np.random.default_rng(self.seed); fs = context.sampling_rate; n = int(1.5 * t60 * fs)
        ir = rng.normal(size=n) * 10 ** (-3 * np.arange(n) / fs / t60)   # -60 dB at t60
        return {context.key(s, r): ImpulseResponse(context.key(s, r), s, r, ir, fs, normalized=False)
                for s in context.sources for r in context.receivers}

@pytest.fixture
def fake(monkeypatch):
    FakeMono.runs.clear(); monkeypatch.setattr(absorption, "_method", lambda seed, *a, **k: FakeMono(seed))

def expected_alpha(chamber, area, c):
    return 55.3 * chamber.volume / c * (1 / 0.8 - 1 / 1.2) / area

def test_alpha_matches_the_iso_354_formula_for_known_decay_times(fake):
    sample = ScatteringSample.disc(flat_tile(), diameter=0.8)
    r = measure_absorption(sample, AcousticMaterial("panel", [0.3] * 7), n_repeats=2)
    chamber = ReverberationChamber.for_area(footprint_area(sample))
    assert np.allclose(r.alpha_s[1:6], expected_alpha(chamber, r.area, AirConditions().speed_of_sound), rtol=0.1)

def test_bands_counts_and_warnings(fake):
    r = measure_absorption(ScatteringSample.disc(flat_tile(), diameter=0.8), AcousticMaterial("panel", [0.3] * 7), n_repeats=2)
    assert r.bands == [125, 250, 500, 1000, 2000, 4000, 8000] and len(r.absorption_coefficients) == 7
    assert r.n_decays == 2 * 6 * 2 and r.material == "panel" and r.chamber_scale == pytest.approx(0.2139, rel=1e-3)
    assert "the 8000 Hz octave is beyond ISO 354 (100-5000 Hz)" in r.warnings

def test_t1_and_t2_use_paired_seeds(fake):
    measure_absorption(ScatteringSample.disc(flat_tile(), diameter=0.8), AcousticMaterial("panel", [0.3] * 7), n_repeats=2)
    assert FakeMono.runs == [(354, False), (354, True), (355, False), (355, True)]

def test_an_empty_chamber_above_the_table_1_limit_warns(fake, monkeypatch):
    monkeypatch.setattr(absorption, "table1_limit", lambda f, v: 0.1)
    r = measure_absorption(ScatteringSample.disc(flat_tile(), diameter=0.8), AcousticMaterial("panel", [0.3] * 7), n_repeats=1)
    assert any("exceeds the ISO 354 limit 0.1 m2 in the 125 Hz octave" in w for w in r.warnings)

def test_a_long_run_is_announced(fake, monkeypatch):
    monkeypatch.setattr(absorption, "estimate_seconds", lambda *a, **k: 1800.0)
    with pytest.warns(UserWarning, match="estimated run time 30 min"):
        measure_absorption(ScatteringSample.disc(flat_tile(), diameter=0.8), AcousticMaterial("panel", [0.3] * 7), n_repeats=1)

def test_no_repeats_is_refused():
    with pytest.raises(ValueError, match="n_repeats"):
        measure_absorption(ScatteringSample.disc(flat_tile(), diameter=0.8), AcousticMaterial("panel", [0.3] * 7), n_repeats=0)

def test_nan_bands_serialise_as_none():
    r = AbsorptionResult(bands=[125], alpha_s=[float("nan")], absorption_area=[float("nan")], T1=[1.0], T2=[float("nan")], A1=[5.0],
                         area=0.5, volume=1.9, chamber_scale=0.21, material="m", n_decays=12, n_repeats=1)
    json.dumps(r.__data__, allow_nan=False)
    assert np.isnan(json_loads(json_dumps(r)).alpha_s[0])

def test_the_result_round_trips_through_json(fake):
    r = measure_absorption(ScatteringSample.disc(flat_tile(), diameter=0.8), AcousticMaterial("panel", [0.3] * 7), n_repeats=1)
    t = json_loads(json_dumps(r))
    assert isinstance(t, AbsorptionResult) and np.allclose(t.alpha_s, r.alpha_s) and t.warnings == r.warnings
```

- [ ] **Step 2: Run them and see them fail**

Run: `pytest tests/lab/test_absorption.py -v`
Expected: FAIL (`ModuleNotFoundError: compas_acoustics.lab.absorption`).

- [ ] **Step 3: Implement `AbsorptionResult` in `lab/results.py` (same pattern as `ScatteringResult`, reuse `_plain`) and `measure_absorption`, `estimate_seconds`, `_method` in `lab/absorption.py`**

`AcousticAnalyser` and `Mono` are imported inside the functions that use them if importing them at module level would import pyfar/pyrato in `import compas_acoustics.lab` (check with `tests/test_python39.py`). Defaults for `n_repeats`, `n_rays`, `image_source_order` and the `estimate_seconds` constants are the spike's. The docstring of `measure_absorption` says why octaves and why the chamber is scaled, and cites ISO 354 §7.4.1, §8.1 and Annex A.

- [ ] **Step 4: Run them and see them pass**

Run: `pytest tests/lab/test_absorption.py tests/test_scene_serialisation.py tests/test_python39.py -v`
Expected: PASS (except the known pre-existing pickle test).

- [ ] **Step 5: Commit**

```bash
git add src/compas_acoustics/lab src/compas_acoustics/__init__.py tests/lab tests/test_scene_serialisation.py
git commit -m "Lab: measure_absorption, ISO 354 absorption of a sample in the scaled ray-traced chamber with paired seeds"
```

---

### Task 5: `lab_material`

> Note (2026-10-24): lab/material.py exists (C plan Task 2) with these rules for numbers and materials; this task adds the AbsorptionResult branch and its data_origin, and its first test now needs method="kirchhoff" passed explicitly (the default is geometric).

**Files:**
- Create: `src/compas_acoustics/lab/material.py`
- Modify: `src/compas_acoustics/lab/__init__.py`
- Test: `tests/lab/test_material.py`

**Interfaces:**
- Consumes: `AbsorptionResult` (Task 4), `ScatteringResult` (`.scattering_coefficients`, `.warnings`, `.method`, `.generation`), `AcousticMaterial`.
- Produces: `lab_material(name: str, absorption: AbsorptionResult, scattering: ScatteringResult = None, category=None, description: str = None) -> AcousticMaterial`: 7 absorption coefficients from `absorption.absorption_coefficients`, 7 scattering coefficients from `scattering.scattering_coefficients` (or none), both clipped to [0, 1] (`AcousticMaterial` refuses anything else).
  - A clipped value warns `f"{kind} coefficient {value:.2f} in the {band} Hz octave clipped to {limit:g}"`.
  - A NaN absorption or scattering value raises `ValueError` naming the octave (`f"no absorption coefficient in the {band} Hz octave"`).
  - Absorption above 0.5 in any octave with a scattering result warns `"ISO 17497-1 6.3.4: the sample absorbs more than 0.5, the scattering coefficient was computed for a rigid sample"`.
  - `data_origin = "compas_acoustics.lab: alpha_s by ISO 354 (ray traced, chamber scale {k:.2f}); s by {method} (preset {preset})"`, or `"... ; no scattering"` without a scattering result.

- [ ] **Step 1: Write the failing tests**

```python
def absorption_result(alpha, area=0.5):
    return AbsorptionResult(bands=[125, 250, 500, 1000, 2000, 4000, 8000], alpha_s=alpha, absorption_area=[a * area for a in alpha],
                            T1=[1.0] * 7, T2=[0.8] * 7, A1=[5.0] * 7, area=area, volume=1.9, chamber_scale=0.21, material="panel", n_decays=12, n_repeats=1)

def test_material_carries_both_sets_of_octave_coefficients():
    scattering = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), preset="survey", band_range=(1000, 1000))
    m = lab_material("test panel", absorption_result([0.1, 0.2, 0.3, 0.4, 0.5, 0.45, 0.4]), scattering)
    assert m.absorption_coefficients == [0.1, 0.2, 0.3, 0.4, 0.5, 0.45, 0.4] and len(m.scattering_coefficients) == 7
    assert "alpha_s by ISO 354" in m.data_origin and "s by kirchhoff (preset standard)" in m.data_origin

def test_without_scattering_the_material_has_none():
    m = lab_material("p", absorption_result([0.2] * 7))
    assert not m.scattering_coefficients and "no scattering" in m.data_origin

def test_out_of_range_values_are_clipped_with_a_warning():
    with pytest.warns(UserWarning, match=r"absorption coefficient 1\.08 in the 500 Hz octave clipped to 1"):
        m = lab_material("p", absorption_result([0.2, 0.2, 1.08, 0.2, 0.2, -0.02, 0.2]))
    assert m.absorption_coefficients[2] == 1.0 and m.absorption_coefficients[5] == 0.0

def test_a_nan_band_is_refused_naming_the_octave():
    with pytest.raises(ValueError, match="no absorption coefficient in the 2000 Hz octave"):
        lab_material("p", absorption_result([0.2, 0.2, 0.2, 0.2, float("nan"), 0.2, 0.2]))

def test_a_strongly_absorbing_sample_warns_about_the_rigid_scattering_assumption():
    scattering = measure_scattering(ScatteringSample.disc(flat_tile(), diameter=0.4), preset="survey", band_range=(1000, 1000))
    with pytest.warns(UserWarning, match="6.3.4"):
        lab_material("p", absorption_result([0.2, 0.2, 0.2, 0.7, 0.2, 0.2, 0.2]), scattering)
```

- [ ] **Step 2: Run them and see them fail**

Run: `pytest tests/lab/test_material.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `lab/material.py`; export `lab_material` from `lab/__init__.py`**

The scattering result's preset is `scattering.generation["preset"]`. Writing the library JSON stays with the existing `AcousticMaterial.to_json` / `add_to_library`; no new writer.

- [ ] **Step 4: Run them and see them pass**

Run: `pytest tests/lab/test_material.py tests/lab/test_scattering.py -k "material or default_preset" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/compas_acoustics/lab/material.py src/compas_acoustics/lab/__init__.py tests/lab/test_material.py
git commit -m "Lab: lab_material, an AcousticMaterial from the measured absorption and scattering, clipped and checked"
```

---

### Task 6: Validation in the real chamber (slow)

**Files:**
- Create: `tests/lab/test_absorption_validation.py`

**Interfaces:**
- Consumes: everything above with the real `Mono` ray tracer and the defaults the spike chose.

- [ ] **Step 1: Write the slow tests**

```python
pytestmark = pytest.mark.slow

PANEL = AcousticMaterial("panel", [0.3] * 7, [0.1] * 7)

def test_a_flat_sample_of_known_absorption_is_recovered():
    r = measure_absorption(ScatteringSample.disc(flat_tile(), diameter=0.8), PANEL)
    assert np.all(np.abs(np.array(r.alpha_s) - 0.3) <= 0.1), r.alpha_s

def test_the_empty_chamber_meets_table_1():
    r = measure_absorption(ScatteringSample.disc(flat_tile(), diameter=0.8), PANEL)
    assert not any("exceeds the ISO 354 limit" in w for w in r.warnings)

def test_a_relief_absorbs_more_than_the_flat_sample_of_the_same_footprint():
    flat = measure_absorption(ScatteringSample.disc(flat_tile(), diameter=0.8), PANEL)
    sine = measure_absorption(ScatteringSample.disc(sine_tile(0.0708, 0.0204, n=8), diameter=0.8), PANEL)
    assert np.mean(sine.alpha_s[1:6]) > np.mean(flat.alpha_s[1:6])
```

The ±0.1 tolerance and the relief test are the intent; if the spike shows the flat-sample standard deviation makes ±0.1 unreachable at the default cost, change the defaults (more rays or repeats), not the tolerance, and record it in the commit message.

- [ ] **Step 2: Run them once**

Run: `pytest tests/lab/test_absorption_validation.py -v -m slow -s`
Expected: PASS. Record the times and the measured `alpha_s` per octave of the three runs in the commit message.

- [ ] **Step 3: Commit**

```bash
git add tests/lab/test_absorption_validation.py
git commit -m "Lab: ray-traced absorption validated on a flat sample of known absorption and a relief"
```

---

## After this plan

- Run `trail check` before pushing.
- Next: the plan for **V1-C** (proto messages, conversions, worker functions, C# components and the Polar Preview), which wraps `measure_scattering`, `measure_absorption` and `lab_material`.
