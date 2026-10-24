# Virtual Coefficient Lab — design

Status: design approved 2026-10-23 (brainstorming). Branch: `feat/coefficient-lab` (from `up/hybrid-split`).
Sources: ISO 354:2003, ISO 17497-1:2004, ISO 17497-2:2012, Vorländer & Mommertz (2000), Heimes & Vorländer (2025)
with its scripts (Zenodo 15673498: `1.3_calculate_s_d.py`, `1.5_random_scattering.py`, `utils.py`).

## Context

compas_acoustics simulates rooms with materials that carry random-incidence **α** and **s** on 7 octave bands
(125 Hz–8 kHz, `AcousticMaterial`, `src/compas_acoustics/materials.py:50`). Those numbers currently come from the
library or from the `scattering_coefficients_from_depth` heuristic. The goal is a **virtual lab in Rhino 8 /
Grasshopper** that derives α and s for 3D surfaces you design, following ISO 354, ISO 17497-1/-2,
Vorländer & Mommertz (2000) and Heimes & Vorländer (2025, Zenodo 15673498). The coefficients then feed back into the
room simulations ("design loop").

### What you said
- Purpose: design-loop coefficients for parametric surfaces.
- Scale: mixed (panels to facade relief).
- Bands: one-third octaves **100 Hz–5 kHz by default** (the range of ISO 354 §5, ISO 17497-1 §5 and ISO 17497-2 §5),
  with an **optional extension to 8 kHz** (beyond the standards). Decided 2026-10-24 after the cost comparison:
  relative BEM cost 0.41 (default) vs 1.00 (extended) for a 0.8 m disc.
- Solver: BEM for scattering, the existing ray tracer for absorption.
- Scattering method chosen per run, trading accuracy for speed: **BEM** (accurate, hours) or **Kirchhoff**
  (fast, minutes). A virtual ISO 17497-1 turntable was dropped from V1 on 2026-10-24: see "Turntable: dropped" below.
- Compute: local Mac; overnight runs are acceptable.
- Sample: both "tile + crop to a disc" and "as drawn", chosen per run.
- Outputs, delivered in versions:
  - **V1:** an AcousticMaterial (plus JSON) and polar previews in Rhino.
  - **V2:** one-third-octave curves and s(θ,φ).
  - **V3:** bidirectional s_d as SOFA.

### Assumptions
- The scattering sample is computed **sound-hard** (rigid BEM). Absorption comes only from the ray-traced ISO 354
  room using the materials assigned to the geometry. ISO 17497-1 §6.3.4 limits samples to α_s ≤ 0.5 anyway; above that the lab warns.
- One-third-octave results fold into octave bands by arithmetic mean of 3, as in ISO 354 §8.3 / ISO 11654. The default
  range gives complete octaves 125 Hz–4 kHz. `AcousticMaterial` needs 7 octaves (125 Hz–8 kHz), so the 8 kHz octave is:
  - default range: the 4 kHz octave's value, with a warning that it was not simulated;
  - extended range: the mean of the 6.3 kHz and 8 kHz bands (10 kHz is not simulated), with a beyond-ISO warning.
  The full one-third-octave data stays in the result and in the SOFA file.
- The s_base check from ISO 17497-1 belongs to the reverberation-room method, which we are **not** using for s. For the BEM I use convergence checks instead (receiver resolution, Embrechts eq. 6;
  frequency points per band). These come in V2, together with the ±2σ uncertainty for α (ray-traced repeats, ISO 17497-1 Annex A.3).
- The sections "Architecture" and "Testing" describe the full V1–V3 scope. "Version split" says which parts ship when.

## What the documents dictate (method summary)

| Quantity | Source | How the lab computes it |
|---|---|---|
| s(θ,φ) | Mommertz correlation, Heimes eq. 3 | `1 − |Σ p₁ p₀* w|² / (Σ|p₁|²w · Σ|p₀|²w)` over the receiver hemisphere; p₀ = flat reference with the same footprint |
| s_d(θS,φS,θR,φR) | Heimes eq. 5; `1.3_calculate_s_d.py` | Same correlation, with p₀ = the reference at incidence (θR, φR+π); specular cell = 1 − s |
| random-incidence s | Paris / `1.5_random_scattering.py` | `1/π ∬ s(θ,φ) cos²θ sinφ dθ dφ` over incidence (cos²θ weighting); sources weighted sin(colatitude) |
| band range | ISO 354 §5; ISO 17497-1 §5; ISO 17497-2 §5, §6.3.3 | 1/3 octaves 100–5000 Hz by default; optional extension to 8000 Hz, flagged as beyond ISO |
| band values | Heimes eq. 11; ISO 17497-2 §7.4.4 | ≥ 3 frequency points per 1/3 band (simulate on a 1/9-octave grid), arithmetic mean |
| α_s | ISO 354 eq. 8–9 | `A_T = 55.3V(1/(c₂T₂) − 1/(c₁T₁)) − 4V(m₂−m₁)`; α_s = A_T/S, or A_obj = A_T/n for single objects |
| room and setup checks | ISO 354 §6, §7 | V ≥ 150 m³ (default ≈ 200), l_max < 1.9 V^⅓, A₁ ≤ Table 1, sample 10–12 m² ≥ 1 m from edges and not parallel to them, ≥ 2 sources × ≥ 3 mics (≥ 12 decays), T20 evaluated −5 to −25 dB |
| receiver grid | Heimes §3; Embrechts eq. 6 | pyfar equal-area grid with Voronoi weights, z > 0, far field (r ≫ d); Δθ ≤ 50λ/d degrees at the highest frequency |
| sample limits | ISO 17497-1 §6.3; Heimes App. A | h ≤ d/16; nΛ/λ ≫ 1. **Frequency-split runs**: larger disc / more periods at low frequency, smaller at high frequency, so elements stay below ~100k (edge length λ/6–λ/7) |
| storage (V3) | Heimes §6.3, Table 1 | SOFA GeneralTF plus SpeedOfSound, SampleStructuralWavelengthX/Y, SampleDiameter, Receiver/SourceWeights, RandomIncidentScattering |

## Scattering methods (accuracy ↔ speed)

| Method | Pipeline | Cost | Valid when | Misses |
|---|---|---|---|---|
| `"bem"` | free field, complex p₁/p₀ on the receiver hemisphere → Mommertz eq. 3 → random incidence | hours (overnight for the full range) | always (reference method) | — |
| `"kirchhoff"` | **same** free-field pipeline; p₁/p₀ from the physical-optics (Kirchhoff) surface integral | seconds to minutes | smooth relief, curvature radius ≫ λ, mostly single reflections | shadowing, multiple scattering, deep cavities, some low-frequency edge effects |

- **Kirchhoff solver.** For plane-wave incidence k̂ᵢ and far-field direction k̂ₛ,
  `p(k̂ₛ) ∝ Σ_facets lit (n·(k̂ᵢ − k̂ₛ)) ∫_T e^{jk(k̂ᵢ−k̂ₛ)·x} dS` (the Beckmann form, chosen 2026-10-24 over the rigid
  n·k̂ₛ form: same cost, reciprocal, which V3's s_d reciprocity check needs; s differs by ≤ 0.02 on the sine surface). The triangle phase integral is evaluated in closed
  form (Gordon's formula), so the mesh does not need λ/6 refinement. Illumination comes from n·k̂ᵢ < 0, with an optional
  ray-cast shadow test using the existing geometry tools. The reference disc is computed the same way, so edge terms
  largely cancel in the correlation, as the paper notes for p₀/p₁. Vectorised numpy, chunked over receivers.
- **Validity warnings.** Kirchhoff warns per band where the feature curvature radius or depth approaches λ, or where
  shadowing is significant. Both solvers write
  the method into `generation` / the result metadata, so a material always records how its s was obtained.
- A **solver interface** `ScatteringSolver.solve(sample, grids, frequencies) → (p₁, p₀)` is used by `"bem"` and `"kirchhoff"`.

### Turntable: dropped (2026-10-24)
The virtual ISO 17497-1 method (T₁–T₄, rotating sample, phase-locked averaging of impulse responses) needs coherent responses,
which only image sources give, and the standard evaluates the decay from −5 to −20 dB (more than 1 s in a ~200 m³ chamber).
Measured with `Mono(ray_tracing=False)` in a 7.2×5.3×5.2 m chamber (walls α 0.05): the response grows ~0.02 s per image order, so
order ~50 is needed, while a closed 12-face obstacle already costs 4.5 s at order 7 (the empty box 0.0 s at order 6, 1.1 s at order 12),
growing steeply. That times 72 positions × source/mic pairs × 2 is out of reach, and in its validity regime (features ≫ λ) the
Kirchhoff method already gives the geometric-optics limit. Revisit with a faster coherent engine.

## Objects, the proxy reference and the geometric method (added 2026-10-24)

Source: evaluation of Shtrepi et al. (Building Acoustics 2020, doi 10.1177/1351010X20967821), which traces specular rays off a rigid panel and
reports a frequency-independent uniformity index, not s. Their method is kept only as the lab's fast default, repaired (below).

**Objects.** Surfaces and objects **mounted** on a wall, floor or ceiling (sound arrives from one side: the hemisphere grids stand), 1 to ~10 m
across, possibly with deep cavities and shadowing. Surfaces with simple relief can keep using `scattering_coefficients_from_depth`.

**Proxy reference.** `ScatteringSample.as_drawn(mesh, reference=proxy)` takes an optional proxy: the simplified mesh the room model will carry the
lab's s on. s then measures the share of the detailed object's reflection that the proxy does not reproduce. Without a proxy the reference is the
flat footprint, as before (the ISO definition). A proxy result is flagged non-ISO (`sample.reference_kind == "proxy"`, a warning). The proxy and the
sample share one outline. In the room model the material goes on the proxy's faces, never also on the detailed mesh (double counting).

**Methods** (`measure_scattering(method=...)`):

| Method | Status | Pipeline |
|---|---|---|
| `"geometric"` (default) | D2 | specular rays off the rigid mesh, binned on the receiver hemisphere; **heuristic** s(f) from the geometry; seconds |
| `"kirchhoff"` | built | as above |
| `"sbr"` | D3, later | shooting and bouncing rays with phase and a physical-optics exit integral, into the same correlation pipeline |
| `"bem"` | built | as above |

**Geometric method.** Per source direction û (the `SourceGrid` presets, default `"standard"`), `n_rays` parallel rays are launched from uniformly random
points of the reference's flat bottom face (the same points for the sample and the reference), aimed at z = 0. The object is **mounted in a rigid baffle
at z = 0 outside the footprint**: a downward ray reaching z = 0 outside the footprint reflects specularly there and counts as a bounce; inside the footprint
it reflects off whatever face it hits. Rays reflect specularly up to `max_bounces` and leave upward into one cell of the receiver grid
(`ReceiverGrid.hemisphere(bin_deg)`, default 10°; the Voronoi weights are the cells' solid angles). A ray that exits along the horizon is binned into the
nearest cell; `lost` is only the rays still bouncing after `max_bounces` (warned above 1 %). The binned and the lost rays are counted separately, so
binned + lost = 1 is a check, not an assumption. The rays are aimed only at the footprint, so for a tall object at oblique incidence the rays that would
strike its sides from outside the footprint are not traced.
- Geometric scattering share `s_max(û) = 1 − Σ_bins min(E_sample, E_reference)` over the energy fractions per cell **plus the lost fraction as one extra bin** (1 − the share in the specular cell for a flat reference).
- Depth `h = max(2·σ_z, |mean_z|)`, z the height difference between the sample's and the reference's top surfaces at the footprint points, σ_z its
  standard deviation: a relief gets 2σ (0.577 H for a sawtooth of height H, 0.707 H for a sine, against a mean of 0.5 H), an object of uniform height over
  its footprint (a box, a cylinder, a flat-topped panel, σ_z = 0) its height. A depth curve below 0.5 at the highest band while the mean s_max exceeds 0.05
  is warned about (s close to 0 although the rays scatter).
- `s(û, f) = s_max(û)·(1 − exp(−(k h)²))`, k = 2πf/c (the form of `AcousticMaterial.scattering_coefficients_from_depth`, with s_max and h taken from the geometry), evaluated at each band's exact centre. The result is labelled heuristic (`generation["heuristic"]`, a warning).
- The polar levels are the cells' energy per steradian, the same at every octave. The rim is not diffracted; the flat-reference correlation cancels it for `kirchhoff` and `bem`, not here.
- `geometric` is the default method of `measure_scattering` and needs `trimesh` (ideally with `embreex`; Python >= 3.11, the `lab` extra); `method="kirchhoff"` works without it. The method's accuracy against Kirchhoff and its validity limits are recorded in the module docstring of `lab/geometric.py` ("Accuracy record").

**Open design choices:** the depth factor (`h = max(2·σ_z, |mean_z|)`, 2σ for a relief; the accuracy record shows the curve above Kirchhoff by up to 0.19 at 2 and 4 kHz on the Heimes sine, not settled by one sample), the bin width (10°), and how d is averaged over random incidence.

**Diffusion coefficient.** Every method also returns d (ISO 17497-2 eq. 6 on the intensities at the receivers, weighted by the cells' solid angles) and the
normalised d_n = (d − d_ref)/(1 − d_ref) per band and octave; random incidence uses the same Paris weights as s.

**Order of work:** D1 the proxy reference and d, D2 the geometric method, C the Grasshopper chain (with the final method list), D3 SBR (starts with a spike on
tube density, ray throughput and the receiver count at 10 m, where Δθ ≤ 50λ/d means ~10⁵ receivers at 5 kHz).

## Architecture

The work is delivered in **three versions**:
- **V1** carries sub-projects A, B and C below, restricted to what is needed for the AcousticMaterial and the polar previews.
- **V2** and **V3** extend the same modules (see "Version split" below).

Each sub-project gets its own implementation plan. They are built in this order:

### A. Scattering core (Python, no GH): `src/compas_acoustics/lab/`
- `sample.py`: `ScatteringSample.disc(tile_mesh, diameter, ...)` tiles, crops to a circle and puts it on a flush
  baseplate. `ScatteringSample.as_drawn(mesh)` keeps the mesh as given. Both build the matching flat reference
  (same outline) and run the checks (h ≤ d/16, element count estimate per frequency range, closed baseplate).
  The geometry is COMPAS `Mesh`.
- `grids.py`: source presets `survey` (ISO 17497-2 Table 1, 13), `standard` (colatitude 0–80° step 10° × azimuth 30°, 97) and
  `fine` (Heimes: colatitude 1.26°…88.74° step 2.52° × azimuth 15°, 864), collapsed to azimuth 0 when the sample is
  rotationally symmetric. **Default is `standard` for both methods** (decided 2026-10-24: first `kirchhoff` → `fine`, switched to `standard`
  after measuring a default fine Kirchhoff run at ~45 min on 14 cores vs ~5 min standard; the fine grid also makes BEM ~9× longer). Comparisons between methods pass the same preset explicitly. Also: the receiver grid (above),
  1/9-octave frequency grid with frequency-range splits.
- `solvers/kirchhoff.py`: the fast free-field solver (above), with no extra dependencies.
- `solvers/bem.py`: NumCalc backend via `mesh2scattering` (`write_scattering_project_numcalc`, `manage_numcalc`,
  `output.write_pressure`). Plane-wave incidence on sample and reference, a background subprocess with progress and cancel,
  results read back as pyfar `FrequencyData` + `Coordinates`. Project folders are cached so an interrupted overnight run resumes.
- `coefficients.py`: pure numpy, no I/O: `scattering_freefield` (eq. 3; reuse `mesh2scattering.process.scattering_freefield`
  where its signature fits), `bidirectional_scattering` (eq. 5, reference rotated in azimuth for disc samples,
  simulated for every incidence direction when as drawn), `random_incidence`, `band_average`, `fold_to_octaves`,
  and `sd_checks` (reciprocity eq. 1, energy sum eq. 2, normalisation before use, as Heimes §7 requires).
- `sofa.py`: write and read pressure / s_d / s / s_rand with the Table 1 fields (`sofar`).
- `results.py`: `ScatteringResult` (1/3-octave s, s(θ,φ), s_d, metadata). Serialises under the package dtype like the other classes.

### B. Absorption room (Python): `src/compas_acoustics/lab/absorption.py`
- `ReverberationChamber`: an ISO 354 shoebox (7.4 × 5.8 × 4.6 m, V = 197.4 m³, shape and ratio rules §6.1.2) **scaled to the sample**:
  all lengths × k = √(S / 11 m²) with S the sample's footprint area, because decay times and areas scale with length but α does not,
  and in a full-size chamber a 0.5 m² sample changes the decay (~1.5 %) less than the ray tracer's seed-to-seed spread (~4 %).
  Walls α 0.025, s 0.3 (the diffusers of Annex A), 2 sources × 6 microphones = 12 decays placed to §7.1.2–7.1.4 (seeded, deterministic).
- `measure_absorption(sample, material, chamber, air, n_repeats, n_rays, ...)`: runs `Mono` (`simulation_methods/pra.py:33`, ray tracing)
  empty (T₁) and with the sample as an obstacle on the floor (T₂), the same seed for both of a repeat. T20 per **octave** comes from
  `AcousticAnalyser.reverberation_times()` (the engine's materials are octave-wise, so third-octave decay times carry no extra
  information); the ISO 354 formulas (eq. 5–9) are in `lab/iso354.py`. Output: an `AbsorptionResult` with α_s per octave and the
  intermediate T₁, T₂, A₁, A_T; the 8000 Hz octave is flagged as beyond ISO 354. ±2σ (ISO 17497-1 A.3) is added in V2.
- `lab/material.py`: `lab_material(name, absorption, scattering=None)` → `AcousticMaterial` (7 octaves, clipped to [0, 1] with a warning,
  NaN refused), the Python side of the Lab Material component.
- The hybrid/FDN method is not used here, because its tail is statistical.

### C. Grasshopper integration: C# components, compas_pb data exchange

**C as planned on 2026-10-24** (`docs/superpowers/plans/2026-10-24-coefficient-lab-c-grasshopper.md`), before B is built:
- **Components first:** Scattering Sample, Scattering Lab, Polar Preview and Lab Material. Lab Material takes α as numbers or from a material, and s from the lab. The Absorption Lab component, `ReverberationChamberData` and `AbsorptionResultData` follow B.
- **Messages:** only `ScatteringSampleData` and `ScatteringResultData`, each mirroring its class's `__data__`, with arrays as `ArrayData` (values plus shape). The method settings travel as a kwargs dict, as `simulate`'s settings do, so there is no `LabGridData` or `ScatteringSettingsData`. The polar levels stay in the result's arrays, and a worker function `polar_balloon` turns one source and octave into a coloured mesh, instead of a `PolarResponseData` per balloon.
- **Runs:** `geometric` runs as soon as its inputs change; `kirchhoff` and `bem` run only on the Run button (minutes to hours).
- **Worker environment:** the installer's worker environment installs `compas_acoustics[lab]`, so trimesh, embreex and mesh2scattering are there. NumCalc for BEM stays a separate setup step.
- **Units:** Rhino geometry is taken in metres, as the other components take it. The sample's existing "model in millimetres?" warning stands.

The pattern follows the existing plugin. There are no CPython/GhPython components:
**C# `ComponentBase` components** call the **Python worker** over its local HTTP API, and every lab object crosses the
boundary as a **compas_pb message**.

- **Proto messages:** new messages in `src/compas_acoustics/pb/protobuf_defs/compas_acoustics/pb/generated/acoustics.proto`,
  laid out like the existing ones (guid and name first, mirroring each class's `__data__`, reusing `compas_pb.data.MeshData` /
  `PointData` / `VectorData` and the existing `AcousticMaterialData`):
  - `ScatteringSampleData`: mode (disc / as_drawn), sample `MeshData`, reference `MeshData`, diameter, structural depth,
    structural wavelength x/y, symmetry.
  - `LabGridData`: source directions plus weights, receiver points plus weights, frequencies, frequency splits.
  - `ScatteringSettingsData`: method (`bem` / `kirchhoff`), band range, quality preset, cache folder.
  - `ReverberationChamberData`: room `MeshData`, wall material, source and mic points, volume, air (temperature, humidity).
  - `PolarResponseData`: incidence direction, band, receiver points, sample and reference levels in dB (for the balloons).
  - `ScatteringResultData`: method, bands, random-incidence s per band, repeated `PolarResponseData`, warnings, generation.
    Fields reserved for V2 (1/3-octave s, s(θ,φ)) and V3 (s_d, SOFA path), so later versions add data without renumbering.
  - `AbsorptionResultData`: bands, α_s or A_obj, T₁/T₂ per band, mode (area / object). V2 adds σ.
  - The lab's material output is the existing `AcousticMaterialData`.
- **Generated bindings:**
  - Python: `invoke generate-proto-classes` → `src/compas_acoustics/pb/generated`.
  - C#: `invoke generate-csharp` → `csharp/CompasAcoustics.Grasshopper/Generated/Acoustics.cs`.
  - The new messages live in the same generated assembly as `SceneData`, so `CompasPbRegistration.Register()`
    (`csharp/CompasAcoustics.Grasshopper/CompasPbRegistration.cs`) picks them up with no change.
- **Python conversions:** `*_to_pb` / `*_from_pb` pairs for each lab class in `src/compas_acoustics/pb/conversions.py`,
  following `material_to_pb` / `scene_to_pb`. Round-trip fixtures go in `tests/pb_fixtures/*.b64`.
- **C# data and params:** Goo wrappers for the new messages in `csharp/CompasAcoustics.Grasshopper/Data/`
  (a `LabGoo.cs` modelled on `ModelGoo.cs` / `MessageGoo.cs`), parameters in `Params/Params.cs`, and balloon and
  chamber drawing in `Preview/` and `Display/`. Curves (V2) keep the existing `ParameterGoo` → Plot 2D path.
- **Worker functions** in `src/compas_acoustics/worker/api.py`, added to `__all__`. They take and return compas_pb messages.
  Long runs use `RunInBackground` + `/progress` + `/cancel`:
  - `make_scattering_sample`
  - `make_reverberation_chamber`
  - `run_scattering_lab`
  - `run_absorption_lab`
  - `lab_material`
- The mesh2scattering / NumCalc dependency stays in the worker's Python environment (3.11+). The C# side only sees messages.
- New components (each with a spec in `Components/Specs/CA_<Name>/metadata.json` plus icon, a `ComponentBase` class, and an entry in `SpecCoverageTests`):
  - **Scattering Sample**: mode (disc / as drawn), diameter, symmetry; previews the sample, the reference and the source/receiver hemispheres.
  - **Scattering Lab**: method (`bem` / `kirchhoff`), frequency range, quality preset, output folder; runs in the background. Outputs: the result (V1); 1/3-octave s and s(θ) as Parameter curves (`_parameter`, `api.py:253`) (V2); the SOFA path (V3).
  - **Absorption Lab**: sample, materials, chamber settings, air; outputs the absorption result (V1), α_s or A_obj curves with uncertainty (V2).
  - **Lab Material**: combines α and s into an `AcousticMaterial` and optionally writes library JSON.
  - **Polar Preview**: per band and incidence direction, the reflected-level polar response as a coloured mesh balloon (V1); an s_d mode as in the paper's Fig. 10 (V3).
- Components exchange the lab messages above as typed Goo, so they can be wired, cached and internalised like the existing Scene and Material parameters.

## Version split

**V1: material and previews**
- Both scattering methods ship in V1: `kirchhoff` and `bem` in sub-project A, and the method input on the Scattering Lab component in C.
- The polar preview balloons are available for both methods.
- The internal pipeline computes s(θ,φ) on the source grid only as an intermediate result. Random-incidence s needs it, but it is not exposed.
- `coefficients.py` provides: `scattering_freefield`, `random_incidence`, `band_average`, `fold_to_octaves`.
- Absorption returns α_s or A_obj per octave band (repeats are averaged; no uncertainty output).
- `results.py` provides `ScatteringResult` and `AbsorptionResult`, holding octave values and the pressures needed for previews.
- **Lab Material**: `to_material()` → `AcousticMaterial`, plus library JSON.
- Grasshopper components: Scattering Sample, Scattering Lab, Absorption Lab, Lab Material and a **Polar Preview** component. The preview shows the sample, the reference, the source/receiver hemispheres, and a balloon per octave band and incidence direction of the **reflected sound pressure level** (ISO 17497-2 §3.8 hemispherical polar response, p₁ against p₀).
- NumCalc pressures are cached in mesh2scattering's own project and output files. This is internal only: no public SOFA API.

**V2: curves and angle dependence**
- Exposes the one-third-octave α_s and s and the s(θ,φ) per incidence direction as Parameter curves (`_parameter`, `api.py:253`) for Plot 2D.
- Adds the ±2σ uncertainty for α (ISO 17497-1 A.3) and the BEM convergence diagnostics.
- Extends the Scattering Lab and Absorption Lab outputs. No new solver work.

**V3: bidirectional s_d**
- `bidirectional_scattering` (eq. 5): the reference is rotated in azimuth for disc samples and simulated per incidence direction for as-drawn samples. Adds `sd_checks` (reciprocity eq. 1, energy sum eq. 2, normalisation).
- `sofa.py` writes and reads the s_d / s / s_rand files with the Table 1 fields.
- The Polar Preview gains an s_d balloon mode (the paper's Fig. 10).
- **Grid consequence:** an as-drawn sample needs the reference simulated for every incidence direction. V1 already simulates the reference for all source directions in as-drawn mode, so V3 needs no new BEM runs on cached projects.

## Data flow
Rhino mesh → Scattering Sample → (sample, reference, grids) → Scattering Lab →
[`bem`: NumCalc ×2 (×frequency splits) | `kirchhoff`: surface integral] → p₁, p₀ → coefficients
→ ScatteringResult (+ SOFA in V3) → Lab Material ← AbsorptionResult ← Absorption Lab (ray tracer T₁/T₂) → AcousticMaterial → existing Room/Scene simulation.

## Error handling
- Bands above 5 kHz (extended range) and an 8 kHz octave filled from 4 kHz (default range) produce warnings.
- Unmet ISO checks (h > d/16, sample outside 10–12 m², A₁ above Table 1, too few decays, α_s > 0.5 for s) produce **warnings**, not errors, via the existing `_messages` → `ShowWarnings` path.
- A missing NumCalc binary or Python < 3.11 for mesh2scattering gives a clear error with setup instructions. The lab extra is optional and does not affect Rhino's 3.9 environment, because the worker is a separate process.
- An estimated element count above the configured limit gives a warning plus a suggested split or a smaller diameter, before launch.
- Cancel kills NumCalc. Finished frequencies stay cached.

## Testing and validation
- Unit (fast, V1): correlation (p₁ = p₀ → 0; p₁ = p₀·e^{jφ} → 0; uncorrelated → ≈1), random-incidence integral of constant s → s, band averaging, octave folding, grid weights summing to 2π, ISO 354 formula against a hand calculation, `to_material()` producing a valid `AcousticMaterial` that round-trips through JSON.
- Unit (V3): s_d reciprocity and energy sum, SOFA round-trip.
- Kirchhoff (fast):
  - Gordon's triangle integral against brute-force quadrature on a fine sub-mesh.
  - A flat disc against itself → s = 0.
  - An infinite-plate limit: a large disc gives a specular lobe at θₛ = θᵢ.
- Kirchhoff vs BEM (slow): sine surface random-incidence s. Expect agreement in the upper bands and documented deviation at low Λ/λ; record the measured gap in the docstring as the method's known accuracy.
- Absorption (slow): a flat sample of a known broadband α in the virtual chamber returns α within tolerance. The empty chamber passes Table 1.
- BEM integration (slow): flat disc vs itself → s ≈ 0. **Reference case**: Heimes' simulated sine surface (h = 20.4 mm, Λ = 80 mm, n = 10 periods across d = 0.8 m, as in Heimes's `0.1_simulate_optimal.py`, which made the Zenodo results) at θ = 31.5° / 61.74° and random incidence, compared with the Zenodo `results_simulation_optimal` SOFA and the Embrechts analytical curve. The tolerance is the paper's one-third-octave agreement (|Δs| ≲ 0.1 above Λ/λ = 0.5). (The paper's text value Λ = 70.8 mm describes the physical sample, not the simulated one compared here.)
- compas_pb: Python round-trip of every new message (`*_to_pb` → bytes → `*_from_pb` equals the original), and
  `.b64` fixtures decoded on the C# side (as in `GooTests.cs`) to prove both bindings agree.
- C#: component spec coverage (`SpecCoverageTests`, `AllComponentsTests`) and the fake-worker tests (`Fakes.cs`, `Worker.Override`) for each lab component.

## Risks → a spike at the start of sub-project A
1. Install `mesh2scattering` + build NumCalc on macOS arm64; confirm a Python 3.11+ worker environment.
2. Check that mesh2scattering's `SampleMesh` accepts as-drawn outlines and full-hemisphere (azimuth) incidence. If not, write the NumCalc project directly (Mesh2HRTF format).
3. Time one 0.8 m disc at 8 kHz on the Mac, to set the quality presets and split defaults.

## Next steps
1. Review of this written spec.
2. `superpowers:writing-plans` for **V1**: sub-project A (spike first), then B, then C, each as its own plan.
   V2 and V3 get their own plans after V1 is in use.
