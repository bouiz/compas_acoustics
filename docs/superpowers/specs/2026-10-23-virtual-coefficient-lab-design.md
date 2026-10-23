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
- Scattering method chosen per run, trading accuracy for speed: **BEM** (accurate, hours), **Kirchhoff**
  (fast, seconds to minutes) or the **ray-traced turntable** (virtual ISO 17497-1, valid only for large features).
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
- The s_base check and the ±2σ turntable uncertainty from ISO 17497-1 belong to the reverberation-room method, which
  we are **not** using for s. For the BEM I use convergence checks instead (receiver resolution, Embrechts eq. 6;
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
| `"turntable"` | virtual ISO 17497-1 in the ISO 354 chamber of sub-project B: T₁–T₄, rotating sample, phase-locked averaging | minutes to an hour (n positions × source/mic pairs) | features ≫ λ (upper bands, large relief) | all diffraction below the feature scale |

- **Kirchhoff solver.** For plane-wave incidence k̂ᵢ and far-field direction k̂ₛ,
  `p(k̂ₛ) ∝ Σ_facets lit (n·(k̂ᵢ − k̂ₛ)) ∫_T e^{jk(k̂ᵢ−k̂ₛ)·x} dS` (the Beckmann form, chosen 2026-10-24 over the rigid
  n·k̂ₛ form: same cost, reciprocal, which V3's s_d reciprocity check needs; s differs by ≤ 0.02 on the sine surface). The triangle phase integral is evaluated in closed
  form (Gordon's formula), so the mesh does not need λ/6 refinement. Illumination comes from n·k̂ᵢ < 0, with an optional
  ray-cast shadow test using the existing geometry tools. The reference disc is computed the same way, so edge terms
  largely cancel in the correlation, as the paper notes for p₀/p₁. Vectorised numpy, chunked over receivers.
- **Turntable solver.**
  - **Responses:** coherent responses only. It uses the `Mono` method (`simulation_methods/pra.py:33`) with `ray_tracing=False` and a high
    `image_source_order`, because the ray-traced tail is random-sign
    and identical across positions with a fixed seed (see the exploration notes).
  - **Rotation:** the sample is rotated in n steps (default 72, ISO 17497-1 §7.3). The responses are averaged per
    source/mic pair before backward integration.
  - **Evaluation:** integration limit −30 dB, T over −5 to −20 dB on the first decay (§7.5).
  - **Results:** α_s (eq. 1), α_spec (eq. 4), s (eq. 5), s_base (eq. 6) checked against Table 1, and ISO 17497-1 room
    checks (V ≥ 200 N⁻³, A₁ ≤ 0.3 V^⅔, disc d ≥ 3 m, h ≤ d/16, displaced centre for rotationally symmetric samples §6.3.3).
- **Validity warnings.** Kirchhoff warns per band where the feature curvature radius or depth approaches λ, or where
  shadowing is significant. The turntable warns per band where λ exceeds the characteristic feature size. Both write
  the method into `generation` / the result metadata, so a material always records how its s was obtained.
- A **solver interface** `ScatteringSolver.solve(sample, grids, frequencies) → (p₁, p₀)` is used by `"bem"` and
  `"kirchhoff"`. The turntable has its own `measure_scattering(...)` because it computes reverberation times, not pressures.

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
  rotationally symmetric. **Default depends on the method** (decided 2026-10-24): `kirchhoff` → `fine`, `bem` → `standard`
  (the fine grid makes BEM ~9× longer). Comparisons between methods pass the same preset explicitly. Also: the receiver grid (above),
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
- `ReverberationChamber`: about 200 m³, non-rectangular-ratio box, wall α tuned to stay within ISO 354 Table 1,
  high wall scattering to stand in for diffusers (Annex A), and source/mic placement satisfying §7.1.2–7.1.4.
- `measure_absorption(sample, materials, chamber, air, n_repeats)`: runs `Mono` (`simulation_methods/pra.py:33`)
  empty (T₁) and with the sample as an obstacle on the floor (T₂). `seed` varies per repeat. T20 comes from
  `AcousticAnalyser.reverberation_times()` (`analyser.py:385`); c and m come from `AirConditions` (`room/air.py:19`). The ISO 354 inverse formula is new.
  Output: an `AbsorptionResult` with α_s or A_obj per band; ±2σ (ISO 17497-1 A.3) is added in V2.
- The hybrid/FDN method is not used here, because its tail is statistical.
- `lab/turntable.py`: `measure_scattering(sample, chamber, n_positions=72, ...)`. It reuses `ReverberationChamber`
  plus a turntable and base plate, and computes T₁–T₄ (ISO 17497-1 Table 2) as described under "Scattering methods".

### C. Grasshopper integration: C# components, compas_pb data exchange
The pattern follows the existing plugin. There are no CPython/GhPython components:
**C# `ComponentBase` components** call the **Python worker** over its local HTTP API, and every lab object crosses the
boundary as a **compas_pb message**.

- **Proto messages:** new messages in `src/compas_acoustics/pb/protobuf_defs/compas_acoustics/pb/generated/acoustics.proto`,
  laid out like the existing ones (guid and name first, mirroring each class's `__data__`, reusing `compas_pb.data.MeshData` /
  `PointData` / `VectorData` and the existing `AcousticMaterialData`):
  - `ScatteringSampleData`: mode (disc / as_drawn), sample `MeshData`, reference `MeshData`, diameter, structural depth,
    structural wavelength x/y, symmetry.
  - `LabGridData`: source directions plus weights, receiver points plus weights, frequencies, frequency splits.
  - `ScatteringSettingsData`: method (`bem` / `kirchhoff` / `turntable`), band range, quality preset, cache folder, n_positions.
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
  - **Scattering Lab**: method (`bem` / `kirchhoff` / `turntable`), frequency range, quality preset, output folder; runs in the background. Outputs: the result (V1); 1/3-octave s and s(θ) as Parameter curves (`_parameter`, `api.py:253`) (V2); the SOFA path (V3).
  - **Absorption Lab**: sample, materials, chamber settings, air; outputs the absorption result (V1), α_s or A_obj curves with uncertainty (V2).
  - **Lab Material**: combines α and s into an `AcousticMaterial` and optionally writes library JSON.
  - **Polar Preview**: per band and incidence direction, the reflected-level polar response as a coloured mesh balloon (V1); an s_d mode as in the paper's Fig. 10 (V3).
- Components exchange the lab messages above as typed Goo, so they can be wired, cached and internalised like the existing Scene and Material parameters.

## Version split

**V1: material and previews**
- All three scattering methods ship in V1:
  - `kirchhoff` and `bem` in sub-project A
  - `turntable` in sub-project B, since it needs the chamber
  - the method input on the Scattering Lab component in C
- The polar preview balloons are available for `bem` and `kirchhoff`. The turntable has no free-field pressures, so for it the preview shows the room, the turntable and the source/mic positions.
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
(or [`turntable`: chamber T₁–T₄ → α_spec, α_s → s]) → ScatteringResult (+ SOFA in V3) → Lab Material ← AbsorptionResult ← Absorption Lab (ray tracer T₁/T₂) → AcousticMaterial → existing Room/Scene simulation.

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
- Turntable (slow):
  - A flat disc gives s ≈ 0, and s_base within Table 1.
  - Large-feature relief (e.g. randomly placed boxes ≫ λ) gives s that rises in the upper bands.
  - Averaging n identical rigid-sample responses reproduces T₂ (the coherence sanity check).
- Absorption (slow): a flat sample of a known broadband α in the virtual chamber returns α within tolerance. The empty chamber passes Table 1.
- BEM integration (slow): flat disc vs itself → s ≈ 0. **Reference case**: Heimes' sine surface (h = 20.4 mm, Λ = 70.8 mm, d = 0.8 m) at θ = 31.5° / 61.74° and random incidence, compared with the Zenodo `results_simulation_optimal` SOFA and the Embrechts analytical curve. The tolerance is the paper's one-third-octave agreement (|Δs| ≲ 0.1 above Λ/λ = 0.5).
- compas_pb: Python round-trip of every new message (`*_to_pb` → bytes → `*_from_pb` equals the original), and
  `.b64` fixtures decoded on the C# side (as in `GooTests.cs`) to prove both bindings agree.
- C#: component spec coverage (`SpecCoverageTests`, `AllComponentsTests`) and the fake-worker tests (`Fakes.cs`, `Worker.Override`) for each lab component.

## Risks → a spike at the start of sub-project A
1. Install `mesh2scattering` + build NumCalc on macOS arm64; confirm a Python 3.11+ worker environment.
2. Check that mesh2scattering's `SampleMesh` accepts as-drawn outlines and full-hemisphere (azimuth) incidence. If not, write the NumCalc project directly (Mesh2HRTF format).
3. Time one 0.8 m disc at 8 kHz on the Mac, to set the quality presets and split defaults.
4. Turntable cost: image-source order × faces of a 3 m relief disc × 72 positions × 6 pairs × 2. Measure this in the B plan before committing to the defaults. If it is too slow, cap the order and the face count (merging coplanar faces) and warn.

## Next steps
1. Review of this written spec.
2. `superpowers:writing-plans` for **V1**: sub-project A (spike first), then B, then C, each as its own plan.
   V2 and V3 get their own plans after V1 is in use.
