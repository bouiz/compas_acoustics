# Coefficient lab C: the Grasshopper chain

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In Grasshopper, a user wires Rhino geometry into **Scattering Sample → Scattering Lab** and gets s per octave, d and polar balloons (**Polar Preview**). **Lab Material** then turns the result and an absorption into an `AcousticMaterial` for the room components, with no Scene, source or receiver.

**Architecture:**
- **Pattern:** the plugin's existing one. C# `ComponentBase` components call the Python worker over its local HTTP API with compas_pb payloads.
- **New messages:** two, `ScatteringSampleData` and `ScatteringResultData`, each mirroring its class's `__data__`.
- **New worker functions:** four, `make_scattering_sample`, `run_scattering_lab`, `polar_balloon` and `lab_material`.
- **Testable helpers:** a component's `Solve` cannot run outside Grasshopper, so request building and result reading live in static helpers (`LabRequests`, `LabResults`) that the xunit tests drive, as `SimulationSettings` is driven.

**Tech Stack:** Python 3.12 worker, compas_pb, protobuf (`invoke generate`), C# net7.0 Grasshopper plugin, xunit (net8.0), pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-virtual-coefficient-lab-design.md`, section "C. Grasshopper integration" and its paragraph "C as planned on 2026-10-24". Absorption (B) is not built. Its component, `ReverberationChamberData` and `AbsorptionResultData` come after B. Also out of scope here: the sample's source and receiver hemisphere preview, and writing library JSON from Lab Material. The material can be internalised in Grasshopper, and `AcousticMaterial.to_json` exists.

## Global Constraints

- **Branch:** `feat/coefficient-lab`. Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- **Python style:** ruff `line-length = 179`, force-single-line imports, numpy docstrings, `target-version = "py39"`. `compas_acoustics.lab` and `compas_acoustics.pb` must still import on Python 3.9.
- **Python checks:** `ruff check src tests` and `pytest tests -q -m "not slow"` before each commit that touches Python.
- **C# check:** `dotnet test csharp/CompasAcoustics.Grasshopper.Tests --nologo` before each commit that touches C#. It runs 434 tests today, and every one must still pass.
- **Messages:** go in `src/compas_acoustics/pb/protobuf_defs/compas_acoustics/pb/generated/acoustics.proto`, guid and name first, mirroring `__data__`. Regenerate both bindings with `invoke generate`, which writes `src/compas_acoustics/pb/generated`, `csharp/CompasAcoustics.Grasshopper/Generated/Acoustics.cs` and `Schema.cs`, and commit them. Never hand-edit generated files.
- **C# fixtures:** after a schema change, run `python tests/pb_fixtures/generate.py`, which writes `csharp/CompasAcoustics.Grasshopper.Tests/Fixtures/*.b64` with fixed guids.
- **Worker functions:** live in `src/compas_acoustics/worker/api.py` and are listed in `__all__`. They take and return what compas_pb carries. Bad input raises `ValueError` or `TypeError`. A function that returns warnings returns them under `"warnings"`, collected with `_messages(caught)` as `simulate` does.
- **Each new component:**
  - a spec in `csharp/CompasAcoustics.Grasshopper/Components/Specs/CA_<Name>/metadata.json` (same schema as `CA_Construct_Material`);
  - subcategory `"08 Lab"`, exposure 2, and an icon from `tools/icons.py`, with a glyph per spec and accent `"08 Lab": (140, 86, 75)`;
  - a `ComponentBase` class, and an entry in `SpecCoverageTests.Components`.
- **GUIDs** are `uuid5(NAMESPACE_URL, "compas_acoustics/grasshopper/<key>")`:

  | Key | GUID |
  |---|---|
  | `CA_Scattering_Sample` | `ff1614f6-bb68-5861-aed6-d8d222af8471` |
  | `CA_Scattering_Lab` | `2a2910fc-7e2b-5734-9c40-d1a082d917f1` |
  | `CA_Polar_Preview` | `caa96e48-6c1b-5a10-bf2f-b8fcb160f04b` |
  | `CA_Lab_Material` | `94143bb0-1ea1-5d3c-9bc9-edbeb8bfeea7` |
  | `Param_ScatteringSample` | `63b9bad1-3835-5f3c-bb45-c2e3b64b96ec` |
  | `Param_ScatteringResult` | `6f521d78-33a5-5839-b636-49a3528a1470` |
- **Units:** Rhino geometry is taken in metres, like the other components, with no unit conversion.
- **C# tests run outside Rhino**, with RhinoCommon's managed assemblies only. No test may construct a Rhino `Mesh`, `Brep` or `Plane`, because their constructors call native code. Code that needs them stays a few lines inside the component. Everything testable is a pure helper on `MeshData`, `Point3d`, arrays and messages.
- **Defaults:** `method` `"geometric"`, `preset` `"standard"`, band range ISO (100–5000 Hz). The method's options are `n_rays`, `bin_deg`, `max_bounces` and `seed` for geometric; `shadowing` and `workers` for kirchhoff.

## Review Focus

- **Kirchhoff and BEM must not start on every input change**, because they take minutes to hours. Only the Run button starts them; geometric runs on change (Task 6).
- **A worker without the lab packages:** an older install without trimesh or embreex must give the user one readable error that names `method="kirchhoff"` and the installer, not a traceback (Task 3, Task 6).
- **A Brep, surface or SubD from Rhino instead of a mesh** is meshed. Anything else gets the input-named warning the other components give (Task 5). The code is Rhino-only, so it is pinned by Task 9's manual check, not a unit test.
- **Old saved results:** an internalised result from before d existed has no d arrays. The Lab outputs an empty `diffusion` and the Goo prints without it (Task 4, Task 6).
- **Cancel during a run:** cancelling a geometric run or a Kirchhoff run stops it at the next source, and the component shows "cancelled" (Task 2 for the worker side; the C# side is `RunInBackground`, unchanged).

---

### Task 1: Messages, conversions and fixtures

**Files:**
- Modify: `src/compas_acoustics/pb/protobuf_defs/compas_acoustics/pb/generated/acoustics.proto`, `src/compas_acoustics/pb/conversions.py`, `tests/pb_fixtures/generate.py`
- Regenerate: `src/compas_acoustics/pb/generated/*`, `csharp/CompasAcoustics.Grasshopper/Generated/Acoustics.cs`, `Schema.cs`, `csharp/CompasAcoustics.Grasshopper.Tests/Fixtures/scattering_sample.b64`, `scattering_result.b64`
- Test: `tests/test_pb.py`

**Interfaces:**
- Produces, in the proto, appended after the existing messages:
  ```proto
  // An n-dimensional array: values in C order, and its shape.
  message ArrayData { repeated double values = 1; repeated int64 shape = 2; }

  message ScatteringSampleData {
    string guid = 1; string name = 2;
    compas_pb.data.MeshData sample = 3; compas_pb.data.MeshData reference = 4;
    string mode = 5; double diameter = 6;
    repeated double period = 7;           // empty: no period (as_drawn)
    double structural_depth = 8; optional double feature_size = 9;
    string symmetry = 10; double baseplate = 11; repeated string warnings = 12; string reference_kind = 13;
  }

  message ScatteringResultData {
    string guid = 1; string name = 2; string method = 3;
    repeated int32 bands = 4; repeated double s_bands = 5; repeated int32 octave_bands = 6; repeated double s_octaves = 7;
    repeated int32 filled_octaves = 8; ArrayData s_angle = 9;
    repeated double source_colatitude = 10; repeated double source_azimuth = 11; repeated double source_weights = 12;
    ArrayData receiver_directions = 13; ArrayData polar_sample_db = 14; ArrayData polar_reference_db = 15;
    repeated int32 polar_source_indices = 16; repeated int32 unconverged_octaves = 17; repeated string warnings = 18;
    compas_pb.data.AnyData generation = 19;
    ArrayData d_angle = 20; ArrayData d_bands = 21; ArrayData d_octaves = 22;   // unset: None (results made before d)
    reserved 30 to 39;   // V3: s_d and the SOFA path
  }
  ```
- Produces: `scattering_sample_to_pb` / `scattering_sample_from_pb` and `scattering_result_to_pb` / `scattering_result_from_pb`, registered with `@pb_serializer` / `@pb_deserializer` like `impulse_response_to_pb`. Arrays go through `ArrayData`, and `generation` goes through `_serializer_any(plain(...))`.

- [ ] **Step 1: Write the failing tests** in `tests/test_pb.py`, using its `roundtrip` and `arm` helpers.
  - `test_a_scattering_sample_round_trips_as_a_message`: `ScatteringSample.disc(sine_tile(0.08, 0.02), 0.4)`. Check that `arm(...)` names `ScatteringSampleData`, then that mode, diameter, period, structural_depth, symmetry, baseplate, warnings, reference_kind and guid are equal, and that both meshes have the same vertex and face counts.
  - `test_an_as_drawn_sample_with_a_proxy_keeps_its_reference_kind`: `period is None` and `reference_kind == "proxy"`.
  - `test_a_scattering_result_round_trips_every_field`: a geometric result of the sawtooth disc at `preset="survey"`, `n_rays=500`. Check that `s_angle`, `d_angle`, `receiver_directions` and the polar arrays are `np.allclose` (polar levels to 1e-3 dB, since they are float32), that the lists and `generation` are equal, and `scattering_coefficients` too.
  - `test_a_result_without_d_round_trips_with_none`: build one with `d_angle=None` and check `d_angle is None`.

  Import `sine_tile` and `sawtooth_tile` with `from lab_shapes import ...`: pytest's `pythonpath` already holds `tests/lab`.
- [ ] **Step 2: Run `pytest tests/test_pb.py -q -k scattering`; expect FAIL (no converter).**
- [ ] **Step 3: Add the messages, run `invoke generate`, write the converters, and add `scattering_sample()` and `scattering_result()` (guids 50 and 51, geometric, `n_rays=500`, `preset="survey"`) to `tests/pb_fixtures/generate.py`. Then run it.**
- [ ] **Step 4: Run `pytest tests/test_pb.py -q`, then `dotnet test csharp/CompasAcoustics.Grasshopper.Tests --nologo` (the regenerated C# must compile); expect PASS. Commit** `Lab: ScatteringSample and ScatteringResult travel as compas_pb messages`.

### Task 2: `lab_material`, and progress and cancel for the geometric method

**Files:**
- Create: `src/compas_acoustics/lab/material.py`
- Modify: `src/compas_acoustics/lab/__init__.py`, `src/compas_acoustics/lab/geometric.py`, `docs/superpowers/plans/2026-10-24-coefficient-lab-v1-b-absorption.md` (one note line, below)
- Test: `tests/lab/test_material.py`, `tests/lab/test_geometric.py`

**Interfaces:**
- Produces: `lab_material(name: str, absorption, scattering=None, category=None, description=None) -> AcousticMaterial`.
  - `absorption` is 1 or 7 numbers, or anything with `absorption_coefficients`: an `AcousticMaterial` now, B's `AbsorptionResult` later.
  - `scattering` is a `ScatteringResult` or None, read through `.scattering_coefficients`, `.method`, `.generation["preset"]` and `.generation.get("heuristic")`.
  - The rules are B plan Task 5's, verbatim:
    - clipping to [0, 1] warns `f"{kind} coefficient {value:.2f} in the {band} Hz octave clipped to {limit:g}"`;
    - a NaN raises `ValueError(f"no {kind} coefficient in the {band} Hz octave")`;
    - absorption above 0.5 with a scattering result warns `"ISO 17497-1 6.3.4: the sample absorbs more than 0.5, the scattering coefficient was computed for a rigid sample"`.
  - `data_origin` is `f"compas_acoustics.lab: alpha_s given; s by {method} (preset {preset}{', heuristic' if heuristic else ''})"`, or `"compas_acoustics.lab: alpha_s given; no scattering"`.
- Produces: `geometric_response` calls `jobs.progress("geometric", i, n_sources)` before each source, importing `from compas_acoustics.worker import jobs` inside the function as `kirchhoff.py` does, so a worker cancel stops it.

- [ ] **Step 1: Write the failing tests.**
  - `test_material_from_numbers_and_a_result`: 7 absorption values and a geometric sawtooth result. Check that α is equal, `len(scattering_coefficients) == 7`, and `data_origin` contains `"s by geometric (preset survey, heuristic)"`.
  - `test_one_absorption_value_fills_every_octave`
  - `test_absorption_from_a_material`: `AcousticMaterial("m", [0.3] * 7)`.
  - `test_without_scattering_the_material_has_none`
  - `test_out_of_range_values_are_clipped_with_a_warning`: use B's exact case, `[0.2, 0.2, 1.08, 0.2, 0.2, -0.02, 0.2]`.
  - `test_a_nan_band_is_refused_naming_the_octave`
  - `test_a_strongly_absorbing_sample_warns_about_the_rigid_scattering_assumption`
  - In `test_geometric.py`, `test_a_cancelled_job_stops_the_geometric_run`: `jobs.start("c1")`, `jobs.cancel("c1")`, then `pytest.raises(jobs.Cancelled)` around `geometric_response(...)`, and `jobs.finish("c1")` in a `finally`.
- [ ] **Step 2: Run them; expect FAIL.**
- [ ] **Step 3: Implement both, and export `lab_material` from `lab/__init__.py`.** Add this line under B plan Task 5's heading: `> Note (2026-10-24): lab/material.py exists (C plan Task 2) with these rules for numbers and materials; this task adds the AbsorptionResult branch and its data_origin, and its first test now needs method="kirchhoff" passed explicitly (the default is geometric).`
- [ ] **Step 4: Run `pytest tests/lab -q -m "not slow"`; expect PASS. Commit** `Lab: lab_material from numbers or a material, and the geometric method reports progress and stops on cancel`.

### Task 3: Worker API and the worker's lab packages

**Files:**
- Modify: `src/compas_acoustics/worker/api.py`, `src/compas_acoustics/worker/install.py`
- Test: `tests/test_worker_api.py`, `tests/test_worker.py`

**Interfaces:**
- `make_scattering_sample(mesh, mode="as_drawn", diameter=None, reference=None, baseplate=0.01, symmetry="none", period=None, name=None) -> ScatteringSample`:
  - `"disc"` tiles `mesh` and needs `diameter`, refusing without one: `ValueError("a disc sample needs a diameter")`. `period` is 2 numbers or None.
  - `"as_drawn"` takes `reference` as the proxy mesh.
  - Any other mode raises `ValueError("mode is 'as_drawn' or 'disc', not ...")`.
  - `name` sets `sample.name`.
- `run_scattering_lab(sample, method="geometric", preset=None, band_range="iso", settings=None) -> dict`:
  - Returns `{"result": ScatteringResult, "scattering_coefficients": [7 floats], "octave_bands": [ints], "warnings": [str]}`.
  - `band_range` is `"iso"` (`ISO_BAND_RANGE`) or `"extended"` (`EXTENDED_BAND_RANGE`), else `ValueError`.
  - `settings` is the method's options, passed on as `**settings`.
  - Warnings are recorded as `simulate` records them.
- `polar_balloon(result, preview=0, octave=1000, body="sample", radius=1.0, floor_db=-30.0) -> dict`:
  - Returns `{"mesh": compas Mesh, "levels": [dB per vertex], "source": [x, y, z], "label": str}`.
  - `preview` indexes `result.polar_source_indices`. `octave` must be in `result.octave_bands`. `body` is `"sample"` or `"reference"`. Anything else raises `ValueError` naming the valid values.
  - **Vertices:** each receiver direction × `radius * max(0, 1 - level / floor_db)`, with `level` the polar dB for that preview, octave and body.
  - **Faces:** `scipy.spatial.ConvexHull` of the unit directions plus the point (0, 0, −1), dropping the faces that use that point, oriented outward.
  - **`source`:** the unit vector of the preview source (from its colatitude and azimuth) × `1.2 * radius`.
  - **`label`:** `f"θ {colatitude:g}°, φ {azimuth:g}°, {octave} Hz, {body}"`.
- `lab_material(name, absorption, result=None, category=None) -> {"material": AcousticMaterial, "warnings": [str]}`, which wraps Task 2's function.
- `install_worker` installs `"compas_acoustics[lab] @ {wheel uri}"` instead of `"compas_acoustics @ ..."`.

- [ ] **Step 1: Write the failing tests** in `tests/test_worker_api.py`, through its `call()` wire helper, which round-trips compas_pb.
  - `test_make_scattering_sample_as_drawn_and_disc`: a flat 1 m quad as drawn; a sine tile as a disc with `diameter=0.4`.
  - `test_a_disc_without_a_diameter_is_refused`
  - `test_run_scattering_lab_returns_the_result_the_material_octaves_and_the_warnings`: geometric, `settings={"n_rays": 500}`, `preset="survey"`. Expect 7 `scattering_coefficients`, a `"heuristic"` warning, and `octave_bands == [125, 250, 500, 1000, 2000, 4000]`.
  - `test_an_unknown_band_range_is_refused`
  - `test_without_trimesh_the_lab_names_kirchhoff`: monkeypatch `sys.modules["trimesh"] = None` and check that the `ImportError` contains `method="kirchhoff"`.
  - `test_polar_balloon_of_a_flat_disc_points_its_peak_to_the_specular_direction`: geometric result, preview 1 (a 30° source). The vertex of the largest radius is within one 10° cell of the mirror direction, and every face index is below the vertex count.
  - `test_polar_balloon_refuses_an_octave_that_was_not_simulated`
  - `test_lab_material_from_the_lab_result`

  In `tests/test_worker.py`, extend the install test: the second uv call's last argument starts with `"compas_acoustics[lab] @ "`.
- [ ] **Step 2: Run them; expect FAIL.**
- [ ] **Step 3: Implement the four functions**, each with a numpy docstring, its lab imports inside the function as the module does, and its `__all__` entry. Then the install change.
- [ ] **Step 4: Run `pytest tests/test_worker_api.py tests/test_worker.py -q`; expect PASS. Commit** `Worker: the scattering lab's functions, and the worker installs the lab extra`.

### Task 4: Goo and parameters

**Files:**
- Create: `csharp/CompasAcoustics.Grasshopper/Data/LabGoo.cs`
- Modify: `csharp/CompasAcoustics.Grasshopper/Params/Params.cs`, `csharp/CompasAcoustics.Grasshopper/Preview/PreviewGeometry.cs` (make its private `Mesh(MeshData)` internal as `MeshOf`, unchanged in body)
- Test: `csharp/CompasAcoustics.Grasshopper.Tests/LabGooTests.cs`

**Interfaces:**
- **`ScatteringSampleGoo : MessageGoo<ScatteringSampleData>, IOwnPreview`:**
  - `TypeName "Scattering Sample"`.
  - `ToString`: `"Scattering Sample '<name>', <mode>, d 0.40 m, h 0.020 m, <reference_kind> reference"`, invariant culture.
  - It draws the sample mesh's edges in the wire colour and the reference's outline edges (edges used by one face of its top) in `PreviewDrawing.ObstacleEdge`.
  - `ClippingBox` is the sample's box.
- **`ScatteringResultGoo : MessageGoo<ScatteringResultData>`:**
  - `TypeName "Scattering Result"`.
  - `ToString`: `"Scattering Result '<name>', <method>, s 0.00–0.44 (125–4000 Hz)"`, plus `", d 0.00–0.01"` when `d_octaves` is set.
  - `Diffusion`: the d octaves, or an empty list when the field is unset.
- **`Param_ScatteringSample : OwnPreviewParam<ScatteringSampleGoo>`** and **`Param_ScatteringResult : CompasAcousticsParam<ScatteringResultGoo>`**, with the GUIDs above.

- [ ] **Step 1: Write the failing tests** from the Task 1 fixtures (`InteropTests.Fixture`).
  - `SampleGoo_ToString_NamesModeDiameterDepthAndReference`
  - `SampleGoo_HasAPreviewBoxAroundTheSample`
  - `ResultGoo_ToString_ShowsTheMethodAndTheRangeOfS`
  - `ResultGoo_WithoutD_HasNoDiffusion`: clear `DOctaves` on a clone of the fixture, then `Diffusion` is empty and `ToString` has no `", d "`.
  - `BothGoo_SurviveAWriteAndRead`, through `GH_LooseChunk` as `AudioPlayerComponentTests` does.
- [ ] **Step 2: Run `dotnet test ... --filter LabGooTests`; expect FAIL (types missing).**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the whole C# suite; expect PASS. Commit** `Grasshopper: Goo and parameters of the scattering sample and result`.

### Task 5: Scattering Sample component

**Files:**
- Create: `csharp/CompasAcoustics.Grasshopper/Components/LabComponents.cs` (Tasks 5–8 add to it), `csharp/CompasAcoustics.Grasshopper/Components/LabRequests.cs`, `Components/Specs/CA_Scattering_Sample/metadata.json` + `icon.png`
- Modify: `tools/icons.py` (the `"08 Lab"` accent and a `sample` glyph: a relief profile over a baseplate line), `csharp/CompasAcoustics.Grasshopper.Tests/SpecCoverageTests.cs`
- Test: `csharp/CompasAcoustics.Grasshopper.Tests/LabComponentTests.cs`, `tests/test_icons.py` (unchanged; it must pass)

**Interfaces:**
- **Spec inputs:**
  - `geometry` (geometrybase, item);
  - `mode` (str, default `"as_drawn"`; a body dropdown `as_drawn` / `disc`);
  - `diameter` (float, optional, disc only);
  - `reference` (geometrybase, optional, the proxy, as_drawn only);
  - `baseplate` (float, default 0.01);
  - `symmetry` (str, default `"none"`; a body dropdown `none` / `rotational`);
  - `name` (str, optional).
- **Output:** `sample` (`Param_ScatteringSample`).
- **`LabRequests.MeshOf(object? geometry) -> MeshData?`:**
  - a Mesh as is;
  - a Brep, Surface or SubD meshed with `MeshingParameters.QualityRenderMesh` and joined;
  - null for anything else.
- **`LabRequests.Sample(MeshData mesh, string mode, double? diameter, MeshData? reference, double? baseplate, string? symmetry, string? name) -> (object?[] args, Dictionary<string, object?> kwargs, string? problem)`:**
  - `problem` is `"A disc sample needs 'diameter'."` for disc without a diameter, `"'reference' is used in as_drawn mode only."` for a reference in disc mode, otherwise null.
  - Leaving an input out leaves out its kwarg.
- **The component:** calls `make_scattering_sample` synchronously, shows each `sample.Warnings` entry as a Warning, and sets `Status($"{mode}, d {diameter:0.00} m")`.

- [ ] **Step 1: Write the failing tests:**
  - `Sample_DiscWithoutDiameter_IsAProblem`;
  - `Sample_ReferenceInDiscMode_IsAProblem`;
  - `Sample_LeavesUnsetInputsOut`;
  - `Sample_PassesTheMeshAndTheModeAsArguments`;
  - `SampleSpec_HasItsPortsAndDefaults` (`ComponentSpec.Load("CA_Scattering_Sample")`).

  `MeshOf` takes Rhino types and is not unit-tested; keep it to its three cases.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement**: the spec, `python tools/icons.py`, the component (`ComponentGuid` as listed), and the `SpecCoverageTests` entry `["CA_Scattering_Sample"] = "ScatteringSampleComponent"`.
- [ ] **Step 4: Run the C# suite and `pytest tests/test_icons.py -q`; expect PASS. Commit** `Grasshopper: the Scattering Sample component`.

### Task 6: Scattering Lab component

**Files:**
- Modify: `Components/LabComponents.cs`, `Components/LabRequests.cs`, `tools/icons.py` (a `lab` glyph: a hemisphere of dots over a relief), `SpecCoverageTests.cs`
- Create: `Components/Specs/CA_Scattering_Lab/metadata.json` + `icon.png`, `Components/LabResults.cs`
- Test: `LabComponentTests.cs`

**Interfaces:**
- **Spec inputs:**
  - `sample` (ghdoc);
  - `method` (str, default `"geometric"`; a body dropdown `geometric` / `kirchhoff` / `bem`);
  - `preset` (str, optional; a dropdown of blank meaning the method's default, `survey`, `standard` and `fine`);
  - `band_range` (str, default `"iso"`; a dropdown `iso` / `extended`);
  - `n_rays` (int, optional);
  - `cache_dir` (str, optional, bem only: NumCalc's project folder, so an interrupted run resumes);
  - `run` (a body button "Run", declared as Auralise declares its `save` button: a spec input plus a `ButtonControl`).
- **Spec outputs:** `result` (`Param_ScatteringResult`), `scattering` (float list, 7 octaves 125–8000 Hz), `diffusion` (float list, the simulated octaves, empty without d), `octaves` (int list).
- **`LabRequests.StartsWithoutRun(string method) -> bool`:** true only for `"geometric"`.
- **`LabRequests.Lab(ScatteringSampleData sample, string method, string? preset, string bandRange, int? nRays, string? cacheDir) -> (object?[] args, Dictionary<string, object?> kwargs, string? problem)`:**
  - `n_rays` goes into `settings` for geometric only. For other methods it is the problem `"'n_rays' is used by the geometric method only."`.
  - `cache_dir` goes into `settings` for bem only. For other methods it is the problem `"'cache_dir' is used by bem only."`.
  - Args are `[sample, method]`; kwargs hold `preset` (if set), `band_range` and `settings` (if any).
- **`LabResults.Read(IDictionary<string, object?> result) -> (ScatteringResultData result, double[] scattering, int[] octaves, double[] diffusion)`**, and `LabResults.Summary(ScatteringResultData) -> string`, for example `"geometric, s 0.00–0.44"`.
- **The component:** `RunsInBackground => true`. It calls `RunInBackground(access, new Background("run_scattering_lab", args, kwargs, sample), deliver, start: StartsWithoutRun(method) || Pressed("run"))`.
  - Before a first kirchhoff or bem run it remarks `"Press Run: kirchhoff and bem take minutes to hours."`.
  - `deliver` shows the warnings with `ShowWarnings`, sets the four outputs, and sets `Status(Summary)`.

- [ ] **Step 1: Write the failing tests:**
  - `OnlyGeometric_StartsWithoutRun`;
  - `Lab_PutsNRaysInTheGeometricSettings`;
  - `Lab_NRaysWithKirchhoff_IsAProblem`;
  - `Lab_CacheDirGoesToTheBemSettings`;
  - `Lab_LeavesAnUnsetPresetOut`;
  - `Read_TakesTheResultAndItsOctaves`, from the fixture result wrapped in a dict as the worker answers;
  - `Read_WithoutD_GivesAnEmptyDiffusion`;
  - `Summary_ShowsMethodAndRange`;
  - `LabSpec_HasARunButtonAndGeometricAsDefault`.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement**: the spec, the icon, the component, and the `SpecCoverageTests` entry `"ScatteringLabComponent"`.
- [ ] **Step 4: Run the C# suite; expect PASS. Commit** `Grasshopper: the Scattering Lab component, geometric on change and the wave methods on Run`.

### Task 7: Polar Preview component

**Files:**
- Modify: `Components/LabComponents.cs`, `Components/LabRequests.cs`, `Components/LabResults.cs` (`FaceLevels`), `Display/EnergyMapColours.cs` (move `VisualiseEnergyMapComponent.Coloured` here as `internal static Mesh Coloured(...)`, with an `offset` flag, false for balloons, and keep the energy map's call identical), `tools/icons.py` (a `balloon` glyph), `SpecCoverageTests.cs`
- Create: `Components/Specs/CA_Polar_Preview/metadata.json` + `icon.png`
- Test: `LabComponentTests.cs`

**Interfaces:**
- **Spec inputs:**
  - `result` (ghdoc);
  - `preview` (int; a dropdown filled from the result's preview sources, labelled `"θ 30°, φ 0°"`);
  - `octave` (int; a dropdown filled from `octave_bands`);
  - `body` (str, default `"sample"`; dropdown `sample` / `reference`);
  - `radius` (float, default 1.0);
  - `floor_db` (float, default −30);
  - `centre` (point, optional, default the origin).
- **Spec outputs:** `mesh` (mesh), `levels` (float list), `source` (line, from `centre` along the source direction).
- **The component:** a `DisplayComponent` with its `PreviewSwitch`. It calls `polar_balloon` synchronously and colours each face by its vertices' mean level with `EnergyMapColours.Of(level, floor_db, 0)`. It moves the mesh by `centre` and sets `Status(label)`.
- **`LabRequests.PreviewLabels(ScatteringResultData r) -> string[]`.**
- **`LabRequests.Balloon(ScatteringResultData result, int preview, int octave, string body, double radius, double floorDb) -> (object?[] args, Dictionary<string, object?> kwargs, string? problem)`:**
  - An octave the result lacks is the problem `"'octave' must be one of 125, 250, ... Hz"`, listing the result's octaves.
  - A preview out of range is the problem `"'preview' must be 0 to N"`.
- **`LabResults.FaceLevels(double[] levels, int[][] faces) -> double[]`:** the mean of each face's vertex levels, which the colours are made from.

- [ ] **Step 1: Write the failing tests:**
  - `PreviewLabels_NameTheRingAndAzimuthOfEachPreviewSource`, from the fixture;
  - `Balloon_PassesPreviewOctaveBodyRadiusAndFloor`;
  - `Balloon_AnOctaveThatWasNotSimulated_IsAProblem`;
  - `FaceLevels_AreTheMeanOfTheirVertices`.

  `Coloured` builds a Rhino `Mesh` and stays untested, as it is today. The energy map's call must stay the same apart from the new argument.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement**: the spec, the icon, the component (`"PolarPreviewComponent"`), and the SpecCoverage entry.
- [ ] **Step 4: Run the C# suite; expect PASS. Commit** `Grasshopper: the Polar Preview component draws the reflected level as a coloured balloon`.

### Task 8: Lab Material component

**Files:**
- Modify: `Components/LabComponents.cs`, `Components/LabRequests.cs`, `tools/icons.py` (a glyph that reuses `material` with a dot hemisphere), `SpecCoverageTests.cs`
- Create: `Components/Specs/CA_Lab_Material/metadata.json` + `icon.png`
- Test: `LabComponentTests.cs`

**Interfaces:**
- **Spec inputs:**
  - `name` (str);
  - `absorption` (float list, optional: 1 or 7 values);
  - `material` (ghdoc, optional, the α source when `absorption` is empty);
  - `result` (ghdoc, optional);
  - `category` (a dropdown of `ConstructMaterialComponent.Categories`, default `"Other"`).
- **Spec output:** `material` (`Param_Material`).
- **`LabRequests.Material(string name, IList<double> absorption, AcousticMaterialData? material, ScatteringResultData? result, string? category) -> (object?[] args, Dictionary<string, object?> kwargs, string? problem)`:**
  - No absorption and no material is the problem `"Connect 'absorption' (1 or 7 values) or a 'material' to take it from."`.
  - An absorption list that is neither 1 nor 7 long is the problem `"'absorption' takes 1 or 7 values (125 Hz to 8 kHz)."`.
  - The description says the material goes on the proxy's faces, not on the detailed mesh.
- **The component:** calls `lab_material` synchronously, shows the warnings, and outputs a `MaterialGoo`.

- [ ] **Step 1: Write the failing tests:**
  - `Material_WithoutAbsorptionOrMaterial_IsAProblem`;
  - `Material_FiveValues_IsAProblem`;
  - `Material_TakesTheMaterialWhenAbsorptionIsEmpty`;
  - `LabMaterialSpec_DescribesTheProxy` (the description contains `"proxy"`).
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement**: the spec, the icon, the component (`"LabMaterialComponent"`), and the entry.
- [ ] **Step 4: Run the C# suite; expect PASS. Commit** `Grasshopper: the Lab Material component`.

### Task 9: End to end with a real worker, and docs

**Files:**
- Modify: `csharp/CompasAcoustics.Grasshopper.Tests/WorkerIntegrationTests.cs`, `CHANGELOG.md`, `docs/superpowers/plans/2026-10-24-coefficient-lab-c-grasshopper.md` (tick the steps)
- Test: `WorkerIntegrationTests.cs`

- [ ] **Step 1: Add `RealWorker_RunsTheScatteringLabChain`** with the same skip guard as its neighbours.
  - It runs `make_scattering_sample`, a flat 1 m quad, as drawn.
  - Then `run_scattering_lab` with `settings {n_rays: 500}` and `preset "survey"`, then `polar_balloon(result, 0, 1000)`, then `lab_material("panel", [0.1], result)`.
  - It asserts:
    - the result `Method == "geometric"`;
    - 7 scattering values, all 0 within 1e-9 (flat);
    - a balloon mesh with faces;
    - a material with 7 scattering coefficients.
- [ ] **Step 2: Run it against the local worker** the way the file's docstring says (its environment variable). It must pass, or skip without one. Report which, in the task report.
- [ ] **Step 3: Write the CHANGELOG entry** under Unreleased.
  - The four components in a new "08 Lab" subcategory.
  - `ScatteringSampleData` and `ScatteringResultData`.
  - The four worker functions.
  - The worker now installs `compas_acoustics[lab]`, so reinstall to use the lab. NumCalc for `bem` is a separate setup.
- [ ] **Step 4: Write the manual Rhino check into the task report.** The implementer cannot run Rhino; the user runs it.
  - A Brep box and a mesh into Scattering Sample both give a sample; a point warns about 'geometry'.
  - Scattering Lab runs on change with geometric, and only on Run with kirchhoff. Cancel during the Kirchhoff run shows "cancelled".
  - Polar Preview draws a balloon at `centre`.
  - Lab Material's output wires into Acoustic Object.
- [ ] **Step 5: Run the C# suite and `pytest tests -q -m "not slow"`; expect PASS. Commit** `Lab: the Grasshopper chain end to end, changelog`.

### Task 10: The check in Grasshopper through lamcp (the controller, after the user installs the rebuilt bundle)

This task is not given to an implementer subagent: it needs the lamcp tools and a Rhino with the plugin installed, and the user installs the bundle. The controller runs it after Task 9 and the final review, and reports what it saw.

**Before it starts**
- The user runs `invoke installer-bundle`, opens `dist/compas_acoustics_installer/compas_acoustics_installer.gh` in Grasshopper to install, restarts Rhino, and switches the LAMCP bridge Toggle on.
- The controller checks `bridge_health` is `ok`.
- The controller checks `search_components("Scattering")` returns the four new components by the GUIDs in Global Constraints. If not, stop and report: the install did not take.

**Rules**
- The canvas holds the LAMCP bridge script, its Toggle and the status Panel. Never move or delete them.
- Place new objects below y = 520, clear of the Panel and of the Sphere at (800, 480).
- `_lamcp_run_on_ui` does not exist in `run_python_script`. Marshal canvas edits with `Rhino.RhinoApp.InvokeOnUiThread` and finish with `ScheduleSolution`, as the team notes on lamcp say.
- Everything the controller places is removed again at the end, and the final `list_grasshopper_objects` shows the same four objects the canvas started with.

- [ ] **Step 1: Place and wire the chain.**
  - Place a Rhino mesh source: a Mesh component of the 1 m sawtooth, or a Python-made mesh object, wired to **Scattering Sample** (`mode` `as_drawn`).
  - Wire the sample to **Scattering Lab** (geometric, `preset` `survey`, `n_rays` 2000).
  - Wire the result to **Polar Preview** (`octave` 1000, `preview` 1) and to **Lab Material** (`absorption` one value 0.1, `name` "lamcp panel").
  - Wire Lab Material's output into **Acoustic Object**.
- [ ] **Step 2: Solve and read.**
  - `solve_grasshopper`, then `list_grasshopper_objects`: no component has an error. Report each warning.
  - `inspect_component_data` on the Lab's `scattering` and `octaves` outputs: 7 and 6 values.
  - On Polar Preview's `mesh`: a mesh with faces.
  - On Lab Material's output: a material with 7 scattering values.
  - `capture_grasshopper_canvas` and `capture_rhino_viewport`: the balloon is in the viewport.
- [ ] **Step 3: The Run rule.**
  - Set Scattering Lab to `kirchhoff`: the component shows "Press Run" and starts no job (`list_grasshopper_objects` shows no running state, the worker log shows no `run_scattering_lab`).
  - Press Run with a tiny sample (`diameter` 0.2, `preset` `survey`, `band_range` iso): it completes in minutes. Report the time.
  - Press Run again and cancel: the message reads "cancelled".
  - Set the method back to `geometric`.
- [ ] **Step 4: Failure messages.**
  - Disconnect `diameter` from a disc-mode sample: the component warns with the input's name.
  - Wire a point into Scattering Sample's `geometry`: the same.
- [ ] **Step 5: Clean up and report.** Remove everything placed, confirm the canvas is as it started, and write the results (errors, warnings, timings, captures) into the final message. A failing step is a finding for the user, not something to fix on the canvas.
