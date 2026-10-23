# Coefficient lab V1-A spike findings

Environment: macOS 26 (Darwin 25.6, arm64, 14 cores, 36 GB RAM, about 15 GB free during runs), Python 3.13, pyfar 0.8, `mesh2scattering` 1.0.5, `trimesh` 5.1.1 (plus `psutil`, `sofar`, `scipy` pulled in or already present). Tasks 7 and 9 read this note.

## 1. Install and NumCalc build

- `pip install "mesh2scattering>=1.0.5" trimesh` works. `mesh2scattering.__version__ == "1.0.5"`.
- `mesh2scattering.numcalc.build_or_fetch_numcalc()` runs `make` with the Xcode command-line tools (g++) on first call and **succeeds** (only `sprintf` deprecation warnings). It returns
  `<site-packages>/mesh2scattering/numcalc/bin/NumCalc`. Later calls just return the path.
- NumCalc is **single-threaded** (user time equals wall time). Parallelism comes only from `manage_numcalc` starting several instances.

## 2. API gotchas hit while writing the spike

- `SoundSource(...)` and `EvaluationGrid.from_spherical(...)` both **require `Coordinates.weights`** (else `ValueError: ... must contain weights`). Pass `weights=` to `pf.Coordinates`. Source weights are stored as `sources_weights`; receiver weights as `ReceiverWeights` in the SOFA.
- `SurfaceDescription(surface_type=SurfaceType.FLAT)` (enum is `SurfaceType`, from `mesh2scattering.input`).
- `manage_numcalc` with a **relative** `project_path` crashes (`FileNotFoundError: proj/proj/manage_numcalc_....txt`). Always pass an absolute path.
- `manage_numcalc` writes its log to `<project>/manage_numcalc_<date>.txt`.
- The default `speed_of_sound` of `write_scattering_project_numcalc` is 346.18 m/s (`density_of_medium` 1.1839); pass `speed_of_sound=` explicitly if another value is wanted.
- `trimesh.remesh.subdivide_to_size` on a `trimesh.creation.cylinder` gives 81 750 faces for d = 0.4 m, edge 1 cm (long sliver triangles from the cap fan are bisected over and over). A quick custom mesh (hex-lattice Delaunay disc, top + bottom + rim strip, welded) gave 5 932 faces for the same disc and is watertight with the right volume. Use a proper tiling, not the cylinder primitive, for the timing runs and in Task 5/7 fixtures.

## 3. Project layout written by `write_scattering_project_numcalc` (before running)

```
<project>/parameters.json
<project>/ObjectMeshes/Reference.stl
<project>/ObjectMeshes/Reference/{Nodes.txt,Elements.txt}
<project>/EvaluationGrids/<grid name>/{Nodes.txt,Elements.txt}
<project>/NumCalc/source_<k>/NC.inp        # k = 1..n_directions
```

### (e) One SoundSource, many directions

`SoundSource` takes a **multi-point `Coordinates`** (with weights) and `source_type=PLANE_WAVE`. The project then has one folder `NumCalc/source_<k>` per point, each with its own `NC.inp` (the frequency list is the same in each). One `SoundSource` with 97 or 864 points is the right call. A plane wave is written in `NC.inp` as `PLANE WAVES / 0 <x> <y> <z> 0.1 -1 0.0 -1` with the **propagation direction** given to `SoundSource` (e.g. `(0, 0, -1)`). In the SOFA the source position is `-direction * 99` (so `(0, 0, 99)`), i.e. the unit vector from the sample towards the source times 99. That matches the global-constraints convention (`û` = from sample to source = minus the propagation direction).
Each source folder is an independent NumCalc run (mesh is re-read and the matrix reassembled per source, no sharing).

## 4. (a) What NumCalc writes per frequency step

Per source folder `NumCalc/source_<k>/` after `manage_numcalc` (step numbers are 1-based, in the order of the `frequencies` array):

```
Memory.txt                     # from `NumCalc -estimate_ram`; lines "step freq est_GB"
NC.inp
NC.out                         # general input log (all steps)
NC<step>-<step>.out            # full NumCalc output of that step; ends with "End time: ..."
NC<step>-<step>_log.txt        # stdout of that step (timing summary), no completion marker needed
be.out/be.<step>/pBoundary     # pressure on the sample surface
be.out/be.<step>/vBoundary
be.out/be.<step>/pEvalGrid     # pressure at receivers: header "mesh2scattering 1.0.5", "1", "1 <n>", then "<node id> <re> <im>" per row
be.out/be.<step>/vEvalGrid
```

**Finished-step marker** (this is exactly what `numcalc._check_project` / `manage_numcalc` uses): `be.out/be.<step>/pEvalGrid` exists **and** `NC<step>-<step>.out` contains the string `End time:`. `pEvalGrid` alone is not sufficient (it exists while post-processing may still be running); `NC<step>-<step>.out` is only reliable once it contains `End time:`. For our own cache check use the `End time:` text plus `pEvalGrid`.
Also: after `write_pressure`, `<project>/report/report_source_<k>.csv` has per-step columns `frequency step, frequency in Hz, NC input file, Input check passed, Converged, Num. iterations, relative error, Comp. time total, Comp. time assembling, Comp. time solving, Comp. time post-proc.` (e.g. `1, 500.0, NC1-1.out,1, 1, 26, 2.4626e-10, 6, 6, 0, 0`). A `report_issues.txt` plus a `UserWarning` appear if something did not converge. The `Converged` flag in this csv is the one to check for non-converged steps.

## 5. (b) `output.write_pressure`

- Call: `mesh2scattering.output.write_pressure(project_path)` (prints "Writing the project report ..."; writes the report csv, then the SOFA).
- **SOFA location: `os.path.join(project_path, '..', f'{project_title}_{grid_name}.pressure.sofa')`**, that is, a **sibling of the project folder**, not inside it (here `.../spike/flat_grid.pressure.sofa` for `project_title="flat"`, grid name `"grid"`). One file per evaluation grid. The path contains `..` literally; normalise it. Two projects with the same title in the same parent folder overwrite each other's SOFA, so use one parent directory per project or a unique title.
- Read with `sofar`, then pyfar: `s = sofar.read_sofa(path)` (a path string is **not** accepted by `pf.io.convert_sofa`), `data, src, rec = pf.io.convert_sofa(s)[:3]` (it returns `FrequencyData, source Coordinates, receiver Coordinates`; `[0]` is the FrequencyData).
- `data.freq.shape == (n_sources, n_receivers, n_frequencies)` (here `(3, 200, 2)`), complex128; `data.cshape == (n_sources, n_receivers)`; `data.frequencies == [500., 1000.]` in the order given (no sorting). The pressure is the **scattered** field for a unit-amplitude plane wave (`|p|` ~ 1e-3 at 10 m for the small flat disc), as the SOFA/docstring says "scattered sound pressure".
- `src.cartesian` is `(n_sources, 3)` in input order (value `-direction * 99`); `src.weights` is `None` in pyfar, the weights are in `s.SourceWeights` (shape `(n_sources,)`, here `[1/3, 1/3, 1/3]`). `rec.cartesian` is `(n_receivers, 3)`; receiver weights in `s.ReceiverWeights` (shape `(n_receivers,)`, sums to 1). Other custom entries: `SampleDiameter`, `SpeedOfSound`, `DensityOfMedium`, `Lbyl`, `OriginalFrequencies`, `RealScaleFrequencies`, symmetry fields.

## 6. (c) Rerunning `manage_numcalc` resumes

Test: 6 frequencies (500...1500 Hz, 200 receivers, 1 source), `max_instances=1`, process killed after 20 s (also `pkill NumCalc`). State after kill: steps 1, 2, 3, 6 had `End time:`; step 5 had `NC5-5.out` without `End time:` and no `be.5`; step 4 had nothing. (Note the default `starting_order="alternate"` runs steps in a 1, N, 2, N-1 ... pattern, so unfinished steps are scattered.)
Rerun printed `Running 2/6 unfinished frequency steps in the project`, started only steps 5 and 4, and left the finished steps' `pEvalGrid` and `NC1-1.out` untouched (identical modification times). So **yes, it skips finished steps and restarts half-finished ones**.
Caveat for Task 7: `manage_numcalc` checks only the markers above. It does **not** compare geometry, frequencies or sources, so the cache key (mesh hash, frequencies, sources, grid) must be ours; on a mismatch the project folder must be rewritten from scratch (there is also `numcalc.remove_outputs`).

## 7. (d) `EvaluationGrid.from_spherical` keeps the point order

`EvaluationGrid.from_spherical(coords, name)` stores `coords` untouched (it only builds the triangle faces through `scipy.spatial.ConvexHull(points)`); `grid.coordinates.cartesian` equals the input exactly (`np.allclose`, max abs diff 0.0). The points reach the SOFA in the same order: `rec.cartesian` from the SOFA vs the input array differs by 0.0, so `data.freq[:, i, :]` is receiver `i` of the input. Convex hull requires the points not to be coplanar (a hemisphere with z > 0 is fine); `Coordinates` with `weights` is mandatory. Evaluation grid triangles are added to the BEM model as extra elements: 3000 points make about 6000 additional triangles.

## 8. Timings at the band tops (d = 0.8 m disc, 10 mm baseplate, 1 source, 3000 receivers, one NumCalc run per frequency, single instance, M-series Mac)

Mesh: top disc + rim + bottom, hex-lattice Delaunay, `edge = 343/(f*6)`. "Total elements" is the first number of the `Main Parameters I` line in `NC.inp` (sample plus receiver-grid triangles). Wall time and peak RAM from `/usr/bin/time -l NumCalc -istart 1 -iend 1` (maximum resident set size); the `estimate_ram` value comes from `NumCalc -estimate_ram` (`Memory.txt`).

| frequency | edge | sample faces | total elements (NC.inp) | nodes | NumCalc estimate | peak RAM measured | wall | CGS iterations | time split (assemble / solve / post) |
|---|---|---|---|---|---|---|---|---|---|
| 5400 Hz | 10.6 mm | 21 004 | 27 000 | 13 504 | 1.02 GB | 1.06 GB (1 058 488 320 B) | 28.8 s | 23 | 13 / 5 / 11 s |
| 8600 Hz | 6.6 mm | 53 044 | 59 040 | 29 524 | 3.39 GB | 3.52 GB (3 521 855 488 B) | 74.7 s | 26 | 20 / 24 / 29 s |

Notes:
- `estimate_ram` is accurate to about 4 percent, so it can be used as the RAM figure for the pre-flight and for `manage_numcalc`'s own scheduling.
- Elements scale as f^2 (21 004 to 53 044 for the 1.6 times higher frequency), wall time roughly as f^2 over this range, RAM roughly f^2 to f^2.1. The 8600 Hz point needs 3.4-3.5 GB per instance.
- Cheap points: the small flat disc (d = 0.4 m, edge 1 cm, 6 328 total elements) took 6 s at 500 Hz and 3 s at 1000 Hz, 0.16 and 0.12 GB. All 3 steps times 2 frequencies finished in 15 s wall with 3 parallel instances.
- Both runs converged (relative error 2e-10 and 5e-11) in about 25 iterations.

### Extrapolation for the plan (rough; not run in full)

- Default range (100-5000 Hz, 18 bands x 3 = 54 frequencies, top 5 040 Hz at `fc*2**(1/9)`): per source and frequency the cost runs from about 1 s (100 Hz) to about 25 s (5 kHz); the sum over the 54 frequencies is roughly 54 x 8 s average = 7 minutes per source for d = 0.8 m. With 97 sources (bem preset "standard") that is about 11 core-hours, so about 1 h on 14 cores (RAM per instance at most about 1 GB, so no RAM limit) provided the mesh follows the 6 elements per wavelength rule at each frequency.
- Extended range (adds 6300 and 8000 Hz x 3 = 6 frequencies up to 8 980 Hz, which is above the 8600 Hz measured): about 60-100 s and 3.5-5 GB per run. 97 sources x 6 = 582 runs, roughly 12 core-hours; RAM-limited to about 3-4 parallel instances on 15 GB free, so about 3-4 h wall.
- Suggested default `max_elements` for Task 7: set from the RAM estimate. 60 000 total elements (about 8.6 kHz at d = 0.8 m) costs about 3.5 GB; 27 000 (5.4 kHz) about 1 GB. A cap near 60 000 elements per frequency step covers the default range and the extension on d <= 0.8 m. Sizes above about 100 000 elements (more than about 10 GB by the f^2 scaling) should stop with a clear error and an estimate. The "overnight" wording applies to the extended range or to many directions on a small machine, not to the default range with the "standard" preset on this 14-core machine.
