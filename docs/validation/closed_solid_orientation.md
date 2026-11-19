# Closed solids in open scenes: orientation from the seeds

Validation record of the fix on branch `fix/closed-solid-orientation`, based on `50f4685` (the `engine-integration`
tip the compas_soundscape app runs). Not a docs page (excluded in `mkdocs.yml`).

## The bug

`prepare_mesh` orients every face so its normal points away from the air. The walk (`visibility.walk_from_seeds`)
counts hits on each side of each face from the sources and receivers (seeds); a seed whose primary rays mostly escape
(>= `seed_outside_escape_fraction`, 0.5) is "outside" and its hits do not vote. `orientation.decide_orientation` then
oriented a closed welded component by a majority of its hits and, with no hit at all, kept its normals pointing *out of
the solid* ("never reached by air: keep normals pointing out of the solid"), i.e. treated it as a room.

In a free field every seed is outside, so no face gets a hit. BRAS RS2_2m_absorbing (an MDF plate with a 20 mm
RockFon slab, seeds at [3.46, +-2.0, 1.02]): the slab, a closed component, kept its normals pointing out into the air
(front face at x = 0 with normal +x towards the source). PFFDTD puts the material behind the normal, so the absorber
material landed inside the slab and the slab acted rigid: the 0 deg and 45 deg absorber responses were bit-identical.
The RS1 absorber blocks, the RS6/RS7 MDF elements and the RS1 floor slabs had the same fault; the open floor planes of
RS1/RS5/RS7 kept their modelled winding (normal +z, into the air above them).

## The fix

A. **Closed orientable welded components** (rule 1): the seeds decide, geometrically. For each component,
`orientation.points_inside_closed` casts 9 rays from each seed in fixed directions (a Fibonacci set turned off the
axes and diagonals, so no ray runs along an axis-aligned plane or edge of an architectural model) and counts their
crossings with that component's faces (`raycast.ray_crossings`). A ray passing within 1e-7 (barycentric) of an edge or
vertex, running in a face's plane, or starting on a face abstains, since there a crossing may count twice or not at
all; the others vote by parity, a strict majority deciding (a tie, or no valid ray, is "uncertain"). A seed outside
the component's bounding box is outside without casting.
- any seed inside -> the component holds the air -> normals out of it;
- no seed inside (all certain) -> a solid -> normals into it, whether the walk reached it or not;
- an uncertain seed and none inside -> the old majority of hits, a solid when there is none;
- no seeds at all, or zero signed volume -> the old rule (majority, else normals out).

The walk's hits only cross-check. A conflict is recorded (`OrientationResult.conflicts`,
`stats["orientation_conflicts"]`, and an `analyze` warning `orientation_conflict` with the component's faces) when the
side the decision makes solid is reached clearly: at least `two_sided_min_hits` hits there and either at least
`two_sided_min_fraction` of the component's hits or over at least half of its area (a seed's walk trapped in a small
solid piles up its hits inside it, so the share of hits alone hides the air outside), or when a seed the parity test
puts inside has its rays mostly escape (status outside): a closed surface cannot let that happen. Such a seed still
counts as inside (the parity test is exact; the escape threshold is a setting).

B. **Loose-face enclosing objects** (rule 2) with no hit: they hold the air only when a seed is inside them (same
parity test), instead of always being solid by default. With hits, unchanged.

C. **All-outdoor scenes**: when every seed is outside, the exterior is the air the simulation happens in. The walk then
runs from all of them after the primary rays (`AirVisibility.outdoor`, `stats["outdoor"]`), their hits voting, so
free-standing sheets hit on both sides become two-sided. The random generator is consumed in the same order as before
for every scene that is not all-outdoor (the extra walks run after the loop), so those scenes are bit-identical. The
leak share, the leaks and the mean free path stay those of the seeds inside (none in such a scene): rays escaping into a
free field are no leaks of a room. Mixed scenes (a seed in a building, one in the street) keep today's behaviour:
outside seeds do not vote.

`prepared_wall_sides` (closed component never hit from its other side -> one-sided solid), `room.geometry` (air side
and other side per wall) and `_classify` read `air_hits` / `other_hits` as "hits on the side the decision puts the air
on / the other side", which they remain; for a closed solid reached only from outside, `other_hits` stays 0.

Files: `geometry/orientation.py`, `geometry/raycast.py`, `geometry/visibility.py`, `geometry/prepare.py`,
`geometry/report.py`; tests `tests/test_closed_solid_orientation.py`; tool `tools/prepared_geometry.py`.

## Tests

`tests/test_closed_solid_orientation.py`, 20 tests (all pass):

| scene | expectation |
| --- | --- |
| parity test on a cube (points level with edges, vertices, diagonals) and an L-prism (notch) | exact, certain; a point on an edge is uncertain |
| (a) closed 20 mm slab, 2 x 1.33 m, seeds 3.5 m away, modelled outward and inward | outdoor walk; normals into the slab, front face -x; front hit from the air side only; no conflict |
| (b) the slab against a loose-face plate (RS2 layout) | slab and plate normals into themselves |
| (c) closed room, seed inside, outward and inward | normals out of the room, no conflict |
| (d) closed column in a closed room, column modelled both ways | column into, room out |
| (e) outdoor panel hit from both sides / from one side (either side) | two-sided / one-sided facing away from the seed |
| (e) outdoor walk statistics | no re-emissions, chords or leaks counted, `leak_fraction` 0, no mean-free-path volume |
| (f) inward building with a seed inside + street seed + block in the street | building out (flipped), street seed does not vote, unreached block into itself |
| (g) seed inside a column | column holds air, conflict reported (stats and `analyze` warning, 12 faces) |
| (g) `seed_outside_escape_fraction=0` around a seed in a closed room | escaping-seed conflict, room still out |
| (h) unwelded closed box, no hits, seed outside / inside (via `decide_orientation`) | solid / air |
| closed box, no hits, seed outside / inside | solid / air, no conflict |
| no seeds, unreached inward box | normals out (old rule kept) |

Scenarios (a), (e, seeds on both sides), (f, block) and (g) fail on `50f4685` (prepared through `prepare_mesh` only):

```
before (50f4685)                                                     after
(a) slab in free field, modelled outward   normals out of the slab, hits 0/0     into the slab, hits 92/0
(a) slab in free field, modelled inward    normals out of the slab, hits 0/0     into the slab, hits 92/0
(d) column in room (column modelled inward) room out, column into                 room out, column into
(e) outdoor panel, seeds on both sides     two-sided 0/2, hits 0/0               two-sided 2/2, hits 119/138
(f) building + street block                building out, block out of it         building out, block into it
(g) a seed inside the column               column out, no diagnostic             column out, conflict (52000 air / 4159 other hits)
```

Full suite (`python -m pytest tests -q -p no:cacheprovider -n 4`, worktree `src` first on the path):

- base `50f4685`: 1205 passed, 2 skipped;
- fix: 1225 passed, 2 skipped. No test changed outcome; the 20 new ones pass.

compas_soundscape backend tests that prepare meshes (`test_acoustic_paths`, `test_pffdtd_preflight`,
`test_simulation_mesh`, `test_simulation_preflight`), `compas-soundscape` env with this `src` overlaid by
`PYTHONPATH`: 39 passed before and after.

`ruff check` and `ruff format --check`: clean.

## Prepared-geometry comparison (instead of simulations)

A simulation of a prepared mesh is a deterministic function of the prepared mesh, so the change is validated on
`prepare_mesh`'s output. `tools/prepared_geometry.py collect` recorded the inputs of every `prepare_mesh` call of the
`examples/hil_d24_1.py` room (as `RoomGeometry.from_scene(scene, prepare=PrepareOptions())` prepares it, its 5
sources and receivers as seeds) and of the test files `test_geometry_prepare`, `test_geometry_report`,
`test_geometry_roles`, `test_open_rooms`, `test_prepared_accounting`, `test_room_from_prepared`, `test_sector_edges`,
`test_sector_wiring` and `test_closed_solid_orientation` (62 distinct scenes). The BRAS scenes were captured through the
app (`services.pffdtd_job.prepare_pffdtd_mesh`, as `orient3.py` prepares them: RS1_rigid, RS1_absorbing, RS1_diffuse,
RS2_1m_rigid, RS2_1m_absorbing, RS2_2m_rigid, RS2_2m_absorbing, RS2_2m_absorbing_45deg, RS5, RS6, RS7). Each corpus
was prepared by `50f4685` (`git archive` of its `src`) and by the fix (`dump`), and the dumps compared face by face
(normals/winding, two-sidedness, face class, air and other hits, walls, enclosing objects, seed statuses, air volume,
key stats, conflicts). The corpora and dumps are under `examples/results/geometry/` (git-ignored).

### compas_acoustics scenes: 51/62 identical

- **hil_d24_1: identical** (376 faces, 45 components of which 9 closed, all 5 seeds inside, 232 flipped, 364
  two-sided, 109 walls, no conflict). Its simulation is therefore unchanged by construction, and its simulation
  comparison (`tools/regression_reference.py`) was not needed. (The regression tool simulates the example with
  `Scene.simulate`, which does not prepare the mesh at all; the reference recording started on `50f4685` was stopped
  during the hybrid run when the validation switched to prepared geometry.)
- `test_objects_the_air_never_reaches_are_listed`: the two closed boxes buried outside the closed room ("buried",
  "buried too", 24 faces) now point into themselves (no seed inside: solids); they are unreached, so nothing the test
  checks or the room counts changes.
- `test_the_room_holding_the_sources_is_never_an_object`: same geometry; a new conflict: a receiver inside the closed
  "cabinet" (seed 4), cabinet holding air (as before), 47300 hits inside, 5453 outside. A real situation: a closed
  surface with air on both sides that a one-sided wall cannot represent.
- The 9 other differing scenes are the new test scenes (a), (b), (e), (f), (g), each changed as designed.

### BRAS scenes via the app: 0/11 identical, every change by design

```
RS1_rigid       winding 26: floor closed slabs 24 (into), floor plane 2 (now -z); class/hits 4
RS1_absorbing   winding 590: RockFon closed blocks 588 (into), floor plane 2; class/hits 113
RS1_diffuse     winding 794: MDF closed elements 792 (into), floor plane 2; class/hits 335 (328 on loose-face enclosing MDF, winding unchanged)
RS2_1m_rigid    class/hits 3 (MDF loose faces now reached), winding unchanged
RS2_1m_absorbing winding 12: absorber slab (into); class/hits 3
RS2_2m_rigid    class/hits 5 (MDF), winding unchanged
RS2_2m_absorbing / _45deg  winding 28: absorber slab (into); class/hits 5; MDF unchanged
RS5             winding 23: floor plane 2, MDF25 diffuser (open, non-manifold) 21 faces by per-face votes; class/hits 23
RS6             winding 468: MDF closed elements (into). Seeds outside + leaky (escape 0.497): mixed, so no outdoor walk; hits unchanged
RS7             winding 362: MDF closed elements 360 (into), floor plane 2; class/hits 117
```

The "divergence" air volume of RS1/RS2/RS7 changes in the 16th digit only (the absolute signed volume of the same
closed component summed with the opposite winding). No BRAS scene has an orientation conflict.

Per scene, as the app prepares it (`orient_summary.py`: per component the final orientation, `in` = signed volume < 0):

| scene | component | before | after |
| --- | --- | --- | --- |
| RS2_2m_absorbing | 'Plaster_Mortar_White' (RockFon slab, faces 28-55, closed) | out, front x = 0 normal +x, 0 hits | **in**, front x = 0 normal **-x**, 153 air hits |
| RS2_2m_absorbing | 'MDF12mm' (loose, enclosing) | front normal -x (into the plate), 0 hits | unchanged |
| RS1_absorbing | 49 RockFon blocks (closed) | out | **in** |
| RS1_absorbing | floor plane (open) | top z = 0 normal +z | **-z** (into the floor), one-sided |
| RS1_rigid | 2 floor slabs (closed) / floor plane | out / top z = 0.05 normal +z | **in** / **-z**, one-sided |
| RS5 | floor plane | +z | **-z** |
| RS6 | 39 MDF elements (closed) | out (1 more already in) | **in** (all 40) |
| RS7 | 30 MDF elements (closed) / floor plane | out / +z | **in** / **-z** |

## PFFDTD repro (RS2_2m_absorbing 0 deg vs 45 deg, 400 Hz, ppw 10, LS1-MP1)

`mat_repro.py` through the app (`compas-soundscape` env, this `src` overlaid), `PFFDTD_GPU_MEMORY_GB=10`:

| | max abs(IR 0 deg - IR 45 deg) / max abs(IR) |
| --- | --- |
| before (`50f4685`) | 0.000e+00 (bit-identical: the absorber acted rigid) |
| after | **7.689e-02** |

## Limits and risks

- Indoor rooms: a closed shell holding a seed is oriented as before (parity and majority agree); every indoor scene of
  the corpus, hil_d24_1 included, is identical. A closed component that the walk reaches but that holds no seed was
  already a solid by the majority. What changes indoors: unreached closed components holding no seed (outside the
  room, behind the shell, an unreached second room) now point into themselves instead of out; they are unreached, so
  they leave the accounting and diffraction as before, and still reflect. A seed inside furniture or a column now also
  gives an `orientation_conflict` warning.
- A self-intersecting or badly modelled closed component could fool the parity test where the old majority was
  right; the conflict diagnostic flags it when the walk disagrees.
- A free-field scene still has `leak_fraction` 0, so `RoomGeometry.from_prepared` calls it closed and takes the
  divergence volume of its biggest closed component, as before this change (not addressed here).
- RS6's second seed is "leaky" (escape 0.497 < 0.5), so RS6 is a mixed scene and its outside seed does not vote; its
  open MDF parts stay mostly unreached.
