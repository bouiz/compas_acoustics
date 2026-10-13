# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Grasshopper installer for Rhino 8: `installer/compas_acoustics_installer.gh`. It installs a `.whl` file (local file or download link) with the Python of Rhino, and adds the Grasshopper components.
- `invoke installer-bundle`: builds the Grasshopper components and the wheel, and zips them with the installer.
- `MaterialCategory` and `AcousticMaterial.category`. `MaterialsLibrary.materials_by_category`.
- `tools/import_backend_materials.py`: imports the materials of pyroomacoustics into the materials library, with a category and the name `"<key> (from pyroomacoustics)"`.
- Material Library component: materials are grouped by category. Construct and Deconstruct Material components: new `category` input and output.
- `ImpulseResponse` data class wrapping a source–receiver pair, raw IR array, sample rate, and simulation platform; includes `resampled()` for poly/FFT resampling
- Export `ImpulseResponse` from the public package API
- `Scene.ir_graph` — a compas `Graph` linking sources and receivers, with `ImpulseResponse` stored on edges
- Scene IR graph API: `add_node_to_ir_graph`, `add_impulse_response`, `populate_ir_graph`, `get_impulse_response`, `clear_ir_edges`
- Automatic IR graph node registration when sources/receivers are added via `add_source`, `add_receiver`, or `add_acoustic_object`
- `Simulator.rir_from_ray_tracing(..., random_seed=None)` — optional seed for reproducible pyroomacoustics ray-tracing runs; `None` re-seeds with fresh entropy each run
- `Ray` — a `compas.geometry.Polyline` subclass representing a traced acoustic ray, with metadata (source, ray index, initial direction, sound speed), travel distance/time, spotlight-cone membership (`is_in_cone`), and truncation by reflection order/distance/time (`up_to_reflection_order`, `up_to_distance`, `up_to_time`)
- `ScatteredRay` — a `Ray` subclass for coefficient-driven scattered rays born at a reflection point, carrying parent ray index, scattered energy, and a distance offset
- `RayField` — a container for traced rays with `filter` (by source and/or ray index), `truncated` (by reflection order or travel time), and `display_in_viewer` (specular and/or scattered rays over the scene)
- `Simulator.trace_rays(..., direction=None, angle=None)` — reconstructs specular ray paths from the pyroomacoustics `libroom` engine, optionally emitting all rays directly into a spotlight cone instead of omnidirectionally, and optionally spawning coefficient-driven scattered rays, returning a `RayField`
- Export `Ray`, `ScatteredRay`, and `RayField` from the public package API
- `AcousticObjectsGroup` — a composite container for `AcousticObject` (and nested groups), used to model a single logical entity (e.g. a room) as several per-surface acoustic objects; provides `iter_acoustic_objects()`, `add_acoustic_object()`, `remove_acoustic_object()`, `remove_acoustic_object_by_name()`, `get_acoustic_object_by_name()`, and an aggregated `sources` property
- `Scene.iter_acoustic_objects()` — yields every leaf `AcousticObject` in the scene, recursing into `AcousticObjectsGroup`
- `Scene.scene_materials`, `Scene.print_scene_materials()`, `Scene.display_materials_in_viewer()` — per-object material introspection and visualisation, replacing the removed surface-group equivalents
- `Source.directivity` (`"omnidirectional"`, `"cardioid"`, `"supercardioid"`, `"hypercardioid"`, `"bidirectional"`) and `Source.sound_file` — assigning a sound file automatically downmixes stereo/multi-channel audio to mono (writing a sibling `mono_<name>` file) so it can be convolved without errors
- `Receiver.direction`, `Receiver.radius`, `Receiver.received_spls`, and `Receiver.total_spl` — energy-summed SPL (dB) across all contributions collected on the receiver
- `compas_acoustics.pra_converter` module — conversion helpers between compas_acoustics and pyroomacoustics objects: `vector_to_pra`, `directivity_to_pra`, `points_to_array`, `source_to_pra`, `receivers_to_pra_mic_array`, `material_to_pra`, `mesh_to_pra_wall`, `acoustic_object_to_pra_walls`, `scene_to_pra_room`
- `Simulator.spls_from_image_source_method(room, sources, receivers, start_db, target_db, time_window=0.125)` — deterministic, image-source-based SPL estimate at each receiver (no ray-count-dependent noise), summing only image sources that arrive within `time_window` of the direct sound
- `Simulator.rir_from_ray_tracing(..., energy_thres=1e-7, time_thres=10.0)` — configurable ray-tracing stopping thresholds, forwarded to pyroomacoustics' `Room.set_ray_tracing`
- `compas_acoustics.stage_acoustics` module — ISO 3382-1 Annex A stage-acoustics parameters computed from an `ImpulseResponse`: `compute_ST1`, `compute_ST2`, `compute_G_stage`, `compute_ITDG`, `stage_support_parameters`
- `compas_acoustics.orchestra_layouts` module — named string-section seating layouts: `string_quartet`, `chamber_strings`, `medium_chamber_strings`, `symphonic_strings`
- `compas_acoustics.numerical.fitness.regularized_fitness` — a MAE + weighted-STD loss function for scoring a list of values against a target, for use in optimisation objectives
- `AudioPlayer` — cross-platform (Windows/macOS) playback of `.wav` files or raw numpy arrays (`play`, `play_file`, `play_array`, `stop`)
- `anechoic_soundfiles_library` — bundled anechoic speech/viola/trumpet recordings (OpenAIR, CC BY 4.0) for auralisation demos
- `hrtfs_library` — bundled MIT KEMAR HRTF sets (`mit_kemar_normal_pinna.sofa`, `mit_kemar_large_pinna.sofa`) used by `Auraliser`
- Preset acoustic materials in `MaterialsLibrary`: `acoustic_plaster`, `brickwall`, `concrete`, `carpet` (renamed from `test_material`), `wall_absorber_50mm`
- `Defaults.HRTF_LIBRARY_PATH`, `Defaults.ANECHOIC_SOUNDFILES_LIBRARY_PATH`, `Defaults.PARAM_NAMES`
- `compas_acoustics_ghpython` — a new package for Rhino and Grasshopper, with 18 Grasshopper components covering the full workflow (Acoustic Object, Source, Receiver, Directivity, Channel, Model, Orchestra Layout, Material Library, Construct/Deconstruct Material, Simulate Ray Trace, Analyse IR, Deconstruct IR, Deconstruct IR Graph, Auralise, Audio Player, Regularized Fitness, Plot 2D, Plot 2D Settings). Build the `.ghuser` files with `invoke build-cpython-ghuser-components`
- `compas_acoustics_ghpython.plotter` — `AcousticParameter`, `Plotter2D`, `Plotter2DSettings` for plotting acoustic parameters in Rhino/Grasshopper
- `sounddevice` added as a runtime dependency (used by `AudioPlayer.play_array`); `pythonnet` added as a dev dependency

### Changed

- **Breaking:** `Simulator.rir_from_image_source_method()` and `Simulator.rir_from_ray_tracing()` now return `dict[str, ImpulseResponse]` (keys `{receiver_guid}_{source_guid}`) instead of a nested `list[list[ndarray]]`
- **Breaking:** RIRs are returned at the simulation sample rate; automatic resampling to `Defaults.SAMPLING_RATE` was removed — use `ImpulseResponse.resampled()` when a different rate is needed
- `Simulator` now populates `scene.ir_graph` automatically after each successful RIR computation
- `Scene.add_acoustic_objects()` now calls `add_acoustic_object()` per item so embedded sources are registered in the IR graph
- `Analyser` now uses numpy and returns a dictionary with parameters and overall values
- **Breaking:** `AcousticObject.__init__` now takes a single `material: AcousticMaterial` instead of `surface_groups: dict`; per-surface materials are now expressed by grouping several single-material `AcousticObject`s in an `AcousticObjectsGroup` rather than tagging faces of one mesh
- **Breaking:** `AcousticObjectsLibrary.shoebox_room()` now returns an `AcousticObjectsGroup` of six single-material surface objects (floor, walls, ceiling) instead of a single `AcousticObject`, and takes a `materials` argument (`None`, one `AcousticMaterial`, or a dict keyed by surface name) instead of materials being assigned after construction
- **Breaking:** `Source.__init__` parameter `directional: bool` replaced by `directivity: str` (`"omnidirectional"`, `"cardioid"`, `"supercardioid"`, `"hypercardioid"`, `"bidirectional"`); `Source.direction` is now a `compas.geometry.Vector` (was a plain list) and defaults to `(0, 0, 1)` (was `(1, 0, 0)`)
- **Breaking:** `Receiver.direction` is now a `compas.geometry.Vector` defaulting to `(0, 0, 1)` (was a plain list defaulting to `(1, 0, 0)`)
- `Simulator._pyroomacoustics_simulation()` now builds walls and materials via `pra_converter.scene_to_pra_room()` instead of inline pyroomacoustics wall construction, recursing into `AcousticObjectsGroup` via `iter_acoustic_objects()`
- `Simulator._add_pyroomacoustics_sources_receivers()` now adds sources/receivers via `pra_converter.source_to_pra()` / `receivers_to_pra_mic_array()`, so source directivity and sound files are respected during simulation
- Grasshopper component build task renamed `build.build_ghuser_components` → `build.build_cpython_ghuser_components`
- The `compas_acoustics` package does not use Rhino, Grasshopper or .NET anymore. This code is now in `compas_acoustics_ghpython`. On Windows, `AudioPlayer` uses `winsound` instead of `System.Media`.
- The package installs on Python 3.9 (Rhino 8). There, it uses older versions of pyfar and pyrato, and STI is not available: `AcousticAnalyser.compute_sti` shows a warning and returns NaN.
- **Breaking:** material names can have spaces and capital letters. Library files are named after `AcousticMaterial.slug`.

### Fixed

- Wheels were missing the materials library, the HRTFs, the sound files and the Grasshopper components.
- `Simulator._pyroomacoustics_simulation` now passes surface scattering coefficients to `pyroomacoustics.Material` correctly; previously the whole material dict was passed as the positional `energy_absorption` argument and scattering was silently dropped (always zero)
- `AcousticObject.display_surface_groups_in_viewer`, `Scene.display_surface_groups_in_viewer`, and `RayField.display_in_viewer` now call `compas.is_rhino()` instead of treating it as a boolean attribute, so the Rhino guard actually raises when appropriate
- `Auraliser` now resolves its default HRTF file via `Defaults.HRTF_LIBRARY_PATH` and ships `mit_kemar_normal_pinna.sofa`; previously it pointed at `KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa`, a file not present anywhere in the repository

### Removed

- `Tools` removed from the public package exports (module still exists but is no longer re-exported)
- `Simulator.resample_room_rir()` removed (replaced by per-IR resampling on `ImpulseResponse`)
- `AcousticObject.surface_groups`, `.acoustic_materials`, `.surface_group_colors`, `.face_colors`, `.acoustic_object_from_obj()`, `.display_surface_groups_in_viewer()`, and `.print_surface_groups()` — replaced by the single-material `AcousticObject` + `AcousticObjectsGroup` model
- `Scene.scene_surface_groups`, `.print_scene_surface_groups()`, and `.display_surface_groups_in_viewer()` — replaced by `.scene_materials`, `.print_scene_materials()`, and `.display_materials_in_viewer()`

