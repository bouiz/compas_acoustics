# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Grasshopper installer for Rhino 8: `installer/compas_acoustics_installer.gh`. It installs a `.whl` file (local file or download link) into a Rhino script environment, and adds the Grasshopper components.
- `invoke installer-bundle`: builds the Grasshopper components and the wheel, and zips them with the installer.
- GitHub Actions use the new `compas-dev/compas-actions` (v1). The build workflow makes the installer bundle (artifact). Releases are made with a pull request (`prepare release` workflow). After the merge, the release workflow builds the Grasshopper components into the wheel, and adds the wheel and the installer bundle to the GitHub release. Tests also run on Python 3.9.
- `invoke pre-build` and `invoke build-cpython-ghuser-components` build the Grasshopper components with `ghpython_componentizer`.
- `MaterialCategory` and `AcousticMaterial.category`. `MaterialsLibrary.materials_by_category`.
- `tools/import_backend_materials.py`: imports the materials of pyroomacoustics into the materials library, with a category, the key as the name, and `data_origin` set to the backend.
- Material Library component: materials are grouped by category. Construct and Deconstruct Material components: new `category` input and output.
- `AcousticMaterial.data_origin`: where the coefficients come from. Defaults to `"custom"`; materials imported from pyroomacoustics store `"pyroomacoustics"`. Included in serialization and in `AcousticMaterial.__str__`.
- Material Library component: a remark when the chosen material has no scattering coefficients, and the component message shows the material name and `data_origin`.
- Deconstruct Material component: `data_origin` output.
- `AirConditions(temperature=20, humidity=50, pressure=101325)`: the air of a simulation, with the speed of sound after Cramer (1993) and the air absorption after ISO 9613-1:1993, both from `pyfar.constants`; for `Pyroomacoustics(air=...)`, the ray tracer and `Hybrid(air=...)`, and recorded in `generation['air']` and `generation['speed_of_sound']`. A `Hybrid` takes its references' air and refuses a different one
- `Hybrid.binaural(hrtf_path=None)`: binaural responses whose early part (image sources, UDFA and BTMS paths) is filtered path by path with the HRTF measured nearest each direction of arrival, as RAVEN renders image sources; the late field is MagLS-decoded as before. `compas_acoustics.binaural` (`HrtfSet`, `render_early_binaural`, `binaural_rir`, and `magls_filters`, moved there from the `Auraliser`), and binaural impulse responses in the `Auraliser`
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
- `Scene.scene_materials`, `Scene.print_scene_materials()`, `Scene.display_materials_in_viewer()` — per-object material introspection and visualisation, replacing the removed surface-group equivalents
- `Source.directivity` (`"omnidirectional"`, `"cardioid"`, `"supercardioid"`, `"hypercardioid"`, `"bidirectional"`) and `Source.sound_file` — assigning a sound file automatically downmixes stereo/multi-channel audio to mono (writing a sibling `mono_<name>` file) so it can be convolved without errors
- `Receiver.direction`, `Receiver.radius`, `Receiver.received_spls`, and `Receiver.total_spl` — energy-summed SPL (dB) across all contributions collected on the receiver
- `compas_acoustics.pra_converter` module — conversion helpers between compas_acoustics and pyroomacoustics objects: `vector_to_pra`, `directivity_to_pra`, `points_to_array`, `source_to_pra`, `receivers_to_pra_mic_array`, `material_to_pra`, `mesh_to_pra_wall`, `acoustic_object_to_pra_walls`, `scene_to_pra_room`
- `Simulator.spls_from_image_source_method(room, sources, receivers, start_db, target_db, time_window=0.125)` — deterministic, image-source-based SPL estimate at each receiver (no ray-count-dependent noise), summing only image sources that arrive within `time_window` of the direct sound
- `Simulator.rir_from_ray_tracing(..., energy_thres=1e-7, time_thres=10.0)` — configurable ray-tracing stopping thresholds, forwarded to pyroomacoustics' `Room.set_ray_tracing`
- `compas_acoustics.orchestra_layouts` module — named string-section seating layouts: `string_quartet`, `chamber_strings`, `medium_chamber_strings`, `symphonic_strings`
- `compas_acoustics.numerical.fitness.regularized_fitness` — a MAE + weighted-STD loss function for scoring a list of values against a target, for use in optimisation objectives
- `AudioPlayer` — cross-platform (Windows/macOS) playback of `.wav` files or raw numpy arrays (`play`, `play_file`, `play_array`, `stop`)
- `anechoic_soundfiles_library` — bundled anechoic speech/viola/trumpet recordings (OpenAIR, CC BY 4.0) for auralisation demos
- `hrtfs_library` — bundled MIT KEMAR HRTF sets (`mit_kemar_normal_pinna.sofa`, `mit_kemar_large_pinna.sofa`) used by `Auraliser`
- Preset acoustic materials in `MaterialsLibrary`: `acoustic_plaster`, `brickwall`, `concrete`, `carpet` (renamed from `test_material`), `wall_absorber_50mm`
- `Defaults.HRTF_LIBRARY_PATH`, `Defaults.ANECHOIC_SOUNDFILES_LIBRARY_PATH`, `Defaults.PARAM_NAMES`
- `compas_acoustics_ghpython` — a new package for Rhino and Grasshopper, with 18 Grasshopper components covering the full workflow (Acoustic Object, Source, Receiver, Directivity, Channel, Model, Orchestra Layout, Material Library, Construct/Deconstruct Material, Simulate Ray Trace, Analyse IR, Deconstruct IR, Deconstruct IR Graph, Auralise, Audio Player, Regularized Fitness, Plot 2D, Plot 2D Settings). Build the `.ghuser` files with `invoke build-cpython-ghuser-components`
- `compas_acoustics_ghpython.plotter` — `AcousticParameter`, `Plotter2D`, `Plotter2DSettings` for plotting acoustic parameters in Rhino/Grasshopper
- `sounddevice` added as a runtime dependency (used by `AudioPlayer.play_array`); `ghpython_componentizer` added as a dev dependency
- `tests/test_python39.py`: the package imports on Python 3.9 without the optional dependencies
- `AcousticAnalyser.stage_support()` — early and late stage support (`ST_early` 20-100 ms, `ST_late` 100-1000 ms relative to the first 10 ms; ISO 3382-1:2009 Annex C, Gade) per octave band, with single-number values averaged over 250-2000 Hz; each time window is cut before octave filtering so low-band filter delay does not leak direct sound into the reflection windows
- `AcousticAnalyser.sound_strength(reference_ir)` — sound strength G per octave band via `pyrato.parameters.sound_strength`, relative to a 10 m free-field reference, with the single-number value averaged over 500 and 1000 Hz
- `AcousticAnalyser` accepts an `ImpulseResponse`, analysed at its raw level, and takes its sample rate
- `Auraliser.decode()` with `decode_mono()` (W channel), `decode_stereo()` (virtual cardioids pointing left and right; mono signals play on both channels) and `decode_binaural()`, which renders ambisonics (order 1 or higher) to binaural via MagLS decoding filters (`Auraliser.binaural_filters()`) computed from the HRTF SOFA file, following `spaudiopy.decoder.magls_bin`/`sh2bin` (Hold et al., 2023)
- Grasshopper Auralise component: `output_format` input (`stereo` default, `binaural`, `mono`, `ambisonics`) and `hrtf` input (`kemar_normal_pinna`, `kemar_large_pinna` or a `.sofa` path), each with a Value List placed automatically when unconnected, and a `sample_rate` output
- Grasshopper Audio Player component: optional `sample_rate` input for numpy array sounds; signals with more than two channels are rejected with an error
- `Auraliser` now accepts a list of `ImpulseResponse` objects as well as the dict format, and a list of sound file paths as well as the dict format; when `anechoic_sounds` is omitted, each source's `sound_file` is used. Impulse responses and sounds are paired by id when the ids match, otherwise by position
- `ImpulseResponse` peak normalisation: the raw impulse response is kept with its `normalization_factor` (peak over all channels) and `level_db`; `normalized=True` (default) makes `ir` return the peak-normalised response; `denormalized()` returns the raw response and `ImpulseResponse.denormalize_group()` scales a group relative to its loudest impulse response
- `ImpulseResponse.to_wav()` / `ImpulseResponse.from_wav()` — peak-normalised 32-bit float WAV plus a JSON metadata file (normalisation factor, generation method and settings, source, receiver, sample rate); `ImpulseResponse.generation` and `ImpulseResponse.metadata`
- `Simulator.rir_from_image_source_method()` / `rir_from_ray_tracing()` `normalize=True` option; impulse responses record how they were generated
- `Auraliser` output level: `level='relative'` (default; one gain for a group of renders so the loudest peaks at `headroom_db`, keeping level differences), `'individual'` or `'none'`, `headroom_db=-1`; `Auraliser.render_gains()` and `Auraliser.peak_dbfs()`; `export_convolved_sound(..., gain=None)`
- `AudioPlayer.play(..., normalize=True)` option to play arrays at their rendered level
- Grasshopper: Auralise `level` input (Value List) applying one gain across all signals of a run and showing the output peak in dBFS; Audio Player `normalize` input; Simulate Ray Tracing `normalize` input
- `RoomGeometry` — a scene's surfaces as one validated room: vertices welded transitively across objects, surfaces oriented consistently and outward, closedness checked, obstacles stored two-sided while the drawn obstacles stay available (`obstacle_faces`), materials extended to a 62.5 Hz band; `RoomGeometry.from_mesh()` for shared vertices and faces; coplanar faces merged by default in `from_scene`; `to_pra()` builds the pyroomacoustics room at any sample rate, holding the 8 kHz coefficients above 8 kHz, and gives a material without scattering coefficients a scattering of 0.05 in every band (`room_geometry.DEFAULT_SCATTERING`)
- `RoomGeometry.is_inside()` — generalised winding numbers over the drawn faces: a point inside a closed obstacle (a pillar, a cabinet) is outside the room, and a scene without a closed boundary (an open field) has no outside; every pyroomacoustics room built by `to_pra` answers `is_inside` with it
- `RoomGeometry.mergeable_faces` and a `UserWarning` when coplanar neighbours of one object and material could be merged (triangulated Rhino surfaces lose image sources at their seams)
- `role` on `AcousticObject`: `'boundary'` (default) or `'obstacle'`, and `Scene.iter_acoustic_objects_with_roles()`: obstacles are free-standing objects inside the room, boundaries close it
- `compas_acoustics.diffuse` — Sabine and Eyring reverberation times through pyrato (`predicted_rt`, `mean_absorption`, `DiffuseFieldWarning`), in closed form on pyrato < 1.0 (Python 3.9)
- Receiver orientation: `Receiver.up`, `Receiver.rotation` (columns facing, left, up), `Receiver.frame` and `Receiver.from_frame()`; the orientation sets the axes of ambisonic responses (x facing, y left, z up), and `transform()` turns it with the receiver
- `Pyroomacoustics(image_source_order=3, ray_tracing=False, ...)` — the pyroomacoustics method: image sources by default, plus hybrid-rir's ray tracer with `ray_tracing=True` (`n_rays`, `energy_thres`, `duration`, `receiver_radius`, `seed`, `n_jobs`)
- `ImpulseResponse` carries `ambisonic_order`, `normalisation`, `channel_order` (also in its JSON metadata), `n_channels` and `direct_sound_index()`; `resampled()` keeps channels
- `Scene.impulse_responses(attribute)`, `Scene.simulation_methods()`; `attribute=` on `Scene.add_impulse_response`, `populate_ir_graph` and `get_impulse_response`
- A `UserWarning` before an image source search that will take long (`methods.image_source_search_seconds()`: it grows about `faces / 2.2` times per order on a mesh room)
- `AcousticObject.from_obj(path, materials, roles='boundary')` — a list of `AcousticObject`s, one per OBJ group (Rhino layer), with a material and a role per group (`roles={'chairs': 'obstacle'}` for furniture); replaces `AcousticObject.acoustic_object_from_obj`. `compas.files.OBJ` reads and welds the geometry; the groups are read from the file itself, as `OBJ.groups` keeps only the last run of a group name that recurs, as Rhino writes them
- `Scene.simulate(method, sampling_rate=48000)` — runs a simulation method on the scene and stores its responses on the IR graph; the one way to simulate. Methods implement `compute(context)`, and `SimulationContext.from_scene(scene, sampling_rate)` builds the context
- `RayField.from_scene(scene, ...)` — traced rays for visualisation, replacing `Simulator.trace_rays`
- Sound strength `G` is a main analyser parameter (`AcousticAnalyser.G`, `G_overall`, in `analyse()`, tables, plots, CSV and the Grasshopper outputs); without a `reference_ir` it uses the package-scale free field at 10 m, and is NaN for responses without the package scale
- Grasshopper: Deconstruct IR Graph `method` input; the ray-tracing component (now "Simulate Pyroomacoustics") gets a `ray_tracing` input and shows the library's warnings (points outside, open rooms, mergeable faces, a slow search) as component messages
- `Pyroomacoustics` and `SimulationMethod` exported from `compas_acoustics`
- `compas_acoustics.methods` package: `base` (`SimulationContext`, `SimulationMethod`), `pra` (`Pyroomacoustics`) and `hybrid` (`Hybrid`, loaded on first use because it needs the `hybrid` extra)
- Demonstration notebook: `examples/notebooks/01_pyroomacoustics_simulation.ipynb`
- `compas_acoustics.Hybrid` (`compas_acoustics.methods.Hybrid`) — hybrid ambisonic room impulse responses (ACN, SN3D or N3D): image sources and, optionally, edge diffraction around obstacles (`diffraction="udfa"`, or `"btms"`: BTMS kernels ported from EDtoolbox, BSD licence, Peter Svensson) for the early field; the next image source order drives a directional feedback delay network with direction-dependent boundary absorption. `tail` sets its decay target: a reference response (`reference=` a method name, a dict or a `Pyroomacoustics` method), the hybrid's own image-source plus ray-traced reference (default without a reference), or an Eyring or Sabine prediction that needs no reference (`tail="eyring"`, warning where the room is not diffuse). Ported from hybrid-rir (revision in `tests/hybrid/UPSTREAM_REVISION`); needs the new `hybrid` extra (`pip install compas_acoustics[hybrid]`, Python >= 3.11)
- Grasshopper: Simulate Hybrid component (`tail` and `diffraction` Value Lists, optional `reference` method); on Python 3.9 (Rhino 8) it reports that the hybrid method needs Python >= 3.11 and the `hybrid` extra
- `hybrid` extra (`pyFDN>=0.5,<0.6`, `spharpy`, `multislope`; Python >= 3.11 only) and a CI job for `tests/hybrid`; `tests/test_wheel.py` pins the wheel's contents
- `compas_acoustics.Hybrid` is loaded on first use (it needs the `hybrid` extra)
- Demonstration notebook: `examples/notebooks/02_hybrid_method.ipynb`
- A test pins the hybrid's late level to the exact answer of a specular shoebox (image sources to order 100), within 1 dB. The network's drive gain stays as it is: the hybrid is within about 1 dB of that answer and of Barron's revised theory, while pyroomacoustics' ray tracer drifts with the scattering coefficient (2.6 dB low to 3.6 dB high), so the ray tracer is not used to calibrate it
- Hybrid diffraction at the edges of the room's own boundary (re-entrant corners, the corners of closed volumes inside the room), ported from hybrid-rir `4b42bfd`: every boundary edge whose air side opens wider than 180 deg diffracts with the obstacle edges, in UDFA and BTMS; an open boundary is not searched (a `ShellEdgeWarning`, `enclosure_edges` returns no edges)

### Changed

- `AcousticAnalyser.direct_to_reverberant_ratio` follows Haapaniemi and Lokki (2018, eq. 1): the direct sound is the first 5 ms from its arrival, with the window opened 1 ms early to hold a band-limited arrival. It was +-2.5 ms around the arrival (or the peak), which counted the 2.5 ms before the arrival as direct and reflections from 2.5 ms on as reverberant. `DRR_overall` is their single-number value: the impulse response filtered with a second-order Butterworth band-pass of 700-4000 Hz, the band of their localisation threshold (-2.7 to 1.7 dB in concert halls); it was the mean of the 500, 1000 and 2000 Hz octaves. Per octave band the windows are cut before filtering, as before; validated against responses whose direct and reverberant parts are filtered separately (within 0.02 dB in every octave)
- `AcousticAnalyser.reverberation_times` uses Schroeder's backward integration (pyrato `schroeder_integration`, normalised at the direct sound) and pyrato's `reverberation_time_linear_regression`, on the response up to its last non-zero sample (trailing zeros are cut). There is no noise compensation: pyrato's `energy_decay_curve_chu_lundeby` found no Lundeby intersection in a ray-traced response that decays into numerical zero (a 408-face office: six of seven bands NaN). A measured response with a noise floor should be truncated before it is analysed
- Without `AirConditions`, every simulation now runs in `AirConditions()`, 20 C, 50 % and 101325 Pa: 343.98 m/s instead of 343 m/s, and ISO 9613-1 air absorption at 20 C and 50 % for every method. Before, `Pyroomacoustics` and the ray tracer used pyroomacoustics' own air table (19.3 C and 0.3 % humidity, derived from 343 m/s) while the hybrid's image sources, diffraction and Eyring/Sabine targets used a fixed ISO 9613-1 table at 10 C and 70 %, documented as 20 C and 50 %. In a 20 x 15 x 6 m room the predicted reverberation changes by -0.4 % below 250 Hz, -2 % at 500-1000 Hz and +3 to +7 % at 4-8 kHz. Responses loaded from older files keep 343 m/s for their direct-sound sample
- `Receiver.direction` defaults to +x instead of +z
- **Breaking:** `Simulator.rir_from_image_source_method()` and `Simulator.rir_from_ray_tracing()` now return `dict[str, ImpulseResponse]` (keys `{receiver_guid}_{source_guid}`) instead of a nested `list[list[ndarray]]`
- **Breaking:** RIRs are returned at the simulation sample rate; automatic resampling to `Defaults.SAMPLING_RATE` was removed — use `ImpulseResponse.resampled()` when a different rate is needed
- `Simulator` now populates `scene.ir_graph` automatically after each successful RIR computation
- `Scene.add_acoustic_objects()` now calls `add_acoustic_object()` per item so embedded sources are registered in the IR graph
- `Analyser` now uses numpy and returns a dictionary with parameters and overall values
- **Breaking:** `AcousticObject.__init__` now takes a single `material: AcousticMaterial` instead of `surface_groups: dict`; per-surface materials take one `AcousticObject` per surface
- **Breaking:** `AcousticObjectsLibrary.shoebox_room()` now returns a list of six single-material `AcousticObject`s (floor, walls, ceiling; added with `Scene.add_acoustic_objects`) instead of a single `AcousticObject`, and takes a `materials` argument (`None`, one `AcousticMaterial`, or a dict keyed by surface name) instead of materials being assigned after construction
- **Breaking:** `Source.__init__` parameter `directional: bool` replaced by `directivity: str` (`"omnidirectional"`, `"cardioid"`, `"supercardioid"`, `"hypercardioid"`, `"bidirectional"`); `Source.direction` is now a `compas.geometry.Vector` (was a plain list) and defaults to `(0, 0, 1)` (was `(1, 0, 0)`)
- **Breaking:** `Receiver.direction` is now a `compas.geometry.Vector` defaulting to `(0, 0, 1)` (was a plain list defaulting to `(1, 0, 0)`)
- `Simulator._pyroomacoustics_simulation()` now builds walls and materials via `pra_converter.scene_to_pra_room()` instead of inline pyroomacoustics wall construction
- `Simulator._add_pyroomacoustics_sources_receivers()` now adds sources/receivers via `pra_converter.source_to_pra()` / `receivers_to_pra_mic_array()`, so source directivity and sound files are respected during simulation
- Grasshopper component build task renamed `build.build_ghuser_components` → `build.build_cpython_ghuser_components`
- The `compas_acoustics` package does not use Rhino, Grasshopper or .NET anymore. This code is now in `compas_acoustics_ghpython`. On Windows, `AudioPlayer` uses `winsound` instead of `System.Media`.
- The package installs on Python 3.9 (Rhino 8). There, it uses older versions of pyfar and pyrato, and STI is not available: `AcousticAnalyser.speech_transmission_index` shows a warning and returns NaN.
- **Breaking:** material names can have spaces and capital letters. Library files are named after `AcousticMaterial.slug`.
- **Breaking:** library material names and filenames no longer include the backend suffix. Example: `carpet_thin_from_pyroomacoustics` is now `carpet_thin`. The backend is stored as `data_origin` instead. `tools/import_backend_materials.py` writes the key as the name and passes the backend as `data_origin`.
- Material Library component no longer sets the component nickname to the selected material name.
- `AcousticAnalyser.center_time()` uses `pyrato.parameters.center_time` instead of a manual computation
- `AcousticAnalyser` methods are named after the quantity they return: `compute_rt` -> `reverberation_times`, `compute_clarity_definition` -> `clarity_definition`, `compute_drr` -> `direct_to_reverberant_ratio`, `compute_sti` -> `speech_transmission_index`, `compute_stage_support` -> `stage_support`, `compute_strength` -> `sound_strength` (`center_time` is unchanged)
- Analyser results change at low frequencies for the same impulse response: the octave filters are applied zero-phase (EDT, T20, T30, Ts) or after cutting the time windows (C50, C80, D50, DRR), which removes a delay of about 18 ms at 250 Hz and 35 ms at 125 Hz
- On pyrato < 1.0 (Python 3.9), Ts is computed in closed form and G is NaN with a warning
- `Auraliser` reads audio and resamples via pyfar (polyphase) and converts N3D to SN3D. `Auraliser.convolve()` sums the impulse responses that share a sound, decodes them to the output format and then convolves once per sound with only the output channels; for 10 third-order sources this takes about 1 s instead of 136 s (stereo) or 26 s (binaural after convolving)
- `Auraliser` defaults to `output_format='stereo'` and to the sample rate of the first impulse response (was a fixed 48000 Hz); binaural decoding filters are cached across instances
- `Auraliser.export_convolved_sound()` writes the signal returned by `convolve()` (computed when not given) instead of decoding it again
- `librosa` is no longer a dependency; `spharpy` is, on Python >= 3.11. Binaural decoding needs it; N3D to SN3D conversion does not
- **Breaking:** `Auraliser.decode_stereo_simple()` removed; use `Auraliser.decode_stereo()`
- **Breaking:** `Auraliser.resample_signal()` removed; resampling uses `pyfar.dsp.resample` directly
- **Breaking:** `Auraliser.multiconvolution()` and `Auraliser.convolve(impulse_response, anechoic_sound)` replaced by `Auraliser.convolve()`, which returns the signal decoded to `output_format`
- **Breaking:** `Auraliser` no longer accepts FuMa channel order; impulse responses must use ACN channel order with N3D or SN3D normalisation
- **Breaking:** simulated impulse responses are at 48000 Hz by default (were always 16000 Hz); set `Simulator.sampling_rate` for another rate
- `Auraliser` auralises `ImpulseResponse` objects at their raw level, and exports peak at -1 dBFS (was 0 dBFS); with the default `level='relative'` a single render is scaled as before
- Assigning `Scene.acoustic_materials` raises an `AttributeError` that says how materials are assigned now (it used to store the dict and do nothing); reading it returns the scene's materials
- Surfaces with `role='obstacle'` reflect from both sides; closed rooms are wound outward automatically; boundary surfaces that do not close a volume raise a `UserWarning`
- A material without scattering coefficients scatters 0.05 in every band in every pyroomacoustics simulation (`room_geometry.DEFAULT_SCATTERING`), ray tracing included; given coefficients are used as they are. Image source rooms used 0 before, and ray tracing refused such materials
- Omnidirectional sources are passed to pyroomacoustics without a directivity object
- `AcousticMaterial`: a coefficient of 0 is a value, not an absence; only `None` or `[]` means "not given"; `frequency_bands` refuses grids other than the package bands
- Sources or receivers outside the room raise a `UserWarning` instead of printing; simulation settings are logged
- `Scene.number_of_sources` counts sources attached to acoustic objects
- `AcousticAnalyser` finds the direct sound as `ita_start_IR` of the ITA Toolbox does (ISO 3382-1): the last sample more than 20 dB below the maximum before it, moved back before an earlier arrival within 14 dB of the maximum, and no start when the maximum is less than 20 dB above the noise. It was the first sample within 20 dB of the maximum, anywhere. On the package scale the search starts 1 ms before the geometric arrival; it used to take the geometric sample itself, so a blocked direct path left the direct-sound windows empty (DRR of about -110 dB on responses behind an obstacle). The start now equals the ITA Toolbox's on 42 existing responses
- `Auraliser` and the Grasshopper Auralise component use an `ImpulseResponse`'s own `channel_order` and `normalisation` (an N3D response was decoded as SN3D)
- Grasshopper Simulate Ray Tracing leaves unset inputs to the method's defaults
- `Auraliser` decodes binaurally with the bundled `KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa` HRTF (free-field compensated, 48 kHz) instead of `mit_kemar_normal_pinna.sofa`
- The example scripts `demo.py`, `demo_obj.py`, `demo_rays.py` and `demo_scattered_rays.py` use the current API again; the README has a getting-started section
- **Breaking:** `Scene.simulate(method)` is the one way to simulate; every method returns responses on the package scale (free-field direct sound `1 / (4 pi r)`) and records how they were made in `ImpulseResponse.generation`
- `pyroomacoustics` is pinned below 0.11: the parallel ray tracer writes private `Room` state (`rt_histograms`, `simulator_state`)
- **Breaking:** `Hybrid` without a reference runs `Pyroomacoustics(ray_tracing=True)` with its default settings (unseeded) and fits the decay to it; `late_envelope` defaults to on for a reference or ray-tracing tail and off for an Eyring or Sabine tail
- `Hybrid` accepts `diffraction` without obstacles (it used to raise `ValueError`): a room without re-entrant edges gives the same response as without diffraction
- Hybrid: the 35 Hz high-pass that keeps DC out of the early field is causal (ported from hybrid-rir a272da7). Applied as a zero-phase magnitude it rang before every arrival, back to the start of the response: around the corner of an L-shaped room the diffracted part held -18 dB (UDFA) and -17 dB (BTMS) of its energy before the direct sound could arrive, now below -60 dB. The magnitude is the same, so spectra, total energy, EDT, T20, T30, Ts and G do not change; values over short windows at the arrival do, most at 125 Hz (DRR over +-2.5 ms, D50)

### Fixed

- `Hybrid`: an image source reached through an edge was counted twice. Where the path to an image passes exactly through an edge, pyroomacoustics' general-room search accepts it from both wall orderings (+6 dB on that path); its `ShoeBox` counts it once. Source and receiver at one height or width, as on a seating grid, hit such edges routinely: 18 to 51 of 99 receivers on a 0.5 m grid in a 6 x 5 x 3 m room, up to 598 extra images and +0.85 dB of image energy at order 4. `hybrid.early.image_sources` now keeps one image per position, and its image set equals `ShoeBox`'s, positions and per-band gains. Deliberately different from hybrid-rir @ a272da7, which keeps the duplicates
- Wheels were missing the materials library, the HRTFs, the sound files and the Grasshopper components.
- `Simulator._pyroomacoustics_simulation` now passes surface scattering coefficients to `pyroomacoustics.Material` correctly; previously the whole material dict was passed as the positional `energy_absorption` argument and scattering was silently dropped (always zero)
- `AcousticObject.display_surface_groups_in_viewer`, `Scene.display_surface_groups_in_viewer`, and `RayField.display_in_viewer` now call `compas.is_rhino()` instead of treating it as a boolean attribute, so the Rhino guard actually raises when appropriate
- `Auraliser` now resolves its default HRTF file via `Defaults.HRTF_LIBRARY_PATH` and ships `mit_kemar_normal_pinna.sofa`; previously it pointed at `KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa`, a file not present anywhere in the repository
- `AcousticAnalyser.speech_transmission_index` now zero-pads IRs shorter than the 1.6 s required by IEC 60268-16:2020 instead of silently returning NaN for STI, and warns when the STI computation fails rather than swallowing the error
- `AcousticAnalyser` EDT, C50, C80, D50, DRR and Ts were biased at low frequencies: the causal order-14 octave filters delay the band-filtered direct sound (about 18 ms at 250 Hz, 35 ms at 125 Hz) past time limits measured from the broadband direct sound (e.g. D50 9 % instead of 34 %, Ts about 50 ms too long, EDT about 15 % too long at 125 Hz). C50, C80, D50 and DRR now cut their time windows from the broadband impulse response before octave filtering; EDT, T20, T30 and Ts use zero-phase (forward-backward) octave filtering with the decay starting at the direct sound, located 20 dB below the peak (ISO 3382-1). Validated against exponential decays, per-band decays, discrete reflections, free-field strength and the analytic STI of an exponential decay
- Grasshopper Analyse IR component analysed simulated impulse responses at the default 48000 Hz instead of their own sample rate (the simulator's 16000 Hz)
- `Auraliser` no longer modifies the caller's impulse response dicts, accepts 1D impulse responses and multichannel sound files (mixed down to mono), keeps the ambisonic order an int after a channel-count mismatch (previously crashed `multiconvolution()`), rejects unknown channel orders and normalisations, re-resamples when `sample_rate` changes, and no longer prints from `maximum_ambisonics_order`
- `Auraliser` resampling no longer wraps impulse response energy around to the start/end (FFT resampling replaced by polyphase) and preserves the impulse response's frequency response
- `Auraliser` mono export used the mean of all ambisonic channels instead of the omnidirectional W channel, stereo export used the front-back (X) dipole instead of the left-right (Y) dipole, and binaural export silently fell back to that stereo; binaural output now raises for mono (order 0) impulse responses
- `Auraliser.export_convolved_sound()` now normalises after decoding, so the exported peak is 1 for every output format
- Grasshopper Auralise component output the undecoded convolution matrix (only saved files were decoded), used a hard-coded 48000 Hz sample rate, and always declared ambisonic order 0
- `Simulator` always simulated at 16000 Hz regardless of `Simulator.sampling_rate`, while its microphone arrays used `sampling_rate`; it now simulates at `sampling_rate` (default 48000 Hz) and the impulse responses carry that rate
- Simulations above 16 kHz with air absorption failed because wall materials kept 7 octave bands while pyroomacoustics uses 8; `pra_converter.material_to_pra()` now resamples materials to the room's octave bands (`room_octave_bands()`), holding the 8 kHz value for the 16 kHz band, and treats missing scattering coefficients as zero
- `pra_converter.receivers_to_pra_mic_array()` read `Defaults.SAMPLING_RATE` from the class, which returned the property object instead of the sample rate
- Grasshopper Simulate Ray Tracing component reports the sample rate of the generated impulse responses
- `Receiver` and `Source` serialise their name, orientation or directivity, and settings (`__data__`), so JSON and `copy()` keep them (they kept only the position and failed to load)
- A source at a receiver's position gave NaN responses; it now raises a `ValueError`
- A source or receiver inside a closed obstacle crashed pyroomacoustics with an unrelated error; scenes of obstacles only simulated nothing; too few walls for pyroomacoustics gave opaque errors (now a `ValueError` saying so)
- The hybrid method failed on a fully absorbing surface (alpha = 1) seen by a whole delay line
- A wall hit on a diffraction apex is no reflection (ported from hybrid-rir 4b42bfd): tilted obstacle rooms lose duplicate zero-length reflections; level rooms are unchanged (checked bit-identical)

### Removed

- `Tools` removed from the public package exports (module still exists but is no longer re-exported)
- `Simulator.resample_room_rir()` removed (replaced by per-IR resampling on `ImpulseResponse`)
- `AcousticObject.surface_groups`, `.acoustic_materials`, `.surface_group_colors`, `.face_colors`, `.display_surface_groups_in_viewer()`, and `.print_surface_groups()` — replaced by single-material `AcousticObject`s, one per surface, `Scene.print_scene_materials()` and `Scene.display_materials_in_viewer()`
- `Scene.scene_surface_groups`, `.print_scene_surface_groups()`, and `.display_surface_groups_in_viewer()` — replaced by `.scene_materials`, `.print_scene_materials()`, and `.display_materials_in_viewer()`
- `compas_acoustics.stage_acoustics` module (ISO 3382-1 Annex A stage parameters `compute_ST1`, `compute_ST2`, `compute_G_stage`, `compute_ITDG`, `stage_support_parameters`), replaced by `AcousticAnalyser.stage_support()`: its `ST2` was ST_late under another name, its values were broadband only, and its ITDG found the floor reflection
- `pra_converter.room_octave_bands`, `material_to_pra`, `mesh_to_pra_wall` and `acoustic_object_to_pra_walls` (replaced by `RoomGeometry.to_pra`); `scene_to_pra_room` delegates to `RoomGeometry`
- The unused `tools.Tools` module
- **Breaking:** `Simulator` removed; use `scene.simulate(method)`. With it go `rir_from_ray_tracing`, `rir_from_image_source_method`, `trace_rays` (now `RayField.from_scene`), `spls_from_image_source_method`, and the `results` and `rirs` attributes (the IR graph holds the responses)
- **Breaking:** `simulation_platform` removed from `Scene` and `ImpulseResponse` (the generation record names the method); old exported metadata that still carries it loads unchanged

