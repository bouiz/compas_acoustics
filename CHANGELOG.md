# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Migration from the previous main

Nothing is kept under its old name. What replaces what:

| Before | Now |
|---|---|
| `Simulator(scene).rir_from_ray_tracing(image_source_order=, n_rays=, air_absorption=, random_seed=)` | `scene.simulate(Pyroomacoustics(ray_tracing=True, image_source_order=, n_rays=, air_absorption=, seed=))` |
| `Simulator(scene).rir_from_image_source_method(order)` | `scene.simulate(Pyroomacoustics(image_source_order=order, ray_tracing=False))` |
| `simulator.sampling_rate = fs` | `Scene(name, sampling_rate=fs)` or `scene.sampling_rate = fs` |
| `simulator.trace_rays(...)` | `RayField.from_scene(scene, ...)` |
| `simulator.rirs`, `simulator.results` | the responses returned by `scene.simulate`, or `scene.impulse_responses(method_name)`, `scene.get_impulse_response(source, receiver)` |
| `Scene(simulation_platform=)`, `ImpulseResponse(simulation_platform=)` | removed; `ir.generation` names the method |
| `AcousticObjectsGroup`; `scene.iter_acoustic_objects()` | `scene.add_acoustic_objects(shoebox_room(...))` (a list); `scene.acoustic_objects` |
| `scene.acoustic_materials = {...}` | `shoebox_room(materials=...)`, `AcousticObject.from_obj(path, materials=...)`: a material per object |
| `AcousticAnalyser.compute_rt`, `compute_clarity_definition`, `compute_drr`, `compute_sti`, `compute_stage_support`, `compute_strength` | `reverberation_times`, `clarity_definition`, `direct_to_reverberant_ratio`, `speech_transmission_index`, `stage_support`, `sound_strength` |
| `stage_acoustics.compute_ST1`, `compute_ST2`, `compute_G_stage`, `compute_ITDG` | `AcousticAnalyser.stage_support()` |
| `Auraliser.multiconvolution()`, `Auraliser.convolve(ir, sound)` | `Auraliser(...).convolve()`, in the format of the impulse responses |
| `Auraliser.decode_stereo_simple()`, `Auraliser.resample_signal()` | no decoder: binaural responses from `Hybrid()`, ambisonic ones saved for a decoder elsewhere; `pyfar.dsp.resample` |
| `Auraliser(output_format=...)` | removed: the signal has the format of its impulse responses, `Auraliser.format` (`mono`, `binaural` or `ambisonic`) |
| `sample_rate` (`ImpulseResponse`, `Auraliser`, `AudioPlayer.play`, `receivers_to_pra_mic_array`) | `sampling_rate` |
| `pra_converter.material_to_pra`, `mesh_to_pra_wall`, `acoustic_object_to_pra_walls` | `RoomGeometry.to_pra`, `pra_converter.scene_to_pra_room` |
| `Tools` | removed |
| `compas_acoustics.orchestra_layouts`; the Orchestra Layout component | `examples/orchestra_layouts.py`; no component |
| `Auraliser`, `AcousticAnalyser`, `MaterialsLibrary` as `Data` | plain classes: no `guid`, `copy()` or `compas.json_dumps` |
| Grasshopper: Simulate Ray Tracing, with a `sample_rate` input | Simulate Pyroomacoustics; the rate is the `sampling_rate` input of Room Scene |
| Grasshopper: Analyse IR output STI | no longer an output, with G: they need pyrato >= 1.0 |
| `Source` direction default +z; `shoebox_room(materials={'left_wall': ..., 'right_wall': ...})` with `left_wall` at y = 0 | direction default +x, as the Grasshopper Source component has it; `left_wall` is at y = width and `right_wall` at y = 0, the sides of a receiver at the centre facing +x (swap the two keys to keep a room) |
| `scene.simulate(hybrid)`, then `hybrid.binaural(hrtf)` | `scene.simulate(Hybrid(output="binaural", hrtf=...))`, stored under `'hybrid_binaural'`; `output=["ambisonic", "binaural"]` for both from one run |
| `Hybrid()` returning ambisonic responses | `Hybrid(output="ambisonic")`; the default output is now binaural |
| `binaural_rir(result, hrtf, filters)` (MagLS-decoded late field) | `binaural_rir(result, hrtf)`: each delay line through its own HRIR |
| `shape_late(late, early, reference, fs)` | `shape_late(signals, early_w, late_w, reference, fs)` |
| Grasshopper: Simulate Ray Tracing output `ir_graph` | Simulate Pyroomacoustics output `mono_ir_graph` |
| Grasshopper: Simulate Hybrid output `ir_graph` (ambisonic) | outputs `binaural_ir_graph` and `ambisonic_ir_graph`; the input `output` (`binaural`, `ambisonic`, `both`) chooses them, default `binaural` |

A Grasshopper definition saved with the old components keeps their old code inside the file, and fails on the removed names. Delete the components
in it and drop the new ones from the toolbar: Room Scene, Receiver, the Simulate components, Deconstruct IR Graph, Analyse IR, Auralise and Audio Player.

### Added

* Added `Scene.simulate(method)` as the one way to simulate, with the methods `Pyroomacoustics(image_source_order=3, ray_tracing=True, ...)` and `Hybrid(...)` in `compas_acoustics.simulation_methods`.
* Added `Scene.sampling_rate` (default 48000, serialised with the scene), the rate at which every method runs; the Room Scene component has the input.
* Added `Hybrid`, ambisonic and binaural room impulse responses (binaural by default) from image sources, optional edge diffraction (`diffraction="udfa"` or `"btms"`) and a directional feedback delay network tail, ported from hybrid-rir; needs Python >= 3.11.
* Added `Hybrid(tail=...)` with a reference response, the hybrid's own ray-traced reference, or an Eyring or Sabine prediction as decay target.
* Added edge diffraction at the edges of the room's own boundary (re-entrant corners) to `Hybrid`.
* Added `Hybrid(scattering=False)`, surface scattering after Ewert et al. (2025) as RAZR 0.96.3 implements it: each image's specular part, the product of `sqrt(1 - s)` over its reflections, stays in the early field, and its scattered part, the product of `sqrt(s)`, enters the delay lines on its own hemisphere through an allpass cascade; checked against RAZR in `tests/hybrid/data/razr_scattering`. Off by default.
* Added `compas_acoustics.binaural` (`HrtfSet`, `render_early_binaural`, `binaural_rir`): the hybrid's early part path by path with the nearest HRTF.
* Added `Hybrid(output="binaural", hrtf=None)`: `output` is `"ambisonic"`, `"binaural"` or a list of both, rendered from one run; `hrtf` is `"kemar"` (the default for a binaural output) or a SOFA file. Binaural responses record `generation["output"] == "binaural"` and the HRTF file in `generation["hrtf"]`, ambisonic ones `generation["output"] == "ambisonic"`.
* Added `HybridResult.late_lines` and `line_directions`: the late field per delay line before the output matrix, and each line's direction in the receiver frame.
* Added `SimulationMethod.attribute` (the IR graph attribute, by default `name`) and `SimulationMethod.extra_responses` (responses in further formats from the same run, by attribute); `Scene.simulate` stores each.
* Added `Pyroomacoustics(ray_tracing=True)`, hybrid-rir's ray tracer on top of the image sources (`n_rays`, `energy_thres`, `duration`, `receiver_radius`, `seed`).
* Added `AirConditions(temperature=20, humidity=50, pressure=101325)`: speed of sound after Cramer (1993) and air absorption after ISO 9613-1, for every method, recorded in `ImpulseResponse.generation`.
* Added `RoomGeometry`, the scene's surfaces as one validated room (welded, outward oriented, closedness checked, coplanar faces merged), with `to_pra()` and `is_inside()`.
* Added `AcousticObject(role="boundary" | "obstacle")` and `AcousticObject.from_obj(path, materials=None)`, a list with one object per OBJ group, every one a boundary; groups `materials` does not cover have none until they are assigned, and `AcousticObject.material` accepts only an `AcousticMaterial` or `None`.
* Added `AcousticObject.assign_roles_by_group(objects, roles)` and `assign_materials_by_group(objects, materials)` to mark OBJ groups as obstacles and give them materials after reading, and `AcousticObject.group`, the OBJ group an object was read from (not serialised); a group name that none of the objects has raises a `ValueError`.
* Added `Receiver.up`, `rotation`, `frame` and `Receiver.from_frame()`: the orientation sets the axes of ambisonic responses (x facing, y left, z up).
* Added `compas_acoustics.diffuse` (`predicted_rt`, `mean_absorption`, `DiffuseFieldWarning`): Sabine and Eyring reverberation times.
* Added JSON serialisation (`compas.json_dumps`) for `Scene`, `AcousticObject`, `Source`, `Receiver`, `ImpulseResponse`, `Ray`, `ScatteredRay` and `RayField`; the IR graph is not serialised.
* Added `Scene.impulse_responses(attribute)`, `Scene.simulation_methods()` and `attribute=` on the IR graph methods: each method stores its responses under its own name.
* Added `ImpulseResponse` with peak normalization (`denormalized()`, `normalization_factor`, `level_db`), `to_wav()`, `from_wav()`, `generation`, `resampled()`, `ambisonic_order`, `normalization`, `channel_order` and `direct_sound_index()`.
* Added `AcousticAnalyser.stage_support()` (ST_early and ST_late, ISO 3382-1 Annex C) and `sound_strength()` (G, needs pyrato >= 1.0), and `G` in the analysis tables.
* Added `AcousticAnalyser` accepting an `ImpulseResponse`, analysed at its raw level and sampling rate.
* Added `Auraliser.format` (`mono`, `binaural` or `ambisonic`, that of its impulse responses, which must share one), `level` (`relative`, `individual`, `none`), `headroom_db`, and lists of impulse responses and sound files as input.
* Added `RayField`, `Ray` and `ScatteredRay` for traced rays (`RayField.from_scene`), with filtering, truncation and viewer display.
* Added `Source.directivity`, `Source.sound_file` (the `Auraliser` mixes a multichannel file down to mono when it reads it), `Receiver.direction`, `radius` and `received_spls`.
* Added `AcousticMaterial.category` and `data_origin`, `MaterialsLibrary.materials_by_category`, preset materials, and `tools/import_backend_materials.py`.
* Added `AudioPlayer` for `.wav` files and numpy arrays, with `normalize=True`.
* Added `compas_acoustics.pra_converter` and `compas_acoustics.numerical.fitness.regularized_fitness`.
* Added a warning before an image source search that will take long (`simulation_methods.image_source_search_seconds()`).
* Added `compas_acoustics.worker`: runs the hybrid method in a second Python >= 3.11 for Rhino 8 (Python 3.9), through a `compas.rpc` server on a free local port with a token per session.
* Added `anechoic_soundfiles_library` (bassoon, female and male speech, trumpet, viola) and `hrtfs_library` (KEMAR, free-field compensated, 48 kHz), with READMEs that credit the sources and licences.
* Added `Defaults.HRTF_LIBRARY_PATH`, `ANECHOIC_SOUNDFILES_LIBRARY_PATH` and `PARAM_NAMES`.
* Added `pyFDN` and `multislope` to the requirements on Python >= 3.11, for the hybrid method (pyFDN brings flamo and PyTorch), and a CI job for it: one `pip install compas_acoustics`, no extra.
* Added the package `compas_acoustics_ghpython` with the Grasshopper components, `AcousticParameter`, `Plotter2D`, `Plotter2DSettings` and `ensure_inputs`, which gives unconnected inputs a value list or a button.
* Added the components Simulate Hybrid (with a `run` button, its last input), Simulate Pyroomacoustics (was Simulate Ray Tracing), Material Library categories, and the inputs `level` (Auralise), `normalize` (Audio Player, Simulate), `method` (Deconstruct IR Graph), `direction`, `up` and `plane` (Receiver).
* Added the input `role` (a value list: `boundary` or `obstacle`) to the Grasshopper Acoustic Object component; a wrong role or material is an error on the component instead of a traceback.
* Added `compas_acoustics_ghpython.MethodGraph`: Simulate Pyroomacoustics and Simulate Hybrid return the scene's IR graph tagged with their method, and Deconstruct IR Graph reads the method its wire comes from (Pyroomacoustics: mono, Hybrid: `hybrid` ambisonic or `hybrid_binaural`), so it has no `method` input.
* Changed the Grasshopper Check Channel component to count the channels of any sound: an audio file path, an impulse response, a Source with a sound file, or an array of samples such as the signals of Auralise (the input is now `sound`).
* Changed the Grasshopper Auralise component to also show the warnings of building the auraliser (impulse responses resampled to a common rate, sounds mixed down to mono), and the Audio Player to remark at which rate it plays an array that comes without one.
* Changed the image source order input of the Grasshopper Simulate Pyroomacoustics (`max_order`) and Simulate Hybrid (`image_source_order`) components to `ism_order`; the Pyroomacoustics description gave the default as 6, the method's default is 3.
* Changed the order of the inputs of the Grasshopper Auralise component to `irs`, `anechoic_sounds`, `level`, `mix_sources`, `save_file_path`, `save_file_name`, `save`.
* Changed the Grasshopper Auralise component to keep the format of its impulse responses, without the inputs `output_format` and `hrtf`: mono responses (Simulate Pyroomacoustics) give mono signals, binaural ones binaural, ambisonic ones the undecoded ACN/SN3D signal, and the file saved has that many channels. Binaural signals come from Simulate Hybrid's binaural output and its `hrtf`. With `mix_sources`, responses of different formats at one receiver are an error. It convolves in Rhino's Python, without the worker.
* Changed the Grasshopper Audio Player component to warn for a signal of more than two channels (an ambisonic one) instead of an error, and `AudioPlayer.play_array` to play a mono signal on both channels.
* Renamed the inputs `signals` and `multiconvolve` of the Grasshopper Auralise component to `anechoic_sounds` and `mix_sources`: `mix_sources` sums the sources at each receiver into one signal.
* Changed the order of the inputs of the Grasshopper Source component to `position`, `name`, `direction`, `directivity`, `sound_file`.
* Changed `AudioPlayer` to read the audio devices again before it plays an array, so that headphones connected after Rhino started are the output.
* Added the input `select` (a value list: `all` or one of EDT, T20, T30, C50, C80, D50, DRR, Ts, STI, G) to the Grasshopper Deconstruct Parameter component: a selected parameter comes out as plain lists, `all` as one branch per parameter. Its `parameters` input takes the list.
* Renamed the output `ir_graph` of the Grasshopper Simulate Pyroomacoustics component to `mono_ir_graph`, after the format of its responses, as Simulate Hybrid's `binaural_ir_graph` and `ambisonic_ir_graph`.
* Added the input `source_idx` to the Grasshopper Deconstruct IR Graph component (inputs `ir_graph`, `source_idx`, `receiver_idx`): with `receiver_idx` it picks source-receiver combinations, e.g. a single impulse response for Analyse Impulse Response. Sources are indexed in the order of the scene's sources; an index out of range is a warning.
* Added `compas_acoustics.analyser.time_response` (level in dB re the peak of all channels, or amplitude) and `frequency_response` (fractional-octave smoothed, in dB re the peak), per channel; levels are floored 100 dB below the peak.
* Added the outputs `time` and `frequency` and the input `time_scale` (a value list: `db` or `linear`) to the Grasshopper Analyse Impulse Response component: the impulse response and its 1/3-octave smoothed frequency response from 20 Hz to 20 kHz, of the W channel of an ambisonic response and both ears of a binaural one. Plot 2D draws them as a curve per channel, on a time axis or a log frequency axis with ticks at 50, 100, 200, 500 ... 20000 Hz, reduced to the minimum and maximum of 1000 bins; a level in dB runs from its floor on the x axis to its peak, an amplitude is symmetric about zero with the x axis at zero. The time output in `linear` ends 5 % after its Schroeder curve falls 60 dB, in `db` it is the whole response, (`time_response(..., trim_db=...)`); the parameters and the frequency output use the whole response. `AcousticParameter.from_time_response` and `from_frequency_response` replace `from_ir`, which Plot 2D could not draw.
* Renamed the output `params` of the Grasshopper Analyse Impulse Response component to `parameters`, and the input `param` of Deconstruct Parameter to `parameters`.
* Renamed the input `param` of the Grasshopper Plot 2D component to `data`: it takes a parameter, or the time or frequency output of Analyse Impulse Response.
* Changed the `hrtf` input of the Grasshopper Simulate Hybrid component to get no value list: empty is KEMAR, and a File Path connected to it is the user's own SOFA set; the component's message names the set in use.
* Changed the `seed` input of the Grasshopper Simulate Pyroomacoustics and Simulate Hybrid components: left empty, each run draws a new random seed (shown in the component's message) instead of the methods' fixed default 42; a number gives repeatable responses. `Pyroomacoustics` and `Hybrid` themselves still default to 42.
* Added a warning to the Grasshopper Simulate Hybrid component when `tail` is `reference` and nothing is connected to `reference`; it shows before Run, and Run does not start the simulation.
* Added the output `parameter` to the Grasshopper Deconstruct Parameter component: the selected AcousticParameter, which Plot 2D plots and a panel prints (`AcousticParameter.__str__` gives the name, unit, value per band and overall value).
* Added STI and G to the Grasshopper Analyse IR component through the worker (`compas_acoustics.worker.analyse`): they need pyrato >= 1.0, which Rhino 8's Python 3.9 cannot install, so the worker computes them and Analyse IR leaves them out without one.
* Changed the Grasshopper Plot 2D chart to leave half a band's spacing between the y axis and the first frequency band.
* Added the inputs `output` (a value list: `binaural`, `ambisonic` or `both`) and `hrtf` (`kemar` or a SOFA path) and the outputs `binaural_ir_graph` and `ambisonic_ir_graph` (`None` for a format not requested) to the Grasshopper Simulate Hybrid component, which replace its output `ir_graph`; its status counts the responses of the requested formats.
* Changed `compas_acoustics.worker.simulate` to store every format of a run on the scene's graph under its attribute and return the first, as `Scene.simulate` does; it used to store whatever came back under the method's name.
* Added a status to the Grasshopper Simulate Hybrid component for when it is not running: it says `done` with the tail and order of the last result on the graph, `settings changed, press Run` when its inputs differ from that result, and `press Run` only when there is none.
* Added the Grasshopper installer for Rhino 8 (`installer/compas_acoustics_installer.gh`), `invoke installer-bundle`, and `compas-dev/compas-actions` v1 workflows (build, release, `prepare release`), with tests on Python 3.9.
* Added `examples/office_hybrid.py`, which runs both methods on an office.
* Added a getting-started section to the README, and `Examples` sections that `invoke testdocs` can run in the docstrings of `Scene`, `Pyroomacoustics`, `Hybrid`, `AcousticAnalyser` and `Auraliser`.

### Changed

* Changed `Hybrid` with a binaural output to use at least 24 delay lines when `num_delays` is not given (`BINAURAL_MIN_DELAYS`, RAZR 0.96.3's default): with 16, the late field's octave-band interaural coherence was up to 0.18 above the KEMAR set's diffuse field, with 24 up to 0.04.
* Changed the spelling of identifiers to `normalization` and `normalized` everywhere, including `ImpulseResponse.normalization` (the ambisonic convention) and its JSON metadata key.
* Changed `AudioPlayer` to raise errors and `warnings.warn` instead of printing them; the Audio Player component shows them as messages. An unsupported sound type raises a `TypeError`.
* Changed `seed` of `Pyroomacoustics` and `Hybrid` (was 0) to `True` by default, which seeds with 42 (`DEFAULT_SEED`); an int seeds with that value; `False` (or `None`) leaves the run unseeded, a new draw each run, recorded as `generation["seed"] = None`. Anything else raises a `ValueError`. `RayField.from_scene(random_seed=)` follows the same rule.
* Changed `Hybrid` to drive the network with the last rendered image source order, as Wendt et al. (2014), RAZR and Ewert et al. (2025) do; hybrid-rir used the next, unrendered order. The search stops at `image_source_order` (29 minutes became 14 seconds on BRAS CR2).
* Changed `Hybrid` without a reference to run `Pyroomacoustics(ray_tracing=True, image_source_order=3, n_rays=100000, seed=42)` and fit the decay to it.
* Changed the hybrid's 35 Hz high-pass to a causal filter, which removes pre-arrival ringing in the diffracted paths.
* Changed every simulation to run in `AirConditions()` (20 C, 50 %, 343.98 m/s) unless `air=` is given; before, the methods used different air tables.
* Changed simulated impulse responses to 48000 Hz (were always 16000 Hz) and to the package scale (free-field direct sound `1 / (4 pi r)`).
* Changed `AcousticObject.__init__` to take one `material`; `AcousticObjectsLibrary.shoebox_room()` returns a list of six single-material objects and takes `materials`.
* Changed `Source.directional` to `Source.directivity`, and `Source.direction` and `Receiver.direction` to `compas.geometry.Vector`.
* Changed `sample_rate` to `sampling_rate` everywhere: `ImpulseResponse`, `Auraliser`, `AudioPlayer.play`, `receivers_to_pra_mic_array`, `HrtfSet` and the Grasshopper components.
* Changed the `AcousticAnalyser` methods to name the quantity: `reverberation_times`, `clarity_definition`, `direct_to_reverberant_ratio`, `speech_transmission_index`, `stage_support`, `sound_strength`.
* Changed the analyser to apply octave filters zero-phase or after cutting the time windows, which removes a delay of 18 ms at 250 Hz and 35 ms at 125 Hz from EDT, Ts, C50, C80, D50 and DRR.
* Changed `direct_to_reverberant_ratio` to Haapaniemi and Lokki (2018): 5 ms of direct sound, `DRR_overall` from a 700-4000 Hz band.
* Changed `reverberation_times` to Schroeder integration without noise compensation; truncate a measured response with a noise floor before analysing it.
* Changed the direct sound detection to the ITA Toolbox's `ita_start_IR` (ISO 3382-1).
* Changed `Auraliser.convolve()` to return the signal in the format of its impulse responses, without decoding, summing the impulse responses of one sound first (136 s became 1 s for 10 third-order sources).
* Changed `Auraliser` to auralise at the raw level, export peak at -1 dBFS, and use an impulse response's own `channel_order` and `normalization`.
* Changed `Auraliser` to read and resample with pyfar (polyphase) and to default to the rate of the first impulse response.
* Changed `AcousticAnalyser`, `Auraliser` and `MaterialsLibrary` to plain classes; they are services, not data.
* Changed the default HRTF to `KEMAR_FreeFieldComp_48kHz.sofa`.
* Changed a material without scattering coefficients to scatter 0.05 in every band in every pyroomacoustics simulation.
* Changed `AcousticMaterial` so that a coefficient of 0 is a value, and material names may have spaces and capitals; library files are named by `AcousticMaterial.slug` and carry no backend suffix.
* Changed sources and receivers outside the room to raise a `UserWarning`, and surfaces with `role="obstacle"` to reflect from both sides.
* Changed `Scene.acoustic_materials` to read-only; assigning it raises an `AttributeError` that says how materials are assigned.
* Changed `Hybrid` to accept `diffraction` without obstacles, and to report non-closed obstacle components in one warning per call.
* Changed the package to run on Python 3.9 (Rhino 8) without Rhino, Grasshopper or .NET; Rhino code is in `compas_acoustics_ghpython`.
* Changed `pyrato >= 1.0` to be required from Python 3.11 (was 3.12), and removed the upper bounds on pyroomacoustics and pyFDN.
* Changed the Analyse IR component to leave out STI and G, which need pyrato >= 1.0; they stay in `AcousticAnalyser` where pyrato allows.
* Changed `matplotlib` to be imported by the `AcousticAnalyser` plot methods only when they run.
* Changed the Grasshopper components to use ruff per-file ignores instead of `# flake8: noqa`, and the installer's `worker` input to `hybrid`.
* Changed `librosa` out of the dependencies and `sounddevice` into them; `spharpy` is required on Python >= 3.11.
* Changed `RoomGeometry` to weld vertices at 1 mm (`WELD_TOLERANCE`, was 1 micrometre), as Rhino's tolerance in a metre model: a wall corner 10 micrometres off no longer leaves the room open. A vertex still lies on an edge only within 1e-6 of the model's extent, as in hybrid-rir and about pyroomacoustics' 1e-5 m. Faces with a corner more than 10 micrometres off their plane are triangulated (pyroomacoustics raised a `RuntimeError` on them), faces that collapse in welding are dropped with a warning, a model more than 500 units across warns that coordinates are read as metres, and a wall pyroomacoustics refuses is a `ValueError` naming the face.
* Changed `AcousticObject.from_obj` to read OBJ files with its own parser: `o` (object) lines are the groups of a file without `g` lines, faces may use `i/j/k` and negative indices, vertices are no longer rounded (`RoomGeometry` welds them), and faces with fewer than three distinct vertices are skipped.
* Changed the default `Source` direction to +x (was +z), the direction the Grasshopper Source component already defaults to and the receiver's facing direction.
* Changed `AcousticObjectsLibrary.shoebox_room` so that `left_wall` is the surface at y = width and `right_wall` the one at y = 0, matching the receiver frame (x facing, y left); the two names were swapped. The surfaces come in the order floor, right, front, left and back wall, ceiling.
* Changed the error of `Receiver` for a name with spaces or capitals to say `Receiver` (it said `Source`).
* Changed `RoomGeometry.surfaces`, `surface_area`, `area_normals` and `volume` to count the air side only: a closed obstacle counts one side and its volume is subtracted, and obstacle faces lying on the boundary are left out, so Eyring and Sabine times, the default ray-tracing duration and the hybrid's delays and scattering decay time move in scenes with closed obstacles or obstacles on the boundary; a free panel still counts both sides.
* Changed `AcousticMaterial` to validate coefficients on creation: numpy arrays and scalars become lists of float, and a length other than 7 or a value outside [0, 1] raises `ValueError`.
* Changed the default output of `Hybrid` to binaural (`output="binaural"`), stored under `'hybrid_binaural'`; `Hybrid(output="ambisonic")` gives the ambisonic responses under `'hybrid'` as before, unchanged up to float rounding.
* Changed the hybrid's binaural late field to be rendered per delay line, each line `/ sqrt(N)` through the HRIR measured nearest its direction, as RAZR and RAVEN render their feedback delay networks; it was the ambisonic late field MagLS-decoded at `sh_order`, so a binaural response no longer needs `sh_order >= 1`.
* Changed `hybrid.late.render` to return the delay lines' outputs, `(n, num_delays)`, after the directional absorption and slope weights; the ambisonic late field is that times `C.T`.
* Changed `Scene.simulate` to store a method's responses under `method.attribute` and its `extra_responses` under theirs; `'data'` still holds a mono method's only.

### Fixed

* Fixed `RoomGeometry` taking a vertex up to 1 mm off an edge for a T-junction, which left rooms with sliver faces open (BRAS CR2): the edge test is 1e-6 of the model's extent again.
* Fixed the single-number values of C50, C80, D50 and Ts averaging 500, 1000 and 2000 Hz: they average 500 and 1000 Hz, as ISO 3382-1:2009 Table A.1 has it for G, EDT, C80, D50 and Ts (C50 follows D50), like EDT, T20 and T30 already did.
* Fixed responses simulated in the worker (the hybrid in Rhino) always coming back as peak-normalized views: they keep the method's `normalize`. Their raw level was already kept.
* Fixed the Grasshopper Material Library component forgetting its material when the file is reopened: the choice is kept in the component's nickname, which Grasshopper saves.
* Fixed the Grasshopper Material Library component failing with `No module named 'compas_acoustics'`: its script pinned an empty Rhino script environment, `compas_acoustics_rhino`, instead of the one the installer fills.
* Fixed the worker setup from the Grasshopper installer failing at its last check: the worker's Python 3.12 inherited the `PYTHONPATH` of Rhino's 3.9 packages and could not import numpy.
* Fixed a simulation failing on a Source whose `sound_file` is not a RIFF WAV (an AIFF or FLAC file; the library's `basoon.wav` is an AIFF): the file is read with soundfile.
* Fixed `Hybrid` counting an image source twice when its path passes through an edge (+6 dB on that path); `hybrid.early.image_sources` now keeps one image per position, as `ShoeBox` does.
* Fixed low-frequency bias in EDT, C50, C80, D50, DRR and Ts caused by the delay of causal octave filters.
* Fixed `speech_transmission_index` returning NaN for impulse responses shorter than 1.6 s.
* Fixed wheels that were missing the materials library, the HRTFs, the sound files and the Grasshopper components.
* Fixed `Auraliser` modifying the caller's dicts, ignoring 1D responses and multichannel sounds, and wrapping energy around when resampling.
* Fixed simulations above 16 kHz with air absorption, which failed on a 7 against 8 octave band mismatch.
* Fixed `Receiver` and `Source` losing name, orientation, directivity and settings in JSON and `copy()`.
* Fixed NaN responses for a source at a receiver's position (now a `ValueError`), and unrelated errors for points inside a closed obstacle, scenes of obstacles only, or too few walls.
* Fixed the hybrid failing on a fully absorbing surface, and counting a wall hit on a diffraction apex as a reflection.
* Fixed `display_in_viewer` guards that treated `compas.is_rhino()` as a boolean, and the default HRTF path that pointed at a missing file.
* Fixed the Grasshopper Auralise and Analyse IR components using a fixed sample rate.
* Fixed `AcousticObject.from_obj` reading OBJ files in the platform's encoding: a UTF-8 byte order mark dropped the first vertex, and non-ASCII group names were garbled on Windows; files are read as UTF-8.
* Fixed obstacle faces lying on a boundary face being counted when the scene is not aligned with the axes: the test used the distance along each axis instead of the distance to the face's plane.

### Removed

* Removed `Simulator`; use `Scene.simulate(method)` and `RayField.from_scene`.
* Removed `simulation_platform` from `Scene` and `ImpulseResponse`; `generation` names the method.
* Removed `Pyroomacoustics(n_jobs=)` and the parallel ray tracer, which wrote private `Room` state.
* Removed `AcousticObjectsGroup`, `AcousticObject.surface_groups`, `Scene.iter_acoustic_objects`, `Scene.scene_surface_groups` and their display and print methods; see `scene_materials` and `print_scene_materials`.
* Removed `compas_acoustics.stage_acoustics` and `tools.Tools`; use `AcousticAnalyser.stage_support()`.
* Removed `pra_converter.material_to_pra`, `mesh_to_pra_wall`, `acoustic_object_to_pra_walls` and `room_octave_bands`; see `RoomGeometry.to_pra`.
* Removed `Auraliser.multiconvolution()`, `decode_stereo_simple()`, `resample_signal()` and FuMa channel order.
* Removed `compas_acoustics.orchestra_layouts` and the Orchestra Layout component; the layouts are now `examples/orchestra_layouts.py`.
* Removed the two MIT KEMAR HRTF sets and `speech.wav`.
* Removed `Hybrid.binaural()`; use `Hybrid(output="binaural")` or `output=["ambisonic", "binaural"]`.
* Removed the ambisonic decoders of `Auraliser` (mono from W, stereo cardioids and MagLS binaural: `output_format`, `decode`, `decode_mono`, `decode_stereo`, `decode_binaural`, `hrtf_path`, `binaural_filters`, `ambisonics_upmix_downmix`), `binaural.magls_filters`, `worker.convolve` and the installer's `worker` option `binaural`. The hybrid renders binaural responses with an HRTF per path and per delay line; an ambisonic response or signal is saved as ACN/SN3D for a decoder elsewhere. Impulse responses of different formats in one `Auraliser` raise a `ValueError`.
