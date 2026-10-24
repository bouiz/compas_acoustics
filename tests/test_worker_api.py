"""The worker API the Grasshopper plugin calls, in-process; every argument and result crosses compas_pb as it would the wire."""

import sys

import numpy as np
import pytest
from compas.datastructures import Mesh
from compas.geometry import Frame
from compas.geometry import Polyline
from compas_pb import pb_dump_bts
from compas_pb import pb_load_bts

import compas_acoustics.pb  # noqa: F401
from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import MaterialCategory
from compas_acoustics import MaterialsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.pb import plain
from compas_acoustics.worker import api


def wire(value):
    # the service wraps every payload in a dict: compas_pb refuses an empty or zero top-level value
    return pb_load_bts(pb_dump_bts({"value": plain(value)}))["value"]


def call(function, *args, **kwargs):
    """``api.<function>`` with its arguments and result sent through compas_pb, as the service does."""
    args, kwargs = wire(list(args)), wire(kwargs)
    return wire(getattr(api, function)(*args, **kwargs))


def box_mesh():
    return Mesh.from_vertices_and_faces(
        [[0, 0, 0], [7, 0, 0], [7, 5, 0], [0, 5, 0], [0, 0, 3], [7, 0, 3], [7, 5, 3], [0, 5, 3]],
        [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]],
    )


# ---------------------------------------------------------------------------
# Task 4: materials and model makers
# ---------------------------------------------------------------------------


def test_every_function_the_service_serves_is_listed():
    for name in ("materials_library", "material_from_name", "make_material", "make_source", "make_receiver",
                 "make_acoustic_object", "make_scene"):
        assert name in api.__all__


def test_the_library_lists_every_material_under_its_category():
    library = MaterialsLibrary()
    listed = call("materials_library")
    assert [category for category, _ in listed] == [category.value for category in library.materials_by_category]
    names = [name for _, group in listed for name in group]
    assert sorted(names) == sorted(library.material_names)
    material = call("material_from_name", names[0])
    assert isinstance(material, AcousticMaterial) and material.name == names[0]


def test_an_unknown_material_name_is_a_value_error():
    with pytest.raises(ValueError, match="No material 'nope' in the materials library"):
        api.material_from_name("nope")


def test_make_material_validates_like_the_class():
    material = call("make_material", "m", [0.1] * 7, [0.05] * 7, "Wood")
    assert material.category is MaterialCategory.WOOD and list(material.scattering_coefficients) == [0.05] * 7
    with pytest.raises(ValueError):
        api.make_material("m", [-0.1] * 7)


def test_make_source_defaults_match_the_component():
    source = call("make_source", None, [1, 2, 3])
    assert isinstance(source, Source)
    assert source.name == "my_source" and list(source.position) == [1, 2, 3]
    assert list(source.direction) == [1, 0, 0] and source.directivity == "omnidirectional"
    given = call("make_source", "s", [0, 0, 1], [0, 1, 0], "cardioid")
    assert given.name == "s" and list(given.direction) == [0, 1, 0] and given.directivity == "cardioid"


def test_make_receiver_defaults_to_room_aligned_at_one_metre():
    result = call("make_receiver", None)
    receiver = result["receiver"]
    assert isinstance(receiver, Receiver) and receiver.name == "my_receiver" and list(receiver.position) == [0, 0, 1]
    assert set(result) == {"receiver", "frame_overrides"}


def test_make_receiver_takes_a_direction_and_stays_upright():
    receiver = call("make_receiver", "r", [1, 2, 3], [0, 1, 0])["receiver"]
    assert np.allclose(list(receiver.direction), [0, 1, 0]) and np.allclose(list(receiver.up), [0, 0, 1])


def test_make_receiver_has_no_up():
    import inspect

    assert list(inspect.signature(api.make_receiver).parameters) == ["name", "position", "direction", "frame"]


def test_make_receiver_from_a_frame_faces_its_x_axis():
    result = call("make_receiver", "r", frame=Frame([1, 2, 3], [0, 1, 0], [-1, 0, 0]))
    receiver = result["receiver"]
    assert np.allclose(list(receiver.direction), [0, 1, 0]) and np.allclose(list(receiver.up), [0, 0, 1])
    assert list(receiver.position) == [1, 2, 3]
    assert result["frame_overrides"] is False


def test_a_frame_with_other_inputs_says_they_are_not_used():
    result = call("make_receiver", "r", position=[0, 0, 0], frame=Frame([1, 2, 3], [0, 1, 0], [-1, 0, 0]))
    assert result["frame_overrides"] is True


def test_make_receiver_facing_up_is_a_value_error():
    with pytest.raises(ValueError, match="parallel"):
        api.make_receiver("r", [0, 0, 1], direction=[0, 0, 1])


def test_make_acoustic_object_from_a_mesh_and_from_a_polyline():
    material = AcousticMaterial("wall", [0.2] * 7)
    obj = call("make_acoustic_object", box_mesh(), material, "box")
    assert isinstance(obj, AcousticObject) and obj.role == "boundary" and obj.name == "box"
    assert obj.acoustic_geometry.number_of_faces() == 6 and obj.material.name == "wall"
    line = call("make_acoustic_object", Polyline([[0, 0, 0], [1, 0, 0], [1, 1, 0]]), material, None, "obstacle")
    assert isinstance(line.acoustic_geometry, Polyline) and line.role == "obstacle"


def test_make_acoustic_object_rejects_other_geometry_and_roles():
    material = AcousticMaterial("wall", [0.2] * 7)
    with pytest.raises(TypeError, match="Unsupported geometry type"):
        api.make_acoustic_object(Frame.worldXY(), material)
    with pytest.raises(ValueError):
        api.make_acoustic_object(box_mesh(), material, role="ceiling")


def test_make_scene_rejects_a_foreign_item_naming_it():
    with pytest.raises(TypeError, match=r"objects\[0\] is a Source"):
        api.make_scene([Source("s", 0, 0, 1)], [], [])
    with pytest.raises(TypeError, match=r"receivers\[1\] is a Source"):
        api.make_scene([], [], [Receiver("r", 0, 0, 1), Source("s", 0, 0, 1)])


def test_make_scene_round_trips_through_compas_pb():
    objects = AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", [0.2] * 7))
    scene = call("make_scene", objects, [Source("src", 1.5, 1.5, 1.6)], [Receiver("rec", 5, 3, 1.5)], "room", 44100)
    assert isinstance(scene, Scene) and scene.name == "room" and scene.sampling_rate == 44100
    assert len(scene.acoustic_objects) == 6 and [s.name for s in scene.sources] == ["src"]


def test_make_scene_keeps_no_name_as_none_and_validates_the_rate():
    assert call("make_scene", [], [], []).name is None
    with pytest.raises(ValueError, match="sampling_rate"):
        api.make_scene([], [], [], sampling_rate=-1)


# ---------------------------------------------------------------------------
# Task 5: analysis, auralise, play, stop
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

import compas_acoustics  # noqa: E402
from compas_acoustics import ImpulseResponse  # noqa: E402
from compas_acoustics.pb.types import Audio  # noqa: E402

VIOLA = str(Path(compas_acoustics.__file__).parent / "anechoic_soundfiles_library" / "viola.wav")


def decay(channels=1, seconds=0.6, rt=0.5, rate=48000, seed=0, level=1.0, source=None, receiver=None):
    """An exponentially decaying noise response with a direct sound."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    envelope = np.exp(-6.9 * np.arange(n) / rate / rt)
    raw = rng.normal(size=(n, channels)) * envelope[:, None] * 0.1
    raw[100] = 1.0
    raw *= level
    source = source or Source("src", 1, 1, 1.5)
    source._sound_file = VIOLA
    return ImpulseResponse("ir", source, receiver or Receiver("rec", 4, 3, 1.5), raw if channels > 1 else raw[:, 0], rate,
                           generation={"method": "test"})


def test_analysis_gives_every_parameter_with_its_units_and_the_two_curves():
    result = call("analysis", decay())
    names = [p["name"] for p in result["parameters"]]
    # STI and G are left out where they cannot be computed (pyrato < 1.0; G needs the simulation's level), LF and LFC
    # for a response that is not ambisonic
    optional = ("STI", "G", "SPL", "LF", "LFC")
    assert [n for n in api.PARAMETERS if n not in optional] == [n for n in names if n not in optional]
    t30 = next(p for p in result["parameters"] if p["name"] == "T30")
    assert t30["title"] == "Reverberation Time (T30)" and t30["x_unit"] == "Frequency (Hz)" and t30["y_unit"] == "Time (s)"
    assert t30["x_scale"] == "bands" and len(t30["values"]) == 1 and len(t30["values"][0]) == len(t30["frequencies"])
    assert isinstance(t30["overall"], float)
    time, frequency = result["time"], result["frequency"]
    assert time["x_scale"] == "linear" and time["y_unit"] == "Level (dB)" and time["channels"] == []
    assert frequency["x_scale"] == "log" and frequency["frequencies"][0] >= 20 and frequency["frequencies"][-1] <= 20000
    assert len(time["values"][0]) == len(time["frequencies"])
    assert isinstance(result["warnings"], list)


def test_analysis_of_a_binaural_response_plots_both_ears():
    result = call("analysis", decay(channels=2))
    assert result["time"]["channels"] == ["left", "right"] and len(result["time"]["values"]) == 2
    assert result["time"]["title"] == "Impulse Response (left, right)"


def test_analysis_of_an_ambisonic_response_plots_w():
    result = call("analysis", decay(channels=4))
    assert result["frequency"]["channels"] == ["W"] and len(result["frequency"]["values"]) == 1


def test_analysis_linear_time_scale_ends_60_db_down():
    db = call("analysis", decay(seconds=2.0), "db")["time"]
    linear = call("analysis", decay(seconds=2.0), time_scale="linear")["time"]
    assert linear["y_unit"] == "Amplitude" and len(linear["frequencies"]) < len(db["frequencies"])


def test_analysis_rejects_an_unknown_time_scale():
    with pytest.raises(ValueError, match="db.*linear"):
        api.analysis(decay(), "log")


def group(irs, branch="{0}", sounds=None):
    return {"branch": branch, "irs": irs, "sounds": sounds if sounds is not None else [None] * len(irs)}


def two_sources_one_receiver(**kwargs):
    receiver = Receiver("rec", 4, 3, 1.5)
    return [decay(source=Source("a", 1, 1, 1.5), receiver=receiver, level=1.0, **kwargs),
            decay(source=Source("b", 2, 1, 1.5), receiver=receiver, level=0.25, seed=1, **kwargs)]


def test_auralise_renders_one_signal_per_source_and_receiver():
    result = call("auralise", [group(two_sources_one_receiver(), "{0;1}")])
    signals = result["signals"]
    assert [s["suffix"] for s in signals] == ["0_0", "0_1"]
    assert all(s["branch"] == "{0;1}" for s in signals)
    assert all(isinstance(s["audio"], Audio) and s["audio"].format == "mono" for s in signals)
    assert signals[0]["audio"].sampling_rate == 48000


def test_auralise_mixes_the_sources_of_a_receiver():
    result = call("auralise", [group(two_sources_one_receiver())], mix_sources=True)
    assert [s["suffix"] for s in result["signals"]] == ["0"]


def test_auralise_relative_level_applies_one_gain_to_all():
    result = call("auralise", [group(two_sources_one_receiver())], "relative")
    assert result["peak_dbfs"] == pytest.approx(-1.0, abs=0.05)
    loud, quiet = (np.max(np.abs(s["audio"].samples)) for s in result["signals"])
    individual = call("auralise", [group(two_sources_one_receiver())], "individual")
    assert [np.max(np.abs(s["audio"].samples)) for s in individual["signals"]] == pytest.approx([10 ** (-1 / 20)] * 2, rel=1e-3)
    assert quiet < loud


def test_auralise_takes_the_given_sounds():
    irs = two_sources_one_receiver()
    result = call("auralise", [group(irs, sounds=[VIOLA, VIOLA])])
    assert len(result["signals"]) == 2


def test_auralise_refuses_mixed_formats_at_one_receiver():
    receiver = Receiver("rec", 4, 3, 1.5)
    irs = [decay(receiver=receiver), decay(channels=2, receiver=receiver, source=Source("b", 2, 1, 1.5))]
    with pytest.raises(ValueError, match="mix responses of one format"):
        api.auralise([group(irs)], mix_sources=True)


def test_auralise_refuses_an_unknown_level():
    with pytest.raises(ValueError, match="relative"):
        api.auralise([group(two_sources_one_receiver())], "loud")


def test_auralise_saves_with_the_components_naming(tmp_path):
    call("auralise", [group(two_sources_one_receiver())], save_folder=str(tmp_path), save_name="take")
    assert sorted(p.name for p in tmp_path.glob("*.wav")) == ["take_mic_0_0.wav", "take_mic_0_1.wav"]
    call("auralise", [group(two_sources_one_receiver())], mix_sources=True, save_folder=str(tmp_path))
    assert (tmp_path / "convolved_sound_mic_0.wav").exists()


def test_play_refuses_a_missing_file():
    with pytest.raises(ValueError, match="does not exist"):
        api.play("/nowhere/sound.wav")


def test_play_refuses_more_than_two_channels():
    with pytest.raises(ValueError, match="4-channel ambisonic signal cannot be played"):
        api.play(Audio(np.zeros((10, 4)), 48000, "ambisonic"))


def test_play_and_stop_reach_the_player(monkeypatch):
    calls = []

    class Player:
        duration = 0.0

        def play(self, sound, sampling_rate=None, normalize=True, gain_db=0.0, start=0.0):
            calls.append(("play", type(sound).__name__, sampling_rate, normalize))

        def stop(self):
            calls.append(("stop",))

    monkeypatch.setattr(api, "_PLAYER", Player())
    assert call("play", Audio(np.zeros((10, 2)), 44100, "binaural"), False) == {"warnings": [], "duration": 0.0}
    call("play", VIOLA)
    call("stop")
    assert calls == [("play", "ndarray", 44100, False), ("play", "str", None, True), ("stop",)]


# ---------------------------------------------------------------------------
# Task 6: rays, reflections, fitness
# ---------------------------------------------------------------------------

from compas_acoustics import RayField  # noqa: E402
from compas_acoustics import SpecularReflections  # noqa: E402
from compas_acoustics import regularized_fitness  # noqa: E402


def shoebox(source=(1.5, 1.5, 1.6), sources=1):
    scene = Scene("room")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", [0.2] * 7, [0.3] * 7)))
    for i in range(sources):
        scene.add_source(Source(f"s{i}", source[0] + i, source[1], source[2]))
    scene.add_receiver(Receiver("rec", 5, 3, 1.5))
    return scene


def test_ray_field_traces_and_truncates():
    result = call("ray_field", shoebox(), n_rays=50, max_order=4, order=2)
    field, shown = result["field"], result["shown"]
    assert isinstance(field, RayField) and isinstance(shown, RayField)
    assert len(field.rays) == len(shown.rays) == 50
    assert all(len(ray.points) <= 3 for ray in shown.rays) and any(len(ray.points) > 3 for ray in field.rays)
    assert isinstance(result["seed"], int)
    assert call("ray_field", shoebox(), n_rays=5, seed=7)["seed"] == 7


def test_ray_field_scatters_with_a_seed_and_traces_one_source():
    result = call("ray_field", shoebox(sources=2), source_index=1, n_rays=20, max_order=3, scattering=True, seed=3)
    assert result["field"].scattered_rays
    assert {ray.source_index for ray in result["field"].rays} == {1}


def test_ray_field_emits_into_a_cone():
    result = call("ray_field", shoebox(), n_rays=30, max_order=1, direction=[1, 0, 0], angle=20)
    for ray in result["field"].rays:
        assert ray.direction[0] > 0.9


def test_ray_field_warns_for_a_source_index_out_of_range():
    with pytest.raises(ValueError, match=r"No source at index 3; the scene has 1 source\(s\), 0 to 0\."):
        api.ray_field(shoebox(), source_index=3)


def test_ray_field_warns_outside_the_room():
    assert call("ray_field", shoebox(source=(20, 1, 1)), n_rays=5, max_order=1)["warnings"]


def test_specular_reflections_first_order_of_every_pair():
    result = call("specular_reflections", shoebox())
    assert isinstance(result["reflections"], SpecularReflections)
    assert len(result["reflections"].paths) == 6
    assert result["counts"] == [[1, 6]] and result["orders"] == [1]
    assert all(len(path.points) == 3 and path.travel_time > 0 for path in result["reflections"].paths)


def test_specular_reflections_pick_orders_and_leave_out_negative_ones():
    result = call("specular_reflections", shoebox(), [2, -1, 0, 2])
    assert result["orders"] == [0, 2]
    assert {path.order for path in result["reflections"].paths} == {0, 2}
    assert "Orders below 0 are left out: -1." in result["warnings"]


def test_specular_reflections_warn_for_an_index_out_of_range():
    result = call("specular_reflections", shoebox(), [1], receiver_index=4)
    assert "No receiver at index 4; the scene has 1 receiver(s), 0 to 0." in result["warnings"]


def test_regularized_fitness_matches_the_function():
    assert call("regularized_fitness", [1, 2], 1.5, 1.0) == regularized_fitness([1, 2], 1.5, 1.0)
    assert call("regularized_fitness", [1.5, 1.5], 1.5) == 0.0


# ---------------------------------------------------------------------------
# Task 15: what Check Channel asks about a file
# ---------------------------------------------------------------------------


def test_sound_info_reads_channels_and_rate_of_any_soundfile_format(tmp_path):
    import soundfile as sf

    path = tmp_path / "s.flac"
    sf.write(str(path), np.zeros((100, 3)), 22050)
    assert call("sound_info", str(path)) == {"channels": 3, "sampling_rate": 22050, "frames": 100}
    assert call("sound_info", VIOLA)["channels"] >= 1


def test_sound_info_of_a_missing_or_unreadable_file_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        api.sound_info(str(tmp_path / "none.wav"))
    (tmp_path / "text.wav").write_text("not audio")
    with pytest.raises(ValueError, match="Cannot read"):
        api.sound_info(str(tmp_path / "text.wav"))


# ---------------------------------------------------------------------------
# Playback control (Grasshopper UI plan, Task 2)
# ---------------------------------------------------------------------------


class FakePlayer:
    def __init__(self):
        self.calls = []
        self.playing, self.paused, self.position, self.duration = False, False, 0.0, 0.0

    def play(self, sound, sampling_rate=None, normalize=True, gain_db=0.0, start=0.0):
        self.calls.append(("play", type(sound).__name__, sampling_rate, normalize, gain_db, start))
        self.playing, self.position = True, start
        self.duration = 2.0

    def pause(self):
        self.calls.append(("pause",))
        self.playing, self.paused = False, True

    def resume(self):
        self.calls.append(("resume",))
        self.playing, self.paused = True, False

    def stop(self):
        self.calls.append(("stop",))
        self.playing, self.paused, self.position = False, False, 0.0


def test_play_takes_gain_and_start_and_reports_the_duration(monkeypatch):
    player = FakePlayer()
    monkeypatch.setattr(api, "_PLAYER", player)
    result = call("play", Audio(np.zeros((10, 2)), 44100, "binaural"), gain_db=-6.0, start=0.25)
    assert result == {"warnings": [], "duration": 2.0}
    assert player.calls == [("play", "ndarray", 44100, True, -6.0, 0.25)]


def test_pause_resume_stop_and_status_reach_the_player(monkeypatch):
    player = FakePlayer()
    monkeypatch.setattr(api, "_PLAYER", player)
    call("play", Audio(np.zeros((10, 2)), 44100, "binaural"))
    call("pause")
    assert call("playback_status") == {"playing": False, "paused": True, "position": 0.0, "duration": 2.0}
    call("resume")
    call("stop")
    assert [c[0] for c in player.calls] == ["play", "pause", "resume", "stop"]


def test_sound_samples_reads_a_file_as_audio(tmp_path):
    import soundfile as sf

    path = tmp_path / "s.wav"
    sf.write(str(path), np.full((20, 2), 0.25), 8000)
    audio = call("sound_samples", str(path))
    assert isinstance(audio, Audio) and audio.format == "binaural" and audio.sampling_rate == 8000
    assert audio.samples.shape == (20, 2) and np.allclose(audio.samples, 0.25, atol=1e-4)
    with pytest.raises(ValueError, match="does not exist"):
        api.sound_samples(str(tmp_path / "none.wav"))


def test_the_worker_leaves_lf_out_for_a_mono_response():
    names = [p["name"] for p in call("analysis", decay())["parameters"]]
    assert "LF" not in names and "LFC" not in names
    assert list(api.PARAMETERS)[-2:] == ["LF", "LFC"]
    assert api.PARAMETERS["LF"] == ("Lateral Energy Fraction (LF)", "Frequency (Hz)", "Fraction (0-1)")


def test_make_source_takes_the_level_at_1_m():
    assert call("make_source", "s", [0, 0, 0], spl=[65.0]).spl == [65.0] * 7
    assert call("make_source", "s", [0, 0, 0], spl=list(range(60, 67))).spl == [60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0]
    assert call("make_source", "s", [0, 0, 0]).spl == Source("x", 0, 0, 0).spl


def _map_scene():
    from compas_acoustics import AcousticMaterial, AcousticObjectsLibrary, Scene

    scene = Scene("map")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=AcousticMaterial("m", 0.3, 0.1)))
    source = Source("s", 1.5, 1.5, 1.6)
    source.spl = 80.0
    scene.add_source(source)
    return scene


def test_energy_map_gives_levels_per_point_without_frames():
    import math

    result = call("energy_map", _map_scene(), [[4, 3, 1.2], [3, 2, 1.2], [100, 0, 0]])
    assert len(result["spl_a"]) == 3 and len(result["spl"]) == 3 and len(result["spl"][0]) == 7
    assert result["bands"] == [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    assert result["inside"] == [True, True, False]
    assert math.isnan(result["spl_a"][2]) and all(math.isnan(v) for v in result["spl"][2])
    assert 60 < result["spl_a"][0] < 90
    # a static map: the frames of the energy arriving are not sent
    assert "frames" not in result and "times" not in result
    assert any("1 of 3 points" in w for w in result["warnings"])


def test_energy_map_fixes_its_rays_seed_and_image_source_order():
    from compas_acoustics.energy_map import map_rays
    from compas_acoustics.room.geometry import RoomGeometry
    from compas_acoustics.room.raytracing import default_duration

    assert "energy_map" in api.__all__
    scene = _map_scene()
    result = call("energy_map", scene, [[4, 3, 1.2]])
    geometry = RoomGeometry.from_scene(scene)
    assert result["n_rays"] == map_rays(geometry.volume, default_duration(geometry, True))
    # the same map every run: a map has no setting to tune but its grid
    assert "seed" not in result
    assert result["spl_a"] == call("energy_map", scene, [[4, 3, 1.2]])["spl_a"]
    for removed in ("n_rays", "image_source_order", "seed"):
        with pytest.raises(TypeError):
            api.energy_map(scene, [[4, 3, 1.2]], **{removed: 1})


def test_energy_map_rejects_a_source_index_out_of_range():
    with pytest.raises(ValueError, match="source"):
        api.energy_map(_map_scene(), [[4, 3, 1.2]], source_index=5)


# ---------------------------------------------------------------------------
# The coefficient lab (Grasshopper chain C, Task 3)
# ---------------------------------------------------------------------------


def _flat_quad(size=1.0):
    return Mesh.from_vertices_and_faces([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]], [[0, 1, 2, 3]])


def _lab_disc():
    from lab_shapes import flat_tile

    return call("make_scattering_sample", flat_tile(), mode="disc", diameter=0.4)


def test_the_lab_functions_are_listed():
    for name in ("make_scattering_sample", "run_scattering_lab", "polar_balloon", "lab_material"):
        assert name in api.__all__


def test_make_scattering_sample_as_drawn_and_disc():
    from lab_shapes import sine_tile

    from compas_acoustics.lab import ScatteringSample

    drawn = call("make_scattering_sample", _flat_quad(), name="drawn")
    assert isinstance(drawn, ScatteringSample) and drawn.mode == "as_drawn" and drawn.name == "drawn"
    assert drawn.diameter == pytest.approx(2 ** 0.5)
    disc = call("make_scattering_sample", sine_tile(0.1, 0.01), mode="disc", diameter=0.4, period=[0.1, 0.05], baseplate=0.02, symmetry="rotational")
    assert disc.mode == "disc" and disc.diameter == 0.4 and disc.period == (0.1, 0.05) and disc.baseplate == 0.02 and disc.symmetry == "rotational"
    assert disc.sample.number_of_faces() > 0


def test_a_disc_without_a_diameter_is_refused():
    with pytest.raises(ValueError, match="a disc sample needs a diameter"):
        api.make_scattering_sample(_flat_quad(), mode="disc")


@pytest.mark.parametrize("diameter", [0, -1, float("nan")])
def test_a_disc_with_a_diameter_that_is_not_positive_is_refused(diameter):
    from lab_shapes import flat_tile

    with pytest.raises(ValueError, match="diameter is a positive length in m, not"):
        call("make_scattering_sample", flat_tile(), mode="disc", diameter=diameter)


def test_an_unknown_sample_mode_is_refused():
    with pytest.raises(ValueError, match="mode is 'as_drawn' or 'disc', not 'tiled'"):
        api.make_scattering_sample(_flat_quad(), mode="tiled")


def test_run_scattering_lab_returns_the_result_the_material_octaves_and_the_warnings():
    from compas_acoustics.lab import ScatteringResult

    out = call("run_scattering_lab", _lab_disc(), method="geometric", preset="survey", settings={"n_rays": 500})
    assert isinstance(out["result"], ScatteringResult) and out["result"].method == "geometric"
    assert len(out["scattering_coefficients"]) == 7
    assert out["octave_bands"] == [125, 250, 500, 1000, 2000, 4000]
    assert any("heuristic" in n for n in out["notes"])
    assert not any("heuristic: s_max" in w for w in out["warnings"])


def test_the_heuristic_note_of_the_geometric_method_is_a_note_and_not_a_warning():
    from compas_acoustics.lab.scattering import HEURISTIC_NOTES

    out = call("run_scattering_lab", _lab_disc(), method="geometric", preset="survey", settings={"n_rays": 500})
    assert out["notes"] == list(HEURISTIC_NOTES)
    assert not set(out["notes"]) & set(out["warnings"])
    # a real warning of the same run stays one
    assert any("8000 Hz octave was not simulated" in w for w in out["warnings"])


def test_split_notes_leaves_a_kirchhoff_run_without_notes():
    from compas_acoustics.lab.scattering import HEURISTIC_NOTES

    kirchhoff = ["the 8000 Hz octave was not simulated; the material takes the 4000 Hz value", "bands above 5000 Hz are beyond ISO 354"]
    assert api._split_notes(kirchhoff, HEURISTIC_NOTES) == (kirchhoff, [])
    geometric = [HEURISTIC_NOTES[0], kirchhoff[0]]
    assert api._split_notes(geometric, HEURISTIC_NOTES) == ([kirchhoff[0]], [HEURISTIC_NOTES[0]])


def test_the_extended_band_range_simulates_8_kHz_too():
    out = call("run_scattering_lab", _lab_disc(), preset="survey", band_range="extended", settings={"n_rays": 200})
    assert out["octave_bands"][-1] == 8000


def test_an_unknown_band_range_is_refused():
    with pytest.raises(ValueError, match="'iso' or 'extended'"):
        api.run_scattering_lab(_lab_disc(), band_range="narrow")


def test_without_trimesh_the_lab_names_kirchhoff(monkeypatch):
    monkeypatch.setitem(sys.modules, "trimesh", None)
    # a ValueError, because the service answers that with its message and an ImportError with a traceback
    with pytest.raises(ValueError, match='method="kirchhoff"'):
        api.run_scattering_lab(_lab_disc(), preset="survey", settings={"n_rays": 100})


def _geometric_result():
    return call("run_scattering_lab", _lab_disc(), preset="survey", settings={"n_rays": 4000})["result"]


def test_polar_balloon_of_a_flat_disc_points_its_peak_to_the_specular_direction():
    result = _geometric_result()
    preview = 1 if len(result.polar_source_indices) > 1 else 0
    index = result.polar_source_indices[preview]
    colatitude, azimuth = np.radians(result.source_colatitude[index]), np.radians(result.source_azimuth[index])
    source = np.array([np.sin(colatitude) * np.cos(azimuth), np.sin(colatitude) * np.sin(azimuth), np.cos(colatitude)])
    mirror = np.array([-source[0], -source[1], source[2]])
    balloon = call("polar_balloon", result, preview=preview, octave=1000, body="reference")
    mesh = balloon["mesh"]
    vertices = np.array(mesh.vertices_attributes("xyz"))
    faces = [mesh.face_vertices(f) for f in mesh.faces()]
    assert len(balloon["levels"]) == len(vertices) == len(result.receiver_directions)
    assert all(0 <= i < len(vertices) for face in faces for i in face)
    peak = vertices[np.argmax(np.linalg.norm(vertices, axis=1))]
    angle = np.degrees(np.arccos(np.clip(peak @ mirror / np.linalg.norm(peak), -1, 1)))
    assert angle <= 10.0
    assert np.allclose(balloon["source"], source * 1.2)
    assert balloon["label"] == "θ {:g}°, φ {:g}°, 1000 Hz, reference".format(result.source_colatitude[index], result.source_azimuth[index])
    # the faces point outward, and none uses the pole point added to close the hull: seen on the unit directions the vertices
    # move along: that check does not depend on the shape of the spike (the centroid of the vertices holds here too, but it is
    # only meaningful for a roughly round balloon)
    directions = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]
    for face in faces:
        a, b, c = directions[face[:3]]
        assert np.cross(b - a, c - a) @ (a + b + c) > 0


def test_polar_balloon_radius_follows_the_level_down_to_a_minimum_of_3_percent():
    result = _geometric_result()
    balloon = api.polar_balloon(result, radius=2.0, floor_db=-20.0)
    vertices = np.array(balloon["mesh"].vertices_attributes("xyz"))
    levels = np.array(balloon["levels"])
    expected = 2.0 * (0.03 + 0.97 * np.clip(1 - levels / -20.0, 0.0, 1.0))
    assert np.allclose(np.linalg.norm(vertices, axis=1), expected, atol=1e-9)
    assert np.linalg.norm(vertices, axis=1).min() >= 0.03 * 2.0 - 1e-9


def _balloon_geometry(balloon):
    mesh = balloon["mesh"]
    vertices = np.array(mesh.vertices_attributes("xyz"))
    faces = np.array([mesh.face_vertices(f)[:3] for f in mesh.faces()])
    return vertices, faces


def test_a_sparse_balloon_is_a_valid_mesh():
    from scipy.spatial import cKDTree

    # a flat disc measured by the geometric method fills a few cells of the 10 degree histogram, the rest are at -300 dB
    result = _geometric_result()
    balloon = call("polar_balloon", result, preview=0, octave=1000, radius=1.5)
    levels = np.array(balloon["levels"])
    vertices, faces = _balloon_geometry(balloon)
    assert np.mean(levels <= -299.9) > 0.5
    # no two vertices coincide
    distance, _ = cKDTree(vertices).query(vertices, k=2)
    assert distance[:, 1].min() > 1e-6
    # no face is degenerate
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    assert (np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2).min() > 1e-12
    # the empty cells are at 3 % of the radius, the loudest cell at its level
    radii = np.linalg.norm(vertices, axis=1)
    assert radii.min() == pytest.approx(0.03 * 1.5, abs=1e-9)
    expected_peak = 1.5 * (0.03 + 0.97 * min(1.0, max(0.0, 1 - levels.max() / -30.0)))
    assert radii.max() == pytest.approx(expected_peak, abs=1e-9)


def test_a_dense_balloon_is_a_valid_mesh_and_its_radius_is_linear_in_the_level_between_the_floor_and_0_dB():
    import copy
    from scipy.spatial import cKDTree

    result = copy.deepcopy(_geometric_result())
    directions = np.asarray(result.receiver_directions, dtype=float)
    # from +3 dB at the pole to -60 dB at the horizon: cells above 0 dB, between the floor and 0 dB and below the floor
    smooth = 3.0 - 63.0 * (1 - directions[:, 2]) ** 1.3 - 2.0 * directions[:, 0] ** 2
    result.polar_sample_db[0, result.octave_bands.index(1000)] = smooth

    for radius, floor_db in ((1.0, -30.0), (2.0, -40.0)):
        balloon = call("polar_balloon", result, preview=0, octave=1000, radius=radius, floor_db=floor_db)
        vertices, faces = _balloon_geometry(balloon)
        levels = np.array(balloon["levels"])
        assert np.allclose(levels, smooth, atol=0.051)  # the wire keeps a tenth of a dB
        assert (levels > 0).any() and (levels < floor_db).any() and ((levels > floor_db) & (levels < 0)).sum() >= 10
        radii = np.linalg.norm(vertices, axis=1)
        # 3 % at and below the floor, linear between, the radius at 0 dB and above (the levels are the ones the tool returned)
        expected = radius * (0.03 + 0.97 * np.clip(1 - levels / floor_db, 0.0, 1.0))
        assert np.allclose(radii, expected, atol=2e-3 * radius)
        assert radii.max() == pytest.approx(radius, abs=1e-9)
        assert radii.min() == pytest.approx(0.03 * radius, abs=1e-9)
        order = np.argsort(levels)
        assert np.all(np.diff(radii[order]) >= -1e-12)
        # the mesh is valid: distinct vertices, faces with an area
        assert cKDTree(vertices).query(vertices, k=2)[0][:, 1].min() > 1e-6
        a, b, c = (vertices[faces[:, i]] for i in range(3))
        assert (np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2).min() > 1e-12

    # the faces of a smooth balloon (a mild level field) point away from its centre
    result.polar_sample_db[0, result.octave_bands.index(1000)] = -25.0 * (1 - directions[:, 2]) - 3.0 * directions[:, 0] ** 2
    vertices, faces = _balloon_geometry(call("polar_balloon", result, preview=0, octave=1000))
    centre = vertices.mean(axis=0)
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    assert np.all(np.einsum("ij,ij->i", np.cross(b - a, c - a), a + b + c - 3 * centre) > 0)


def test_polar_balloon_refuses_an_octave_that_was_not_simulated():
    result = _geometric_result()
    with pytest.raises(ValueError, match="8000"):
        api.polar_balloon(result, octave=8000)
    with pytest.raises(ValueError, match="'sample' or 'reference'"):
        api.polar_balloon(result, body="both")
    with pytest.raises(ValueError, match="preview"):
        api.polar_balloon(result, preview=len(result.polar_source_indices))


@pytest.mark.parametrize("radius", [0, -1, float("nan")])
def test_polar_balloon_refuses_a_radius_that_is_not_positive(radius):
    with pytest.raises(ValueError, match="radius is a positive length, not"):
        call("polar_balloon", _geometric_result(), radius=radius)


@pytest.mark.parametrize("floor_db", [0, 5.0])
def test_polar_balloon_refuses_a_floor_that_is_not_negative(floor_db):
    with pytest.raises(ValueError, match="floor_db is a negative level in dB, not"):
        call("polar_balloon", _geometric_result(), floor_db=floor_db)


def test_lab_material_from_the_lab_result():
    result = _geometric_result()
    out = call("lab_material", "my panel", [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3], result=result, category="Wood")
    material = out["material"]
    assert isinstance(material, AcousticMaterial) and material.name == "my panel"
    assert list(material.absorption_coefficients) == pytest.approx([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3])
    assert list(material.scattering_coefficients) == pytest.approx(result.scattering_coefficients)
    assert out["warnings"] == []
    assert out["notes"] == [api.MATERIAL_HEURISTIC_NOTE]
    clipped = call("lab_material", "x", [1.5], category="")
    assert any("clipped" in w for w in clipped["warnings"])


def test_lab_material_takes_its_absorption_from_a_material():
    library = MaterialsLibrary()
    source = library.material_from_name(library.material_names[0])
    out = call("lab_material", "copy", source)
    assert list(out["material"].absorption_coefficients) == pytest.approx(list(source.absorption_coefficients))


@pytest.mark.parametrize("absorption", [None, []])
def test_lab_material_refuses_a_missing_absorption(absorption):
    with pytest.raises(ValueError, match="absorption is required: 1 or 7 values, or a material to take them from"):
        api.lab_material("x", absorption)


# ---------------------------------------------------------------------------
# The coefficient lab: the orientation of the object (final review, I1)
# ---------------------------------------------------------------------------


def _closed_box(x=1.2, y=0.8, z=0.3):
    from compas.geometry import Box

    return Mesh.from_shape(Box(x, y, z, Frame([x / 2, y / 2, z / 2], [1, 0, 0], [0, 1, 0])))


def _on_the_wall(mesh):
    """``mesh`` turned 90 degrees about X and moved: what lay on the floor now stands on a wall at y = 4, facing -y, and the frame of that wall."""
    from compas.geometry import Rotation
    from compas.geometry import Translation

    turn = Rotation.from_axis_and_angle([1, 0, 0], np.pi / 2, point=[0, 0, 0]).concatenated(Translation.from_vector([2, 4, 1.0]))
    return mesh.transformed(turn), Frame([2, 4, 1.0], [1, 0, 0], [0, 0, 1])


def _wavy_panel(n=13):
    """An open relief panel, 0.6 m square, z = 0.04 sin(2 pi x / 0.2)."""
    xs = np.linspace(-0.3, 0.3, n)
    vertices = [[xs[i], xs[j], 0.04 * np.sin(2 * np.pi * xs[i] / 0.2)] for j in range(n) for i in range(n)]
    faces = [[j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i] for j in range(n - 1) for i in range(n - 1)]
    return Mesh.from_vertices_and_faces(vertices, faces)


def test_a_frame_crosses_the_wire_to_the_worker():
    frame = Frame([2, 4, 1.0], [1, 0, 0], [0, 0, 1])
    sample = call("make_scattering_sample", _on_the_wall(_closed_box())[0], frame=frame)
    assert sample.diameter == pytest.approx(np.hypot(1.2, 0.8))


def test_a_box_on_a_wall_is_the_box_on_the_floor_when_its_frame_is_given():
    floor = call("make_scattering_sample", _closed_box())
    wall_mesh, frame = _on_the_wall(_closed_box())
    wall = call("make_scattering_sample", wall_mesh, frame=frame)
    assert wall.diameter == pytest.approx(floor.diameter)
    assert wall.structural_depth == pytest.approx(floor.structural_depth)
    results = [api.run_scattering_lab(s, "geometric", preset="survey", settings={"n_rays": 2000}) for s in (floor, wall)]
    assert results[1]["scattering_coefficients"] == pytest.approx(results[0]["scattering_coefficients"], abs=1e-3)


def test_the_reference_is_turned_with_the_mesh():
    from lab_shapes import bumped_box

    from compas.geometry import Rotation
    from compas.geometry import Translation

    turn = Rotation.from_axis_and_angle([1, 0, 0], np.pi / 2, point=[0, 0, 0]).concatenated(Translation.from_vector([2, 4, 1.0]))
    frame = Frame([2, 4, 1.0], [1, 0, 0], [0, 0, 1])
    floor = api.make_scattering_sample(bumped_box(), reference=bumped_box(bump=0))
    wall = api.make_scattering_sample(bumped_box().transformed(turn), reference=bumped_box(bump=0).transformed(turn), frame=frame)
    assert wall.reference_kind == floor.reference_kind == "proxy"
    assert wall.structural_depth == pytest.approx(floor.structural_depth)


def test_a_disc_tile_is_turned_with_its_frame():
    from lab_shapes import sine_tile

    tile = sine_tile(0.1, 0.01)
    wall_tile, frame = _on_the_wall(tile)
    floor = api.make_scattering_sample(tile, mode="disc", diameter=0.4, period=[0.1, 0.05])
    wall = api.make_scattering_sample(wall_tile, mode="disc", diameter=0.4, period=[0.1, 0.05], frame=frame)
    assert wall.structural_depth == pytest.approx(floor.structural_depth)


def test_the_world_xy_frame_changes_nothing():
    plain_sample = api.make_scattering_sample(_closed_box())
    framed = api.make_scattering_sample(_closed_box(), frame=Frame.worldXY())
    assert framed.diameter == plain_sample.diameter and framed.structural_depth == plain_sample.structural_depth
    assert framed.sample.number_of_faces() == plain_sample.sample.number_of_faces()


def test_a_frame_that_is_not_a_frame_is_a_readable_error():
    with pytest.raises(ValueError, match="frame"):
        api.make_scattering_sample(_closed_box(), frame=[0, 0, 1])


def test_a_relief_lying_on_the_wrong_plane_is_told_to_lie_on_xy():
    wall_mesh, frame = _on_the_wall(_wavy_panel())
    with pytest.raises(ValueError, match="the footprint has no area: lay the object on the XY plane"):
        api.make_scattering_sample(wall_mesh)
    # given its plane it is a sample as any other
    assert api.make_scattering_sample(wall_mesh, frame=frame).diameter == pytest.approx(api.make_scattering_sample(_wavy_panel()).diameter)


def test_lab_material_flags_a_heuristic_scattering_and_not_a_measured_one():
    geometric = call("lab_material", "heuristic", [0.1], result=_geometric_result())
    assert any("heuristic geometric method: use kirchhoff or bem for a measurement" in n for n in geometric["notes"])
    assert not any("heuristic" in w for w in geometric["warnings"])
    measured = _geometric_result()
    measured.generation = {"method": "kirchhoff", "preset": "survey"}
    out = api.lab_material("measured", [0.1], result=measured)
    assert out["warnings"] == [] and out["notes"] == []
