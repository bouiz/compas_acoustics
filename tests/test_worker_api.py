"""The worker API the Grasshopper plugin calls, in-process; every argument and result crosses compas_pb as it would the wire."""

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
# Materials and model makers
# ---------------------------------------------------------------------------


def test_every_function_the_service_serves_is_listed():
    for name in ("materials_library", "material_from_name", "make_material", "make_source", "make_receiver", "make_acoustic_object", "make_scene"):
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
# Analysis, auralise, play, stop
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
    return ImpulseResponse("ir", source, receiver or Receiver("rec", 4, 3, 1.5), raw if channels > 1 else raw[:, 0], rate, generation={"method": "test"})


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
    return [decay(source=Source("a", 1, 1, 1.5), receiver=receiver, level=1.0, **kwargs), decay(source=Source("b", 2, 1, 1.5), receiver=receiver, level=0.25, seed=1, **kwargs)]


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
# Rays, reflections, fitness
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
# What the Check Channel component asks about a file
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
# Playback control
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
    for fixed in ("n_rays", "image_source_order", "seed"):
        with pytest.raises(TypeError):
            api.energy_map(scene, [[4, 3, 1.2]], **{fixed: 1})


def test_energy_map_rejects_a_source_index_out_of_range():
    with pytest.raises(ValueError, match="source"):
        api.energy_map(_map_scene(), [[4, 3, 1.2]], source_index=5)
