"""Grasshopper components, run outside Rhino with Grasshopper, Rhino and ghenv stubbed."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Pyroomacoustics
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics import analyser

needs_hybrid = pytest.mark.skipif(importlib.util.find_spec("pyFDN") is None, reason="needs pyFDN (Python >= 3.11)")
needs_sti = pytest.mark.skipif(not analyser._STI_SUPPORTED, reason="STI and G need pyrato >= 1.0 (Python >= 3.11)")

@pytest.fixture(autouse=True)
def no_configured_worker(monkeypatch, tmp_path):
    from compas_acoustics import worker

    monkeypatch.delenv("COMPAS_ACOUSTICS_WORKER_PYTHON", raising=False)
    monkeypatch.setattr(worker, "CONFIG", tmp_path / "worker.json")
    monkeypatch.setattr(worker, "_PROXY", None)


COMPONENTS = Path(__file__).parents[1] / "src" / "compas_acoustics_ghpython" / "components"


class Geometry(types.SimpleNamespace):
    def __getattr__(self, name):
        return MagicMock()


class FakeAttributes:
    Pivot = None


class FakeButton:
    def __init__(self):
        self.Attributes, self.NickName, self.expired = FakeAttributes(), None, False

    def CreateAttributes(self):
        self.Attributes = FakeAttributes()

    def ExpireSolution(self, recompute):
        self.expired = True


class FakeValueListItem:
    def __init__(self, name, expression):
        self.name, self.expression = name, expression


class FakeValueList(FakeButton):
    def __init__(self):
        super().__init__()
        self.ListItems = FakeItems()


class FakeItems(list):
    def Clear(self):
        self.clear()

    def Add(self, item):
        self.append(item)


@pytest.fixture(autouse=True)
def fresh_ghpython_package():
    # compas_acoustics_ghpython imports Rhino when it is imported; each test gets its own import against its own stubs
    yield
    for name in [m for m in sys.modules if m.startswith("compas_acoustics_ghpython")]:
        del sys.modules[name]


class Tree:
    """Grasshopper's DataTree, as {path: [items]}."""

    def __class_getitem__(cls, item):
        return cls

    def __init__(self):
        self.branches = {}

    def Add(self, item, path):
        self.branches.setdefault(path, []).append(item)

    def AddRange(self, items, path):
        self.branches.setdefault(path, []).extend(items)


def xyz(x, y, z):
    return types.SimpleNamespace(X=x, Y=y, Z=z)


def plane(origin, xaxis, yaxis):
    """Rhino's Plane(origin, x axis, y axis), with the attributes the components read."""
    return types.SimpleNamespace(Origin=xyz(*origin), XAxis=xyz(*xaxis), YAxis=xyz(*yaxis))


def load_component(name, monkeypatch, levels=False):
    """The component's module and its runtime messages; ``levels=True`` records ``(level, text)``."""
    grasshopper = types.ModuleType("Grasshopper")
    grasshopper.Kernel = types.SimpleNamespace(
        GH_ScriptInstance=object,
        GH_RuntimeMessageLevel=types.SimpleNamespace(Warning="warning", Error="error", Remark="remark"),
    )
    grasshopper.DataTree = Tree
    system = types.ModuleType("System")
    system.Drawing = types.SimpleNamespace(PointF=lambda x, y: (x, y))
    special = types.ModuleType("Grasshopper.Kernel.Special")
    special.GH_ButtonObject, special.GH_ValueList, special.GH_ValueListItem = FakeButton, FakeValueList, FakeValueListItem
    kernel_data = types.ModuleType("Grasshopper.Kernel.Data")
    kernel_data.GH_Path = lambda *indices: indices
    rhino = types.ModuleType("Rhino")
    rhino.Geometry = Geometry(Point3d=lambda x, y, z: (x, y, z), Vector3d=lambda x, y, z: (x, y, z),
                              Line=lambda *coordinates: coordinates, Plane=plane)
    rhino_geometry = types.ModuleType("Rhino.Geometry")
    rhino_geometry.Line = rhino.Geometry.Line
    # The package compas_acoustics_ghpython, which the components import, reads more of Rhino.Geometry when it is imported
    rhino_geometry.__getattr__ = lambda name: MagicMock()
    for module_name, module in (("Grasshopper", grasshopper), ("Grasshopper.Kernel", types.ModuleType("Grasshopper.Kernel")),
                                ("Grasshopper.Kernel.Data", kernel_data), ("Rhino", rhino), ("Rhino.Geometry", rhino_geometry),
                                ("Grasshopper.Kernel.Special", special), ("System", system), ("scriptcontext", types.SimpleNamespace(sticky={}))):
        monkeypatch.setitem(sys.modules, module_name, module)
    spec = importlib.util.spec_from_file_location(f"gh_{name}", COMPONENTS / name / "code.py")
    module = importlib.util.module_from_spec(spec)
    messages = []
    record = (lambda level, text: messages.append((level, text))) if levels else (lambda level, text: messages.append(text))
    module.ghenv = types.SimpleNamespace(
        Component=types.SimpleNamespace(AddRuntimeMessage=record, Message="", InstanceGuid="test",
                                        OnPingDocument=lambda: None)
    )
    spec.loader.exec_module(module)
    return module, messages


def scene():
    s = Scene("gh")
    s.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", 0.2, 0.1)))
    s.add_source(Source("src", 1.5, 1.5, 1.6))
    s.add_receiver(Receiver("rec", 5, 3, 1.5))
    return s


def test_the_source_component_takes_position_name_direction_directivity_sound_file(monkeypatch):
    import inspect
    import json

    module, messages = load_component("CA_Source", monkeypatch)
    metadata = json.loads((COMPONENTS / "CA_Source" / "metadata.json").read_text())["ghpython"]
    names = [p["name"] for p in metadata["inputParameters"]]
    assert names == ["position", "name", "direction", "directivity", "sound_file"]
    assert list(inspect.signature(module.CreateSource.RunScript).parameters)[1:] == names
    source = module.CreateSource().RunScript((1.0, 2.0, 1.6), "talker", (0.0, 1.0, 0.0), "cardioid", None)
    assert (source.name, source.position, source.directivity, list(source.direction)) == ("talker", [1.0, 2.0, 1.6], "cardioid", [0.0, 1.0, 0.0])
    assert not messages


def test_the_auralise_inputs_go_sound_rendering_then_saving(monkeypatch):
    import inspect
    import json

    module, _ = load_component("CA_Auralise", monkeypatch)
    metadata = json.loads((COMPONENTS / "CA_Auralise" / "metadata.json").read_text())["ghpython"]
    names = [p["name"] for p in metadata["inputParameters"]]
    assert names == ["irs", "anechoic_sounds", "level", "mix_sources", "save_file_path", "save_file_name", "save"]
    assert list(inspect.signature(module.Auralise.RunScript).parameters)[1:] == names


def test_the_receiver_component_defaults_to_room_aligned(monkeypatch):
    module, _ = load_component("CA_Receiver", monkeypatch)
    receiver, orientation = module.CreateReceiver().RunScript((1.0, 2.0, 1.5), "r", None, None, None)
    assert receiver.position == [1.0, 2.0, 1.5]
    np.testing.assert_allclose(receiver.rotation, np.eye(3), atol=1e-12)
    assert (orientation.XAxis.X, orientation.YAxis.Y) == (1.0, 1.0)


def test_the_receiver_component_takes_direction_and_up(monkeypatch):
    module, _ = load_component("CA_Receiver", monkeypatch)
    receiver, orientation = module.CreateReceiver().RunScript((1.0, 2.0, 1.5), "r", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), None)
    np.testing.assert_allclose(receiver.rotation[:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    # facing +y, the receiver's left is -x
    np.testing.assert_allclose([orientation.YAxis.X, orientation.YAxis.Y], [-1.0, 0.0], atol=1e-12)


def test_the_receiver_component_takes_a_plane(monkeypatch):
    module, messages = load_component("CA_Receiver", monkeypatch)
    receiver, _ = module.CreateReceiver().RunScript((9.0, 9.0, 9.0), "r", None, None, plane((1.0, 2.0, 1.5), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))
    assert receiver.position == [1.0, 2.0, 1.5]
    np.testing.assert_allclose(receiver.rotation, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert messages and "plane" in messages[0]


def test_a_receiver_facing_up_is_an_error_not_a_traceback(monkeypatch):
    module, messages = load_component("CA_Receiver", monkeypatch)
    assert module.CreateReceiver().RunScript((1.0, 2.0, 1.5), "r", (0.0, 0.0, 1.0), None, None) == (None, None)
    assert messages and "parallel" in messages[0]


def capture(monkeypatch):
    calls = []

    def simulate(self, method):
        calls.append(vars(method).copy())
        return {}

    monkeypatch.setattr(Scene, "simulate", simulate)
    return calls


def method_defaults(**changes):
    from compas_acoustics.simulation_methods import Pyroomacoustics

    return dict(vars(Pyroomacoustics(**changes)))


def test_the_ism_order_input_states_the_methods_default(monkeypatch):
    import inspect
    import json

    from compas_acoustics.simulation_methods import Pyroomacoustics

    module, _ = load_component("CA_Simulate_Ray_Tracing", monkeypatch)
    metadata = json.loads((COMPONENTS / "CA_Simulate_Ray_Tracing" / "metadata.json").read_text())["ghpython"]
    assert list(inspect.signature(module.SimulateRayTracing.RunScript).parameters)[1:] == [p["name"] for p in metadata["inputParameters"]]
    assert [p["name"] for p in metadata["outputParameters"]] == ["mono_ir_graph"]
    (ism_order,) = [p for p in metadata["inputParameters"] if p["name"] == "ism_order"]
    assert f"(default {Pyroomacoustics().image_source_order})" in ism_order["description"]
    hybrid = json.loads((COMPONENTS / "CA_Simulate_Hybrid" / "metadata.json").read_text())["ghpython"]["inputParameters"]
    assert "ism_order" in [p["name"] for p in hybrid] and "image_source_order" not in [p["name"] for p in hybrid]


def test_empty_inputs_leave_the_method_defaults_but_draw_a_seed(monkeypatch):
    module, _ = load_component("CA_Simulate_Ray_Tracing", monkeypatch)
    calls = capture(monkeypatch)
    draw_seed(module, monkeypatch)
    s = scene()
    graph = module.SimulateRayTracing().RunScript(s, None, None, None, None, None, None, None, None)
    assert calls == [method_defaults(ray_tracing=True, seed=7)]
    assert (graph.graph, graph.method) == (s.ir_graph, "pyroomacoustics")
    assert module.ghenv.Component.Message.endswith("seed=7")


def test_an_empty_seed_is_a_new_draw_each_run(monkeypatch):
    module, _ = load_component("CA_Simulate_Ray_Tracing", monkeypatch)
    calls = capture(monkeypatch)
    for _ in range(3):
        module.SimulateRayTracing().RunScript(scene(), None, None, None, None, None, None, None, None)
    assert len({call["seed"] for call in calls}) == 3


def test_given_inputs_are_passed_through(monkeypatch):
    module, _ = load_component("CA_Simulate_Ray_Tracing", monkeypatch)
    calls = capture(monkeypatch)
    module.SimulateRayTracing().RunScript(scene(), 4, True, 5000, False, 1e-7, 2.0, 3, False)
    assert calls == [method_defaults(image_source_order=4, ray_tracing=True, n_rays=5000, air_absorption=False,
                                     energy_thres=1e-7, duration=2.0, seed=3, normalize=False)]


def test_without_ray_tracing_the_ray_inputs_are_dropped_with_a_remark(monkeypatch):
    module, messages = load_component("CA_Simulate_Ray_Tracing", monkeypatch)
    calls = capture(monkeypatch)
    module.SimulateRayTracing().RunScript(scene(), 3, False, 5000, None, None, None, 3, None)
    assert calls == [method_defaults(image_source_order=3, ray_tracing=False)]
    assert messages and "n_rays" in messages[0]


def test_the_message_shows_the_scenes_sampling_rate(monkeypatch):
    module, _ = load_component("CA_Simulate_Ray_Tracing", monkeypatch)
    capture(monkeypatch)
    s = scene()
    s.sampling_rate = 44100
    module.SimulateRayTracing().RunScript(s, None, None, None, None, None, None, None, None)
    assert module.ghenv.Component.Message.startswith("44100Hz")


def test_a_missing_scene_warns(monkeypatch):
    module, messages = load_component("CA_Simulate_Ray_Tracing", monkeypatch)
    assert module.SimulateRayTracing().RunScript(None, None, None, None, None, None, None, None, None) is None
    assert messages and "scene" in messages[0]


# ------------------------------------------------------- Deconstruct IR Graph --
def graph_with_methods():
    from compas_acoustics import ImpulseResponse

    s = scene()
    source, receiver = s.sources[0], s.receivers[0]
    mono = ImpulseResponse("m", source, receiver, np.ones(4), 48000)
    ambisonic = ImpulseResponse("a", source, receiver, np.ones((4, 4)), 48000, ambisonic_order=1, normalization="sn3d", channel_order="acn")
    s.populate_ir_graph([mono], attribute="pyroomacoustics")
    s.populate_ir_graph([mono], attribute="data")
    return s, mono, ambisonic


def test_deconstruct_reads_data_by_default(monkeypatch):
    module, _ = load_component("CA_Deconstruct_IR_Graph", monkeypatch)
    s, mono, _ = graph_with_methods()
    irs, *_ = module.DeconstructIRGraph().RunScript(s.ir_graph, None, None)
    assert irs.branches == {(0,): [mono]}


def test_deconstruct_skips_edges_without_the_method_and_warns(monkeypatch):
    module, messages = load_component("CA_Deconstruct_IR_Graph", monkeypatch)
    s, _, _ = graph_with_methods()
    from compas_acoustics_ghpython import MethodGraph

    irs, *_ = module.DeconstructIRGraph().RunScript(MethodGraph(s.ir_graph, "hybrid"), None, None)
    assert irs.branches.get((0,), []) == []
    assert messages and "pyroomacoustics" in messages[0]


def test_deconstruct_reads_the_method_its_wire_comes_from(monkeypatch):
    module, messages = load_component("CA_Deconstruct_IR_Graph", monkeypatch)
    from compas_acoustics_ghpython import MethodGraph

    s, mono, ambisonic = graph_with_methods()
    s.populate_ir_graph([ambisonic], attribute="hybrid")
    for method, expected in (("hybrid", ambisonic), ("pyroomacoustics", mono)):
        irs, *_ = module.DeconstructIRGraph().RunScript(MethodGraph(s.ir_graph, method), None, None)
        assert irs.branches == {(0,): [expected]} and module.ghenv.Component.Message == method
    assert not messages
    assert MethodGraph(s.ir_graph, "hybrid").graph is s.ir_graph and not hasattr(MethodGraph(s.ir_graph, "x"), "nothing")


def test_deconstruct_picks_a_source_receiver_combination(monkeypatch):
    module, messages = load_component("CA_Deconstruct_IR_Graph", monkeypatch)
    from compas_acoustics import ImpulseResponse

    s = scene()
    s.add_source(Source("src2", 2.5, 1.5, 1.6))
    s.add_receiver(Receiver("rec2", 6, 4, 1.5))
    responses = {(i, j): ImpulseResponse(f"{i}{j}", src, rec, np.ones(4), 48000)
                 for i, src in enumerate(s.sources) for j, rec in enumerate(s.receivers)}
    s.populate_ir_graph(list(responses.values()))
    irs, sources, receivers, _ = module.DeconstructIRGraph().RunScript(s.ir_graph, [0], [1])
    assert irs.branches == {(1,): [responses[0, 1]]} and sources.branches == {(1,): [s.sources[0]]}
    irs, *_ = module.DeconstructIRGraph().RunScript(s.ir_graph, [1], None)
    assert irs.branches == {(0,): [responses[1, 0]], (1,): [responses[1, 1]]}
    assert not messages
    irs, *_ = module.DeconstructIRGraph().RunScript(s.ir_graph, [1, 5], [0])
    assert irs.branches == {(0,): [responses[1, 0]]}
    assert len(messages) == 1 and "5" in messages[0]


# -------------------------------------------------------------- Analyse IR --
def analysed_response():
    return next(iter(scene().simulate(Pyroomacoustics(image_source_order=2, ray_tracing=False)).values()))


@needs_sti
def test_analyse_ir_gives_sti_and_g_where_pyrato_can_compute_them(monkeypatch):
    module, messages = load_component("CA_Analyse_IR", monkeypatch, levels=True)
    params, *_ = module.AnalyseIR().RunScript(analysed_response(), None)
    assert [p.name for p in params] == ["EDT", "T20", "T30", "C50", "C80", "D50", "DRR", "Ts", "STI", "G"]
    sti, g = params[-2:]
    assert 0.0 < sti.overall < 1.0 and len(sti.values) == 7 and np.isfinite(g.overall)
    assert (sti.y_unit, g.y_unit) == ("Index (0-1)", "dB")
    assert not [text for level, text in messages if level == "remark"]


def test_analyse_ir_leaves_out_sti_and_g_when_they_could_not_be_computed(monkeypatch):
    from compas_acoustics import AcousticAnalyser
    from compas_acoustics import worker

    module, messages = load_component("CA_Analyse_IR", monkeypatch, levels=True)

    def without_pyrato_1(ir):
        analyser = AcousticAnalyser(ir)
        analyser.analyse()
        for name in ("STI", "G"):
            analyser.analysis_data[name] = [float("nan")] * 7
            analyser.analysis_data[f"{name}_overall"] = float("nan")
        return analyser, []

    monkeypatch.setattr(worker, "analyse", without_pyrato_1)
    params, *_ = module.AnalyseIR().RunScript(analysed_response(), None)
    assert [p.name for p in params] == ["EDT", "T20", "T30", "C50", "C80", "D50", "DRR", "Ts"]
    assert any(level == "remark" and "worker 'hybrid'" in text for level, text in messages)


def test_a_worker_failure_leaves_the_analysis_to_this_python(monkeypatch):
    from compas_acoustics import worker

    module, messages = load_component("CA_Analyse_IR", monkeypatch, levels=True)

    def failing(ir):
        raise RuntimeError("run the installer again")

    monkeypatch.setattr(worker, "analyse", failing)
    params, *_ = module.AnalyseIR().RunScript(analysed_response(), None)
    assert [p.name for p in params][:8] == ["EDT", "T20", "T30", "C50", "C80", "D50", "DRR", "Ts"]
    assert ("warning", "STI and G could not be computed in the worker: run the installer again") in messages


def test_analyse_ir_gives_the_time_and_frequency_responses_of_w_or_both_ears(monkeypatch):
    from compas_acoustics import ImpulseResponse

    module, messages = load_component("CA_Analyse_IR", monkeypatch, levels=True)
    mono = analysed_response()
    _, time, frequency = module.AnalyseIR().RunScript(mono, None)
    assert (time.x_scale, time.y_unit, frequency.x_scale) == ("linear", "Level (dB)", "log")
    assert time.title == "Impulse Response" and np.max(time.values) == 0.0 and time.values.shape == (1, len(mono.denormalized()))
    assert 20 <= frequency.frequencies[0] and frequency.frequencies[-1] <= 20000
    assert str(time).startswith("Impulse Response: Time (s) 0 to")
    _, time, _ = module.AnalyseIR().RunScript(mono, "linear")
    np.testing.assert_allclose(time.values[0], mono.denormalized()[:time.values.shape[1]])

    ir = np.zeros((4800, 4))
    ir[10] = [1.0, 0.5, 0.2, 0.1]
    ambisonic = ImpulseResponse("a", mono.source, mono.receiver, ir, 48000, ambisonic_order=1, normalization="sn3d", channel_order="acn")
    binaural = ImpulseResponse("b", mono.source, mono.receiver, ir[:, :2], 48000, generation={"method": "hybrid", "output": "binaural"})
    for response, channels in ((ambisonic, ["W"]), (binaural, ["left", "right"])):
        time = module.AcousticParameter.from_time_response(response)
        frequency = module.AcousticParameter.from_frequency_response(response)
        assert time.channels == frequency.channels == channels and len(time.values) == len(frequency.values) == len(channels)
    assert module.AnalyseIR().RunScript(mono, "log") == (None, None, None)
    assert ("error", "Input parameter 'time_scale' must be one of ['db', 'linear'].") in messages


def test_the_plotter_offers_every_parameter_with_its_units(monkeypatch):
    from compas_acoustics import AcousticAnalyser

    load_component("CA_Source", monkeypatch)
    try:
        from compas_acoustics_ghpython import AcousticParameter

        ir = next(iter(scene().simulate(Pyroomacoustics(image_source_order=2, ray_tracing=False)).values()))
        analyser = AcousticAnalyser(ir)
        analyser.analyse()
        assert AcousticParameter.NAMES == ("EDT", "T20", "T30", "C50", "C80", "D50", "DRR", "Ts", "STI", "G")
        parameter = AcousticParameter.from_analyser(analyser, "D50")
        assert (parameter.title, parameter.x_unit, parameter.y_unit) == ("Definition (D50)", "Frequency (Hz)", "Value (%)")
        sti = AcousticParameter.from_analyser(analyser, "STI")
        assert (sti.title, sti.values, sti.overall) == ("Speech Transmission Index (STI)", analyser.analysis_data["STI"], analyser.analysis_data["STI_overall"])
        with pytest.raises(ValueError, match="not available in Grasshopper"):
            AcousticParameter.from_analyser(analyser, "ST1")
    finally:
        for name in [m for m in sys.modules if m.startswith("compas_acoustics_ghpython")]:
            del sys.modules[name]


# ------------------------------------------------------------- Audio Player --
def test_the_audio_player_inputs_are_audio_sampling_rate_normalize_play_stop(monkeypatch):
    import inspect
    import json

    module, _ = load_component("CA_Audio_Player", monkeypatch)
    metadata = json.loads((COMPONENTS / "CA_Audio_Player" / "metadata.json").read_text())["ghpython"]
    names = [p["name"] for p in metadata["inputParameters"]]
    assert names == ["audio", "sampling_rate", "normalize", "play", "stop"]
    assert list(inspect.signature(module.PlayAudio.RunScript).parameters)[1:] == names


def test_the_audio_player_shows_a_failure_as_an_error_and_a_clipping_warning(monkeypatch):
    from compas_acoustics import AudioPlayer

    module, messages = load_component("CA_Audio_Player", monkeypatch, levels=True)
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    def play(self, sound, sampling_rate, normalize):
        import warnings

        warnings.warn("The sound array peaks at +6.0 dBFS and will clip.")
        raise RuntimeError("no output device")

    monkeypatch.setattr(AudioPlayer, "play", play)
    module.PlayAudio().RunScript(np.ones(4) * 2.0, 48000, False, True, False)
    assert ("error", "The sound could not be played: no output device") in messages
    assert ("warning", "The sound array peaks at +6.0 dBFS and will clip.") in messages


def test_the_audio_player_says_which_rate_an_array_without_one_is_played_at(monkeypatch, tmp_path):
    import soundfile as sf

    from compas_acoustics import AudioPlayer

    module, messages = load_component("CA_Audio_Player", monkeypatch, levels=True)
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(AudioPlayer, "play", lambda self, sound, sampling_rate, normalize: None)
    module.PlayAudio().RunScript(np.zeros(4), None, None, True, False)
    assert messages == [("remark", "No sampling_rate: the signal is played at 48000 Hz. Connect the sampling_rate of Auralise.")]
    messages.clear()
    module.PlayAudio().RunScript(np.zeros(4), 44100, None, True, False)
    path = tmp_path / "s.wav"
    sf.write(str(path), np.zeros(480), 44100)
    module.PlayAudio().RunScript(str(path), None, None, True, False)
    assert messages == []


class FakeTree:
    """Grasshopper's DataTree as the Auralise component reads its inputs."""

    def __init__(self, branches):
        self.branches = branches

    @property
    def BranchCount(self):
        return len(self.branches)

    @property
    def DataCount(self):
        return sum(len(items) for items in self.branches.values())

    @property
    def Paths(self):
        return list(self.branches)

    def Branch(self, i):
        return list(self.branches.values())[i]


def test_auralise_shows_the_warnings_of_building_the_auraliser(monkeypatch, tmp_path):
    import soundfile as sf

    from compas_acoustics import ImpulseResponse

    module, messages = load_component("CA_Auralise", monkeypatch, levels=True)
    path = tmp_path / "speech.wav"
    sf.write(str(path), np.random.default_rng(1).normal(size=4800) * 0.1, 48000)
    receiver = Receiver("rec", 5, 3, 1.5)
    responses = [ImpulseResponse(f"ir{rate}", Source(f"s{rate}", 1, 1, 1, sound_file=str(path)), receiver, np.r_[1.0, np.zeros(479)], rate)
                 for rate in (48000, 44100)]
    signals, rates = module.Auralise().RunScript(FakeTree({(0,): responses}), None, "relative", True, None, None, None)
    assert signals.branches and rates.branches == {(0,): [48000]}
    assert any(level == "warning" and "resampled to 48000 Hz" in text for level, text in messages)


def test_auralise_convolves_a_binaural_response_as_it_is(monkeypatch, tmp_path):
    import soundfile as sf

    from compas_acoustics import ImpulseResponse

    module, messages = load_component("CA_Auralise", monkeypatch, levels=True)
    path = tmp_path / "speech.wav"
    sf.write(str(path), np.random.default_rng(1).normal(size=4800) * 0.1, 48000)
    ir = np.zeros((480, 2))
    ir[10] = [1.0, 0.5]
    response = ImpulseResponse("b", Source("s", 1, 1, 1, sound_file=str(path)), Receiver("rec", 5, 3, 1.5), ir, 48000,
                               generation={"method": "hybrid", "output": "binaural"})
    signals, _ = module.Auralise().RunScript(FakeTree({(0,): [response]}), None, "relative", True, None, None, None)
    (signal,) = signals.branches[(0,)]
    assert signal.shape[1] == 2 and np.all(np.isfinite(signal)) and not any(level == "error" for level, _ in messages)
    np.testing.assert_allclose(signal[10 : 10 + 4800, 1] / signal[10 : 10 + 4800, 0], 0.5, atol=1e-9)


def auralise_responses(tmp_path):
    """A mono, an ambisonic and a binaural response of one source at one receiver."""
    import soundfile as sf

    from compas_acoustics import ImpulseResponse

    path = tmp_path / "speech.wav"
    sf.write(str(path), np.random.default_rng(1).normal(size=4800) * 0.1, 48000)
    source, receiver = Source("s", 1, 1, 1, sound_file=str(path)), Receiver("rec", 5, 3, 1.5)
    ambisonic = np.zeros((480, 4))
    ambisonic[10] = [1.0, 0.3, 0.2, 0.1]
    return {
        "mono": ImpulseResponse("m", source, receiver, np.r_[np.zeros(10), 1.0, np.zeros(469)], 48000),
        "ambisonic": ImpulseResponse("a", source, receiver, ambisonic, 48000, ambisonic_order=1, normalization="sn3d", channel_order="acn"),
        "binaural": ImpulseResponse("b", source, receiver, ambisonic[:, :2], 48000, generation={"method": "hybrid", "output": "binaural"}),
    }


def test_auralise_keeps_the_format_of_the_impulse_responses(monkeypatch, tmp_path):
    module, messages = load_component("CA_Auralise", monkeypatch, levels=True)
    for name, response in auralise_responses(tmp_path).items():
        messages.clear()
        signals, _ = module.Auralise().RunScript(FakeTree({(0,): [response]}), None, "none", True, None, None, None)
        (signal,) = signals.branches[(0,)]
        assert signal.shape[1] == {"mono": 1, "binaural": 2, "ambisonic": 4}[name]
        assert module.ghenv.Component.Message.startswith(name + ", none")
        assert not any(level == "error" for level, _ in messages)
        assert any(level == "remark" and "ACN/SN3D" in text for level, text in messages) == (name == "ambisonic")
    # the ambisonic signal is the undecoded ACN/SN3D convolution, its channels in the response's ratios
    np.testing.assert_allclose(signal[:, 1], 0.3 * signal[:, 0], atol=1e-12)


def test_auralise_mixes_only_responses_of_one_format(monkeypatch, tmp_path):
    module, messages = load_component("CA_Auralise", monkeypatch, levels=True)
    responses = auralise_responses(tmp_path)
    tree = FakeTree({(0,): [responses["mono"], responses["binaural"]]})
    assert module.Auralise().RunScript(tree, None, "relative", True, None, None, None) == (None, None)
    assert ("error", "The impulse responses at receiver 0 are binaural and mono: mix responses of one format, or turn mix_sources off.") in messages
    signals, _ = module.Auralise().RunScript(tree, None, "relative", False, None, None, None)
    assert [s.shape[1] for s in signals.branches[(0,)]] == [1, 2]
    assert module.ghenv.Component.Message.startswith("binaural + mono, relative")


def test_auralise_offers_only_the_level_list_and_hybrid_no_hrtf_list(monkeypatch):
    for name, cls, run, params in (
        ("CA_Auralise", "Auralise", lambda c: c.RunScript(*[None] * 7), ("level",)),
        ("CA_Simulate_Hybrid", "SimulateHybrid", lambda c: c.RunScript(*[None] * 12), ("hrtf", "output")),
    ):
        module, _ = load_component(name, monkeypatch)
        document, inputs = FakeDocument(), [FakeParam(p) for p in params]
        component = fake_component(document, *inputs, guid=name)
        component.AddRuntimeMessage = lambda level, text: None
        module.ghenv = types.SimpleNamespace(Component=component)
        run(getattr(module, cls)())
        document.solve()
        assert {p.Name: len(p.added) for p in inputs} == {"level": 1} if name == "CA_Auralise" else {"hrtf": 0, "output": 1}


def test_the_audio_player_warns_for_an_ambisonic_signal_and_plays_mono_on_both_channels(monkeypatch):
    from compas_acoustics import AudioPlayer

    module, messages = load_component("CA_Audio_Player", monkeypatch, levels=True)
    played = []
    monkeypatch.setattr(AudioPlayer, "play", lambda self, sound, sampling_rate, normalize: played.append(sound))
    module.PlayAudio().RunScript(np.zeros((480, 4)), 48000, None, True, False)
    assert played == [] and [level for level, _ in messages] == ["warning"] and "4-channel ambisonic" in messages[0][1]

    fake = types.SimpleNamespace(play=lambda data, rate: played.append(data), check_output_settings=lambda **kwargs: None,
                                 _terminate=lambda: None, _initialize=lambda: None, stop=lambda: None)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    AudioPlayer().play_array(np.r_[0.5, -0.25], 48000, normalize=False)
    np.testing.assert_allclose(played[-1], [[0.5, 0.5], [-0.25, -0.25]])


# --------------------------------------------------------------- Value lists --
class FakeParam:
    def __init__(self, name, sources=0):
        self.Name, self.SourceCount = name, sources
        self.Attributes = types.SimpleNamespace(Pivot=types.SimpleNamespace(Y=100.0))
        self.added = []

    def AddSource(self, source):
        self.added.append(source)
        self.SourceCount += 1


class FakeDocument:
    def __init__(self):
        self.objects, self.scheduled = [], []

    def ScheduleSolution(self, delay, callback):
        self.scheduled.append(callback)

    def AddObject(self, obj, update):
        self.objects.append(obj)

    def solve(self):
        for callback in self.scheduled:
            callback(self)


def fake_component(document, *params, guid="guid-1"):
    return types.SimpleNamespace(InstanceGuid=guid, OnPingDocument=lambda: document, Attributes=types.SimpleNamespace(Pivot=types.SimpleNamespace(X=500.0)),
                                 Params=types.SimpleNamespace(Input=list(params)))


def value_list_helper(monkeypatch):
    load_component("CA_Source", monkeypatch)
    from compas_acoustics_ghpython import ensure_inputs

    return ensure_inputs


def test_ensure_inputs_adds_a_list_or_a_button_to_each_unconnected_input_it_is_asked_for(monkeypatch):
    ensure_inputs = value_list_helper(monkeypatch)
    document = FakeDocument()
    tail, run, order, diffraction = FakeParam("tail"), FakeParam("run"), FakeParam("sh_order"), FakeParam("diffraction", sources=1)
    ensure_inputs(fake_component(document, tail, run, order, diffraction), {"tail": ["auto", "eyring"], "run": None, "diffraction": ["none", "udfa"]})
    assert document.objects == []
    document.solve()
    assert len(document.objects) == 2
    (value_list,), (button,) = tail.added, run.added
    assert [(item.name, item.expression) for item in value_list.ListItems] == [("auto", '"auto"'), ("eyring", '"eyring"')]
    assert value_list.NickName == "tail" and button.NickName == "Run"
    assert value_list.Attributes.Pivot == (280.0, 90.0) and value_list.expired and button.expired
    assert order.added == [] and diffraction.added == []


def test_ensure_inputs_acts_once_per_component_and_session(monkeypatch):
    ensure_inputs = value_list_helper(monkeypatch)
    document = FakeDocument()
    component = fake_component(document, FakeParam("run"))
    ensure_inputs(component, {"run": None})
    ensure_inputs(component, {"run": None})
    assert len(document.scheduled) == 1
    other = FakeDocument()
    ensure_inputs(fake_component(other, FakeParam("run"), guid="guid-2"), {"run": None})
    assert len(other.scheduled) == 1


def test_ensure_inputs_skips_what_got_connected_meanwhile_and_needs_no_document(monkeypatch):
    ensure_inputs = value_list_helper(monkeypatch)
    document, run = FakeDocument(), FakeParam("run")
    ensure_inputs(fake_component(document, run), {"run": None})
    run.SourceCount = 1
    document.solve()
    assert document.objects == [] and run.added == []
    ensure_inputs(fake_component(None, FakeParam("run"), guid="guid-3"), {"run": None})
    ensure_inputs(fake_component(document, FakeParam("sh_order"), guid="guid-4"), {"run": None})
    assert len(document.scheduled) == 1


# ---------------------------------------------------------- Simulate Hybrid --
def fake_hybrid(monkeypatch, warn=None):
    """Stand-in for compas_acoustics.simulation_methods.Hybrid, so the component is tested without pyFDN."""
    import warnings

    calls = []

    class Hybrid:
        name, output = "hybrid", "ambisonic"

        def __init__(self, **kwargs):
            if kwargs.get("tail") in ("eyring", "sabine") and kwargs.get("reference"):
                raise ValueError("tail='eyring' takes no reference")
            calls.append(kwargs)
            self.tail = kwargs.get("tail") or ("reference" if kwargs.get("reference") else "ray_tracing")
            self.sh_order = kwargs.get("sh_order", 1)

    def simulate(self, method):
        if warn:
            warnings.warn(warn, UserWarning)
        return {}

    import compas_acoustics.simulation_methods
    monkeypatch.setattr(compas_acoustics.simulation_methods, "Hybrid", Hybrid, raising=False)
    monkeypatch.setattr(Scene, "simulate", simulate)
    return calls


def draw_seed(module, monkeypatch, seed=7):
    """An empty seed input draws a random one: fix the draw."""
    monkeypatch.setattr(module, "random", types.SimpleNamespace(randrange=lambda n: seed))


def run_hybrid(module, s, **inputs):
    inputs.setdefault("run", True)
    names = ("tail", "reference", "sh_order", "ism_order", "diffraction", "late_envelope", "seed", "normalize", "output", "hrtf", "run")
    return module.SimulateHybrid().RunScript(s, *(inputs.get(n) for n in names))


def test_the_reference_tail_without_a_reference_warns_and_does_not_run(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    monkeypatch.setitem(sys.modules, "compas_acoustics.simulation_methods.hybrid", None)
    for run in (False, True):
        binaural, ambisonic = run_hybrid(module, scene(), tail="reference", run=run)
        assert binaural is not None and ambisonic is None
        assert module.ghenv.Component.Message == "connect a reference"
    assert [level for level, _ in messages] == ["warning", "warning"] and "reference" in messages[0][1]


def test_a_missing_hybrid_extra_is_an_error_not_a_traceback(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    monkeypatch.setitem(sys.modules, "compas_acoustics.simulation_methods.hybrid", None)
    assert run_hybrid(module, scene()) is None
    assert messages and messages[0][0] == "error" and "hybrid" in messages[0][1]


def test_simulate_hybrid_without_the_extra_runs_in_a_configured_worker(monkeypatch):
    from compas_acoustics import worker

    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    monkeypatch.setitem(sys.modules, "compas_acoustics.simulation_methods.hybrid", None)
    monkeypatch.setattr(worker, "worker_python", lambda: "/a/worker/python")
    calls = []

    def simulate(scene, method, settings, references):
        calls.append((method, settings, references))
        return {}, ["a warning from the worker"]

    monkeypatch.setattr(worker, "simulate", simulate)
    draw_seed(module, monkeypatch)
    s = scene()
    binaural, ambisonic = run_hybrid(module, s, tail="eyring")
    assert (binaural.graph, binaural.method, ambisonic) == (s.ir_graph, "hybrid_binaural", None)
    assert calls == [("hybrid", {"tail": "eyring", "seed": 7}, None)]
    assert module.ghenv.Component.Message.endswith("seed 7, hrtf kemar")
    assert ("warning", "a warning from the worker") in messages


def test_the_worker_gets_the_requested_outputs_and_hrtf(monkeypatch):
    from compas_acoustics import worker

    module, _ = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    monkeypatch.setitem(sys.modules, "compas_acoustics.simulation_methods.hybrid", None)
    monkeypatch.setattr(worker, "worker_python", lambda: "/a/worker/python")
    calls = []
    monkeypatch.setattr(worker, "simulate", lambda scene, method, settings, references: calls.append(settings) or ({}, []))
    s = scene()
    binaural, ambisonic = run_hybrid(module, s, output="both", hrtf="kemar", seed=3)
    assert (binaural.method, ambisonic.method) == ("hybrid_binaural", "hybrid")
    assert calls == [{"output": ["ambisonic", "binaural"], "hrtf": "kemar", "seed": 3}]


@needs_hybrid
def test_the_hybrid_runs_only_when_run_is_pressed(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    calls = fake_hybrid(monkeypatch)
    draw_seed(module, monkeypatch)
    s = scene()
    for run in (None, False):
        binaural, ambisonic = run_hybrid(module, s, run=run)
        assert (binaural.graph, binaural.method, ambisonic) == (s.ir_graph, "hybrid_binaural", None)
    assert calls == [] and module.ghenv.Component.Message == "press Run"
    assert not messages
    run_hybrid(module, s, run=True)
    assert calls == [{"seed": 7}]


def hybrid_graph(attributes=("hybrid_binaural",), **generation):
    from compas_acoustics import ImpulseResponse

    s = scene()
    source, receiver = s.sources[0], s.receivers[0]
    last = dict(method="hybrid", tail="ray_tracing", sh_order=1, image_source_order=2, diffraction="udfa", **generation)
    for attribute in attributes:
        s.populate_ir_graph([ImpulseResponse("a", source, receiver, np.ones((4, 4)), 48000, ambisonic_order=1, normalization="sn3d",
                                             channel_order="acn", generation=last)], attribute=attribute)
    return s


def test_the_hybrid_says_done_when_not_running_after_a_result(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch)
    s = hybrid_graph(attributes=("hybrid_binaural", "hybrid"))
    for inputs in ({}, {"tail": "auto"}, {"tail": "ray_tracing", "sh_order": 1, "ism_order": 2, "diffraction": "udfa"}, {"diffraction": True}):
        assert run_hybrid(module, s, run=False, **inputs)[0].graph is s.ir_graph
        assert module.ghenv.Component.Message == "done, 1 IR, tail: ray_tracing, order 1 (Run to redo)", inputs
    assert not messages


def test_the_hybrid_status_counts_the_responses_of_the_requested_outputs(monkeypatch):
    module, _ = load_component("CA_Simulate_Hybrid", monkeypatch)
    s = hybrid_graph(attributes=("hybrid", "hybrid_binaural"))
    for output, count in (("ambisonic", "1 IR"), ("binaural", "1 IR"), ("both", "2 IRs")):
        run_hybrid(module, s, run=False, output=output)
        assert module.ghenv.Component.Message == "done, {}, tail: ray_tracing, order 1 (Run to redo)".format(count), output


def test_the_hybrid_says_when_a_requested_output_has_no_responses_yet(monkeypatch):
    module, _ = load_component("CA_Simulate_Hybrid", monkeypatch)
    s = hybrid_graph(attributes=("hybrid",))
    run_hybrid(module, s, run=False, output="binaural")
    assert module.ghenv.Component.Message == "press Run"
    run_hybrid(module, s, run=False, output="both")
    assert module.ghenv.Component.Message == "settings changed, press Run (tail: ray_tracing, order 1)"


def test_the_hybrid_outputs_are_the_requested_formats_only(monkeypatch):
    module, _ = load_component("CA_Simulate_Hybrid", monkeypatch)
    s = scene()
    for output, expected in (("binaural", ("hybrid_binaural", None)), ("ambisonic", (None, "hybrid")), ("both", ("hybrid_binaural", "hybrid")),
                             (None, ("hybrid_binaural", None))):
        result = run_hybrid(module, s, run=False, output=output)
        assert tuple(r and r.method for r in result) == expected, output


def test_the_hybrid_says_when_the_settings_changed_since_the_last_result(monkeypatch):
    module, _ = load_component("CA_Simulate_Hybrid", monkeypatch)
    s = hybrid_graph()
    for inputs in ({"tail": "eyring"}, {"sh_order": 2}, {"ism_order": 3}, {"diffraction": "btms"}, {"diffraction": "none"}):
        run_hybrid(module, s, run=False, **inputs)
        assert module.ghenv.Component.Message == "settings changed, press Run (tail: ray_tracing, order 1)", inputs


def test_the_hybrid_inputs_match_its_script_with_run_last(monkeypatch):
    import inspect
    import json

    module, _ = load_component("CA_Simulate_Hybrid", monkeypatch)
    metadata = json.loads((COMPONENTS / "CA_Simulate_Hybrid" / "metadata.json").read_text())
    names = [p["name"] for p in metadata["ghpython"]["inputParameters"]]
    assert list(inspect.signature(module.SimulateHybrid.RunScript).parameters)[1:] == names
    assert names[0] == "scene" and names[-1] == "run" and names[-3:-1] == ["output", "hrtf"]
    assert [p["name"] for p in metadata["ghpython"]["outputParameters"]] == ["binaural_ir_graph", "ambisonic_ir_graph"]


def test_not_running_does_not_need_the_hybrid_extra(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    monkeypatch.setitem(sys.modules, "compas_acoustics.simulation_methods.hybrid", None)
    s = scene()
    assert run_hybrid(module, s, run=False)[0].graph is s.ir_graph
    assert not messages


@needs_hybrid
def test_the_hybrid_component_passes_only_given_inputs(monkeypatch):
    module, _ = load_component("CA_Simulate_Hybrid", monkeypatch)
    calls = fake_hybrid(monkeypatch)
    draw_seed(module, monkeypatch)
    run_hybrid(module, scene())
    run_hybrid(module, scene(), tail="eyring", diffraction="btms", sh_order=2, seed=3)
    assert calls == [{"seed": 7}, {"tail": "eyring", "diffraction": "btms", "sh_order": 2, "seed": 3}]
    assert "tail: eyring" in module.ghenv.Component.Message and module.ghenv.Component.Message.endswith("seed 3, hrtf kemar")
    run_hybrid(module, scene(), output="both", hrtf="kemar")
    run_hybrid(module, scene(), output="ambisonic", hrtf="kemar")
    assert module.ghenv.Component.Message.endswith("seed 7")
    run_hybrid(module, scene(), output="binaural", hrtf="/a/set.sofa")
    assert module.ghenv.Component.Message.endswith("hrtf set.sofa")
    assert calls[2:] == [{"output": ["ambisonic", "binaural"], "hrtf": "kemar", "seed": 7}, {"output": "ambisonic", "seed": 7},
                         {"output": "binaural", "hrtf": "/a/set.sofa", "seed": 7}]


@needs_hybrid
def test_a_local_run_fills_the_graph_of_each_requested_output(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    s = scene()
    binaural, ambisonic = run_hybrid(module, s, tail="eyring", sh_order=1, ism_order=1, output="both")
    assert not [m for m in messages if m[0] == "error"]
    assert [ir.n_channels for ir in s.impulse_responses(binaural.method).values()] == [2]
    assert [ir.n_channels for ir in s.impulse_responses(ambisonic.method).values()] == [4]
    binaural, ambisonic = run_hybrid(module, s, tail="eyring", sh_order=1, ism_order=1, output="ambisonic", hrtf="kemar")
    assert binaural is None and ambisonic.method == "hybrid"


@needs_hybrid
def test_auto_and_none_are_the_defaults(monkeypatch):
    module, _ = load_component("CA_Simulate_Hybrid", monkeypatch)
    calls = fake_hybrid(monkeypatch)
    draw_seed(module, monkeypatch)
    run_hybrid(module, scene(), tail="auto", diffraction="none")
    assert calls == [{"seed": 7}]


@needs_hybrid
def test_eyring_with_a_reference_is_an_error(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    fake_hybrid(monkeypatch)
    s, _, _ = graph_with_methods()
    assert run_hybrid(module, s, tail="eyring", reference="pyroomacoustics") is None
    assert messages and messages[0][0] == "error"


@needs_hybrid
def test_a_missing_reference_method_is_an_error(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    fake_hybrid(monkeypatch)
    assert run_hybrid(module, scene(), reference="pyroomacoustics") is None
    assert messages and messages[0][0] == "error" and "pyroomacoustics" in messages[0][1]


@needs_hybrid
def test_a_reference_at_another_rate_than_the_scene_is_an_error(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    fake_hybrid(monkeypatch)
    s, _, _ = graph_with_methods()
    s.sampling_rate = 44100
    assert run_hybrid(module, s, reference="pyroomacoustics") is None
    assert messages and messages[0][0] == "error" and "48000" in messages[0][1] and "44100" in messages[0][1]


@needs_hybrid
def test_a_diffuse_field_warning_becomes_a_warning(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    fake_hybrid(monkeypatch, warn="Eyring assumes a diffuse field")
    run_hybrid(module, scene(), tail="eyring")
    assert ("warning", "Eyring assumes a diffuse field") in messages


def test_deconstruct_defaults_to_the_only_method_when_there_is_no_data(monkeypatch):
    module, messages = load_component("CA_Deconstruct_IR_Graph", monkeypatch)
    s, _, ambisonic = graph_with_methods()
    s.populate_ir_graph([], attribute="data")
    s.populate_ir_graph([], attribute="pyroomacoustics")
    s.populate_ir_graph([ambisonic], attribute="hybrid")
    irs, *_ = module.DeconstructIRGraph().RunScript(s.ir_graph, None, None)
    assert irs.branches == {(0,): [ambisonic]}
    assert module.ghenv.Component.Message == "hybrid" and not messages


def test_the_deconstruct_inputs_match_its_script(monkeypatch):
    import inspect
    import json

    module, _ = load_component("CA_Deconstruct_IR_Graph", monkeypatch)
    metadata = json.loads((COMPONENTS / "CA_Deconstruct_IR_Graph" / "metadata.json").read_text())
    parameters = list(inspect.signature(module.DeconstructIRGraph.RunScript).parameters)[1:]
    assert parameters == [p["name"] for p in metadata["ghpython"]["inputParameters"]] == ["ir_graph", "source_idx", "receiver_idx"]


@needs_hybrid
def test_the_hybrid_reference_can_be_the_ray_tracing_graph_wire(monkeypatch):
    module, _ = load_component("CA_Simulate_Hybrid", monkeypatch)
    calls = fake_hybrid(monkeypatch)
    from compas_acoustics_ghpython import MethodGraph

    s, mono, _ = graph_with_methods()
    for wire in (s.ir_graph, MethodGraph(s.ir_graph, "pyroomacoustics")):
        run_hybrid(module, s, reference=wire)
    assert [call["reference"] for call in calls] == [{f"{mono.receiver.guid}_{mono.source.guid}": mono}] * 2


def test_the_ray_tracing_component_shows_warnings(monkeypatch):
    import warnings

    module, messages = load_component("CA_Simulate_Ray_Tracing", monkeypatch, levels=True)

    def simulate(self, method):
        warnings.warn("Receiver 'x' is outside the room and will not be simulated.", UserWarning)
        return {}

    monkeypatch.setattr(Scene, "simulate", simulate)
    module.SimulateRayTracing().RunScript(scene(), None, None, None, None, None, None, None, None)
    assert ("warning", "Receiver 'x' is outside the room and will not be simulated.") in messages


# ---------------------------------------------------------- Acoustic Object --
class RhinoMesh:
    def __init__(self, mesh):
        self.mesh = mesh


def load_acoustic_object(monkeypatch):
    conversions = types.ModuleType("compas_rhino.conversions")
    conversions.mesh_to_compas = lambda rhino_mesh: rhino_mesh.mesh
    conversions.polyline_to_compas = lambda polyline: polyline
    monkeypatch.setitem(sys.modules, "compas_rhino", types.ModuleType("compas_rhino"))
    monkeypatch.setitem(sys.modules, "compas_rhino.conversions", conversions)
    module, messages = load_component("CA_Acoustic_Object", monkeypatch, levels=True)
    module.Rhino.Geometry.Mesh, module.Rhino.Geometry.Polyline, module.Rhino.Geometry.PolylineCurve = RhinoMesh, tuple, list
    return module, messages


def panel_mesh():
    from compas.datastructures import Mesh
    return RhinoMesh(Mesh.from_vertices_and_faces([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]]))


def test_the_acoustic_object_component_defaults_to_a_boundary(monkeypatch):
    module, messages = load_acoustic_object(monkeypatch)
    material = AcousticMaterial("wall", 0.2, 0.1)
    for role in (None, ""):
        acoustic_object = module.CreateAcousticObject().RunScript(panel_mesh(), material, "wall", role)
        assert (acoustic_object.role, acoustic_object.material) == ("boundary", material)
    assert messages == []


def test_the_acoustic_object_component_takes_a_role(monkeypatch):
    module, _ = load_acoustic_object(monkeypatch)
    acoustic_object = module.CreateAcousticObject().RunScript(panel_mesh(), AcousticMaterial("wall", 0.2, 0.1), "table", "obstacle")
    assert (acoustic_object.name, acoustic_object.role) == ("table", "obstacle")


def test_a_bad_role_or_material_is_an_error_not_a_traceback(monkeypatch):
    module, messages = load_acoustic_object(monkeypatch)
    component = module.CreateAcousticObject()
    assert component.RunScript(panel_mesh(), AcousticMaterial("wall", 0.2, 0.1), "x", "wall") is None
    assert component.RunScript(panel_mesh(), 0.3, "x", "boundary") is None
    assert ["role" in messages[0][1], "AcousticMaterial" in messages[1][1]] == [True, True]
    assert {level for level, _ in messages} == {"error"}


def test_the_acoustic_object_inputs_match_its_script(monkeypatch):
    import inspect
    import json

    module, _ = load_acoustic_object(monkeypatch)
    metadata = json.loads((COMPONENTS / "CA_Acoustic_Object" / "metadata.json").read_text())
    parameters = list(inspect.signature(module.CreateAcousticObject.RunScript).parameters)[1:]
    assert parameters == [p["name"] for p in metadata["ghpython"]["inputParameters"]]


# ------------------------------------------------------ Deconstruct Parameter --
def two_parameters(module):
    t30 = module.AcousticParameter("T30", [125, 250, 1000], [0.984, 1.17, 1.94], 1.7, "Reverberation Time (T30)", "Frequency (Hz)", "Time (s)")
    c50 = module.AcousticParameter("C50", [125, 250, 1000], [-1.46, 1.94, -2.58], -2.49, "Clarity (C50)", "Frequency (Hz)", "dB")
    return t30, c50


def test_deconstruct_parameter_gives_every_parameter_apart_by_default(monkeypatch):
    module, messages = load_component("CA_Deconstruct_IR", monkeypatch)
    t30, c50 = two_parameters(module)
    for select in (None, "all"):
        param, name, values, overall, frequencies = module.DeconstructParam().RunScript([t30, c50], select)
        assert param == [t30, c50]
        assert name == ["Reverberation Time (T30)", "Clarity (C50)"] and overall == [1.7, -2.49]
        assert values.branches == {(0,): [0.984, 1.17, 1.94], (1,): [-1.46, 1.94, -2.58]}
        assert list(frequencies) == [125, 250, 1000]
    assert not messages


def test_deconstruct_parameter_gives_a_selected_parameter_as_plain_lists(monkeypatch):
    module, messages = load_component("CA_Deconstruct_IR", monkeypatch)
    t30, c50 = two_parameters(module)
    param, name, values, overall, frequencies = module.DeconstructParam().RunScript([t30, c50], "C50")
    assert param is c50
    assert (name, list(values), overall, list(frequencies)) == ("Clarity (C50)", [-1.46, 1.94, -2.58], -2.49, [125, 250, 1000])
    assert not messages


def test_deconstruct_parameter_prints_its_bands_in_a_panel(monkeypatch):
    module, _ = load_component("CA_Deconstruct_IR", monkeypatch)
    t30, _ = two_parameters(module)
    assert str(t30) == "T30, Time (s): 125 Hz 0.984, 250 Hz 1.17, 1000 Hz 1.94; overall 1.7"
    t30.overall, t30.source_name, t30.receiver_name = None, "S1", "R1"
    assert str(t30) == "T30, Time (s): 125 Hz 0.984, 250 Hz 1.17, 1000 Hz 1.94 (S1 -> R1)"
    assert str(module.AcousticParameter(title="Impulse Response (IR)")) == "Impulse Response (IR)"


def test_deconstruct_parameter_feeds_plot_2d(monkeypatch):
    module, _ = load_component("CA_Deconstruct_IR", monkeypatch)
    t30, c50 = two_parameters(module)
    param = module.DeconstructParam().RunScript([t30, c50], "T30")[0]
    import compas_acoustics_ghpython.plotter as plotter

    monkeypatch.setattr(plotter, "rg", fake_rhino_geometry())
    chart = plotter.Plotter2D(param)
    chart.plot(plane=FakePlane(Vec(0, 0), Vec(1, 0), Vec(0, 1)), width=10.0, height=5.0)
    assert chart.title.PlainText == "Reverberation Time (T30)" and len(chart.data_points) == 3


def test_deconstruct_parameter_warns_without_data_or_without_the_selection(monkeypatch):
    module, messages = load_component("CA_Deconstruct_IR", monkeypatch)
    t30, _ = two_parameters(module)
    for call in (([], None), (None, None), ([t30], "EDT")):
        assert module.DeconstructParam().RunScript(*call) == (None,) * 5
    assert len(messages) == 3 and "EDT" in messages[2]


def test_deconstruct_parameter_offers_the_parameters_in_a_list_and_matches_its_metadata(monkeypatch):
    import inspect
    import json

    module, _ = load_component("CA_Deconstruct_IR", monkeypatch)
    t30, _ = two_parameters(module)
    document, select = FakeDocument(), FakeParam("select")
    module.ghenv = types.SimpleNamespace(Component=fake_component(document, select))
    module.DeconstructParam().RunScript([t30], None)
    document.solve()
    (value_list,) = select.added
    assert [item.name for item in value_list.ListItems] == ["all", "EDT", "T20", "T30", "C50", "C80", "D50", "DRR", "Ts", "STI", "G"]
    metadata = json.loads((COMPONENTS / "CA_Deconstruct_IR" / "metadata.json").read_text())["ghpython"]
    assert list(inspect.signature(module.DeconstructParam.RunScript).parameters)[1:] == [p["name"] for p in metadata["inputParameters"]]
    assert [o["name"] for o in metadata["outputParameters"]] == ["parameter", "name", "values", "overall", "frequencies"]
    assert metadata["inputParameters"][0]["scriptParamAccess"] == 1


# ------------------------------------------------------------------ Plot 2D --
class Vec:
    def __init__(self, x, y, z=0.0):
        self.X, self.Y, self.Z = x, y, z

    def __add__(self, other):
        return Vec(self.X + other.X, self.Y + other.Y, self.Z + other.Z)

    def __mul__(self, k):
        return Vec(self.X * k, self.Y * k, self.Z * k)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0


class FakePlane:
    def __init__(self, *args):
        source = args[0] if len(args) == 1 else None
        self.Origin, self.XAxis, self.YAxis = (source.Origin, source.XAxis, source.YAxis) if source else args


class FakePolyline(list):
    def AddRange(self, points):
        self.extend(points)


def fake_rhino_geometry():
    FakePlane.WorldXY = FakePlane(Vec(0, 0), Vec(1, 0), Vec(0, 1))
    justification = types.SimpleNamespace(**{name: name for name in ("BottomCenter", "TopCenter", "MiddleRight", "BottomLeft", "MiddleCenter")})
    return types.SimpleNamespace(Plane=FakePlane, Line=lambda a, b: (a, b), Polyline=FakePolyline,
                                 TextEntity=types.SimpleNamespace, TextJustification=justification)


def test_the_plot_leaves_half_a_band_between_the_y_axis_and_the_first_band(monkeypatch):
    load_component("CA_Plot_2D", monkeypatch)
    import compas_acoustics_ghpython.plotter as plotter

    monkeypatch.setattr(plotter, "rg", fake_rhino_geometry())
    bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    t30 = plotter.AcousticParameter("T30", bands, [0.98, 1.17, 1.69, 1.94, 2.16, 1.77, 1.15], 1.81, "Reverberation Time (T30)", "Frequency (Hz)", "Time (s)")
    chart = plotter.Plotter2D(t30)
    chart.plot(plane=FakePlane(Vec(2, 15), Vec(1, 0), Vec(0, 1)), width=13.0, height=5.0)
    xs = [point.X - 2 for point in chart.x_axis_points]
    step = 13.0 / 6.5
    assert xs == pytest.approx([step / 2 + i * step for i in range(7)])
    assert xs[-1] == pytest.approx(13.0) and chart.x_axis_line[1].X - 2 == pytest.approx(13.0)
    assert [point.X - 2 for point in chart.data_points] == pytest.approx(xs)
    assert [label.PlainText for label in chart.x_tick_labels] == [str(b) for b in bands]


def test_a_time_response_plots_as_a_curve_per_channel_on_a_time_axis(monkeypatch):
    load_component("CA_Plot_2D", monkeypatch)
    import compas_acoustics_ghpython.plotter as plotter

    monkeypatch.setattr(plotter, "rg", fake_rhino_geometry())
    rng = np.random.default_rng(0)
    values = np.vstack([rng.normal(size=96000), 0.5 * rng.normal(size=96000)])
    values[0, 12345] = 9.0
    curve = plotter.AcousticParameter("IR", np.arange(96000) / 48000, values, None, "Impulse Response (left, right)",
                                      "Time (s)", "Amplitude", x_scale="linear", channels=["left", "right"])
    chart = plotter.Plotter2D(curve)
    chart.plot(plane=FakePlane(Vec(0, 0), Vec(1, 0), Vec(0, 1)), width=10.0, height=5.0)
    assert len(chart.data_polyline) == 2 and chart.data_points == [] and chart.data_values == []
    assert all(len(polyline) <= 2 * plotter.CURVE_BINS for polyline in chart.data_polyline)
    assert [label.PlainText for label in chart.x_tick_labels] == ["0", "0.5", "1", "1.5", "2"]
    assert chart.x_axis_points[-1].X == pytest.approx(10.0)
    # the decimated curve keeps its peak
    top = max(point.Y for point in chart.data_polyline[0])
    assert top == pytest.approx(plotter.remap_number(9.0, float(chart.y_tick_labels[0].PlainText), float(chart.y_tick_labels[-1].PlainText), 0, 5.0))
    # an amplitude is symmetric about zero, with the x axis at zero and its labels below the chart
    y_ticks = [float(label.PlainText) for label in chart.y_tick_labels]
    assert y_ticks == [-t for t in reversed(y_ticks)]
    assert chart.x_axis_line[0].Y == pytest.approx(2.5) and chart.x_axis_points[0].Y == pytest.approx(2.5)
    assert chart.x_tick_labels[0].Plane.Origin.Y == pytest.approx(-0.1)


def test_a_level_in_db_runs_from_its_floor_on_the_x_axis_to_its_peak(monkeypatch):
    load_component("CA_Plot_2D", monkeypatch)
    import compas_acoustics_ghpython.plotter as plotter
    from compas_acoustics.analyser import time_response

    monkeypatch.setattr(plotter, "rg", fake_rhino_geometry())
    ir = np.random.default_rng(0).normal(size=4800) * np.exp(-np.arange(4800) / 500)
    times, level = time_response(ir, 48000)
    curve = plotter.AcousticParameter("IR", times, level, None, "Impulse Response", "Time (s)", "Level (dB)", x_scale="linear")
    chart = plotter.Plotter2D(curve)
    chart.plot(plane=FakePlane(Vec(0, 0), Vec(1, 0), Vec(0, 1)), width=10.0, height=5.0)
    assert [label.PlainText for label in chart.y_tick_labels] == ["-100", "-80", "-60", "-40", "-20", "0"]
    (polyline,) = chart.data_polyline
    assert min(point.Y for point in polyline) == pytest.approx(0.0) and max(point.Y for point in polyline) == pytest.approx(5.0)
    assert chart.x_axis_line[0].Y == pytest.approx(0.0)


def test_a_frequency_response_plots_on_a_log_axis_with_ticks_at_1_2_5_times_powers_of_ten(monkeypatch):
    load_component("CA_Plot_2D", monkeypatch)
    import compas_acoustics_ghpython.plotter as plotter

    monkeypatch.setattr(plotter, "rg", fake_rhino_geometry())
    frequencies = np.geomspace(20, 20000, 500)
    curve = plotter.AcousticParameter("FR", frequencies, -np.log10(frequencies)[np.newaxis], None, "Frequency Response, 1/3 octave",
                                      "Frequency (Hz)", "Level (dB)", x_scale="log")
    chart = plotter.Plotter2D(curve)
    chart.plot(plane=FakePlane(Vec(0, 0), Vec(1, 0), Vec(0, 1)), width=12.0, height=5.0)
    assert [label.PlainText for label in chart.x_tick_labels] == ["50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000"]
    assert chart.x_axis_points[4].X == pytest.approx(12.0 * np.log10(1000 / 20) / 3)
    (polyline,) = chart.data_polyline
    assert len(polyline) == 500 and polyline[0].X == pytest.approx(0.0) and polyline[-1].X == pytest.approx(12.0)


def test_the_y_axis_title_reads_upwards(monkeypatch):
    load_component("CA_Plot_2D", monkeypatch)
    import compas_acoustics_ghpython.plotter as plotter

    monkeypatch.setattr(plotter, "rg", fake_rhino_geometry())
    c50 = plotter.AcousticParameter("C50", [125, 250], [-1.0, 2.0], 0.5, "Clarity (C50)", "Frequency (Hz)", "dB")
    chart = plotter.Plotter2D(c50)
    chart.plot(plane=FakePlane(Vec(0, 0), Vec(1, 0), Vec(0, 1)), width=10.0, height=5.0)
    title = chart.y_axis_title
    assert title.PlainText == "dB" and title.Justification == "BottomCenter"
    # Rhino keeps the plane's direction only without DrawForward; the horizontal texts keep Rhino's default
    assert title.DrawForward is False and not hasattr(chart.x_axis_title, "DrawForward")
    # the text runs up the y axis, its letters on the outer side of the anchor
    assert (title.Plane.XAxis.X, title.Plane.XAxis.Y) == (0, 1) and (title.Plane.YAxis.X, title.Plane.YAxis.Y) == (-1.0, 0.0)
    assert (title.Plane.Origin.X, title.Plane.Origin.Y) == pytest.approx((-1.0, 2.5))


# ------------------------------------------------------ Specular Reflections --
def test_specular_reflections_default_to_first_order_of_every_pair(monkeypatch):
    module, messages = load_component("CA_Specular_Reflections", monkeypatch)
    reflections, paths, images, orders, walls, times = module.SpecularReflectionsComponent().RunScript(scene(), [], None, None)
    assert set(orders) == {1} and len(paths) == len(images) == len(times) == 6
    assert sorted(walls.branches) == [(i,) for i in range(6)] and all(len(w) == 1 for w in walls.branches.values())
    assert walls.branches[(0,)] == reflections.paths[0].wall_names
    assert all(isinstance(name, str) for w in walls.branches.values() for name in w)
    assert ["shoebox_room_7x5x3_floor"] in walls.branches.values()
    assert all(t > 0 for t in times) and not messages
    assert module.ghenv.Component.Message == "order 1: 6"


def test_specular_reflections_pick_a_pair_and_orders(monkeypatch):
    s = scene()
    s.add_receiver(Receiver("rec2", 2, 4, 1.5))
    module, _ = load_component("CA_Specular_Reflections", monkeypatch)
    out = module.SpecularReflectionsComponent().RunScript(s, [0, 1], 0, 1)
    assert set(out[3]) == {0, 1} and all(p.receiver_index == 1 for p in out[0])
    assert len(out[1]) == len(out[2]) == len(out[3]) == len(out[5]) == len(out[0]) == 7
    assert module.ghenv.Component.Message == "order 0: 1, order 1: 6"


def test_specular_reflections_take_numpy_orders_and_ignore_empty_items(monkeypatch):
    module, _ = load_component("CA_Specular_Reflections", monkeypatch)
    out = module.SpecularReflectionsComponent().RunScript(scene(), [None, np.int64(2)], None, None)
    assert set(out[3]) == {2}
    out = module.SpecularReflectionsComponent().RunScript(scene(), [None], None, None)
    assert set(out[3]) == {1}


def test_specular_reflections_leave_out_negative_orders_with_a_warning(monkeypatch):
    module, messages = load_component("CA_Specular_Reflections", monkeypatch, levels=True)
    out = module.SpecularReflectionsComponent().RunScript(scene(), [-1, 0], None, None)
    assert set(out[3]) == {0}
    assert [level for level, text in messages] == ["warning"] and "-1" in messages[0][1]
    messages.clear()
    out = module.SpecularReflectionsComponent().RunScript(scene(), [-2, -1], None, None)
    assert set(out[3]) == {1} and len(out[1]) == 6
    assert [level for level, text in messages] == ["warning"] and "-2, -1" in messages[0][1]


class RecordingPolyline(list):
    def __init__(self):
        super().__init__()

    def AddRange(self, points):
        self.extend(points)


def test_specular_reflections_give_the_paths_vertices_and_image_points(monkeypatch):
    module, _ = load_component("CA_Specular_Reflections", monkeypatch)
    module.Rhino.Geometry.Polyline = RecordingPolyline
    reflections, paths, images, *_ = module.SpecularReflectionsComponent().RunScript(scene(), [0, 2], None, None)
    assert len(paths) == len(images) == len(reflections) > 0
    for polyline, image, path in zip(paths, images, reflections):
        assert polyline == [tuple(point) for point in path.points]
        assert len(polyline) == path.order + 2
        assert image == tuple(path.image)


def test_an_out_of_range_receiver_warns(monkeypatch):
    module, messages = load_component("CA_Specular_Reflections", monkeypatch, levels=True)
    out = module.SpecularReflectionsComponent().RunScript(scene(), [1], 0, 5)
    assert any(level == "warning" and "receiver" in text for level, text in messages)
    assert len(out[1]) == 0


def test_an_out_of_range_source_warns(monkeypatch):
    module, messages = load_component("CA_Specular_Reflections", monkeypatch, levels=True)
    module.SpecularReflectionsComponent().RunScript(scene(), [1], -1, None)
    assert any(level == "warning" and "source" in text for level, text in messages)


def test_specular_reflections_without_a_scene_warn(monkeypatch):
    module, messages = load_component("CA_Specular_Reflections", monkeypatch, levels=True)
    assert module.SpecularReflectionsComponent().RunScript(None, [1], None, None) == (None,) * 6
    assert messages == [("warning", "Input parameter 'scene' failed to collect data.")]


def test_specular_reflections_show_the_scenes_warnings(monkeypatch):
    module, messages = load_component("CA_Specular_Reflections", monkeypatch, levels=True)
    s = scene()
    s.add_receiver(Receiver("outside", 50, 50, 50))
    out = module.SpecularReflectionsComponent().RunScript(s, [1], None, None)
    assert ("warning", "Receiver 'outside' is outside the room and has no specular paths.") in messages
    assert len(out[1]) == 6


def test_a_scene_without_a_room_is_an_error(monkeypatch):
    module, messages = load_component("CA_Specular_Reflections", monkeypatch, levels=True)
    out = module.SpecularReflectionsComponent().RunScript(Scene("empty"), [1], None, None)
    assert out == (None,) * 6 and [level for level, _ in messages] == ["error"]


def test_the_specular_reflections_inputs_match_its_script(monkeypatch):
    import inspect
    import json

    module, _ = load_component("CA_Specular_Reflections", monkeypatch)
    metadata = json.loads((COMPONENTS / "CA_Specular_Reflections" / "metadata.json").read_text())
    parameters = list(inspect.signature(module.SpecularReflectionsComponent.RunScript).parameters)[1:]
    assert parameters == [p["name"] for p in metadata["ghpython"]["inputParameters"]] == ["scene", "order", "source_idx", "receiver_idx"]
    assert [p["name"] for p in metadata["ghpython"]["outputParameters"]] == ["reflections", "paths", "images", "order", "walls", "time"]
    assert (metadata["name"], metadata["nickname"], metadata["subcategory"]) == ("Specular Reflections", "Spec", "05 Display")


# ------------------------------------------------------------------ Ray Field --
def test_ray_field_traces_with_the_python_defaults_and_a_drawn_seed(monkeypatch):
    module, _ = load_component("CA_Ray_Field", monkeypatch)
    field, rays, scattered = module.RayFieldComponent().RunScript(scene(), None, None, None, None, None, None, None, None)
    assert len(field) == 500 and scattered == []
    assert "seed" in module.ghenv.Component.Message


def test_ray_field_order_truncates_only_the_drawn_rays(monkeypatch):
    module, _ = load_component("CA_Ray_Field", monkeypatch)
    module.Rhino.Geometry.Polyline = RecordingPolyline
    field, rays, _ = module.RayFieldComponent().RunScript(scene(), None, 20, 6, None, None, None, 1, 2)
    assert max(r.reflection_order for r in field) > 2
    assert len(rays) == 20 and max(len(r) for r in rays) == 3
    for polyline, ray in zip(rays, field):
        assert polyline == [tuple(point) for point in ray.points[:3]]


def test_ray_field_gives_the_whole_rays_without_an_order(monkeypatch):
    module, _ = load_component("CA_Ray_Field", monkeypatch)
    module.Rhino.Geometry.Polyline = RecordingPolyline
    field, rays, _ = module.RayFieldComponent().RunScript(scene(), None, 5, 4, None, None, None, 1, None)
    assert [len(r) for r in rays] == [len(ray.points) for ray in field]


def test_ray_field_scattering_gives_the_scattered_polylines_and_a_given_seed(monkeypatch):
    module, _ = load_component("CA_Ray_Field", monkeypatch)
    module.Rhino.Geometry.Polyline = RecordingPolyline
    field, rays, scattered = module.RayFieldComponent().RunScript(scene(), None, 10, 3, None, None, True, 7, None)
    assert len(scattered) == len(field.scattered_rays) > 0
    assert [tuple(p) for p in field.scattered_rays[0].points] == scattered[0]
    assert module.ghenv.Component.Message.endswith("seed=7")
    again, *_ = module.RayFieldComponent().RunScript(scene(), None, 10, 3, None, None, True, 7, None)
    assert [tuple(p) for p in again.scattered_rays[0].points] == scattered[0]


def test_ray_field_order_truncates_the_scattered_rays_too(monkeypatch):
    module, _ = load_component("CA_Ray_Field", monkeypatch)
    module.Rhino.Geometry.Polyline = RecordingPolyline
    field, _, scattered = module.RayFieldComponent().RunScript(scene(), None, 10, 4, None, None, True, 7, 1)
    assert len(scattered) == len(field.truncated(reflection_order=1).scattered_rays) < len(field.scattered_rays)


def test_ray_field_emits_into_a_cone(monkeypatch):
    module, messages = load_component("CA_Ray_Field", monkeypatch)
    field, *_ = module.RayFieldComponent().RunScript(scene(), None, 20, 2, xyz(1, 0, 0), 30, None, 1, None)
    assert len(field) == 20 and not messages
    assert all(ray.direction.angle([1, 0, 0], degrees=True) <= 30 + 1e-6 for ray in field)


def test_direction_without_angle_is_a_warning_not_a_traceback(monkeypatch):
    module, messages = load_component("CA_Ray_Field", monkeypatch)
    out = module.RayFieldComponent().RunScript(scene(), None, 20, 3, xyz(1, 0, 0), None, None, 1, None)
    assert any("angle" in m for m in messages) and out == (None,) * 3


def test_ray_field_without_a_scene_warns(monkeypatch):
    module, messages = load_component("CA_Ray_Field", monkeypatch, levels=True)
    assert module.RayFieldComponent().RunScript(None, None, None, None, None, None, None, None, None) == (None,) * 3
    assert messages == [("warning", "Input parameter 'scene' failed to collect data.")]


def test_ray_field_shows_the_scenes_warnings(monkeypatch):
    module, messages = load_component("CA_Ray_Field", monkeypatch, levels=True)
    s = scene()
    s.add_source(Source("outside", 50, 50, 50))
    module.RayFieldComponent().RunScript(s, None, 5, 2, None, None, None, 1, None)
    assert any(level == "warning" and "outside" in text for level, text in messages)


def test_ray_field_traces_only_the_chosen_source(monkeypatch):
    module, messages = load_component("CA_Ray_Field", monkeypatch)
    s = scene()
    s.add_source(Source("second", 5, 2, 1.5))
    field, rays, _ = module.RayFieldComponent().RunScript(s, 1, 5, 2, None, None, None, 1, None)
    assert len(field) == 5 and {ray.source_index for ray in field} == {1} and not messages
    field, *_ = module.RayFieldComponent().RunScript(s, None, 5, 2, None, None, None, 1, None)
    assert {ray.source_index for ray in field} == {0, 1}


def test_ray_field_warns_for_a_source_index_out_of_range(monkeypatch):
    module, messages = load_component("CA_Ray_Field", monkeypatch, levels=True)
    out = module.RayFieldComponent().RunScript(scene(), 3, 5, 2, None, None, None, 1, None)
    assert out == (None,) * 3
    assert messages == [("warning", "No source at index 3; the scene has 1 source(s), 0 to 0.")]


def test_the_ray_field_inputs_match_its_script(monkeypatch):
    import inspect
    import json

    module, _ = load_component("CA_Ray_Field", monkeypatch)
    metadata = json.loads((COMPONENTS / "CA_Ray_Field" / "metadata.json").read_text())
    parameters = list(inspect.signature(module.RayFieldComponent.RunScript).parameters)[1:]
    assert parameters == [p["name"] for p in metadata["ghpython"]["inputParameters"]]
    assert parameters == ["scene", "source_idx", "n_rays", "max_order", "direction", "angle", "scattering", "seed", "order"]
    assert [p["name"] for p in metadata["ghpython"]["outputParameters"]] == ["ray_field", "rays", "scattered"]
    assert (metadata["name"], metadata["nickname"], metadata["subcategory"]) == ("Ray Field", "Rays", "05 Display")


# ------------------------------------------------------------------ Billiard --
class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def billiard(monkeypatch, with_document=True):
    module, messages = load_component("CA_Billiard", monkeypatch)
    clock, document = Clock(), types.SimpleNamespace(scheduled=[])
    module.ghenv.Component.Params = types.SimpleNamespace(Input=[])
    document.ScheduleSolution = lambda delay, callback: document.scheduled.append((delay, callback))
    module.ghenv.Component.OnPingDocument = lambda: document if with_document else None
    module.ghenv.Component.ExpireSolution = lambda recompute: document.__dict__.setdefault("expired", []).append(recompute)
    monkeypatch.setattr(module.time, "time", clock)
    return module, messages, clock, document


def billiard_field(seed=1):
    from compas_acoustics import RayField

    return RayField.from_scene(scene(), n_rays=20, max_reflection_order=4, random_seed=seed)


def run_billiard(module, field, time, play, speed):
    return module.BilliardComponent().RunScript(field, time, play, speed)


def test_billiard_without_play_or_time_shows_the_start(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    field = billiard_field()
    particles, t = run_billiard(module, field, None, False, None)
    assert t == 0.0 and len(particles) == len(field)
    assert all(tuple(p) == tuple(ray.points[0]) for p, ray in zip(particles, field))
    assert module.ghenv.Component.Message == "press Play or connect time"
    assert document.scheduled == []


def test_billiard_time_alone_is_a_frozen_frame(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    field = billiard_field()
    particles, t = run_billiard(module, field, 10.0, False, None)
    assert t == 10.0 and document.scheduled == []
    assert [tuple(p) for p in particles] == [tuple(p) for p in field.particles(10.0)[0]]
    assert module.ghenv.Component.Message == "t = 10.0 ms"


def test_play_animates_up_to_time_and_stops(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    field = billiard_field()
    clock.now = 0.0
    assert run_billiard(module, field, 10.0, True, 5.0)[1] == 0.0
    assert len(document.scheduled) == 1 and document.scheduled[0][0] == 33
    clock.now = 1.0
    assert run_billiard(module, field, 10.0, False, 5.0)[1] == 5.0
    assert len(document.scheduled) == 2
    clock.now = 3.0
    assert run_billiard(module, field, 10.0, False, 5.0)[1] == 10.0
    assert len(document.scheduled) == 2
    clock.now = 9.0
    assert run_billiard(module, field, 10.0, False, 5.0)[1] == 10.0
    assert len(document.scheduled) == 2


def test_play_without_time_runs_to_the_end_at_the_default_speed(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    field = billiard_field()
    clock.now = 0.0
    run_billiard(module, field, None, True, None)
    clock.now = 5.0
    assert run_billiard(module, field, None, False, None)[1] == pytest.approx(field.duration_ms / 2)
    clock.now = 20.0
    assert run_billiard(module, field, None, False, None)[1] == pytest.approx(field.duration_ms)


def test_a_new_ray_field_keeps_playing(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    clock.now = 0.0
    run_billiard(module, billiard_field(1), 10.0, True, 5.0)
    clock.now = 1.0
    scheduled = len(document.scheduled)
    particles, t = run_billiard(module, billiard_field(2), 10.0, False, 5.0)
    assert t == 5.0 and len(document.scheduled) == scheduled + 1


def test_the_scheduled_frame_expires_the_component_without_a_second_solution(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    run_billiard(module, billiard_field(), 10.0, True, 5.0)
    document.scheduled[0][1](document)
    assert document.expired == [False]


def test_pressing_play_while_playing_restarts(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    field = billiard_field()
    clock.now = 0.0
    run_billiard(module, field, 10.0, True, 5.0)
    clock.now = 1.5
    assert run_billiard(module, field, 10.0, False, 5.0)[1] == 7.5
    assert run_billiard(module, field, 10.0, True, 5.0)[1] == 0.0


def test_scrubbing_time_after_a_finished_play_scrubs(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    field = billiard_field()
    run_billiard(module, field, 10.0, True, 5.0)
    clock.now = 100.0
    assert run_billiard(module, field, 10.0, False, 5.0)[1] == 10.0
    scheduled = len(document.scheduled)
    assert run_billiard(module, field, 3.0, False, 5.0)[1] == 3.0
    assert len(document.scheduled) == scheduled
    clock.now = 200.0
    assert run_billiard(module, field, 3.0, False, 5.0)[1] == 3.0


def test_changing_time_during_playback_stops_it(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    field = billiard_field()
    run_billiard(module, field, 10.0, True, 5.0)
    clock.now = 1.0
    scheduled = len(document.scheduled)
    assert run_billiard(module, field, 4.0, False, 5.0)[1] == 4.0
    assert len(document.scheduled) == scheduled


def test_the_held_end_frame_stays_while_time_stays_empty(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    field = billiard_field()
    run_billiard(module, field, None, True, None)
    clock.now = 50.0
    assert run_billiard(module, field, None, False, None)[1] == pytest.approx(field.duration_ms)
    clock.now = 60.0
    assert run_billiard(module, field, None, False, None)[1] == pytest.approx(field.duration_ms)
    assert len(document.scheduled) == 1


def test_billiard_times_are_clamped_to_the_ray_fields_duration(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    field = billiard_field()
    assert run_billiard(module, field, -5.0, True, 5.0)[1] == 0.0
    assert document.scheduled == []
    clock.now = 3.0
    assert run_billiard(module, field, -5.0, False, 5.0)[1] == 0.0
    assert run_billiard(module, field, 1e9, False, 5.0)[1] == field.duration_ms
    assert run_billiard(module, field, -5.0, False, 5.0)[1] == 0.0
    assert module.ghenv.Component.Message == "t = 0.0 ms"


def test_billiard_draws_only_the_particles(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    field = billiard_field()
    out = run_billiard(module, field, 15.0, False, None)
    points, _ = field.particles(15.0)
    assert len(out) == 2 and out[0] == [tuple(p) for p in points] and out[1] == 15.0


def test_billiard_without_a_ray_field_warns(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    module.ghenv.Component.Message = "t = 3.0 ms"
    assert run_billiard(module, None, 1.0, True, None) == (None,) * 2
    assert messages == ["Input parameter 'ray_field' failed to collect data."] and document.scheduled == []
    assert module.ghenv.Component.Message == ""


def test_billiard_with_an_empty_ray_field_warns(monkeypatch):
    from compas_acoustics import RayField

    module, messages, clock, document = billiard(monkeypatch)
    module.ghenv.Component.Message = "t = 3.0 ms"
    assert run_billiard(module, RayField([]), 1.0, True, None) == (None,) * 2
    assert messages == ["The ray field has no rays."] and document.scheduled == []
    assert module.ghenv.Component.Message == ""


def test_a_held_play_toggle_plays_once_and_holds(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    field = billiard_field()
    clock.now = 0.0
    assert run_billiard(module, field, 10.0, True, 5.0)[1] == 0.0
    clock.now = 1.0
    assert run_billiard(module, field, 10.0, True, 5.0)[1] == 5.0
    clock.now = 3.0
    assert run_billiard(module, field, 10.0, True, 5.0)[1] == 10.0
    scheduled = len(document.scheduled)
    clock.now = 4.0
    assert run_billiard(module, field, 10.0, True, 5.0)[1] == 10.0
    assert len(document.scheduled) == scheduled


def test_play_turned_off_and_on_again_restarts(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    field = billiard_field()
    clock.now = 0.0
    run_billiard(module, field, 10.0, True, 5.0)
    clock.now = 1.0
    assert run_billiard(module, field, 10.0, False, 5.0)[1] == 5.0
    assert run_billiard(module, field, 10.0, True, 5.0)[1] == 0.0
    clock.now = 1.5
    assert run_billiard(module, field, 10.0, True, 5.0)[1] == 2.5


def test_scrubbing_with_a_held_play_toggle_scrubs(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    field = billiard_field()
    run_billiard(module, field, 10.0, True, 5.0)
    clock.now = 1.0
    assert run_billiard(module, field, 4.0, True, 5.0)[1] == 4.0
    clock.now = 2.0
    assert run_billiard(module, field, 4.0, True, 5.0)[1] == 4.0


def test_playing_to_the_end_follows_a_longer_ray_field(monkeypatch):
    from compas_acoustics import RayField

    module, messages, clock, document = billiard(monkeypatch)
    short = billiard_field()
    longer = RayField.from_scene(scene(), n_rays=20, max_reflection_order=12, random_seed=1)
    assert longer.duration_ms > short.duration_ms
    clock.now = 0.0
    run_billiard(module, short, None, True, 5.0)
    clock.now = 1000.0
    assert run_billiard(module, short, None, False, 5.0)[1] == pytest.approx(short.duration_ms)
    assert run_billiard(module, longer, None, False, 5.0)[1] == pytest.approx(longer.duration_ms)


def test_the_default_speed_follows_a_new_ray_field(monkeypatch):
    from compas_acoustics import RayField

    module, messages, clock, document = billiard(monkeypatch)
    short = billiard_field()
    longer = RayField.from_scene(scene(), n_rays=20, max_reflection_order=12, random_seed=1)
    clock.now = 0.0
    run_billiard(module, short, None, True, None)
    clock.now = 2.0
    assert run_billiard(module, longer, None, False, None)[1] == pytest.approx(short.duration_ms / 5)
    clock.now = 3.0
    assert run_billiard(module, longer, None, False, None)[1] == pytest.approx(short.duration_ms / 5 + longer.duration_ms / 10)


def test_a_new_speed_while_playing_runs_on_from_the_current_frame(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch)
    field = billiard_field()
    assert field.duration_ms > 10.0
    clock.now = 0.0
    run_billiard(module, field, None, True, 1.0)
    clock.now = 2.0
    assert run_billiard(module, field, None, False, 1.0)[1] == pytest.approx(2.0)
    assert run_billiard(module, field, None, False, 4.0)[1] == pytest.approx(2.0)
    clock.now = 3.0
    assert run_billiard(module, field, None, False, 4.0)[1] == pytest.approx(6.0)


def test_billiard_ignores_a_speed_that_is_not_positive_and_does_not_need_a_document(monkeypatch):
    module, messages, clock, document = billiard(monkeypatch, with_document=False)
    field = billiard_field()
    clock.now = 0.0
    run_billiard(module, field, None, True, 0.0)
    clock.now = 5.0
    assert run_billiard(module, field, None, False, -1.0)[1] == pytest.approx(field.duration_ms / 2)
    assert any("speed" in m for m in messages)


def test_the_billiard_inputs_match_its_script(monkeypatch):
    import inspect
    import json

    module, *_ = billiard(monkeypatch)
    metadata = json.loads((COMPONENTS / "CA_Billiard" / "metadata.json").read_text())
    parameters = list(inspect.signature(module.BilliardComponent.RunScript).parameters)[1:]
    assert parameters == [p["name"] for p in metadata["ghpython"]["inputParameters"]] == ["ray_field", "time", "play", "speed"]
    assert [p["name"] for p in metadata["ghpython"]["outputParameters"]] == ["particles", "t"]
    assert (metadata["name"], metadata["nickname"], metadata["subcategory"]) == ("Billiard", "Billiard", "05 Display")


def test_components_with_list_inputs_import_system():
    import json
    import re

    # Rhino types a list input as System.Collections.Generic.List in the script's signature; without the import the script cannot load
    # ("Error getting GH_ScriptInstance instance")
    missing = []
    for metadata in COMPONENTS.glob("*/metadata.json"):
        inputs = json.loads(metadata.read_text())["ghpython"]["inputParameters"]
        code = (metadata.parent / "code.py").read_text()
        if any(i.get("scriptParamAccess") == 1 for i in inputs) and not re.search(r"^\s*(import System\b|from System import)", code, re.MULTILINE):
            missing.append(metadata.parent.name)
    assert missing == []


# ------------------------------------------------------------ Check Channel --
def test_check_channel_counts_the_channels_of_any_sound(monkeypatch, tmp_path):
    import inspect
    import json

    import soundfile as sf

    module, messages = load_component("CA_Channel", monkeypatch, levels=True)
    check = module.CheckChannel().RunScript
    for channels in (1, 2):
        path = tmp_path / f"{channels}.wav"
        sf.write(str(path), np.zeros((480, channels)), 48000)
        assert check(str(path)) == channels and check(path) == channels
    source, receiver = scene().sources[0], scene().receivers[0]
    mono = analysed_response()
    from compas_acoustics import ImpulseResponse

    ambisonic = ImpulseResponse("a", source, receiver, np.ones((4, 4)), 48000, ambisonic_order=1, normalization="sn3d", channel_order="acn")
    assert (check(mono), check(ambisonic)) == (1, 4)
    assert (check(np.zeros(100)), check(np.zeros((100, 2))), check(np.zeros((100, 4)).tolist())) == (1, 2, 4)
    sf.write(str(tmp_path / "s.wav"), np.zeros((480, 2)), 48000)
    assert check(Source("talker", 0, 0, 0, sound_file=str(tmp_path / "s.wav"))) == 2
    assert module.ghenv.Component.Message == "channels: 2" and not messages

    metadata = json.loads((COMPONENTS / "CA_Channel" / "metadata.json").read_text())["ghpython"]
    assert list(inspect.signature(module.CheckChannel.RunScript).parameters)[1:] == [p["name"] for p in metadata["inputParameters"]] == ["sound"]


def test_check_channel_reports_what_it_cannot_read_as_an_error(monkeypatch, tmp_path):
    module, messages = load_component("CA_Channel", monkeypatch, levels=True)
    check = module.CheckChannel().RunScript
    assert [check(None), check(""), check(Source("silent", 0, 0, 0)), check(object()), check(str(tmp_path / "missing.wav"))] == [None] * 5
    assert [level for level, _ in messages] == ["warning", "warning", "error", "error", "error"]
    assert "no sound file" in messages[2][1]


def test_no_component_pins_its_own_script_environment():
    import re

    # A directive in a component's script makes Rhino run it in that environment, where compas_acoustics is not installed.
    # The installer adds one itself when the user chooses a venv.
    pinned = [path.parent.name for path in COMPONENTS.glob("*/code.py") if re.search(r"^#\s*(venv|r|requirements)\s*:", path.read_text(), re.MULTILINE)]
    assert pinned == []


def test_the_plot_2d_inputs_match_its_script(monkeypatch):
    import inspect
    import json

    module, messages = load_component("CA_Plot_2D", monkeypatch)
    metadata = json.loads((COMPONENTS / "CA_Plot_2D" / "metadata.json").read_text())["ghpython"]
    parameters = list(inspect.signature(module.Plot2D.RunScript).parameters)[1:]
    assert parameters == [p["name"] for p in metadata["inputParameters"]] == ["data", "plane", "x_size", "y_size", "settings"]
    assert module.Plot2D().RunScript(None, None, None, None, None) is None and "'data'" in messages[0]


def test_the_analyse_ir_inputs_and_outputs_match_its_script(monkeypatch):
    import inspect
    import json

    module, _ = load_component("CA_Analyse_IR", monkeypatch)
    metadata = json.loads((COMPONENTS / "CA_Analyse_IR" / "metadata.json").read_text())["ghpython"]
    assert list(inspect.signature(module.AnalyseIR.RunScript).parameters)[1:] == [p["name"] for p in metadata["inputParameters"]] == ["ir", "time_scale"]
    assert [p["name"] for p in metadata["outputParameters"]] == ["parameters", "time", "frequency"]
