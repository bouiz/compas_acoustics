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

needs_hybrid = pytest.mark.skipif(importlib.util.find_spec("pyFDN") is None, reason="needs the hybrid extra (Python >= 3.11)")

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
    assert names == ["irs", "signals", "output_format", "hrtf", "level", "multiconvolve", "save_file_path", "save_file_name", "save"]
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

    module, _ = load_component("CA_Simulate_Ray_Trace", monkeypatch)
    metadata = json.loads((COMPONENTS / "CA_Simulate_Ray_Trace" / "metadata.json").read_text())["ghpython"]
    assert list(inspect.signature(module.SimulateRayTracing.RunScript).parameters)[1:] == [p["name"] for p in metadata["inputParameters"]]
    (ism_order,) = [p for p in metadata["inputParameters"] if p["name"] == "ism_order"]
    assert f"(default {Pyroomacoustics().image_source_order})" in ism_order["description"]
    hybrid = json.loads((COMPONENTS / "CA_Simulate_Hybrid" / "metadata.json").read_text())["ghpython"]["inputParameters"]
    assert "ism_order" in [p["name"] for p in hybrid] and "image_source_order" not in [p["name"] for p in hybrid]


def test_empty_inputs_leave_the_method_defaults(monkeypatch):
    module, _ = load_component("CA_Simulate_Ray_Trace", monkeypatch)
    calls = capture(monkeypatch)
    s = scene()
    graph = module.SimulateRayTracing().RunScript(s, None, None, None, None, None, None, None, None)
    assert calls == [method_defaults(ray_tracing=True)]
    assert (graph.graph, graph.method) == (s.ir_graph, "pyroomacoustics")


def test_given_inputs_are_passed_through(monkeypatch):
    module, _ = load_component("CA_Simulate_Ray_Trace", monkeypatch)
    calls = capture(monkeypatch)
    module.SimulateRayTracing().RunScript(scene(), 4, True, 5000, False, 1e-7, 2.0, 3, False)
    assert calls == [method_defaults(image_source_order=4, ray_tracing=True, n_rays=5000, air_absorption=False,
                                     energy_thres=1e-7, duration=2.0, seed=3, normalize=False)]


def test_without_ray_tracing_the_ray_inputs_are_dropped_with_a_remark(monkeypatch):
    module, messages = load_component("CA_Simulate_Ray_Trace", monkeypatch)
    calls = capture(monkeypatch)
    module.SimulateRayTracing().RunScript(scene(), 3, False, 5000, None, None, None, 3, None)
    assert calls == [method_defaults(image_source_order=3, ray_tracing=False)]
    assert messages and "n_rays" in messages[0]


def test_the_message_shows_the_scenes_sampling_rate(monkeypatch):
    module, _ = load_component("CA_Simulate_Ray_Trace", monkeypatch)
    capture(monkeypatch)
    s = scene()
    s.sampling_rate = 44100
    module.SimulateRayTracing().RunScript(s, None, None, None, None, None, None, None, None)
    assert module.ghenv.Component.Message.startswith("44100Hz")


def test_a_missing_scene_warns(monkeypatch):
    module, messages = load_component("CA_Simulate_Ray_Trace", monkeypatch)
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
    irs, *_ = module.DeconstructIRGraph().RunScript(s.ir_graph, None)
    assert irs.branches == {(0,): [mono]}


def test_deconstruct_skips_edges_without_the_method_and_warns(monkeypatch):
    module, messages = load_component("CA_Deconstruct_IR_Graph", monkeypatch)
    s, _, _ = graph_with_methods()
    from compas_acoustics_ghpython import MethodGraph

    irs, *_ = module.DeconstructIRGraph().RunScript(MethodGraph(s.ir_graph, "hybrid"), None)
    assert irs.branches.get((0,), []) == []
    assert messages and "pyroomacoustics" in messages[0]


def test_deconstruct_reads_the_method_its_wire_comes_from(monkeypatch):
    module, messages = load_component("CA_Deconstruct_IR_Graph", monkeypatch)
    from compas_acoustics_ghpython import MethodGraph

    s, mono, ambisonic = graph_with_methods()
    s.populate_ir_graph([ambisonic], attribute="hybrid")
    for method, expected in (("hybrid", ambisonic), ("pyroomacoustics", mono)):
        irs, *_ = module.DeconstructIRGraph().RunScript(MethodGraph(s.ir_graph, method), None)
        assert irs.branches == {(0,): [expected]} and module.ghenv.Component.Message == method
    assert not messages
    assert MethodGraph(s.ir_graph, "hybrid").graph is s.ir_graph and not hasattr(MethodGraph(s.ir_graph, "x"), "nothing")


# -------------------------------------------------------------- Analyse IR --
def analysed_response():
    return next(iter(scene().simulate(Pyroomacoustics(image_source_order=2, ray_tracing=False)).values()))


def test_analyse_ir_gives_sti_and_g_where_pyrato_can_compute_them(monkeypatch):
    module, messages = load_component("CA_Analyse_IR", monkeypatch, levels=True)
    params = module.AnalyseIR().RunScript(analysed_response())
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
    params = module.AnalyseIR().RunScript(analysed_response())
    assert [p.name for p in params] == ["EDT", "T20", "T30", "C50", "C80", "D50", "DRR", "Ts"]
    assert any(level == "remark" and "worker 'binaural' or 'hybrid'" in text for level, text in messages)


def test_a_worker_failure_leaves_the_analysis_to_this_python(monkeypatch):
    from compas_acoustics import worker

    module, messages = load_component("CA_Analyse_IR", monkeypatch, levels=True)

    def failing(ir):
        raise RuntimeError("run the installer again")

    monkeypatch.setattr(worker, "analyse", failing)
    params = module.AnalyseIR().RunScript(analysed_response())
    assert [p.name for p in params][:8] == ["EDT", "T20", "T30", "C50", "C80", "D50", "DRR", "Ts"]
    assert ("warning", "STI and G could not be computed in the worker: run the installer again") in messages


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
def test_the_audio_player_shows_a_failure_as_an_error_and_a_clipping_warning(monkeypatch):
    from compas_acoustics import AudioPlayer

    module, messages = load_component("CA_Audio_Player", monkeypatch, levels=True)
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    def play(self, sound, sampling_rate, normalize):
        import warnings

        warnings.warn("The sound array peaks at +6.0 dBFS and will clip.")
        raise RuntimeError("no output device")

    monkeypatch.setattr(AudioPlayer, "play", play)
    module.PlayAudio().RunScript(np.ones(4) * 2.0, True, False, 48000, False)
    assert ("error", "The sound could not be played: no output device") in messages
    assert ("warning", "The sound array peaks at +6.0 dBFS and will clip.") in messages


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


def run_hybrid(module, s, **inputs):
    inputs.setdefault("run", True)
    names = ("tail", "reference", "sh_order", "ism_order", "diffraction", "late_envelope", "seed", "normalize", "run")
    return module.SimulateHybrid().RunScript(s, *(inputs.get(n) for n in names))


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
    s = scene()
    graph = run_hybrid(module, s, tail="eyring")
    assert (graph.graph, graph.method) == (s.ir_graph, "hybrid")
    assert calls == [("hybrid", {"tail": "eyring"}, None)]
    assert ("warning", "a warning from the worker") in messages


@needs_hybrid
def test_the_hybrid_runs_only_when_run_is_pressed(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    calls = fake_hybrid(monkeypatch)
    s = scene()
    for run in (None, False):
        graph = run_hybrid(module, s, run=run)
        assert (graph.graph, graph.method) == (s.ir_graph, "hybrid")
    assert calls == [] and module.ghenv.Component.Message == "press Run"
    assert not messages
    run_hybrid(module, s, run=True)
    assert calls == [{}]


def hybrid_graph(**generation):
    from compas_acoustics import ImpulseResponse

    s = scene()
    source, receiver = s.sources[0], s.receivers[0]
    last = dict(method="hybrid", tail="ray_tracing", sh_order=1, image_source_order=2, diffraction="udfa", **generation)
    s.populate_ir_graph([ImpulseResponse("a", source, receiver, np.ones((4, 4)), 48000, ambisonic_order=1, normalization="sn3d",
                                         channel_order="acn", generation=last)], attribute="hybrid")
    return s


def test_the_hybrid_says_done_when_not_running_after_a_result(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch)
    s = hybrid_graph()
    for inputs in ({}, {"tail": "auto"}, {"tail": "ray_tracing", "sh_order": 1, "ism_order": 2, "diffraction": "udfa"}, {"diffraction": True}):
        assert run_hybrid(module, s, run=False, **inputs).graph is s.ir_graph
        assert module.ghenv.Component.Message == "done, 1 response, tail: ray_tracing, order 1 (Run to redo)", inputs
    assert not messages


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
    assert names[0] == "scene" and names[-1] == "run"


def test_not_running_does_not_need_the_hybrid_extra(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    monkeypatch.setitem(sys.modules, "compas_acoustics.simulation_methods.hybrid", None)
    s = scene()
    assert run_hybrid(module, s, run=False).graph is s.ir_graph
    assert not messages


@needs_hybrid
def test_the_hybrid_component_passes_only_given_inputs(monkeypatch):
    module, _ = load_component("CA_Simulate_Hybrid", monkeypatch)
    calls = fake_hybrid(monkeypatch)
    run_hybrid(module, scene())
    run_hybrid(module, scene(), tail="eyring", diffraction="btms", sh_order=2)
    assert calls == [{}, {"tail": "eyring", "diffraction": "btms", "sh_order": 2}]
    assert "tail: eyring" in module.ghenv.Component.Message


@needs_hybrid
def test_auto_and_none_are_the_defaults(monkeypatch):
    module, _ = load_component("CA_Simulate_Hybrid", monkeypatch)
    calls = fake_hybrid(monkeypatch)
    run_hybrid(module, scene(), tail="auto", diffraction="none")
    assert calls == [{}]


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
    irs, *_ = module.DeconstructIRGraph().RunScript(s.ir_graph, None)
    assert irs.branches == {(0,): [ambisonic]}
    assert module.ghenv.Component.Message == "hybrid" and not messages


def test_the_deconstruct_inputs_match_its_script(monkeypatch):
    import inspect
    import json

    module, _ = load_component("CA_Deconstruct_IR_Graph", monkeypatch)
    metadata = json.loads((COMPONENTS / "CA_Deconstruct_IR_Graph" / "metadata.json").read_text())
    parameters = list(inspect.signature(module.DeconstructIRGraph.RunScript).parameters)[1:]
    assert parameters == [p["name"] for p in metadata["ghpython"]["inputParameters"]] == ["ir_graph", "receiver_idx"]


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

    module, messages = load_component("CA_Simulate_Ray_Trace", monkeypatch, levels=True)

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
        name, values, overall, frequencies = module.DeconstructParam().RunScript([t30, c50], select)
        assert name == ["Reverberation Time (T30)", "Clarity (C50)"] and overall == [1.7, -2.49]
        assert values.branches == {(0,): [0.984, 1.17, 1.94], (1,): [-1.46, 1.94, -2.58]}
        assert list(frequencies) == [125, 250, 1000]
    assert not messages


def test_deconstruct_parameter_gives_a_selected_parameter_as_plain_lists(monkeypatch):
    module, messages = load_component("CA_Deconstruct_IR", monkeypatch)
    t30, c50 = two_parameters(module)
    name, values, overall, frequencies = module.DeconstructParam().RunScript([t30, c50], "C50")
    assert (name, list(values), overall, list(frequencies)) == ("Clarity (C50)", [-1.46, 1.94, -2.58], -2.49, [125, 250, 1000])
    assert not messages


def test_deconstruct_parameter_warns_without_data_or_without_the_selection(monkeypatch):
    module, messages = load_component("CA_Deconstruct_IR", monkeypatch)
    t30, _ = two_parameters(module)
    for call in (([], None), (None, None), ([t30], "EDT")):
        assert module.DeconstructParam().RunScript(*call) == (None,) * 4
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
    assert [o["name"] for o in metadata["outputParameters"]] == ["name", "values", "overall", "frequencies"]
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
