"""Grasshopper components, run outside Rhino with Grasshopper, Rhino and ghenv stubbed."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source

needs_hybrid = pytest.mark.skipif(importlib.util.find_spec("pyFDN") is None, reason="needs the hybrid extra (Python >= 3.11)")

COMPONENTS = Path(__file__).parents[1] / "src" / "compas_acoustics_ghpython" / "components"


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
    kernel_data = types.ModuleType("Grasshopper.Kernel.Data")
    kernel_data.GH_Path = lambda *indices: indices
    rhino = types.ModuleType("Rhino")
    rhino.Geometry = types.SimpleNamespace(Point3d=lambda x, y, z: (x, y, z), Vector3d=lambda x, y, z: (x, y, z),
                                           Line=lambda *coordinates: coordinates, Plane=plane)
    rhino_geometry = types.ModuleType("Rhino.Geometry")
    rhino_geometry.Line = rhino.Geometry.Line
    for module_name, module in (("Grasshopper", grasshopper), ("Grasshopper.Kernel", types.ModuleType("Grasshopper.Kernel")),
                                ("Grasshopper.Kernel.Data", kernel_data), ("Rhino", rhino), ("Rhino.Geometry", rhino_geometry),
                                ("System", types.ModuleType("System")), ("scriptcontext", types.SimpleNamespace(sticky={}))):
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

    def simulate(self, method, sampling_rate=48000):
        calls.append(vars(method).copy())
        return {}

    monkeypatch.setattr(Scene, "simulate", simulate)
    return calls


def method_defaults(**changes):
    from compas_acoustics.methods import Pyroomacoustics

    return dict(vars(Pyroomacoustics(**changes)))


def test_empty_inputs_leave_the_method_defaults(monkeypatch):
    module, _ = load_component("CA_Simulate_Ray_Trace", monkeypatch)
    calls = capture(monkeypatch)
    module.SimulateRayTracing().RunScript(scene(), None, None, None, None, None, None, None, None, None)
    assert calls == [method_defaults(ray_tracing=True)]


def test_given_inputs_are_passed_through(monkeypatch):
    module, _ = load_component("CA_Simulate_Ray_Trace", monkeypatch)
    calls = capture(monkeypatch)
    module.SimulateRayTracing().RunScript(scene(), None, 4, True, 5000, False, 1e-7, 2.0, 3, False)
    assert calls == [method_defaults(image_source_order=4, ray_tracing=True, n_rays=5000, air_absorption=False,
                                     energy_thres=1e-7, duration=2.0, seed=3, normalize=False)]


def test_without_ray_tracing_the_ray_inputs_are_dropped_with_a_remark(monkeypatch):
    module, messages = load_component("CA_Simulate_Ray_Trace", monkeypatch)
    calls = capture(monkeypatch)
    module.SimulateRayTracing().RunScript(scene(), None, 3, False, 5000, None, None, None, 3, None)
    assert calls == [method_defaults(image_source_order=3, ray_tracing=False)]
    assert messages and "n_rays" in messages[0]


def test_a_float_sample_rate_is_accepted(monkeypatch):
    module, _ = load_component("CA_Simulate_Ray_Trace", monkeypatch)
    capture(monkeypatch)
    module.SimulateRayTracing().RunScript(scene(), 44100.0, None, None, None, None, None, None, None, None)
    assert module.ghenv.Component.Message.startswith("44100Hz")


def test_a_missing_scene_warns(monkeypatch):
    module, messages = load_component("CA_Simulate_Ray_Trace", monkeypatch)
    assert module.SimulateRayTracing().RunScript(None, None, None, None, None, None, None, None, None, None) is None
    assert messages and "scene" in messages[0]


# ------------------------------------------------------- Deconstruct IR Graph --
def graph_with_methods():
    from compas_acoustics import ImpulseResponse

    s = scene()
    source, receiver = s.sources[0], s.receivers[0]
    mono = ImpulseResponse("m", source, receiver, np.ones(4), 48000)
    ambisonic = ImpulseResponse("a", source, receiver, np.ones((4, 4)), 48000, ambisonic_order=1, normalisation="sn3d", channel_order="acn")
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
    irs, *_ = module.DeconstructIRGraph().RunScript(s.ir_graph, None, "hybrid")
    assert irs.branches.get((0,), []) == []
    assert messages and "pyroomacoustics" in messages[0]


def test_deconstruct_reads_a_named_method(monkeypatch):
    module, _ = load_component("CA_Deconstruct_IR_Graph", monkeypatch)
    s, _, ambisonic = graph_with_methods()
    s.populate_ir_graph([ambisonic], attribute="hybrid")
    irs, *_ = module.DeconstructIRGraph().RunScript(s.ir_graph, None, "hybrid")
    assert irs.branches == {(0,): [ambisonic]}


# ---------------------------------------------------------- Simulate Hybrid --
def fake_hybrid(monkeypatch, warn=None):
    """Stand-in for compas_acoustics.methods.Hybrid, so the component is tested without pyFDN."""
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

    def simulate(self, method, sampling_rate=48000):
        if warn:
            warnings.warn(warn, UserWarning)
        return {}

    import compas_acoustics.methods
    monkeypatch.setattr(compas_acoustics.methods, "Hybrid", Hybrid, raising=False)
    monkeypatch.setattr(Scene, "simulate", simulate)
    return calls


def run_hybrid(module, s, **inputs):
    names = ("tail", "reference", "sample_rate", "sh_order", "image_source_order", "diffraction", "late_envelope", "seed", "normalize")
    return module.SimulateHybrid().RunScript(s, *(inputs.get(n) for n in names))


def test_a_missing_hybrid_extra_is_an_error_not_a_traceback(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    monkeypatch.setitem(sys.modules, "compas_acoustics.methods.hybrid", None)
    assert run_hybrid(module, scene()) is None
    assert messages and messages[0][0] == "error" and "hybrid" in messages[0][1]


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
def test_the_sample_rate_follows_the_reference(monkeypatch):
    module, messages = load_component("CA_Simulate_Hybrid", monkeypatch, levels=True)
    fake_hybrid(monkeypatch)
    s, _, _ = graph_with_methods()
    assert run_hybrid(module, s, reference="pyroomacoustics", sample_rate=44100) is None
    assert messages and messages[0][0] == "error" and "48000" in messages[0][1]


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


@needs_hybrid
def test_the_hybrid_reference_can_be_the_ray_tracing_graph_wire(monkeypatch):
    module, _ = load_component("CA_Simulate_Hybrid", monkeypatch)
    calls = fake_hybrid(monkeypatch)
    s, mono, _ = graph_with_methods()
    run_hybrid(module, s, reference=s.ir_graph)
    (call,) = calls
    assert call["reference"] == {f"{mono.receiver.guid}_{mono.source.guid}": mono}


def test_the_ray_tracing_component_shows_warnings(monkeypatch):
    import warnings

    module, messages = load_component("CA_Simulate_Ray_Trace", monkeypatch, levels=True)

    def simulate(self, method, sampling_rate=48000):
        warnings.warn("Receiver 'x' is outside the room and will not be simulated.", UserWarning)
        return {}

    monkeypatch.setattr(Scene, "simulate", simulate)
    module.SimulateRayTracing().RunScript(scene(), None, None, None, None, None, None, None, None, None)
    assert ("warning", "Receiver 'x' is outside the room and will not be simulated.") in messages
