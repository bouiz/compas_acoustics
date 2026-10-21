import numpy as np
import pytest

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import ImpulseResponse
from compas_acoustics import Pyroomacoustics
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.simulation_methods import SimulationContext
from compas_acoustics.simulation_methods import SimulationMethod

FS = 48000


def scene():
    s = Scene("methods")
    s.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", 0.2, 0.1)))
    s.add_source(Source("src", 1.5, 1.5, 1.6))
    s.add_receiver(Receiver("near", 3, 2, 1.5))
    s.add_receiver(Receiver("far", 6, 4, 1.5))
    return s


class Delta(SimulationMethod):
    """A free-field delta at the geometric delay: the smallest method meeting the contract."""

    name = "delta"
    output = "mono"

    def __init__(self, broken=None, channels=None):
        self.broken = broken
        self.channels = channels
        self.normalize = True

    def compute(self, context):
        fs = context.sampling_rate if self.broken != "rate" else context.sampling_rate // 2
        rirs = {}
        for source in context.sources:
            for receiver in context.receivers:
                distance = float(np.linalg.norm(np.subtract(receiver.position, source.position)))
                data = np.zeros(int(0.1 * fs))
                data[round(distance / 343.0 * fs)] = 1.0 / (4 * np.pi * distance) * (4 * np.pi if self.broken == "scale" else 1.0)
                if self.channels:
                    data = np.tile(data[:, None], (1, self.channels))
                generation = {"method": self.name, "scale": "package", "valid_range_hz": None}
                if self.broken == "generation":
                    generation.pop("method")
                key = context.key(source, receiver)
                rirs[key] = ImpulseResponse(key, source, receiver, data, fs,
                                            normalized=self.broken != "normalized", generation=generation)
        if self.broken == "missing":
            rirs.popitem()
        return rirs


class AmbisonicDelta(Delta):
    name = "ambisonic_delta"
    output = "ambisonic"

    def compute(self, context):
        rirs = Delta(channels=4).compute(context)
        for ir in rirs.values():
            ir.ambisonic_order, ir.normalization, ir.channel_order = 1, "sn3d", "acn"
            ir.generation["method"] = self.name
        return rirs


def test_a_second_method_keeps_the_first_on_the_graph():
    s = scene()
    first = s.simulate(Delta())
    s.simulate(AmbisonicDelta())
    s.simulate(AmbisonicDelta())
    assert s.impulse_responses("delta") == first
    assert sorted(s.simulation_methods()) == ["ambisonic_delta", "delta"]


def test_the_fake_methods_meet_the_contract(assert_meets_contract):
    s = scene()
    context = SimulationContext.from_scene(s)
    for method in (Delta(), AmbisonicDelta()):
        assert_meets_contract(s.simulate(method), context, method)


@pytest.mark.parametrize("broken", ["scale", "rate", "generation", "normalized", "missing"])
def test_each_broken_contract_point_fails(assert_meets_contract, broken):
    s = scene()
    method = Delta(broken=broken)
    with pytest.raises(AssertionError):
        assert_meets_contract(s.simulate(method), SimulationContext.from_scene(s), method)


def test_a_mono_method_with_channels_fails_the_contract(assert_meets_contract):
    s = scene()
    method = Delta(channels=4)
    with pytest.raises(AssertionError):
        assert_meets_contract(s.simulate(method), SimulationContext.from_scene(s), method)


def test_a_receiver_outside_the_room_warns_and_is_left_out():
    s = scene()
    s.add_receiver(Receiver("outside", 9, 2, 1.5))
    with pytest.warns(UserWarning, match="outside"):
        rirs = s.simulate(Delta())
    assert len(rirs) == 2


def test_simulate_stores_responses_under_the_method_name_and_data():
    s = scene()
    rirs = s.simulate(Delta())
    assert s.impulse_responses("delta") == rirs
    assert s.impulse_responses("data") == rirs
    assert s.simulation_methods() == ["delta"]


def test_simulate_leaves_data_alone_for_an_ambisonic_method():
    s = scene()
    mono = s.simulate(Delta())
    ambisonic = s.simulate(AmbisonicDelta())
    assert s.impulse_responses("data") == mono
    assert s.impulse_responses("ambisonic_delta") == ambisonic


def test_simulating_twice_replaces_the_responses():
    s = scene()
    s.simulate(Delta())
    second = s.simulate(Delta())
    assert s.impulse_responses("delta") == second
    assert len(s.impulse_responses("delta")) == len(second) == 2


def test_a_scene_simulates_at_48000_hz_unless_told_otherwise():
    assert scene().sampling_rate == 48000
    (first, *_) = scene().simulate(Delta()).values()
    assert first.sampling_rate == 48000


def test_the_scenes_sampling_rate_reaches_the_method():
    s = scene()
    s.sampling_rate = 44100
    (first, *_) = s.simulate(Delta()).values()
    assert first.sampling_rate == 44100
    assert SimulationContext.from_scene(s).sampling_rate == 44100


def test_a_whole_number_float_rate_is_accepted_and_stored_as_int():
    s = Scene("rate", sampling_rate=44100.0)
    assert s.sampling_rate == 44100 and isinstance(s.sampling_rate, int)


@pytest.mark.parametrize("rate", [44100.5, 0, -48000])
def test_a_rate_that_is_not_a_positive_whole_number_raises(rate):
    with pytest.raises(ValueError, match="whole number"):
        Scene("rate", sampling_rate=rate)
    s = scene()
    with pytest.raises(ValueError, match="whole number"):
        s.sampling_rate = rate
    assert s.sampling_rate == 48000


def traced_responses(**kwargs):
    method = Pyroomacoustics(ray_tracing=True, n_rays=2000, duration=0.5, **kwargs)
    return {ir.receiver.name: ir.denormalized() for ir in scene().simulate(method).values()}


def test_the_seed_is_true_42_by_default_false_unseeded_or_an_int():
    assert Pyroomacoustics().seed == 42 and Pyroomacoustics(seed=True).seed == 42 and Pyroomacoustics(seed=7).seed == 7
    assert Pyroomacoustics(seed=False).seed is None and Pyroomacoustics(seed=None).seed is None
    default, again, explicit, other = traced_responses(), traced_responses(), traced_responses(seed=42), traced_responses(seed=7)
    unseeded, unseeded_again = traced_responses(seed=False), traced_responses(seed=False)
    for name in default:
        np.testing.assert_array_equal(again[name], default[name])
        np.testing.assert_array_equal(explicit[name], default[name])
        assert not np.array_equal(other[name], default[name])
        assert not np.array_equal(unseeded[name], default[name]) and not np.array_equal(unseeded_again[name], unseeded[name])


def test_an_unseeded_run_records_no_seed():
    method = Pyroomacoustics(ray_tracing=True, n_rays=2000, duration=0.5, seed=False)
    assert all(ir.generation["seed"] is None for ir in scene().simulate(method).values())


@pytest.mark.parametrize("seed", [1.5, "42"])
def test_a_seed_that_is_not_a_bool_or_an_int_raises(seed):
    with pytest.raises(ValueError, match="seed"):
        Pyroomacoustics(seed=seed)


def test_a_scene_without_receivers_returns_nothing():
    s = Scene("empty")
    s.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(5, 4, 3, materials=AcousticMaterial("wall", 0.2, 0.1)))
    s.add_source(Source("s", 1, 1, 1.5))
    assert s.simulate(Delta()) == {}


def test_a_source_outside_the_room_warns_at_the_caller_and_the_rest_simulates():
    s = scene()
    s.add_source(Source("outside", 99, 2, 1.5))
    with pytest.warns(UserWarning, match="outside") as caught:
        rirs = s.simulate(Delta())
    assert len(rirs) == 2
    assert caught[0].filename.endswith("test_simulation_methods.py")


def test_simulate_raises_for_a_source_at_a_receiver():
    s = scene()
    s.add_receiver(Receiver("on_source", 1.5, 1.5, 1.6))
    with pytest.raises(ValueError, match="same position"):
        s.simulate(Delta())


def test_a_method_must_implement_compute():
    class Bare(SimulationMethod):
        name = "bare"

    with pytest.raises(TypeError):
        Bare()
