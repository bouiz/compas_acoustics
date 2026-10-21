"""The hybrid's ambisonic and binaural outputs, rendered from one run; the binaural late field per delay line."""

import sys
from pathlib import Path

import numpy as np
import pytest

import compas_acoustics as ca
from compas_acoustics.binaural import DEFAULT_HRTF
from compas_acoustics.binaural import HrtfSet
from compas_acoustics.binaural import binaural_rir
from compas_acoustics.hybrid.envelope import shape_late
from compas_acoustics.hybrid.late import ambisonic_output_matrix
from compas_acoustics.hybrid.late import fdn_directions
from compas_acoustics.hybrid.synthesis import _from_n3d

sys.path.insert(0, str(Path(__file__).parent / "data"))
from hybrid_reference import CASES  # noqa: E402
from hybrid_reference import reference_rir  # noqa: E402

FS = 48000
DATA = Path(__file__).parent / "data"
FAST = dict(tail="eyring", duration=0.3)


def evaluation_scene(direction=None):
    """The evaluation's shoebox: 8 x 6 x 3.5 m, alpha 0.15, s 0.2."""
    scene = ca.Scene("outputs", sampling_rate=FS)
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3.5, materials=ca.AcousticMaterial("wall", 0.15, 0.2)))
    scene.add_source(ca.Source("src", 4.0, 5.0, 1.5))
    scene.add_receiver(ca.Receiver("rec", 4.0, 2.0, 1.5, direction=direction))
    return scene


def one(responses):
    (ir,) = responses.values()
    return ir


def run(**kwargs):
    hybrid = ca.Hybrid(**dict(FAST, **kwargs))
    evaluation_scene().simulate(hybrid)
    (result,) = hybrid.results.values()
    return result


def deltas(directions):
    """An HRTF set of unit impulses at both ears for every direction: ideal ears."""
    hrirs = np.zeros((len(directions), 2, 8))
    hrirs[:, :, 0] = 1.0
    return HrtfSet(hrirs, np.asarray(directions, dtype=float), FS)


def test_every_output_has_kirschs_96_delay_lines():
    for output in ("binaural", "ambisonic", ["ambisonic", "binaural"]):
        hybrid = ca.Hybrid(**FAST, output=output)
        ir = one(evaluation_scene().simulate(hybrid))
        assert ir.generation["num_delays"] == 96
        assert one(hybrid.results).late_lines.shape[1] == 96
        assert one(hybrid.results).drive_gain == pytest.approx(np.sqrt(96))


def test_the_default_output_is_binaural():
    ir = one(evaluation_scene().simulate(ca.Hybrid(**FAST)))
    assert ir.n_channels == 2 and ir.ambisonic_order is None
    assert ir.generation["output"] == "binaural" and ir.generation["hrtf"] == DEFAULT_HRTF.name
    assert ir.generation["method"] == "hybrid"


def test_one_run_gives_both():
    scene = evaluation_scene()
    hybrid = ca.Hybrid(output=["ambisonic", "binaural"], **FAST)
    out = scene.simulate(hybrid)
    ambisonic = one(out)
    assert ambisonic.n_channels == 4 and ambisonic.generation["output"] == "ambisonic"
    assert scene.impulse_responses("hybrid") == out
    (binaural,) = scene.impulse_responses("hybrid_binaural").values()
    assert binaural.n_channels == 2 and binaural.generation["output"] == "binaural"
    assert binaural.denormalized().shape[0] == ambisonic.denormalized().shape[0]
    assert scene.impulse_responses("data") == {}
    assert hybrid.outputs == ("ambisonic", "binaural") and hybrid.attribute == "hybrid"
    assert set(hybrid.extra_responses) == {"hybrid_binaural"}


def test_the_first_format_is_returned_and_the_second_is_extra():
    scene = evaluation_scene()
    hybrid = ca.Hybrid(output=["binaural", "ambisonic"], **FAST)
    out = scene.simulate(hybrid)
    assert one(out).n_channels == 2 and hybrid.attribute == "hybrid_binaural"
    assert scene.impulse_responses("hybrid_binaural") == out
    assert one(scene.impulse_responses("hybrid")).n_channels == 4


def test_binaural_only_stores_and_returns_binaural():
    scene = evaluation_scene()
    out = scene.simulate(ca.Hybrid(output="binaural", **FAST))
    assert scene.impulse_responses("hybrid") == {} and scene.impulse_responses("data") == {}
    assert one(out).n_channels == 2
    assert scene.impulse_responses("hybrid_binaural") == out


def test_both_outputs_meet_the_contract(assert_meets_contract):
    from compas_acoustics.simulation_methods import SimulationContext

    scene = evaluation_scene()
    hybrid = ca.Hybrid(output=["binaural", "ambisonic"], **FAST)
    binaural = scene.simulate(hybrid)
    context = SimulationContext.from_scene(scene)
    # the ears' direct sound is the free field's through an HRIR, not 1 / (4 pi r)
    assert_meets_contract(binaural, context, hybrid, direct_sound=False)
    assert_meets_contract(hybrid.extra_responses["hybrid"], context, hybrid)


def test_ambisonic_only_is_as_before():
    scene = evaluation_scene()
    hybrid = ca.Hybrid(output="ambisonic", **FAST)
    out = scene.simulate(hybrid)
    assert one(out).n_channels == 4 and one(out).generation["output"] == "ambisonic"
    assert scene.impulse_responses("hybrid") == out and scene.impulse_responses("hybrid_binaural") == {}
    assert hybrid.extra_responses == {} and "hrtf" not in one(out).generation


def test_hrtf_without_binaural_raises():
    with pytest.raises(ValueError, match="hrtf"):
        ca.Hybrid(output="ambisonic", hrtf="kemar")


@pytest.mark.parametrize("output", ["stereo", [], ["binaural", "binaural"], ["ambisonic", "mono"]])
def test_an_unknown_or_repeated_output_raises(output):
    with pytest.raises(ValueError, match="output"):
        ca.Hybrid(output=output)


def test_the_hrtf_is_kemar_or_a_sofa_file():
    assert ca.Hybrid().hrtf == "kemar" and ca.Hybrid(output="ambisonic").hrtf is None
    ir = one(evaluation_scene().simulate(ca.Hybrid(hrtf=str(DEFAULT_HRTF), **FAST)))
    assert ir.generation["hrtf"] == DEFAULT_HRTF.name


@pytest.mark.parametrize("normalization", ["SN3D", "N3D"])
def test_both_outputs_share_one_late_field(normalization):
    result = run(normalization=normalization)
    n = result.late_lines.shape[1]
    directions = fdn_directions(1, n)
    C = ambisonic_output_matrix(1, directions)
    np.testing.assert_allclose(result.line_directions, directions.cartesian / np.linalg.norm(directions.cartesian, axis=1, keepdims=True))
    late = result.late_lines @ C.T
    assert np.allclose(result.late, _from_n3d(late, normalization), rtol=0, atol=1e-12 * np.abs(late).max())


@pytest.mark.parametrize("n_directions", [1, 200])
def test_per_line_binaural_with_ideal_ears_is_w(n_directions):
    result = run()
    golden = np.pi * (3.0 - np.sqrt(5.0)) * np.arange(n_directions)
    z = 1.0 - 2.0 * (np.arange(n_directions) + 0.5) / n_directions
    directions = np.stack([np.sqrt(1 - z ** 2) * np.cos(golden), np.sqrt(1 - z ** 2) * np.sin(golden), z], axis=1)
    w = result.early[:, 0] + result.late[:, 0]
    ears = binaural_rir(result, deltas(directions))
    assert np.allclose(ears[:, 0], w, rtol=0, atol=1e-9 * np.abs(w).max())
    np.testing.assert_array_equal(ears[:, 0], ears[:, 1])


def test_shape_late_gives_every_column_the_gains_of_w():
    """Shaping the lines and then projecting is shaping the ambisonic channels: the gains depend on W only."""
    result = run(normalization="N3D")
    C = ambisonic_output_matrix(1, fdn_directions(1, result.late_lines.shape[1]))
    early_w, late_w = result.early[:, 0], result.late_lines @ C[0]
    reference = early_w + 0.5 * late_w
    lines = shape_late(result.late_lines, early_w, late_w, reference, FS)
    ambisonic = shape_late(result.late_lines @ C.T, early_w, late_w, reference, FS)
    assert lines.shape == result.late_lines.shape
    assert np.allclose(lines @ C.T, ambisonic, rtol=0, atol=1e-12 * np.abs(ambisonic).max())
    assert np.sum(lines ** 2) < np.sum(result.late_lines ** 2)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_ambisonic_output_is_unchanged(case):
    """The code at cb644a8 (96 lines, the hemisphere drive; data/hybrid_reference.py) gives the same response."""
    before = np.load(DATA / "hybrid_reference_cb644a8.npz")[case]
    after = reference_rir(output="ambisonic", **CASES[case])
    assert np.any(after[-200:] != 0.0)  # the late field is inside the window
    assert np.allclose(after, before, rtol=0, atol=1e-10 * np.abs(before).max())


def direct_ild_db(ir):
    """Left minus right ear level in dB over the 3 ms from the direct sound, before the first reflection."""
    ears = ir.denormalized()
    start = int(np.flatnonzero(np.abs(ears).max(axis=1) > 1e-3 * np.abs(ears).max())[0])
    window = ears[start:start + int(0.003 * FS)]
    left, right = np.sum(window ** 2, axis=0)
    return 10 * np.log10(left / right)


def test_the_source_on_the_left_is_louder_at_the_left_ear():
    # The source at (4, 5) is at +y of the receiver at (4, 2): on its left facing +x, on its right facing -x
    facing = one(evaluation_scene().simulate(ca.Hybrid(**FAST)))
    turned = one(evaluation_scene(direction=[-1, 0, 0]).simulate(ca.Hybrid(**FAST)))
    assert direct_ild_db(facing) > 10.0 and direct_ild_db(turned) < -10.0
