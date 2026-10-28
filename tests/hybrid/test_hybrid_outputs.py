"""The binaural and ambisonics methods over one hybrid core; the binaural late field per delay line."""

import sys
from pathlib import Path

import numpy as np
import pytest

import compas_acoustics as ca
from compas_acoustics.hybrid.binaural import DEFAULT_HRTF
from compas_acoustics.hybrid.binaural import HrtfSet
from compas_acoustics.hybrid.binaural import render_binaural
from compas_acoustics.hybrid.envelope import shape_late
from compas_acoustics.hybrid.ambisonics import ambisonic_output_matrix
from compas_acoustics.hybrid.late import fdn_directions
from compas_acoustics.hybrid.ambisonics import _from_n3d

sys.path.insert(0, str(Path(__file__).parent / "data"))
from hybrid_reference import AMBISONIC_ONLY  # noqa: E402
from hybrid_reference import CASES  # noqa: E402
from hybrid_reference import reference_binaural  # noqa: E402
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
    method = ca.Ambisonics(**dict(FAST, **kwargs))
    evaluation_scene().simulate(method)
    return one(method.results)


def deltas(directions):
    """An HRTF set of unit impulses at both ears for every direction: ideal ears."""
    hrirs = np.zeros((len(directions), 2, 8))
    hrirs[:, :, 0] = 1.0
    return HrtfSet(hrirs, np.asarray(directions, dtype=float), FS)


def test_binaural_has_two_ears_and_the_kemar_set():
    ir = one(evaluation_scene().simulate(ca.Binaural(**FAST)))
    assert ir.n_channels == 2 and ir.ambisonic_order is None and ir.format == "binaural"
    assert ir.generation["method"] == "binaural" and ir.generation["hrtf"] == DEFAULT_HRTF.name
    assert "output" not in ir.generation and "sh_order" not in ir.generation


def test_ambisonics_has_the_order_and_convention():
    ir = one(evaluation_scene().simulate(ca.Ambisonics(sh_order=2, normalization="N3D", **FAST)))
    assert ir.n_channels == 9 and ir.ambisonic_order == 2 and ir.normalization == "n3d" and ir.channel_order == "acn"
    assert ir.format == "ambisonic" and ir.generation["method"] == "ambisonics" and ir.generation["sh_order"] == 2
    assert "hrtf" not in ir.generation and "output" not in ir.generation


def test_both_methods_keep_each_other_and_data_on_the_graph():
    scene = evaluation_scene()
    mono = scene.simulate(ca.Mono(n_rays=2000, duration=0.3))
    binaural = scene.simulate(ca.Binaural(**FAST))
    ambisonics = scene.simulate(ca.Ambisonics(**FAST))
    assert scene.impulse_responses("binaural") == binaural and scene.impulse_responses("ambisonics") == ambisonics
    assert scene.impulse_responses("mono") == mono == scene.impulse_responses("data")


def test_every_method_has_kirschs_96_delay_lines():
    for method in (ca.Binaural(**FAST), ca.Ambisonics(**FAST), ca.Ambisonics(sh_order=3, **FAST)):
        ir = one(evaluation_scene().simulate(method))
        assert method.num_delays == 96 and ir.generation["num_delays"] == 96
        assert one(method.results).late_lines.shape[1] == 96
        assert one(method.results).core.drive_gain == pytest.approx(np.sqrt(96))


def test_fourth_order_takes_more_lines():
    assert ca.Ambisonics(sh_order=4).num_delays > 96


@pytest.mark.parametrize("make", [
    lambda: ca.Binaural(output="binaural"), lambda: ca.Binaural(sh_order=1), lambda: ca.Binaural(normalization="N3D"),
    lambda: ca.Ambisonics(hrtf="kemar"), lambda: ca.Ambisonics(output="ambisonic")])
def test_the_other_formats_arguments_are_rejected(make):
    with pytest.raises(TypeError):
        make()


def test_both_methods_share_one_late_field():
    ears, channels = ca.Binaural(seed=3, **FAST), ca.Ambisonics(seed=3, **FAST)
    evaluation_scene().simulate(ears)
    evaluation_scene().simulate(channels)
    np.testing.assert_allclose(one(ears.results).late_lines, one(channels.results).late_lines, atol=1e-12)
    assert one(ears.results).rir.shape[1] == 2 and one(channels.results).rir.shape[1] == 4


def test_both_methods_meet_the_contract(assert_meets_contract):
    from compas_acoustics.simulation_methods import SimulationContext

    scene = evaluation_scene()
    binaural, ambisonics = ca.Binaural(**FAST), ca.Ambisonics(**FAST)
    ears, channels = scene.simulate(binaural), scene.simulate(ambisonics)
    context = SimulationContext.from_scene(scene)
    # the ears' direct sound is the free field's through an HRIR, not 1 / (4 pi r)
    assert_meets_contract(ears, context, binaural, direct_sound=False)
    assert_meets_contract(channels, context, ambisonics)


def test_the_hrtf_is_kemar_or_a_sofa_file():
    assert ca.Binaural().hrtf == "kemar"
    ir = one(evaluation_scene().simulate(ca.Binaural(hrtf=str(DEFAULT_HRTF), **FAST)))
    assert ir.generation["hrtf"] == DEFAULT_HRTF.name
    with pytest.raises(ValueError, match="hrtf"):
        ca.Binaural(hrtf="/nowhere.sofa")


@pytest.mark.parametrize("normalization", ["SN3D", "N3D"])
def test_the_ambisonic_late_field_is_the_lines_through_the_output_matrix(normalization):
    result = run(normalization=normalization)
    n = result.late_lines.shape[1]
    directions = fdn_directions(n)
    C = ambisonic_output_matrix(1, directions)
    np.testing.assert_allclose(result.core.line_directions, directions.cartesian / np.linalg.norm(directions.cartesian, axis=1, keepdims=True))
    late = result.late_lines @ C.T
    assert np.allclose(result.late, _from_n3d(late, normalization), rtol=0, atol=1e-12 * np.abs(late).max())


@pytest.mark.parametrize("n_directions", [1, 200])
def test_per_line_binaural_with_ideal_ears_is_w(n_directions):
    result = run()
    golden = np.pi * (3.0 - np.sqrt(5.0)) * np.arange(n_directions)
    z = 1.0 - 2.0 * (np.arange(n_directions) + 0.5) / n_directions
    directions = np.stack([np.sqrt(1 - z ** 2) * np.cos(golden), np.sqrt(1 - z ** 2) * np.sin(golden), z], axis=1)
    w = result.early[:, 0] + result.late[:, 0]
    ears = render_binaural(result.core, deltas(directions)).rir
    assert np.allclose(ears[:, 0], w, rtol=0, atol=1e-9 * np.abs(w).max())
    np.testing.assert_array_equal(ears[:, 0], ears[:, 1])


def test_shape_late_gives_every_column_the_gains_of_w():
    """Shaping the lines and then projecting is shaping the ambisonic channels: the gains depend on W only."""
    result = run(normalization="N3D")
    raw = result.late_lines
    C = ambisonic_output_matrix(1, fdn_directions(raw.shape[1]))
    early_w, late_w = result.early[:, 0], raw @ C[0]
    reference = early_w + 0.5 * late_w
    lines = shape_late(raw, early_w, late_w, reference, FS)
    ambisonic = shape_late(raw @ C.T, early_w, late_w, reference, FS)
    assert lines.shape == raw.shape
    assert np.allclose(lines @ C.T, ambisonic, rtol=0, atol=1e-12 * np.abs(ambisonic).max())
    assert np.sum(lines ** 2) < np.sum(raw ** 2)


def _comparable_here(case):
    # the references were stored on macOS; pyroomacoustics' seeded ray tracer draws another stream elsewhere
    # (1-3.5 % of the peak on Linux), while every other case agrees within 2.3e-4 of the peak
    if {**CASES, **AMBISONIC_ONLY}[case].get("tail") == "ray_tracing" and sys.platform != "darwin":
        pytest.skip("a ray-traced reference is reproducible only on the platform that stored it")


@pytest.mark.parametrize("case", sorted({**CASES, **AMBISONIC_ONLY}))
def test_the_ambisonic_output_is_unchanged_since_3b9333d(case):
    """The code before the split into the binaural and ambisonics methods (data/hybrid_reference.py) gives the same response."""
    _comparable_here(case)
    before = np.load(DATA / "ambisonic_reference_3b9333d.npz")[case]
    after = reference_rir(**{**CASES, **AMBISONIC_ONLY}[case])
    assert after.shape == before.shape and np.any(after[-200:] != 0.0)
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-3 * np.abs(before).max())


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_binaural_output_is_unchanged_since_3b9333d(case):
    """The same for the ears, within 1e-3 of the peak as BTMS needs (the binaural BTMS takes the ambisonic encoder after the split)."""
    _comparable_here(case)
    before = np.load(DATA / "binaural_reference_3b9333d.npz")[case]
    after = reference_binaural(**CASES[case])
    assert after.shape == before.shape and np.any(after[-200:] != 0.0)
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-3 * np.abs(before).max())


def direct_ild_db(ir):
    """Left minus right ear level in dB over the 3 ms from the direct sound, before the first reflection."""
    ears = ir.denormalized()
    start = int(np.flatnonzero(np.abs(ears).max(axis=1) > 1e-3 * np.abs(ears).max())[0])
    window = ears[start:start + int(0.003 * FS)]
    left, right = np.sum(window ** 2, axis=0)
    return 10 * np.log10(left / right)


def test_the_source_on_the_left_is_louder_at_the_left_ear():
    # The source at (4, 5) is at +y of the receiver at (4, 2): on its left facing +x, on its right facing -x
    facing = one(evaluation_scene().simulate(ca.Binaural(**FAST)))
    turned = one(evaluation_scene(direction=[-1, 0, 0]).simulate(ca.Binaural(**FAST)))
    assert direct_ild_db(facing) > 10.0 and direct_ild_db(turned) < -10.0


def test_settings_changed_after_construction_are_the_ones_rendered_and_recorded(tmp_path):
    """A method object is reused between runs in notebooks: what it renders and what it records must agree."""
    import shutil

    own = tmp_path / "own.sofa"
    shutil.copy(DEFAULT_HRTF, own)
    ears = ca.Binaural(**FAST)
    evaluation_scene().simulate(ears)
    kemar = ears._hrtf_set
    ears.hrtf = str(own)
    ir = one(evaluation_scene().simulate(ears))
    assert ir.generation["hrtf"] == "own.sofa" and ears._hrtf_set is not kemar
    channels = ca.Ambisonics(**FAST)
    evaluation_scene().simulate(channels)
    channels.sh_order = 4
    ir = one(evaluation_scene().simulate(channels))
    assert ir.n_channels == 25 and ir.generation["num_delays"] == channels.num_delays == ca.Ambisonics(sh_order=4).num_delays > 96
