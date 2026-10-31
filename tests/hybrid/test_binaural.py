"""Binaural rendering of the hybrid: an HRTF per early path and per delay line."""

import numpy as np
import pytest
from compas.datastructures import Mesh

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.simulation_methods import Binaural

FS = 48000
WALL = AcousticMaterial("wall", [0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35], 0.1)
FAST = dict(tail="eyring", image_source_order=2, duration=0.3)


def scene(receiver=None, panel=False):
    s = Scene("binaural")
    s.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(10, 8, 4, materials=WALL))
    if panel:
        mesh = Mesh.from_vertices_and_faces([[5, 3, 0.5], [5, 5, 0.5], [5, 5, 2.5], [5, 3, 2.5]], [[0, 1, 2, 3]])
        s.add_acoustic_object(AcousticObject(mesh, WALL, name="panel", role="obstacle"))
    s.add_source(Source("src", 2.0, 4.0, 1.5))
    s.add_receiver(receiver or Receiver("rec", 8.0, 4.2, 1.5))
    return s


def run(method, **kwargs):
    s = scene(**kwargs)
    s.simulate(method)
    (result,) = method.results.values()
    return result


def test_the_core_keeps_the_early_images():
    core = run(Binaural(**FAST)).core
    assert core.images.order.max() == core.max_order
    assert core.diffraction_paths == () and core.btms_contributions == ()


def test_the_core_keeps_the_diffracted_paths():
    assert len(run(Binaural(diffraction=True, **FAST), panel=True).core.diffraction_paths) > 0


def test_the_nearest_measured_direction_is_picked():
    from compas_acoustics.hybrid.binaural import DEFAULT_HRTF
    from compas_acoustics.hybrid.binaural import HrtfSet

    hrtf = HrtfSet.from_sofa(DEFAULT_HRTF, FS)
    assert hrtf.hrirs.shape[1] == 2 and hrtf.directions.shape == (hrtf.hrirs.shape[0], 3)
    picked = hrtf.nearest(hrtf.directions[[0, 100, 500]] * 3.0)
    np.testing.assert_array_equal(picked, [0, 100, 500])


def test_a_set_at_another_rate_is_resampled():
    from compas_acoustics.hybrid.binaural import DEFAULT_HRTF
    from compas_acoustics.hybrid.binaural import HrtfSet

    native, resampled = HrtfSet.from_sofa(DEFAULT_HRTF, FS), HrtfSet.from_sofa(DEFAULT_HRTF, 44100)
    assert resampled.sampling_rate == 44100
    assert resampled.hrirs.shape[2] == int(np.ceil(native.hrirs.shape[2] * 44100 / FS))  # pyfar rounds up


def identity(n_dir=1):
    """An HRTF set that passes every direction through unchanged to both ears."""
    from compas_acoustics.hybrid.binaural import HrtfSet

    hrirs = np.zeros((n_dir, 2, 8))
    hrirs[:, :, 0] = 1.0
    return HrtfSet(hrirs, np.tile([[1.0, 0.0, 0.0]], (n_dir, 1)), FS)


@pytest.mark.parametrize("diffraction", [False, True, "btms"])
def test_with_a_transparent_hrtf_both_ears_hear_the_omnidirectional_early_part(diffraction):
    """Same paths, same encoders, same band interpolation, delays and taper: the core's W exactly, BTMS included."""
    from compas_acoustics.hybrid.binaural import render_binaural
    from compas_acoustics.hybrid.ambisonics import render_ambisonics

    core = run(Binaural(diffraction=diffraction, **FAST), panel=bool(diffraction)).core
    assert diffraction != "btms" or len(core.btms_contributions) > 0
    ears = render_binaural(core, identity()).early
    w = render_ambisonics(core).early[:, 0]
    np.testing.assert_allclose(ears[:, 0], w, rtol=0, atol=1e-12 * np.max(np.abs(w)))
    np.testing.assert_array_equal(ears[:, 0], ears[:, 1])


def test_the_binaural_response_is_early_plus_each_line_through_its_hrir():
    from scipy.signal import fftconvolve
    from compas_acoustics.hybrid.binaural import DEFAULT_HRTF
    from compas_acoustics.hybrid.binaural import HrtfSet, render_binaural

    core = run(Binaural(**FAST)).core
    hrtf = HrtfSet.from_sofa(DEFAULT_HRTF, FS)
    result = render_binaural(core, hrtf)
    n, lines = core.n_samples, result.late_lines.shape[1]
    late = sum(fftconvolve(result.late_lines[:, i:i + 1] / np.sqrt(lines), hrtf.hrirs[hrtf.nearest(core.line_directions[i])[0]].T,
                           axes=0)[:n] for i in range(lines))
    assert result.rir.shape == (n, 2) and result.normalization is None and result.core is core
    np.testing.assert_allclose(result.late, late, atol=1e-12)
    np.testing.assert_allclose(result.rir, result.early + result.late, atol=0)


def test_the_hrtf_must_be_at_the_simulations_rate():
    from compas_acoustics.hybrid.binaural import DEFAULT_HRTF
    from compas_acoustics.hybrid.binaural import HrtfSet, render_binaural

    core = run(Binaural(**FAST)).core
    with pytest.raises(ValueError, match="sample rate"):
        render_binaural(core, HrtfSet.from_sofa(DEFAULT_HRTF, 44100))


def test_binaural_returns_two_channel_responses_of_the_cores_length():
    method = Binaural(**FAST)
    (ir,) = scene().simulate(method).values()
    assert ir.n_channels == 2 and ir.ambisonic_order is None and ir.format == "binaural"
    assert ir.generation["hrtf"] == "KEMAR_FreeFieldComp_48kHz.sofa" and ir.generation["method"] == "binaural"
    (result,) = method.results.values()
    assert ir.denormalized().shape == (result.core.n_samples, 2)
