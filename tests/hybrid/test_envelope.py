"""The ray-traced late envelope imposed on the network output.

Ported from hybrid-rir (gpapadimitri/hybrid-rir @ a272da7, tests/test_envelope.py).
"""

import numpy as np
import pytest

from _compat import Material, RoomSpec, ShoeboxGeometry, synthesize
from compas_acoustics.hybrid.envelope import envelope_gains, octave_band_split, shape_late

FS = 48000.0
N = int(1.6 * FS)
T = np.arange(N) / FS


def parts(rt=0.8, onset=0.03, seed=0):
    """A few early impulses, and a decaying 4-channel noise from ``onset``."""
    rng = np.random.default_rng(seed)
    early = np.zeros((N, 4))
    for t, a in ((0.01, 1.0), (0.015, 0.5), (0.022, 0.4), (0.027, 0.3)):
        early[int(t * FS), 0] = a
    decay = 10 ** (-3 * np.clip(T - onset, 0.0, None) / rt) * (T >= onset)
    late = 0.02 * rng.standard_normal((N, 4)) * decay[:, None]
    return early, late


def energy_db(x, a, b):
    return 10 * np.log10(np.sum(x[int(a * FS):int(b * FS)] ** 2))


# ----------------------------------------------------------------- the bank --
def test_the_band_split_sums_back_to_the_input():
    x = np.random.default_rng(1).standard_normal((4000, 3))
    bands = octave_band_split(x, FS)
    assert bands.shape == (9, 4000, 3)
    np.testing.assert_allclose(bands.sum(axis=0), x, atol=1e-6)


def test_the_band_split_is_pyfar_reconstructing_bank():
    """pyfar's linear-phase bank applied by pyfar itself, its delay removed, to 1e-12; one channel and several."""
    import pyfar as pf

    from compas_acoustics.hybrid.envelope import _N_FILT, _TOP_BAND_HZ

    x = np.random.default_rng(2).standard_normal((3000, 4)) * np.exp(-np.arange(3000) / 600.0)[:, None]
    padded = np.concatenate([x.T, np.zeros((4, _N_FILT))], axis=1)
    out = pf.dsp.filter.reconstructing_fractional_octave_bands(
        pf.Signal(padded, FS), num_fractions=1, frequency_range=(63, _TOP_BAND_HZ), n_samples=_N_FILT)
    filtered = out[0] if isinstance(out, tuple) else out
    delay = _N_FILT // 2
    expected = np.transpose(filtered.time[:, :, delay:delay + 3000], (0, 2, 1))
    bands = octave_band_split(x, FS)
    np.testing.assert_allclose(bands, expected, rtol=0, atol=1e-12 * np.max(np.abs(expected)))
    np.testing.assert_allclose(octave_band_split(x[:, 0], FS), expected[:, :, 0], rtol=0, atol=1e-12 * np.max(np.abs(expected)))


# ---------------------------------------------------------------- the gains --
def test_gains_never_exceed_one():
    early, late = parts()
    louder = early[:, 0] + 3.0 * late[:, 0]
    g = envelope_gains(louder, early[:, 0], late[:, 0], FS)
    assert g.shape == (9, N)
    assert g.max() <= 1.0


def test_a_louder_reference_leaves_the_late_field_alone():
    early, late = parts()
    shaped = shape_late(late, early[:, 0], late[:, 0], early[:, 0] + 3.0 * late[:, 0], FS)
    np.testing.assert_allclose(shaped, late, atol=1e-6)


def test_a_quieter_reference_attenuates_by_the_difference():
    early, late = parts()
    reference = early[:, 0] + 10 ** (-6 / 20) * late[:, 0]
    shaped = shape_late(late, early[:, 0], late[:, 0], reference, FS)
    lost = energy_db(shaped[:, 0], 0.15, 0.6) - energy_db(late[:, 0], 0.15, 0.6)
    assert lost == pytest.approx(-6.0, abs=0.5)


def test_every_channel_gets_the_same_gain():
    early, late = parts()
    reference = early[:, 0] + 0.5 * late[:, 0]
    shaped = shape_late(late, early[:, 0], late[:, 0], reference, FS)
    ratio = [energy_db(shaped[:, c], 0.15, 0.6) - energy_db(late[:, c], 0.15, 0.6)
             for c in range(4)]
    assert np.ptp(ratio) < 0.5


def test_the_gain_is_held_once_the_reference_has_decayed_20_db():
    """A burst in the reference after the hold point does not reach the output."""
    early, late = parts()
    reference = early[:, 0] + 0.5 * late[:, 0]
    burst = reference.copy()
    i = int(1.0 * FS)
    burst[i:i + int(0.02 * FS)] *= 30.0
    plain = shape_late(late, early[:, 0], late[:, 0], reference, FS)
    bursty = shape_late(late, early[:, 0], late[:, 0], burst, FS)
    np.testing.assert_allclose(bursty[int(0.9 * FS):], plain[int(0.9 * FS):], atol=1e-9)


def test_a_short_reference_holds_its_last_gain():
    early, late = parts()
    reference = (early[:, 0] + 0.5 * late[:, 0])[: int(0.8 * FS)]
    shaped = shape_late(late, early[:, 0], late[:, 0], reference, FS)
    assert shaped.shape == late.shape
    tail = energy_db(shaped[:, 0], 1.0, 1.4) - energy_db(late[:, 0], 1.0, 1.4)
    assert tail == pytest.approx(-6.0, abs=1.0)


def test_a_reference_on_another_scale_warns():
    early, late = parts()
    reference = 100.0 * (early[:, 0] + late[:, 0])
    with pytest.warns(UserWarning, match="scale"):
        shape_late(late, early[:, 0], late[:, 0], reference, FS)


# -------------------------------------------------------------- the pipeline --
def shoebox():
    mat = Material.from_scalar(0.2)
    return RoomSpec(
        geometry=ShoeboxGeometry((7.0, 5.0, 3.0), {n: mat for n in ShoeboxGeometry.WALL_NAMES}),
        source=np.array([1.5, 1.5, 1.6]),
        receiver=np.array([5.0, 3.0, 1.5]),
        fs=FS,
        air_absorption=False,
    )


def test_without_a_reference_nothing_changes():
    spec = shoebox()
    default = synthesize(spec, max_order=2, duration_s=0.5)
    none = synthesize(spec, max_order=2, duration_s=0.5, late_envelope=None)
    np.testing.assert_array_equal(default.rir, none.rir)


def test_a_reference_shapes_only_the_late_field():
    spec = shoebox()
    plain = synthesize(spec, max_order=2, duration_s=0.5)
    reference = plain.early[:, 0] + 0.5 * plain.late[:, 0]
    shaped = synthesize(spec, max_order=2, duration_s=0.5, late_envelope=reference)
    np.testing.assert_array_equal(shaped.early, plain.early)
    np.testing.assert_allclose(shaped.rir, shaped.early + shaped.late)
    assert np.sum(shaped.late ** 2) < np.sum(plain.late ** 2)
