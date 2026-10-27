"""The early field is causal: nothing is heard before it arrives.

The radiation taper (a fourth-order 35 Hz high-pass that keeps DC out, see
``early._radiation_taper``) multiplies every path's spectrum. Applied as a real,
zero-phase magnitude it rang symmetrically about each arrival, and its
low-frequency half before the arrival reached back to the start of the response:
around the corner of an L-shaped room the diffracted part held 18 dB below its
energy before the direct sound could arrive. It is now the causal Butterworth
high-pass of the same magnitude.

Not tested here: the sinc tail of a fractional delay (about -27 dB beyond 1 ms,
the cost of an exact band-limited delay), and the band magnitude, which is still
applied zero-phase (a reflection losing 20 dB over the eight bands rings about
25 dB down a millisecond before it arrives).

Ported from hybrid-rir (gpapadimitri/hybrid-rir @ a272da7, tests/test_early_causality.py).
"""

import numpy as np
import pyfar as pf
import pytest

from _compat import RoomSpec, image_sources
from compas_acoustics.hybrid.synthesis import prepare
from compas_acoustics.hybrid.ambisonics import render_ambisonics
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.hybrid.early import (
    ImageSourceSet,
    _radiation_taper,
    render_images,
)
from compas_acoustics.hybrid.bands import BANDS
from _compat import Material, MeshGeometry

FS = 48_000.0
C = 343.0
SOURCE, RECEIVER = (7.0, 1.0, 1.5), (4.0, 5.5, 1.5)


def before_arrival_db(signal, arrival, margin=48):
    """Energy more than ``margin`` samples (1 ms) before ``arrival``, relative to the whole signal."""
    return 10 * np.log10(np.sum(signal[: arrival - margin] ** 2) / np.sum(signal**2))


def test_the_taper_keeps_the_butterworth_magnitude():
    freq = np.fft.rfftfreq(1 << 16, 1 / FS)
    ratio = (freq / 35.0) ** 4
    np.testing.assert_allclose(np.abs(_radiation_taper(freq)), ratio / np.sqrt(1 + ratio**2), rtol=1e-12, atol=1e-15)
    assert _radiation_taper(np.array([0.0]))[0] == 0.0


def test_the_taper_is_causal():
    nfft = 1 << 16
    h = np.fft.irfft(_radiation_taper(np.fft.rfftfreq(nfft, 1 / FS)), nfft)
    # The second half of the circular response is negative time. Its last millisecond holds the Gibbs ripple of
    # band-limiting the analog response at Nyquist (2e-6 of the energy, in the samples next to t = 0); before
    # that the zero-phase taper left 4.6e-4, the causal one 3e-8.
    assert np.sum(h[nfft // 2: nfft - 48] ** 2) / np.sum(h**2) < 1e-6


def test_an_image_on_a_sample_is_silent_before_it_arrives():
    # flat band gains and an integer delay: only the taper can ring early
    delay = 0.02
    image = ImageSourceSet(delay=np.array([delay]), gain=np.ones((1, BANDS.size)),
                           doa=pf.Coordinates(np.array([1.0]), np.array([0.0]), np.array([0.0])),
                           order=np.array([0]), bands=BANDS.copy(), fs=FS)
    h = render_images(image, np.ones((1, 1)), 9600)[:, 0]
    assert before_arrival_db(h, int(delay * FS)) < -60.0


def l_room():
    outline = [(0, 0), (8, 0), (8, 3), (5, 3), (5, 6), (0, 6)]
    n = len(outline)
    v = np.array([[x, y, 0.0] for x, y in outline] + [[x, y, 3.0] for x, y in outline], float)
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))] + [(k, (k + 1) % n, (k + 1) % n + n, k + n) for k in range(n)]
    # frequency-dependent: with every material flat, pyroomacoustics returns a single band
    materials = {name: Material(0.1 * np.linspace(0.8, 1.2, 8), BANDS.copy()) for name in ("floor", "ceiling", "wall")}
    return MeshGeometry(v, tuple(faces), tuple(["floor", "ceiling"] + ["wall"] * n), materials)


@pytest.mark.parametrize("model", ["udfa", "btms"])
def test_diffraction_adds_nothing_before_the_direct_sound_could_arrive(model):
    # around the corner the wall hides the direct sound, and no diffracted path is shorter than it
    spec = RoomSpec(geometry=l_room(), source=np.array(SOURCE), receiver=np.array(RECEIVER),
                    air_absorption=False, speed_of_sound=C)
    off = render_ambisonics(prepare(spec, image_sources(spec, 2), RoomTarget.from_room(spec), max_order=1, duration_s=0.2)).early[:, 0]
    on = render_ambisonics(prepare(spec, image_sources(spec, 2), RoomTarget.from_room(spec), max_order=1, duration_s=0.2, diffraction=model)).early[:, 0]
    hidden = int(np.linalg.norm(np.subtract(SOURCE, RECEIVER)) / C * FS)
    assert before_arrival_db(on - off, hidden) < -50.0
