"""One-third-octave simulation grid, band averaging and octave folding for the virtual lab."""

import numpy as np

from compas_acoustics.defaults import Defaults

THIRD_OCTAVE_CENTRES = (100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000)
ISO_BAND_RANGE = (100, 5000)
EXTENDED_BAND_RANGE = (100, 8000)

_EXACT_CENTRES = 1000.0 * 10.0 ** (np.arange(-10, 10) / 10.0)


def _band_index(nominal):
    """Index of the band whose exact centre is nearest (on a log scale) to a nominal value."""
    return int(np.argmin(np.abs(np.log(_EXACT_CENTRES) - np.log(float(nominal)))))


def third_octave_centres(band_range=ISO_BAND_RANGE):
    """Nominal and exact centres of the one-third-octave bands in an inclusive range.

    The exact centres are ``1000 * 10**(n / 10)`` (ISO 266; ISO 354 §5, ISO 17497-2 §5).

    Parameters
    ----------
    band_range : tuple of float
        Lowest and highest nominal centre, both inclusive; ``lo == hi`` gives a single band.
        A value is mapped to the band with the nearest exact centre. Must lie within 100...8000 Hz.

    Returns
    -------
    tuple of np.ndarray
        ``(nominal, exact)`` for the bands in the range.

    Raises
    ------
    ValueError
        If the range lies outside 100...8000 Hz or is reversed.
    """
    lo, hi = band_range
    if lo < THIRD_OCTAVE_CENTRES[0] * 0.98 or hi > THIRD_OCTAVE_CENTRES[-1] * 1.02:
        raise ValueError("band_range {} must lie within 100...8000 Hz".format(tuple(band_range)))
    i, j = _band_index(lo), _band_index(hi)
    if i > j:
        raise ValueError("band_range {} is reversed; the lower centre must not exceed the upper (100...8000 Hz)".format(tuple(band_range)))
    return np.array(THIRD_OCTAVE_CENTRES[i : j + 1]), _EXACT_CENTRES[i : j + 1].copy()


def simulation_frequencies(band_range=ISO_BAND_RANGE):
    """Three frequencies per band, ``fc * 2**(-1/9)``, ``fc``, ``fc * 2**(1/9)``, ascending.

    ISO 17497-2 §7.4.4 asks for at least three points per band; band values are their mean (Heimes eq. 11).

    Returns
    -------
    np.ndarray
        Shape ``(3 * n_bands,)``.
    """
    _, exact = third_octave_centres(band_range)
    return np.stack([exact * 2 ** (-1 / 9), exact, exact * 2 ** (1 / 9)], axis=-1).ravel()


def band_average(values):
    """Arithmetic mean of each consecutive triple on the last axis (Heimes eq. 11).

    Parameters
    ----------
    values : np.ndarray
        Last axis of length ``3 * n_bands``.

    Returns
    -------
    np.ndarray
        Last axis of length ``n_bands``.
    """
    values = np.asarray(values, dtype=float)
    if values.shape[-1] % 3:
        raise ValueError("the last axis must hold 3 values per band, got {}".format(values.shape[-1]))
    return values.reshape(values.shape[:-1] + (-1, 3)).mean(axis=-1)


def _nearest_octave(f, octaves):
    return int(octaves[int(np.argmin(np.abs(np.log(np.asarray(octaves, dtype=float)) - np.log(float(f)))))])


def fold_to_octaves(values, nominal):
    """Mean of the one-third-octave values inside each octave (ISO 354 §8.3, ISO 11654).

    Each band belongs to the octave whose nominal centre is nearest on a log scale (125 <- 100, 125, 160; ...; 8000 <- 6300, 8000, 10000).

    Parameters
    ----------
    values : np.ndarray
        Last axis: one value per band in ``nominal``.
    nominal : np.ndarray
        Nominal band centres.

    Returns
    -------
    tuple
        ``(octaves, means)``: the octave centres with at least one band and their means on the last axis.
    """
    values = np.asarray(values, dtype=float)
    standard = list(Defaults._DEFAULT_FREQUENCY_BANDS)
    owner = np.array([_nearest_octave(f, standard) for f in nominal])
    octaves = [o for o in standard if np.any(owner == o)]
    means = np.stack([values[..., owner == o].mean(axis=-1) for o in octaves], axis=-1)
    return octaves, means


def fill_octaves(octaves, values):
    """Expand simulated octaves to all material octaves; a missing one takes the nearest simulated octave's value.

    Parameters
    ----------
    octaves : list of int
        Simulated octave centres.
    values : np.ndarray
        Last axis: one value per simulated octave.

    Returns
    -------
    tuple
        ``(filled, pairs)``: values on the 7 octaves of ``Defaults._DEFAULT_FREQUENCY_BANDS`` and the ``(filled, source)`` pairs.
    """
    values = np.asarray(values, dtype=float)
    index = {o: k for k, o in enumerate(octaves)}
    columns, pairs = [], []
    for o in Defaults._DEFAULT_FREQUENCY_BANDS:
        if o in index:
            columns.append(values[..., index[o]])
        else:
            source = _nearest_octave(o, octaves)
            columns.append(values[..., index[source]])
            pairs.append((o, source))
    return np.stack(columns, axis=-1), pairs
