"""Scattering coefficients from free-field pressures, on plain numpy arrays."""

import numpy as np

__all__ = ["random_incidence", "scattering_freefield"]


def scattering_freefield(p_sample, p_reference, weights, axis=1):
    r"""Correlation scattering coefficient of a sample against its flat reference.

    Implements the free-field correlation method of Mommertz (2000), written as
    eq. 3 in Heimes & Vorländer (2025):

    .. math::

        s = 1 - \frac{\left|\sum p_1 p_0^* w\right|^2}
                     {\sum |p_1|^2 w \cdot \sum |p_0|^2 w}

    where :math:`p_1` is the pressure with the sample, :math:`p_0` the pressure
    with the reference, and the sums run over the receivers.

    Parameters
    ----------
    p_sample : numpy.ndarray
        Complex pressure of the sample, receivers along `axis`.
    p_reference : numpy.ndarray
        Complex pressure of the reference, same shape as `p_sample`.
    weights : numpy.ndarray
        Receiver weights, one per receiver; broadcast along `axis`.
    axis : int, optional
        Receiver axis. Default is 1.

    Returns
    -------
    numpy.ndarray
        Scattering coefficient with `axis` removed. Where both patterns are
        identically zero the coefficient is 0 rather than NaN.
    """
    p1 = np.asarray(p_sample)
    p0 = np.asarray(p_reference)
    w = np.asarray(weights, dtype=float)
    shape = [1] * p1.ndim
    shape[axis] = -1
    w = w.reshape(shape)

    cross = np.sum(p1 * np.conj(p0) * w, axis=axis)
    e1 = np.sum(np.abs(p1) ** 2 * w, axis=axis)
    e0 = np.sum(np.abs(p0) ** 2 * w, axis=axis)
    denom = e1 * e0
    safe = np.where(denom > 0.0, denom, 1.0)
    ratio = np.where(denom > 0.0, np.abs(cross) ** 2 / safe, 1.0)
    return 1.0 - ratio


def random_incidence(s_angle, source_grid):
    """Random-incidence scattering coefficient from per-direction values.

    Weights the coefficient of each source direction with the Paris weights
    (Kuttruff), :math:`s_{rand} = \\sum_i w_i s_i`, which approximate the
    :math:`\\cos\\theta\\,\\sin\\theta` diffuse-field weighting and sum to 1.

    Parameters
    ----------
    s_angle : numpy.ndarray
        Coefficients with the sources along axis 0.
    source_grid : compas_acoustics.lab.SourceGrid
        Source grid providing ``paris_weights``.

    Returns
    -------
    numpy.ndarray
        Weighted average over axis 0.
    """
    s = np.asarray(s_angle)
    return np.tensordot(np.asarray(source_grid.paris_weights), s, axes=(0, 0))
