"""The solver interface of the scattering lab and the free-field pressures it returns (Heimes & Vorländer §3)."""

from dataclasses import dataclass
from dataclasses import field
from typing import List

import numpy as np

__all__ = ["Pressures", "ScatteringSolver"]


@dataclass
class Pressures:
    """Free-field pressures of a sample and its flat reference, the input of the correlation method (Mommertz 2000; Heimes & Vorländer eq. 3).

    Parameters
    ----------
    sample, reference : numpy.ndarray
        Complex pressures, shape (n_sources, n_receivers, n_frequencies).
    frequencies : numpy.ndarray
        The simulated frequencies in Hz, shape (n_frequencies,).
    generation : dict
        How they were computed: the method and its settings.
    """

    sample: np.ndarray
    reference: np.ndarray
    frequencies: np.ndarray
    generation: dict = field(default_factory=dict)


class ScatteringSolver(object):
    """A free-field solver: plane waves from each source direction onto the sample and its reference, pressures at the receivers.

    Directions follow the lab's convention: a source direction û points from the sample towards the source, so its plane wave travels along −û.
    """

    name = None

    def solve(self, sample, sources, receivers, frequencies, c):
        """The free-field pressures of ``sample`` and its reference.

        Parameters
        ----------
        sample : :class:`compas_acoustics.lab.sample.ScatteringSample`
        sources : :class:`compas_acoustics.lab.grids.SourceGrid`
        receivers : :class:`compas_acoustics.lab.grids.ReceiverGrid`
        frequencies : numpy.ndarray
            Frequencies in Hz.
        c : float
            The speed of sound in m/s.

        Returns
        -------
        :class:`Pressures`
        """
        raise NotImplementedError

    def warnings(self, sample, nominal_bands, exact_bands, c) -> List[str]:
        """Where this solver's results are unreliable for ``sample`` over the bands (nominal and exact centres in Hz)."""
        return []
