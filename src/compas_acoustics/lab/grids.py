"""Source (incidence) and receiver grids for the virtual scattering lab."""

import math
import warnings

import numpy as np
from scipy.spatial import SphericalVoronoi

_SYMMETRIES = ("none", "rotational")
_MAX_RECEIVERS = 50000


def _preset_rings(name):
    """Return ``(colatitudes, azimuth_step)`` in degrees for a preset name."""
    if name == "survey":
        return [0.0, 30.0, 60.0], 60.0
    if name == "standard":
        return [10.0 * k for k in range(9)], 30.0
    if name == "fine":
        return [1.26 + 2.52 * k for k in range(36)], 15.0
    raise ValueError("unknown source preset {!r}; valid presets are 'survey', 'standard', 'fine'".format(name))


class SourceGrid(object):
    """Incidence directions of the sources, with Paris random-incidence weights.

    Parameters
    ----------
    colatitude : array-like, shape (n,)
        Colatitude of each source in degrees (0 = straight above the sample).
    azimuth : array-like, shape (n,)
        Azimuth of each source in degrees.

    Notes
    -----
    Presets: ``survey`` is ISO 17497-2 Table 1 (colatitude 0, 30 and 60 degrees, azimuth step 60 degrees, 13 sources),
    ``standard`` uses colatitudes 0..80 degrees in 10 degree steps with a 30 degree azimuth step (97 sources), and ``fine``
    uses Heimes and Vorlaender's colatitudes 1.26..88.74 degrees in 2.52 degree steps with a 15 degree azimuth step (864 sources).
    The Paris weights (Kuttruff, *Room Acoustics*) of a ring are ``(sin^2(theta_hi) - sin^2(theta_lo)) / n_azimuth``, with ring
    bounds at the midpoints between consecutive colatitudes, 0 degrees below the first ring and 90 degrees above the last.
    """

    def __init__(self, colatitude, azimuth):
        self.colatitude = np.asarray(colatitude, dtype=float).ravel()
        self.azimuth = np.asarray(azimuth, dtype=float).ravel()
        if self.colatitude.shape != self.azimuth.shape:
            raise ValueError("colatitude and azimuth must have the same length")

    @classmethod
    def preset(cls, name, symmetry="none"):
        """Build one of the named source grids.

        Parameters
        ----------
        name : {"survey", "standard", "fine"}
        symmetry : {"none", "rotational"}
            ``"rotational"`` keeps azimuth 0 only.
        """
        colats, step = _preset_rings(name)
        if symmetry not in _SYMMETRIES:
            raise ValueError("unknown symmetry {!r}; valid symmetries are 'none', 'rotational'".format(symmetry))
        azimuths = np.array([0.0]) if symmetry == "rotational" else np.arange(0.0, 360.0, step)
        colat, azim = [], []
        for c in colats:
            for a in [0.0] if c == 0.0 else azimuths:
                colat.append(c)
                azim.append(a)
        return cls(colat, azim)

    @property
    def n(self):
        return int(self.colatitude.size)

    @property
    def directions(self):
        """Unit vectors from the sample towards the sources, shape (n, 3)."""
        t = np.radians(self.colatitude)
        p = np.radians(self.azimuth)
        return np.column_stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])

    @property
    def paris_weights(self):
        """Random-incidence weights (Paris), summing to 1, shape (n,)."""
        rings = np.unique(self.colatitude)
        mids = 0.5 * (rings[:-1] + rings[1:])
        lo = np.concatenate([[0.0], mids])
        hi = np.concatenate([mids, [90.0]])
        ring_weight = np.sin(np.radians(hi)) ** 2 - np.sin(np.radians(lo)) ** 2
        w = np.zeros(self.n)
        for r, rw in zip(rings, ring_weight):
            mask = self.colatitude == r
            w[mask] = rw / mask.sum()
        return w


class ReceiverGrid(object):
    """Receiver points on a hemisphere, with the solid angle each represents.

    Parameters
    ----------
    directions : array-like, shape (n, 3)
        Unit vectors from the sample centre to the receivers.
    weights : array-like, shape (n,)
        Solid angle (Voronoi cell area on the unit sphere) of each receiver; they sum to 2 pi on a hemisphere.
    radius : float
        Distance of the receivers from the origin.
    """

    def __init__(self, directions, weights, radius):
        self.directions = np.asarray(directions, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.radius = float(radius)

    @property
    def points(self):
        return self.directions * self.radius

    @classmethod
    def hemisphere(cls, spacing_deg, radius):
        """Receivers on the upper hemisphere, nominally ``spacing_deg`` apart (ISO 17497-2 section 6.2.2).

        A Fibonacci (golden-angle) spiral of ``ceil(2 pi / spacing^2)`` points; the weights are the areas of the spherical
        Voronoi cells of the points together with their mirror images in z = 0, which keeps the tiling symmetric.
        """
        delta = math.radians(spacing_deg)
        n = int(math.ceil(2.0 * math.pi / delta**2))
        if n > _MAX_RECEIVERS:
            warnings.warn("receiver grid has {} points (spacing {} deg); this is slow".format(n, spacing_deg))
        i = np.arange(n)
        z = (i + 0.5) / n
        phi = i * math.pi * (3.0 - math.sqrt(5.0))
        rho = np.sqrt(1.0 - z**2)
        up = np.column_stack([rho * np.cos(phi), rho * np.sin(phi), z])
        mirrored = up * np.array([1.0, 1.0, -1.0])
        sv = SphericalVoronoi(np.vstack([up, mirrored]), radius=1.0, center=np.zeros(3))
        sv.sort_vertices_of_regions()
        areas = sv.calculate_areas()[:n]
        return cls(up, areas, radius)


def receiver_spacing(diameter, f_max, c):
    """Receiver spacing in degrees: ``min(5, 50 * lambda_min / d)`` (ISO 17497-2 section 6.2.2, Embrechts eq. 6, Heimes eq. 6).

    Parameters
    ----------
    diameter : float
        Footprint diameter d of the sample (m).
    f_max : float
        Highest frequency (Hz).
    c : float
        Speed of sound (m/s).
    """
    return min(5.0, 50.0 * (c / f_max) / diameter)
