"""Edge diffraction by the universal diffraction filter approximation (UDFA).

Single wedges: Kirsch and Ewert (2023), "A universal filter approximation of
edge diffraction for geometrical acoustics", IEEE/ACM Trans. Audio Speech
Lang. Process. 31, 1636-1651. Three-sided barriers (:class:`DoubleEdgePath`):
Kirsch and Ewert (2024), "Filter-based first- and higher-order diffraction
modeling for geometrical acoustics", Acta Acustica 8, 73, Sec. 3.3. Equation
numbers are the respective paper's.

For the barrier, two simplifications the 2024 paper allows (end of Sec.
3.3.1) are taken: beta = 1 at all angles, and no low-frequency gain
compensation (its Appendix). Where the paper is silent the choice is marked
"reading:". The barrier is defined for an infinite double edge and is not
applied to room geometry (:mod:`compas_acoustics.hybrid.diffraction`).

A wedge path from a point source over one rigid wedge carries
``exp(-j k d) / d * udfa_transfer(path, f)`` relative to a unit free field
``exp(-j k r) / r``, ``d`` being :attr:`WedgePath.d_constrained`. The
transfer function holds the factor 1/2 and the sign of both terms of
eq. (2): on the shadow side of the incident shadow boundary its incident term
tends to +1/2, on the lit side to -1/2, and the direct sound's unit step is
made up exactly. Diffraction is therefore rendered on both sides of every
boundary, and the geometric sound must be switched by the same geometry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class WedgePath:
    """Source, receiver and one rigid wedge in the wedge's cylindrical frame.

    The edge runs along z from ``-length / 2`` to ``+length / 2``
    (``length = inf`` for an infinite wedge), Fig. 1. Azimuths are measured
    through the open region from the face nearer the source, so
    ``0 <= theta_s <= theta_r <= theta_w``; :func:`compas_acoustics.hybrid.diffraction.wedge_path`
    builds this from Cartesian points.
    """

    theta_w: float
    theta_s: float
    theta_r: float
    r_s: float
    r_r: float
    z_s: float
    z_r: float
    length: float = np.inf

    def __post_init__(self) -> None:
        if not np.pi < self.theta_w <= 2.0 * np.pi:
            raise ValueError("theta_w must lie in (pi, 2 pi]")
        if not 0.0 <= self.theta_s <= self.theta_r <= self.theta_w:
            raise ValueError("need 0 <= theta_s <= theta_r <= theta_w")
        if self.r_s <= 0.0 or self.r_r <= 0.0:
            raise ValueError("source and receiver must lie off the edge")
        if not self.length > 0.0:
            raise ValueError("length must be positive")

    @classmethod
    def infinite(
        cls, theta_w: float, theta_s: float, theta_r: float,
        d_s: float, d_r: float, phi: float,
    ) -> WedgePath:
        """An infinite wedge from distances to the apex and the incidence angle.

        Angles may come in either order; measuring from the other face swaps
        them, which leaves the diffracted field unchanged.
        """
        if theta_s > theta_r:
            theta_s, theta_r = theta_w - theta_s, theta_w - theta_r
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        return cls(theta_w, theta_s, theta_r, d_s * sin_phi, d_r * sin_phi,
                   -d_s * cos_phi, d_r * cos_phi)

    def length_via(self, z: float) -> float:
        """Path length from source to receiver through the edge point at z."""
        return float(np.hypot(self.r_s, z - self.z_s) + np.hypot(self.r_r, self.z_r - z))

    @property
    def z_apex(self) -> float:
        return self.z_s + (self.z_r - self.z_s) * self.r_s / (self.r_s + self.r_r)

    @property
    def d(self) -> float:
        return self.length_via(self.z_apex)

    @property
    def d_s(self) -> float:
        return float(np.hypot(self.r_s, self.z_apex - self.z_s))

    @property
    def d_r(self) -> float:
        return self.d - self.d_s

    @property
    def sin2_phi(self) -> float:
        return (self.r_s / self.d_s) ** 2

    @property
    def apex_on_edge(self) -> bool:
        return bool(np.isinf(self.length) or abs(self.z_apex) <= self.length / 2.0)

    @property
    def d_constrained(self) -> float:
        """Path length over the apex clamped onto the physical edge (Sec. VI)."""
        if self.apex_on_edge:
            return self.d
        half = self.length / 2.0
        return self.length_via(float(np.clip(self.z_apex, -half, half)))

    @property
    def zone(self) -> str:
        if self.theta_r - self.theta_s > np.pi:
            return "shadow"
        plus = self.theta_r + self.theta_s
        if plus < np.pi or plus > 2.0 * self.theta_w - np.pi:
            return "reflection"
        return "view"


def fractional_lowpass(
    freq: np.ndarray, fc: float, blend: float = 1.44, q: float = 0.2
) -> np.ndarray:
    """Eq. (6), of fractional order 1/2 and roll-off 1.6. ``fc = inf`` gives a flat response."""
    jx = 1j * np.asarray(freq, dtype=float) / fc
    return (jx ** (2.0 / blend) + (jx / q) ** (1.0 / blend**1.6) + 1.0) ** (
        -0.5 * blend / 2.0
    )


def first_order_lowpass(freq: np.ndarray, fc: float) -> np.ndarray:
    return 1.0 / (1.0 + 1j * np.asarray(freq, dtype=float) / fc)


def term_parameters(path: WedgePath, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cutoff (eq. 4), gain (eq. 3) and polarity of the (+, -) terms of eq. (2).

    The polarity is the sign of N_nu (eq. 5), i.e. of cos(nu pi) - cos(nu theta).
    On [0, 2 theta_w - pi] that is eq. (2)'s sign(theta - pi); beyond, in the
    reflection zone of the second face (Sec. II), it flips again, as the
    reflected sound from that face appears there.
    """
    nu = np.pi / path.theta_w
    theta = np.array([path.theta_r + path.theta_s, path.theta_r - path.theta_s])
    cos_nupi = np.cos(nu * np.pi)
    root = np.sqrt(1.0 - cos_nupi * np.cos(nu * theta))
    denom = cos_nupi - np.cos(nu * theta)
    d_star = 2.0 * path.d_s * path.d_r / path.d
    with np.errstate(divide="ignore"):
        fc = 2.0 * c * (nu * root / denom) ** 2 / (np.pi**2 * d_star * path.sin2_phi)
    gain = np.sin(nu * np.pi) / root
    polarity = np.where(denom > 0.0, 1.0, -1.0)
    return fc, gain, polarity


def single_term_cutoff(path: WedgePath, c: float, *, flat_outside_shadow: bool = True) -> float:
    """Eq. (8): one low-pass with unit DC gain, matching eq. (2) asymptotically.

    Eq. (7) taken with each term's polarity, which is what eq. (8) writes out.
    Outside the shadow zone the single filter is flat (Sec. II-B and Kirsch
    and Ewert 2024 Sec. 3.1) unless ``flat_outside_shadow`` is False, which
    the double-edge weighting (2024 eq. 12) needs.
    """
    if flat_outside_shadow and path.zone != "shadow":
        return float("inf")
    fc, gain, polarity = term_parameters(path, c)
    with np.errstate(invalid="ignore"):
        s = float(np.sum(polarity * gain * np.sqrt(fc)))
    return (s / 2.0) ** 2 if np.isfinite(s) else float("inf")


def udfa_single_transfer(path: WedgePath, freq: np.ndarray, *, c: float) -> np.ndarray:
    """Eq. (8) as one filter with unit DC gain; on a finite wedge with the
    half wedges of Sec. III, as Sec. VII proposes ("including a DC gain").
    Flat outside the shadow zone."""
    freq = np.asarray(freq, dtype=float)
    fc = single_term_cutoff(path, c)
    if np.isinf(fc):
        return np.ones(freq.shape, dtype=complex)
    # _term_transfer carries eq. (2)'s 1/2; the single term has unit gain
    return 2.0 * _term_transfer(freq, fc, fc, path, c)


def udfa_transfer(path: WedgePath, freq: np.ndarray, *, c: float) -> np.ndarray:
    """Two-term UDFA, eq. (2), with the finite-wedge forms of Sec. III."""
    freq = np.asarray(freq, dtype=float)
    fc, gain, polarity = term_parameters(path, c)
    fc_max = float(np.max(fc))
    out = np.zeros(freq.shape, dtype=complex)
    for k in range(2):
        out += polarity[k] * gain[k] * _term_transfer(freq, float(fc[k]), fc_max, path, c)
    return out


def _half_wedge_gain(fc: float, t: float) -> float:
    """Eq. (10): DC gain of a half wedge truncated t seconds after the apex."""
    if t <= 0.0:
        return 0.0
    if np.isinf(fc):
        return 1.0
    return 2.0 / np.pi * float(np.arctan(np.pi * np.sqrt(2.0 * fc * t)))


def _finite_half_wedge(freq: np.ndarray, fc: float, g: float) -> np.ndarray:
    """Eqs. (11), (12): the half-order low-pass of a half wedge with DC gain g."""
    if g == 0.0:
        return np.zeros(freq.shape, dtype=complex)
    return g * fractional_lowpass(
        freq, fc / g**2, 1.0 + (1.44 - 1.0) * g**2, 0.5 + (0.2 - 0.5) * g**2
    )


def _term_transfer(freq, fc, fc_max, path, c):
    """One term of eq. (2) with its 1/2: eq. (13) on the edge, (17) off it."""
    if np.isinf(path.length):
        return fractional_lowpass(freq, fc) / 2.0
    half = path.length / 2.0
    t_near, t_far = sorted(
        ((path.length_via(-half) - path.d) / c, (path.length_via(half) - path.d) / c)
    )
    if path.apex_on_edge:
        return (
            _finite_half_wedge(freq, fc, _half_wedge_gain(fc, t_near))
            + _finite_half_wedge(freq, fc, _half_wedge_gain(fc, t_far))
        ) / 4.0
    g_far = _half_wedge_gain(fc, t_far)
    g_off = g_far - _half_wedge_gain(fc, t_near)
    # Sec. III-D: the blend reads the initial decay with the higher cutoff
    # of the two terms
    p = (1.0 - _half_wedge_gain(fc_max, t_near) / _half_wedge_gain(fc_max, t_far)) ** 4
    off = g_off * first_order_lowpass(freq, 1.0 / (np.pi * (t_far - t_near)))
    return (p * _finite_half_wedge(freq, fc, g_far) + (1.0 - p) * off) / 4.0


# Three-sided barrier, Kirsch and Ewert (2024), Sec. 3.3


def _normal_1(theta_w1):
    a = np.pi - theta_w1
    return np.array([-np.sin(a), np.cos(a)])


def _normal_2(theta_w2):
    return np.array([np.sin(theta_w2), -np.cos(theta_w2)])


def _angle(v) -> float:
    return float(np.arctan2(v[1], v[0]))


@dataclass(frozen=True)
class DoubleEdgePath:
    """Source and receiver around a three-sided barrier, in its cross-section.

    Edge 1 at (0, 0), edge 2 at (-width, 0), the face between them along y = 0
    with the barrier below (Fig. 7). Side face 1 leaves edge 1 at angle
    pi - theta_w1, side face 2 leaves edge 2 at angle theta_w2. ``phi`` is the
    incidence angle on both edges (Fig. 1); cross-section distances divided
    by sin(phi) are path lengths.
    """

    theta_w1: float
    theta_w2: float
    width: float
    source: np.ndarray
    receiver: np.ndarray
    phi: float = np.pi / 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", np.asarray(self.source, dtype=float))
        object.__setattr__(self, "receiver", np.asarray(self.receiver, dtype=float))
        for w in (self.theta_w1, self.theta_w2):
            if not np.pi < w <= 1.5 * np.pi + 1e-12:
                raise ValueError("exterior angles must lie in (pi, 3 pi / 2]")
        if self.width <= 0.0:
            raise ValueError("width must be positive")
        if not 0.0 < self.phi <= np.pi / 2 + 1e-12:
            raise ValueError("phi must lie in (0, pi / 2]")
        if self.source @ _normal_1(self.theta_w1) <= 0.0:
            raise ValueError("the source must lie beyond the plane of side face 1")
        if (self.receiver - self.edge2) @ _normal_2(self.theta_w2) <= 0.0:
            raise ValueError("the receiver must lie beyond the plane of side face 2")

    @property
    def edge2(self) -> np.ndarray:
        return np.array([-self.width, 0.0])


@dataclass(frozen=True)
class RouteTerm:
    """One rendered contribution: pressure exp(-jkL)/L * transfer(f)."""

    length: float
    last_edge: int
    zone: str
    transfer: Callable[[np.ndarray], np.ndarray]


def _canonical(path: DoubleEdgePath) -> tuple[DoubleEdgePath, bool]:
    """Source at or below the top plane whenever the receiver is below it."""
    if path.source[1] > 0.0 and path.receiver[1] < 0.0:
        def mirror(p):
            return np.array([-path.width - p[0], p[1]])
        return DoubleEdgePath(path.theta_w2, path.theta_w1, path.width,
                              mirror(path.receiver), mirror(path.source), path.phi), True
    return path, False


def _angles(p: DoubleEdgePath):
    e1 = np.pi - p.theta_w1
    th_s1 = (_angle(p.source) - e1) % TWO_PI
    th_r1 = (_angle(p.receiver) - e1) % TWO_PI
    th_r2 = (p.theta_w2 - _angle(p.receiver - p.edge2)) % TWO_PI
    th_s2 = (p.theta_w2 - _angle(p.source - p.edge2)) % TWO_PI
    return th_s1, th_r1, th_s2, th_r2


def _zone(p: DoubleEdgePath, th_s1: float, th_r1: float) -> str:
    if p.source[1] >= 0.0 and p.receiver[1] >= 0.0:
        return "double_view"
    if p.receiver[1] < 0.0:
        return "double_shadow"
    return "single_shadow" if th_r1 > th_s1 + np.pi else "single_view"


def double_edge_zone(path: DoubleEdgePath) -> str:
    p, _ = _canonical(path)
    th_s1, th_r1, _, _ = _angles(p)
    return _zone(p, th_s1, th_r1)


def _alpha12(fc12: float, fc21: float) -> float:
    """Eq. (12), with the limits of either cutoff at infinity."""
    if np.isinf(fc12) and np.isinf(fc21):
        return 0.5
    if np.isinf(fc12):
        return 0.0
    if np.isinf(fc21):
        return 1.0
    return 1.0 / (fc12 / fc21 + 1.0)


def double_edge_terms(path: DoubleEdgePath, *, c: float) -> list[RouteTerm]:
    """Eq. (11) as rendered terms, each with its own path length and last edge."""
    p, swapped = _canonical(path)
    th_s1, th_r1, th_s2, th_r2 = _angles(p)
    zone = _zone(p, th_s1, th_r1)
    sin_phi = np.sin(p.phi)
    d_s1 = float(np.linalg.norm(p.source)) / sin_phi
    d_r1 = float(np.linalg.norm(p.receiver)) / sin_phi
    d_s2 = float(np.linalg.norm(p.source - p.edge2)) / sin_phi
    d_r2 = float(np.linalg.norm(p.receiver - p.edge2)) / sin_phi
    d_w = p.width / sin_phi

    # eq. (8), beta = 1; d_{1,2} = d_s + d_r across either edge
    d1, d2 = d_s1 + d_r1, d_s2 + d_r2
    hat1 = p.theta_w1 + d1**2 / (p.width + d1**2) * (p.theta_w2 - np.pi)
    hat2 = p.theta_w2 + d2**2 / (p.width + d2**2) * (p.theta_w1 - np.pi)

    shadow = zone == "double_shadow"
    # eq. (9): detour distance in the double shadow zone.
    # reading: the angle beyond the modified face is clamped onto it.
    wp1 = WedgePath.infinite(hat1, th_s1, min(th_r1, hat1), d_s1,
                             d_w + d_r2 if shadow else d_r1, p.phi)
    wp2 = WedgePath.infinite(hat2, th_r2, min(th_s2, hat2), d_r2,
                             d_w + d_s1 if shadow else d_s2, p.phi)
    # eq. (10): the other edge as a source or receiver on the shared face,
    # unmodified wedge angle, single-term filter
    wp12 = WedgePath.infinite(p.theta_w2, p.theta_w2, th_r2, d_w, d_r2, p.phi)
    wp21 = WedgePath.infinite(p.theta_w1, th_s1, p.theta_w1, d_s1, d_w, p.phi)
    fc12, fc21 = single_term_cutoff(wp12, c), single_term_cutoff(wp21, c)
    # reading: the weight uses the asymptotic cutoffs, which stay finite in
    # the view zone where the filters themselves are flat
    alpha12 = _alpha12(single_term_cutoff(wp12, c, flat_outside_shadow=False),
                       single_term_cutoff(wp21, c, flat_outside_shadow=False))

    def d1_tf(f):
        return udfa_transfer(wp1, f, c=c)

    def d2_tf(f):
        return udfa_transfer(wp2, f, c=c)

    last = (lambda e: 3 - e) if swapped else (lambda e: e)
    if zone == "double_shadow":
        alpha = alpha12

        def tf(f):
            return ((1.0 - alpha) * d1_tf(f) * fractional_lowpass(f, fc12)
                    + alpha * d2_tf(f) * fractional_lowpass(f, fc21))

        return [RouteTerm(d_s1 + d_w + d_r2, last(2), zone, tf)]
    if zone == "double_view":
        alpha = 1.0 - alpha12
        return [
            RouteTerm(d_s1 + d_r1, last(1), zone, lambda f: (1.0 - alpha) * d1_tf(f)),
            RouteTerm(d_s2 + d_r2, last(2), zone, lambda f: alpha * d2_tf(f)),
        ]
    return [RouteTerm(d_s1 + d_r1, last(1), zone, d1_tf)]
