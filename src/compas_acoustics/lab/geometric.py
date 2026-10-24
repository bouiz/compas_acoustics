"""The geometric scattering engine: specular rays off the rigid sample and reference, binned on the receiver hemisphere.

Per source direction, parallel rays aimed at uniformly random points of the reference's footprint are reflected specularly off the rigid
sample and, with the same rays, off the rigid reference (the flat footprint or a proxy), and leave upward into one cell of a receiver
hemisphere grid. Every ray is either binned or counted as lost, and the two are counted separately. There is no phase and no
diffraction: the rim and any wave effect are ignored, and the frequency dependence is not computed but imposed afterwards by
:func:`heuristic_s`, in the form of ``AcousticMaterial.scattering_coefficients_from_depth``, from the depth of the relief: ``h = max(2 σ, |mean|)``
of the height difference between the sample and the reference over the footprint, so a relief gets twice its standard deviation and an object of
uniform height over its footprint (a box, a cylinder, a flat-topped panel) its height.

The object is treated as mounted in a rigid infinite baffle at z = 0 (the objects of the lab are mounted on a wall, floor or ceiling), outside
its footprint: a ray travelling downward that reaches z = 0 before any face, at a point outside the footprint (the xy projection of the reference's
flat bottom face), is reflected specularly on the plane, which counts as one bounce. Inside the footprint there is no baffle: a ray reflects off
whatever face it meets, below z = 0 included (the bottom of a closed mesh lies below z = 0). A ray that leaves upward is binned, and so is one that
leaves exactly along the horizon (``d_z == 0``: it grazes the baffle plane, it is not trapped), into the cell nearest its direction. A ray still
bouncing after ``max_bounces`` is lost; so would be one leaving downward without meeting a face or the baffle, which a closed body does not let happen.

Accuracy record
---------------
Against the Kirchhoff method, 2026-10-24 (``tests/lab/test_geometric_validation.py``, which prints this table). The sample is Heimes &
Vorländer's sine relief (period 0.08 m, depth 0.051/2.5 m = 20.4 mm, disc of 0.8 m, c = 346.18 m/s), both methods with ``preset="survey"``
and the ISO band range, the geometric one at its defaults (``n_rays=50000``, ``bin_deg=10``). The last two columns are the geometric s at
5000 and 50000 rays per source, the Monte-Carlo spread (the seed is fixed, so this is the spread over the footprint points)::

    octave  s_geometric  s_kirchhoff  |delta|   g(5000)  g(50000)
       125        0.001        0.002    0.001     0.001     0.001
       250        0.004        0.001    0.003     0.004     0.004
       500        0.017        0.002    0.016     0.017     0.017
      1000        0.066        0.003    0.063     0.067     0.066
      2000        0.234        0.043    0.191     0.238     0.234
      4000        0.625        0.440    0.186     0.632     0.625
    max |delta| = 0.191, mean |delta| = 0.077

Run time on the development machine (macOS, 8 or more cores): geometric 1.6 s, Kirchhoff 45.4 s (about 28 times as long). The ray count is not what limits the
accuracy: 5000 rays per source change s by at most 0.007.

What the numbers show: the heuristic follows the rise of s with frequency but sits above Kirchhoff from 500 Hz up, by 0.06 at 1 kHz and
by about 0.19 at 2 and 4 kHz, the octaves where s is large. Below 500 Hz both are near 0. Kirchhoff itself warns that its s is unreliable
below 4327 Hz for this sample (features of 0.08 m smaller than the wavelength), so the lower octaves are not a clean reference; at 4 kHz,
where the features are about as large as the wavelength, the gap is the least ambiguous one, and it is an over-estimate. Open question for the owner: the
depth of the curve is ``h = 2 σ`` of the height difference for a relief like this one (the depth is ``max(2 σ, |mean|)``; here 2 σ = 0.0142 m and
|mean| = 0.0102 m, so the table holds for the later definition too); whether that factor, or the curve's form, should be changed is not settled by
this one sample against one approximate method (the BEM, or Heimes's s, would be the better reference), and nothing was changed here.

Validity limits. The method has no phase: grating lobes, notches and phase-grating diffusers (periodic relief whose scattering is an interference effect)
are invisible to it. It has no diffraction (the rim, edges, the
shadow zone) and no cavity resonance, and a relief much smaller than the wavelength still gets the geometric s_max, reduced only by
the depth curve. The frequency dependence is that curve, a heuristic, not computed. The object is taken as mounted in a rigid baffle at
z = 0: a free-standing object is not represented. The rays are aimed only at the footprint, so for a tall object at oblique incidence the rays
that would strike its sides from outside the footprint are not traced. Directions are binned in cells of ``bin_deg``: a facet tilted by less than about half a
bin from the reference's reflection counts as specular, so s_max is blind to slopes below that angle. A warning is raised only
when the lost fraction exceeds 1 % for some source (of the sample or the reference); up to 1 % goes unreported and enters s_max as one extra bin.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .grids import ReceiverGrid
from .raycast import RayCaster

__all__ = ["GeometricResponse", "geometric_response", "heuristic_s", "scattering_share"]

_BOTTOM = 0.999  # a triangle with a normal z below minus this is part of a flat bottom face
_BELOW = -1e-9  # a hit lower than this (m) is below the baffle; the double-precision hit points on z = 0 are within 1e-15 of it
_EDGE = 1e-12  # relative to the footprint's size squared: the tolerance of the point-in-footprint test, which includes the edge
_CHUNK = 2_000_000  # rays x footprint triangles tested at once
_MARGIN = 0.05  # the rays start this fraction of the bodies' height range above the tallest vertex


@dataclass
class GeometricResponse:
    """The energy of the specular rays in the cells of the receiver hemisphere.

    Attributes
    ----------
    directions : :class:`numpy.ndarray`
        The cells' unit directions, shape (n_bins, 3).
    weights : :class:`numpy.ndarray`
        The cells' solid angles in sr, shape (n_bins,).
    sample, reference : :class:`numpy.ndarray`
        The fraction of the incident energy that leaves into each cell, shape (n_sources, n_bins).
    lost_sample, lost_reference : :class:`numpy.ndarray`
        The fraction still bouncing after ``max_bounces`` (trapped in a cavity), shape (n_sources,); rays leaving along the horizon are binned.
        It is counted separately from the cells, so ``sample.sum(axis=1) + lost_sample`` is 1 as a check.
        Pass it on as one more bin when comparing the two (see :func:`scattering_share`).
    depth : float
        ``max(2 σ, |mean|)`` of the height difference between sample and reference over the footprint, in m.
    height_mean, height_spread : float
        The mean and the standard deviation σ of that height difference in m. A mean above ``2 σ`` with a spread above 0 means the object stands on a
        base or backing taller than its relief (see :func:`compas_acoustics.lab.scattering.measure_scattering`).
    """

    directions: np.ndarray
    weights: np.ndarray
    sample: np.ndarray
    reference: np.ndarray
    lost_sample: np.ndarray
    lost_reference: np.ndarray
    depth: float
    height_mean: float = 0.0
    height_spread: float = 0.0


def scattering_share(e_sample, e_reference):
    """One minus the overlap of two energy histograms, ``1 − Σ min(a, b)`` over the last axis.

    Each histogram must sum to 1, so the lost fraction of a :class:`GeometricResponse` must be included as one extra bin::

        scattering_share(np.column_stack([r.sample, r.lost_sample]), np.column_stack([r.reference, r.lost_reference]))

    Passed the cells alone, energy that both bodies lose (rays still bouncing after ``max_bounces``) is
    counted as scattering, and a body compared with itself gives a share above 0.
    """
    return 1.0 - np.minimum(np.asarray(e_sample, dtype=float), np.asarray(e_reference, dtype=float)).sum(axis=-1)


def heuristic_s(s_max, depth, frequencies, c):
    """The heuristic frequency dependence ``s_max · (1 − exp(−(2π f h / c)²))``.

    Parameters
    ----------
    s_max : array-like
        The geometric scattering share, any shape.
    depth : float
        The relief depth h in m.
    frequencies : array-like, shape (n_f,)
        Frequencies in Hz.
    c : float
        Speed of sound in m/s.

    Returns
    -------
    :class:`numpy.ndarray`
        Shape ``s_max.shape + (n_f,)``.
    """
    s_max = np.asarray(s_max, dtype=float)
    k = 2.0 * np.pi * np.asarray(frequencies, dtype=float) / c
    return s_max[..., None] * (1.0 - np.exp(-((k * depth) ** 2)))


def geometric_response(sample, sources, bin_deg=10.0, n_rays=50000, max_bounces=20, seed=0):
    """Trace specular rays off the sample and the reference for every source and bin their exit directions.

    Parameters
    ----------
    sample : :class:`~compas_acoustics.lab.ScatteringSample`
    sources : :class:`~compas_acoustics.lab.SourceGrid`
    bin_deg : float, optional
        Spacing of the receiver cells (``ReceiverGrid.hemisphere``), in degrees: more than 0, at most 90.
    n_rays : int, optional
        Rays per source, at least 1; each carries ``1 / n_rays`` of the energy.
    max_bounces : int, optional
        A ray is reflected at most this many times (on a face or on the baffle plane z = 0), at least once; one that would be reflected once more is lost.
    seed : int, optional
        Seed of the footprint points, which are the same for the sample, the reference and every source.

    Returns
    -------
    :class:`GeometricResponse`

    Raises
    ------
    ValueError
        If ``n_rays`` or ``max_bounces`` is not an integer of at least 1, or ``bin_deg`` is not in (0, 90].
    ImportError
        If ``trimesh`` or its Embree backend is missing (:class:`~compas_acoustics.lab.raycast.RayCaster`).

    Notes
    -----
    The footprint points are drawn from the flat bottom face of the reference (its triangles with normal −z at z = −baseplate),
    area-weighted, and the rays are aimed at their xy position at z = 0. A ray starts above both bodies and travels along −û.
    The bodies are treated as mounted in a rigid baffle at z = 0 (module docstring), for the sample and the reference alike.
    """
    _check_options(bin_deg, n_rays, max_bounces)
    reference = sample.reference
    corners, twice = _bottom_triangles(reference, sample.baseplate)
    launch = _launch_points(corners, twice, n_rays, seed)
    footprint = corners[:, :, :2]
    sample_caster = RayCaster(sample.sample)
    reference_caster = RayCaster(reference)
    z = np.array([vertex[2] for body in (sample.sample, reference) for vertex in (body.vertex_coordinates(v) for v in body.vertices())])
    z_top, z_low = float(z.max()), float(z.min())
    margin = _MARGIN * max(z_top - z_low, 1e-9)
    grid = ReceiverGrid.hemisphere(bin_deg, 1.0)
    tree = cKDTree(grid.directions)
    directions = sources.directions
    n_sources = len(directions)
    e_sample = np.zeros((n_sources, len(grid.weights)))
    e_reference = np.zeros_like(e_sample)
    lost_sample = np.zeros(n_sources)
    lost_reference = np.zeros(n_sources)
    for i, u in enumerate(directions):
        start = launch + u * ((z_top + margin) / u[2])
        e_sample[i], lost_sample[i] = _trace(sample_caster, footprint, start, -u, tree, len(grid.weights), max_bounces)
        e_reference[i], lost_reference[i] = _trace(reference_caster, footprint, start, -u, tree, len(grid.weights), max_bounces)
    depth, mean, spread = _depth(sample_caster, reference_caster, launch, z_top + margin)
    return GeometricResponse(grid.directions, grid.weights, e_sample, e_reference, lost_sample, lost_reference, depth, mean, spread)


def _check_options(bin_deg, n_rays, max_bounces):
    """Raise ``ValueError``, naming the option, unless ``n_rays`` and ``max_bounces`` are integers ≥ 1 and ``0 < bin_deg ≤ 90``."""
    for name, value in (("n_rays", n_rays), ("max_bounces", max_bounces)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("{} must be an integer of at least 1, not {!r}".format(name, value))
    if not 0.0 < float(bin_deg) <= 90.0:
        raise ValueError("bin_deg must be more than 0 and at most 90 degrees, not {!r}".format(bin_deg))


def _bottom_triangles(reference, baseplate):
    """The triangles of the reference's flat bottom face (normal −z, all vertices at z = −baseplate): corners (m, 3, 3) and twice their areas (m,)."""
    vertices, faces = reference.to_vertices_and_faces(triangulated=True)
    vertices = np.asarray(vertices, dtype=float)
    triangles = vertices[np.asarray(faces, dtype=int)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice = np.linalg.norm(cross, axis=1)
    normal_z = cross[:, 2] / np.where(twice > 0, twice, 1.0)
    tolerance = 1e-6 * max(float(np.ptp(vertices, axis=0).max()), 1e-9)
    bottom = (normal_z < -_BOTTOM) & (np.abs(triangles[:, :, 2] + baseplate) < tolerance).all(axis=1)
    if not bottom.any():
        raise ValueError("the reference has no flat bottom face at z = -baseplate to launch the rays from")
    return triangles[bottom], twice[bottom]


def _launch_points(corners, twice, n_rays, seed):
    """``n_rays`` points (x, y, 0) uniform over the triangles ``corners``, area-weighted by triangle."""
    rng = np.random.default_rng(seed)
    which = rng.choice(len(corners), size=n_rays, p=twice / twice.sum())
    r1 = np.sqrt(rng.random(n_rays))
    r2 = rng.random(n_rays)
    a, b, c = corners[which, 0], corners[which, 1], corners[which, 2]
    points = (1.0 - r1)[:, None] * a + (r1 * (1.0 - r2))[:, None] * b + (r1 * r2)[:, None] * c
    points[:, 2] = 0.0
    return points


def _inside(footprint, points):
    """Whether each of ``points`` (k, 2) lies in the footprint, the union of the triangles ``footprint`` (m, 3, 2), edges included.

    A point is in a triangle when the three cross products of its edges with the vector to the point share a sign (orientation does not matter),
    up to ``_EDGE`` times the footprint's size squared.
    """
    tolerance = _EDGE * float(np.ptp(footprint.reshape(-1, 2), axis=0).max()) ** 2
    a, b, c = footprint[:, 0], footprint[:, 1], footprint[:, 2]
    inside = np.zeros(len(points), dtype=bool)
    step = max(_CHUNK // len(footprint), 1)
    for start in range(0, len(points), step):
        p = points[start : start + step, None, :]
        d1 = (b[:, 0] - a[:, 0]) * (p[..., 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (p[..., 0] - a[:, 0])
        d2 = (c[:, 0] - b[:, 0]) * (p[..., 1] - b[:, 1]) - (c[:, 1] - b[:, 1]) * (p[..., 0] - b[:, 0])
        d3 = (a[:, 0] - c[:, 0]) * (p[..., 1] - c[:, 1]) - (a[:, 1] - c[:, 1]) * (p[..., 0] - c[:, 0])
        low = np.minimum(np.minimum(d1, d2), d3)
        high = np.maximum(np.maximum(d1, d2), d3)
        inside[start : start + step] = ((low >= -tolerance) | (high <= tolerance)).any(axis=1)
    return inside


def _trace(caster, footprint, origins, direction, tree, n_bins, max_bounces):
    """Bounce the rays, all with the same first ``direction``; return the energy fraction per cell and the lost fraction.

    Outside ``footprint`` (triangles (m, 3, 2) in xy) the body is mounted in a rigid baffle at z = 0: a ray travelling downward that reaches
    z = 0 before its first face (it misses, or the face is below the plane), at a point outside the footprint, is reflected specularly on the
    plane. That counts as a bounce, like a reflection off a face. Inside the footprint a ray reflects off the face it hits, whatever its height.
    A ray that meets nothing more leaves: upward or along the horizon (``d_z == 0``, grazing the baffle plane) into the nearest cell. The binned
    and the lost rays (still bouncing after ``max_bounces``, or leaving downward through a leak in the body) are counted separately, so that their
    sum is a check.
    """
    n_rays = len(origins)
    origins = np.array(origins)
    directions = np.tile(direction, (n_rays, 1))
    exclude = np.full(n_rays, -1, dtype=int)
    binned = np.zeros(n_bins, dtype=int)
    lost = 0
    for bounce in range(max_bounces + 1):
        if len(origins) == 0:
            break
        triangle, point = caster.first_hits(origins, directions, exclude)
        dz = directions[:, 2]
        hit = triangle >= 0
        # a downward ray from at or above the plane meets it before its first face if it has none or the face is below the plane
        ground = (dz < 0) & (origins[:, 2] >= _BELOW) & (~hit | (point[:, 2] < _BELOW))
        if ground.any():
            g = np.flatnonzero(ground)
            t_ground = np.maximum(-origins[g, 2] / dz[g], 0.0)
            crossing = origins[g, :2] + t_ground[:, None] * directions[g, :2]
            ground[g] = ~_inside(footprint, crossing)
        face = hit & ~ground
        out = ~hit & ~ground & (dz >= 0)  # leaves upward, or along the horizon grazing the baffle plane
        if out.any():
            binned += np.bincount(tree.query(directions[out])[1], minlength=n_bins)
        lost += int((~hit & ~ground & (dz < 0)).sum())  # leaves downward without meeting a face or the baffle: a leak in the body
        again = face | ground
        if bounce == max_bounces:
            lost += int(again.sum())
            break
        if not again.any():
            break
        origins, directions, exclude = origins[again], directions[again], np.full(int(again.sum()), -1, dtype=int)
        ground, face, triangle, point = ground[again], face[again], triangle[again], point[again]
        if ground.any():
            g = directions[ground]
            t_ground = np.maximum(-origins[ground, 2] / g[:, 2], 0.0)
            on_plane = origins[ground] + t_ground[:, None] * g
            on_plane[:, 2] = 0.0
            origins[ground] = on_plane
            g[:, 2] = -g[:, 2]
            directions[ground] = g
        if face.any():
            normal = caster.normals[triangle[face]]
            d = directions[face]
            d = d - 2.0 * np.einsum("ij,ij->i", d, normal)[:, None] * normal
            directions[face] = d / np.linalg.norm(d, axis=1)[:, None]
            origins[face] = point[face]
            exclude[face] = triangle[face]
    return binned / n_rays, lost / n_rays


def _depth(sample_caster, reference_caster, launch, height):
    """The depth ``h = max(2 σ, |mean|)`` of the difference z between the heights where straight-down rays from ``height`` first hit the sample and the reference.

    ``2 σ`` is the depth of a relief (``2 H / sqrt(12)`` = 0.577 H for a sawtooth of height H, 0.707 H for a sine, against a mean of 0.5 H); the mean
    is the height of an object of uniform height over its footprint (a box, a cylinder, a flat-topped panel), whose σ is 0.
    Returns ``(h, mean, σ)``; the mean and σ tell a relief from a base under it (the mean far above 2 σ).
    """
    start = launch + [0.0, 0.0, height]
    down = np.tile([0.0, 0.0, -1.0], (len(launch), 1))
    t_sample, p_sample = sample_caster.first_hits(start, down)
    t_reference, p_reference = reference_caster.first_hits(start, down)
    both = (t_sample >= 0) & (t_reference >= 0)
    if not both.any():
        return 0.0, 0.0, 0.0
    z = p_sample[both, 2] - p_reference[both, 2]
    mean, spread = float(np.mean(z)), float(np.std(z))
    return max(2.0 * spread, abs(mean)), mean, spread
