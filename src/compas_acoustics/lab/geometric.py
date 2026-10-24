"""The geometric scattering engine: specular rays off the rigid sample and reference, binned on the receiver hemisphere.

Per source direction, parallel rays aimed at uniformly random points of the reference's footprint are reflected specularly off the rigid
sample and, with the same rays, off the rigid reference (the flat footprint or a proxy), and leave upward into one cell of a receiver
hemisphere grid. Energy is conserved by construction: what does not leave upward is counted as lost. There is no phase and no
diffraction: the rim and any wave effect are ignored, and the frequency dependence is not computed but imposed afterwards by
:func:`heuristic_s`, in the form of ``AcousticMaterial.scattering_coefficients_from_depth``, from the depth of the relief.

The object is treated as mounted in a rigid infinite baffle at z = 0 (the objects of the lab are mounted on a wall, floor or ceiling): a ray
travelling downward that meets no face above z = 0 is reflected specularly on the plane z = 0, which counts as one bounce. A ray that leaves
upward is binned; one that leaves exactly along the horizon, or is still bouncing after ``max_bounces``, is lost.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .grids import ReceiverGrid
from .raycast import RayCaster

__all__ = ["GeometricResponse", "geometric_response", "heuristic_s", "scattering_share"]

_BOTTOM = 0.999  # a triangle with a normal z below minus this is part of a flat bottom face
_BELOW = -1e-9  # a hit lower than this (m) is below the baffle; the double-precision hit points on z = 0 are within 1e-15 of it
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
        The fraction that never leaves upward: still bouncing after ``max_bounces``, or leaving exactly along the horizon (``d_z == 0``),
        shape (n_sources,). It is counted separately from the cells, so ``sample.sum(axis=1) + lost_sample`` is 1 as a check.
        Pass it on as one more bin when comparing the two (see :func:`scattering_share`).
    depth : float
        ``2 σ`` of the height difference between sample and reference over the footprint, in m.
    """

    directions: np.ndarray
    weights: np.ndarray
    sample: np.ndarray
    reference: np.ndarray
    lost_sample: np.ndarray
    lost_reference: np.ndarray
    depth: float


def scattering_share(e_sample, e_reference):
    """One minus the overlap of two energy histograms, ``1 − Σ min(a, b)`` over the last axis.

    Each histogram must sum to 1, so the lost fraction of a :class:`GeometricResponse` must be included as one extra bin::

        scattering_share(np.column_stack([r.sample, r.lost_sample]), np.column_stack([r.reference, r.lost_reference]))

    Passed the cells alone, energy that both bodies lose (rays still bouncing after ``max_bounces``, rays leaving along the horizon) is
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
        Spacing of the receiver cells (``ReceiverGrid.hemisphere``).
    n_rays : int, optional
        Rays per source; each carries ``1 / n_rays`` of the energy.
    max_bounces : int, optional
        A ray is reflected at most this many times (on a face or on the baffle plane z = 0); one that would be reflected once more is lost.
    seed : int, optional
        Seed of the footprint points, which are the same for the sample, the reference and every source.

    Returns
    -------
    :class:`GeometricResponse`

    Notes
    -----
    The footprint points are drawn from the flat bottom face of the reference (its triangles with normal −z at z = −baseplate),
    area-weighted, and the rays are aimed at their xy position at z = 0. A ray starts above both bodies and travels along −û.
    The bodies are treated as mounted in a rigid baffle at z = 0 (module docstring), for the sample and the reference alike.
    """
    reference = sample.reference
    launch = _launch_points(reference, sample.baseplate, n_rays, seed)
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
        e_sample[i], lost_sample[i] = _trace(sample_caster, start, -u, tree, len(grid.weights), max_bounces)
        e_reference[i], lost_reference[i] = _trace(reference_caster, start, -u, tree, len(grid.weights), max_bounces)
    depth = _depth(sample_caster, reference_caster, launch, z_top + margin)
    return GeometricResponse(grid.directions, grid.weights, e_sample, e_reference, lost_sample, lost_reference, depth)


def _launch_points(reference, baseplate, n_rays, seed):
    """``n_rays`` points (x, y, 0) uniform over the reference's flat bottom face, area-weighted by triangle."""
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
    corners = triangles[bottom]
    area = twice[bottom]
    rng = np.random.default_rng(seed)
    which = rng.choice(len(corners), size=n_rays, p=area / area.sum())
    r1 = np.sqrt(rng.random(n_rays))
    r2 = rng.random(n_rays)
    a, b, c = corners[which, 0], corners[which, 1], corners[which, 2]
    points = (1.0 - r1)[:, None] * a + (r1 * (1.0 - r2))[:, None] * b + (r1 * r2)[:, None] * c
    points[:, 2] = 0.0
    return points


def _trace(caster, origins, direction, tree, n_bins, max_bounces):
    """Bounce the rays, all with the same first ``direction``; return the energy fraction per cell and the lost fraction.

    The body is mounted in a rigid baffle at z = 0: a ray travelling downward that meets no face above z = 0 (it misses, or its first face
    lies below the baffle) is reflected specularly on the plane z = 0. That counts as a bounce, like a reflection off a face.
    The binned and the lost rays are counted separately, so that their sum is a check.
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
        # a downward ray whose first face is below the baffle meets the baffle first
        ground = (dz < 0) & (~hit | (point[:, 2] < _BELOW))
        face = hit & ~ground
        up = ~hit & (dz > 0)
        if up.any():
            binned += np.bincount(tree.query(directions[up])[1], minlength=n_bins)
        flat = ~hit & (dz == 0)  # leaves along the horizon, neither up nor down
        lost += int(flat.sum())
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
    """``2 σ`` of the difference between the heights where straight-down rays from ``height`` first hit the sample and the reference."""
    start = launch + [0.0, 0.0, height]
    down = np.tile([0.0, 0.0, -1.0], (len(launch), 1))
    t_sample, p_sample = sample_caster.first_hits(start, down)
    t_reference, p_reference = reference_caster.first_hits(start, down)
    both = (t_sample >= 0) & (t_reference >= 0)
    if not both.any():
        return 0.0
    return float(2.0 * np.std(p_sample[both, 2] - p_reference[both, 2]))
