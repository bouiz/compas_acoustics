r"""The BEM free-field solver: mesh2scattering writes NumCalc projects, NumCalc solves them (Kreuzer et al. 2024).

The sample and its flat reference are rigid closed bodies hit by unit plane waves; NumCalc's boundary element method gives the scattered
pressure at the receivers, the input of the correlation method (Heimes & Vorländer §3.2 and App. A; Mommertz 2000).

- **Splits.** The frequencies are grouped by octave band. Each split is meshed once, for its highest frequency, at
  ``elements_per_wavelength`` (6) elements per wavelength: the edge is :math:`c / (6 f_{max})`.
- **Size check.** An estimate of the element count, the body's area over the area of an equilateral triangle of that edge, is checked
  before anything is written or imported: above ``max_elements`` the solver warns, above ten times that it stops. A model left in
  millimetres has 10⁶ times the elements.
- **Cache.** The projects live in ``<cache_dir>/<key>/split_<k>/{sample,reference}``, the key a hash of the geometry, the directions,
  the frequencies, c and the edges. ``manage_numcalc`` skips finished steps but checks no geometry, so the key decides what is reused:
  an interrupted run resumes, a changed one starts in a new folder.
- **Run.** Each split runs in a child process (``python -m compas_acoustics.lab.solvers._numcalc``), polled every 2 s for progress;
  cancelling the job kills it and its NumCalc instances, and the finished steps stay on disk.

mesh2scattering needs Python ≥ 3.11 and is imported only inside the functions that need it, so this module imports on Python 3.9.

References
----------
Kreuzer, W., Pollack, K., Brinkmann, F., Majdak, P. (2024). NumCalc: An open-source BEM code for solving acoustic scattering problems.
*Engineering Analysis with Boundary Elements* 161, 157-178. mesh2scattering: https://github.com/ahms5/mesh2scattering.

Heimes, A., Vorländer, M. (2025). Scattering coefficients of 3D surfaces from free-field BEM simulations, §3.2 and App. A.
"""

import contextlib
import hashlib
import io
import math
import os
import subprocess
import sys
import warnings
from typing import List

import numpy as np

from compas_acoustics.defaults import Defaults

from .base import Pressures
from .base import ScatteringSolver

__all__ = ["BemSolver", "cache_key", "estimate_elements", "plan_splits"]

_INSTALL = 'the BEM solver needs Python >= 3.11 and mesh2scattering >= 1.0.5: pip install "compas_acoustics[lab]"'
_GRID = "receivers"
_BODIES = ("sample", "reference")
_POLL = 2.0  # s
_FLAT = math.cos(1e-3)  # faces within 1 mrad (0.06°) of each other count as one plane for the flips and collapses
_TOO_BIG = (
    "the BEM mesh of the {body} would have about {n:,} elements at {f:.0f} Hz (edge {edge:.3g} m), more than 10 x max_elements = {limit:,}; "
    "is the model in millimetres? Use a smaller diameter or the Kirchhoff method"
)
_BIG = "the BEM mesh of the {body} has about {n:,} elements at {f:.0f} Hz, above max_elements = {limit:,}; this is slow: use a smaller diameter or the Kirchhoff method"


def plan_splits(frequencies, nominal_octaves=Defaults._DEFAULT_FREQUENCY_BANDS) -> List[np.ndarray]:
    """The frequencies grouped by the octave band whose centre is nearest on a log scale, low to high.

    With the lab's simulation frequencies (three per one-third-octave band) every split holds the bands of one octave
    (ISO 354 §8.3: 125 ← 100, 125, 160; …), and each split is meshed once, for its highest frequency.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Frequencies in Hz.
    nominal_octaves : sequence of float, optional
        The octave centres.

    Returns
    -------
    list of numpy.ndarray
        The non-empty splits, each ascending.
    """
    f = np.sort(np.atleast_1d(np.asarray(frequencies, dtype=float)))
    octaves = np.log(np.asarray(nominal_octaves, dtype=float))
    owner = np.argmin(np.abs(np.log(f)[:, None] - octaves[None, :]), axis=1)
    return [f[owner == k] for k in range(len(octaves)) if np.any(owner == k)]


def _arrays(mesh):
    vertices, faces = mesh.to_vertices_and_faces()
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def _area(V, F):
    return float(np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum() / 2)


def estimate_elements(mesh, edge) -> int:
    """The element count of a mesh of equilateral triangles of side ``edge`` over the area of ``mesh``: ``ceil(A / (√3/4 · edge²))``.

    Parameters
    ----------
    mesh : :class:`compas.datastructures.Mesh`
        A triangulated mesh.
    edge : float
        The element edge in m.
    """
    V, F = _arrays(mesh)
    return int(math.ceil(_area(V, F) / (math.sqrt(3) / 4 * edge**2)))


def cache_key(sample, sources, receivers, frequencies, c, edge) -> str:
    """12 hex characters of a SHA-1 over what the NumCalc projects depend on.

    That is the vertex coordinates of the sample and the reference rounded to 1e-6 m, their faces, the source and receiver directions,
    the frequencies, the speed of sound and the element edge (one or one per split).
    """

    def add(label, array, decimals=None):
        array = np.asarray(array, dtype=float)
        if decimals is not None:
            array = np.round(array, decimals) + 0.0  # + 0.0 turns -0.0 into 0.0
        h.update(label.encode())
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(array).tobytes())

    h = hashlib.sha1()
    for name in _BODIES:
        V, F = _arrays(getattr(sample, name))
        add(name + ".vertices", np.round(V / 1e-6))
        add(name + ".faces", F)
    add("sources", sources.directions, 12)
    add("receivers", receivers.directions, 12)
    add("frequencies", np.sort(np.atleast_1d(frequencies)), 6)
    add("c", [c], 9)
    add("edge", np.atleast_1d(edge), 12)
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------------------------------------------------------------
# remeshing: trimesh's midpoint subdivision with Delaunay edge flips in the planes
# --------------------------------------------------------------------------------------------------------------------------------


def _normals(V, F):
    cross = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    twice = np.linalg.norm(cross, axis=1)
    return cross / np.where(twice > 0, twice, 1.0)[:, None], twice


def _cot(apex, p, q):
    """The cotangent of the angle at ``apex`` in the triangles (apex, p, q)."""
    a, b = p - apex, q - apex
    return np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(np.cross(a, b), axis=1), 1e-300)


def _flip_to_delaunay(V, F, max_length=None, max_passes=100000):
    """Lawson flips of the edges between two faces in one plane (within 1 mrad) that are not locally Delaunay.

    A flip there leaves the surface as it is and removes the slivers that fans (earclipped caps) and long quads leave, which
    midpoint subdivision would otherwise copy at every level. Each pass flips a set of edges with no face in common; with
    ``max_length`` no flip makes an edge longer than that.
    """
    F = F.copy()
    n = len(V)
    for _ in range(max_passes):
        a = F.ravel()
        b = F[:, [1, 2, 0]].ravel()
        opposite = F[:, [2, 0, 1]].ravel()
        face = np.repeat(np.arange(len(F)), 3)
        keys = a * n + b
        twin_keys = b * n + a
        order = np.argsort(keys)
        pos = np.searchsorted(keys[order], twin_keys)
        pos = np.minimum(pos, len(keys) - 1)
        twin = order[pos]
        found = keys[twin] == twin_keys
        first = found & (a < b)
        h1 = np.flatnonzero(first)
        h2 = twin[h1]
        f1, f2 = face[h1], face[h2]
        va, vb, vc, vd = a[h1], b[h1], opposite[h1], opposite[h2]
        normals, twice = _normals(V, F)
        coplanar = (twice[f1] > 0) & (twice[f2] > 0) & (np.einsum("ij,ij->i", normals[f1], normals[f2]) > _FLAT)
        badness = _cot(V[vc], V[va], V[vb]) + _cot(V[vd], V[va], V[vb])
        candidate = coplanar & (badness < -1e-9) & (vc != vd)
        if max_length is not None:
            candidate &= np.linalg.norm(V[vc] - V[vd], axis=1) <= max_length
        if candidate.any():
            # the new edge (c, d) must not exist yet
            existing = np.sort(np.column_stack([a, b]), axis=1)
            existing = existing[:, 0] * n + existing[:, 1]
            new = np.sort(np.column_stack([vc, vd]), axis=1)
            new = new[:, 0] * n + new[:, 1]
            candidate &= ~np.isin(new, existing)
        if candidate.any():
            # both new faces must keep the orientation
            n1 = np.cross(V[vd] - V[va], V[vc] - V[va])
            n2 = np.cross(V[vb] - V[vd], V[vc] - V[vd])
            candidate &= (np.einsum("ij,ij->i", n1, normals[f1]) > 0) & (np.einsum("ij,ij->i", n2, normals[f1]) > 0)
        index = np.flatnonzero(candidate)
        if not len(index):
            return F
        index = index[np.argsort(badness[index], kind="stable")]
        # keep a flip only where it is the worst one of both its faces, so no face takes part in two flips
        pair_faces = np.concatenate([f1[index], f2[index]])
        rank = np.concatenate([np.arange(len(index))] * 2)
        _, first_seen = np.unique(pair_faces, return_index=True)
        winners = np.zeros(len(index), dtype=int)
        np.add.at(winners, rank[first_seen], 1)
        chosen = index[winners == 2]
        _, unique_new = np.unique(new[chosen], return_index=True)
        chosen = chosen[np.sort(unique_new)]
        F[f1[chosen]] = np.column_stack([va[chosen], vd[chosen], vc[chosen]])
        F[f2[chosen]] = np.column_stack([vd[chosen], vb[chosen], vc[chosen]])
    raise RuntimeError("the edge flips did not converge")


def _collapse_short_edges(V, F, edge, shortest):
    """Collapse the edges shorter than ``shortest`` where every face that remains keeps its plane (within 1 mrad) and orientation.

    A collapse of v onto u replaces v by u in v's other faces and removes the two faces on the edge. When u lies in the plane of each
    of those faces the surface stays as it is: so a vertex inside a plane can move onto any neighbour, a vertex on a straight crease
    only along the crease, and a corner not at all. No new edge may be longer than ``edge``, and the link condition keeps the mesh
    manifold. Shortest edges first.

    Returns
    -------
    tuple
        The vertices (unused ones removed), the faces and the number of collapses.
    """
    F = [list(f) for f in F]
    alive = [True] * len(F)
    normals, twice = _normals(V, np.asarray(F, dtype=np.int64))
    faces_of = [set() for _ in range(len(V))]
    for index, f in enumerate(F):
        for v in f:
            faces_of[v].add(index)
    edges = np.unique(np.sort(np.asarray(F, dtype=np.int64)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    short = np.flatnonzero(lengths < shortest)
    short = short[np.argsort(lengths[short], kind="stable")]
    limit = edge * edge
    tiny = 1e-12 * edge * edge

    def neighbours(v):
        return {w for f in faces_of[v] for w in F[f]} - {v}

    def collapse(v, u):
        shared = [f for f in faces_of[v] if u in F[f]]
        if len(shared) != 2:
            return False
        if len(neighbours(v) & neighbours(u)) != 2:
            return False
        moved = [f for f in faces_of[v] if f not in shared]
        new_faces = []
        for f in moved:
            new = [u if w == v else w for w in F[f]]
            p = V[new]
            cross = np.cross(p[1] - p[0], p[2] - p[0])
            area = float(np.linalg.norm(cross))
            if area <= tiny or float(cross @ normals[f]) <= _FLAT * area:
                return False
            for w in new:
                if w != u and float(((V[w] - V[u]) ** 2).sum()) > limit:
                    return False
            new_faces.append((f, new, cross / area))
        for f in shared:
            alive[f] = False
            for w in F[f]:
                faces_of[w].discard(f)
        for f, new, normal in new_faces:
            F[f] = new
            normals[f] = normal
            faces_of[u].add(f)
        faces_of[v] = set()
        return True

    count = 0
    for a, b in edges[short]:
        if not faces_of[a] or not faces_of[b] or b not in neighbours(a):
            continue
        if collapse(a, b) or collapse(b, a):
            count += 1
    F = np.asarray([f for f, keep in zip(F, alive) if keep], dtype=np.int64)
    used = np.unique(F)
    index = np.full(len(V), -1, dtype=np.int64)
    index[used] = np.arange(len(used))
    return V[used], index[F], count


def _remesh(mesh, edge, max_rounds=8):
    """A closed triangle mesh with all edges at most ``edge`` and near-equilateral faces in the planes of the input.

    1. Delaunay flips in the planes (:func:`_flip_to_delaunay`) clear the sliver fans of the earclipped caps.
    2. ``trimesh.remesh.subdivide_to_size`` bisects every edge longer than ``edge`` at shared midpoints, so the mesh stays closed.
    3. Collapses of edges shorter than ``0.6 edge`` that leave every face in its plane (:func:`_collapse_short_edges`) and flips,
       repeated, thin out the points that midpoint refinement crowds where many long edges meet.

    This is isotropic remeshing (Botsch & Kobbelt 2004) restricted to moves that keep the surface: the vertices stay where they are.
    Subdivision alone keeps the shape of every input triangle and gave 30-45 times the area estimate (:func:`estimate_elements`) on a
    disc; this gives a few times it, more where the input's own faces are narrower than ``edge``.

    Returns
    -------
    tuple of numpy.ndarray
        Vertices (n, 3) and faces (m, 3), oriented as the input.
    """
    import trimesh

    V, F = _arrays(mesh)
    F = _flip_to_delaunay(V, F)
    V, F = trimesh.remesh.subdivide_to_size(V, F, edge, max_iter=64)
    V, F = np.asarray(V, dtype=float), np.asarray(F, dtype=np.int64)
    F = _flip_to_delaunay(V, F, max_length=edge)
    for _ in range(max_rounds):
        V, F, collapsed = _collapse_short_edges(V, F, edge, 0.6 * edge)
        F = _flip_to_delaunay(V, F, max_length=edge)
        if not collapsed:
            break
    return V, F


# --------------------------------------------------------------------------------------------------------------------------------
# the solver
# --------------------------------------------------------------------------------------------------------------------------------


def _require():
    """mesh2scattering and its companions, or an ImportError that names the extra."""
    if sys.version_info < (3, 11):
        raise ImportError(_INSTALL + " (this is Python {}.{})".format(*sys.version_info[:2]))
    try:
        import mesh2scattering
        import mesh2scattering.input  # noqa: F401
        import mesh2scattering.numcalc  # noqa: F401
        import mesh2scattering.output  # noqa: F401
        import psutil  # noqa: F401
        import pyfar  # noqa: F401
        import sofar  # noqa: F401
        import trimesh  # noqa: F401
    except ImportError as error:
        raise ImportError("{} ({})".format(_INSTALL, error)) from error
    return mesh2scattering


def _finished(source_folder, step):
    """Whether NumCalc finished a frequency step (as ``mesh2scattering.numcalc``'s own check): its pEvalGrid and "End time:" exist."""
    if not os.path.isfile(os.path.join(source_folder, "be.out", "be.{}".format(step), "pEvalGrid")):
        return False
    out = os.path.join(source_folder, "NC{0}-{0}.out".format(step))
    if not os.path.isfile(out):
        return False
    with open(out, "r", encoding="utf8", errors="replace") as handle:
        return "End time:" in handle.read()


def _kill_tree(process):
    """Kill a child process and everything it started (its NumCalc instances)."""
    import psutil

    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return
    try:
        parent.suspend()  # so it starts nothing more while its children are collected
    except psutil.Error:
        pass
    try:
        children = parent.children(recursive=True)
    except psutil.Error:
        children = []
    for p in children + [parent]:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(children + [parent], timeout=10)


class BemSolver(ScatteringSolver):
    """Free-field BEM pressures of a rigid sample and its reference, with mesh2scattering and NumCalc (Kreuzer et al. 2024).

    The setup follows Heimes & Vorländer §3.2 and App. A: both bodies closed on the baseplate, unit plane waves travelling along −û,
    receivers on a hemisphere in the far field. See the module docstring for the splits, the cache and the run.

    Parameters
    ----------
    cache_dir : str, optional
        Where the NumCalc projects are kept; ``~/.compas_acoustics/lab_cache`` by default.
    numcalc_path : str, optional
        The NumCalc binary; by default ``$COMPAS_ACOUSTICS_NUMCALC``, else ``mesh2scattering.numcalc.build_or_fetch_numcalc()``.
    max_elements : int, optional
        The elements per body above which the solver warns; above ten times this it stops. At 100 000 elements a frequency step
        takes about 10 GB of RAM (53 000 took 3.5 GB and 75 s on the spike machine).
    elements_per_wavelength : int, optional
        The mesh resolution at each split's highest frequency.
    density : float, optional
        The density of air in kg/m³.
    """

    name = "bem"

    def __init__(self, cache_dir=None, numcalc_path=None, max_elements=100_000, elements_per_wavelength=6, density=1.1839):
        self.cache_dir = cache_dir if cache_dir is not None else os.path.join(os.path.expanduser("~"), ".compas_acoustics", "lab_cache")
        self.numcalc_path = numcalc_path
        self.max_elements = int(max_elements)
        self.elements_per_wavelength = int(elements_per_wavelength)
        self.density = float(density)
        if self.max_elements < 1 or self.elements_per_wavelength < 1:
            raise ValueError("max_elements and elements_per_wavelength must be at least 1")

    # -- planning -------------------------------------------------------------------------------------------------------------

    def _edges(self, splits, c):
        return [float(c) / (float(s.max()) * self.elements_per_wavelength) for s in splits]

    def _check_size(self, sample, splits, edges):
        """Warn above ``max_elements``, raise above ten times that (before anything is imported or written)."""
        for split, edge in zip(splits, edges):
            for body in _BODIES:
                n = estimate_elements(getattr(sample, body), edge)
                if n > 10 * self.max_elements:
                    raise ValueError(_TOO_BIG.format(body=body, n=n, f=split.max(), edge=edge, limit=10 * self.max_elements))
        largest = max((estimate_elements(getattr(sample, body), edges[-1]), body) for body in _BODIES)
        if largest[0] > self.max_elements:
            warnings.warn(_BIG.format(body=largest[1], n=largest[0], f=splits[-1].max(), limit=self.max_elements), stacklevel=3)

    def _numcalc(self):
        if self.numcalc_path:
            return self.numcalc_path
        if os.environ.get("COMPAS_ACOUSTICS_NUMCALC"):
            return os.environ["COMPAS_ACOUSTICS_NUMCALC"]
        from mesh2scattering import numcalc

        return numcalc.build_or_fetch_numcalc()

    @staticmethod
    def radius(sample, f_max, c):
        """The evaluation radius ``max(10 d, 2 d² / λ_min)``, in the far field of the sample (Heimes & Vorländer App. A)."""
        d = sample.diameter
        return float(max(10.0 * d, 2.0 * d**2 * f_max / c))

    # -- solving --------------------------------------------------------------------------------------------------------------

    def solve(self, sample, sources, receivers, frequencies, c):
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        splits = plan_splits(frequencies)
        edges = self._edges(splits, c)
        self._check_size(sample, splits, edges)
        m2s = _require()
        numcalc_path = self._numcalc()

        ordered = np.concatenate(splits)
        root = os.path.abspath(os.path.join(self.cache_dir, cache_key(sample, sources, receivers, ordered, c, edges)))
        radius = self.radius(sample, ordered.max(), c)
        pressures = {body: [] for body in _BODIES}
        elements = []
        for k, (split, edge) in enumerate(zip(splits, edges)):
            folder = os.path.join(root, "split_{}".format(k + 1))
            counts = {}
            for body in _BODIES:
                counts[body] = self._write(os.path.join(folder, body), body, sample, sources, receivers, split, edge, radius, c)
            elements.append(counts)
            self._run(folder, numcalc_path, "bem: split {}/{}".format(k + 1, len(splits)), sources.n, len(split))
            for body in _BODIES:
                pressures[body].append(self._read(os.path.join(folder, body), body, sources, receivers, radius, split))

        # back to the order of the frequencies given
        order = np.argsort(frequencies, kind="stable")
        out = {}
        for body in _BODIES:
            joined = np.concatenate(pressures[body], axis=2)
            out[body] = np.empty_like(joined)
            out[body][:, :, order] = joined
        generation = {
            "method": self.name,
            "mesh2scattering": m2s.__version__,
            "elements": elements,
            "edge": edges,
            "radius": radius,
            "cache": root,
        }
        return Pressures(out["sample"], out["reference"], frequencies, generation)

    def _write(self, project, title, sample, sources, receivers, frequencies, edge, radius, c):
        """Write one NumCalc project unless its ``parameters.json`` (written last) exists; return its element count."""
        elements_file = os.path.join(project, "ObjectMeshes", "Reference", "Elements.txt")
        if os.path.isfile(os.path.join(project, "parameters.json")) and os.path.isfile(elements_file):
            with open(elements_file) as handle:
                return int(handle.readline().split()[0])

        import pyfar as pf
        import trimesh
        from mesh2scattering import input as m2s_input

        V, F = _remesh(getattr(sample, title), edge)
        if len(F) > self.max_elements:
            warnings.warn(_BIG.format(body=title, n=len(F), f=frequencies.max(), limit=self.max_elements), stacklevel=3)
        period = sample.period or (0.0, 0.0)
        if title == "sample":
            surface = m2s_input.SurfaceDescription(
                structural_wavelength_x=float(period[0]),
                structural_wavelength_y=float(period[1]),
                structural_depth=float(sample.structural_depth),
                surface_type=m2s_input.SurfaceType.PERIODIC if sample.period else m2s_input.SurfaceType.STOCHASTIC,
            )
        else:
            surface = m2s_input.SurfaceDescription(
                structural_wavelength_x=float(period[0]), structural_wavelength_y=float(period[1]), structural_depth=0.0, surface_type=m2s_input.SurfaceType.FLAT
            )
        body = m2s_input.SampleMesh(trimesh.Trimesh(V, F, process=False), surface, sample_baseplate_hight=float(sample.baseplate), sample_diameter=float(sample.diameter))
        travel = -np.asarray(sources.directions, dtype=float)  # a plane wave from û travels along −û
        weights = np.asarray(sources.paris_weights, dtype=float)
        source = m2s_input.SoundSource(pf.Coordinates(travel[:, 0], travel[:, 1], travel[:, 2], weights=weights), source_type=m2s_input.SoundSourceType.PLANE_WAVE)
        points = np.asarray(receivers.directions, dtype=float) * radius
        grid = m2s_input.EvaluationGrid.from_spherical(pf.Coordinates(points[:, 0], points[:, 1], points[:, 2], weights=np.asarray(receivers.weights, dtype=float)), _GRID)
        os.makedirs(project, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            m2s_input.write_scattering_project_numcalc(
                project_path=project,
                project_title=title,
                frequencies=np.asarray(frequencies, dtype=float),
                sound_sources=source,
                evaluation_grids=[grid],
                sample_mesh=body,
                speed_of_sound=float(c),
                density_of_medium=self.density,
            )
        return int(len(F))

    def _run(self, folder, numcalc_path, stage, n_sources, n_steps):
        """Run the unfinished steps of a split's projects in a child process, reporting progress every 2 s; kill it on cancel."""
        from compas_acoustics.worker import jobs

        steps = [(os.path.join(folder, body, "NumCalc", "source_{}".format(s + 1)), i + 1) for body in _BODIES for s in range(n_sources) for i in range(n_steps)]
        done = set()

        def count():
            for item in steps:
                if item not in done and _finished(*item):
                    done.add(item)
            return len(done)

        jobs.progress(stage, count(), len(steps))
        if len(done) == len(steps):
            return
        log_path = os.path.join(folder, "numcalc.log")
        with open(log_path, "a") as log:
            process = subprocess.Popen([sys.executable, "-m", "compas_acoustics.lab.solvers._numcalc", folder, numcalc_path], stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        process.wait(timeout=_POLL)
                    except subprocess.TimeoutExpired:
                        pass
                    jobs.progress(stage, count(), len(steps))
                    if process.poll() is not None:
                        break
            except BaseException:
                _kill_tree(process)
                raise
        if process.returncode != 0 or count() < len(steps):
            with open(log_path, errors="replace") as handle:
                tail = "".join(handle.readlines()[-20:])
            raise RuntimeError("NumCalc finished {} of {} steps in {} (exit code {}); see {}:\n{}".format(len(done), len(steps), folder, process.returncode, log_path, tail))

    def _read(self, project, title, sources, receivers, radius, frequencies):
        """The scattered pressures of a finished project, (n_sources, n_receivers, n_frequencies) in the order of the grids given."""
        import pyfar as pf
        import sofar
        from mesh2scattering import output
        from scipy.spatial import cKDTree

        path = os.path.normpath(os.path.join(project, "..", "{}_{}.pressure.sofa".format(title, _GRID)))
        if not os.path.isfile(path):
            with contextlib.redirect_stdout(io.StringIO()):
                output.write_pressure(project)
        data, source_coordinates, receiver_coordinates = pf.io.convert_sofa(sofar.read_sofa(path, verify=False))[:3]
        p = np.asarray(data.freq)
        f_order = np.argsort(data.frequencies)
        if len(f_order) != len(frequencies) or not np.allclose(np.asarray(data.frequencies)[f_order], frequencies, rtol=1e-9):
            raise RuntimeError("{} holds the frequencies {}, expected {}".format(path, data.frequencies, frequencies))
        # the SOFA source position of a plane wave is -(travel direction) x 99 = û x 99
        distance, s_order = cKDTree(source_coordinates.cartesian / 99.0).query(np.asarray(sources.directions, dtype=float))
        if distance.max() > 1e-6:
            raise RuntimeError("{}: the sources do not match the source grid (off by {:.3g})".format(path, distance.max()))
        distance, r_order = cKDTree(receiver_coordinates.cartesian).query(np.asarray(receivers.directions, dtype=float) * radius)
        if distance.max() > 1e-6 * radius:
            raise RuntimeError("{}: the receivers do not match the receiver grid (off by {:.3g} m)".format(path, distance.max()))
        return p[s_order][:, r_order][:, :, f_order]

    def warnings(self, sample, nominal_bands, exact_bands, c) -> List[str]:
        """A note when the mesh for the highest simulated frequency (the top band's ``fc·2^(1/9)``) exceeds ``max_elements``."""
        exact = np.atleast_1d(np.asarray(exact_bands, dtype=float))
        if not exact.size:
            return []
        f_max = exact.max() * 2 ** (1 / 9)
        edge = c / (f_max * self.elements_per_wavelength)
        n, body = max((estimate_elements(getattr(sample, b), edge), b) for b in _BODIES)
        return [_BIG.format(body=body, n=n, f=f_max, limit=self.max_elements)] if n > self.max_elements else []
