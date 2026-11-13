"""User-facing issues of a :class:`~compas_acoustics.geometry.prepare.PreparedMesh`: what will go wrong, and why.

:func:`analyze` checks the sources and receivers (outside the model, inside an object, on a surface), the paths
between them (blocked, or in separate volumes), leaks and holes, the automatic fixes (flipped and two-sided faces),
the faces and objects sound never reaches, and the cost of the image source search. Every issue has a stable ``code``, a ``severity`` (``"error"``,
``"warning"`` or ``"info"``) and what to highlight (faces, objects, boundary loops, a point or a segment). Checks
specific to one solver (a wave solver's grid) are left to the caller, which adds them with :meth:`Issues.add`.

Ported from compas_soundscape (``backend/services/simulation_preflight_service.py``).
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

from .prepare import FACE_CLASS_SKIPPED
from .prepare import PreparedMesh
from .raycast import TriangleSet
from .raycast import point_triangle_distances
from .raycast import segment_crossings
from .visibility import SEED_LEAKY
from .visibility import SEED_OUTSIDE
from .visibility import seed_enclosures

__all__ = ["Seed", "GeometryIssue", "Issues", "GeometryReport", "ReportOptions", "analyze", "summarize"]


@dataclass
class Seed:
    """A source or receiver, in the order of :attr:`PreparedMesh.seeds`."""

    id: str
    kind: str  # "source" | "receiver"
    position: List[float]


@dataclass
class GeometryIssue:
    id: str
    severity: str  # "error" | "warning" | "info"
    code: str
    title: str
    detail: str
    position: Optional[List[float]] = None
    segment: Optional[List[List[float]]] = None
    face_ids: List[int] = field(default_factory=list)
    object_ids: List[str] = field(default_factory=list)
    loop_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReportOptions:
    """Thresholds and what the simulation will do. Lengths in metres.

    ``image_source_order`` 0 skips the image source cost check (a wave solver); ``ray_tracing`` warns about
    receivers inside the tracer's detection sphere of ``receiver_radius``.
    """

    image_source_order: int = 0
    ray_tracing: bool = False
    receiver_radius: float = 0.5
    surface_error_distance: float = 0.05
    leak_warning_fraction: float = 0.005
    leak_error_fraction: float = 0.05
    hole_leak_match_distance: float = 0.5
    ism_paths_warning: float = 2e7
    ism_paths_error: float = 5e8
    model_diagonal_min: float = 1.0
    model_diagonal_max: float = 1000.0
    max_listed_items: int = 25
    max_issue_face_ids: int = 5000
    interior_object_max_depth: float = 1.0


class Issues:
    """An ordered list of issues with stable ids ``"<code>:<index>"``."""

    def __init__(self, options: ReportOptions):
        self.items: List[GeometryIssue] = []
        self.options = options

    def add(self, severity: str, code: str, title: str, detail: str, **kw) -> None:
        if "face_ids" in kw:
            kw["face_ids"] = list(kw["face_ids"])[: self.options.max_issue_face_ids]
        if "object_ids" in kw:
            kw["object_ids"] = list(kw["object_ids"])[: self.options.max_listed_items]
        self.items.append(GeometryIssue(id=f"{code}:{len(self.items)}", severity=severity, code=code, title=title, detail=detail, **kw))


@dataclass
class GeometryReport:
    issues: List[GeometryIssue]
    enclosure: List[int]  # enclosure id per seed
    loop_leaking: List[bool]  # per boundary loop
    blocked_paths: List[dict] = field(default_factory=list)
    seed_surface_distance: List[float] = field(default_factory=list)
    ism_paths_estimate: float = 0.0


def summarize(issues: Sequence[GeometryIssue]) -> Dict[str, int]:
    return {
        "n_errors": sum(i.severity == "error" for i in issues),
        "n_warnings": sum(i.severity == "warning" for i in issues),
        "n_infos": sum(i.severity == "info" for i in issues),
    }


def enclosures(mesh: PreparedMesh) -> List[int]:
    """Seeds reaching a common (face, side) share the same air volume (:func:`.visibility.seed_enclosures`)."""
    return seed_enclosures(mesh.visibility)


def _object_list(mesh: PreparedMesh, face_ids, limit: int) -> List[str]:
    objs = sorted({mesh.objects[o] for o in mesh.face_object[np.asarray(face_ids, dtype=np.int64)] if o >= 0})
    return objs[:limit]


def _label(seed: Seed) -> str:
    return f"{seed.kind.capitalize()} '{seed.id}'"


def analyze(
    mesh: PreparedMesh,
    seeds: Sequence[Seed],
    pairs: Sequence[Tuple[str, str]],
    options: Optional[ReportOptions] = None,
    issues: Optional[Issues] = None,
) -> GeometryReport:
    """The issues of ``mesh`` for the ``pairs`` of ``(source id, receiver id)`` among ``seeds``.

    ``issues`` may hold issues the caller added first; the solver-specific ones go after, through the returned
    report's list.
    """
    options = options or ReportOptions()
    out = issues or Issues(options)
    tris = TriangleSet(mesh.vertices, mesh.faces)
    vis = mesh.visibility
    enclosure = enclosures(mesh)
    seed_index = {s.id: i for i, s in enumerate(seeds)}
    limit = options.max_listed_items

    # ── Sources / receivers ──────────────────────────────────────────────
    dists: List[float] = []
    for i, (meta, sv) in enumerate(zip(seeds, vis.seeds)):
        d = float(point_triangle_distances(tris, sv.position).min()) if len(tris) else float("inf")
        dists.append(d)
        pos = [float(x) for x in sv.position]
        if sv.status == SEED_OUTSIDE:
            out.add(
                "error",
                "seed_outside",
                f"{_label(meta)} is outside the model",
                f"{sv.escape_fraction:.0%} of its rays escape without hitting a surface. "
                "It has no enclosing room, so it will not be simulated correctly "
                "(outer walls never block it). Move it inside the room.",
                position=pos,
            )
            continue
        if sv.status == SEED_LEAKY:
            out.add(
                "warning",
                "seed_leaky",
                f"{_label(meta)} sees an opening",
                f"{sv.escape_fraction:.0%} of its direct rays leave the model. The room is open in its line of sight (missing wall, ceiling or material).",
                position=pos,
            )
        alone = all(enclosure[j] != enclosure[i] for j in range(len(seeds)) if j != i)
        if alone and len(seeds) > 1 and sv.mean_primary_hit_m < options.interior_object_max_depth:
            out.add(
                "error",
                "seed_inside_object",
                f"{_label(meta)} is inside an object",
                f"It is enclosed by surfaces {sv.mean_primary_hit_m:.2f} m away on average and shares no air with the other sources/receivers.",
                position=pos,
                object_ids=_object_list(mesh, sv.seen_faces, limit),
            )
        if d < options.surface_error_distance:
            out.add(
                "error",
                "seed_on_surface",
                f"{_label(meta)} touches a surface",
                f"It is {d * 100:.1f} cm from a surface. Move it at least {options.surface_error_distance * 100:.0f} cm away.",
                position=pos,
            )
        elif options.ray_tracing and meta.kind == "receiver" and d < options.receiver_radius:
            out.add(
                "warning",
                "receiver_near_surface",
                f"{_label(meta)} is close to a surface",
                f"It is {d:.2f} m from a surface, inside the {options.receiver_radius} m ray-tracing receiver sphere. The late reverberation may be mis-estimated.",
                position=pos,
            )

    # ── Source → receiver paths ──────────────────────────────────────────
    blocked: List[dict] = []
    margin = options.surface_error_distance
    for src_id, rcv_id in pairs:
        si, ri = seed_index.get(src_id), seed_index.get(rcv_id)
        if si is None or ri is None:
            continue
        a, b = seeds[si].position, seeds[ri].position
        crossing = segment_crossings(tris, a, b, margin)
        separated = enclosure[si] != enclosure[ri]
        if len(crossing) == 0 and not separated:
            continue
        seg = [[float(x) for x in a], [float(x) for x in b]]
        blocked.append({"source_id": src_id, "receiver_id": rcv_id, "segment": seg, "face_ids": crossing.tolist(), "separated": separated})
        objs = _object_list(mesh, crossing, limit)
        if separated:
            out.add(
                "error",
                "no_acoustic_path",
                f"No sound path from '{src_id}' to '{rcv_id}'",
                "They are in separate closed volumes. Transmission through walls is not simulated, so the impulse response will be empty.",
                segment=seg,
                face_ids=crossing.tolist(),
                object_ids=objs,
            )
        else:
            out.add(
                "warning",
                "direct_path_blocked",
                f"Direct sound blocked: '{src_id}' → '{rcv_id}'",
                f"{len(objs)} object(s) stand between them. Transmission and diffraction are not simulated, so only reflections around the obstacle reach the receiver.",
                segment=seg,
                face_ids=crossing.tolist(),
                object_ids=objs,
            )

    # ── Leaks and holes ──────────────────────────────────────────────────
    loop_leaking = attribute_leaks(mesh, options.hole_leak_match_distance)
    leak = vis.leak_fraction
    if leak >= options.leak_warning_fraction:
        severity = "error" if leak >= options.leak_error_fraction else "warning"
        out.add(
            severity,
            "sound_leak",
            f"{leak:.1%} of reflected sound escapes the model",
            "Rays leave through openings: the reverberation will be too short. Close the room or assign materials to the missing surfaces.",
        )
    for k, (loop, leaking) in enumerate(zip(mesh.loops, loop_leaking)):
        if leaking:
            out.add(
                "error",
                "leaking_hole",
                f"Hole leaks sound ({loop.extent:.2f} m)",
                f"An opening with a {loop.length:.2f} m perimeter lets sound escape.",
                position=[float(x) for x in loop.centroid],
                loop_ids=[k],
            )
    silent = [k for k, leaking in enumerate(loop_leaking) if not leaking]
    if silent:
        out.add("info", "open_edges", f"{len(silent)} open edge loop(s)", "Edges used by a single face. No sound escapes through them (seams, panel edges).", loop_ids=silent)

    # ── Topology / auto-fixes ────────────────────────────────────────────
    st = mesh.stats
    if st["non_manifold_edges"]:
        out.add(
            "info",
            "non_manifold",
            f"{st['non_manifold_edges']} non-manifold edge(s)",
            "Edges shared by three or more faces (T-junctions, touching solids). Kept as is: no geometry is removed.",
        )
    n2 = int(mesh.two_sided.sum())
    if n2:
        ids = np.flatnonzero(mesh.two_sided).tolist()
        out.add(
            "info",
            "two_sided_surfaces",
            f"Auto-fixed: {n2} face(s) made double-sided",
            "Thin surfaces with air on both sides reflect sound on both faces.",
            face_ids=ids,
            object_ids=_object_list(mesh, ids, limit),
        )
    if st["flipped_faces"]:
        ids = np.flatnonzero(mesh.flipped).tolist()
        out.add(
            "info",
            "flipped_faces",
            f"Auto-fixed: {st['flipped_faces']} face(s) flipped",
            "Their winding pointed into the air; the reflecting side now faces the room.",
            face_ids=ids,
            object_ids=_object_list(mesh, ids, limit),
        )
    skipped = np.flatnonzero(mesh.face_class == FACE_CLASS_SKIPPED).tolist()
    if skipped:
        unreached = mesh.unreached_objects()
        detail = "Never reached by sound (inside solids or behind the shell). They are kept as reflecting walls, left out of the surface area, the absorption and the diffraction."
        if unreached:
            detail += f" {len(unreached)} object(s) have no face reached at all: leaving them out of the simulation makes the image-source search cheaper."
        out.add(
            "info",
            "skipped_faces",
            f"{len(skipped)} skipped face(s)",
            detail,
            face_ids=skipped,
            object_ids=unreached,
        )
    for obj in mesh.enclosing_objects:
        out.add(
            "info",
            "closed_not_welded",
            f"'{mesh.objects[obj]}' is closed but not welded",
            "Its faces do not share vertices. It was treated as a closed solid; raise the weld tolerance if the gaps are modelling noise.",
            object_ids=[mesh.objects[obj]],
        )

    # ── Materials, cost, units ───────────────────────────────────────────
    if st["dropped_unassigned_objects"]:
        objs = st["dropped_unassigned_objects"]
        out.add(
            "warning",
            "unassigned_material",
            f"{len(objs)} object(s) without material are excluded",
            "Objects with no material are not part of the simulation. If they close the room, sound will leak through the gap.",
            object_ids=objs[:limit],
        )
    ism = 0.0
    if options.image_source_order > 0:
        n_walls = len(mesh.walls) + n2
        ism = float((max(n_walls, 2) / 2.0) ** options.image_source_order)
        if ism >= options.ism_paths_warning:
            out.add(
                "error" if ism >= options.ism_paths_error else "warning",
                "ism_cost",
                f"Image-source search is very large (~{ism:.1e} paths)",
                f"{n_walls} walls at order {options.image_source_order}. Enable coplanar merging, simplify the model or lower the max order (ray tracing covers the late tail).",
            )
    if len(mesh.vertices):
        diag = float(np.linalg.norm(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)))
        if not (options.model_diagonal_min <= diag <= options.model_diagonal_max):
            out.add("warning", "model_units", f"Unusual model size ({diag:.2f} m)", "Check that the model is in metres.")

    return GeometryReport(
        issues=out.items,
        enclosure=enclosure,
        loop_leaking=loop_leaking,
        blocked_paths=blocked,
        seed_surface_distance=dists,
        ism_paths_estimate=ism,
    )


def attribute_leaks(mesh: PreparedMesh, match_distance: float = 0.5) -> List[bool]:
    """Per boundary loop, whether an escaping ray crosses its plane within its extent."""
    vis = mesh.visibility
    flags = [False] * len(mesh.loops)
    if len(vis.leak_origins) == 0:
        return flags
    for k, loop in enumerate(mesh.loops):
        pts = mesh.vertices[loop.vertex_ids]
        if len(pts) < 3:
            continue
        c = loop.centroid
        n = np.linalg.svd(pts - c)[2][-1]
        denom = vis.leak_dirs @ n
        ok = np.abs(denom) > 1e-9
        t = np.where(ok, ((c - vis.leak_origins) @ n) / np.where(ok, denom, 1.0), -1.0)
        p = vis.leak_origins + t[:, None] * vis.leak_dirs
        near = ok & (t > 0) & (np.linalg.norm(p - c, axis=1) <= loop.extent / 2.0 + match_distance)
        flags[k] = bool(near.any())
    return flags
