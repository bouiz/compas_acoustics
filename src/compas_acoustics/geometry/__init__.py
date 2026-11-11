"""Mesh preparation for simulation, with numpy and scipy only (numba speeds the ray casting up when installed).

:func:`prepare_mesh` turns a raw triangle mesh (open, non-manifold, unwelded, wound any way) into a
:class:`PreparedMesh`: welded, oriented from the side the air is on, thin surfaces made two-sided, faces classified,
coplanar triangles merged into walls, with the air volume. :func:`analyze` reports what will go wrong.
:meth:`compas_acoustics.RoomGeometry.from_prepared` builds the geometric methods' room from it; a wave solver
reads :attr:`PreparedMesh.faces` and :attr:`PreparedMesh.two_sided` directly.

Importing this package loads neither pyroomacoustics nor COMPAS.
"""

from .coplanar import MergedWall
from .prepare import FACE_CLASS_INTERIOR
from .prepare import FACE_CLASS_SHELL
from .prepare import FACE_CLASS_SKIPPED
from .prepare import VISIBILITY_QUALITY
from .prepare import PreparedMesh
from .prepare import PrepareOptions
from .prepare import material_key
from .prepare import prepare_mesh
from .report import GeometryIssue
from .report import GeometryReport
from .report import Issues
from .report import ReportOptions
from .report import Seed
from .report import analyze
from .report import summarize
from .visibility import SEED_INSIDE
from .visibility import SEED_LEAKY
from .visibility import SEED_OUTSIDE

__all__ = [
    "FACE_CLASS_INTERIOR",
    "FACE_CLASS_SHELL",
    "FACE_CLASS_SKIPPED",
    "SEED_INSIDE",
    "SEED_LEAKY",
    "SEED_OUTSIDE",
    "VISIBILITY_QUALITY",
    "GeometryIssue",
    "GeometryReport",
    "Issues",
    "MergedWall",
    "PreparedMesh",
    "PrepareOptions",
    "ReportOptions",
    "Seed",
    "analyze",
    "material_key",
    "prepare_mesh",
    "summarize",
]
