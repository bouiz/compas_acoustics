"""Diffuse random walk from the "air seeds" (sources + receivers).

Each seed emits rays uniformly; every surface hit re-emits a Lambertian ray
into the half-space the ray came from. Recording which SIDE of each face the
rays arrive on tells us, without any convexity assumption, which side of each
surface touches the air the simulation actually happens in. The same walk
yields: seeds that are outside the model (most primary rays escape), leaks
(rays escaping after a bounce), the set of faces each seed can reach
(enclosures), and the mean free path (air volume via 4V/S).

Ported from compas_soundscape (``backend/utils/air_visibility.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np

from .raycast import TriangleSet
from .raycast import cosine_hemisphere_directions
from .raycast import first_hits
from .raycast import uniform_sphere_directions

SEED_INSIDE = "inside"
SEED_LEAKY = "leaky"
SEED_OUTSIDE = "outside"


@dataclass
class SeedVisibility:
    position: np.ndarray
    escape_fraction: float  # share of primary rays that hit nothing
    status: str  # SEED_INSIDE | SEED_LEAKY | SEED_OUTSIDE
    seen_faces: np.ndarray  # faces reached by this seed's walk
    seen_sides: np.ndarray  # face * 2 + (0 normal side | 1 back side) reached
    nearest_surface_hit_m: float  # closest primary-ray hit distance
    mean_primary_hit_m: float  # mean primary-ray hit distance (enclosure size)


@dataclass
class AirVisibility:
    hits_normal: np.ndarray  # (F,) hits arriving on the side the normal points to
    hits_back: np.ndarray  # (F,) hits arriving on the opposite side
    seeds: list[SeedVisibility]
    leak_origins: np.ndarray  # (L, 3) surface points rays escaped from
    leak_dirs: np.ndarray  # (L, 3) escape directions
    n_reemitted: int = 0  # rays re-emitted from surfaces (bounce >= 1)
    n_escaped: int = 0  # of those, rays that hit nothing
    chord_sum: float = 0.0  # summed bounce >= 1 hit distances
    n_chords: int = 0
    rays_per_seed: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def leak_fraction(self) -> float:
        return self.n_escaped / self.n_reemitted if self.n_reemitted else 0.0

    @property
    def mean_free_path(self) -> float:
        return self.chord_sum / self.n_chords if self.n_chords else 0.0


def seed_enclosures(vis: AirVisibility) -> list[int]:
    """The air volume (enclosure) of each seed, labelled 0, 1, ...: seeds reaching a common (face, side) share one.

    A face side is therefore reached from one enclosure at most.
    """
    sides = [set(s.seen_sides.tolist()) for s in vis.seeds]
    parent = list(range(len(sides)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(sides)):
        for j in range(i + 1, len(sides)):
            if sides[i] & sides[j]:
                parent[find(i)] = find(j)
    roots = {r: k for k, r in enumerate(sorted({find(i) for i in range(len(sides))}))}
    return [roots[find(i)] for i in range(len(sides))]


def side_enclosures(vis: AirVisibility, n_faces: int) -> np.ndarray:
    """The enclosure (:func:`seed_enclosures`) reaching each face side, shape (n_faces, 2): column 0 the side the
    walk's normal points to, column 1 the other; -1 where no seed reaches it."""
    out = np.full(2 * n_faces, -1, dtype=np.int64)
    for seed, label in zip(vis.seeds, seed_enclosures(vis)):
        out[np.asarray(seed.seen_sides, dtype=np.int64)] = label
    return out.reshape(n_faces, 2)


def _seed_status(escape: float, outside_frac: float, leaky_frac: float) -> str:
    if escape >= outside_frac:
        return SEED_OUTSIDE
    if escape >= leaky_frac:
        return SEED_LEAKY
    return SEED_INSIDE


def walk_from_seeds(
    tris: TriangleSet,
    normals: np.ndarray,
    seeds: np.ndarray,
    rays_per_seed: int,
    bounces: int,
    eps: float,
    rng: np.random.Generator,
    outside_frac: float,
    leaky_frac: float,
    max_leaks: int,
) -> AirVisibility:
    """
    Run the diffuse walk from every seed.

    Seeds classified as outside only contribute their primary-ray statistics:
    their rays see the exterior of the model, which must not vote on which
    side of a wall faces the room.
    """
    n_faces = len(tris)
    vis = AirVisibility(
        hits_normal=np.zeros(n_faces, dtype=np.int64),
        hits_back=np.zeros(n_faces, dtype=np.int64),
        seeds=[],
        leak_origins=np.zeros((0, 3)),
        leak_dirs=np.zeros((0, 3)),
        rays_per_seed=rays_per_seed,
    )
    leak_o: list[np.ndarray] = []
    leak_d: list[np.ndarray] = []

    for seed in np.asarray(seeds, dtype=np.float64).reshape(-1, 3):
        d = uniform_sphere_directions(rays_per_seed, rng)
        o = np.repeat(seed[None, :], rays_per_seed, axis=0)
        t, f = first_hits(tris, o, d, tmin=0.0)
        hit = f >= 0
        escape = 1.0 - float(hit.mean()) if rays_per_seed else 1.0
        status = _seed_status(escape, outside_frac, leaky_frac)
        seen = np.zeros(2 * n_faces, dtype=bool)
        primary_cos = np.einsum("ij,ij->i", d[hit], normals[f[hit]])
        seen[2 * f[hit] + (primary_cos >= 0)] = True
        nearest = float(t[hit].min()) if hit.any() else float("inf")
        mean_hit = float(t[hit].mean()) if hit.any() else float("inf")

        if status != SEED_OUTSIDE:
            for bounce in range(bounces + 1):
                hit = f >= 0
                if bounce > 0:
                    vis.n_reemitted += len(f)
                    vis.n_escaped += int((~hit).sum())
                    vis.chord_sum += float(t[hit].sum())
                    vis.n_chords += int(hit.sum())
                    if (~hit).any():
                        leak_o.append(o[~hit])
                        leak_d.append(d[~hit])
                if not hit.any():
                    break
                o, d, t, f = o[hit], d[hit], t[hit], f[hit]
                cos = np.einsum("ij,ij->i", d, normals[f])
                on_normal_side = cos < 0
                np.add.at(vis.hits_normal, f[on_normal_side], 1)
                np.add.at(vis.hits_back, f[~on_normal_side], 1)
                seen[2 * f + (~on_normal_side)] = True
                if bounce == bounces:
                    break
                # Re-emit into the half-space the ray arrived from.
                side = np.where(on_normal_side[:, None], normals[f], -normals[f])
                o = o + t[:, None] * d + eps * side
                d = cosine_hemisphere_directions(side, rng)
                t, f = first_hits(tris, o, d, tmin=0.0)

        vis.seeds.append(
            SeedVisibility(
                position=seed,
                escape_fraction=escape,
                status=status,
                seen_faces=np.unique(np.flatnonzero(seen) // 2),
                seen_sides=np.flatnonzero(seen),
                nearest_surface_hit_m=nearest,
                mean_primary_hit_m=mean_hit,
            )
        )

    if leak_o:
        lo = np.concatenate(leak_o)
        ld = np.concatenate(leak_d)
        if len(lo) > max_leaks:
            pick = rng.choice(len(lo), size=max_leaks, replace=False)
            lo, ld = lo[pick], ld[pick]
        vis.leak_origins, vis.leak_dirs = lo, ld
    return vis


def rays_for_budget(n_faces: int, n_seeds: int, bounces: int, budget: float, lo: int, hi: int, quality: float) -> int:
    """Rays per seed so that rays x faces x bounces stays within ``budget``."""
    if n_faces == 0 or n_seeds == 0:
        return 0
    per_seed = budget / (n_faces * (bounces + 1) * n_seeds)
    return int(np.clip(per_seed * quality, lo, hi * quality))
