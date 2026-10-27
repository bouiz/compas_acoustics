"""Rays for display: traced rays and ray fields (:meth:`RayField.from_scene`), and the specular
image-source paths between sources and receivers (:meth:`SpecularReflections.from_scene`), a ray field too."""

import numbers
import warnings
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from compas.data import Data
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import Vector
from compas.tolerance import TOL

from compas_acoustics.defaults import Defaults
from compas_acoustics.defaults import _PackageDtype

__all__ = ['Ray', 'ScatteredRay', 'RayField', 'SpecularPath', 'SpecularReflections']


def _distance(time_ms: float) -> float:
    """The distance sound travels in ``time_ms`` at ``Defaults().DEFAULT_SPEED_OF_SOUND``."""
    return Defaults().DEFAULT_SPEED_OF_SOUND * time_ms / 1000.0


def _inside(geometry, items, message: str) -> list:
    """The ``(index, item)`` pairs whose item is inside the room, a ``UserWarning`` ``message`` for each other."""
    inside = []
    for index, item in items:
        if geometry.is_inside(item.position):
            inside.append((index, item))
        else:
            warnings.warn(message.format(name=item.name), UserWarning, stacklevel=3)
    return inside


class Ray(_PackageDtype, Polyline):
    """A ray traced through a scene, the polyline ``[source_position, hit_1, hit_2, ...]``.

    Times are distances at ``Defaults().DEFAULT_SPEED_OF_SOUND``.

    Parameters
    ----------
    points : list
    source : :class:`compas_acoustics.Source`, optional
    source_index : int, optional
        The source's index in the scene.
    ray_index : int, optional
        The ray's index among those of its source.
    direction : :class:`compas.geometry.Vector` or list, optional
        The initial direction; default that of the first segment.
    distance_offset : float, optional
        The distance travelled before ``points[0]``: non-zero for a scattered ray born at a hit.
    name : str, optional
    """

    def __init__(
        self,
        points,
        source=None,
        source_index: Optional[int] = None,
        ray_index: Optional[int] = None,
        direction=None,
        distance_offset: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(points, name=name)
        self.source = source
        self.source_index = source_index
        self.ray_index = ray_index
        self.distance_offset = distance_offset

        if direction is not None:
            self.direction = direction
        elif len(self.points) >= 2:
            self.direction = Vector.from_start_end(self.points[0], self.points[1])
        else:
            self.direction = Vector(1, 0, 0)

    @property
    def __data__(self):
        # The source is not stored: a RayField keeps each source once, a lone ray keeps source_index
        data = super().__data__
        data.update(
            source_index=self.source_index,
            ray_index=self.ray_index,
            direction=list(self.direction),
            distance_offset=self.distance_offset,
        )
        scattered = getattr(self, 'scattered_reflections', None)
        if scattered:
            data["scattered_reflections"] = scattered
        return data

    @classmethod
    def __from_data__(cls, data):
        data = dict(data)
        data.pop("sound_speed", None)  # stored before the speed of sound came from Defaults
        scattered = data.pop("scattered_reflections", None)
        ray = cls(**data)
        if scattered:
            ray.scattered_reflections = list(scattered)
        return ray

    @property
    def direction(self) -> Vector:
        return self._direction

    @direction.setter
    def direction(self, value):
        vector = Vector(*value)
        if vector.length:
            vector.unitize()
        self._direction = vector

    @property
    def segment_lengths(self) -> List[float]:
        """The length of each segment of the ray."""
        return [self.points[i].distance_to_point(self.points[i + 1]) for i in range(len(self.points) - 1)]

    @property
    def cumulative_distances(self) -> List[float]:
        """The travelled distance from the ray origin at each vertex, including ``distance_offset``."""
        distances = [self.distance_offset]
        accumulated = self.distance_offset
        for segment in self.segment_lengths:
            accumulated += segment
            distances.append(accumulated)
        return distances

    @property
    def total_distance(self) -> float:
        """The total distance travelled by the ray, including ``distance_offset``."""
        return self.distance_offset + sum(self.segment_lengths)

    @property
    def travel_time(self) -> float:
        """The total travel time of the ray in milliseconds."""
        return self.total_distance / Defaults().DEFAULT_SPEED_OF_SOUND * 1000.0

    @property
    def reflection_order(self) -> int:
        """The number of segments (hops) of the ray."""
        return max(len(self.points) - 1, 0)

    def is_in_cone(self, axis, half_angle_deg: float) -> bool:
        """Whether the ray's direction is within ``half_angle_deg`` of ``axis``, a cone with its apex at the origin."""
        axis_vector = Vector(*axis)
        if not axis_vector.length or not self.direction.length:
            return False
        return self.direction.angle(axis_vector, degrees=True) <= half_angle_deg + 1e-9

    def _clone_with_points(self, points) -> "Ray":
        return Ray(
            points,
            source=self.source,
            source_index=self.source_index,
            ray_index=self.ray_index,
            direction=self.direction,
            distance_offset=self.distance_offset,
            name=self.name,
        )

    def up_to_reflection_order(self, order: Optional[int]) -> "Ray":
        """A copy of the ray with its first ``order`` segments; None: all."""
        if order is None:
            return self._clone_with_points(self.points[:])
        order = max(int(order), 0)
        points = self.points[: order + 1]
        if len(points) < 1:
            points = self.points[:1]
        return self._clone_with_points(points)

    def up_to_distance(self, distance: Optional[float]) -> Optional["Ray"]:
        """A copy of the ray cut at ``distance`` from the source (``distance_offset`` included); None: all.

        Returns None if the ray has not emerged yet at ``distance``.
        """
        if distance is None:
            return self._clone_with_points(self.points[:])

        cumulative = self.cumulative_distances
        if distance <= cumulative[0]:
            return None
        if distance >= cumulative[-1]:
            return self._clone_with_points(self.points[:])

        points = [self.points[0]]
        for i in range(1, len(self.points)):
            if cumulative[i] <= distance:
                points.append(self.points[i])
                continue
            segment = cumulative[i] - cumulative[i - 1]
            t = (distance - cumulative[i - 1]) / segment if segment else 0.0
            previous = self.points[i - 1]
            current = self.points[i]
            points.append(
                Point(
                    previous[0] + t * (current[0] - previous[0]),
                    previous[1] + t * (current[1] - previous[1]),
                    previous[2] + t * (current[2] - previous[2]),
                )
            )
            break
        return self._clone_with_points(points)

    def up_to_time(self, time_ms: Optional[float]) -> Optional["Ray"]:
        """A copy of the ray cut at ``time_ms`` milliseconds, as :meth:`up_to_distance`."""
        if time_ms is None:
            return self._clone_with_points(self.points[:])
        return self.up_to_distance(_distance(time_ms))


class ScatteredRay(Ray):
    """A scattered ray born at a reflection point of a specular ray.

    Parameters
    ----------
    points : list
        From the reflection point it scattered from.
    parent_ray_index : int, optional
    energy : float, optional
        The scattered energy it carries.
    origin_reflection_order : int, optional
        The parent's segments up to that reflection point.
    **kwargs
        Of :class:`compas_acoustics.Ray`.
    """

    def __init__(
        self,
        points,
        parent_ray_index: Optional[int] = None,
        energy: Optional[float] = None,
        origin_reflection_order: int = 0,
        **kwargs,
    ):
        super().__init__(points, **kwargs)
        self.parent_ray_index = parent_ray_index
        self.energy = energy
        self._origin_reflection_order = origin_reflection_order

    @property
    def __data__(self):
        data = super().__data__
        data.update(parent_ray_index=self.parent_ray_index, energy=self.energy, origin_reflection_order=self._origin_reflection_order)
        return data

    @property
    def reflection_order(self) -> int:
        """The specular reflection order of the originating hit point."""
        return self._origin_reflection_order

    def _clone_with_points(self, points) -> "ScatteredRay":
        return ScatteredRay(
            points,
            parent_ray_index=self.parent_ray_index,
            energy=self.energy,
            origin_reflection_order=self._origin_reflection_order,
            source=self.source,
            source_index=self.source_index,
            ray_index=self.ray_index,
            direction=self.direction,
            distance_offset=self.distance_offset,
            name=self.name,
        )


class RayField(_PackageDtype, Data):
    """Traced rays, with filters and display.

    Parameters
    ----------
    rays : list of :class:`compas_acoustics.Ray`, optional
        The specular rays, each with its scattered rays in ``ray.scattered_reflections``.
    name : str, optional
    """

    def __init__(self, rays: Optional[List[Ray]] = None, name: Optional[str] = None):
        super().__init__(name=name)
        self.rays = list(rays) if rays else []

    @property
    def __data__(self):
        # Each source once, by its index, instead of once per ray
        sources = {ray.source_index: ray.source for ray in self.rays if ray.source is not None and ray.source_index is not None}
        return {"rays": self.rays, "sources": [[index, source] for index, source in sorted(sources.items())]}

    @classmethod
    def __from_data__(cls, data):
        sources = {int(index): source for index, source in data.get("sources", [])}
        field = cls(data["rays"])
        for ray in field.rays:
            for traced in [ray, *getattr(ray, 'scattered_reflections', [])]:
                traced.source = sources.get(traced.source_index)
        return field

    def __len__(self) -> int:
        return len(self.rays)

    def __iter__(self):
        return iter(self.rays)

    @classmethod
    def from_scene(
        cls,
        scene,
        n_rays: int = 500,
        max_reflection_order: int = 10,
        direction=None,
        angle: Optional[float] = None,
        duration: Optional[float] = None,
        energy_thres: Optional[float] = None,
        sources=None,
        include_scattering: bool = False,
        scatter_density: int = 10,
        scatter_max_bounces: int = 1,
        scatter_band: Optional[int] = None,
        random_seed: Union[bool, int] = True,
    ):
        """Trace rays through the scene for display.

        Each source emits with the density of its radiated energy, the square of its first-order pattern
        ``p + (1 - p) cos(theta)`` about its ``direction``; with a cone (``direction`` and ``angle``) all
        ``n_rays`` go into the cone, with the same density. With ``include_scattering`` each hit spawns
        scattered rays, their number and spread set by the scattering coefficient.

        Parameters
        ----------
        scene : :class:`compas_acoustics.Scene`
        n_rays : int, default=500
            Per source.
        max_reflection_order : int, default=10
        direction : :class:`compas.geometry.Vector` or list, optional
            The axis of a cone to emit into, apex at each source. Requires ``angle``.
        angle : float, optional
            The cone's half angle in degrees. Requires ``direction``.
        duration : float, optional
            In seconds: a ray stops at the first reflection past the distance sound travels in it.
        energy_thres : float, optional
            A ray stops when the energy its surfaces left it falls below this.
        sources : list of :class:`compas_acoustics.Source`, optional
            Default: all of the scene's. ``source_index`` is the index in ``scene.sources``.
        include_scattering : bool, default=False
        scatter_density : int, default=10
            Scattered rays per hit at scattering coefficient 1.
        scatter_max_bounces : int, default=1
            Reflections traced for each scattered ray.
        scatter_band : int, optional
            The band whose scattering coefficient is used; default the mean of all.
        random_seed : bool or int, default=True
            True seeds with 42, an int with that value, False or None leaves the scattering unseeded.

        Returns
        -------
        :class:`compas_acoustics.RayField`
        """
        if (direction is None) != (angle is None):
            raise ValueError("Both 'direction' and 'angle' must be provided together to emit rays into a cone.")

        from compas_acoustics.room.geometry import RoomGeometry
        from compas_acoustics.simulation_methods.base import _seed

        geometry = RoomGeometry.from_scene(scene)
        simulation = geometry.to_pra(Defaults().SAMPLING_RATE, max_reflection_order, ray_tracing=False, air_absorption=False)
        engine = simulation.room_engine
        walls = engine.walls
        max_dist = float(engine.max_dist)

        if sources is None:
            sources = scene.sources

        # A ray keeps its source's index in the scene, whichever sources are traced
        scene_index = {id(source): index for index, source in enumerate(scene.sources)}
        sim_sources = _inside(
            geometry,
            [(scene_index.get(id(source), position), source) for position, source in enumerate(sources)],
            "Source '{name}' is outside the room and no rays will be traced from it.",
        )

        rng = np.random.default_rng(_seed(random_seed))
        cone = direction

        distance_thres = _distance(duration * 1000.0) if duration is not None else None

        from compas_acoustics.worker import jobs

        rays = []
        for n, (source_index, source) in enumerate(sim_sources):
            jobs.progress("tracing", n, len(sim_sources))
            source_position = np.asarray(source.position, dtype=float)
            directions = _source_directions(source, cone, angle, n_rays)

            for ray_index in range(directions.shape[0]):
                initial_direction = directions[ray_index] / np.linalg.norm(directions[ray_index])
                direction = initial_direction.copy()
                start = source_position.copy()
                points = [Point(*start)]
                scattered_reflections = []
                transmitted = 1.0
                travel_distance = 0.0

                for bounce in range(max_reflection_order):
                    hit, wall_index, hit_distance = engine.next_wall_hit(start, start + direction * max_dist, False)
                    if wall_index == -1:
                        break

                    hit = np.asarray(hit, dtype=float)
                    points.append(Point(*hit))
                    travel_distance += float(hit_distance)
                    wall = walls[wall_index]

                    absorption = float(np.mean(np.asarray(wall.absorption, dtype=float)))
                    transmitted *= max(1.0 - absorption, 0.0)

                    if include_scattering:
                        scatter = _scatter_coefficient(wall, scatter_band)
                        if scatter > 0.0:
                            scattered_reflections.extend(
                                _make_scattered_rays(
                                    engine=engine,
                                    walls=walls,
                                    hit=hit,
                                    incident_direction=direction,
                                    wall=wall,
                                    scatter=scatter,
                                    transmitted=transmitted,
                                    travel_distance=travel_distance,
                                    origin_reflection_order=bounce + 1,
                                    source=source,
                                    source_index=source_index,
                                    ray_index=ray_index,
                                    max_dist=max_dist,
                                    scatter_density=scatter_density,
                                    scatter_max_bounces=scatter_max_bounces,
                                    rng=rng,
                                )
                            )
                            transmitted *= max(1.0 - scatter, 0.0)

                    if distance_thres is not None and travel_distance > distance_thres:
                        break
                    if energy_thres is not None and transmitted < energy_thres:
                        break

                    direction = np.asarray(wall.normal_reflect(direction), dtype=float)
                    start = hit

                ray = Ray(
                    points,
                    source=source,
                    source_index=source_index,
                    ray_index=ray_index,
                    direction=Vector(*initial_direction),
                )
                ray.scattered_reflections = scattered_reflections
                rays.append(ray)

        return cls(rays)

    @property
    def scattered_rays(self) -> List[ScatteredRay]:
        """All scattered rays across every ray in the field."""
        scattered = []
        for ray in self.rays:
            scattered.extend(getattr(ray, 'scattered_reflections', []))
        return scattered

    @property
    def duration_ms(self) -> float:
        """The largest travel time of the specular rays in milliseconds, 0.0 when empty."""
        return max((ray.travel_time for ray in self.rays), default=0.0)

    def particles(self, time_ms: float):
        """The position of every specular ray at ``time_ms`` (clamped to ``[0, duration_ms]``), for a billiard view.

        Returns
        -------
        tuple of (list of :class:`compas.geometry.Point`, list of int, list of int)
            The positions, their rays' ``ray_index`` and the reflections each has passed. A ray that
            has ended gives no particle.
        """
        time_ms = min(max(time_ms, 0.0), self.duration_ms)
        points = []
        indices = []
        orders = []
        distance = _distance(time_ms)
        for ray in self.rays:
            if ray.travel_time < time_ms:
                continue
            part = ray.up_to_distance(distance)
            if part is None:
                # up_to_distance reports "not emerged" at distance == distance_offset too
                if distance < ray.cumulative_distances[0]:
                    continue
                points.append(Point(*ray.points[0]))
            else:
                points.append(Point(*part.points[-1]))
            indices.append(ray.ray_index)
            # A particle on a wall is reflected only once it moves on, so the end of a ray is no reflection
            orders.append(sum(1 for reached in ray.cumulative_distances[1:] if reached < distance - TOL.absolute))
        return points, indices, orders

    def filter(
        self,
        source=None,
        ray_index: Optional[Union[int, List[int]]] = None,
    ) -> "RayField":
        """A new field of the rays of ``source`` (object or index) and of ``ray_index`` (one or several)."""
        ray_indices = None
        if ray_index is not None:
            ray_indices = {ray_index} if isinstance(ray_index, numbers.Integral) else set(ray_index)

        kept = []
        for ray in self.rays:
            if source is not None and not self._matches_source(ray, source):
                continue
            if ray_indices is not None and ray.ray_index not in ray_indices:
                continue
            kept.append(ray)

        return type(self)(kept, name=self.name)

    @staticmethod
    def _matches_source(ray: Ray, source) -> bool:
        if isinstance(source, numbers.Integral):
            return ray.source_index == source
        return ray.source is source

    def truncated(self, reflection_order: Optional[int] = None, travel_time_ms: Optional[float] = None) -> "RayField":
        """A new field of the rays cut to their first ``reflection_order`` segments and to ``travel_time_ms``."""
        new_rays = []
        for ray in self.rays:
            truncated_ray = self._truncate_ray(ray, reflection_order, travel_time_ms)
            if truncated_ray is None:
                continue

            scattered = []
            for scattered_ray in getattr(ray, 'scattered_reflections', []):
                truncated_scattered = self._truncate_scattered(scattered_ray, reflection_order, travel_time_ms)
                if truncated_scattered is not None:
                    scattered.append(truncated_scattered)
            truncated_ray.scattered_reflections = scattered

            new_rays.append(truncated_ray)
        # Truncated rays are plain rays, whatever the field held
        return RayField(new_rays, name=self.name)

    @staticmethod
    def _truncate_ray(ray: Ray, reflection_order: Optional[int], travel_time_ms: Optional[float]) -> Optional[Ray]:
        result = ray
        if reflection_order is not None:
            result = result.up_to_reflection_order(reflection_order)
        if travel_time_ms is not None:
            result = result.up_to_time(travel_time_ms)
        return result

    @staticmethod
    def _truncate_scattered(ray: ScatteredRay, reflection_order: Optional[int], travel_time_ms: Optional[float]) -> Optional[ScatteredRay]:
        result = ray
        if reflection_order is not None:
            budget = reflection_order - ray.reflection_order
            if budget < 1:
                return None
            result = result.up_to_reflection_order(budget)
        if travel_time_ms is not None:
            result = result.up_to_time(travel_time_ms)
        return result

    def display_in_viewer(
        self,
        scene=None,
        reflection_order: Optional[int] = None,
        travel_time_ms: Optional[float] = None,
        show_specular: bool = True,
        show_scattered: bool = False,
        color=None,
        scattered_color=None,
        linewidth: float = 1.0,
        show: bool = True,
    ):
        """Display the rays, and the scene's geometry, sources and receivers when given, in a viewer.

        ``reflection_order`` and ``travel_time_ms`` are those of :meth:`truncated`. With ``show=False``
        the viewer is returned without being opened.
        """
        import compas

        if compas.is_rhino():
            raise RuntimeError("Displaying rays in the viewer is not supported in Rhino.")

        from compas.colors import Color
        from compas_viewer import Viewer

        if color is None:
            color = Color.blue()
        if scattered_color is None:
            scattered_color = Color.red()

        viewer = Viewer()

        if scene is not None:
            self._add_scene_to_viewer(viewer, scene)

        field = self.truncated(reflection_order=reflection_order, travel_time_ms=travel_time_ms)

        if show_specular:
            for ray in field.rays:
                if len(ray.points) < 2:
                    continue
                viewer.scene.add(ray, name=f"ray_{ray.ray_index}", linecolor=color, linewidth=linewidth)

        if show_scattered:
            for scattered_ray in field.scattered_rays:
                if len(scattered_ray.points) < 2:
                    continue
                name = f"scattered_{scattered_ray.parent_ray_index}"
                viewer.scene.add(scattered_ray, name=name, linecolor=scattered_color, linewidth=linewidth)

        if show:
            viewer.show()

        return viewer

    @staticmethod
    def _add_scene_to_viewer(viewer, scene) -> None:
        from compas.colors import Color

        for acoustic_object in scene.acoustic_objects:
            viewer.scene.add(acoustic_object.acoustic_geometry, name=acoustic_object.name, opacity=0.5)

        for source in scene.sources:
            viewer.scene.add(Point(*source.position), name=source.name, pointcolor=Color.green(), pointsize=10)

        for receiver in scene.receivers:
            viewer.scene.add(Point(*receiver.position), name=receiver.name, pointcolor=Color.red(), pointsize=10)


def _basis(axis, label):
    """The unit ``axis`` and two unit tangents completing it to an orthonormal basis."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if TOL.is_zero(norm):
        raise ValueError(f"{label} must be a non-zero vector.")
    axis = axis / norm
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent_1 = np.cross(axis, helper)
    tangent_1 /= np.linalg.norm(tangent_1)
    return axis, tangent_1, np.cross(axis, tangent_1)


def _cone_sampling(direction, angle: float, n_rays: int):
    """``n_rays`` unit directions (n_rays, 3) on the fibonacci lattice of the cap of half angle ``angle`` degrees about ``direction``."""
    cosines = 1.0 - (np.arange(n_rays) + 0.5) / n_rays * (1.0 - np.cos(np.radians(angle)))
    return _spiral(direction, cosines, "Cone 'direction'")


def _source_directions(source, cone, angle: Optional[float], n_rays: int):
    """The ``n_rays`` unit directions a source emits, shape ``(n_rays, 3)``: the lattice of the cone, if given, or
    of the sphere, with the density of the energy of the source's pattern about its facing."""
    import pyroomacoustics as pra

    from compas_acoustics.room.pra import CARDIOID_P

    p = CARDIOID_P.get(source.directivity or "omnidirectional", 1.0)
    if cone is None:
        if p == 1.0:
            directions = np.asarray(pra.fibonacci_spherical_sampling(n_rays)).T
            return directions / np.linalg.norm(directions, axis=1)[:, None]
        return _pattern_sampling(source.direction, p, n_rays)
    if p == 1.0:
        return _cone_sampling(cone, angle, n_rays)
    # the cone's lattice, 64 candidates per ray, thinned to the pattern
    candidates = _cone_sampling(cone, angle, n_rays * 64)
    facing = np.asarray(source.direction, dtype=float)
    weights = (p + (1.0 - p) * candidates @ (facing / np.linalg.norm(facing))) ** 2
    if weights.sum() <= 0.0:
        return _cone_sampling(cone, angle, n_rays)
    # systematic resampling along the lattice, which runs from the axis outwards
    cumulative = np.cumsum(weights) / weights.sum()
    picks = np.searchsorted(cumulative, (np.arange(n_rays) + 0.5) / n_rays)
    return candidates[np.minimum(picks, len(candidates) - 1)]


def _pattern_sampling(axis, p: float, n_rays: int):
    """``n_rays`` unit directions, shape ``(n_rays, 3)``, with the density (p + (1 - p) cos(theta))^2 about ``axis``.

    The fibonacci lattice of the sphere with its cosines through the inverse of the pattern's distribution: the
    energy within cos(theta) < u is ((p + (1 - p) u)^3 - (2p - 1)^3) / (1 - (2p - 1)^3), for p < 1.
    """
    q = (2.0 * p - 1.0) ** 3
    share = 1.0 - (np.arange(n_rays) + 0.5) / n_rays
    cosines = np.clip((np.cbrt(q + share * (1.0 - q)) - p) / (1.0 - p), -1.0, 1.0)
    return _spiral(axis, cosines, "A source's 'direction'")


def _spiral(axis, cosines, label):
    """Unit directions at ``cosines`` to ``axis``, turned by the golden angle one after the other."""
    axis, tangent_1, tangent_2 = _basis(axis, label)
    phi = np.arange(len(cosines)) * np.pi * (3.0 - np.sqrt(5.0))
    radius = np.sqrt(np.maximum(1.0 - cosines**2, 0.0))
    directions = (radius * np.cos(phi))[:, None] * tangent_1 + (radius * np.sin(phi))[:, None] * tangent_2 + cosines[:, None] * axis
    return directions / np.linalg.norm(directions, axis=1)[:, None]


def _scatter_coefficient(wall, scatter_band: Optional[int]) -> float:
    coefficients = np.asarray(wall.scatter, dtype=float)
    if coefficients.size == 0:
        return 0.0
    value = float(coefficients[scatter_band]) if scatter_band is not None else float(np.mean(coefficients))
    return min(max(value, 0.0), 1.0)


def _make_scattered_rays(
    engine,
    walls,
    hit,
    incident_direction,
    wall,
    scatter: float,
    transmitted: float,
    travel_distance: float,
    origin_reflection_order: int,
    source,
    source_index: int,
    ray_index: int,
    max_dist: float,
    scatter_density: int,
    scatter_max_bounces: int,
    rng,
):
    n_scatter = int(round(scatter * scatter_density))
    if n_scatter <= 0:
        return []

    normal = np.asarray(wall.normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    if np.dot(normal, incident_direction) > 0:
        normal = -normal

    specular = np.asarray(wall.normal_reflect(incident_direction), dtype=float)
    specular = specular / np.linalg.norm(specular)

    axis = (1.0 - scatter) * specular + scatter * normal
    axis_norm = np.linalg.norm(axis)
    axis = normal if TOL.is_zero(axis_norm) else axis / axis_norm

    theta_max = np.radians(scatter * 90.0)
    directions = _sample_cone_directions(axis, normal, n_scatter, theta_max, rng)

    energy = transmitted * scatter / n_scatter
    scattered_rays = []
    for scatter_direction in directions:
        points = [Point(*hit)]
        start = np.asarray(hit, dtype=float)
        direction = scatter_direction
        for _ in range(max(scatter_max_bounces, 1)):
            sub_hit, wall_index, _ = engine.next_wall_hit(start, start + direction * max_dist, False)
            if wall_index == -1:
                break
            sub_hit = np.asarray(sub_hit, dtype=float)
            points.append(Point(*sub_hit))
            start = sub_hit
            direction = np.asarray(walls[wall_index].normal_reflect(direction), dtype=float)

        if len(points) < 2:
            continue

        scattered_rays.append(
            ScatteredRay(
                points,
                parent_ray_index=ray_index,
                energy=energy,
                origin_reflection_order=origin_reflection_order,
                source=source,
                source_index=source_index,
                ray_index=ray_index,
                direction=Vector(*scatter_direction),
                distance_offset=travel_distance,
            )
        )
    return scattered_rays


def _sample_cone_directions(axis, normal, n_directions: int, theta_max: float, rng):
    """Cosine-weighted random directions within ``theta_max`` of ``axis``, on the front of ``normal``."""
    axis, tangent_1, tangent_2 = _basis(axis, "The scattering axis")

    sin_max = np.sin(theta_max)
    directions = []
    attempts = 0
    max_attempts = max(n_directions * 20, 20)
    while len(directions) < n_directions and attempts < max_attempts:
        attempts += 1
        u1 = rng.random()
        u2 = rng.random()
        radius = sin_max * np.sqrt(u1)
        phi = 2.0 * np.pi * u2
        x = radius * np.cos(phi)
        y = radius * np.sin(phi)
        z = np.sqrt(max(1.0 - x * x - y * y, 0.0))
        direction = x * tangent_1 + y * tangent_2 + z * axis
        direction /= np.linalg.norm(direction)
        if np.dot(direction, normal) <= 1e-6:
            continue
        directions.append(direction)
    return directions


class SpecularPath(Ray):
    """An image-source path ``[source_position, hit_1, ..., hit_order, receiver_position]``.

    Parameters
    ----------
    points : list
    order : int
        The number of reflections.
    walls : list of int
        The wall of each reflection, in path order: indices into :attr:`RoomGeometry.faces` of
        ``RoomGeometry.from_scene(scene)``, obstacle copies included.
    image : :class:`compas.geometry.Point` or list
    receiver_index : int
        The receiver's index in the scene.
    **kwargs
        Of :class:`compas_acoustics.Ray`.
    """

    def __init__(self, points, order: int, walls: List[int], image, receiver_index: int, **kwargs):
        super().__init__(points, **kwargs)
        self.order = order
        self.walls = list(walls)
        self.image = Point(*image)
        self.receiver_index = receiver_index

    @property
    def __data__(self):
        data = super().__data__
        data.update(order=self.order, walls=self.walls, image=list(self.image), receiver_index=self.receiver_index)
        return data

    @classmethod
    def __from_data__(cls, data):
        data = dict(data)
        data.pop("wall_names", None)  # stored by the first version
        return super().__from_data__(data)

    @property
    def reflection_order(self) -> int:
        """The number of walls the path is reflected by: its last segment ends at the receiver, not a wall."""
        return self.order

    def _clone_with_points(self, points) -> Ray:
        # A cut path no longer reaches its receiver: a plain ray
        return Ray(points, source=self.source, source_index=self.source_index, ray_index=self.ray_index,
                   direction=self.direction, name=self.name)


class SpecularReflections(RayField):
    """The image-source paths between sources and receivers: a :class:`RayField` of :class:`SpecularPath`.

    Parameters
    ----------
    paths : list of :class:`SpecularPath`, optional
    name : str, optional
    """

    def __init__(self, paths: Optional[List[SpecularPath]] = None, name: Optional[str] = None):
        super().__init__(paths, name=name)

    @classmethod
    def __from_data__(cls, data):
        data = dict(data)
        if "rays" not in data:  # stored by the first version
            data["rays"] = data.pop("paths")
        return super().__from_data__(data)

    @property
    def paths(self) -> List[SpecularPath]:
        """The reflection paths, the field's rays."""
        return self.rays

    @classmethod
    def from_scene(cls, scene, max_order: int = 3) -> "SpecularReflections":
        """The paths of the visible image sources, orders 0 to ``max_order``, from every source to every receiver.

        Only the last wall of each image is recorded, so each path is traced back from the receiver: through
        that wall to the parent image, and on through whichever wall leads back to the source. An image found
        twice (through an edge, from both wall orderings) is kept once. Sources and receivers outside the room
        are left out with a ``UserWarning``.

        Parameters
        ----------
        scene : :class:`compas_acoustics.Scene`
        max_order : int, default=3

        Returns
        -------
        :class:`SpecularReflections`
        """
        from compas_acoustics.room.geometry import RoomGeometry
        from compas_acoustics.simulation_methods.base import _warn_if_the_search_is_slow

        if max_order < 0:
            raise ValueError(f"max_order must be 0 or more, not {max_order}.")
        geometry = RoomGeometry.from_scene(scene)
        sources = _inside(geometry, enumerate(scene.sources), "Source '{name}' is outside the room and has no specular paths.")
        receivers = _inside(geometry, enumerate(scene.receivers), "Receiver '{name}' is outside the room and has no specular paths.")
        if not sources or not receivers:
            return cls()

        _warn_if_the_search_is_slow(geometry, max_order)
        room = geometry.to_pra(Defaults().SAMPLING_RATE, max_order, ray_tracing=False, air_absorption=False)
        for _, source in sources:
            room.add_source(np.asarray(source.position, dtype=float))
        room.add_microphone_array(np.array([receiver.position for _, receiver in receivers], dtype=float).T)
        room.image_source_model()

        walls = room.room_engine.walls
        tolerance = 1e-5 * max(float(np.ptp(geometry.vertices, axis=0).max()), 1.0)
        paths = []
        untraced = 0
        for s, (source_index, source) in enumerate(sources):
            pra_source = room.sources[s]
            images = np.asarray(pra_source.images, dtype=float).T
            orders = np.asarray(pra_source.orders, dtype=int).ravel()
            last_walls = np.asarray(pra_source.walls, dtype=int).ravel()
            start = np.asarray(source.position, dtype=float)
            for m, (receiver_index, receiver) in enumerate(receivers):
                end = np.asarray(receiver.position, dtype=float)
                kept = []
                failed = []
                for i in np.flatnonzero(np.asarray(room.visibility[s][m], dtype=bool)):
                    # the first of each group at one position, as hybrid.early._first_of_each_position
                    if _near(kept, images[i], tolerance):
                        continue
                    traced = _trace_back(walls, end, images[i], int(last_walls[i]), int(orders[i]), start, tolerance)
                    if traced is None:
                        if not _near(failed, images[i], tolerance):
                            failed.append(images[i])
                        continue
                    kept.append(images[i])
                    hits, path_walls = traced
                    paths.append(SpecularPath(
                        [start, *hits, end],
                        order=int(orders[i]),
                        walls=path_walls,
                        image=images[i],
                        receiver_index=receiver_index,
                        source=source,
                        source_index=source_index,
                        ray_index=len(paths),
                    ))
                # another image of the group may have traced a position whose first image did not
                untraced += sum(1 for image in failed if not _near(kept, image, tolerance))
        if untraced:
            warnings.warn(f"{untraced} visible image sources could not be traced back to their source.", UserWarning, stacklevel=2)
        return cls(paths)

    @property
    def orders(self) -> List[int]:
        """The sorted distinct reflection orders present."""
        return sorted({path.order for path in self.paths})

    def filter(
        self,
        source=None,
        ray_index: Optional[Union[int, List[int]]] = None,
        receiver_index: Optional[int] = None,
        order: Optional[Union[int, List[int]]] = None,
    ) -> "SpecularReflections":
        """As :meth:`RayField.filter`, and the paths to ``receiver_index`` of ``order`` (one or several)."""
        orders = None
        if order is not None:
            orders = {order} if isinstance(order, numbers.Integral) else set(order)
        kept = [
            path
            for path in super().filter(source=source, ray_index=ray_index)
            if (receiver_index is None or path.receiver_index == receiver_index)
            and (orders is None or path.order in orders)
        ]
        return SpecularReflections(kept, name=self.name)


def _near(images: list, image, tolerance: float) -> bool:
    """Whether ``image`` is within ``tolerance`` of any of ``images``."""
    return bool(images) and float(np.min(np.linalg.norm(np.asarray(images) - image, axis=1))) <= tolerance


def _crossing(wall, start, end) -> Optional[np.ndarray]:
    """Where the segment ``start`` to ``end`` crosses ``wall``, by the wall's own polygon test, or None."""
    out = np.zeros(3, dtype=np.float32)
    if wall.intersection(start.astype(np.float32), end.astype(np.float32), out) < 0:
        return None
    normal = np.asarray(wall.normal, dtype=float)
    direction = end - start
    t = normal @ (np.asarray(wall.origin, dtype=float) - start) / (normal @ direction)
    return start + t * direction


def _mirror(wall, point) -> np.ndarray:
    normal = np.asarray(wall.normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    return point + 2.0 * (normal @ (np.asarray(wall.origin, dtype=float) - point)) * normal


def _trace_back(walls, receiver, image, last_wall: int, order: int, source, tolerance: float):
    """The hit points and walls, source side first, of the path from ``source`` to ``receiver`` via ``image``.

    ``image`` is a visible image of ``order`` generated by ``last_wall``; its parent is its mirror image in that
    wall. The parent's generating wall is one the segment from the hit to the parent crosses, with the parent
    behind it; at an edge several do, so each is tried until the mirrors lead back to the source. None if none does.
    """
    if order == 0:
        return ([], []) if np.linalg.norm(image - source) <= tolerance else None

    def back(point, image, wall_index, order):
        hit = _crossing(walls[wall_index], point, image)
        if hit is None:
            return None
        parent = _mirror(walls[wall_index], image)
        if order == 1:
            return ([hit], [wall_index]) if np.linalg.norm(parent - source) <= tolerance else None
        for candidate in range(len(walls)):
            if candidate == wall_index or walls[candidate].side(parent.astype(np.float32)) <= 0:
                continue
            found = back(hit, parent, candidate, order - 1)
            if found is not None:
                return found[0] + [hit], found[1] + [wall_index]
        return None

    return back(receiver, image, last_wall, order)
