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

__all__ = ['Ray', 'ScatteredRay', 'RayField']

DEFAULT_SOUND_SPEED = 343.0


class Ray(Polyline):
    """A single acoustic ray traced through a scene, represented as a polyline.

    The vertices of the polyline are ``[source_position, hit_1, hit_2, ...]`` where
    each ``hit_i`` is a point where the ray was reflected by a surface.

    Parameters
    ----------
    points : list
        The polyline vertices, starting at the ray origin.
    source : :class:`compas_acoustics.Source`, optional
        The source the ray was emitted from.
    source_index : int, optional
        The index of the source in the simulation.
    ray_index : int, optional
        The index of the ray among all rays shot from the source.
    direction : :class:`compas.geometry.Vector` or list, optional
        The initial (unit) direction of the ray. If not provided it is derived
        from the first segment of the polyline.
    sound_speed : float, optional
        The speed of sound in m/s used to convert travel distance to travel time.
    distance_offset : float, optional
        The distance already travelled before ``points[0]``. This is ``0`` for
        specular rays emitted from the source, and non-zero for scattered rays
        that are born at a reflection point further along the path.
    name : str, optional
        The name of the ray.
    """

    def __init__(
        self,
        points,
        source=None,
        source_index: Optional[int] = None,
        ray_index: Optional[int] = None,
        direction=None,
        sound_speed: float = DEFAULT_SOUND_SPEED,
        distance_offset: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(points, name=name)
        self.source = source
        self.source_index = source_index
        self.ray_index = ray_index
        self.sound_speed = sound_speed
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
            sound_speed=self.sound_speed,
            distance_offset=self.distance_offset,
        )
        scattered = getattr(self, 'scattered_reflections', None)
        if scattered:
            data["scattered_reflections"] = scattered
        return data

    @classmethod
    def __from_data__(cls, data):
        data = dict(data)
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
        return self.total_distance / self.sound_speed * 1000.0

    @property
    def reflection_order(self) -> int:
        """The number of segments (hops) of the ray."""
        return max(len(self.points) - 1, 0)

    def is_in_cone(self, axis, half_angle_deg: float) -> bool:
        """Test whether the ray falls inside a spotlight cone.

        The cone apex is the ray origin (``points[0]``), its axis is ``axis`` and
        its aperture is ``half_angle_deg`` measured from the axis to the side.
        Because the ray emanates from its origin, membership reduces to the angle
        between the ray direction and the cone axis being within the half angle.

        Parameters
        ----------
        axis : :class:`compas.geometry.Vector` or list
            The direction the cone points in.
        half_angle_deg : float
            The half angle of the cone in degrees.

        Returns
        -------
        bool
        """
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
            sound_speed=self.sound_speed,
            distance_offset=self.distance_offset,
            name=self.name,
        )

    def up_to_reflection_order(self, order: Optional[int]) -> "Ray":
        """Return a copy of the ray truncated to the first ``order`` segments.

        Parameters
        ----------
        order : int or None
            The number of segments (hops) to keep. ``None`` returns the full ray.

        Returns
        -------
        :class:`compas_acoustics.Ray`
        """
        if order is None:
            return self._clone_with_points(self.points[:])
        order = max(int(order), 0)
        points = self.points[: order + 1]
        if len(points) < 1:
            points = self.points[:1]
        return self._clone_with_points(points)

    def up_to_distance(self, distance: Optional[float]) -> Optional["Ray"]:
        """Return a copy of the ray truncated at a total travelled ``distance``.

        The ``distance`` is measured from the true ray origin (the source), so it
        accounts for ``distance_offset``. The final partial segment is interpolated.

        Parameters
        ----------
        distance : float or None
            The travel distance (from the source) at which to cut the ray.
            ``None`` returns the full ray.

        Returns
        -------
        :class:`compas_acoustics.Ray` or None
            ``None`` if the ray has not emerged yet at the given distance.
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
        """Return a copy of the ray truncated at a travel ``time_ms`` in milliseconds.

        Parameters
        ----------
        time_ms : float or None
            The travel time in milliseconds. ``None`` returns the full ray.

        Returns
        -------
        :class:`compas_acoustics.Ray` or None
        """
        if time_ms is None:
            return self._clone_with_points(self.points[:])
        distance = self.sound_speed * time_ms / 1000.0
        return self.up_to_distance(distance)


class ScatteredRay(Ray):
    """A scattered ray born at a reflection point of a specular ray.

    Parameters
    ----------
    points : list
        The polyline vertices, starting at the reflection point it scattered from.
    parent_ray_index : int, optional
        The index of the specular ray this ray branched from.
    energy : float, optional
        The scattered energy carried by the ray (for colouring / thresholding).
    origin_reflection_order : int, optional
        The number of segments the parent ray had travelled to reach the hit
        point this scattered ray originates from.
    **kwargs
        Additional arguments forwarded to :class:`compas_acoustics.Ray`.
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
            sound_speed=self.sound_speed,
            distance_offset=self.distance_offset,
            name=self.name,
        )


class RayField(Data):
    """A collection of traced acoustic rays with filtering and visualisation helpers.

    Parameters
    ----------
    rays : list of :class:`compas_acoustics.Ray`, optional
        The specular rays. Each ray may carry its scattered rays in
        ``ray.scattered_reflections``.
    sound_speed : float, optional
        The speed of sound in m/s used for time based operations.
    name : str, optional
        The name of the ray field.
    """

    def __init__(self, rays: Optional[List[Ray]] = None, sound_speed: float = DEFAULT_SOUND_SPEED, name: Optional[str] = None):
        super().__init__(name=name)
        self.rays = list(rays) if rays else []
        self.sound_speed = sound_speed

    @property
    def __data__(self):
        # Each source once, by its index, instead of once per ray
        sources = {ray.source_index: ray.source for ray in self.rays if ray.source is not None and ray.source_index is not None}
        return {"rays": self.rays, "sound_speed": self.sound_speed, "sources": [[index, source] for index, source in sorted(sources.items())]}

    @classmethod
    def __from_data__(cls, data):
        sources = {int(index): source for index, source in data.get("sources", [])}
        field = cls(data["rays"], sound_speed=data["sound_speed"])
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
        time_thres: Optional[float] = None,
        energy_thres: Optional[float] = None,
        sources=None,
        include_scattering: bool = False,
        scatter_density: int = 10,
        scatter_max_bounces: int = 1,
        scatter_band: Optional[int] = None,
        random_seed: Union[bool, int] = True,
    ):
        """Trace acoustic rays through the scene for visualisation.

        The specular ray paths are reconstructed by replicating the ray-tracing loop
        of the pyroomacoustics ``libroom`` engine (which does not itself expose ray
        paths) using its ``next_wall_hit`` and ``normal_reflect`` primitives. When
        ``include_scattering`` is set, extra scattered rays are spawned at each hit,
        with their number and directions driven by the surface scattering coefficient.

        By default rays are emitted omnidirectionally from each source. If a spotlight
        cone is specified (via ``direction`` and ``angle``), all ``n_rays`` are instead
        emitted directly into that cone (apex at each source), so no separate filtering
        step is needed.

        Parameters
        ----------
        scene : :class:`compas_acoustics.Scene`
            The scene whose room the rays are traced in.
        n_rays : int, default=500
            The number of rays to shoot from each source.
        max_reflection_order : int, default=10
            The maximum number of reflections (segments) to trace per ray.
        direction : :class:`compas.geometry.Vector` or list, optional
            The axis of a spotlight cone (apex at each source) to emit rays into.
            When omitted, rays are emitted omnidirectionally. Requires ``angle``.
        angle : float, optional
            The half angle of the spotlight cone in degrees. Requires ``direction``.
        time_thres : float, optional
            Maximum travel time of a ray in seconds. Converted to a distance using
            the speed of sound. When reached, the ray is stopped.
        energy_thres : float, optional
            Energy threshold below which a ray is stopped (based on surface absorption).
        sources : list of :class:`compas_acoustics.Source`, optional
            The sources to shoot rays from. Defaults to all sources in the scene.
            Each ray's ``source_index`` is its source's index in ``scene.sources``.
        include_scattering : bool, default=False
            Whether to spawn coefficient-driven scattered (diffuse) rays at each reflection.
        scatter_density : int, default=10
            The maximum number of scattered rays per hit (scaled by the scattering coefficient).
        scatter_max_bounces : int, default=1
            The number of further reflections to trace for each scattered ray.
        scatter_band : int, optional
            Index of the frequency band whose scattering coefficient to use. Defaults
            to the mean across all bands.
        random_seed : bool or int, default=True
            Seed for the scattered-direction sampler. True seeds with 42, so that the same scene gives the same rays;
            an int seeds with that value; False (or None) leaves it unseeded, new directions each call.

        Returns
        -------
        :class:`compas_acoustics.RayField`
        """
        if (direction is None) != (angle is None):
            raise ValueError("Both 'direction' and 'angle' must be provided together to emit rays into a cone.")

        import pyroomacoustics as pra

        from compas_acoustics.defaults import Defaults
        from compas_acoustics.room_geometry import RoomGeometry
        from compas_acoustics.simulation_methods.base import _seed

        geometry = RoomGeometry.from_scene(scene)
        simulation = geometry.to_pra(Defaults().SAMPLING_RATE, max_reflection_order, ray_tracing=False, air_absorption=False)
        engine = simulation.room_engine
        walls = engine.walls
        max_dist = float(engine.max_dist)
        sound_speed = float(getattr(simulation, 'c', 343.0))

        if sources is None:
            sources = scene.sources

        # A ray keeps its source's index in the scene, whichever sources are traced
        scene_index = {id(source): index for index, source in enumerate(scene.sources)}
        sim_sources = []
        for position, source in enumerate(sources):
            if simulation.is_inside(source.position):
                sim_sources.append((scene_index.get(id(source), position), source))
            else:
                warnings.warn(f"Source '{source.name}' is outside the room and no rays will be traced from it.", UserWarning, stacklevel=2)

        rng = np.random.default_rng(_seed(random_seed))
        if direction is None:
            directions = np.asarray(pra.fibonacci_spherical_sampling(n_rays)).T
        else:
            directions = _cone_sampling(direction, angle, n_rays)

        distance_thres = time_thres * sound_speed if time_thres is not None else None

        rays = []
        for source_index, source in sim_sources:
            source_position = np.asarray(source.position, dtype=float)

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
                                    sound_speed=sound_speed,
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
                    sound_speed=sound_speed,
                )
                ray.scattered_reflections = scattered_reflections
                rays.append(ray)

        return cls(rays, sound_speed=sound_speed)

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
        """Return the position of every specular ray at ``time_ms``, as particles for a billiard view.

        Parameters
        ----------
        time_ms : float
            The time in milliseconds, clamped to ``[0, duration_ms]``.

        Returns
        -------
        tuple of (list of :class:`compas.geometry.Point`, list of int, list of int)
            The positions, the ``ray_index`` of the ray each belongs to, and the
            number of reflections each has passed (its reflection order).
            A ray that has ended by ``time_ms`` gives no particle.
        """
        time_ms = min(max(time_ms, 0.0), self.duration_ms)
        points = []
        indices = []
        orders = []
        for ray in self.rays:
            if ray.travel_time < time_ms:
                continue
            part = ray.up_to_time(time_ms)
            if part is None:
                # up_to_distance reports "not emerged" at distance == distance_offset too
                if ray.sound_speed * time_ms / 1000.0 < ray.cumulative_distances[0]:
                    continue
                points.append(Point(*ray.points[0]))
            else:
                points.append(Point(*part.points[-1]))
            indices.append(ray.ray_index)
            distance = ray.sound_speed * time_ms / 1000.0
            orders.append(sum(1 for reached in ray.cumulative_distances[1:] if reached <= distance))
        return points, indices, orders

    def filter(
        self,
        source=None,
        ray_index: Optional[Union[int, List[int]]] = None,
    ) -> "RayField":
        """Return a new :class:`RayField` keeping only the rays matching the filters.

        Parameters
        ----------
        source : :class:`compas_acoustics.Source` or int, optional
            Keep only rays emitted from this source (matched by object or index).
        ray_index : int or list of int, optional
            Keep only rays with these ray indices.

        Returns
        -------
        :class:`RayField`
        """
        ray_indices = None
        if ray_index is not None:
            ray_indices = {ray_index} if isinstance(ray_index, int) else set(ray_index)

        kept = []
        for ray in self.rays:
            if source is not None and not self._matches_source(ray, source):
                continue
            if ray_indices is not None and ray.ray_index not in ray_indices:
                continue
            kept.append(ray)

        return RayField(kept, sound_speed=self.sound_speed, name=self.name)

    @staticmethod
    def _matches_source(ray: Ray, source) -> bool:
        if isinstance(source, int):
            return ray.source_index == source
        return ray.source is source

    def truncated(self, reflection_order: Optional[int] = None, travel_time_ms: Optional[float] = None) -> "RayField":
        """Return a new :class:`RayField` with every ray truncated for partial display.

        Parameters
        ----------
        reflection_order : int, optional
            Keep only the first ``reflection_order`` segments of each ray.
        travel_time_ms : float, optional
            Keep only the portion of each ray reachable within this travel time
            (converted to distance via the speed of sound).

        Returns
        -------
        :class:`RayField`
        """
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
        return RayField(new_rays, sound_speed=self.sound_speed, name=self.name)

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
        """Display the rays (and optionally the scene) in a compas viewer.

        Parameters
        ----------
        scene : :class:`compas_acoustics.Scene`, optional
            If provided, the acoustic geometry, sources and receivers are drawn as context.
        reflection_order : int, optional
            Only display the first ``reflection_order`` segments of each ray.
        travel_time_ms : float, optional
            Only display the portion of each ray reachable within this travel time.
        show_specular : bool, optional
            Whether to draw the specular rays.
        show_scattered : bool, optional
            Whether to draw the scattered rays.
        color : :class:`compas.colors.Color`, optional
            The colour of the specular rays.
        scattered_color : :class:`compas.colors.Color`, optional
            The colour of the scattered rays.
        linewidth : float, optional
            The width of the drawn ray lines.
        show : bool, optional
            Whether to open the viewer. If ``False`` the configured viewer is returned
            without showing (useful for testing / further customisation).

        Returns
        -------
        The configured viewer instance.
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


def _cone_sampling(direction, angle: float, n_rays: int):
    """Evenly sample ``n_rays`` unit directions on the spherical cap of a cone.

    The cone axis is ``direction`` and its half aperture is ``angle`` (degrees).
    A deterministic fibonacci-cap lattice is used so results are reproducible and
    do not depend on the random seed.

    Parameters
    ----------
    direction : :class:`compas.geometry.Vector` or list
        The cone axis. Does not need to be normalized.
    angle : float
        The half angle of the cone in degrees.
    n_rays : int
        The number of directions to sample.

    Returns
    -------
    numpy.ndarray
        An array of shape ``(n_rays, 3)`` of unit direction vectors.
    """
    axis = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(axis)
    if TOL.is_zero(norm):
        raise ValueError("Cone 'direction' must be a non-zero vector.")
    axis = axis / norm

    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent_1 = np.cross(axis, helper)
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(axis, tangent_1)

    cos_half = np.cos(np.radians(angle))
    golden = np.pi * (3.0 - np.sqrt(5.0))

    directions = np.empty((n_rays, 3), dtype=float)
    for i in range(n_rays):
        z = 1.0 - (i + 0.5) / n_rays * (1.0 - cos_half)
        radius = np.sqrt(max(1.0 - z * z, 0.0))
        phi = i * golden
        vector = radius * np.cos(phi) * tangent_1 + radius * np.sin(phi) * tangent_2 + z * axis
        directions[i] = vector / np.linalg.norm(vector)
    return directions


def _scatter_coefficient(wall, scatter_band: Optional[int]) -> float:
    coefficients = np.asarray(wall.scatter, dtype=float)
    if coefficients.size == 0:
        return 0.0
    if scatter_band is not None:
        value = float(coefficients[scatter_band])
    else:
        value = float(np.mean(coefficients))
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
    sound_speed: float,
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
                sound_speed=sound_speed,
                distance_offset=travel_distance,
            )
        )
    return scattered_rays


def _sample_cone_directions(axis, normal, n_directions: int, theta_max: float, rng):
    """Sample cosine-weighted directions inside a cone around ``axis`` (half-angle ``theta_max``)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent_1 = np.cross(axis, helper)
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(axis, tangent_1)

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
