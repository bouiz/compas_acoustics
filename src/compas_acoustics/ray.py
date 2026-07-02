from typing import List
from typing import Optional
from typing import Union

from compas.data import Data
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import Vector

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

    def __len__(self) -> int:
        return len(self.rays)

    def __iter__(self):
        return iter(self.rays)

    @property
    def scattered_rays(self) -> List[ScatteredRay]:
        """All scattered rays across every ray in the field."""
        scattered = []
        for ray in self.rays:
            scattered.extend(getattr(ray, 'scattered_reflections', []))
        return scattered

    def filter(
        self,
        source=None,
        ray_index: Optional[Union[int, List[int]]] = None,
        direction=None,
        angle: Optional[float] = None,
    ) -> "RayField":
        """Return a new :class:`RayField` keeping only the rays matching the filters.

        Parameters
        ----------
        source : :class:`compas_acoustics.Source` or int, optional
            Keep only rays emitted from this source (matched by object or index).
        ray_index : int or list of int, optional
            Keep only rays with these ray indices.
        direction : :class:`compas.geometry.Vector` or list, optional
            The axis of a spotlight cone (apex at each ray's source). Requires ``angle``.
        angle : float, optional
            The half angle of the spotlight cone in degrees. Requires ``direction``.

        Returns
        -------
        :class:`RayField`
        """
        if (direction is None) != (angle is None):
            raise ValueError("Both 'direction' and 'angle' must be provided together to filter by cone.")

        ray_indices = None
        if ray_index is not None:
            ray_indices = {ray_index} if isinstance(ray_index, int) else set(ray_index)

        kept = []
        for ray in self.rays:
            if source is not None and not self._matches_source(ray, source):
                continue
            if ray_indices is not None and ray.ray_index not in ray_indices:
                continue
            if direction is not None and not ray.is_in_cone(direction, angle):
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
