from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from compas_acoustics import Defaults

# from compas_acoustics import Tools
from compas_acoustics import ImpulseResponse
from compas_acoustics.receiver import Receiver
from compas_acoustics.source import Source
from compas_acoustics.pra_converter import acoustic_object_to_pra_walls

__all__ = ['Simulator']


class Simulator:
    """Base class for acoustic simulators."""

    def __init__(self, scene, simulation_platform='pyroomacoustics'):
        self.scene = scene
        self._supported_simulation_platforms = ['pyroomacoustics']
        self.simulation_platform = simulation_platform
        self.sampling_rate = Defaults().SAMPLING_RATE
        self.rirs = None
    
    
    @property
    def simulation_platform(self):
        return self._simulation_platform
    
    
    @property
    def supported_simulation_platforms(self):
        return self._supported_simulation_platforms


    @simulation_platform.setter
    def simulation_platform(self, simulation_platform:str):
        if simulation_platform not in self._supported_simulation_platforms:
            raise ValueError(f"Simulation platform '{simulation_platform}' is not supported. Supported platforms are: {self._supported_simulation_platforms}")
        
        self._simulation_platform = simulation_platform
    
    
    def _pyroomacoustics_simulation(self, image_source_order:int=3, ray_tracing:bool=False, n_rays:int=10000, air_absorption:bool=True):
        """
        Prepare the walls and materials for a pyroomacoustics simulation.
        
        Parameters
        ----------
        image_source_order : int, default=3
            The maximum order of image sources to consider in the simulation. This is used to determine the maximum path length for ray tracing.
        ray_tracing : bool, default=False
            Whether to use ray tracing in the simulation. If False, the image source method will be used.
        n_rays : int, default=10000
            The number of rays to trace in the simulation. Only used if ray_tracing is True.
        air_absorption : bool, default=True
            Whether to include air absorption in the simulation.
        Returns
        -------
        pyroomacoustics.Room
            A pyroomacoustics Room object prepared for simulation.
        """
        
        import pyroomacoustics as pra
        
        # create walls
        walls = []
        for obj in self.scene.acoustic_objects:
            walls.extend(acoustic_object_to_pra_walls(obj))
    
        # create the room
        room = pra.Room(
            walls,
            fs=16000, #TODO Figure out how to set this to something more than 16000. Currently, anything above 16000 will cause an error when air absorption is enabled.
            max_order=image_source_order,
            ray_tracing=ray_tracing,
            air_absorption=air_absorption,
        )
        
        print("Simulation setting:\n"+
              f"Sampling rate={self.sampling_rate}\n"+
              f"Image source order={image_source_order}\n"+
              f"Air absorption={air_absorption}"
              )
        
        if ray_tracing:
            room.set_ray_tracing(n_rays=n_rays)
            print(f"Ray tracing={ray_tracing}\n"+
                f"N rays={n_rays}\n")

        return room
    
    
    def _add_pyroomacoustics_sources_receivers(self, room) -> Tuple[List[Source], List[Receiver]]:
        sim_sources = []
        sim_receivers = []

        for source in self.scene.sources:
            if room.is_inside(source.position):
                room.add_source(source.position)
                sim_sources.append(source)
            else:
                print(f"Warning: Source '{source.name}' is outside the room and will not be added to the simulation.")

        for receiver in self.scene.receivers:
            if room.is_inside(receiver.position):
                room.add_microphone(receiver.position)
                sim_receivers.append(receiver)
            else:
                print(f"Warning: Receiver '{receiver.name}' is outside the room and will not be added to the simulation.")

        return sim_sources, sim_receivers

    def _seed_pyroomacoustics(self, random_seed: Optional[int]) -> None:
        import pyroomacoustics as pra

        if random_seed is not None:
            pra.random.seed(numpy=random_seed, libroom=random_seed)
        else:
            pra.random.seed()

    
    def _update_scene_ir_graph(self, rirs: dict) -> None:
        self.scene.populate_ir_graph(rirs)
    

    def rir_from_image_source_method(self, image_source_order:int=6, air_absorption:bool=True):
        """
        Compute the room impulse response (RIR) using the image source method.
        
        Parameters
        ----------
        image_source_order : int, default=6
            The maximum order of image sources to consider in the simulation.
        air_absorption : bool, default=True
            Whether to include air absorption in the simulation.
        Returns
        -------
        Dict[str, ImpulseResponse]
            A dictionary of ImpulseResponse objects, where each object contains the source index, receiver index, impulse response,
            and sample rate for a single source-receiver pair.
        """
        
        if self.simulation_platform == 'pyroomacoustics':
            # prepare the walls and materials for the simulation
            simulation = self._pyroomacoustics_simulation(image_source_order, ray_tracing=False, air_absorption=air_absorption)
            
            sim_sources, sim_receivers = self._add_pyroomacoustics_sources_receivers(simulation)
            
            simulation.image_source_model()
            simulation.compute_rir()
            self.rirs = self._convert_pyroomacoustics_rirs(
                room_rir=simulation.rir,
                sample_rate=simulation.fs,
                sim_sources=sim_sources,
                sim_receivers=sim_receivers,
            )
            self._update_scene_ir_graph(self.rirs)
        else:
            raise NotImplementedError(f"Simulation platform '{self.simulation_platform}' is not implemented yet.")
        
        return self.rirs
    
    
    def rir_from_ray_tracing(
        self,
        image_source_order: int = 3,
        n_rays: int = 10000,
        air_absorption: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        Compute the room impulse response (RIR) using ray tracing.
        
        Parameters
        ----------
        image_source_order : int, default=3
            The maximum order of image sources to consider in the simulation. This is used to determine the maximum path length for ray tracing.
        n_rays : int, default=10000
            The number of rays to trace in the simulation.
        air_absorption : bool, default=True
            Whether to include air absorption in the simulation.
        random_seed : int, optional
            Seed for the pyroomacoustics random number generator (NumPy and libroom).
            When ``None``, both generators are re-seeded with fresh entropy before
            each simulation so results vary across runs. Pass an integer for
            reproducible ray-tracing results.
        Returns
        -------
        Dict[str, ImpulseResponse]
            A dictionary of ImpulseResponse objects, where each object contains the source index, receiver index, impulse response,
            and sample rate for a single source-receiver pair.
        """
        
        if self.simulation_platform == 'pyroomacoustics':
            self._seed_pyroomacoustics(random_seed)

            # prepare the walls and materials for the simulation
            simulation = self._pyroomacoustics_simulation(image_source_order, ray_tracing=True, n_rays=n_rays, air_absorption=air_absorption)
            
            sim_sources, sim_receivers = self._add_pyroomacoustics_sources_receivers(simulation)

            self._seed_pyroomacoustics(random_seed)

            simulation.compute_rir()
            self.rirs = self._convert_pyroomacoustics_rirs(
                room_rir=simulation.rir,
                sample_rate=simulation.fs,
                sim_sources=sim_sources,
                sim_receivers=sim_receivers,
            )
            self._update_scene_ir_graph(self.rirs)
        else:
            raise NotImplementedError(f"Simulation platform '{self.simulation_platform}' is not implemented yet.")
        
        return self.rirs
    
    
    def trace_rays(
        self,
        n_rays: int = 500,
        max_reflection_order: int = 10,
        direction=None,
        angle: Optional[float] = None,
        time_thres: Optional[float] = None,
        energy_thres: Optional[float] = None,
        sources: Optional[List[Source]] = None,
        include_scattering: bool = False,
        scatter_density: int = 10,
        scatter_max_bounces: int = 1,
        scatter_band: Optional[int] = None,
        random_seed: Optional[int] = None,
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
        include_scattering : bool, default=False
            Whether to spawn coefficient-driven scattered (diffuse) rays at each reflection.
        scatter_density : int, default=10
            The maximum number of scattered rays per hit (scaled by the scattering coefficient).
        scatter_max_bounces : int, default=1
            The number of further reflections to trace for each scattered ray.
        scatter_band : int, optional
            Index of the frequency band whose scattering coefficient to use. Defaults
            to the mean across all bands.
        random_seed : int, optional
            Seed for the scattered-direction sampler, for reproducible results.

        Returns
        -------
        :class:`compas_acoustics.RayField`
        """
        if (direction is None) != (angle is None):
            raise ValueError("Both 'direction' and 'angle' must be provided together to emit rays into a cone.")
        if self.simulation_platform != 'pyroomacoustics':
            raise NotImplementedError(f"Simulation platform '{self.simulation_platform}' is not implemented yet.")

        import pyroomacoustics as pra
        from compas.geometry import Point
        from compas.geometry import Vector

        from compas_acoustics.ray import Ray
        from compas_acoustics.ray import RayField

        simulation = self._pyroomacoustics_simulation(
            image_source_order=max_reflection_order,
            ray_tracing=False,
            air_absorption=False,
        )
        engine = simulation.room_engine
        walls = engine.walls
        max_dist = float(engine.max_dist)
        sound_speed = float(getattr(simulation, 'c', 343.0))

        if sources is None:
            sources = self.scene.sources

        sim_sources = []
        for source in sources:
            if simulation.is_inside(source.position):
                sim_sources.append(source)
            else:
                print(f"Warning: Source '{source.name}' is outside the room and no rays will be traced from it.")

        rng = np.random.default_rng(random_seed)
        if direction is None:
            directions = np.asarray(pra.fibonacci_spherical_sampling(n_rays)).T
        else:
            directions = self._cone_sampling(direction, angle, n_rays)

        distance_thres = time_thres * sound_speed if time_thres is not None else None

        rays = []
        for source_index, source in enumerate(sim_sources):
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
                        scatter = self._scatter_coefficient(wall, scatter_band)
                        if scatter > 0.0:
                            scattered_reflections.extend(
                                self._make_scattered_rays(
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

        return RayField(rays, sound_speed=sound_speed)

    @staticmethod
    def _cone_sampling(direction, angle: float, n_rays: int) -> NDArray[np.float64]:
        """Evenly sample ``n_rays`` unit directions on the spherical cap of a cone.

        The cone axis is ``direction`` and its half aperture is ``angle`` (degrees).
        A deterministic fibonacci-cap lattice is used so results are reproducible and
        do not depend on the random seed.

        Parameters
        ----------
        direction : :class:`compas.geometry.Vector` or list
            The cone axis. Does not need to be normalised.
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
        if norm < 1e-12:
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

    @staticmethod
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
        self,
        engine,
        walls,
        hit,
        incident_direction,
        wall,
        scatter: float,
        transmitted: float,
        travel_distance: float,
        origin_reflection_order: int,
        source: Source,
        source_index: int,
        ray_index: int,
        max_dist: float,
        sound_speed: float,
        scatter_density: int,
        scatter_max_bounces: int,
        rng,
    ):
        from compas.geometry import Point
        from compas.geometry import Vector

        from compas_acoustics.ray import ScatteredRay

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
        axis = normal if axis_norm < 1e-9 else axis / axis_norm

        theta_max = np.radians(scatter * 90.0)
        directions = self._sample_cone_directions(axis, normal, n_scatter, theta_max, rng)

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

    @staticmethod
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

    def _convert_pyroomacoustics_rirs(
        self,
        room_rir: List[List[NDArray[np.float64]]],
        sample_rate: int,
        sim_sources: List[Source],
        sim_receivers: List[Receiver],
    ) -> dict:
        """
        Store the room impulse responses (RIRs) from a pyroomacoustics simulation in a structured format.
        
        Parameters
        ----------
        room_rir : List[List[NDArray[np.float64]]]
            A nested list of impulse responses, where the outer list corresponds to microphones and the inner list corresponds to sources.
            Each impulse response is a 1D numpy array.
        sample_rate : int
            The sampling rate of the impulse responses.
        Returns
        -------
        Dict[str, ImpulseResponse]
            A dictionary of ImpulseResponse objects, where each object contains the source index, receiver index, impulse response,
            and sample rate for a single source-receiver pair.
        """
        rirs = {}
        for mic_idx, mic_rirs in enumerate(room_rir):
            for source_idx, ir in enumerate(mic_rirs):
                source = sim_sources[source_idx]
                receiver = sim_receivers[mic_idx]
                key = f"{receiver.guid}_{source.guid}"
                response = ImpulseResponse(
                    name=key,
                    source=source,
                    receiver=receiver,
                    ir=ir,
                    sample_rate=sample_rate,
                    simulation_platform=self.simulation_platform
                )
                rirs[key] = response
        return rirs
    
    
    # def resample_room_rir(self, room_rir:List[List], orig_sr:int, target_sr:int, method:str = "poly") -> List[List[NDArray[np.float64]]]:
    #     """
    #     Resample all impulse responses in a pyroomacoustics room.rir structure.

    #     Parameters
    #     ----------
    #     room_rir : Sequence[Sequence[ArrayLike]]
    #         room.rir structure:
    #         room_rir[mic_idx][source_idx] -> 1D impulse response
    #     orig_sr : int
    #         Original sample rate.
    #     target_sr : int
    #         Target sample rate.
    #     method : str, default="poly"
    #         Resampling method passed to resample_ir().

    #     Returns
    #     -------
    #     List[List[NDArray[np.float64]]]
    #         Resampled room.rir with the same nested-list structure.
    #     """
    #     #TODO does not work in Rhino.
    #     # return [
    #     #     [Tools().resample_signal(ir, orig_sr, target_sr, method=method) for ir in mic_rirs]
    #     #     for mic_rirs in room_rir
    #     # ]
        
    #     #TODO this one works in Rhino, but not when run in a Jupyter notebook. Not sure why.
    #     results = []
    #     for mic_rirs in room_rir:
    #         mic_results = []
    #         for ir in mic_rirs:
    #             result = Tools().resample_signal(ir, orig_sr, target_sr, method=method)
    #         results.append(mic_results)
    #     return results