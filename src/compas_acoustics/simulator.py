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
        
        # create the walls
        walls = []
        for acoustic_geometry in self.scene.acoustic_objects:
            for fkey in acoustic_geometry.acoustic_geometry.faces():
                # verticies of the face
                face_vertices = acoustic_geometry.acoustic_geometry.face_vertices(fkey)
                pts = np.asarray([acoustic_geometry.acoustic_geometry.vertex_coordinates(vkey) for vkey in face_vertices])
                
                # acoustic material of the face
                surface_group = acoustic_geometry.acoustic_geometry.face_attribute(fkey, 'surface_group')
                acoustic_material = acoustic_geometry.acoustic_materials.get(surface_group, None)
                if acoustic_material is None:
                    acoustic_material = self.scene.acoustic_materials.get(surface_group, None)
                    if acoustic_material is None:
                        raise ValueError(f"No acoustic material assigned to surface group '{surface_group}'.")
                    
                material_settings = {
                    'description': acoustic_material.name,
                    'coeffs': acoustic_material.absorption_coefficients,
                    'scattering': acoustic_material.scattering_coefficients,
                    'center_freqs': acoustic_material.frequency_bands
                }
                
                mat = pra.Material(material_settings)
               
                # create wall
                wall = pra.wall_factory(pts.T,
                                        mat.energy_absorption['coeffs'],
                                        mat.scattering['coeffs'])
                walls.append(wall)
                            
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