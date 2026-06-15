from typing import List
import numpy as np
from numpy.typing import NDArray

from compas_acoustics import Defaults
from compas_acoustics import Tools


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
        
        print(f"Simulation setting:\n"+
              f"Sampling rate={self.sampling_rate}\n"+
              f"Image source order={image_source_order}\n"+
              f"Air absorption={air_absorption}"
              )
        
        if ray_tracing:
            room.set_ray_tracing(n_rays=n_rays)
            print(f"Ray tracing={ray_tracing}\n"+
                f"N rays={n_rays}\n")

        return room
    

    def rir_from_image_source_method(self, image_source_order:int=6, air_absorption:bool=True):
        if self.simulation_platform == 'pyroomacoustics':
            # prepare the walls and materials for the simulation
            simulation = self._pyroomacoustics_simulation(image_source_order, ray_tracing=False, air_absorption=air_absorption)
            
            # add sources and receivers to the simulation
            for source in self.sources:
                if(simulation.is_inside(source.position)):
                    simulation.add_source(source.position)
                else:
                    print(f"Warning: Source '{source.name}' is outside the room and will not be added to the simulation.")
            for receiver in self.receivers:
                if(simulation.is_inside(receiver.position)):
                    simulation.add_microphone(receiver.position)
                else:
                    print(f"Warning: Receiver '{receiver.name}' is outside the room and will not be added to the simulation.")
        else:
            raise NotImplementedError(f"Simulation platform '{self.simulation_platform}' is not implemented yet.")
        
        simulation.image_source_model()
        simulation.compute_rir()
        
        self.rirs = self.resample_room_rir(room_rir=simulation.rir, orig_sr=simulation.fs, target_sr=self.sampling_rate)
        return self.rirs
    
    
    def rir_from_ray_tracing(self, image_source_order:int=3, n_rays:int=10000, air_absorption:bool=True):
        if self.simulation_platform == 'pyroomacoustics':
            # prepare the walls and materials for the simulation
            simulation = self._pyroomacoustics_simulation(image_source_order, ray_tracing=True, n_rays=n_rays, air_absorption=air_absorption)
            
            # add sources and receivers to the simulation
            for source in self.scene.sources:
                if(simulation.is_inside(source.position)):
                    simulation.add_source(source.position)
                else:
                    print(f"Warning: Source '{source.name}' is outside the room and will not be added to the simulation.")
            for receiver in self.scene.receivers:
                if(simulation.is_inside(receiver.position)):
                    simulation.add_microphone(receiver.position)
                else:
                    print(f"Warning: Receiver '{receiver.name}' is outside the room and will not be added to the simulation.")
        else:
            raise NotImplementedError(f"Simulation platform '{self.simulation_platform}' is not implemented yet.")
        
        simulation.compute_rir()
        
        self.rirs = self.resample_room_rir(room_rir=simulation.rir, orig_sr=simulation.fs, target_sr=self.sampling_rate)
        return self.rirs
    
    
    def resample_room_rir(self, room_rir:List[List], orig_sr:int, target_sr:int, method:str = "poly") -> List[List[NDArray[np.float64]]]:
        """
        Resample all impulse responses in a pyroomacoustics room.rir structure.

        Parameters
        ----------
        room_rir : Sequence[Sequence[ArrayLike]]
            room.rir structure:
            room_rir[mic_idx][source_idx] -> 1D impulse response
        orig_sr : int
            Original sample rate.
        target_sr : int
            Target sample rate.
        method : str, default="poly"
            Resampling method passed to resample_ir().

        Returns
        -------
        List[List[NDArray[np.float64]]]
            Resampled room.rir with the same nested-list structure.
        """
        #TODO does not work in Rhino.
        # return [
        #     [Tools().resample_signal(ir, orig_sr, target_sr, method=method) for ir in mic_rirs]
        #     for mic_rirs in room_rir
        # ]
        
        #TODO this one works in Rhino, but not when run in a Jupyter notebook. Not sure why.
        results = []
        for mic_rirs in room_rir:
            mic_results = []
            for ir in mic_rirs:
                result = Tools().resample_signal(ir, orig_sr, target_sr, method=method)
                mic_results.append(result)
            results.append(mic_results)
        return results 