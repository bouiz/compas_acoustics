from typing import List
import numpy as np
from numpy.typing import NDArray

from compas_viewer import Viewer
from compas_viewer.scene import Tag

from compas.data import Data
from compas_acoustics import Defaults
from compas_acoustics.tools import Tools

__all__ = ['Scene']

class Scene(Data):
    
    def __init__(self, name:str, description:str="", simulation_platform:str='pyroomacoustics'):
        super().__init__()
        self.name = name
        self.description = description
        self._acoustic_objects = []
        self._sources = []
        self._receivers = []
        self.acoustic_materials = {}
        self._supported_simulation_platforms = ['pyroomacoustics']
        self.simulation_platform = simulation_platform
        self.sampling_rate = Defaults().SAMPLING_RATE
        # self.irs = None
        
        
    @property
    def id(self):
        return self.guid
    
    
    @property
    def acoustic_objects(self):
        return self._acoustic_objects
    
    
    @property
    def sources(self):
        all_sources = []
        
        # add standalone sources
        all_sources.extend(self._sources)
        
        # add sources from acoustic objects
        for acoustic_object in self.acoustic_objects:
            all_sources.extend(acoustic_object.sources)
        
        return all_sources
    
    
    @property
    def receivers(self):
        return self._receivers
    
    
    @property
    def number_of_acoustic_objects(self):
        return len(self._acoustic_objects)
    
    
    @property
    def number_of_sources(self):
        return len(self._sources)
    
    
    @property
    def number_of_receivers(self):
        return len(self._receivers)
    
    
    @property
    def description(self):
        return self._description
    
    
    @description.setter
    def description(self, description:str):
        self._description = description
        return self._description
    
    
    @property
    def name(self):
        return self._name
    
    
    @name.setter
    def name(self, name:str):
        self._name = name
        return self._name
    
    
    @property
    def acoustic_materials(self):
        return self._acoustic_materials
    
    
    @acoustic_materials.setter
    def acoustic_materials(self, acoustic_materials:dict):
        self._acoustic_materials = acoustic_materials          
    
    
    @property
    def simulation_platform(self):
        return self._simulation_platform
    
    
    @property
    def supported_simulation_platforms(self):
        return self._supported_simulation_platforms
    
    
    @property
    def scene_surface_groups(self):
        surface_groups = []
        for acoustic_geometry in self.acoustic_objects:
            groups = acoustic_geometry.surface_groups.keys()
            surface_groups.extend(groups)
        return surface_groups
    
    
    def print_scene_surface_groups(self):
        print(f"Scene '{self.name}' surface groups:")
        for acoustic_geometry in self.acoustic_objects:
            print(f"Acoustic Object '{acoustic_geometry.name}':")
            for group_name in acoustic_geometry.surface_groups.keys():
                print(f"  {group_name}")
                
    
    def display_surface_groups(self):
        viewer = Viewer()
        
        for acoustic_geometry in self.acoustic_objects:
            group_colors = acoustic_geometry.surface_group_colors
            
            for fkey in acoustic_geometry.acoustic_geometry.faces():
                f_centroid = acoustic_geometry.acoustic_geometry.face_centroid(fkey)
                surface_group = acoustic_geometry.acoustic_geometry.face_attribute(fkey, 'surface_group')
                tag = Tag(surface_group, f_centroid, height=30, color=group_colors[surface_group])
                viewer.scene.add(tag, name=f"tag_{fkey}")
            
            viewer.scene.add(acoustic_geometry.acoustic_geometry, name=acoustic_geometry.name, show_points=True, facecolor=acoustic_geometry.face_colors)
        viewer.show()
    
    
    @simulation_platform.setter
    def simulation_platform(self, simulation_platform:str):
        if simulation_platform not in self._supported_simulation_platforms:
            raise ValueError(f"Simulation platform '{simulation_platform}' is not supported. Supported platforms are: {self._supported_simulation_platforms}")
        
        self._simulation_platform = simulation_platform
        
    
    def simulation(self, image_source_order, ray_tracing, n_rays, air_absorption):
        simulation = None
        
        match self.simulation_platform:
            case 'pyroomacoustics':
                # prepare the walls and materials for the simulation
                simulation = self._pyroomacoustics_simulation(image_source_order, ray_tracing, n_rays, air_absorption)
                
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
            case _:
                raise NotImplementedError(f"Simulation platform '{self.simulation_platform}' is not implemented yet.")
            
        return simulation
    
    
    def _pyroomacoustics_simulation(self, image_source_order:int=3, ray_tracing:bool=False, n_rays:int=10000, air_absorption:bool=True):
        import pyroomacoustics as pra
        import numpy as np
        
        # create the walls
        walls = []
        for acoustic_geometry in self.acoustic_objects:
            for fkey in acoustic_geometry.acoustic_geometry.faces():
                # verticies of the face
                face_vertices = acoustic_geometry.acoustic_geometry.face_vertices(fkey)
                pts = np.asarray([acoustic_geometry.acoustic_geometry.vertex_coordinates(vkey) for vkey in face_vertices])
                
                # acoustic material of the face
                surface_group = acoustic_geometry.acoustic_geometry.face_attribute(fkey, 'surface_group')
                acoustic_material = acoustic_geometry.acoustic_materials.get(surface_group, None)
                if acoustic_material is None:
                    acoustic_material = self.acoustic_materials.get(surface_group, None)
                    if acoustic_material is None:
                        raise ValueError(f"No acoustic material assigned to surface group '{surface_group}'.")
                    
                material_settings = {
                    'description': acoustic_material.name,
                    'coeffs': acoustic_material.absorption_coefficients,
                    'scattering': acoustic_material.scattering_coefficients,
                    'center_freqs': acoustic_material.frequency_bands
                }
                
                mat = pra.Material(material_settings)
                    
                # mat = pra.Material(energy_absorption=acoustic_material.absorption_coefficients, scattering=acoustic_material.scattering_coefficients)
               
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
    

    def rir_from_image_source_method(self, image_source_order:int=3, air_absorption:bool=True): 
        simulation = self.simulation(image_source_order=image_source_order, ray_tracing=False, n_rays=10000, air_absorption=air_absorption)
        simulation.image_source_model()
        simulation.compute_rir()
        
        return self.resample_room_rir(room_rir=simulation.rir, orig_sr=simulation.fs, target_sr=self.sampling_rate)
    
    
    def rir_from_ray_tracing(self, image_source_order:int=3, n_rays:int=10000, air_absorption:bool=True):
        simulation = self.simulation(image_source_order=image_source_order, ray_tracing=True, n_rays=n_rays, air_absorption=air_absorption)
        simulation.compute_rir()
        
        return self.resample_room_rir(room_rir=simulation.rir, orig_sr=simulation.fs, target_sr=self.sampling_rate)
    
    
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

        return [
            [Tools().resample_signal(ir, orig_sr, target_sr, method=method) for ir in mic_rirs]
            for mic_rirs in room_rir
        ]
    


    def add_acoustic_object(self, acoustic_object):
        self._acoustic_objects.append(acoustic_object)
        
    
    def add_acoustic_objects(self, acoustic_objects:list):
        self._acoustic_objects.extend(acoustic_objects)


    def add_source(self, source):
        self._sources.append(source)

    def add_receiver(self, receiver):
        self._receivers.append(receiver)
