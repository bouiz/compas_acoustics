from compas.data import Data

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
        self.simulation = None
        # self.irs = None
        
        
    @property
    def id(self):
        return self.guid
    
    
    @property
    def acoustic_objects(self):
        return self._acoustic_objects
    
    
    @property
    def sources(self):
        return self._sources
    
    
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
    
    
    @simulation_platform.setter
    def simulation_platform(self, simulation_platform:str):
        if simulation_platform not in self._supported_simulation_platforms:
            raise ValueError(f"Simulation platform '{simulation_platform}' is not supported. Supported platforms are: {self._supported_simulation_platforms}")
        
        self._simulation_platform = simulation_platform
        
    
    @property
    def simulation(self):
        if self._simulation is None:
            match self.simulation_platform:
                case 'pyroomacoustics':
                    self.simulation = self._prepare_pyroomacoustics_room()
                case _:
                    raise NotImplementedError(f"Simulation platform '{self.simulation_platform}' is not implemented yet.")
        return self._simulation
    
        
    @simulation.setter
    def simulation(self, simulation):
        self._simulation = simulation
        return self._simulation
    
    
    def _prepare_pyroomacoustics_room(self):
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
                wall = pra.wall_factory(pts.T, mat.energy_absorption['coeffs'], mat.scattering['coeffs'])
                walls.append(wall)
                
        # create the room
        room = pra.Room(
            walls,
            fs=16000,
            max_order=2,
            ray_tracing=False,
            air_absorption=False,
        )

        return room
    
    @name.setter
    def name(self, name:str):
        self._name = name
        return self._name


    def add_acoustic_object(self, acoustic_object):
        self._acoustic_objects.append(acoustic_object)
        
    
    def add_acoustic_objects(self, acoustic_objects:list):
        self._acoustic_objects.extend(acoustic_objects)


    def add_source(self, source):
        self._sources.append(source)

    def add_receiver(self, receiver):
        self._receivers.append(receiver)
