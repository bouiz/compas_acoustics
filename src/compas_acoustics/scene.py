# from compas_viewer import Viewer # Does not work in Rhino
# from compas_viewer.scene import Tag # Does not work in Rhino

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
           

    def add_acoustic_object(self, acoustic_object):
        self._acoustic_objects.append(acoustic_object)
        
    
    def add_acoustic_objects(self, acoustic_objects:list):
        self._acoustic_objects.extend(acoustic_objects)


    def add_source(self, source):
        self._sources.append(source)

    def add_receiver(self, receiver):
        self._receivers.append(receiver)
