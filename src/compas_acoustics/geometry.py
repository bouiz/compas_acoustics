from collections import defaultdict
from random import random

from compas.colors import Color
from compas.data import Data
from compas.datastructures import Mesh
from compas.files import OBJ
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Transformation

# from compas_viewer import Viewer # Does not work in Rhino
# from compas_viewer.scene import Tag # Does not work in Rhino
from compas_acoustics import Source

__all__ = ['AcousticObject', 'AcousticObjectsLibrary']

class AcousticObject(Data):
    
    def __init__(self, acoustic_geometry:Mesh, surface_groups:dict, x:float=0, y:float=0, z:float=0, name:str=None):
        """Initialize Acoustic object

        Parameters
        ----------
        acoustic_geometry : :class:`compas.datastructures.Mesh`
            A COMPAS Mesh representing the acoustic geometry.
        surface_groups : dict
            A dictionary where keys are group names and values are lists of face indices belonging to each group.
        x : float
            The x-coordinate of the geometry's reference point.
        y : float
            The y-coordinate of the geometry's reference point.
        z : float
            The z-coordinate of the geometry's reference point.
        name : str
            The name of the acoustic object. Optional. If not provided, a unique identifier will be generated.
        """
        super().__init__(name=name)
        self._x = x
        self._y = y
        self._z = z
        self.acoustic_geometry = acoustic_geometry
        self.surface_groups = surface_groups # {group name : list of face indices}
        self.acoustic_materials = {} # {surface group name : AcousticMaterial}
        self.sources = []
        self.visualisation_geometry = None
        self.surface_group_colors = None
        self._prepare_mesh()
        
    
    def _prepare_mesh(self):
        """
        Prepare the acoustic geometry mesh by setting default face attributes.
        """
        self.acoustic_geometry.update_default_face_attributes({'surface_group': 'default'})
        
        for fkey in list(self.acoustic_geometry.faces()):
            surface_group = self.acoustic_geometry.face_attribute(fkey, 'surface_group')
            vertices = self.acoustic_geometry.face_vertices(fkey)
            
            if len(vertices) == 4:
                _a, b, _c, d = vertices
                f1, f2 = self.acoustic_geometry.split_face(fkey, b, d)
                self.acoustic_geometry.face_attribute(f1, 'surface_group', surface_group)
                self.acoustic_geometry.face_attribute(f2, 'surface_group', surface_group)
                
                # update surface groups dictionary
                self.surface_groups[surface_group].remove(fkey)
                self.surface_groups[surface_group].extend([f1, f2])
       
       
    @property
    def id(self):
        return self.guid
    
    
    @property
    def position(self):
        return [self._x, self._y, self._z]
        
    
    @position.setter
    def position(self, position:list):
        self._x = position[0]
        self._y = position[1]
        self._z = position[2]

    
    @property
    def acoustic_geometry(self):
        return self._acoustic_geometry
    
    
    @acoustic_geometry.setter
    def acoustic_geometry(self, acoustic_geometry:Mesh):
        self._acoustic_geometry = acoustic_geometry
        
    
    @property
    def surface_groups(self):
        return self._surface_groups
    
    
    @surface_groups.setter
    def surface_groups(self, surface_groups:dict):
        """
        Add surface groups to the acoustic object and assign face attributes.
        
        Parameters
        ----------
        surface_groups : dict
            A dictionary where keys are group names and values are lists of face indices belonging to each group.
        """
        # Set surface groups attribute
        self._surface_groups = surface_groups
        # Assign face attributes for each face in the mesh
        for group_name, face_indices in surface_groups.items():
            for face_index in face_indices:
                self.acoustic_geometry.face_attribute(face_index, 'surface_group', group_name)
    
    
    @property
    def sources(self):
        return self._sources
    
    
    @sources.setter
    def sources(self, sources:list):
        self._sources = sources
    
    
    def add_source(self, source:Source):
        self.sources.append(source)


    @property
    def visualisation_geometry(self):
        return self._visualisation_geometry


    @visualisation_geometry.setter
    def visualisation_geometry(self, visualisation_geometry):
        if isinstance(visualisation_geometry, Mesh):
            self._visualisation_geometry = visualisation_geometry
        elif visualisation_geometry is None:
            self._visualisation_geometry = self.acoustic_geometry
        else:
            raise ValueError("Visualisation geometry must be a compas.datastructures.Mesh object.")
    
    
    @property
    def surface_group_colors(self):
        """
        Generate unique random colors for each surface group.
        
        Returns
        -------
        dict
            A dictionary where keys are surface group names and values are random Color objects.
        """
        if self._surface_group_colors is None:
            self._surface_group_colors = {sk: Color(random(), random(), random(), 0.5) for sk in self.surface_groups.keys()}
        return self._surface_group_colors
    
    
    @surface_group_colors.setter
    def surface_group_colors(self, surface_group_colors:dict):
        self._surface_group_colors = surface_group_colors
    
    
    @property
    def face_colors(self):
        """
        Generate a face color mapping based on surface groups.
        
        Returns
        -------
        dict
            A dictionary where keys are face indices and values are Color objects corresponding to their surface group.
        """
        group_colors = self.surface_group_colors
        facecolor = {fkey: group_colors[self.acoustic_geometry.face_attribute(fkey, 'surface_group')] for fkey in self.acoustic_geometry.faces()}
        return facecolor
    
    
    @classmethod
    def acoustic_object_from_obj(cls, obj_path:str, name:str=None):
        """
        Create an AcousticObject from an OBJ file.
        
        Parameters
        ----------
        obj_path : str
            The file path to the OBJ file containing the acoustic geometry.
        name : str
            The name of the acoustic object. Optional. If not provided, a unique identifier will be generated.
        Returns
        -------
        AcousticObject
            An instance of AcousticObject created from the OBJ file.
        """
        
        obj = OBJ(obj_path)

        surface_groups = defaultdict(list)
        for group_name, group_items in obj.groups.items():
            for item in group_items:
                surface_groups[group_name].append(item[1])
                
        mesh = Mesh.from_obj(obj_path)
        acoustic_object = cls(acoustic_geometry=mesh, surface_groups=surface_groups, name=name)
        return acoustic_object
    
    
    def display_surface_groups(self):
        viewer = Viewer()
        group_colors = self.surface_group_colors
        print(self.surface_groups)
        print(group_colors)
        print(self.face_colors)
        
        for fkey in self.acoustic_geometry.faces():
            f_centroid = self.acoustic_geometry.face_centroid(fkey)
            surface_group = self.acoustic_geometry.face_attribute(fkey, 'surface_group')
            tag = Tag(surface_group, f_centroid, height=30, color=group_colors[surface_group])
            viewer.scene.add(tag, name=f"tag_{fkey}")
        
        viewer.scene.add(self.acoustic_geometry, name=self.name, show_points=True, facecolor=self.face_colors)
        viewer.show()
        
    
    def print_surface_groups(self, print_fkeys=False):
        print("Surface groups:")
        if not print_fkeys:
            for group_name in self.surface_groups.keys():
                print(f"  {group_name}: {len(self.surface_groups[group_name])} faces")
        else:
            for group_name, face_indices in self.surface_groups.items():
                print(f"  {group_name}: {len(self.surface_groups[group_name])} faces {face_indices}")
       
        
class AcousticObjectsLibrary(AcousticObject):
    """
    A library of predefined acoustic objects.
    """

    @classmethod
    def shoebox_room(cls, length:float, width:float, height:float, sources:list=None, name:str=None):
        """
        Create a shoebox room acoustic object.
        
        Parameters
        ----------
        length : float
            The length of the shoebox room.
        width : float
            The width of the shoebox room.
        height : float
            The height of the shoebox room.
        sources : list
            A list of Source objects to be added to the room. Optional.
        name : str
            The name of the acoustic object. Optional. If not provided, a name will be generated using the room's dimensions.
        Returns
        -------
        AcousticObject
        An instance of AcousticObject representing the shoebox room.
        """
        
        box = Box(length, width, height)
        f1 = box.frame
        f2 = Frame([box.xmax, box.ymax, box.zmax])
        T = Transformation.from_frame_to_frame(f1, f2)
        box.transform(T)
        
        if name is None:
            name = f"shoebox_room_{length}x{width}x{height}"
        acoustic_geometry = Mesh.from_shape(box)
        
        # Set surface groups attribute
        surface_groups = {
            'floor': [0],
            'left_wall': [1],
            'front_wall': [2],
            'right_wall': [3],
            'back_wall': [4],
            'ceiling': [5],
        }
        
        acoustic_object = cls(acoustic_geometry=acoustic_geometry, surface_groups=surface_groups, name=name)

        if sources:
            acoustic_object.sources=sources

        return acoustic_object

