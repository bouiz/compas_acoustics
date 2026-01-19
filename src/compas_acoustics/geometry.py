from random import random

from collections import defaultdict
from compas.colors import Color
from compas.data import Data
from compas.datastructures import Mesh
from compas.geometry import Box
from compas.files import OBJ

from compas_acoustics import Source


__all__ = ['AcousticObject', 'AcousticObjectsLibrary']

class AcousticObject(Data):
    
    def __init__(self, acoustic_geometry:Mesh, x:float=0, y:float=0, z:float=0):
        """Initialize Acoustic object

        Parameters
        ----------
        acoustic_geometry : :class:`compas.datastructures.Mesh`
            A COMPAS Mesh representing the acoustic geometry.
        x : float
            The x-coordinate of the geometry's reference point.
        y : float
            The y-coordinate of the geometry's reference point.
        z : float
            The z-coordinate of the geometry's reference point.
        """
        
        self._x = x
        self._y = y
        self._z = z
        self.acoustic_geometry = self._prepare_mesh(acoustic_geometry)
        self.surface_groups = {} # {group name : list of face indices}
        self.sources = []
        self.visualisation_geometry = None
        
    
    def _prepare_mesh(self, mesh:Mesh):
        mesh.update_default_face_attributes({'surface_group': 'default'})
        return mesh
       
       
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
        group_colors = {sk: Color(random(), random(), random(), 0.5) for sk in self.surface_groups.keys()}
        return group_colors
    
    
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
    def acoustic_object_from_obj(cls, obj_path:str):
        obj = OBJ(obj_path)

        surface_groups = defaultdict(list)
        for group_name, group_items in obj.groups.items():
            for item in group_items:
                surface_groups[group_name].append(item[1])
                
        mesh = Mesh.from_obj(obj_path)
        acoustic_object = cls(acoustic_geometry=mesh)
        acoustic_object.surface_groups = surface_groups
        return acoustic_object
        
        
class AcousticObjectsLibrary(AcousticObject):
    """
    A library of predefined acoustic objects.
    """

    @classmethod
    def shoebox_room(cls, length:float, width:float, height:float, sources:list=None):
        box = Box(length, width, height)
        acoustic_geometry = Mesh.from_shape(box)
        
        acoustic_object = cls(acoustic_geometry=acoustic_geometry)
        
        # Set surface groups attribute
        surface_groups = {
            'floor': [0],
            'left_wall': [1],
            'front_wall': [2],
            'right_wall': [3],
            'back_wall': [4],
            'ceiling': [5],
        }
        acoustic_object.surface_groups = surface_groups

        if sources:
            acoustic_object.sources=sources

        return acoustic_object

