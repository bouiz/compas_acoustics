import os

from compas.data import Data
from compas.datastructures import Mesh
from compas.files import OBJ
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Transformation

from compas_acoustics import Source
from compas_acoustics.acoustic_material import AcousticMaterial

__all__ = ['AcousticObject', 'AcousticObjectsLibrary']

#: Roles a surface can play: part of the closed room, or a free-standing object inside it.
ROLES = ('boundary', 'obstacle')


def _checked_role(role):
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    return role


class AcousticObject(Data):

    def __init__(self, acoustic_geometry:Mesh, material:AcousticMaterial, x:float=0, y:float=0, z:float=0, name:str=None, role:str='boundary'):
        """Initialize Acoustic object

        Parameters
        ----------
        acoustic_geometry : :class:`compas.datastructures.Mesh`
            A COMPAS Mesh representing the acoustic geometry.
        material : :class:`compas_acoustics.acoustic_material.AcousticMaterial`
            The acoustic material assigned to the acoustic object.
        x : float
            The x-coordinate of the geometry's reference point.
        y : float
            The y-coordinate of the geometry's reference point.
        z : float
            The z-coordinate of the geometry's reference point.
        name : str
            The name of the acoustic object. Optional. If not provided, a unique identifier will be generated.
        role : str
            ``'boundary'`` (part of the closed room) or ``'obstacle'`` (a free-standing
            object inside it).
        """
        super().__init__(name=name)
        self._x = x
        self._y = y
        self._z = z
        self.acoustic_geometry = acoustic_geometry
        self.material = material
        self.role = role
        self.sources = []
        self.visualisation_geometry = None


    @classmethod
    def from_obj(cls, filepath, materials, roles='boundary', name=None):
        """One acoustic object per ``g`` group of a Wavefront OBJ file.

        Rhino writes one group per layer or object; faces before any ``g`` line are in
        ``'default'``. Each object is named ``'<name>_<group>'``. Coordinates are taken as
        metres. :class:`compas.files.OBJ` welds the vertices to
        :attr:`compas.tolerance.TOL.precision` and faces that collapse are dropped.

        Parameters
        ----------
        filepath : str
            The OBJ file.
        materials : :class:`AcousticMaterial` or dict
            One material for every group, or a dict keyed by group name. A group
            without a material raises a ``ValueError`` naming it.
        roles : str or dict, optional
            ``'boundary'`` or ``'obstacle'`` for the whole file (``'obstacle'`` for
            furniture), or a dict keyed by group name; groups not in it are boundaries.
        name : str, optional
            The prefix of the objects' names; defaults to the file name.

        Returns
        -------
        list of :class:`AcousticObject`
        """
        obj = OBJ(filepath)
        obj.read()
        # obj.groups keeps only the last run of a group name that recurs, as in Rhino exports
        face_groups, group = [], 'default'
        with open(filepath) as f:
            for line in f:
                tokens = line.split()
                if tokens[:1] == ['g']:
                    group = " ".join(tokens[1:])
                elif tokens[:1] == ['f'] and len(tokens) > 3:
                    face_groups.append(group)
        groups = {}
        for face, group in zip(obj.faces, face_groups):
            groups.setdefault(group, []).append(face)
        if name is None:
            name = os.path.splitext(os.path.basename(filepath))[0]
        objects = []
        for group, faces in groups.items():
            if isinstance(materials, AcousticMaterial):
                material = materials
            elif group in materials:
                material = materials[group]
            else:
                raise ValueError(f"No material for OBJ group '{group}' of {filepath}; groups: {sorted(groups)}")
            role = roles.get(group, 'boundary') if isinstance(roles, dict) else roles
            mesh = Mesh.from_vertices_and_faces(obj.vertices, faces)
            mesh.remove_unused_vertices()
            objects.append(cls(mesh, material, name=f"{name}_{group}", role=role))
        return objects


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
    def sources(self):
        return self._sources
    
    
    @sources.setter
    def sources(self, sources:list):
        self._sources = sources
    
    
    def add_source(self, source:Source):
        self.sources.append(source)


    @property
    def role(self):
        return self._role


    @role.setter
    def role(self, role):
        self._role = _checked_role(role)


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
       

class AcousticObjectsLibrary:
    """
    A factory of predefined, commonly used acoustic objects.

    This class is never instantiated; it is a namespace of classmethods that
    construct and return :class:`AcousticObject` instances.
    """

    #: Face index -> surface name for a shoebox room built from :class:`compas.geometry.Box`.
    SHOEBOX_SURFACES = {
        0: 'floor',
        1: 'left_wall',
        2: 'front_wall',
        3: 'right_wall',
        4: 'back_wall',
        5: 'ceiling',
    }

    @staticmethod
    def _default_material():
        return AcousticMaterial(
            name="default",
            absorption_coefficients=0.1,
            scattering_coefficients=0.1,
            description="Default material",
        )

    @classmethod
    def _resolve_surface_material(cls, materials, surface_name, default):
        """Resolve the material for a given surface from the ``materials`` argument.

        ``materials`` may be ``None`` (use ``default`` everywhere), a single
        :class:`AcousticMaterial` (used for every surface), or a dict keyed by
        surface name (missing surfaces fall back to ``default``).
        """
        if materials is None:
            return default
        if isinstance(materials, AcousticMaterial):
            return materials
        if isinstance(materials, dict):
            return materials.get(surface_name, default)
        raise TypeError(
            "materials must be None, an AcousticMaterial, or a dict keyed by surface name."
        )

    @classmethod
    def shoebox_room(cls, length:float, width:float, height:float, materials=None, sources:list=None, name:str=None):
        """
        Create a shoebox room as six single-material surfaces.

        Each surface (floor, walls, ceiling) is its own :class:`AcousticObject`
        with its own material.

        Parameters
        ----------
        length : float
            The length of the shoebox room.
        width : float
            The width of the shoebox room.
        height : float
            The height of the shoebox room.
        materials : dict or :class:`AcousticMaterial`, optional
            Per-surface materials. May be a dict keyed by surface name
            (``'floor'``, ``'left_wall'``, ``'front_wall'``, ``'right_wall'``,
            ``'back_wall'``, ``'ceiling'``), a single material applied to all
            surfaces, or ``None`` to use a default material everywhere.
        sources : list, optional
            A list of Source objects to be added to the room (attached to the
            floor surface).
        name : str, optional
            The prefix of the surfaces' names. If not provided, a name is generated
            using the room's dimensions.

        Returns
        -------
        list of :class:`AcousticObject`
            One per surface, in the order floor, left, front, right and back wall, ceiling.
        """

        box = Box(length, width, height)
        f1 = box.frame
        f2 = Frame([box.xmax, box.ymax, box.zmax])
        T = Transformation.from_frame_to_frame(f1, f2)
        box.transform(T)

        if name is None:
            name = f"shoebox_room_{length}x{width}x{height}"
        box_mesh = Mesh.from_shape(box)

        default_material = cls._default_material()
        room = []
        for fkey, surface_name in cls.SHOEBOX_SURFACES.items():
            surface_mesh = Mesh.from_vertices_and_faces(box_mesh.face_coordinates(fkey), [[0, 1, 2, 3]])
            surface_material = cls._resolve_surface_material(materials, surface_name, default_material)
            room.append(AcousticObject(surface_mesh, surface_material, name=f"{name}_{surface_name}"))
        if sources:
            room[0].sources = sources
        return room

