import os

from compas.data import Data
from compas.datastructures import Mesh
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Transformation

from compas_acoustics import Source
from compas_acoustics.defaults import _PackageDtype
from compas_acoustics.materials import AcousticMaterial

__all__ = ['AcousticObject', 'AcousticObjectsLibrary']

#: Roles a surface can play: part of the closed room, or a free-standing object inside it.
ROLES = ('boundary', 'obstacle')


def _checked_role(role):
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    return role


def _objects_by_group(objects, names):
    """The objects read from each OBJ group in ``names``; a name none of them has raises."""
    by_group = {}
    for acoustic_object in objects:
        by_group.setdefault(acoustic_object.group, []).append(acoustic_object)
    unknown = sorted(set(names) - set(by_group))
    if unknown:
        known = sorted(group for group in by_group if group is not None)
        raise ValueError(f"No OBJ group {unknown}; groups: {known}")
    return {name: by_group[name] for name in names}


class AcousticObject(_PackageDtype, Data):

    def __init__(self, acoustic_geometry:Mesh, material:AcousticMaterial, x:float=0, y:float=0, z:float=0, name:str=None, role:str='boundary', group:str=None):
        """Initialize Acoustic object

        Parameters
        ----------
        acoustic_geometry : :class:`compas.datastructures.Mesh`
            A COMPAS Mesh representing the acoustic geometry.
        material : :class:`compas_acoustics.AcousticMaterial`
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
        group : str
            The OBJ group the object was read from, which :meth:`assign_roles_by_group` and
            :meth:`assign_materials_by_group` look it up by. Optional; not serialised.
        """
        super().__init__(name=name)
        self._x = x
        self._y = y
        self._z = z
        self.acoustic_geometry = acoustic_geometry
        self.material = material
        self.role = role
        self.group = group
        self.sources = []
        self.visualisation_geometry = None


    @classmethod
    def from_obj(cls, filepath, materials=None, name=None):
        """One acoustic object per ``g`` group of a Wavefront OBJ file.

        Rhino writes one group per layer or object; faces before any ``g`` line are in
        ``'default'``. A file without any ``g`` line takes its groups from its ``o``
        (object) lines instead; where it has both, the ``g`` lines win. A group that
        recurs later in the file is merged into the first. Each object is named
        ``'<name>_<group>'``. Coordinates are taken as metres and are not rounded, since
        :class:`RoomGeometry` welds them. Faces may use ``i``, ``i/j``, ``i//k`` or
        ``i/j/k`` indices, negative ones counted from the vertices read so far; faces
        with fewer than three distinct vertices are dropped.

        Every object is a boundary of the room; mark furniture as obstacles with
        :meth:`assign_roles_by_group`.

        Parameters
        ----------
        filepath : str
            The OBJ file.
        materials : :class:`AcousticMaterial` or dict, optional
            One material for every group, or a dict keyed by group name, as
            :meth:`assign_materials_by_group` takes it. Groups without one have no
            material until :meth:`assign_materials_by_group` gives them one; a scene
            with an object that has none cannot be simulated.
        name : str, optional
            The prefix of the objects' names; defaults to the file name.

        Returns
        -------
        list of :class:`AcousticObject`
        """
        vertices, faces, face_groups, face_objects = [], [], [], []
        group = obj = None
        has_groups = False
        with open(filepath, encoding="utf-8-sig") as f:
            for line in f:
                tokens = line.split()
                if not tokens:
                    continue
                if tokens[0] == 'v':
                    vertices.append([float(value) for value in tokens[1:4]])
                elif tokens[0] == 'g':
                    group = " ".join(tokens[1:]) or None
                    has_groups = True
                elif tokens[0] == 'o':
                    obj = " ".join(tokens[1:]) or None
                elif tokens[0] == 'f':
                    indices = []
                    for token in tokens[1:]:
                        index = int(token.split('/')[0])
                        indices.append(len(vertices) + index if index < 0 else index - 1)
                    if len(set(indices)) >= 3:
                        faces.append(indices)
                        face_groups.append(group)
                        face_objects.append(obj)
        names = face_groups if has_groups else face_objects
        groups = {}
        for face, group in zip(faces, names):
            groups.setdefault(group or 'default', []).append(face)
        if name is None:
            name = os.path.splitext(os.path.basename(filepath))[0]
        objects = []
        material = None if isinstance(materials, dict) else materials
        for group, faces in groups.items():
            mesh = Mesh.from_vertices_and_faces(vertices, faces)
            mesh.remove_unused_vertices()
            objects.append(cls(mesh, material, name=f"{name}_{group}", group=group))
        if isinstance(materials, dict):
            cls.assign_materials_by_group(objects, materials)
        return objects


    @staticmethod
    def assign_roles_by_group(objects, roles):
        """Set the role of every object read from the named OBJ groups.

        Parameters
        ----------
        objects : list of :class:`AcousticObject`
            Objects from :meth:`from_obj`, from one or several files.
        roles : dict
            ``'boundary'`` or ``'obstacle'`` by OBJ group name. A name that none of the
            objects was read from raises a ``ValueError`` listing the groups they have.

        Returns
        -------
        list of :class:`AcousticObject`
            ``objects``.

        Examples
        --------
        >>> furniture = {'chairs': 'obstacle', 'tables': 'obstacle'}
        >>> objects = AcousticObject.from_obj('office.obj', materials)  # doctest: +SKIP
        >>> AcousticObject.assign_roles_by_group(objects, furniture)  # doctest: +SKIP
        """
        for group, members in _objects_by_group(objects, roles).items():
            for acoustic_object in members:
                acoustic_object.role = roles[group]
        return objects


    @staticmethod
    def assign_materials_by_group(objects, materials):
        """Set the material of every object read from the named OBJ groups.

        Parameters
        ----------
        objects : list of :class:`AcousticObject`
            Objects from :meth:`from_obj`, from one or several files.
        materials : dict
            :class:`AcousticMaterial` by OBJ group name. A name that none of the objects
            was read from raises a ``ValueError`` listing the groups they have; groups
            not in it keep their material.

        Returns
        -------
        list of :class:`AcousticObject`
            ``objects``.

        Examples
        --------
        >>> objects = AcousticObject.from_obj('office.obj')  # doctest: +SKIP
        >>> AcousticObject.assign_materials_by_group(objects, {'chairs': upholstery, 'glass': glass})  # doctest: +SKIP
        """
        for group, members in _objects_by_group(objects, materials).items():
            for acoustic_object in members:
                acoustic_object.material = materials[group]
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
    def material(self):
        return self._material


    @material.setter
    def material(self, material):
        if material is not None and not isinstance(material, AcousticMaterial):
            raise TypeError(f"material must be an AcousticMaterial or None, got {type(material).__name__}")
        self._material = material


    @property
    def role(self):
        return self._role


    @role.setter
    def role(self, role):
        self._role = _checked_role(role)


    @property
    def __data__(self):
        return {
            "acoustic_geometry": self.acoustic_geometry,
            "material": self.material,
            "position": self.position,
            "role": self.role,
            "sources": self.sources,
        }


    @classmethod
    def __from_data__(cls, data):
        acoustic_object = cls(data["acoustic_geometry"], data["material"], *data["position"], role=data["role"])
        acoustic_object.sources = data["sources"]
        return acoustic_object


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
        1: 'right_wall',
        2: 'front_wall',
        3: 'left_wall',
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
            One per surface, in the order floor, right, front, left and back wall, ceiling.
            Seen from a receiver at the centre facing +x (y to its left), ``'left_wall'`` is
            the surface at y = width and ``'right_wall'`` the one at y = 0; ``'front_wall'``
            is at x = length.
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

