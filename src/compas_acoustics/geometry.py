from compas.data import Data
from compas.datastructures import Mesh
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Transformation

from compas_acoustics import Source
from compas_acoustics.acoustic_material import AcousticMaterial

__all__ = ['AcousticObject', 'AcousticObjectsGroup', 'AcousticObjectsLibrary']

#: Roles a surface can play: part of the closed room, or a free-standing object inside it.
ROLES = ('boundary', 'obstacle')


def _checked_role(role):
    if role is not None and role not in ROLES:
        raise ValueError(f"role must be None or one of {ROLES}, got {role!r}")
    return role


class AcousticObject(Data):
    
    def __init__(self, acoustic_geometry:Mesh, material: AcousticMaterial, x:float=0, y:float=0, z:float=0, name:str=None, role:str=None):
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
            ``'boundary'`` (part of the closed room), ``'obstacle'`` (a free-standing
            object inside it), or None to take the enclosing group's role, else ``'boundary'``.
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


    def iter_acoustic_objects(self):
        """Yield the leaf acoustic objects contained in this element.

        An :class:`AcousticObject` is a leaf, so it yields itself. This mirrors
        :meth:`AcousticObjectsGroup.iter_acoustic_objects`, allowing consumers to
        treat single objects and groups uniformly.

        Yields
        ------
        :class:`AcousticObject`
        """
        yield self


    @property
    def role(self):
        return self._role


    @role.setter
    def role(self, role):
        self._role = _checked_role(role)


    def iter_acoustic_objects_with_roles(self, inherited=None):
        """Yield ``(self, role)``: its own role, else ``inherited``, else ``'boundary'``."""
        yield self, self.role or inherited or 'boundary'


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
       

class AcousticObjectsGroup(Data):
    """
    A composite that groups :class:`AcousticObject` instances (and, optionally,
    nested :class:`AcousticObjectsGroup` instances).

    A group has no geometry or material of its own; it is a container that lets
    a single logical entity (for example, a room with per-surface materials) be
    expressed as several acoustic objects.

    Parameters
    ----------
    name : str
        The name of the group. Optional.
    acoustic_objects : list
        The acoustic objects (or nested groups) contained in this group. Optional.
    role : str
        ``'boundary'``, ``'obstacle'``, or None to take the enclosing group's role.
        A role set on a contained object or nested group overrides it.
    """

    def __init__(self, name:str=None, acoustic_objects:list=None, role:str=None):
        super().__init__(name=name)
        self.acoustic_objects = acoustic_objects if acoustic_objects is not None else []
        self.role = role

    @property
    def id(self):
        return self.guid

    @property
    def acoustic_objects(self):
        return self._acoustic_objects
    
    
    @acoustic_objects.setter
    def acoustic_objects(self, acoustic_objects:list):
        self._acoustic_objects = acoustic_objects

    @property
    def sources(self):
        """All sources aggregated from the contained (leaf) acoustic objects."""
        all_sources = []
        for acoustic_object in self.iter_acoustic_objects():
            all_sources.extend(acoustic_object.sources)
        return all_sources

    def iter_acoustic_objects(self):
        """Yield the leaf :class:`AcousticObject` instances in this group.

        Nested :class:`AcousticObjectsGroup` instances are recursed into, so only
        leaf acoustic objects are yielded.

        Yields
        ------
        :class:`AcousticObject`
        """
        for item in self.acoustic_objects:
            yield from item.iter_acoustic_objects()

    @property
    def role(self):
        return self._role

    @role.setter
    def role(self, role):
        self._role = _checked_role(role)

    def iter_acoustic_objects_with_roles(self, inherited=None):
        """Yield ``(leaf, role)`` for every leaf, the nearest explicit role winning."""
        role = self.role or inherited
        for item in self.acoustic_objects:
            yield from item.iter_acoustic_objects_with_roles(role)

    def add_acoustic_object(self, acoustic_object:AcousticObject):
        self.acoustic_objects.append(acoustic_object)

    def remove_acoustic_object(self, acoustic_object:AcousticObject):
        self.acoustic_objects.remove(acoustic_object)

    def remove_acoustic_object_by_name(self, name:str):
        self.acoustic_objects.remove(self.get_acoustic_object_by_name(name))

    def get_acoustic_object_by_name(self, name:str):
        return next((acoustic_object for acoustic_object in self.acoustic_objects if acoustic_object.name == name), None)


class AcousticObjectsLibrary:
    """
    A factory of predefined, commonly used acoustic objects.

    This class is never instantiated; it is a namespace of classmethods that
    construct and return :class:`AcousticObject` or :class:`AcousticObjectsGroup`
    instances, whichever fits the shape.
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
    def from_obj(cls, filepath, materials, roles=None, name=None):
        """
        Read a Wavefront OBJ file as a group with one :class:`AcousticObject` per OBJ group.

        Each ``g`` group of the file (Rhino writes one per layer or object) becomes an
        acoustic object named after it; faces before any ``g`` line belong to ``"default"``.
        Coordinates are taken as metres. Vertices shared between groups are welded later,
        by :class:`compas_acoustics.RoomGeometry`. The group-to-material convention is
        hybrid-rir's ``MeshGeometry.from_obj``.

        Parameters
        ----------
        filepath : str
            The OBJ file.
        materials : :class:`AcousticMaterial` or dict
            One material for every group, or a dict keyed by group name. A group
            without a material raises a ``ValueError`` naming it.
        roles : str or dict, optional
            ``'boundary'`` or ``'obstacle'`` for the whole file (``'obstacle'`` for
            furniture), or a dict keyed by group name; groups not in it take the
            enclosing group's role, else ``'boundary'``.
        name : str, optional
            The name of the returned group; defaults to the file name.

        Returns
        -------
        :class:`AcousticObjectsGroup`
        """
        import os

        points = []
        groups = {}
        group = "default"
        with open(filepath) as f:
            for line in f:
                tokens = line.split()
                if not tokens:
                    continue
                if tokens[0] == "v":
                    points.append([float(c) for c in tokens[1:4]])
                elif tokens[0] == "g" and len(tokens) > 1:
                    group = " ".join(tokens[1:])
                elif tokens[0] == "f":
                    # 1-based, possibly v/vt/vn or negative (relative to the vertices read so far)
                    indices = [int(token.split("/")[0]) for token in tokens[1:]]
                    groups.setdefault(group, []).append([i - 1 if i > 0 else len(points) + i for i in indices])

        if name is None:
            name = os.path.splitext(os.path.basename(filepath))[0]
        whole_file_role = roles if isinstance(roles, str) or roles is None else None
        objects = []
        for group, faces in groups.items():
            if isinstance(materials, AcousticMaterial):
                material = materials
            elif group in materials:
                material = materials[group]
            else:
                raise ValueError(f"No material for OBJ group '{group}' of {filepath}; groups: {sorted(groups)}")
            used = sorted({i for face in faces for i in face})
            index = {old: new for new, old in enumerate(used)}
            mesh = Mesh.from_vertices_and_faces([points[i] for i in used], [[index[i] for i in face] for face in faces])
            role = roles.get(group) if isinstance(roles, dict) else None
            objects.append(AcousticObject(mesh, material, name=f"{name}_{group}", role=role))
        return AcousticObjectsGroup(name=name, acoustic_objects=objects, role=whole_file_role)

    @classmethod
    def shoebox_room(cls, length:float, width:float, height:float, materials=None, sources:list=None, name:str=None):
        """
        Create a shoebox room as a group of six single-material surfaces.

        Each surface (floor, walls, ceiling) becomes its own
        :class:`AcousticObject` with its own material, allowing distinct acoustic
        materials per surface.

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
            The name of the group. If not provided, a name is generated using the
            room's dimensions.

        Returns
        -------
        :class:`AcousticObjectsGroup`
            A group containing one :class:`AcousticObject` per surface.
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

        acoustic_objects = []
        for fkey, surface_name in cls.SHOEBOX_SURFACES.items():
            coords = [box_mesh.vertex_coordinates(vkey) for vkey in box_mesh.face_vertices(fkey)]
            surface_mesh = Mesh.from_vertices_and_faces(coords, [list(range(len(coords)))])
            surface_material = cls._resolve_surface_material(materials, surface_name, default_material)
            surface_object = AcousticObject(
                acoustic_geometry=surface_mesh,
                material=surface_material,
                name=f"{name}_{surface_name}",
            )
            acoustic_objects.append(surface_object)

        room = AcousticObjectsGroup(name=name, acoustic_objects=acoustic_objects)

        if sources:
            floor = room.get_acoustic_object_by_name(f"{name}_floor")
            floor.sources = sources

        return room

