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
    """A surface of the scene with its material.

    Parameters
    ----------
    acoustic_geometry : :class:`compas.datastructures.Mesh`
    material : :class:`compas_acoustics.AcousticMaterial`
    x, y, z : float
        The geometry's reference point.
    name : str, optional
        Default: the class name.
    role : str, optional
        ``'boundary'`` (part of the closed room, the default) or ``'obstacle'`` (free-standing inside it).
    group : str, optional
        The OBJ group the object was read from, for :meth:`assign_roles_by_group` and
        :meth:`assign_materials_by_group`. Not serialised.
    """

    def __init__(self, acoustic_geometry:Mesh, material:AcousticMaterial, x:float=0, y:float=0, z:float=0, name:str=None, role:str='boundary', group:str=None):
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

        Faces before any ``g`` line are in ``'default'``; a file without ``g`` lines is grouped by
        its ``o`` lines. A group that recurs is merged into the first. Objects are named
        ``'<name>_<group>'``. Coordinates are metres, not rounded (:class:`RoomGeometry` welds them).
        Faces may use ``i``, ``i/j``, ``i//k`` or ``i/j/k`` indices, negative ones relative; faces
        with fewer than three distinct vertices are dropped. Every object is a boundary; mark
        furniture with :meth:`assign_roles_by_group`.

        Parameters
        ----------
        filepath : str
        materials : :class:`AcousticMaterial` or dict, optional
            One material for every group, or one per group name as :meth:`assign_materials_by_group`
            takes them. A scene with an object without a material cannot be simulated.
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
            From :meth:`from_obj`, of one or several files.
        roles : dict
            ``'boundary'`` or ``'obstacle'`` by group name. An unknown name raises ``ValueError``.

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
            From :meth:`from_obj`, of one or several files.
        materials : dict
            :class:`AcousticMaterial` by group name; other groups keep theirs. An unknown name raises ``ValueError``.

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
        roles = ('boundary', 'obstacle')
        if role not in roles:
            raise ValueError(f"role must be one of {roles}, got {role!r}")
        self._role = role

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
    """Factories of commonly used acoustic objects."""

    @classmethod
    def shoebox_room(cls, length:float, width:float, height:float, materials=None, sources:list=None, name:str=None):
        """A shoebox room as six surfaces, each its own :class:`AcousticObject`.

        Parameters
        ----------
        length, width, height : float
        materials : dict or :class:`AcousticMaterial`, optional
            One material for every surface, or one per surface name (``'floor'``, ``'left_wall'``,
            ``'front_wall'``, ``'right_wall'``, ``'back_wall'``, ``'ceiling'``). Missing ones:
            absorption and scattering 0.1.
        sources : list, optional
            Sources attached to the floor.
        name : str, optional
            The prefix of the surfaces' names; default from the dimensions.

        Returns
        -------
        list of :class:`AcousticObject`
            Floor, right, front, left and back wall, ceiling. Facing +x from the centre, ``'left_wall'``
            is at y = width, ``'right_wall'`` at y = 0 and ``'front_wall'`` at x = length.
        """
        box = Box(length, width, height)
        box.transform(Transformation.from_frame_to_frame(box.frame, Frame([box.xmax, box.ymax, box.zmax])))
        if name is None:
            name = f"shoebox_room_{length}x{width}x{height}"
        box_mesh = Mesh.from_shape(box)

        default = AcousticMaterial(name="default", absorption_coefficients=0.1, scattering_coefficients=0.1,
                                   description="Default material")
        if not (materials is None or isinstance(materials, (AcousticMaterial, dict))):
            raise TypeError("materials must be None, an AcousticMaterial, or a dict keyed by surface name.")
        room = []
        surfaces = ('floor', 'right_wall', 'front_wall', 'left_wall', 'back_wall', 'ceiling')
        for fkey, surface_name in enumerate(surfaces):
            surface_mesh = Mesh.from_vertices_and_faces(box_mesh.face_coordinates(fkey), [[0, 1, 2, 3]])
            if isinstance(materials, dict):
                material = materials.get(surface_name, default)
            else:
                material = default if materials is None else materials
            room.append(AcousticObject(surface_mesh, material, name=f"{name}_{surface_name}"))
        if sources:
            room[0].sources = sources
        return room
