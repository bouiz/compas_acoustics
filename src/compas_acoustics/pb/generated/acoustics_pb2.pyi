from compas_pb.generated import datastructures_pb2 as _datastructures_pb2
from compas_pb.generated import geometry_pb2 as _geometry_pb2
from compas_pb.generated import message_pb2 as _message_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AcousticMaterialData(_message.Message):
    __slots__ = ("guid", "name", "description", "category", "absorption_coefficients", "scattering_coefficients", "transmission_coefficients", "data_origin")
    GUID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    ABSORPTION_COEFFICIENTS_FIELD_NUMBER: _ClassVar[int]
    SCATTERING_COEFFICIENTS_FIELD_NUMBER: _ClassVar[int]
    TRANSMISSION_COEFFICIENTS_FIELD_NUMBER: _ClassVar[int]
    DATA_ORIGIN_FIELD_NUMBER: _ClassVar[int]
    guid: str
    name: str
    description: str
    category: str
    absorption_coefficients: _containers.RepeatedScalarFieldContainer[float]
    scattering_coefficients: _containers.RepeatedScalarFieldContainer[float]
    transmission_coefficients: _containers.RepeatedScalarFieldContainer[float]
    data_origin: str
    def __init__(self, guid: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., category: _Optional[str] = ..., absorption_coefficients: _Optional[_Iterable[float]] = ..., scattering_coefficients: _Optional[_Iterable[float]] = ..., transmission_coefficients: _Optional[_Iterable[float]] = ..., data_origin: _Optional[str] = ...) -> None: ...

class SourceData(_message.Message):
    __slots__ = ("guid", "name", "point", "directivity", "direction", "sound_file", "spl", "description")
    GUID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    POINT_FIELD_NUMBER: _ClassVar[int]
    DIRECTIVITY_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    SOUND_FILE_FIELD_NUMBER: _ClassVar[int]
    SPL_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    guid: str
    name: str
    point: _geometry_pb2.PointData
    directivity: str
    direction: _geometry_pb2.VectorData
    sound_file: str
    spl: _containers.RepeatedScalarFieldContainer[float]
    description: str
    def __init__(self, guid: _Optional[str] = ..., name: _Optional[str] = ..., point: _Optional[_Union[_geometry_pb2.PointData, _Mapping]] = ..., directivity: _Optional[str] = ..., direction: _Optional[_Union[_geometry_pb2.VectorData, _Mapping]] = ..., sound_file: _Optional[str] = ..., spl: _Optional[_Iterable[float]] = ..., description: _Optional[str] = ...) -> None: ...

class ReceiverData(_message.Message):
    __slots__ = ("guid", "name", "point", "direction", "up", "radius")
    GUID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    POINT_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    UP_FIELD_NUMBER: _ClassVar[int]
    RADIUS_FIELD_NUMBER: _ClassVar[int]
    guid: str
    name: str
    point: _geometry_pb2.PointData
    direction: _geometry_pb2.VectorData
    up: _geometry_pb2.VectorData
    radius: float
    def __init__(self, guid: _Optional[str] = ..., name: _Optional[str] = ..., point: _Optional[_Union[_geometry_pb2.PointData, _Mapping]] = ..., direction: _Optional[_Union[_geometry_pb2.VectorData, _Mapping]] = ..., up: _Optional[_Union[_geometry_pb2.VectorData, _Mapping]] = ..., radius: _Optional[float] = ...) -> None: ...

class AcousticObjectData(_message.Message):
    __slots__ = ("guid", "name", "mesh", "polyline", "material", "position", "role", "sources")
    GUID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    MESH_FIELD_NUMBER: _ClassVar[int]
    POLYLINE_FIELD_NUMBER: _ClassVar[int]
    MATERIAL_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    guid: str
    name: str
    mesh: _datastructures_pb2.MeshData
    polyline: _geometry_pb2.PolylineData
    material: AcousticMaterialData
    position: _geometry_pb2.PointData
    role: str
    sources: _containers.RepeatedCompositeFieldContainer[SourceData]
    def __init__(self, guid: _Optional[str] = ..., name: _Optional[str] = ..., mesh: _Optional[_Union[_datastructures_pb2.MeshData, _Mapping]] = ..., polyline: _Optional[_Union[_geometry_pb2.PolylineData, _Mapping]] = ..., material: _Optional[_Union[AcousticMaterialData, _Mapping]] = ..., position: _Optional[_Union[_geometry_pb2.PointData, _Mapping]] = ..., role: _Optional[str] = ..., sources: _Optional[_Iterable[_Union[SourceData, _Mapping]]] = ...) -> None: ...

class SceneData(_message.Message):
    __slots__ = ("guid", "name", "description", "sampling_rate", "acoustic_objects", "sources", "receivers")
    GUID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_RATE_FIELD_NUMBER: _ClassVar[int]
    ACOUSTIC_OBJECTS_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    RECEIVERS_FIELD_NUMBER: _ClassVar[int]
    guid: str
    name: str
    description: str
    sampling_rate: int
    acoustic_objects: _containers.RepeatedCompositeFieldContainer[AcousticObjectData]
    sources: _containers.RepeatedCompositeFieldContainer[SourceData]
    receivers: _containers.RepeatedCompositeFieldContainer[ReceiverData]
    def __init__(self, guid: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., sampling_rate: _Optional[int] = ..., acoustic_objects: _Optional[_Iterable[_Union[AcousticObjectData, _Mapping]]] = ..., sources: _Optional[_Iterable[_Union[SourceData, _Mapping]]] = ..., receivers: _Optional[_Iterable[_Union[ReceiverData, _Mapping]]] = ...) -> None: ...

class ImpulseResponseData(_message.Message):
    __slots__ = ("guid", "name", "source", "receiver", "samples", "shape", "sampling_rate", "normalized", "generation", "ambisonic_order", "normalization", "channel_order")
    GUID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    RECEIVER_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_RATE_FIELD_NUMBER: _ClassVar[int]
    NORMALIZED_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    AMBISONIC_ORDER_FIELD_NUMBER: _ClassVar[int]
    NORMALIZATION_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_ORDER_FIELD_NUMBER: _ClassVar[int]
    guid: str
    name: str
    source: SourceData
    receiver: ReceiverData
    samples: _containers.RepeatedScalarFieldContainer[float]
    shape: _containers.RepeatedScalarFieldContainer[int]
    sampling_rate: int
    normalized: bool
    generation: _message_pb2.AnyData
    ambisonic_order: int
    normalization: str
    channel_order: str
    def __init__(self, guid: _Optional[str] = ..., name: _Optional[str] = ..., source: _Optional[_Union[SourceData, _Mapping]] = ..., receiver: _Optional[_Union[ReceiverData, _Mapping]] = ..., samples: _Optional[_Iterable[float]] = ..., shape: _Optional[_Iterable[int]] = ..., sampling_rate: _Optional[int] = ..., normalized: bool = ..., generation: _Optional[_Union[_message_pb2.AnyData, _Mapping]] = ..., ambisonic_order: _Optional[int] = ..., normalization: _Optional[str] = ..., channel_order: _Optional[str] = ...) -> None: ...

class AudioData(_message.Message):
    __slots__ = ("guid", "samples", "shape", "sampling_rate", "format")
    GUID_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_RATE_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    guid: str
    samples: _containers.RepeatedScalarFieldContainer[float]
    shape: _containers.RepeatedScalarFieldContainer[int]
    sampling_rate: int
    format: str
    def __init__(self, guid: _Optional[str] = ..., samples: _Optional[_Iterable[float]] = ..., shape: _Optional[_Iterable[int]] = ..., sampling_rate: _Optional[int] = ..., format: _Optional[str] = ...) -> None: ...

class RayData(_message.Message):
    __slots__ = ("guid", "name", "points", "source_index", "ray_index", "direction", "distance_offset", "scattered_reflections")
    GUID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_INDEX_FIELD_NUMBER: _ClassVar[int]
    RAY_INDEX_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    SCATTERED_REFLECTIONS_FIELD_NUMBER: _ClassVar[int]
    guid: str
    name: str
    points: _containers.RepeatedScalarFieldContainer[float]
    source_index: int
    ray_index: int
    direction: _geometry_pb2.VectorData
    distance_offset: float
    scattered_reflections: _containers.RepeatedCompositeFieldContainer[ScatteredRayData]
    def __init__(self, guid: _Optional[str] = ..., name: _Optional[str] = ..., points: _Optional[_Iterable[float]] = ..., source_index: _Optional[int] = ..., ray_index: _Optional[int] = ..., direction: _Optional[_Union[_geometry_pb2.VectorData, _Mapping]] = ..., distance_offset: _Optional[float] = ..., scattered_reflections: _Optional[_Iterable[_Union[ScatteredRayData, _Mapping]]] = ...) -> None: ...

class ScatteredRayData(_message.Message):
    __slots__ = ("guid", "name", "points", "source_index", "ray_index", "direction", "distance_offset", "parent_ray_index", "energy", "origin_reflection_order")
    GUID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_INDEX_FIELD_NUMBER: _ClassVar[int]
    RAY_INDEX_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    PARENT_RAY_INDEX_FIELD_NUMBER: _ClassVar[int]
    ENERGY_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_REFLECTION_ORDER_FIELD_NUMBER: _ClassVar[int]
    guid: str
    name: str
    points: _containers.RepeatedScalarFieldContainer[float]
    source_index: int
    ray_index: int
    direction: _geometry_pb2.VectorData
    distance_offset: float
    parent_ray_index: int
    energy: float
    origin_reflection_order: int
    def __init__(self, guid: _Optional[str] = ..., name: _Optional[str] = ..., points: _Optional[_Iterable[float]] = ..., source_index: _Optional[int] = ..., ray_index: _Optional[int] = ..., direction: _Optional[_Union[_geometry_pb2.VectorData, _Mapping]] = ..., distance_offset: _Optional[float] = ..., parent_ray_index: _Optional[int] = ..., energy: _Optional[float] = ..., origin_reflection_order: _Optional[int] = ...) -> None: ...

class SpecularPathData(_message.Message):
    __slots__ = ("guid", "name", "points", "source_index", "ray_index", "direction", "distance_offset", "order", "walls", "image", "receiver_index")
    GUID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_INDEX_FIELD_NUMBER: _ClassVar[int]
    RAY_INDEX_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    WALLS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    RECEIVER_INDEX_FIELD_NUMBER: _ClassVar[int]
    guid: str
    name: str
    points: _containers.RepeatedScalarFieldContainer[float]
    source_index: int
    ray_index: int
    direction: _geometry_pb2.VectorData
    distance_offset: float
    order: int
    walls: _containers.RepeatedScalarFieldContainer[int]
    image: _geometry_pb2.PointData
    receiver_index: int
    def __init__(self, guid: _Optional[str] = ..., name: _Optional[str] = ..., points: _Optional[_Iterable[float]] = ..., source_index: _Optional[int] = ..., ray_index: _Optional[int] = ..., direction: _Optional[_Union[_geometry_pb2.VectorData, _Mapping]] = ..., distance_offset: _Optional[float] = ..., order: _Optional[int] = ..., walls: _Optional[_Iterable[int]] = ..., image: _Optional[_Union[_geometry_pb2.PointData, _Mapping]] = ..., receiver_index: _Optional[int] = ...) -> None: ...

class RayFieldData(_message.Message):
    __slots__ = ("guid", "name", "rays", "sources")
    class SourcesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: SourceData
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[SourceData, _Mapping]] = ...) -> None: ...
    GUID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    RAYS_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    guid: str
    name: str
    rays: _containers.RepeatedCompositeFieldContainer[RayData]
    sources: _containers.MessageMap[int, SourceData]
    def __init__(self, guid: _Optional[str] = ..., name: _Optional[str] = ..., rays: _Optional[_Iterable[_Union[RayData, _Mapping]]] = ..., sources: _Optional[_Mapping[int, SourceData]] = ...) -> None: ...

class SpecularReflectionsData(_message.Message):
    __slots__ = ("guid", "name", "paths", "sources")
    class SourcesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: SourceData
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[SourceData, _Mapping]] = ...) -> None: ...
    GUID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PATHS_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    guid: str
    name: str
    paths: _containers.RepeatedCompositeFieldContainer[SpecularPathData]
    sources: _containers.MessageMap[int, SourceData]
    def __init__(self, guid: _Optional[str] = ..., name: _Optional[str] = ..., paths: _Optional[_Iterable[_Union[SpecularPathData, _Mapping]]] = ..., sources: _Optional[_Mapping[int, SourceData]] = ...) -> None: ...
