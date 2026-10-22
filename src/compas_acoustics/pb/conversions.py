"""Converters between compas_acoustics' Data types and their protobuf messages, registered with compas_pb.

Each message mirrors the class's ``__data__``; objects are rebuilt with the class's own ``__from_data__``, so the
classes' validation applies, and keep their guid and name.
"""

from uuid import UUID

import numpy as np
from compas.datastructures import Mesh
from compas.geometry import Polyline
from compas_pb import conversions as _core  # noqa: F401  (registers compas_pb's own types first)
from compas_pb.conversions import mesh_from_pb
from compas_pb.conversions import mesh_to_pb
from compas_pb.conversions import polyline_from_pb
from compas_pb.conversions import polyline_to_pb
from compas_pb.core import _deserialize_any
from compas_pb.core import _serializer_any
from compas_pb.registry import pb_deserializer
from compas_pb.registry import pb_serializer

from compas_acoustics.acoustic_object import AcousticObject
from compas_acoustics.ir import ImpulseResponse
from compas_acoustics.materials import AcousticMaterial
from compas_acoustics.pb import plain
from compas_acoustics.pb.generated import acoustics_pb2 as pb
from compas_acoustics.pb.types import Audio
from compas_acoustics.rays import Ray
from compas_acoustics.rays import RayField
from compas_acoustics.rays import ScatteredRay
from compas_acoustics.rays import SpecularPath
from compas_acoustics.rays import SpecularReflections
from compas_acoustics.receiver import Receiver
from compas_acoustics.scene import Scene
from compas_acoustics.source import Source

__all__ = []


def _set_xyz(dest, xyz):
    dest.x, dest.y, dest.z = (float(v) for v in xyz)


def _xyz(message):
    return [message.x, message.y, message.z]


def _identify(obj, message):
    """Give ``obj`` the guid and name of ``message``, as the object it was made from."""
    if message.guid:
        obj._guid = UUID(message.guid)
    return obj


# =============================================================================
# AcousticMaterial
# =============================================================================


@pb_serializer(AcousticMaterial)
def material_to_pb(material):
    data = material.__data__
    message = pb.AcousticMaterialData(
        guid=str(material.guid),
        name=material.name,
        description=data["description"],
        category=data["category"],
        data_origin=data["data_origin"] or "",
    )
    message.absorption_coefficients.extend(data["absorption_coefficients"] or [])
    message.scattering_coefficients.extend(data["scattering_coefficients"] or [])
    message.transmission_coefficients.extend(data["transmission_coefficients"] or [])
    return message


@pb_deserializer(pb.AcousticMaterialData)
def material_from_pb(message):
    material = AcousticMaterial(
        name=message.name,
        absorption_coefficients=list(message.absorption_coefficients),
        scattering_coefficients=list(message.scattering_coefficients) or None,
        transmission_coefficients=list(message.transmission_coefficients) or None,
        description=message.description or None,
        category=message.category or None,
        data_origin=message.data_origin or None,
    )
    return _identify(material, message)


# =============================================================================
# Source, Receiver
# =============================================================================


@pb_serializer(Source)
def source_to_pb(source):
    data = source.__data__
    message = pb.SourceData(guid=str(source.guid), name=source.name or "", directivity=data["directivity"] or "")
    _set_xyz(message.point, data["point"])
    _set_xyz(message.direction, data["direction"])
    if data["sound_file"] is not None:
        message.sound_file = data["sound_file"]
    if data["description"] is not None:
        message.description = data["description"]
    message.spl.extend(data["spl"] or [])
    return message


@pb_deserializer(pb.SourceData)
def source_from_pb(message):
    data = {
        "name": message.name,
        "point": _xyz(message.point),
        "directivity": message.directivity or None,
        "direction": _xyz(message.direction),
        "sound_file": message.sound_file if message.HasField("sound_file") else None,
        "spl": list(message.spl) or None,
        "description": message.description if message.HasField("description") else None,
    }
    return _identify(Source.__from_data__(data), message)


@pb_serializer(Receiver)
def receiver_to_pb(receiver):
    data = receiver.__data__
    message = pb.ReceiverData(guid=str(receiver.guid), name=receiver.name or "", radius=float(data["radius"]))
    _set_xyz(message.point, data["point"])
    _set_xyz(message.direction, data["direction"])
    _set_xyz(message.up, data["up"])
    return message


@pb_deserializer(pb.ReceiverData)
def receiver_from_pb(message):
    data = {
        "name": message.name,
        "point": _xyz(message.point),
        "direction": _xyz(message.direction),
        "up": _xyz(message.up),
        "radius": message.radius,
    }
    return _identify(Receiver.__from_data__(data), message)


# =============================================================================
# AcousticObject
# =============================================================================


@pb_serializer(AcousticObject)
def acoustic_object_to_pb(obj):
    data = obj.__data__
    message = pb.AcousticObjectData(guid=str(obj.guid), role=data["role"])
    if obj._name is not None:
        message.name = obj._name
    geometry = data["acoustic_geometry"]
    if isinstance(geometry, Mesh):
        message.mesh.CopyFrom(mesh_to_pb(geometry))
    elif isinstance(geometry, Polyline):
        message.polyline.CopyFrom(polyline_to_pb(geometry))
    else:
        raise TypeError(f"An acoustic object's geometry must be a Mesh or a Polyline to travel as a message, not {type(geometry).__name__}.")
    if data["material"] is not None:
        message.material.CopyFrom(material_to_pb(data["material"]))
    _set_xyz(message.position, data["position"])
    message.sources.extend(source_to_pb(source) for source in data["sources"])
    return message


@pb_deserializer(pb.AcousticObjectData)
def acoustic_object_from_pb(message):
    which = message.WhichOneof("geometry")
    geometry = mesh_from_pb(message.mesh) if which == "mesh" else polyline_from_pb(message.polyline) if which == "polyline" else None
    data = {
        "acoustic_geometry": geometry,
        "material": material_from_pb(message.material) if message.HasField("material") else None,
        "position": _xyz(message.position),
        "role": message.role,
        "sources": [source_from_pb(source) for source in message.sources],
    }
    obj = AcousticObject.__from_data__(data)
    if message.HasField("name"):
        obj.name = message.name
    return _identify(obj, message)


# =============================================================================
# Scene
# =============================================================================


@pb_serializer(Scene)
def scene_to_pb(scene):
    message = pb.SceneData(guid=str(scene.guid), name=scene.name or "", description=scene.description or "", sampling_rate=scene.sampling_rate)
    message.acoustic_objects.extend(acoustic_object_to_pb(obj) for obj in scene.acoustic_objects)
    message.sources.extend(source_to_pb(source) for source in scene.sources)
    message.receivers.extend(receiver_to_pb(receiver) for receiver in scene.receivers)
    return message


@pb_deserializer(pb.SceneData)
def scene_from_pb(message):
    data = {
        "name": message.name or None,
        "description": message.description,
        "sampling_rate": message.sampling_rate or None,
        "acoustic_objects": [acoustic_object_from_pb(obj) for obj in message.acoustic_objects],
        "sources": [source_from_pb(source) for source in message.sources],
        "receivers": [receiver_from_pb(receiver) for receiver in message.receivers],
    }
    return _identify(Scene.__from_data__(data), message)


# =============================================================================
# ImpulseResponse, Audio
# =============================================================================


def _fill_samples(message, samples):
    samples = np.asarray(samples, dtype=float)
    message.shape.extend(int(n) for n in samples.shape)
    message.samples.extend(samples.ravel().tolist())


def _samples(message):
    return np.asarray(message.samples, dtype=float).reshape(tuple(message.shape))


@pb_serializer(ImpulseResponse)
def impulse_response_to_pb(ir):
    message = pb.ImpulseResponseData(guid=str(ir.guid), name=ir.name or "", sampling_rate=int(ir.sampling_rate), normalized=bool(ir.normalized))
    message.source.CopyFrom(source_to_pb(ir.source))
    message.receiver.CopyFrom(receiver_to_pb(ir.receiver))
    _fill_samples(message, ir.denormalized())
    message.generation.CopyFrom(_serializer_any(plain(ir.generation or {})))
    if ir.ambisonic_order is not None:
        message.ambisonic_order = int(ir.ambisonic_order)
    if ir.normalization is not None:
        message.normalization = ir.normalization
    if ir.channel_order is not None:
        message.channel_order = ir.channel_order
    return message


@pb_deserializer(pb.ImpulseResponseData)
def impulse_response_from_pb(message):
    ir = ImpulseResponse(
        message.name,
        source_from_pb(message.source),
        receiver_from_pb(message.receiver),
        _samples(message),
        message.sampling_rate,
        normalized=message.normalized,
        generation=_deserialize_any(message.generation) if message.HasField("generation") else None,
        ambisonic_order=message.ambisonic_order if message.HasField("ambisonic_order") else None,
        normalization=message.normalization if message.HasField("normalization") else None,
        channel_order=message.channel_order if message.HasField("channel_order") else None,
    )
    return _identify(ir, message)


@pb_serializer(Audio)
def audio_to_pb(audio):
    message = pb.AudioData(guid=str(audio.guid), sampling_rate=audio.sampling_rate, format=audio.format or "")
    _fill_samples(message, audio.samples)
    return message


@pb_deserializer(pb.AudioData)
def audio_from_pb(message):
    audio = Audio(_samples(message), message.sampling_rate, message.format or "mono")
    if message.guid:
        audio.guid = UUID(message.guid)
    return audio


# =============================================================================
# Rays: Ray, ScatteredRay, SpecularPath, RayField, SpecularReflections
# =============================================================================


def _fill_ray(message, ray):
    message.guid = str(ray.guid)
    if ray._name is not None:
        message.name = ray._name
    message.points.extend(float(v) for point in ray.points for v in point)
    if ray.source_index is not None:
        message.source_index = int(ray.source_index)
    if ray.ray_index is not None:
        message.ray_index = int(ray.ray_index)
    _set_xyz(message.direction, ray.direction)
    message.distance_offset = float(ray.distance_offset)


def _ray_data(message):
    data = {
        "points": [list(message.points[i:i + 3]) for i in range(0, len(message.points), 3)],
        "source_index": message.source_index if message.HasField("source_index") else None,
        "ray_index": message.ray_index if message.HasField("ray_index") else None,
        "direction": _xyz(message.direction),
        "distance_offset": message.distance_offset,
    }
    if message.HasField("name"):
        data["name"] = message.name
    return data


def _scattered_to_pb(ray):
    message = pb.ScatteredRayData(origin_reflection_order=int(ray.reflection_order))
    _fill_ray(message, ray)
    if ray.parent_ray_index is not None:
        message.parent_ray_index = int(ray.parent_ray_index)
    if ray.energy is not None:
        message.energy = float(ray.energy)
    return message


def _scattered_from_pb(message):
    ray = ScatteredRay(
        parent_ray_index=message.parent_ray_index if message.HasField("parent_ray_index") else None,
        energy=message.energy if message.HasField("energy") else None,
        origin_reflection_order=message.origin_reflection_order,
        **_ray_data(message),
    )
    return _identify(ray, message)


def _ray_to_pb(ray):
    message = pb.RayData()
    _fill_ray(message, ray)
    message.scattered_reflections.extend(_scattered_to_pb(s) for s in getattr(ray, "scattered_reflections", None) or [])
    return message


def _ray_from_pb(message):
    ray = _identify(Ray(**_ray_data(message)), message)
    if message.scattered_reflections:
        ray.scattered_reflections = [_scattered_from_pb(s) for s in message.scattered_reflections]
    return ray


def _path_to_pb(path):
    message = pb.SpecularPathData(order=int(path.order), receiver_index=int(path.receiver_index))
    _fill_ray(message, path)
    message.walls.extend(int(w) for w in path.walls)
    _set_xyz(message.image, path.image)
    return message


def _path_from_pb(message):
    path = SpecularPath(order=message.order, walls=list(message.walls), image=_xyz(message.image),
                        receiver_index=message.receiver_index, **_ray_data(message))
    return _identify(path, message)


def _field_sources(field):
    return {ray.source_index: ray.source for ray in field.rays if ray.source is not None and ray.source_index is not None}


def _give_sources(field, sources):
    for ray in field.rays:
        for traced in [ray, *(getattr(ray, "scattered_reflections", None) or [])]:
            traced.source = sources.get(traced.source_index)
    return field


@pb_serializer(RayField)
def ray_field_to_pb(field):
    message = pb.RayFieldData(guid=str(field.guid))
    if field._name is not None:
        message.name = field._name
    message.rays.extend(_ray_to_pb(ray) for ray in field.rays)
    for index, source in _field_sources(field).items():
        message.sources[int(index)].CopyFrom(source_to_pb(source))
    return message


@pb_deserializer(pb.RayFieldData)
def ray_field_from_pb(message):
    field = RayField([_ray_from_pb(ray) for ray in message.rays], name=message.name if message.HasField("name") else None)
    _give_sources(field, {index: source_from_pb(source) for index, source in message.sources.items()})
    return _identify(field, message)


@pb_serializer(SpecularReflections)
def specular_reflections_to_pb(found):
    message = pb.SpecularReflectionsData(guid=str(found.guid))
    if found._name is not None:
        message.name = found._name
    message.paths.extend(_path_to_pb(path) for path in found.paths)
    for index, source in _field_sources(found).items():
        message.sources[int(index)].CopyFrom(source_to_pb(source))
    return message


@pb_deserializer(pb.SpecularReflectionsData)
def specular_reflections_from_pb(message):
    found = SpecularReflections([_path_from_pb(path) for path in message.paths], name=message.name if message.HasField("name") else None)
    _give_sources(found, {index: source_from_pb(source) for index, source in message.sources.items()})
    return _identify(found, message)
