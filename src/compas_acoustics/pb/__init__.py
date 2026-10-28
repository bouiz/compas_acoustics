"""compas_acoustics' plugin for compas_pb: its Data types as protobuf messages.

The schema is ``protobuf_defs/compas_acoustics/pb/generated/acoustics.proto``; the Python bindings in
``generated`` and the C# bindings of the Grasshopper plugin are generated from it (``invoke generate``).
compas_pb finds the converters through the ``compas_pb.plugins`` entry point; importing this package
registers them too, for a checkout that is on the path without being installed.
"""

import hashlib
import os
from pathlib import Path

__all__ = ["PROTOBUF_DEFS", "SCHEMA", "schema_hash", "plain"]

PROTOBUF_DEFS = os.path.join(os.path.dirname(__file__), "protobuf_defs")


def schema_hash(root=PROTOBUF_DEFS):
    """The sha256 of every ``.proto`` under ``root``, by relative path: equal on both ends of a wire that agree on the schema."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.proto")):
        digest.update(path.relative_to(root).as_posix().encode())
        # a Windows checkout writes CRLF; the plugin's hash is of the LF files
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


SCHEMA = schema_hash()


def plain(value):
    """``value`` with numpy scalars and arrays as Python numbers and lists, and tuples as lists: what compas_pb can write."""
    import numpy as np

    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _register():
    from compas_pb.plugin import PLUGIN_MANAGER

    # compas_pb discovers its plugins only while its registry is empty: let it do so before adding ours
    PLUGIN_MANAGER.discover_plugins()
    from compas_acoustics.pb import conversions  # noqa: F401


_register()
