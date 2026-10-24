# C# Audio Player and compas_pb Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Python `CA_Audio_Player` with a C# Grasshopper component (waveform, seek, volume, channel display and selection), fed by `compas_pb` bytes from the Python components.

**Architecture:** Python components keep their Python outputs and gain a `pb` output carrying `compas_pb` bytes as a .NET `byte[]`. A netstandard2.0 library (`CompasAcoustics.Core`) decodes WAV files and `AudioData` / `SourceData` messages into an in-memory float buffer and plays it through NAudio (Windows) or AudioToolbox (macOS). A net48 `.gha` (`CompasAcoustics.GH`) hosts the component and draws its face.

**Tech Stack:** Python 3.9+ (Rhino 8 CPython), compas, compas_pb 1.2, protobuf; C# (LangVersion latest), netstandard2.0 / net48 / net8.0, CompasPb 1.0.0, Google.Protobuf 3.31.0, Grpc.Tools 2.72.0, NAudio 2.2.1, Grasshopper SDK 8.0.23304.9001, xUnit 2.9.3.

Spec: `docs/superpowers/specs/2026-10-14-audio-player-csharp-design.md`

## Global Constraints

- Platforms: Rhino 8 on Windows and macOS.
- Audio formats: WAV only (PCM 16/24/32-bit integer, 32-bit float, `WAVE_FORMAT_EXTENSIBLE`). No MP3.
- Python pin: `compas_pb >=1.2, <2`. C#: `CompasPb` 1.0.0 (compas_pb 1.2 wire format).
- Wire layout of `AudioData.samples`: float32, little-endian, interleaved, length `frames * channels * 4`.
- Python array shape convention: `(frames,)` or `(frames, channels)`.
- C# folder: `src/compas_acoustics_ghcsharp/`. Proto source of truth: `src/compas_acoustics/protobuf_defs/compas_acoustics/generated/acoustics.proto`.
- `CompasAcoustics.Core` has no Rhino or Grasshopper references.
- The Audio Player has no outputs. At most one player plays at a time per Rhino session.
- Python line length 179, ruff rules `E,F,I`, `force-single-line` imports.
- Component category "COMPAS Acoustics", subcategory "06 Util".

---

## File Structure

Python (core, CAD-agnostic):
- Create `src/compas_acoustics/protobuf_defs/compas_acoustics/generated/acoustics.proto`: schema.
- Create `src/compas_acoustics/generated/__init__.py`, `acoustics_pb2.py`, `acoustics_pb2.pyi`: generated bindings (committed).
- Create `src/compas_acoustics/audio_signal.py`: `AudioSignal` data class.
- Create `src/compas_acoustics/pb.py`: compas_pb serializers and `to_pb_bytes`.
- Modify `src/compas_acoustics/__init__.py`, `pyproject.toml`, `requirements.txt`, `.gitignore`, `tasks.py`.

Python (Grasshopper):
- Create `src/compas_acoustics_ghpython/pb.py`: `to_dotnet_bytes`.
- Modify `components/CA_Source/*`, `components/CA_Auralise/*`.
- Create `components/CA_To_compas_pb/{code.py,metadata.json,icon.png}`.
- Delete `components/CA_Audio_Player/`.

Tests (Python):
- Create `tests/test_pb_schema.py`, `tests/test_pb.py`, `tests/test_pb_fixtures.py`, `tests/fixtures/make_pb_fixtures.py`, `tests/fixtures/pb/*.pb`.

C# (`src/compas_acoustics_ghcsharp/`):
- `CompasAcoustics.sln`, `Directory.Build.props`
- `protos/compas_pb/generated/geometry.proto`: vendored import.
- `CompasAcoustics.Core/`: `Audio/AudioBuffer.cs`, `Audio/WaveformPeaks.cs`, `Audio/AudioExceptions.cs`, `Audio/AudioDecoder.cs`, `Audio/AudioInputResolver.cs`, `Audio/PlaybackEngine.cs`, `Audio/IAudioOutput.cs`, `Audio/AudioOutputFactory.cs`, `Audio/WindowsAudioOutput.cs`, `Audio/MacAudioOutput.cs`, `Audio/AudioPlayer.cs`, `Audio/PlaybackCoordinator.cs`, `Data/AcousticsRegistrations.cs`.
- `CompasAcoustics.Core.Tests/`: `TestWav.cs`, `AudioBufferTests.cs`, `AudioDecoderTests.cs`, `AudioInputResolverTests.cs`, `FixtureCompatibilityTests.cs`, `PlaybackEngineTests.cs`, `FakeAudioOutput.cs`, `AudioPlayerTests.cs`.
- `CompasAcoustics.GH/`: `CompasAcousticsInfo.cs`, `Components/GooUnwrap.cs`, `Components/AudioPlayerComponent.cs`, `Components/AudioPlayerAttributes.cs`, `Resources/audio_player.png`.

---

### Task 1: Protobuf schema and Python bindings

**Files:**
- Create: `src/compas_acoustics/protobuf_defs/compas_acoustics/generated/acoustics.proto`
- Create: `src/compas_acoustics/generated/__init__.py` (empty) and generated `acoustics_pb2.py`, `acoustics_pb2.pyi`
- Modify: `requirements.txt`, `pyproject.toml` (package-data), `.gitignore`, `tasks.py`
- Test: `tests/test_pb_schema.py`

**Interfaces:**
- Produces: Python module `compas_acoustics.generated.acoustics_pb2` with messages `AudioData` (`guid`, `name`, `sample_rate`, `channels`, `frames`, `samples`) and `SourceData` (`guid`, `name`, `description`, `position: PointData`, `directivity`, `direction: VectorData`, `spl`, `sound_file`). C# namespace `CompasAcoustics.Data`.

- [ ] **Step 1: Write the failing test** in `tests/test_pb_schema.py`

```python
from compas_acoustics.generated import acoustics_pb2


def test_audio_data_round_trips_through_wire_format():
    message = acoustics_pb2.AudioData(sample_rate=48000, channels=2, frames=3, samples=b"\x00" * 24)
    assert acoustics_pb2.AudioData.FromString(message.SerializeToString()) == message


def test_source_data_uses_compas_pb_geometry():
    message = acoustics_pb2.SourceData(name="speaker")
    message.position.x = 1.0
    assert message.position.DESCRIPTOR.full_name == "compas_pb.data.PointData"
    assert message.direction.DESCRIPTOR.full_name == "compas_pb.data.VectorData"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pb_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compas_acoustics.generated'`

- [ ] **Step 3: Add the dependency and schema**

Append to `requirements.txt`:

```text
compas_pb >=1.2, <2
```

Install: `pip install "compas_pb>=1.2,<2"`

Create `src/compas_acoustics/protobuf_defs/compas_acoustics/generated/acoustics.proto`:

```protobuf
syntax = "proto3";

package compas_acoustics.data;

option csharp_namespace = "CompasAcoustics.Data";

import "compas_pb/generated/geometry.proto";

message AudioData {
  string guid = 1;
  string name = 2;
  int32 sample_rate = 3;
  int32 channels = 4;
  int64 frames = 5;
  // float32, little-endian, interleaved; length is frames * channels * 4
  bytes samples = 6;
}

message SourceData {
  string guid = 1;
  string name = 2;
  string description = 3;
  compas_pb.data.PointData position = 4;
  string directivity = 5;
  compas_pb.data.VectorData direction = 6;
  repeated double spl = 7;
  string sound_file = 8;
}
```

In `pyproject.toml`, extend `[tool.setuptools.package-data]`:

```toml
compas_acoustics = [
    "materials_library/*.json",
    "hrtfs_library/*.sofa",
    "anechoic_soundfiles_library/*",
    "protobuf_defs/**/*.proto",
]
```

In `.gitignore`, directly below the existing `generated/` line, add:

```text
!src/compas_acoustics/generated/
```

- [ ] **Step 4: Add the generation task to `tasks.py`**

Add imports at the top (after `import sys`):

```python
import compas_pb
from compas_pb.invocations import generate_proto_classes
```

Add `generate_proto_classes,` to the `Collection(...)` arguments, and add these keys to the `ns.configure({...})` dict:

```python
        "proto_folder": "src/compas_acoustics/protobuf_defs/compas_acoustics/generated",
        "proto_include_paths": [
            "src/compas_acoustics/protobuf_defs",
            os.path.join(os.path.dirname(compas_pb.__file__), "protobuf_defs"),
        ],
        "proto_out_folder": "src",
```

- [ ] **Step 5: Generate the bindings**

Run: `invoke generate-proto-classes -t python`
Expected: creates `src/compas_acoustics/generated/acoustics_pb2.py` and `acoustics_pb2.pyi` (protoc is downloaded on first run).

Create empty `src/compas_acoustics/generated/__init__.py`.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_pb_schema.py -v`
Expected: 2 passed

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pyproject.toml .gitignore tasks.py src/compas_acoustics/protobuf_defs src/compas_acoustics/generated tests/test_pb_schema.py
git commit -m "Add acoustics protobuf schema and Python bindings"
```

---

### Task 2: AudioSignal and compas_pb serializers

**Files:**
- Create: `src/compas_acoustics/audio_signal.py`, `src/compas_acoustics/pb.py`
- Modify: `src/compas_acoustics/__init__.py`, `pyproject.toml` (entry point)
- Test: `tests/test_pb.py`

**Interfaces:**
- Consumes: `acoustics_pb2.AudioData`, `acoustics_pb2.SourceData` (Task 1).
- Produces: `compas_acoustics.AudioSignal(samples, sample_rate, name=None)` with `.samples` (float32, shape `(frames, channels)`), `.sample_rate: int`, `.frames: int`, `.channels: int`; `compas_acoustics.pb.to_pb_bytes(obj, sample_rate=None) -> bytes`; compas_pb round trip for `AudioSignal` and `Source`.

- [ ] **Step 1: Write the failing tests** in `tests/test_pb.py`

```python
import sys

import numpy as np
import pytest
import soundfile as sf
from compas_pb import pb_dump_bts
from compas_pb import pb_load_bts

from compas_acoustics import AudioSignal
from compas_acoustics import Source
from compas_acoustics.pb import to_pb_bytes

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


def test_audio_signal_mono_becomes_one_channel():
    signal = AudioSignal(np.zeros(10), 48000)
    assert signal.samples.shape == (10, 1)
    assert signal.samples.dtype == np.float32
    assert signal.frames == 10
    assert signal.channels == 1


def test_audio_signal_rejects_three_dimensional_samples():
    with pytest.raises(ValueError):
        AudioSignal(np.zeros((2, 2, 2)), 48000)


def test_audio_signal_rejects_non_positive_sample_rate():
    with pytest.raises(ValueError):
        AudioSignal(np.zeros(4), 0)


def test_audio_signal_round_trip():
    samples = np.arange(12, dtype=np.float32).reshape(6, 2) / 12
    signal = AudioSignal(samples, 44100, name="clip")

    loaded = pb_load_bts(pb_dump_bts(signal))

    assert isinstance(loaded, AudioSignal)
    assert loaded.sample_rate == 44100
    assert loaded.channels == 2
    assert loaded.name == "clip"
    assert loaded.guid == signal.guid
    np.testing.assert_array_equal(loaded.samples, samples)


def test_source_round_trip(tmp_path):
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), np.zeros(10), 48000)
    source = Source("speaker", 1.0, 2.0, 3.0, directivity="cardioid", direction=[0, 1, 0], sound_file=str(wav))
    source.description = "test speaker"

    loaded = pb_load_bts(pb_dump_bts(source))

    assert isinstance(loaded, Source)
    assert loaded.name == "speaker"
    assert loaded.description == "test speaker"
    assert loaded.position == [1.0, 2.0, 3.0]
    assert loaded.directivity == "cardioid"
    assert list(loaded.direction) == [0.0, 1.0, 0.0]
    assert loaded.spl == source.spl
    assert loaded.sound_file == str(wav)
    assert loaded.guid == source.guid


def test_to_pb_bytes_wraps_numpy_array_with_sample_rate():
    loaded = pb_load_bts(to_pb_bytes(np.zeros((5, 2)), 44100))
    assert isinstance(loaded, AudioSignal)
    assert loaded.sample_rate == 44100
    assert loaded.channels == 2


def test_to_pb_bytes_defaults_to_48k():
    loaded = pb_load_bts(to_pb_bytes(np.zeros(5)))
    assert loaded.sample_rate == 48000


def test_to_pb_bytes_rejects_other_types():
    with pytest.raises(TypeError):
        to_pb_bytes("not audio")


def test_compas_pb_plugin_entry_point_is_registered():
    assert "compas_acoustics.pb" in {ep.value for ep in entry_points(group="compas_pb.plugins")}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pb.py -v`
Expected: FAIL with `ImportError: cannot import name 'AudioSignal'`

- [ ] **Step 3: Implement `src/compas_acoustics/audio_signal.py`**

```python
import numpy as np
from compas.data import Data

__all__ = ["AudioSignal"]


class AudioSignal(Data):
    """An audio signal held in memory.

    Parameters
    ----------
    samples : array-like
        Shape ``(frames,)`` for mono or ``(frames, channels)``.
    sample_rate : int
        Samples per second.
    name : str, optional
        Name of the signal.
    """

    def __init__(self, samples, sample_rate, name=None):
        super().__init__(name=name)
        self.samples = samples
        self.sample_rate = sample_rate

    @property
    def __data__(self):
        return {"samples": self.samples.tolist(), "sample_rate": self.sample_rate}

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, value):
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, np.newaxis]
        if array.ndim != 2:
            raise ValueError("Samples must have shape (frames,) or (frames, channels), got {}.".format(array.shape))
        self._samples = array

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError("Sample rate must be positive, got {}.".format(value))
        self._sample_rate = value

    @property
    def frames(self):
        return self._samples.shape[0]

    @property
    def channels(self):
        return self._samples.shape[1]
```

- [ ] **Step 4: Implement `src/compas_acoustics/pb.py`**

```python
"""compas_pb serializers for compas_acoustics types."""

import uuid

import numpy as np
from compas.geometry import Point
from compas.geometry import Vector
from compas_pb import pb_dump_bts
from compas_pb.conversions import point_to_pb
from compas_pb.conversions import vector_to_pb
from compas_pb.registry import pb_deserializer
from compas_pb.registry import pb_serializer

from compas_acoustics.audio_signal import AudioSignal
from compas_acoustics.defaults import Defaults
from compas_acoustics.generated import acoustics_pb2
from compas_acoustics.source import Source

__all__ = ["to_pb_bytes"]


def _set_guid(obj, guid):
    if guid:
        obj._guid = uuid.UUID(guid)


@pb_serializer(AudioSignal)
def audio_signal_to_pb(obj):
    samples = np.ascontiguousarray(obj.samples, dtype="<f4")
    return acoustics_pb2.AudioData(
        guid=str(obj.guid),
        name=obj.name,
        sample_rate=obj.sample_rate,
        channels=obj.channels,
        frames=obj.frames,
        samples=samples.tobytes(),
    )


@pb_deserializer(acoustics_pb2.AudioData)
def audio_signal_from_pb(proto):
    expected = proto.frames * proto.channels * 4
    if len(proto.samples) != expected:
        raise ValueError("AudioData holds {} bytes of samples, expected {}.".format(len(proto.samples), expected))
    samples = np.frombuffer(proto.samples, dtype="<f4").reshape(proto.frames, proto.channels)
    signal = AudioSignal(samples.copy(), proto.sample_rate, name=proto.name or None)
    _set_guid(signal, proto.guid)
    return signal


@pb_serializer(Source)
def source_to_pb(obj):
    return acoustics_pb2.SourceData(
        guid=str(obj.guid),
        name=obj.name,
        description=obj.description or "",
        position=point_to_pb(Point(obj.x, obj.y, obj.z)),
        directivity=obj.directivity,
        direction=vector_to_pb(Vector(*obj.direction)),
        spl=list(obj.spl),
        sound_file=obj.sound_file or "",
    )


@pb_deserializer(acoustics_pb2.SourceData)
def source_from_pb(proto):
    source = Source(
        proto.name,
        proto.position.x,
        proto.position.y,
        proto.position.z,
        directivity=proto.directivity or None,
        direction=[proto.direction.x, proto.direction.y, proto.direction.z],
    )
    source.description = proto.description or None
    if proto.spl:
        source.spl = list(proto.spl)
    # the sound_file setter reads and may rewrite the file; a decoded source must not touch disk
    source._sound_file = proto.sound_file or None
    _set_guid(source, proto.guid)
    return source


def to_pb_bytes(obj, sample_rate=None):
    """Serialize a Source, an AudioSignal or a numpy array of samples to compas_pb bytes.

    Parameters
    ----------
    obj : :class:`Source` | :class:`AudioSignal` | numpy.ndarray
        The object to serialize. Arrays have shape ``(frames,)`` or ``(frames, channels)``.
    sample_rate : int, optional
        Sample rate for numpy arrays. Defaults to ``Defaults().SAMPLING_RATE``.

    Returns
    -------
    bytes
    """
    if isinstance(obj, np.ndarray):
        obj = AudioSignal(obj, sample_rate or Defaults().SAMPLING_RATE)
    elif not isinstance(obj, (AudioSignal, Source)):
        raise TypeError("Cannot serialize {} for the C# components; expected a Source, an AudioSignal or a numpy array.".format(type(obj).__name__))
    return pb_dump_bts(obj)
```

- [ ] **Step 5: Export and register**

In `src/compas_acoustics/__init__.py`, add after `from .audio_player import AudioPlayer`:

```python
from .audio_signal import AudioSignal
```

and add `'AudioSignal',` to `__all__`.

In `pyproject.toml`, after `[project.urls]`, add:

```toml
[project.entry-points."compas_pb.plugins"]
compas_acoustics = "compas_acoustics.pb"
```

Reinstall so the entry point is registered: `pip install -e .`

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_pb.py tests/test_pb_schema.py -v`
Expected: all pass

- [ ] **Step 7: Lint and commit**

Run: `invoke lint`
Expected: no errors in the new files.

```bash
git add src/compas_acoustics/audio_signal.py src/compas_acoustics/pb.py src/compas_acoustics/__init__.py pyproject.toml tests/test_pb.py
git commit -m "Add AudioSignal and compas_pb serializers for AudioSignal and Source"
```

---

### Task 3: Cross-language fixtures

**Files:**
- Create: `tests/fixtures/make_pb_fixtures.py`, `tests/fixtures/pb/audio_stereo.pb`, `tests/fixtures/pb/source.pb`
- Test: `tests/test_pb_fixtures.py`

**Interfaces:**
- Consumes: `AudioSignal`, `Source` serializers (Task 2).
- Produces: `tests/fixtures/pb/audio_stereo.pb` (48000 Hz, 2 channels, 4 frames, rows `[0, .25] [.5, -.25] [-.5, .75] [1, -1]`, name `fixture`) and `tests/fixtures/pb/source.pb` (name `speaker`, position `(1, 2, 3)`, directivity `cardioid`, direction `(0, 1, 0)`, sound_file `tone.wav`). Task 6 reads both from C#.

- [ ] **Step 1: Write the failing test** in `tests/test_pb_fixtures.py`

```python
import os

import numpy as np
from compas_pb import pb_load_bts

from compas_acoustics import AudioSignal
from compas_acoustics import Source

PB_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "pb")


def _load(name):
    with open(os.path.join(PB_DIR, name), "rb") as f:
        return pb_load_bts(f.read())


def test_audio_fixture_decodes():
    signal = _load("audio_stereo.pb")
    assert isinstance(signal, AudioSignal)
    assert signal.sample_rate == 48000
    assert signal.name == "fixture"
    np.testing.assert_array_equal(signal.samples, [[0.0, 0.25], [0.5, -0.25], [-0.5, 0.75], [1.0, -1.0]])


def test_source_fixture_decodes():
    source = _load("source.pb")
    assert isinstance(source, Source)
    assert source.name == "speaker"
    assert source.position == [1.0, 2.0, 3.0]
    assert source.directivity == "cardioid"
    assert source.sound_file == "tone.wav"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pb_fixtures.py -v`
Expected: FAIL with `FileNotFoundError` for `audio_stereo.pb`

- [ ] **Step 3: Write the generator** `tests/fixtures/make_pb_fixtures.py`

```python
"""Writes the compas_pb fixtures shared by the Python and C# test suites.

Run from the repository root: ``python tests/fixtures/make_pb_fixtures.py``
"""

import os

import numpy as np
from compas_pb import pb_dump_bts

from compas_acoustics import AudioSignal
from compas_acoustics import Source

PB_DIR = os.path.join(os.path.dirname(__file__), "pb")

STEREO_SAMPLES = np.array([[0.0, 0.25], [0.5, -0.25], [-0.5, 0.75], [1.0, -1.0]], dtype=np.float32)


def make_audio():
    return AudioSignal(STEREO_SAMPLES, 48000, name="fixture")


def make_source():
    source = Source("speaker", 1.0, 2.0, 3.0, directivity="cardioid", direction=[0, 1, 0])
    # bypasses the setter, which requires the file to exist on this machine
    source._sound_file = "tone.wav"
    return source


def main():
    os.makedirs(PB_DIR, exist_ok=True)
    for filename, obj in (("audio_stereo.pb", make_audio()), ("source.pb", make_source())):
        with open(os.path.join(PB_DIR, filename), "wb") as f:
            f.write(pb_dump_bts(obj))
        print("Wrote {}".format(filename))


if __name__ == "__main__":
    main()
```

Run: `python tests/fixtures/make_pb_fixtures.py`
Expected: `Wrote audio_stereo.pb`, `Wrote source.pb`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pb_fixtures.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures tests/test_pb_fixtures.py
git commit -m "Add compas_pb fixtures shared with the C# tests"
```

---

### Task 4: C# solution scaffold and AudioBuffer

**Files:**
- Create: `src/compas_acoustics_ghcsharp/CompasAcoustics.sln`, `src/compas_acoustics_ghcsharp/Directory.Build.props`
- Create: `src/compas_acoustics_ghcsharp/CompasAcoustics.Core/CompasAcoustics.Core.csproj`, `Audio/AudioBuffer.cs`, `Audio/WaveformPeaks.cs`
- Create: `src/compas_acoustics_ghcsharp/CompasAcoustics.Core.Tests/CompasAcoustics.Core.Tests.csproj`, `AudioBufferTests.cs`
- Modify: `.gitignore`

**Interfaces:**
- Produces (namespace `CompasAcoustics.Audio`):
  - `AudioBuffer(int sampleRate, int channels, float[] samples)`; `int SampleRate`, `int Channels`, `int Frames`, `float[] Samples` (interleaved), `double Duration`, `float Peak()`, `AudioBuffer NormalizedTo(float target)`, `WaveformPeaks GetPeaks(int bucketCount)`.
  - `WaveformPeaks`: `float[][] Min`, `float[][] Max` (indexed `[channel][bucket]`), `int Channels`, `int BucketCount`.

- [ ] **Step 1: Create the solution and projects**

```bash
mkdir -p src/compas_acoustics_ghcsharp && cd src/compas_acoustics_ghcsharp
dotnet new sln -n CompasAcoustics
```

`Directory.Build.props`:

```xml
<Project>
  <PropertyGroup>
    <LangVersion>latest</LangVersion>
    <Nullable>enable</Nullable>
    <ImplicitUsings>disable</ImplicitUsings>
  </PropertyGroup>
</Project>
```

`CompasAcoustics.Core/CompasAcoustics.Core.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>netstandard2.0</TargetFramework>
    <AssemblyName>CompasAcoustics.Core</AssemblyName>
    <RootNamespace>CompasAcoustics</RootNamespace>
  </PropertyGroup>
  <ItemGroup>
    <InternalsVisibleTo Include="CompasAcoustics.Core.Tests" />
  </ItemGroup>
</Project>
```

`CompasAcoustics.Core.Tests/CompasAcoustics.Core.Tests.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <IsPackable>false</IsPackable>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.14.1" />
    <PackageReference Include="xunit" Version="2.9.3" />
    <PackageReference Include="xunit.runner.visualstudio" Version="2.8.2" />
  </ItemGroup>
  <ItemGroup>
    <ProjectReference Include="../CompasAcoustics.Core/CompasAcoustics.Core.csproj" />
  </ItemGroup>
</Project>
```

```bash
dotnet sln add CompasAcoustics.Core/CompasAcoustics.Core.csproj CompasAcoustics.Core.Tests/CompasAcoustics.Core.Tests.csproj
```

Append to the repository `.gitignore`:

```text
src/compas_acoustics_ghcsharp/**/bin/
src/compas_acoustics_ghcsharp/**/obj/
```

- [ ] **Step 2: Write the failing tests** `CompasAcoustics.Core.Tests/AudioBufferTests.cs`

```csharp
using System;
using CompasAcoustics.Audio;
using Xunit;

namespace CompasAcoustics.Core.Tests;

public class AudioBufferTests
{
    [Fact]
    public void Constructor_SetsFramesAndDuration()
    {
        var buffer = new AudioBuffer(4, 2, new float[8]);
        Assert.Equal(4, buffer.Frames);
        Assert.Equal(1.0, buffer.Duration);
    }

    [Fact]
    public void Constructor_RejectsSampleCountNotMultipleOfChannels()
    {
        Assert.Throws<ArgumentException>(() => new AudioBuffer(48000, 2, new float[3]));
    }

    [Fact]
    public void Constructor_RejectsNonPositiveSampleRate()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new AudioBuffer(0, 1, new float[1]));
    }

    [Fact]
    public void GetPeaks_ReturnsMinAndMaxPerBucket()
    {
        var buffer = new AudioBuffer(8, 1, new[] { 0f, 1f, -1f, 0.5f, -0.25f, 0.25f, 0f, 0f });
        var peaks = buffer.GetPeaks(2);
        Assert.Equal(new[] { -1f, -0.25f }, peaks.Min[0]);
        Assert.Equal(new[] { 1f, 0.25f }, peaks.Max[0]);
    }

    [Fact]
    public void GetPeaks_SeparatesChannels()
    {
        var buffer = new AudioBuffer(2, 2, new[] { 1f, -1f, 0.5f, -0.5f });
        var peaks = buffer.GetPeaks(1);
        Assert.Equal(2, peaks.Channels);
        Assert.Equal(0.5f, peaks.Min[0][0]);
        Assert.Equal(1f, peaks.Max[0][0]);
        Assert.Equal(-1f, peaks.Min[1][0]);
        Assert.Equal(-0.5f, peaks.Max[1][0]);
    }

    [Fact]
    public void GetPeaks_WithMoreBucketsThanFrames_RepeatsFrames()
    {
        var buffer = new AudioBuffer(2, 1, new[] { 0.2f, 0.8f });
        var peaks = buffer.GetPeaks(4);
        Assert.Equal(new[] { 0.2f, 0.2f, 0.8f, 0.8f }, peaks.Max[0]);
    }

    [Fact]
    public void GetPeaks_ReturnsCachedInstanceForSameBucketCount()
    {
        var buffer = new AudioBuffer(2, 1, new[] { 0.2f, 0.8f });
        Assert.Same(buffer.GetPeaks(3), buffer.GetPeaks(3));
    }

    [Fact]
    public void NormalizedTo_ScalesPeakToTarget()
    {
        var buffer = new AudioBuffer(2, 1, new[] { 0.5f, -0.25f }).NormalizedTo(0.99f);
        Assert.Equal(0.99, buffer.Samples[0], 5);
        Assert.Equal(-0.495, buffer.Samples[1], 5);
    }

    [Fact]
    public void NormalizedTo_LeavesSilenceUnchanged()
    {
        var buffer = new AudioBuffer(2, 1, new[] { 0f, 0f });
        Assert.Same(buffer, buffer.NormalizedTo(0.99f));
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `dotnet test src/compas_acoustics_ghcsharp/CompasAcoustics.sln`
Expected: build FAIL, `The type or namespace name 'AudioBuffer' could not be found`

- [ ] **Step 4: Implement** `CompasAcoustics.Core/Audio/WaveformPeaks.cs`

```csharp
namespace CompasAcoustics.Audio;

/// <summary>Per-channel minimum and maximum sample values, one pair per horizontal bucket.</summary>
public sealed class WaveformPeaks
{
    public WaveformPeaks(float[][] min, float[][] max)
    {
        Min = min;
        Max = max;
    }

    public float[][] Min { get; }

    public float[][] Max { get; }

    public int Channels => Min.Length;

    public int BucketCount => Min.Length == 0 ? 0 : Min[0].Length;
}
```

`CompasAcoustics.Core/Audio/AudioBuffer.cs`:

```csharp
using System;

namespace CompasAcoustics.Audio;

/// <summary>Decoded audio held in memory as interleaved float samples. Treat <see cref="Samples"/> as read-only.</summary>
public sealed class AudioBuffer
{
    private readonly object _peaksGate = new object();
    private WaveformPeaks? _peaks;

    public AudioBuffer(int sampleRate, int channels, float[] samples)
    {
        if (sampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "Sample rate must be positive.");
        }
        if (channels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(channels), channels, "Channel count must be positive.");
        }
        if (samples is null)
        {
            throw new ArgumentNullException(nameof(samples));
        }
        if (samples.Length % channels != 0)
        {
            throw new ArgumentException($"Sample count {samples.Length} is not a multiple of the channel count {channels}.", nameof(samples));
        }

        SampleRate = sampleRate;
        Channels = channels;
        Samples = samples;
        Frames = samples.Length / channels;
    }

    public int SampleRate { get; }

    public int Channels { get; }

    public int Frames { get; }

    public float[] Samples { get; }

    public double Duration => (double)Frames / SampleRate;

    public float Peak()
    {
        float peak = 0f;
        foreach (float sample in Samples)
        {
            float magnitude = Math.Abs(sample);
            if (magnitude > peak)
            {
                peak = magnitude;
            }
        }
        return peak;
    }

    public AudioBuffer NormalizedTo(float target)
    {
        float peak = Peak();
        if (peak == 0f)
        {
            return this;
        }

        float scale = target / peak;
        var scaled = new float[Samples.Length];
        for (int i = 0; i < Samples.Length; i++)
        {
            scaled[i] = Samples[i] * scale;
        }
        return new AudioBuffer(SampleRate, Channels, scaled);
    }

    public WaveformPeaks GetPeaks(int bucketCount)
    {
        if (bucketCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(bucketCount), bucketCount, "Bucket count must be positive.");
        }

        lock (_peaksGate)
        {
            if (_peaks is null || _peaks.BucketCount != bucketCount)
            {
                _peaks = ComputePeaks(bucketCount);
            }
            return _peaks;
        }
    }

    private WaveformPeaks ComputePeaks(int bucketCount)
    {
        var min = new float[Channels][];
        var max = new float[Channels][];
        for (int channel = 0; channel < Channels; channel++)
        {
            min[channel] = new float[bucketCount];
            max[channel] = new float[bucketCount];
        }

        if (Frames == 0)
        {
            return new WaveformPeaks(min, max);
        }

        for (int bucket = 0; bucket < bucketCount; bucket++)
        {
            long start = (long)bucket * Frames / bucketCount;
            long end = Math.Max(start + 1, (long)(bucket + 1) * Frames / bucketCount);
            for (int channel = 0; channel < Channels; channel++)
            {
                float lo = float.MaxValue;
                float hi = float.MinValue;
                for (long frame = start; frame < end; frame++)
                {
                    float sample = Samples[frame * Channels + channel];
                    if (sample < lo)
                    {
                        lo = sample;
                    }
                    if (sample > hi)
                    {
                        hi = sample;
                    }
                }
                min[channel][bucket] = lo;
                max[channel][bucket] = hi;
            }
        }
        return new WaveformPeaks(min, max);
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `dotnet test src/compas_acoustics_ghcsharp/CompasAcoustics.sln`
Expected: `Passed! - Failed: 0, Passed: 9`

- [ ] **Step 6: Commit**

```bash
git add .gitignore src/compas_acoustics_ghcsharp
git commit -m "Scaffold C# solution with AudioBuffer and waveform peaks"
```

---

### Task 5: WAV decoder

**Files:**
- Create: `CompasAcoustics.Core/Audio/AudioExceptions.cs`, `CompasAcoustics.Core/Audio/AudioDecoder.cs`
- Test: `CompasAcoustics.Core.Tests/TestWav.cs`, `CompasAcoustics.Core.Tests/AudioDecoderTests.cs`

(All paths in Tasks 5-10 are relative to `src/compas_acoustics_ghcsharp/`.)

**Interfaces:**
- Consumes: `AudioBuffer` (Task 4).
- Produces:
  - `class AudioInputException : Exception` (`(string)`, `(string, Exception)`), `sealed class AudioFormatException : AudioInputException` (same ctors), `sealed class AudioDeviceException : Exception` (same ctors).
  - `static AudioDecoder.DecodeFile(string path) -> AudioBuffer` (throws `AudioInputException` for missing file, `AudioFormatException` for bad data); `static AudioDecoder.DecodeWav(Stream stream) -> AudioBuffer`.
  - Test helper `TestWav.Create(ushort formatTag, int channels, int sampleRate, int bitsPerSample, byte[] data, bool extensible = false, byte[]? extraChunk = null, bool includeData = true) -> byte[]`, `TestWav.Pcm16(params short[])`, `TestWav.Float32(params float[])`, `TestWav.WriteTemp(byte[]) -> string`.

- [ ] **Step 1: Write the test helper** `CompasAcoustics.Core.Tests/TestWav.cs`

```csharp
using System;
using System.IO;
using System.Text;

namespace CompasAcoustics.Core.Tests;

internal static class TestWav
{
    public static byte[] Create(
        ushort formatTag,
        int channels,
        int sampleRate,
        int bitsPerSample,
        byte[] data,
        bool extensible = false,
        byte[]? extraChunk = null,
        bool includeData = true)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        int blockAlign = channels * bitsPerSample / 8;

        writer.Write(Encoding.ASCII.GetBytes("RIFF"));
        writer.Write(0u);
        writer.Write(Encoding.ASCII.GetBytes("WAVE"));

        writer.Write(Encoding.ASCII.GetBytes("fmt "));
        writer.Write(extensible ? 40u : 16u);
        writer.Write(extensible ? (ushort)0xFFFE : formatTag);
        writer.Write((ushort)channels);
        writer.Write((uint)sampleRate);
        writer.Write((uint)(sampleRate * blockAlign));
        writer.Write((ushort)blockAlign);
        writer.Write((ushort)bitsPerSample);
        if (extensible)
        {
            writer.Write((ushort)22);
            writer.Write((ushort)bitsPerSample);
            writer.Write(0u);
            writer.Write(formatTag);
            writer.Write(new byte[14]);
        }

        if (extraChunk != null)
        {
            writer.Write(Encoding.ASCII.GetBytes("LIST"));
            writer.Write((uint)extraChunk.Length);
            writer.Write(extraChunk);
            if (extraChunk.Length % 2 == 1)
            {
                writer.Write((byte)0);
            }
        }

        if (includeData)
        {
            writer.Write(Encoding.ASCII.GetBytes("data"));
            writer.Write((uint)data.Length);
            writer.Write(data);
        }

        writer.Flush();
        byte[] bytes = stream.ToArray();
        BitConverter.GetBytes((uint)(bytes.Length - 8)).CopyTo(bytes, 4);
        return bytes;
    }

    public static byte[] Pcm16(params short[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            BitConverter.GetBytes(values[i]).CopyTo(bytes, i * 2);
        }
        return bytes;
    }

    public static byte[] Float32(params float[] values)
    {
        var bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
        {
            BitConverter.GetBytes(values[i]).CopyTo(bytes, i * 4);
        }
        return bytes;
    }

    public static string WriteTemp(byte[] wav)
    {
        string path = Path.Combine(Path.GetTempPath(), $"compas_acoustics_{Guid.NewGuid():N}.wav");
        File.WriteAllBytes(path, wav);
        return path;
    }
}
```

- [ ] **Step 2: Write the failing tests** `CompasAcoustics.Core.Tests/AudioDecoderTests.cs`

```csharp
using System;
using System.IO;
using System.Text;
using CompasAcoustics.Audio;
using Xunit;

namespace CompasAcoustics.Core.Tests;

public class AudioDecoderTests
{
    private static AudioBuffer Decode(byte[] wav) => AudioDecoder.DecodeWav(new MemoryStream(wav));

    [Fact]
    public void DecodeWav_Pcm16Stereo()
    {
        var buffer = Decode(TestWav.Create(1, 2, 44100, 16, TestWav.Pcm16(16384, -16384, 32767, -32768)));
        Assert.Equal(44100, buffer.SampleRate);
        Assert.Equal(2, buffer.Channels);
        Assert.Equal(2, buffer.Frames);
        Assert.Equal(0.5, buffer.Samples[0], 5);
        Assert.Equal(-0.5, buffer.Samples[1], 5);
        Assert.Equal(-1.0, buffer.Samples[3], 5);
    }

    [Fact]
    public void DecodeWav_Pcm24()
    {
        var data = new byte[] { 0x00, 0x00, 0x40, 0x00, 0x00, 0xC0 };
        var buffer = Decode(TestWav.Create(1, 1, 48000, 24, data));
        Assert.Equal(0.5, buffer.Samples[0], 5);
        Assert.Equal(-0.5, buffer.Samples[1], 5);
    }

    [Fact]
    public void DecodeWav_Pcm32()
    {
        var buffer = Decode(TestWav.Create(1, 1, 48000, 32, BitConverter.GetBytes(1073741824)));
        Assert.Equal(0.5, buffer.Samples[0], 5);
    }

    [Fact]
    public void DecodeWav_Float32()
    {
        var buffer = Decode(TestWav.Create(3, 1, 48000, 32, TestWav.Float32(0.25f, -0.75f)));
        Assert.Equal(new[] { 0.25f, -0.75f }, buffer.Samples);
    }

    [Fact]
    public void DecodeWav_Extensible4ChannelFloat()
    {
        var buffer = Decode(TestWav.Create(3, 4, 48000, 32, TestWav.Float32(0.1f, 0.2f, 0.3f, 0.4f), extensible: true));
        Assert.Equal(4, buffer.Channels);
        Assert.Equal(1, buffer.Frames);
        Assert.Equal(new[] { 0.1f, 0.2f, 0.3f, 0.4f }, buffer.Samples);
    }

    [Fact]
    public void DecodeWav_SkipsUnknownOddSizedChunks()
    {
        var buffer = Decode(TestWav.Create(1, 1, 48000, 16, TestWav.Pcm16(16384), extraChunk: new byte[] { 1, 2, 3 }));
        Assert.Equal(0.5, buffer.Samples[0], 5);
    }

    [Fact]
    public void DecodeWav_RejectsNonWav()
    {
        var ex = Assert.Throws<AudioFormatException>(() => Decode(Encoding.ASCII.GetBytes("ID3\u0004 this is an mp3")));
        Assert.Contains("Only WAV", ex.Message);
    }

    [Fact]
    public void DecodeWav_RejectsAdpcm()
    {
        var ex = Assert.Throws<AudioFormatException>(() => Decode(TestWav.Create(2, 1, 48000, 4, new byte[4])));
        Assert.Contains("format tag 2", ex.Message);
    }

    [Fact]
    public void DecodeWav_RejectsMissingDataChunk()
    {
        var ex = Assert.Throws<AudioFormatException>(() => Decode(TestWav.Create(1, 1, 48000, 16, Array.Empty<byte>(), includeData: false)));
        Assert.Contains("'data'", ex.Message);
    }

    [Fact]
    public void DecodeFile_MissingFile()
    {
        var ex = Assert.Throws<AudioInputException>(() => AudioDecoder.DecodeFile("/nonexistent/clip.wav"));
        Assert.Contains("not found", ex.Message);
    }

    [Fact]
    public void DecodeFile_ReadsFromDisk()
    {
        string path = TestWav.WriteTemp(TestWav.Create(1, 1, 8000, 16, TestWav.Pcm16(16384)));
        try
        {
            Assert.Equal(8000, AudioDecoder.DecodeFile(path).SampleRate);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `dotnet test CompasAcoustics.sln`
Expected: build FAIL, `AudioDecoder` not found

- [ ] **Step 4: Implement** `CompasAcoustics.Core/Audio/AudioExceptions.cs`

```csharp
using System;

namespace CompasAcoustics.Audio;

/// <summary>The audio input cannot be turned into an <see cref="AudioBuffer"/>.</summary>
public class AudioInputException : Exception
{
    public AudioInputException(string message)
        : base(message) { }

    public AudioInputException(string message, Exception inner)
        : base(message, inner) { }
}

/// <summary>The audio data is not a supported WAV encoding.</summary>
public sealed class AudioFormatException : AudioInputException
{
    public AudioFormatException(string message)
        : base(message) { }

    public AudioFormatException(string message, Exception inner)
        : base(message, inner) { }
}

/// <summary>The audio output device could not be opened or driven.</summary>
public sealed class AudioDeviceException : Exception
{
    public AudioDeviceException(string message)
        : base(message) { }

    public AudioDeviceException(string message, Exception inner)
        : base(message, inner) { }
}
```

`CompasAcoustics.Core/Audio/AudioDecoder.cs`:

```csharp
using System;
using System.IO;
using System.Text;

namespace CompasAcoustics.Audio;

public static class AudioDecoder
{
    private const ushort FormatPcm = 1;
    private const ushort FormatIeeeFloat = 3;
    private const ushort FormatExtensible = 0xFFFE;

    public static AudioBuffer DecodeFile(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new AudioInputException("The audio file path is empty.");
        }
        if (!File.Exists(path))
        {
            throw new AudioInputException($"Audio file not found: {path}");
        }

        using var stream = File.OpenRead(path);
        try
        {
            return DecodeWav(stream);
        }
        catch (AudioFormatException ex)
        {
            throw new AudioFormatException($"{ex.Message} ({path})", ex);
        }
    }

    public static AudioBuffer DecodeWav(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
        if (stream.Length - stream.Position < 12 || ReadId(reader) != "RIFF")
        {
            throw NotWav();
        }
        reader.ReadUInt32();
        if (ReadId(reader) != "WAVE")
        {
            throw NotWav();
        }

        WavFormat? format = null;
        byte[]? data = null;
        while (stream.Length - stream.Position >= 8)
        {
            string id = ReadId(reader);
            long size = reader.ReadUInt32();
            size = Math.Min(size, stream.Length - stream.Position);

            if (id == "fmt ")
            {
                format = ReadFormat(reader, size);
            }
            else if (id == "data")
            {
                data = reader.ReadBytes((int)size);
            }
            else
            {
                stream.Seek(size, SeekOrigin.Current);
            }

            if ((size & 1) == 1 && stream.Position < stream.Length)
            {
                stream.Seek(1, SeekOrigin.Current);
            }
        }

        if (format is null)
        {
            throw new AudioFormatException("The WAV file has no 'fmt ' chunk.");
        }
        if (data is null)
        {
            throw new AudioFormatException("The WAV file has no 'data' chunk.");
        }
        return new AudioBuffer(format.SampleRate, format.Channels, ConvertSamples(format, data));
    }

    private static AudioFormatException NotWav() =>
        new AudioFormatException("Not a WAV file (missing RIFF/WAVE header). Only WAV files are supported.");

    private static string ReadId(BinaryReader reader) => Encoding.ASCII.GetString(reader.ReadBytes(4));

    private static WavFormat ReadFormat(BinaryReader reader, long size)
    {
        if (size < 16)
        {
            throw new AudioFormatException("The WAV 'fmt ' chunk is too short.");
        }

        long start = reader.BaseStream.Position;
        var format = new WavFormat
        {
            Tag = reader.ReadUInt16(),
            Channels = reader.ReadUInt16(),
            SampleRate = (int)reader.ReadUInt32(),
        };
        reader.ReadUInt32();
        format.BlockAlign = reader.ReadUInt16();
        format.BitsPerSample = reader.ReadUInt16();

        if (format.Tag == FormatExtensible)
        {
            if (size < 40)
            {
                throw new AudioFormatException("The WAV extensible 'fmt ' chunk is too short.");
            }
            reader.ReadUInt16();
            reader.ReadUInt16();
            reader.ReadUInt32();
            // the first two bytes of the sub-format GUID hold the actual format tag
            format.Tag = reader.ReadUInt16();
        }

        reader.BaseStream.Position = start + size;
        return format;
    }

    private static float[] ConvertSamples(WavFormat format, byte[] data)
    {
        if (format.Channels <= 0)
        {
            throw new AudioFormatException("The WAV file declares no channels.");
        }
        if (format.SampleRate <= 0)
        {
            throw new AudioFormatException("The WAV file declares no sample rate.");
        }

        Func<byte[], int, float> read = (format.Tag, format.BitsPerSample) switch
        {
            (FormatPcm, 16) => (bytes, offset) => BitConverter.ToInt16(bytes, offset) / 32768f,
            (FormatPcm, 24) => (bytes, offset) => ((bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16)) << 8 >> 8) / 8388608f,
            (FormatPcm, 32) => (bytes, offset) => BitConverter.ToInt32(bytes, offset) / 2147483648f,
            (FormatIeeeFloat, 32) => (bytes, offset) => BitConverter.ToSingle(bytes, offset),
            _ => throw new AudioFormatException(
                $"Unsupported WAV encoding (format tag {format.Tag}, {format.BitsPerSample} bit). Supported: 16, 24 or 32-bit PCM and 32-bit float."),
        };

        int bytesPerSample = format.BitsPerSample / 8;
        if (format.BlockAlign != bytesPerSample * format.Channels)
        {
            throw new AudioFormatException($"Inconsistent WAV block alignment {format.BlockAlign} for {format.Channels} channels of {format.BitsPerSample} bit.");
        }

        int frames = data.Length / format.BlockAlign;
        var samples = new float[frames * format.Channels];
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = read(data, i * bytesPerSample);
        }
        return samples;
    }

    private sealed class WavFormat
    {
        public ushort Tag;
        public int Channels;
        public int SampleRate;
        public int BlockAlign;
        public int BitsPerSample;
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `dotnet test CompasAcoustics.sln`
Expected: all pass (20 tests)

- [ ] **Step 6: Commit**

```bash
git add CompasAcoustics.Core CompasAcoustics.Core.Tests
git commit -m "Add WAV decoder for PCM, float and extensible formats"
```

---

### Task 6: C# protobuf bindings and AudioInputResolver

**Files:**
- Create: `protos/compas_pb/generated/geometry.proto` (vendored copy)
- Create: `CompasAcoustics.Core/Data/AcousticsRegistrations.cs`, `CompasAcoustics.Core/Audio/AudioInputResolver.cs`
- Modify: `CompasAcoustics.Core/CompasAcoustics.Core.csproj`, `CompasAcoustics.Core.Tests/CompasAcoustics.Core.Tests.csproj`
- Modify: `tests/test_pb_schema.py` (drift test, repository root)
- Test: `CompasAcoustics.Core.Tests/AudioInputResolverTests.cs`, `CompasAcoustics.Core.Tests/FixtureCompatibilityTests.cs`

**Interfaces:**
- Consumes: `AudioDecoder.DecodeFile`, `AudioBuffer.NormalizedTo`, exceptions (Tasks 4-5); fixtures (Task 3).
- Produces:
  - Generated `CompasAcoustics.Data.AudioData`, `CompasAcoustics.Data.SourceData`.
  - `static AcousticsRegistrations.Register()`.
  - `static AudioInputResolver.Resolve(object input) -> AudioBuffer` (input is `string` path or `byte[]` compas_pb bytes; throws `AudioInputException`).
  - `static AudioInputResolver.CacheKey(object input) -> string` (throws `AudioInputException` for unsupported input).
  - `static AudioInputResolver.FromAudioData(AudioData audio) -> AudioBuffer` (peak-normalized to 0.99).

- [ ] **Step 1: Vendor the imported proto and add the drift test**

```bash
mkdir -p protos/compas_pb/generated
cp "$(python -c 'import compas_pb, os; print(os.path.join(os.path.dirname(compas_pb.__file__), "protobuf_defs", "compas_pb", "generated", "geometry.proto"))')" protos/compas_pb/generated/geometry.proto
```

Append to `tests/test_pb_schema.py` (repository root), adding `import os` and `import compas_pb` at the top:

```python
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_vendored_geometry_proto_matches_installed_compas_pb():
    installed = os.path.join(os.path.dirname(compas_pb.__file__), "protobuf_defs", "compas_pb", "generated", "geometry.proto")
    vendored = os.path.join(REPO, "src", "compas_acoustics_ghcsharp", "protos", "compas_pb", "generated", "geometry.proto")
    with open(installed, "rb") as a, open(vendored, "rb") as b:
        assert a.read().replace(b"\r\n", b"\n") == b.read().replace(b"\r\n", b"\n")
```

Run: `pytest tests/test_pb_schema.py -v` (from the repository root)
Expected: 3 passed

- [ ] **Step 2: Wire up protobuf generation in the Core project**

Add to `CompasAcoustics.Core/CompasAcoustics.Core.csproj`:

```xml
  <ItemGroup>
    <PackageReference Include="CompasPb" Version="1.0.0" />
    <PackageReference Include="Google.Protobuf" Version="3.31.0" />
    <PackageReference Include="Grpc.Tools" Version="2.72.0" PrivateAssets="all" IncludeAssets="runtime; build; native; contentfiles; analyzers; buildtransitive" />
  </ItemGroup>
  <ItemGroup>
    <Protobuf Include="../../compas_acoustics/protobuf_defs/compas_acoustics/generated/acoustics.proto"
              ProtoRoot="../../compas_acoustics/protobuf_defs"
              AdditionalImportDirs="../protos"
              GrpcServices="None"
              Access="Public" />
  </ItemGroup>
```

Add to `CompasAcoustics.Core.Tests/CompasAcoustics.Core.Tests.csproj`:

```xml
  <ItemGroup>
    <None Include="../../../tests/fixtures/pb/*.pb" Link="Fixtures/%(Filename)%(Extension)" CopyToOutputDirectory="PreserveNewest" />
  </ItemGroup>
```

Run: `dotnet build CompasAcoustics.sln`
Expected: build succeeds (generated `Acoustics.cs` in `obj/`, referencing `CompasPb.Data.PointData`).

- [ ] **Step 3: Write the failing tests** `CompasAcoustics.Core.Tests/AudioInputResolverTests.cs`

```csharp
using System;
using System.IO;
using CompasAcoustics.Audio;
using CompasAcoustics.Data;
using CompasPb;
using CompasPb.Data;
using Google.Protobuf;
using Xunit;

namespace CompasAcoustics.Core.Tests;

public class AudioInputResolverTests
{
    private static readonly CompasPbSerializer Serializer = new CompasPbSerializer();

    public AudioInputResolverTests() => AcousticsRegistrations.Register();

    private static byte[] Floats(params float[] values) => TestWav.Float32(values);

    [Fact]
    public void Resolve_PathDecodesWav()
    {
        string path = TestWav.WriteTemp(TestWav.Create(1, 1, 8000, 16, TestWav.Pcm16(16384)));
        try
        {
            var buffer = AudioInputResolver.Resolve(path);
            Assert.Equal(8000, buffer.SampleRate);
            Assert.Equal(0.5, buffer.Samples[0], 5);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Resolve_AudioDataIsNormalized()
    {
        var audio = new AudioData { SampleRate = 1000, Channels = 1, Frames = 2, Samples = ByteString.CopyFrom(Floats(0.5f, -0.25f)) };
        var buffer = AudioInputResolver.Resolve(Serializer.Pack(audio));
        Assert.Equal(1000, buffer.SampleRate);
        Assert.Equal(0.99, buffer.Samples[0], 5);
        Assert.Equal(-0.495, buffer.Samples[1], 5);
    }

    [Fact]
    public void Resolve_SourceDataDecodesItsSoundFile()
    {
        string path = TestWav.WriteTemp(TestWav.Create(1, 1, 8000, 16, TestWav.Pcm16(16384)));
        try
        {
            var buffer = AudioInputResolver.Resolve(Serializer.Pack(new SourceData { Name = "speaker", SoundFile = path }));
            Assert.Equal(8000, buffer.SampleRate);
            Assert.Equal(0.5, buffer.Samples[0], 5);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Resolve_SourceWithoutSoundFileThrows()
    {
        var ex = Assert.Throws<AudioInputException>(() => AudioInputResolver.Resolve(Serializer.Pack(new SourceData { Name = "speaker" })));
        Assert.Contains("speaker", ex.Message);
        Assert.Contains("no sound file", ex.Message);
    }

    [Fact]
    public void Resolve_OtherMessageThrowsNamingTheType()
    {
        var ex = Assert.Throws<AudioInputException>(() => AudioInputResolver.Resolve(Serializer.Pack(new PointData { X = 1 })));
        Assert.Contains("PointData", ex.Message);
    }

    [Fact]
    public void Resolve_GarbageBytesThrow()
    {
        Assert.Throws<AudioInputException>(() => AudioInputResolver.Resolve(new byte[] { 1, 2, 3, 4, 5 }));
    }

    [Fact]
    public void Resolve_UnsupportedObjectPointsToToCompasPb()
    {
        var ex = Assert.Throws<AudioInputException>(() => AudioInputResolver.Resolve(42));
        Assert.Contains("To compas_pb", ex.Message);
    }

    [Fact]
    public void FromAudioData_RejectsWrongSampleLength()
    {
        var audio = new AudioData { SampleRate = 1000, Channels = 2, Frames = 2, Samples = ByteString.CopyFrom(Floats(0.5f)) };
        Assert.Throws<AudioInputException>(() => AudioInputResolver.FromAudioData(audio));
    }

    [Fact]
    public void CacheKey_SameBytesGiveSameKey()
    {
        Assert.Equal(AudioInputResolver.CacheKey(new byte[] { 1, 2 }), AudioInputResolver.CacheKey(new byte[] { 1, 2 }));
        Assert.NotEqual(AudioInputResolver.CacheKey(new byte[] { 1, 2 }), AudioInputResolver.CacheKey(new byte[] { 1, 3 }));
    }

    [Fact]
    public void CacheKey_ChangesWhenFileIsModified()
    {
        string path = TestWav.WriteTemp(TestWav.Create(1, 1, 8000, 16, TestWav.Pcm16(1)));
        try
        {
            string before = AudioInputResolver.CacheKey(path);
            File.SetLastWriteTimeUtc(path, DateTime.UtcNow.AddMinutes(1));
            Assert.NotEqual(before, AudioInputResolver.CacheKey(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void CacheKey_RejectsEmptyPath()
    {
        Assert.Throws<AudioInputException>(() => AudioInputResolver.CacheKey(" "));
    }
}
```

`CompasAcoustics.Core.Tests/FixtureCompatibilityTests.cs`:

```csharp
using System;
using System.IO;
using CompasAcoustics.Audio;
using CompasAcoustics.Data;
using CompasPb;
using Xunit;

namespace CompasAcoustics.Core.Tests;

public class FixtureCompatibilityTests
{
    public FixtureCompatibilityTests() => AcousticsRegistrations.Register();

    private static byte[] Fixture(string name) => File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", name));

    [Fact]
    public void PythonAudioFixture_ResolvesToNormalizedStereoBuffer()
    {
        var buffer = AudioInputResolver.Resolve(Fixture("audio_stereo.pb"));
        Assert.Equal(48000, buffer.SampleRate);
        Assert.Equal(2, buffer.Channels);
        Assert.Equal(4, buffer.Frames);
        float[] expected = { 0f, 0.25f, 0.5f, -0.25f, -0.5f, 0.75f, 1f, -1f };
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i] * 0.99, buffer.Samples[i], 5);
        }
    }

    [Fact]
    public void PythonSourceFixture_UnpacksAsSourceData()
    {
        var source = Assert.IsType<SourceData>(new CompasPbSerializer().Unpack(Fixture("source.pb")));
        Assert.Equal("speaker", source.Name);
        Assert.Equal("cardioid", source.Directivity);
        Assert.Equal("tone.wav", source.SoundFile);
        Assert.Equal(1.0, source.Position.X);
        Assert.Equal(3.0, source.Position.Z);
        Assert.Equal(1.0, source.Direction.Y);
    }
}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `dotnet test CompasAcoustics.sln`
Expected: build FAIL, `AcousticsRegistrations` / `AudioInputResolver` not found

- [ ] **Step 5: Implement** `CompasAcoustics.Core/Data/AcousticsRegistrations.cs`

```csharp
using CompasPb.Data;

[assembly: CompasPbRegistrations(typeof(CompasAcoustics.Data.AcousticsRegistrations))]

namespace CompasAcoustics.Data;

/// <summary>Registers the compas_acoustics protobuf messages with CompasPb.</summary>
public static class AcousticsRegistrations
{
    public static void Register() => Registry.RegisterAssembly(typeof(AudioData).Assembly);
}
```

`CompasAcoustics.Core/Audio/AudioInputResolver.cs`:

```csharp
using System;
using System.IO;
using System.Security.Cryptography;
using CompasAcoustics.Data;
using CompasPb;

namespace CompasAcoustics.Audio;

/// <summary>Turns an Audio Player input (file path or compas_pb bytes) into an <see cref="AudioBuffer"/>.</summary>
public static class AudioInputResolver
{
    private const float NormalizedPeak = 0.99f;
    private static readonly CompasPbSerializer Serializer = new CompasPbSerializer();

    static AudioInputResolver() => AcousticsRegistrations.Register();

    public static AudioBuffer Resolve(object input)
    {
        switch (input)
        {
            case string path:
                return AudioDecoder.DecodeFile(path);
            case byte[] bytes:
                return ResolvePb(bytes);
            default:
                throw Unsupported(input);
        }
    }

    public static string CacheKey(object input)
    {
        switch (input)
        {
            case string path:
                if (string.IsNullOrWhiteSpace(path))
                {
                    throw new AudioInputException("The audio file path is empty.");
                }
                string full;
                try
                {
                    full = Path.GetFullPath(path);
                }
                catch (Exception ex) when (ex is ArgumentException || ex is NotSupportedException || ex is PathTooLongException)
                {
                    throw new AudioInputException($"Invalid audio file path: {path}", ex);
                }
                long stamp = File.Exists(full) ? File.GetLastWriteTimeUtc(full).Ticks : 0;
                return $"file:{full}:{stamp}";
            case byte[] bytes:
                using (var sha = SHA256.Create())
                {
                    return "pb:" + BitConverter.ToString(sha.ComputeHash(bytes));
                }
            default:
                throw Unsupported(input);
        }
    }

    public static AudioBuffer FromAudioData(AudioData audio)
    {
        if (audio.SampleRate <= 0 || audio.Channels <= 0 || audio.Frames < 0)
        {
            throw new AudioInputException($"AudioData has an invalid layout ({audio.SampleRate} Hz, {audio.Channels} channels, {audio.Frames} frames).");
        }

        long expected = audio.Frames * audio.Channels * 4;
        if (audio.Samples.Length != expected)
        {
            throw new AudioInputException($"AudioData holds {audio.Samples.Length} bytes of samples, expected {expected}.");
        }

        var samples = new float[audio.Frames * audio.Channels];
        System.Buffer.BlockCopy(audio.Samples.ToByteArray(), 0, samples, 0, (int)expected);
        return new AudioBuffer(audio.SampleRate, audio.Channels, samples).NormalizedTo(NormalizedPeak);
    }

    private static AudioBuffer ResolvePb(byte[] bytes)
    {
        object? message;
        try
        {
            message = Serializer.Unpack(bytes);
        }
        catch (Exception ex)
        {
            throw new AudioInputException($"Could not decode the compas_pb data: {ex.Message}", ex);
        }

        switch (message)
        {
            case AudioData audio:
                return FromAudioData(audio);
            case SourceData source:
                if (string.IsNullOrEmpty(source.SoundFile))
                {
                    throw new AudioInputException($"Source '{source.Name}' has no sound file.");
                }
                return AudioDecoder.DecodeFile(source.SoundFile);
            default:
                string found = message is null ? "nothing" : message.GetType().Name;
                throw new AudioInputException($"The compas_pb data contains {found}, expected AudioData or SourceData.");
        }
    }

    private static AudioInputException Unsupported(object? input) =>
        new AudioInputException(
            $"Unsupported audio input of type {input?.GetType().FullName ?? "null"}. "
                + "Connect a WAV file path, or the 'pb' output of a COMPAS Acoustics component (use 'To compas_pb' for numpy arrays).");
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `dotnet test CompasAcoustics.sln`
Expected: all pass (33 tests)

- [ ] **Step 7: Commit**

```bash
git add protos CompasAcoustics.Core CompasAcoustics.Core.Tests ../../tests/test_pb_schema.py
git commit -m "Add C# protobuf bindings and AudioInputResolver"
```

---

### Task 7: PlaybackEngine

**Files:**
- Create: `CompasAcoustics.Core/Audio/PlaybackEngine.cs`
- Test: `CompasAcoustics.Core.Tests/PlaybackEngineTests.cs`

**Interfaces:**
- Consumes: `AudioBuffer` (Task 4).
- Produces: `PlaybackEngine(AudioBuffer buffer)`; `const int OutputChannels = 2`; `AudioBuffer Buffer`; `long PositionFrames`; `double PositionSeconds`; `double PositionFraction`; `float Volume { get; set; }` (clamped 0-1); `int Channel { get; set; }` (0-based, clamped); `void Seek(double fraction)`; `void SeekFrames(long frame)`; `int Fill(float[] destination, int frames)` (writes interleaved stereo, zero-fills past the end, returns audio frames written); `event EventHandler? PlaybackFinished` (raised once on the audio thread when the end is reached).

- [ ] **Step 1: Write the failing tests** `CompasAcoustics.Core.Tests/PlaybackEngineTests.cs`

```csharp
using CompasAcoustics.Audio;
using Xunit;

namespace CompasAcoustics.Core.Tests;

public class PlaybackEngineTests
{
    private static float[] Fill(PlaybackEngine engine, int frames)
    {
        var destination = new float[frames * PlaybackEngine.OutputChannels];
        engine.Fill(destination, frames);
        return destination;
    }

    [Fact]
    public void Fill_MonoIsCopiedToBothOutputs()
    {
        var engine = new PlaybackEngine(new AudioBuffer(1000, 1, new[] { 0.1f, 0.2f, 0.3f }));
        Assert.Equal(new[] { 0.1f, 0.1f, 0.2f, 0.2f, 0.3f, 0.3f }, Fill(engine, 3));
        Assert.Equal(3, engine.PositionFrames);
    }

    [Fact]
    public void Fill_StereoPassesThrough()
    {
        var engine = new PlaybackEngine(new AudioBuffer(1000, 2, new[] { 0.1f, -0.1f, 0.2f, -0.2f }));
        Assert.Equal(new[] { 0.1f, -0.1f, 0.2f, -0.2f }, Fill(engine, 2));
    }

    [Fact]
    public void Fill_MultichannelPlaysSelectedChannelOnBothOutputs()
    {
        var engine = new PlaybackEngine(new AudioBuffer(1000, 4, new[] { 0.1f, 0.2f, 0.3f, 0.4f })) { Channel = 2 };
        Assert.Equal(new[] { 0.3f, 0.3f }, Fill(engine, 1));
    }

    [Fact]
    public void Channel_IsClampedToAvailableChannels()
    {
        var engine = new PlaybackEngine(new AudioBuffer(1000, 4, new float[4]));
        engine.Channel = 9;
        Assert.Equal(3, engine.Channel);
        engine.Channel = -1;
        Assert.Equal(0, engine.Channel);
    }

    [Fact]
    public void Fill_PastEndZeroFillsAndRaisesFinishedOnce()
    {
        var engine = new PlaybackEngine(new AudioBuffer(1000, 1, new[] { 0.5f }));
        int finished = 0;
        engine.PlaybackFinished += (s, e) => finished++;

        var first = new float[4];
        Assert.Equal(1, engine.Fill(first, 2));
        Assert.Equal(new[] { 0.5f, 0.5f, 0f, 0f }, first);
        Assert.Equal(0, engine.Fill(new float[4], 2));
        Assert.Equal(1, finished);
    }

    [Fact]
    public void Seek_MovesPositionAndRearmsFinished()
    {
        var samples = new float[10];
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = i / 10f;
        }
        var engine = new PlaybackEngine(new AudioBuffer(1000, 1, samples));
        int finished = 0;
        engine.PlaybackFinished += (s, e) => finished++;

        engine.Seek(0.5);
        Assert.Equal(5, engine.PositionFrames);
        Assert.Equal(0.5f, Fill(engine, 1)[0]);

        engine.Fill(new float[40], 20);
        engine.Seek(0.9);
        engine.Fill(new float[40], 20);
        Assert.Equal(2, finished);
    }

    [Fact]
    public void Volume_RampsTowardsTarget()
    {
        var samples = new float[10];
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = 1f;
        }
        var engine = new PlaybackEngine(new AudioBuffer(1000, 1, samples)) { Volume = 0f };

        var output = Fill(engine, 10);

        Assert.Equal(0.8, output[0], 4);
        Assert.Equal(0.6, output[2], 4);
        Assert.Equal(0.0, output[8], 4);
        Assert.Equal(0.0, output[18], 4);
    }

    [Fact]
    public void Volume_IsClamped()
    {
        var engine = new PlaybackEngine(new AudioBuffer(1000, 1, new float[1])) { Volume = 2f };
        Assert.Equal(1f, engine.Volume);
        engine.Volume = -1f;
        Assert.Equal(0f, engine.Volume);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test CompasAcoustics.sln`
Expected: build FAIL, `PlaybackEngine` not found

- [ ] **Step 3: Implement** `CompasAcoustics.Core/Audio/PlaybackEngine.cs`

```csharp
using System;
using System.Threading;

namespace CompasAcoustics.Audio;

/// <summary>
/// Produces stereo output samples from an <see cref="AudioBuffer"/>. <see cref="Fill"/> runs on the
/// audio thread; the other members may be called from any thread.
/// </summary>
public sealed class PlaybackEngine
{
    public const int OutputChannels = 2;
    private const double VolumeRampSeconds = 0.005;

    private long _position;
    private float _targetVolume = 1f;
    private float _currentVolume = 1f;
    private int _channel;
    private int _finishedRaised;

    public PlaybackEngine(AudioBuffer buffer)
    {
        Buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
    }

    public event EventHandler? PlaybackFinished;

    public AudioBuffer Buffer { get; }

    public long PositionFrames => Interlocked.Read(ref _position);

    public double PositionSeconds => (double)PositionFrames / Buffer.SampleRate;

    public double PositionFraction => Buffer.Frames == 0 ? 0.0 : (double)PositionFrames / Buffer.Frames;

    public float Volume
    {
        get => Volatile.Read(ref _targetVolume);
        set => Volatile.Write(ref _targetVolume, Math.Max(0f, Math.Min(1f, value)));
    }

    public int Channel
    {
        get => Volatile.Read(ref _channel);
        set => Volatile.Write(ref _channel, Math.Max(0, Math.Min(Buffer.Channels - 1, value)));
    }

    public void Seek(double fraction)
    {
        fraction = Math.Max(0.0, Math.Min(1.0, fraction));
        SeekFrames((long)(fraction * Buffer.Frames));
    }

    public void SeekFrames(long frame)
    {
        Interlocked.Exchange(ref _position, Math.Max(0, Math.Min(Buffer.Frames, frame)));
        Interlocked.Exchange(ref _finishedRaised, 0);
    }

    public int Fill(float[] destination, int frames)
    {
        long start = Interlocked.Read(ref _position);
        int count = (int)Math.Max(0, Math.Min(frames, Buffer.Frames - start));
        int channels = Buffer.Channels;
        float[] source = Buffer.Samples;
        int selected = Channel;
        float target = Volume;
        float step = (float)(1.0 / (VolumeRampSeconds * Buffer.SampleRate));

        for (int i = 0; i < count; i++)
        {
            if (_currentVolume < target)
            {
                _currentVolume = Math.Min(target, _currentVolume + step);
            }
            else if (_currentVolume > target)
            {
                _currentVolume = Math.Max(target, _currentVolume - step);
            }

            long index = (start + i) * channels;
            float left;
            float right;
            if (channels == 1)
            {
                left = right = source[index];
            }
            else if (channels == 2)
            {
                left = source[index];
                right = source[index + 1];
            }
            else
            {
                left = right = source[index + selected];
            }

            destination[2 * i] = left * _currentVolume;
            destination[2 * i + 1] = right * _currentVolume;
        }

        Array.Clear(destination, 2 * count, 2 * (frames - count));
        Interlocked.CompareExchange(ref _position, start + count, start);

        if (count < frames && Interlocked.Exchange(ref _finishedRaised, 1) == 0)
        {
            PlaybackFinished?.Invoke(this, EventArgs.Empty);
        }
        return count;
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test CompasAcoustics.sln`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add CompasAcoustics.Core CompasAcoustics.Core.Tests
git commit -m "Add PlaybackEngine with seek, volume ramp and channel selection"
```

---

### Task 8: AudioPlayer, PlaybackCoordinator and platform outputs

**Files:**
- Create: `CompasAcoustics.Core/Audio/IAudioOutput.cs`, `AudioOutputFactory.cs`, `WindowsAudioOutput.cs`, `MacAudioOutput.cs`, `AudioPlayer.cs`, `PlaybackCoordinator.cs`
- Modify: `CompasAcoustics.Core/CompasAcoustics.Core.csproj` (NAudio)
- Test: `CompasAcoustics.Core.Tests/FakeAudioOutput.cs`, `CompasAcoustics.Core.Tests/AudioPlayerTests.cs`

**Interfaces:**
- Consumes: `PlaybackEngine` (Task 7), `AudioDeviceException` (Task 5).
- Produces:
  - `delegate int FillCallback(float[] destination, int frames)`; `interface IAudioOutput : IDisposable { void Start(int sampleRate, int channels, FillCallback fill); void Stop(); }`.
  - `static AudioOutputFactory.Create() -> IAudioOutput`.
  - `enum PlaybackState { Stopped, Playing, Paused }`.
  - `AudioPlayer(AudioBuffer buffer)`, `AudioPlayer(AudioBuffer buffer, Func<IAudioOutput> outputFactory)`; `PlaybackEngine Engine`; `PlaybackState State`; `event EventHandler? StateChanged` (any thread); `Play()` (throws `AudioDeviceException`), `Pause()`, `Stop()` (returns to start), `TogglePlayPause()`, `Dispose()`.
  - `static PlaybackCoordinator.Activate(AudioPlayer)`, `Release(AudioPlayer)`, `StopAll()`.

- [ ] **Step 1: Write the fake output and failing tests**

`CompasAcoustics.Core.Tests/FakeAudioOutput.cs`:

```csharp
using CompasAcoustics.Audio;

namespace CompasAcoustics.Core.Tests;

internal sealed class FakeAudioOutput : IAudioOutput
{
    public int StartCount { get; private set; }
    public int StopCount { get; private set; }
    public int SampleRate { get; private set; }
    public int Channels { get; private set; }
    private FillCallback? _fill;

    public void Start(int sampleRate, int channels, FillCallback fill)
    {
        StartCount++;
        SampleRate = sampleRate;
        Channels = channels;
        _fill = fill;
    }

    public void Stop() => StopCount++;

    public void Dispose() { }

    public int Pull(int frames) => _fill!(new float[frames * Channels], frames);
}
```

`CompasAcoustics.Core.Tests/AudioPlayerTests.cs`:

```csharp
using System.Threading;
using CompasAcoustics.Audio;
using Xunit;

namespace CompasAcoustics.Core.Tests;

public class AudioPlayerTests
{
    public AudioPlayerTests() => PlaybackCoordinator.StopAll();

    private static (AudioPlayer Player, FakeAudioOutput Output) Create(int frames = 100)
    {
        var output = new FakeAudioOutput();
        return (new AudioPlayer(new AudioBuffer(1000, 1, new float[frames]), () => output), output);
    }

    [Fact]
    public void Play_StartsOutputInStereoAtBufferSampleRate()
    {
        var (player, output) = Create();
        player.Play();
        Assert.Equal(PlaybackState.Playing, player.State);
        Assert.Equal(1000, output.SampleRate);
        Assert.Equal(2, output.Channels);
    }

    [Fact]
    public void Play_StopsOtherActivePlayer()
    {
        var (first, _) = Create();
        var (second, _) = Create();
        first.Play();
        second.Play();
        Assert.Equal(PlaybackState.Stopped, first.State);
        Assert.Equal(PlaybackState.Playing, second.State);
    }

    [Fact]
    public void Pause_KeepsPosition()
    {
        var (player, output) = Create();
        player.Play();
        output.Pull(10);
        player.Pause();
        Assert.Equal(PlaybackState.Paused, player.State);
        Assert.Equal(10, player.Engine.PositionFrames);
        Assert.Equal(1, output.StopCount);
    }

    [Fact]
    public void Stop_ReturnsToStart()
    {
        var (player, output) = Create();
        player.Play();
        output.Pull(10);
        player.Stop();
        Assert.Equal(PlaybackState.Stopped, player.State);
        Assert.Equal(0, player.Engine.PositionFrames);
    }

    [Fact]
    public void TogglePlayPause_Alternates()
    {
        var (player, _) = Create();
        player.TogglePlayPause();
        Assert.Equal(PlaybackState.Playing, player.State);
        player.TogglePlayPause();
        Assert.Equal(PlaybackState.Paused, player.State);
    }

    [Fact]
    public void ReachingEnd_ReturnsToStopped()
    {
        var (player, output) = Create(frames: 5);
        using var stopped = new ManualResetEventSlim();
        player.StateChanged += (s, e) =>
        {
            if (player.State == PlaybackState.Stopped)
            {
                stopped.Set();
            }
        };

        player.Play();
        output.Pull(10);

        Assert.True(stopped.Wait(2000));
        Assert.Equal(0, player.Engine.PositionFrames);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test CompasAcoustics.sln`
Expected: build FAIL, `IAudioOutput` / `AudioPlayer` not found

- [ ] **Step 3: Add NAudio** to `CompasAcoustics.Core/CompasAcoustics.Core.csproj` (inside the existing package `ItemGroup`):

```xml
    <PackageReference Include="NAudio.Core" Version="2.2.1" />
    <PackageReference Include="NAudio.WinMM" Version="2.2.1" />
```

- [ ] **Step 4: Implement the output interface and factory**

`CompasAcoustics.Core/Audio/IAudioOutput.cs`:

```csharp
using System;

namespace CompasAcoustics.Audio;

/// <summary>Fills <paramref name="destination"/> with <paramref name="frames"/> interleaved frames; returns frames of real audio written.</summary>
public delegate int FillCallback(float[] destination, int frames);

/// <summary>A platform audio device that pulls float samples from a <see cref="FillCallback"/>.</summary>
public interface IAudioOutput : IDisposable
{
    void Start(int sampleRate, int channels, FillCallback fill);

    void Stop();
}
```

`CompasAcoustics.Core/Audio/AudioOutputFactory.cs`:

```csharp
using System;
using System.Runtime.InteropServices;

namespace CompasAcoustics.Audio;

public static class AudioOutputFactory
{
    public static IAudioOutput Create()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return new WindowsAudioOutput();
        }
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return new MacAudioOutput();
        }
        throw new PlatformNotSupportedException("Audio playback is only supported on Windows and macOS.");
    }
}
```

`CompasAcoustics.Core/Audio/WindowsAudioOutput.cs`:

```csharp
using System;
using NAudio.Wave;

namespace CompasAcoustics.Audio;

internal sealed class WindowsAudioOutput : IAudioOutput
{
    private WaveOutEvent? _device;

    public void Start(int sampleRate, int channels, FillCallback fill)
    {
        Stop();
        _device = new WaveOutEvent { DesiredLatency = 100 };
        _device.Init(new CallbackSampleProvider(WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, channels), fill));
        _device.Play();
    }

    public void Stop()
    {
        if (_device is null)
        {
            return;
        }
        _device.Stop();
        _device.Dispose();
        _device = null;
    }

    public void Dispose() => Stop();

    private sealed class CallbackSampleProvider : ISampleProvider
    {
        private readonly FillCallback _fill;
        private float[] _scratch = Array.Empty<float>();

        public CallbackSampleProvider(WaveFormat format, FillCallback fill)
        {
            WaveFormat = format;
            _fill = fill;
        }

        public WaveFormat WaveFormat { get; }

        public int Read(float[] buffer, int offset, int count)
        {
            int frames = count / WaveFormat.Channels;
            if (_scratch.Length < count)
            {
                _scratch = new float[count];
            }
            _fill(_scratch, frames);
            Array.Copy(_scratch, 0, buffer, offset, frames * WaveFormat.Channels);
            return frames * WaveFormat.Channels;
        }
    }
}
```

`CompasAcoustics.Core/Audio/MacAudioOutput.cs`:

```csharp
using System;
using System.Runtime.InteropServices;

namespace CompasAcoustics.Audio;

internal sealed class MacAudioOutput : IAudioOutput
{
    private const string AudioToolbox = "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox";
    private const int BufferCount = 3;
    private const double BufferSeconds = 0.02;
    private const uint FormatLinearPcm = 0x6C70636D;
    private const uint FlagsFloatPacked = 1 | 8;

    private static readonly int ByteSizeOffset = Marshal.OffsetOf<AudioQueueBuffer>(nameof(AudioQueueBuffer.AudioDataByteSize)).ToInt32();

    private readonly AudioQueueOutputCallback _callback;
    private IntPtr _queue;
    private FillCallback? _fill;
    private float[] _scratch = Array.Empty<float>();
    private int _channels;
    private volatile bool _stopping;

    public MacAudioOutput()
    {
        // held in a field so the delegate outlives every native callback
        _callback = OnBufferDone;
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void AudioQueueOutputCallback(IntPtr userData, IntPtr queue, IntPtr buffer);

    public void Start(int sampleRate, int channels, FillCallback fill)
    {
        Stop();
        _fill = fill;
        _channels = channels;
        _stopping = false;

        int framesPerBuffer = Math.Max(256, (int)(sampleRate * BufferSeconds));
        _scratch = new float[framesPerBuffer * channels];
        uint bytesPerFrame = (uint)(4 * channels);
        var format = new AudioStreamBasicDescription
        {
            SampleRate = sampleRate,
            FormatId = FormatLinearPcm,
            FormatFlags = FlagsFloatPacked,
            BytesPerPacket = bytesPerFrame,
            FramesPerPacket = 1,
            BytesPerFrame = bytesPerFrame,
            ChannelsPerFrame = (uint)channels,
            BitsPerChannel = 32,
        };

        Check(AudioQueueNewOutput(ref format, _callback, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, 0, out _queue), "AudioQueueNewOutput");
        for (int i = 0; i < BufferCount; i++)
        {
            Check(AudioQueueAllocateBuffer(_queue, (uint)(framesPerBuffer * bytesPerFrame), out IntPtr buffer), "AudioQueueAllocateBuffer");
            FillAndEnqueue(buffer);
        }
        Check(AudioQueueStart(_queue, IntPtr.Zero), "AudioQueueStart");
    }

    public void Stop()
    {
        if (_queue == IntPtr.Zero)
        {
            return;
        }
        _stopping = true;
        AudioQueueStop(_queue, true);
        AudioQueueDispose(_queue, true);
        _queue = IntPtr.Zero;
    }

    public void Dispose() => Stop();

    private void OnBufferDone(IntPtr userData, IntPtr queue, IntPtr buffer)
    {
        if (!_stopping)
        {
            FillAndEnqueue(buffer);
        }
    }

    private void FillAndEnqueue(IntPtr bufferPointer)
    {
        var buffer = Marshal.PtrToStructure<AudioQueueBuffer>(bufferPointer);
        int frames = (int)(buffer.AudioDataBytesCapacity / (uint)(4 * _channels));
        _fill!(_scratch, frames);
        Marshal.Copy(_scratch, 0, buffer.AudioData, frames * _channels);
        Marshal.WriteInt32(bufferPointer, ByteSizeOffset, frames * _channels * 4);
        AudioQueueEnqueueBuffer(_queue, bufferPointer, 0, IntPtr.Zero);
    }

    private static void Check(int status, string call)
    {
        if (status != 0)
        {
            throw new AudioDeviceException($"{call} failed with OSStatus {status}.");
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct AudioStreamBasicDescription
    {
        public double SampleRate;
        public uint FormatId;
        public uint FormatFlags;
        public uint BytesPerPacket;
        public uint FramesPerPacket;
        public uint BytesPerFrame;
        public uint ChannelsPerFrame;
        public uint BitsPerChannel;
        public uint Reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct AudioQueueBuffer
    {
        public uint AudioDataBytesCapacity;
        public IntPtr AudioData;
        public uint AudioDataByteSize;
        public IntPtr UserData;
        public uint PacketDescriptionCapacity;
        public IntPtr PacketDescriptions;
        public uint PacketDescriptionCount;
    }

    [DllImport(AudioToolbox)]
    private static extern int AudioQueueNewOutput(
        ref AudioStreamBasicDescription format,
        AudioQueueOutputCallback callback,
        IntPtr userData,
        IntPtr callbackRunLoop,
        IntPtr callbackRunLoopMode,
        uint flags,
        out IntPtr queue);

    [DllImport(AudioToolbox)]
    private static extern int AudioQueueAllocateBuffer(IntPtr queue, uint bufferByteSize, out IntPtr buffer);

    [DllImport(AudioToolbox)]
    private static extern int AudioQueueEnqueueBuffer(IntPtr queue, IntPtr buffer, uint packetDescriptionCount, IntPtr packetDescriptions);

    [DllImport(AudioToolbox)]
    private static extern int AudioQueueStart(IntPtr queue, IntPtr startTime);

    [DllImport(AudioToolbox)]
    private static extern int AudioQueueStop(IntPtr queue, [MarshalAs(UnmanagedType.U1)] bool immediate);

    [DllImport(AudioToolbox)]
    private static extern int AudioQueueDispose(IntPtr queue, [MarshalAs(UnmanagedType.U1)] bool immediate);
}
```

- [ ] **Step 5: Implement the player and coordinator**

`CompasAcoustics.Core/Audio/AudioPlayer.cs`:

```csharp
using System;
using System.Threading;

namespace CompasAcoustics.Audio;

public enum PlaybackState
{
    Stopped,
    Playing,
    Paused,
}

/// <summary>Plays one <see cref="AudioBuffer"/> through one <see cref="IAudioOutput"/>.</summary>
public sealed class AudioPlayer : IDisposable
{
    private readonly object _gate = new object();
    private readonly Func<IAudioOutput> _outputFactory;
    private IAudioOutput? _output;
    private PlaybackState _state = PlaybackState.Stopped;
    private bool _disposed;

    public AudioPlayer(AudioBuffer buffer)
        : this(buffer, AudioOutputFactory.Create) { }

    public AudioPlayer(AudioBuffer buffer, Func<IAudioOutput> outputFactory)
    {
        _outputFactory = outputFactory;
        Engine = new PlaybackEngine(buffer);
        // the engine raises this on the audio thread, which must not stop its own device
        Engine.PlaybackFinished += (sender, e) => ThreadPool.QueueUserWorkItem(_ => Stop());
    }

    public event EventHandler? StateChanged;

    public PlaybackEngine Engine { get; }

    public PlaybackState State
    {
        get
        {
            lock (_gate)
            {
                return _state;
            }
        }
    }

    public void Play()
    {
        PlaybackCoordinator.Activate(this);
        lock (_gate)
        {
            if (_disposed || _state == PlaybackState.Playing)
            {
                return;
            }
            if (Engine.PositionFrames >= Engine.Buffer.Frames)
            {
                Engine.SeekFrames(0);
            }
            try
            {
                _output ??= _outputFactory();
                _output.Start(Engine.Buffer.SampleRate, PlaybackEngine.OutputChannels, Engine.Fill);
            }
            catch (AudioDeviceException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new AudioDeviceException($"Could not open the audio device: {ex.Message}", ex);
            }
            _state = PlaybackState.Playing;
        }
        OnStateChanged();
    }

    public void Pause()
    {
        lock (_gate)
        {
            if (_state != PlaybackState.Playing)
            {
                return;
            }
            _output?.Stop();
            _state = PlaybackState.Paused;
        }
        OnStateChanged();
    }

    public void Stop()
    {
        bool changed;
        lock (_gate)
        {
            if (_state == PlaybackState.Playing)
            {
                _output?.Stop();
            }
            Engine.SeekFrames(0);
            changed = _state != PlaybackState.Stopped;
            _state = PlaybackState.Stopped;
        }
        PlaybackCoordinator.Release(this);
        if (changed)
        {
            OnStateChanged();
        }
    }

    public void TogglePlayPause()
    {
        if (State == PlaybackState.Playing)
        {
            Pause();
        }
        else
        {
            Play();
        }
    }

    public void Dispose()
    {
        Stop();
        lock (_gate)
        {
            _disposed = true;
            _output?.Dispose();
            _output = null;
        }
    }

    private void OnStateChanged() => StateChanged?.Invoke(this, EventArgs.Empty);
}
```

`CompasAcoustics.Core/Audio/PlaybackCoordinator.cs`:

```csharp
namespace CompasAcoustics.Audio;

/// <summary>Keeps at most one <see cref="AudioPlayer"/> playing per process.</summary>
public static class PlaybackCoordinator
{
    private static readonly object Gate = new object();
    private static AudioPlayer? _active;

    public static void Activate(AudioPlayer player)
    {
        AudioPlayer? previous;
        lock (Gate)
        {
            previous = ReferenceEquals(_active, player) ? null : _active;
            _active = player;
        }
        previous?.Stop();
    }

    public static void Release(AudioPlayer player)
    {
        lock (Gate)
        {
            if (ReferenceEquals(_active, player))
            {
                _active = null;
            }
        }
    }

    public static void StopAll()
    {
        AudioPlayer? active;
        lock (Gate)
        {
            active = _active;
            _active = null;
        }
        active?.Stop();
    }
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `dotnet test CompasAcoustics.sln`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add CompasAcoustics.Core CompasAcoustics.Core.Tests
git commit -m "Add AudioPlayer, playback coordinator and Windows/macOS outputs"
```

---

### Task 9: Grasshopper plugin and AudioPlayerComponent

**Files:**
- Create: `CompasAcoustics.GH/CompasAcoustics.GH.csproj`, `CompasAcoustics.GH/CompasAcousticsInfo.cs`, `CompasAcoustics.GH/Components/GooUnwrap.cs`, `CompasAcoustics.GH/Components/AudioPlayerComponent.cs`
- Create: `CompasAcoustics.GH/Resources/audio_player.png` (copy of `src/compas_acoustics_ghpython/components/CA_Audio_Player/icon.png`)

**Interfaces:**
- Consumes: `AudioInputResolver`, `AudioPlayer`, `PlaybackCoordinator`, `PlaybackState`, exceptions (Tasks 5-8).
- Produces (used by Task 10's attributes): on `AudioPlayerComponent`: `AudioBuffer? Buffer`, `PlaybackState State`, `double PositionFraction`, `double PositionSeconds`, `bool IsLoading`, `double Volume`, `int Channel` (0-based), `bool VolumeDrivenByInput`, `bool ChannelDrivenByInput`, `void TogglePlayPause()`, `void StopPlayback()`, `void SeekTo(double fraction)`, `void SetVolumeFromFace(double volume)`, `void SelectChannelFromFace(int channel)`.

There is no automated test harness for Grasshopper; this task is verified by building and a manual smoke test.

- [ ] **Step 1: Create the project**

```bash
mkdir -p CompasAcoustics.GH/Components CompasAcoustics.GH/Resources
cp ../compas_acoustics_ghpython/components/CA_Audio_Player/icon.png CompasAcoustics.GH/Resources/audio_player.png
```

`CompasAcoustics.GH/CompasAcoustics.GH.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net48</TargetFramework>
    <AssemblyName>CompasAcoustics.GH</AssemblyName>
    <RootNamespace>CompasAcoustics.GH</RootNamespace>
    <TargetExt>.gha</TargetExt>
    <CopyLocalLockFileAssemblies>true</CopyLocalLockFileAssemblies>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Grasshopper" Version="8.0.23304.9001" ExcludeAssets="runtime" />
    <PackageReference Include="RhinoCommon" Version="8.0.23304.9001" ExcludeAssets="runtime" />
    <PackageReference Include="Microsoft.NETFramework.ReferenceAssemblies" Version="1.0.3" PrivateAssets="all" />
  </ItemGroup>
  <ItemGroup>
    <Reference Include="System.Drawing" />
    <Reference Include="System.Windows.Forms" />
  </ItemGroup>
  <ItemGroup>
    <ProjectReference Include="../CompasAcoustics.Core/CompasAcoustics.Core.csproj" />
  </ItemGroup>
  <ItemGroup>
    <EmbeddedResource Include="Resources/audio_player.png" LogicalName="CompasAcoustics.GH.audio_player.png" />
  </ItemGroup>
</Project>
```

```bash
dotnet sln add CompasAcoustics.GH/CompasAcoustics.GH.csproj
```

- [ ] **Step 2: Plugin info** `CompasAcoustics.GH/CompasAcousticsInfo.cs`

```csharp
using System;
using CompasAcoustics.Audio;
using Grasshopper.Kernel;

namespace CompasAcoustics.GH;

public sealed class CompasAcousticsInfo : GH_AssemblyInfo
{
    public override string Name => "COMPAS Acoustics";

    public override string Description => "Grasshopper components for COMPAS Acoustics.";

    public override Guid Id => new Guid("0d7c2a5e-3f4b-4a6e-9b1d-8c2e5f7a9b31");

    public override string AuthorName => "Achilleas Xydis";

    public override string AuthorContact => "xydis@arch.ethz.ch";

    public override string AssemblyVersion => GetType().Assembly.GetName().Version.ToString();
}

public sealed class CompasAcousticsPriority : GH_AssemblyPriority
{
    public override GH_LoadingInstruction PriorityLoad()
    {
        Rhino.RhinoApp.Closing += (sender, e) => PlaybackCoordinator.StopAll();
        return GH_LoadingInstruction.Proceed;
    }
}
```

- [ ] **Step 3: Goo unwrapping** `CompasAcoustics.GH/Components/GooUnwrap.cs`

```csharp
using Grasshopper.Kernel.Types;

namespace CompasAcoustics.GH.Components;

internal static class GooUnwrap
{
    public static object Unwrap(object value)
    {
        object current = value;
        for (int depth = 0; depth < 4 && current is IGH_Goo goo; depth++)
        {
            current = goo.ScriptVariable();
        }
        return current;
    }
}
```

- [ ] **Step 4: The component** `CompasAcoustics.GH/Components/AudioPlayerComponent.cs`

```csharp
using System;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
using CompasAcoustics.Audio;
using GH_IO.Serialization;
using Grasshopper;
using Grasshopper.Kernel;
using Rhino;

namespace CompasAcoustics.GH.Components;

public sealed class AudioPlayerComponent : GH_Component
{
    private const int InAudio = 0;
    private const int InPlay = 1;
    private const int InStop = 2;
    private const int InSeek = 3;
    private const int InVolume = 4;
    private const int InChannel = 5;

    private static readonly Lazy<Bitmap?> IconBitmap = new Lazy<Bitmap?>(LoadIcon);

    private AudioPlayer? _player;
    private string? _inputKey;
    private int _loadGeneration;
    private string? _loadError;
    private string? _deviceError;
    private bool _lastPlay;
    private bool _lastStop;
    private double? _lastSeek;
    private bool _pendingPlay;
    private bool _pendingStop;
    private double? _pendingSeek;
    private Timer? _redrawTimer;

    public AudioPlayerComponent()
        : base(
            "Audio Player",
            "Player",
            "Play WAV audio from a file path, a Source or an audio signal, with waveform, seek and volume.",
            "COMPAS Acoustics",
            "06 Util") { }

    public override Guid ComponentGuid => new Guid("5b0a8f8e-2a41-4a8e-9d55-6a1f0c7e3b21");

    public override GH_Exposure Exposure => GH_Exposure.secondary;

    protected override Bitmap? Icon => IconBitmap.Value;

    public AudioBuffer? Buffer => _player?.Engine.Buffer;

    public PlaybackState State => _player?.State ?? PlaybackState.Stopped;

    public double PositionFraction => _player?.Engine.PositionFraction ?? 0.0;

    public double PositionSeconds => _player?.Engine.PositionSeconds ?? 0.0;

    public bool IsLoading { get; private set; }

    public double Volume { get; private set; } = 1.0;

    public int Channel { get; private set; }

    public bool VolumeDrivenByInput => Params.Input[InVolume].SourceCount > 0;

    public bool ChannelDrivenByInput => Params.Input[InChannel].SourceCount > 0;

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddGenericParameter("audio", "audio", "WAV file path, or the 'pb' output of Source, Auralise or To compas_pb.", GH_ParamAccess.item);
        pManager.AddBooleanParameter("play", "play", "Start playback when this turns true.", GH_ParamAccess.item);
        pManager.AddBooleanParameter("stop", "stop", "Stop playback and return to the start when this turns true.", GH_ParamAccess.item);
        pManager.AddNumberParameter("seek", "seek", "Playback position as a fraction of the duration (0-1).", GH_ParamAccess.item);
        pManager.AddNumberParameter("volume", "volume", "Volume (0-1). Default 1.", GH_ParamAccess.item);
        pManager.AddIntegerParameter("channel", "channel", "Channel to play when the audio has more than two channels (1-based). Default 1.", GH_ParamAccess.item);
        for (int i = 0; i < pManager.ParamCount; i++)
        {
            pManager[i].Optional = true;
        }
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager) { }

    protected override void SolveInstance(IGH_DataAccess access)
    {
        ReadSettings(access);

        object? raw = null;
        if (!access.GetData(InAudio, ref raw) || raw is null)
        {
            ClearAudio();
            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, "Connect a WAV file path or a 'pb' output.");
            return;
        }

        object input = GooUnwrap.Unwrap(raw);
        string key;
        try
        {
            key = AudioInputResolver.CacheKey(input);
        }
        catch (AudioInputException ex)
        {
            ClearAudio();
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, ex.Message);
            return;
        }

        if (key != _inputKey)
        {
            StartLoading(input, key);
        }

        ReadCommands(access);
        ApplyToPlayer(inSolution: true);
        ReportMessages();
    }

    public void TogglePlayPause()
    {
        if (_player is null)
        {
            return;
        }
        if (_player.State == PlaybackState.Playing)
        {
            _player.Pause();
        }
        else
        {
            TryPlay(inSolution: false);
        }
    }

    public void StopPlayback() => _player?.Stop();

    public void SeekTo(double fraction) => _player?.Engine.Seek(Clamp01(fraction));

    public void SetVolumeFromFace(double volume)
    {
        if (VolumeDrivenByInput)
        {
            return;
        }
        Volume = Clamp01(volume);
        if (_player != null)
        {
            _player.Engine.Volume = (float)Volume;
        }
    }

    public void SelectChannelFromFace(int channel)
    {
        if (ChannelDrivenByInput)
        {
            return;
        }
        Channel = Math.Max(0, channel);
        if (_player != null)
        {
            _player.Engine.Channel = Channel;
        }
    }

    public override void RemovedFromDocument(GH_Document document)
    {
        ClearAudio();
        base.RemovedFromDocument(document);
    }

    public override void DocumentContextChanged(GH_Document document, GH_DocumentContext context)
    {
        if (context == GH_DocumentContext.Close || context == GH_DocumentContext.Unloaded)
        {
            _player?.Stop();
        }
        base.DocumentContextChanged(document, context);
    }

    public override bool Write(GH_IWriter writer)
    {
        writer.SetDouble("Volume", Volume);
        writer.SetInt32("Channel", Channel);
        return base.Write(writer);
    }

    public override bool Read(GH_IReader reader)
    {
        if (reader.ItemExists("Volume"))
        {
            Volume = Clamp01(reader.GetDouble("Volume"));
        }
        if (reader.ItemExists("Channel"))
        {
            Channel = Math.Max(0, reader.GetInt32("Channel"));
        }
        return base.Read(reader);
    }

    private void ReadSettings(IGH_DataAccess access)
    {
        double volume = 1.0;
        if (access.GetData(InVolume, ref volume))
        {
            Volume = Clamp01(volume);
        }
        int channel = 1;
        if (access.GetData(InChannel, ref channel))
        {
            Channel = channel - 1;
        }
    }

    private void ReadCommands(IGH_DataAccess access)
    {
        bool play = false;
        bool stop = false;
        access.GetData(InPlay, ref play);
        access.GetData(InStop, ref stop);
        if (play && !_lastPlay)
        {
            _pendingPlay = true;
        }
        if (stop && !_lastStop)
        {
            _pendingStop = true;
            _pendingPlay = false;
        }
        _lastPlay = play;
        _lastStop = stop;

        double seek = 0.0;
        if (access.GetData(InSeek, ref seek))
        {
            if (_lastSeek != seek)
            {
                _lastSeek = seek;
                _pendingSeek = Clamp01(seek);
            }
        }
        else
        {
            _lastSeek = null;
        }
    }

    private void ApplyToPlayer(bool inSolution)
    {
        if (_player is null)
        {
            return;
        }

        var engine = _player.Engine;
        engine.Volume = (float)Volume;
        engine.Channel = Channel;
        if (_pendingSeek is double seek)
        {
            engine.Seek(seek);
            _pendingSeek = null;
        }
        if (_pendingStop)
        {
            _pendingStop = false;
            _player.Stop();
        }
        if (_pendingPlay)
        {
            _pendingPlay = false;
            TryPlay(inSolution);
        }
    }

    private void TryPlay(bool inSolution)
    {
        string? previous = _deviceError;
        try
        {
            _player!.Play();
            _deviceError = null;
        }
        catch (AudioDeviceException ex)
        {
            _deviceError = ex.Message;
        }
        if (!inSolution && previous != _deviceError)
        {
            ExpireSolution(true);
        }
    }

    private void ReportMessages()
    {
        if (IsLoading)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, "Loading audio...");
        }
        if (_loadError != null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, _loadError);
        }
        if (_deviceError != null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, _deviceError);
        }

        var buffer = Buffer;
        if (buffer is null)
        {
            return;
        }
        if (buffer.Frames == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "The audio is empty.");
        }
        if (buffer.Channels > 2 && ChannelDrivenByInput && (Channel < 0 || Channel >= buffer.Channels))
        {
            int used = Math.Max(0, Math.Min(buffer.Channels - 1, Channel));
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Channel {Channel + 1} is out of range for {buffer.Channels}-channel audio; playing channel {used + 1}.");
        }
    }

    private void StartLoading(object input, string key)
    {
        ClearAudio();
        _inputKey = key;
        IsLoading = true;
        int generation = ++_loadGeneration;
        Task.Run(() => AudioInputResolver.Resolve(input))
            .ContinueWith(task => RhinoApp.InvokeOnUiThread((Action)(() => OnLoaded(generation, task))));
    }

    private void OnLoaded(int generation, Task<AudioBuffer> task)
    {
        if (generation != _loadGeneration)
        {
            return;
        }

        IsLoading = false;
        if (task.IsFaulted)
        {
            _loadError = task.Exception!.GetBaseException().Message;
        }
        else
        {
            _player = new AudioPlayer(task.Result);
            _player.StateChanged += OnPlayerStateChanged;
            ApplyToPlayer(inSolution: false);
        }
        Attributes.ExpireLayout();
        ExpireSolution(true);
    }

    private void ClearAudio()
    {
        _loadGeneration++;
        IsLoading = false;
        _inputKey = null;
        _loadError = null;
        _deviceError = null;
        if (_player != null)
        {
            _player.StateChanged -= OnPlayerStateChanged;
            _player.Dispose();
            _player = null;
        }
        StopRedraw();
        Attributes?.ExpireLayout();
    }

    private void OnPlayerStateChanged(object? sender, EventArgs e) =>
        RhinoApp.InvokeOnUiThread((Action)(() =>
        {
            if (_player?.State == PlaybackState.Playing)
            {
                StartRedraw();
            }
            else
            {
                StopRedraw();
            }
            Instances.InvalidateCanvas();
        }));

    private void StartRedraw() =>
        _redrawTimer ??= new Timer(_ => RhinoApp.InvokeOnUiThread((Action)Instances.InvalidateCanvas), null, 0, 33);

    private void StopRedraw()
    {
        _redrawTimer?.Dispose();
        _redrawTimer = null;
    }

    private static double Clamp01(double value) => Math.Max(0.0, Math.Min(1.0, value));

    private static Bitmap? LoadIcon()
    {
        using var stream = typeof(AudioPlayerComponent).Assembly.GetManifestResourceStream("CompasAcoustics.GH.audio_player.png");
        if (stream is null)
        {
            return null;
        }
        using var image = Image.FromStream(stream);
        return new Bitmap(image, 24, 24);
    }
}
```

- [ ] **Step 5: Build**

Run: `dotnet build CompasAcoustics.GH/CompasAcoustics.GH.csproj`
Expected: `Build succeeded`, output `CompasAcoustics.GH/bin/Debug/net48/CompasAcoustics.GH.gha` next to `CompasAcoustics.Core.dll`, `CompasPb.dll`, `Google.Protobuf.dll`, `NAudio.*.dll`, and no `Grasshopper.dll` or `RhinoCommon.dll`.

- [ ] **Step 6: Manual smoke test (macOS Rhino 8)**

1. In Rhino, run `GrasshopperDeveloperSettings`, add the folder `src/compas_acoustics_ghcsharp/CompasAcoustics.GH/bin/Debug/net48`, restart Rhino and open Grasshopper.
2. Place COMPAS Acoustics > 06 Util > Audio Player. Connect a Panel containing the absolute path of `examples/rir_mono.wav` to `audio`, and a Button to `play` and another to `stop`.
3. Expected: no errors; pressing play plays the file, stop stops it. A Panel with a `.mp3` path shows the "Only WAV files are supported" error.

- [ ] **Step 7: Commit**

```bash
git add CompasAcoustics.sln CompasAcoustics.GH
git commit -m "Add Grasshopper plugin with C# Audio Player component"
```

---

### Task 10: Component face (waveform, buttons, seek, volume)

**Files:**
- Create: `CompasAcoustics.GH/Components/AudioPlayerAttributes.cs`
- Modify: `CompasAcoustics.GH/Components/AudioPlayerComponent.cs` (add `CreateAttributes`)

**Interfaces:**
- Consumes: the public members of `AudioPlayerComponent` listed in Task 9; `AudioBuffer.GetPeaks` (Task 4).

- [ ] **Step 1: Hook up the attributes** in `AudioPlayerComponent.cs`, after `RegisterOutputParams`:

```csharp
    public override void CreateAttributes() => m_attributes = new AudioPlayerAttributes(this);
```

- [ ] **Step 2: Implement** `CompasAcoustics.GH/Components/AudioPlayerAttributes.cs`

```csharp
using System;
using System.Drawing;
using System.Windows.Forms;
using CompasAcoustics.Audio;
using Grasshopper.GUI;
using Grasshopper.GUI.Canvas;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Attributes;

namespace CompasAcoustics.GH.Components;

internal sealed class AudioPlayerAttributes : GH_ComponentAttributes
{
    private const float MinWidth = 320f;
    private const float Pad = 6f;
    private const float InfoHeight = 16f;
    private const float ControlsHeight = 20f;
    private const float LaneHeight = 28f;
    private const float LaneGap = 2f;
    private const float LabelWidth = 28f;
    private const float MoreHeight = 14f;
    private const float VolumeWidth = 110f;
    private const int MaxLanes = 8;

    private static readonly StringFormat Centered = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center };
    private static readonly Color WaveColor = Color.FromArgb(70, 130, 180);
    private static readonly Color DimWaveColor = Color.FromArgb(170, 190, 210);

    private readonly AudioPlayerComponent _owner;
    private RectangleF _panel;
    private RectangleF _info;
    private RectangleF _playButton;
    private RectangleF _stopButton;
    private RectangleF _volumeTrack;
    private RectangleF _waveformArea;
    private RectangleF _moreText;
    private RectangleF[] _lanes = Array.Empty<RectangleF>();
    private RectangleF[] _labels = Array.Empty<RectangleF>();
    private bool _draggingSeek;
    private bool _draggingVolume;

    public AudioPlayerAttributes(AudioPlayerComponent owner)
        : base(owner) => _owner = owner;

    private RectangleF VolumeHitArea => RectangleF.Inflate(_volumeTrack, 6f, 6f);

    protected override void Layout()
    {
        base.Layout();
        RectangleF top = Bounds;
        float width = Math.Max(top.Width, MinWidth);
        int channels = _owner.Buffer?.Channels ?? 0;
        int lanes = Math.Max(1, Math.Min(channels, MaxLanes));

        float x = top.X + Pad;
        float inner = width - 2 * Pad;
        float y = top.Bottom + Pad;

        _info = new RectangleF(x, y, inner, InfoHeight);
        y += InfoHeight;

        _playButton = new RectangleF(x, y + 1, ControlsHeight - 2, ControlsHeight - 2);
        _stopButton = new RectangleF(_playButton.Right + 4, y + 1, ControlsHeight - 2, ControlsHeight - 2);
        _volumeTrack = new RectangleF(x + inner - VolumeWidth - 6, y + ControlsHeight / 2 - 3, VolumeWidth, 6);
        y += ControlsHeight + 2;

        _lanes = new RectangleF[lanes];
        _labels = new RectangleF[lanes];
        for (int i = 0; i < lanes; i++)
        {
            _labels[i] = new RectangleF(x, y, LabelWidth, LaneHeight);
            _lanes[i] = new RectangleF(x + LabelWidth, y, inner - LabelWidth, LaneHeight);
            y += LaneHeight + LaneGap;
        }
        _waveformArea = RectangleF.FromLTRB(_lanes[0].Left, _lanes[0].Top, _lanes[lanes - 1].Right, _lanes[lanes - 1].Bottom);

        if (channels > MaxLanes)
        {
            _moreText = new RectangleF(x, y, inner, MoreHeight);
            y += MoreHeight;
        }
        else
        {
            _moreText = RectangleF.Empty;
        }

        _panel = RectangleF.FromLTRB(top.X + 2, top.Bottom + 2, top.X + width - 2, y + Pad - 2);
        Bounds = RectangleF.FromLTRB(top.X, top.Y, top.X + width, y + Pad);
    }

    protected override void Render(GH_Canvas canvas, Graphics graphics, GH_CanvasChannel channel)
    {
        base.Render(canvas, graphics, channel);
        if (channel != GH_CanvasChannel.Objects)
        {
            return;
        }

        using (var background = new SolidBrush(Color.FromArgb(235, 235, 235)))
        {
            graphics.FillRectangle(background, _panel);
        }

        AudioBuffer? buffer = _owner.Buffer;
        DrawInfo(graphics, buffer);
        DrawButtons(graphics, buffer != null);
        DrawVolume(graphics);
        DrawLanes(graphics, buffer);
    }

    public override GH_ObjectResponse RespondToMouseDown(GH_Canvas sender, GH_CanvasMouseEvent e)
    {
        if (e.Button != MouseButtons.Left)
        {
            return base.RespondToMouseDown(sender, e);
        }

        PointF point = e.CanvasLocation;
        if (_playButton.Contains(point))
        {
            _owner.TogglePlayPause();
            sender.Invalidate();
            return GH_ObjectResponse.Handled;
        }
        if (_stopButton.Contains(point))
        {
            _owner.StopPlayback();
            sender.Invalidate();
            return GH_ObjectResponse.Handled;
        }
        if (!_owner.VolumeDrivenByInput && VolumeHitArea.Contains(point))
        {
            _draggingVolume = true;
            SetVolumeAt(point.X);
            sender.Invalidate();
            return GH_ObjectResponse.Capture;
        }

        AudioBuffer? buffer = _owner.Buffer;
        if (buffer != null)
        {
            if (buffer.Channels > 2 && !_owner.ChannelDrivenByInput)
            {
                for (int i = 0; i < _labels.Length; i++)
                {
                    if (_labels[i].Contains(point))
                    {
                        _owner.SelectChannelFromFace(i);
                        sender.Invalidate();
                        return GH_ObjectResponse.Handled;
                    }
                }
            }
            if (_waveformArea.Contains(point))
            {
                _draggingSeek = true;
                SeekAt(point.X);
                sender.Invalidate();
                return GH_ObjectResponse.Capture;
            }
        }
        return base.RespondToMouseDown(sender, e);
    }

    public override GH_ObjectResponse RespondToMouseMove(GH_Canvas sender, GH_CanvasMouseEvent e)
    {
        if (_draggingVolume)
        {
            SetVolumeAt(e.CanvasLocation.X);
            sender.Invalidate();
            return GH_ObjectResponse.Handled;
        }
        if (_draggingSeek)
        {
            SeekAt(e.CanvasLocation.X);
            sender.Invalidate();
            return GH_ObjectResponse.Handled;
        }
        return base.RespondToMouseMove(sender, e);
    }

    public override GH_ObjectResponse RespondToMouseUp(GH_Canvas sender, GH_CanvasMouseEvent e)
    {
        if (_draggingVolume || _draggingSeek)
        {
            _draggingVolume = false;
            _draggingSeek = false;
            return GH_ObjectResponse.Release;
        }
        return base.RespondToMouseUp(sender, e);
    }

    private void SetVolumeAt(float x) => _owner.SetVolumeFromFace((x - _volumeTrack.Left) / _volumeTrack.Width);

    private void SeekAt(float x) => _owner.SeekTo((x - _waveformArea.Left) / _waveformArea.Width);

    private void DrawInfo(Graphics graphics, AudioBuffer? buffer)
    {
        string text = _owner.IsLoading
            ? "Loading..."
            : buffer is null
                ? "No audio"
                : $"{buffer.Channels} ch | {buffer.SampleRate / 1000.0:0.#} kHz | {FormatTime(_owner.PositionSeconds)} / {FormatTime(buffer.Duration)}";
        graphics.DrawString(text, GH_FontServer.Small, Brushes.Black, _info, Centered);
    }

    private void DrawButtons(Graphics graphics, bool enabled)
    {
        using var brush = new SolidBrush(enabled ? Color.FromArgb(60, 60, 60) : Color.Silver);
        foreach (RectangleF button in new[] { _playButton, _stopButton })
        {
            graphics.FillRectangle(Brushes.White, button);
            graphics.DrawRectangle(Pens.Gray, button.X, button.Y, button.Width, button.Height);
        }

        RectangleF play = RectangleF.Inflate(_playButton, -4f, -4f);
        if (_owner.State == PlaybackState.Playing)
        {
            float bar = play.Width / 3f;
            graphics.FillRectangle(brush, play.Left, play.Top, bar, play.Height);
            graphics.FillRectangle(brush, play.Right - bar, play.Top, bar, play.Height);
        }
        else
        {
            graphics.FillPolygon(brush, new[] { new PointF(play.Left, play.Top), new PointF(play.Right, play.Top + play.Height / 2f), new PointF(play.Left, play.Bottom) });
        }
        graphics.FillRectangle(brush, RectangleF.Inflate(_stopButton, -5f, -5f));
    }

    private void DrawVolume(Graphics graphics)
    {
        bool enabled = !_owner.VolumeDrivenByInput;
        graphics.DrawString("vol", GH_FontServer.Small, enabled ? Brushes.Black : Brushes.Gray, _volumeTrack.Left - 22f, _volumeTrack.Top - 4f);
        using (var track = new SolidBrush(Color.FromArgb(200, 200, 200)))
        {
            graphics.FillRectangle(track, _volumeTrack);
        }

        float filled = _volumeTrack.Width * (float)_owner.Volume;
        using (var fill = new SolidBrush(enabled ? WaveColor : Color.Silver))
        {
            graphics.FillRectangle(fill, _volumeTrack.Left, _volumeTrack.Top, filled, _volumeTrack.Height);
        }

        float centerY = _volumeTrack.Top + _volumeTrack.Height / 2f;
        graphics.FillEllipse(enabled ? Brushes.White : Brushes.Gainsboro, _volumeTrack.Left + filled - 5f, centerY - 5f, 10f, 10f);
        graphics.DrawEllipse(Pens.Gray, _volumeTrack.Left + filled - 5f, centerY - 5f, 10f, 10f);
    }

    private void DrawLanes(Graphics graphics, AudioBuffer? buffer)
    {
        if (buffer is null)
        {
            graphics.FillRectangle(Brushes.White, _waveformArea);
            return;
        }

        WaveformPeaks peaks = buffer.GetPeaks(Math.Max(1, (int)_lanes[0].Width));
        bool selectable = buffer.Channels > 2;
        int active = Math.Max(0, Math.Min(buffer.Channels - 1, _owner.Channel));

        for (int i = 0; i < _lanes.Length; i++)
        {
            bool isActive = !selectable || i == active;
            if (selectable && isActive)
            {
                using var highlight = new SolidBrush(Color.FromArgb(210, 225, 240));
                graphics.FillRectangle(highlight, _labels[i]);
            }
            graphics.DrawString($"ch{i + 1}", GH_FontServer.Small, Brushes.Black, _labels[i], Centered);
            graphics.FillRectangle(Brushes.White, _lanes[i]);
            DrawEnvelope(graphics, _lanes[i], peaks.Min[i], peaks.Max[i], isActive ? WaveColor : DimWaveColor);
        }

        if (!_moreText.IsEmpty)
        {
            string more = $"+{buffer.Channels - MaxLanes} more channels";
            if (active >= MaxLanes)
            {
                more += $", playing ch{active + 1}";
            }
            graphics.DrawString(more, GH_FontServer.Small, Brushes.DimGray, _moreText, Centered);
        }

        float x = _waveformArea.Left + _waveformArea.Width * (float)_owner.PositionFraction;
        using var playhead = new Pen(Color.Firebrick, 1.5f);
        graphics.DrawLine(playhead, x, _waveformArea.Top, x, _waveformArea.Bottom);
    }

    private static void DrawEnvelope(Graphics graphics, RectangleF lane, float[] min, float[] max, Color color)
    {
        float mid = lane.Top + lane.Height / 2f;
        graphics.DrawLine(Pens.Gainsboro, lane.Left, mid, lane.Right, mid);

        int count = min.Length;
        if (count == 0)
        {
            return;
        }

        float half = lane.Height / 2f - 1f;
        float step = lane.Width / count;
        var points = new PointF[2 * count];
        for (int i = 0; i < count; i++)
        {
            float x = lane.Left + (i + 0.5f) * step;
            points[i] = new PointF(x, mid - Clamp(max[i]) * half);
            points[2 * count - 1 - i] = new PointF(x, mid - Clamp(min[i]) * half);
        }
        using var brush = new SolidBrush(color);
        graphics.FillPolygon(brush, points);
    }

    private static float Clamp(float value) => Math.Max(-1f, Math.Min(1f, value));

    private static string FormatTime(double seconds)
    {
        TimeSpan time = TimeSpan.FromSeconds(seconds);
        return $"{(int)time.TotalMinutes:00}:{time.Seconds:00}.{time.Milliseconds / 100}";
    }
}
```

- [ ] **Step 3: Build**

Run: `dotnet build CompasAcoustics.GH/CompasAcoustics.GH.csproj`
Expected: `Build succeeded`

- [ ] **Step 4: Manual check (macOS Rhino 8)**

With the setup from Task 9 Step 6, restart Rhino and verify:
1. The face shows `1 ch | 48 kHz | 00:00.0 / 00:0x.x`, one waveform lane, play/stop buttons and a volume slider.
2. The play button toggles play and pause; the playhead moves while playing and the time updates.
3. Clicking or dragging on the waveform seeks while playing.
4. Dragging the volume slider changes loudness smoothly. Connecting a Number Slider to `volume` greys out the face slider.
5. Connect `examples/rir_ambi.wav`: one lane per channel; clicking a lane label selects it (highlighted) and changes what you hear.

- [ ] **Step 5: Commit**

```bash
git add CompasAcoustics.GH
git commit -m "Draw Audio Player face with waveform, transport, seek and volume"
```

---

### Task 11: Python components output compas_pb bytes

**Files:**
- Create: `src/compas_acoustics_ghpython/pb.py`
- Modify: `src/compas_acoustics_ghpython/components/CA_Source/code.py`, `metadata.json`
- Modify: `src/compas_acoustics_ghpython/components/CA_Auralise/code.py`, `metadata.json`
- Create: `src/compas_acoustics_ghpython/components/CA_To_compas_pb/code.py`, `metadata.json`, `icon.png`

**Interfaces:**
- Consumes: `compas_acoustics.pb.to_pb_bytes` and `compas_acoustics.AudioSignal` (Task 2).
- Produces: `compas_acoustics_ghpython.pb.to_dotnet_bytes(data: bytes) -> System.Byte[]`; `pb` outputs on Source, Auralise and To compas_pb.

These run only inside Rhino 8 (they import `System` / `Grasshopper`), so they are verified manually.

- [ ] **Step 1: Create** `src/compas_acoustics_ghpython/pb.py`

```python
"""Hand compas_pb bytes to the C# Grasshopper components."""

import base64

__all__ = ["to_dotnet_bytes"]


def to_dotnet_bytes(data):
    """Convert Python ``bytes`` into a .NET ``byte[]`` that C# components can read.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    System.Byte[]
    """
    import System  # noqa: E402

    return System.Convert.FromBase64String(base64.b64encode(data).decode("ascii"))
```

- [ ] **Step 2: Source gets a `pb` output**

In `components/CA_Source/code.py`, add imports:

```python
from compas_acoustics.pb import to_pb_bytes
from compas_acoustics_ghpython.pb import to_dotnet_bytes
```

and replace the final `return Source(...)` with:

```python
        source = Source(
            name,
            *position,
            directivity=dirt,
            direction=[*direction],
            sound_file=sound_file
        )
        return source, to_dotnet_bytes(to_pb_bytes(source))
```

In `components/CA_Source/metadata.json`, append to `outputParameters`:

```json
            {
                "name": "pb",
                "description": "The source as compas_pb bytes, for C# components such as Audio Player."
            }
```

- [ ] **Step 3: Auralise gets a `pb` output**

In `components/CA_Auralise/code.py`, add imports:

```python
from compas_acoustics import AudioSignal
from compas_acoustics.pb import to_pb_bytes
from compas_acoustics_ghpython.pb import to_dotnet_bytes
```

After `signals_out = DataTree[object]()` add:

```python
        pb_out = DataTree[object]()
        sample_rate = 48000
```

Change `sample_rate=48000` in the `Auraliser(...)` call to `sample_rate=sample_rate`, and after `signals_out.Add(convolved_matrix, path)` add:

```python
                    pb_out.Add(to_dotnet_bytes(to_pb_bytes(AudioSignal(convolved_matrix, sample_rate))), path)
```

Change `return signals_out` to `return signals_out, pb_out`.

In `components/CA_Auralise/metadata.json`, append to `outputParameters`:

```json
            {
                "name": "pb",
                "description": "Convolved signal(s) as compas_pb bytes, for C# components such as Audio Player."
            }
```

- [ ] **Step 4: New To compas_pb component**

```bash
mkdir -p src/compas_acoustics_ghpython/components/CA_To_compas_pb
cp src/compas_acoustics_ghpython/components/CA_Audio_Player/icon.png src/compas_acoustics_ghpython/components/CA_To_compas_pb/icon.png
```

`components/CA_To_compas_pb/code.py`:

```python
# flake8: noqa
import Grasshopper
from compas_acoustics.pb import to_pb_bytes
from compas_acoustics_ghpython.pb import to_dotnet_bytes


class ToCompasPb(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, data, sample_rate):
        if data is None:
            return None
        try:
            return to_dotnet_bytes(to_pb_bytes(data, sample_rate))
        except (TypeError, ValueError) as e:
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, str(e))
            return None
```

`components/CA_To_compas_pb/metadata.json`:

```json
{
    "name": "To compas_pb",
    "nickname": "ToPb",
    "category": "COMPAS Acoustics",
    "subcategory": "06 Util",
    "description": "Serialize a Source, an AudioSignal or a numpy array of samples to compas_pb bytes for the C# components.",
    "exposure": 2,
    "ghpython": {
        "isAdvancedMode": true,
        "iconDisplay": 0,
        "inputParameters": [
            {
                "name": "data",
                "description": "Source, AudioSignal, or numpy array of shape (frames,) or (frames, channels).",
                "optional": false,
                "scriptParamAccess": 0,
                "typeHintID": "ghdoc"
            },
            {
                "name": "sample_rate",
                "description": "Sample rate for numpy arrays. Default 48000.",
                "optional": true,
                "scriptParamAccess": 0,
                "typeHintID": "int"
            }
        ],
        "outputParameters": [
            {
                "name": "pb",
                "description": "compas_pb bytes."
            }
        ]
    }
}
```

- [ ] **Step 5: Build components and lint**

Run: `invoke lint` then (on Windows) `invoke build-cpython-ghuser-components`
Expected: lint clean; `components/ghuser/` contains `compas_acoustics: To compas_pb.ghuser` and updated Source/Auralise components.

- [ ] **Step 6: Manual check in Rhino 8**

1. Install the updated wheel and `.ghuser` files with the installer `.gh`.
2. Source component: connect a WAV path to `sound_file`; its `pb` output into the C# Audio Player plays that file.
3. Auralise: its `pb` output into the Audio Player shows the convolved signal's channel count and plays it.
4. A Python script outputting `np.random.uniform(-0.5, 0.5, (48000, 2))` through To compas_pb plays one second of stereo noise.
5. Connecting a raw numpy array straight to the Audio Player shows the error pointing to To compas_pb.

- [ ] **Step 7: Commit**

```bash
git add src/compas_acoustics_ghpython
git commit -m "Add compas_pb outputs to Source and Auralise, and a To compas_pb component"
```

---

### Task 12: Retire the Python player, build tasks, CI and docs

**Files:**
- Delete: `src/compas_acoustics_ghpython/components/CA_Audio_Player/`
- Modify: `tasks.py`, `.github/workflows/build.yml`, `CHANGELOG.md`, `README.md`

- [ ] **Step 1: Remove the Python component**

```bash
git rm -r src/compas_acoustics_ghpython/components/CA_Audio_Player
```

- [ ] **Step 2: Add build tasks** to `tasks.py`

Add below the existing tasks:

```python
GHCSHARP_DIR = os.path.join(os.path.dirname(__file__), "src", "compas_acoustics_ghcsharp")

YAK_MANIFEST = """name: compas-acoustics
version: {version}
authors:
- Achilleas Xydis
description: COMPAS Acoustics Grasshopper components (C#).
url: https://github.com/gramaziokohler/compas_acoustics
icon: icon.png
keywords:
- acoustics
- compas
"""


def _default_yak():
    if sys.platform == "darwin":
        return "/Applications/Rhino 8.app/Contents/Resources/bin/yak"
    return r"C:\Program Files\Rhino 8\System\Yak.exe"


@task(help={"configuration": "Build configuration (default: Release)"})
def build_ghcsharp(ctx, configuration="Release"):
    """Test the C# core library and build the Grasshopper plugin."""
    with ctx.cd(GHCSHARP_DIR):
        ctx.run("dotnet test CompasAcoustics.sln -c {}".format(configuration))
        ctx.run("dotnet build CompasAcoustics.GH/CompasAcoustics.GH.csproj -c {}".format(configuration))


@task(pre=[build_ghcsharp], help={"yak": "Path to the yak executable (default: the one shipped with Rhino 8)"})
def yak_package(ctx, yak=None):
    """Package the C# Grasshopper plugin as a Yak package in dist/yak."""
    import compas_acoustics

    staging = os.path.join(os.path.dirname(__file__), "dist", "yak")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(os.path.join(GHCSHARP_DIR, "CompasAcoustics.GH", "bin", "Release", "net48"), staging)
    shutil.copy2(os.path.join(GHCSHARP_DIR, "CompasAcoustics.GH", "Resources", "audio_player.png"), os.path.join(staging, "icon.png"))
    with open(os.path.join(staging, "manifest.yml"), "w") as f:
        f.write(YAK_MANIFEST.format(version=compas_acoustics.__version__))
    with ctx.cd(staging):
        ctx.run('"{}" build --platform any'.format(yak or _default_yak()))
```

Add `build_ghcsharp,` and `yak_package,` to the `Collection(...)` arguments.

Run: `invoke yak-package`
Expected: tests pass, then `dist/yak/compas-acoustics-0.1.0-rh8_0-any.yak` is created.

- [ ] **Step 3: CI job** in `.github/workflows/build.yml`, add under `jobs:`

```yaml
  ghcsharp:
    needs: release-change
    if: github.event_name == 'pull_request' || needs.release-change.outputs.is-release != 'true'
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v7.0.1
        with:
          persist-credentials: false

      - uses: actions/setup-dotnet@v4
        with:
          dotnet-version: "8.0.x"

      - name: Test core library
        run: dotnet test src/compas_acoustics_ghcsharp/CompasAcoustics.sln -c Release

      - name: Build Grasshopper plugin
        run: dotnet build src/compas_acoustics_ghcsharp/CompasAcoustics.GH/CompasAcoustics.GH.csproj -c Release

      - uses: actions/upload-artifact@v7.0.1
        if: matrix.os == 'windows-latest'
        with:
          name: compas_acoustics_ghcsharp
          path: src/compas_acoustics_ghcsharp/CompasAcoustics.GH/bin/Release/net48
          if-no-files-found: error
```

- [ ] **Step 4: Docs**

In `CHANGELOG.md` under `## Unreleased`:

```markdown
### Added

* Added a C# Grasshopper plugin (`src/compas_acoustics_ghcsharp`) with an Audio Player showing the waveform per channel, channel count, seek, volume and channel selection, on Windows and macOS.
* Added `AudioSignal` and `compas_pb` serializers for `AudioSignal` and `Source` (`compas_acoustics.pb`), with the schema in `protobuf_defs/compas_acoustics/generated/acoustics.proto`.
* Added `pb` outputs to the Source and Auralise components, and a To compas_pb component.

### Removed

* Removed the Python Audio Player component, replaced by the C# Audio Player.
```

In `README.md`, add an installation note:

```markdown
### Grasshopper (C#) components

The C# components are distributed as a Yak package. Build it with `invoke yak-package`, then install
`dist/yak/compas-acoustics-<version>-rh8_0-any.yak` from Rhino 8 with the `_PackageManager` command or `yak install`.
```

- [ ] **Step 5: Run all tests**

Run: `pytest` and `dotnet test src/compas_acoustics_ghcsharp/CompasAcoustics.sln`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add -A tasks.py .github/workflows/build.yml CHANGELOG.md README.md src/compas_acoustics_ghpython/components
git commit -m "Retire Python Audio Player; add C# build, Yak packaging and CI"
```

---

### Task 13: Cross-platform manual verification

Run on both macOS and Windows Rhino 8 with the Yak package from Task 12 installed (uninstall the developer-settings folder first) and the updated Python wheel and components.

- [ ] File path: play, pause (playhead stops), resume, stop (returns to start).
- [ ] Seek by clicking and dragging on the waveform while playing and while paused; seek via a Number Slider on `seek`.
- [ ] Volume via the face slider (no clicks when dragging) and via the `volume` input (face greyed out).
- [ ] Multichannel (`examples/rir_ambi.wav`): lane per channel, label click selects the played channel; `channel` input out of range shows the clamping warning.
- [ ] More than 8 channels: "+N more channels" row appears.
- [ ] Source `pb` input and Auralise `pb` input both play.
- [ ] Two Audio Players: starting one stops the other.
- [ ] Deleting the component while playing stops audio; closing the document stops audio.
- [ ] Volume and channel survive save and reopen of the `.gh` file.
- [ ] `.mp3` path, missing file and raw numpy input each show the expected error message.
