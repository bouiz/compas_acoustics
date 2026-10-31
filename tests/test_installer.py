"""The installer script (installer/installer.py) outside Rhino, with Grasshopper and scriptcontext stubbed."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

INSTALLER = Path(__file__).parents[1] / "installer" / "installer.py"
GHA_FILES = ["CompasAcoustics.gha", "CompasPb.dll", "Google.Protobuf.dll", "Newtonsoft.Json.dll"]


@pytest.fixture
def installer(monkeypatch, tmp_path):
    folders = types.SimpleNamespace(DefaultAssemblyFolder=str(tmp_path / "Libraries"), DefaultUserObjectFolder=str(tmp_path / "UserObjects"))
    monkeypatch.setitem(sys.modules, "Grasshopper", types.SimpleNamespace(Folders=folders, Kernel=types.SimpleNamespace(GH_ScriptInstance=object)))
    monkeypatch.setitem(sys.modules, "scriptcontext", types.SimpleNamespace(sticky={}))
    spec = importlib.util.spec_from_file_location("compas_acoustics_installer", INSTALLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle(tmp_path):
    folder = tmp_path / "bundle"
    (folder / "gha").mkdir(parents=True)
    for name in GHA_FILES:
        (folder / "gha" / name).write_bytes(b"new " + name.encode())
    wheel = folder / "compas_acoustics-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    return wheel


def test_install_gha_copies_the_assembly_and_removes_previously_installed_userobjects(installer, tmp_path):
    wheel = bundle(tmp_path)
    old = tmp_path / "UserObjects" / "compas_acoustics"
    old.mkdir(parents=True)
    (old / "compas_acoustics: Room Scene.ghuser").write_bytes(b"old")
    stale = tmp_path / "UserObjects" / "compas_acoustics: Source.ghuser"
    stale.write_bytes(b"old")
    other = tmp_path / "UserObjects" / "someone_else.ghuser"
    other.write_bytes(b"keep")
    previous = tmp_path / "Libraries" / "compas_acoustics"
    previous.mkdir(parents=True)
    (previous / "Obsolete.dll").write_bytes(b"old")
    log = []
    installer.install_gha(str(wheel), log.append)
    installed = tmp_path / "Libraries" / "compas_acoustics"
    assert sorted(p.name for p in installed.iterdir()) == sorted(GHA_FILES)
    assert (installed / "CompasAcoustics.gha").read_bytes() == b"new CompasAcoustics.gha"
    assert not old.exists() and not stale.exists() and other.exists()
    assert any("CompasAcoustics.gha" in line for line in log)


def test_install_gha_without_the_plugin_next_to_the_wheel_says_what_to_download(installer, tmp_path):
    wheel = tmp_path / "compas_acoustics-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    with pytest.raises(RuntimeError, match="installer bundle"):
        installer.install_gha(str(wheel), print)


def test_the_installer_installs_the_worker_always(installer):
    assert '"--hybrid"' in INSTALLER.read_text()
