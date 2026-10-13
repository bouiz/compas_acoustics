#! python3
"""COMPAS Acoustics installer for Rhino 8 / Grasshopper.

This is the source of the script component inside ``compas_acoustics_installer.gh``.
It must stay self-contained: it runs before ``compas_acoustics`` is installed.

What it does, in a background thread so Rhino stays responsive:

1. Resolve the wheel to install: ``source`` can be a local ``.whl`` path,
an http(s) URL to a ``.whl``, or empty (picks the newest ``compas_acoustics-*.whl``
next to the ``.gh`` file).
2. Install it with Rhino's own CPython (``python -m pip install``), so it lands
in Rhino's base ``site-packages`` and is visible from every script component,
regardless of the script environment they use.
3. Copy the Grasshopper user objects (``.ghuser``) into the Grasshopper
UserObjects folder (``compas_acoustics`` subfolder).
4. Unload previously imported ``compas_acoustics`` modules.

Rhino must be restarted afterwards: reloading Grasshopper's component libraries
at runtime (``ComponentServer.LoadExternalFiles``) re-registers every plugin
and makes Grasshopper show "Component ID conflict" dialogs.

Inputs: source (str), install (bool, connect a button).
Outputs: log (str), ok (bool).

Rhino quirks to keep in mind when editing this file: Rhino pre-processes the
script text, so do not indent lines inside this docstring, and do not write
script directives (hash + venv/r/requirements) anywhere, not even in comments.
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback

import Grasshopper
import scriptcontext as sc

PACKAGE = "compas_acoustics"
STATE_KEY = "compas_acoustics_installer_state"


def rhino_python():
    """Path to the CPython executable bundled with Rhino 8."""
    if os.name == "nt":
        exe = os.path.join(sys.prefix, "python.exe")
    else:
        exe = os.path.join(sys.prefix, "python{}.{}".format(*sys.version_info[:2]))
    if not os.path.isfile(exe):
        raise RuntimeError("Could not find Rhino's Python executable at: {}".format(exe))
    return exe


def resolve_source(source, gh_file, log):
    source = (source or "").strip().strip('"')

    if not source:
        if not gh_file:
            raise RuntimeError("No source given and this .gh file is not saved: cannot look for a wheel next to it.")
        folder = os.path.dirname(gh_file)
        wheels = sorted(glob.glob(os.path.join(folder, PACKAGE + "-*.whl")), key=os.path.getmtime)
        if not wheels:
            raise RuntimeError("No source given and no {}-*.whl found next to the .gh file ({}).".format(PACKAGE, folder))
        source = wheels[-1]
        log("Using wheel found next to the .gh file: {}".format(os.path.basename(source)))

    if source.startswith(("http://", "https://")):
        return download(source, log)

    source = os.path.abspath(os.path.expanduser(source))
    if not os.path.isfile(source):
        raise RuntimeError("Wheel not found: {}".format(source))
    if not source.endswith(".whl"):
        raise RuntimeError("Source must be a .whl file: {}".format(source))
    return source


def download(url, log):
    from email.message import Message
    from urllib.parse import unquote
    from urllib.parse import urlparse
    from urllib.request import urlopen

    log("Downloading {} ...".format(url))
    with urlopen(url, timeout=120) as response:
        # the wheel filename carries the version/tags pip needs, so keep the server's name
        msg = Message()
        msg["content-disposition"] = response.headers.get("Content-Disposition", "")
        filename = msg.get_param("filename", header="content-disposition") or os.path.basename(unquote(urlparse(response.geturl()).path))
        if not filename.endswith(".whl"):
            raise RuntimeError("The URL does not point to a .whl file (got '{}'). Use a direct download link.".format(filename))
        target = os.path.join(tempfile.mkdtemp(prefix="compas_acoustics_"), filename)
        with open(target, "wb") as f:
            shutil.copyfileobj(response, f)
    log("Downloaded {} ({:.1f} MB)".format(filename, os.path.getsize(target) / 1e6))
    return target


def run(cmd, log):
    log("> " + " ".join('"{}"'.format(c) if " " in c else c for c in cmd))
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, **kwargs)
    for line in process.stdout:
        line = line.rstrip()
        if line:
            log("  " + line)
    if process.wait() != 0:
        raise RuntimeError("Command failed with exit code {}".format(process.returncode))


def query(python, code):
    kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
    return subprocess.check_output([python, "-c", code], universal_newlines=True, **kwargs).strip()


def install_userobjects(python, log):
    # locate the package without importing it: it needs RhinoCommon, which a plain Python process does not have
    package_dir = query(python, "import importlib.util as u; print(u.find_spec('compas_acoustics_ghpython').submodule_search_locations[0])")
    src = os.path.join(package_dir, "components", "ghuser")
    ghusers = sorted(glob.glob(os.path.join(src, "*.ghuser")))
    if not ghusers:
        raise RuntimeError("No .ghuser files found in {}".format(src))

    root = Grasshopper.Folders.DefaultUserObjectFolder
    dst = os.path.join(root, PACKAGE)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    for path in ghusers:
        name = os.path.basename(path)
        # remove stale copies from manual installs into the UserObjects root
        stale = os.path.join(root, name)
        if os.path.isfile(stale):
            os.remove(stale)
        shutil.copy2(path, os.path.join(dst, name))
    log("Installed {} Grasshopper components into {}".format(len(ghusers), dst))


def check_shadowing(python, log):
    """Warn when a package in a Rhino site-env hides the version pip just installed."""
    base = query(python, "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    for name in ("compas", "numpy", "scipy"):
        installed = query(python, "import {0}; print({0}.__version__)".format(name))
        for entry in sys.path:
            if os.path.normcase(os.path.abspath(entry)) == os.path.normcase(os.path.abspath(base)):
                break
            dist_infos = glob.glob(os.path.join(entry, name + "-*.dist-info"))
            if dist_infos:
                found = os.path.basename(dist_infos[0])[len(name) + 1 : -len(".dist-info")]
                if found != installed:
                    log("WARNING: {} {} in {} hides the installed {} {}.".format(name, found, entry, name, installed))
                break


def unload_modules():
    names = [n for n in sys.modules if n == PACKAGE or n.startswith(PACKAGE + ".") or n.startswith(PACKAGE + "_ghpython")]
    for name in names:
        del sys.modules[name]
    return len(names)


def run_install(state, source, gh_file):
    def log(message):
        state["log"].append(message)

    try:
        if sys.version_info[0] < 3:
            raise RuntimeError("This installer needs a Python 3 script component (Rhino 8 or newer).")

        python = rhino_python()
        log("Rhino Python {} at {}".format(sys.version.split()[0], python))

        wheel = resolve_source(source, gh_file, log)

        log("")
        log("Installing {} and its dependencies, this can take a few minutes...".format(os.path.basename(wheel)))
        run([python, "-m", "pip", "install", "--upgrade", "--no-warn-script-location", "--disable-pip-version-check", wheel], log)
        # the version number does not change between pre-release builds, so force a reinstall of the package itself
        run([python, "-m", "pip", "install", "--force-reinstall", "--no-deps", "--disable-pip-version-check", wheel], log)

        version = query(python, "import {0}; print({0}.__version__)".format(PACKAGE))
        log("")
        log("{} {} installed.".format(PACKAGE, version))

        install_userobjects(python, log)
        check_shadowing(python, log)
        unloaded = unload_modules()
        if unloaded:
            log("Unloaded {} previously imported compas_acoustics modules.".format(unloaded))

        log("")
        log("Done! Restart Rhino to see the COMPAS Acoustics tab in Grasshopper.")
        state["ok"] = True
    except Exception as e:
        log("")
        log("ERROR: {}".format(e))
        log(traceback.format_exc())
        state["ok"] = False
    finally:
        state["running"] = False


class Script_Instance(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, source, install):
        component = ghenv.Component  # noqa: F821
        state = sc.sticky.setdefault(STATE_KEY, {"running": False, "log": [], "ok": None})

        if install and not state["running"]:
            gh_doc = component.OnPingDocument()
            state.update(running=True, log=[], ok=None)
            thread = threading.Thread(target=run_install, args=(state, source, gh_doc.FilePath if gh_doc else None))
            thread.daemon = True
            thread.start()

        if state["running"]:
            component.Message = "Installing..."
            # keep polling the background thread to refresh the log
            doc = component.OnPingDocument()
            if doc:
                doc.ScheduleSolution(1000, Grasshopper.Kernel.GH_Document.GH_ScheduleDelegate(lambda d: component.ExpireSolution(False)))
        elif state["ok"] is True:
            component.Message = "Restart Rhino"
            component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Remark, "Installed. Restart Rhino to load the COMPAS Acoustics components.")
        elif state["ok"] is False:
            component.Message = "Failed"
            component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, "Installation failed, check the log output.")
        else:
            component.Message = "Ready"
            component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Remark,
                "Press the button to install. Leave 'source' empty to use the wheel next to this file.",
            )

        return "\n".join(state["log"]), state["ok"]
