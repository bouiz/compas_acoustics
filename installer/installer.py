#! python3
"""COMPAS Acoustics installer for Rhino 8 / Grasshopper.

This is the source of the script component inside ``compas_acoustics_installer.gh``.
It must stay self-contained: it runs before ``compas_acoustics`` is installed.

What it does, in a background thread so Rhino stays responsive:

1. Resolve the wheel to install: ``source`` can be a local ``.whl`` path,
an http(s) URL to a ``.whl``, or empty (picks the newest ``compas_acoustics-*.whl``
next to the ``.gh`` file).
2. Install it with Rhino's own CPython into a Rhino script environment, the
same way Rhino installs the packages that scripts ask for. ``venv`` is the
name of the environment (created if needed). Empty means ``default``, the
environment of scripts that do not ask for one.
3. Copy the Grasshopper plugin (``CompasAcoustics.gha`` and the assemblies it
needs, the ``gha`` folder of the installer bundle, next to the wheel) into
Grasshopper's Libraries folder (``compas_acoustics`` subfolder), and remove
the CPython components (``.ghuser``) of earlier versions from UserObjects.
4. Create the worker: a Python 3.12 environment with compas_acoustics, in
which the plugin's components run everything (``compas_acoustics.worker.install``).
5. Unload previously imported ``compas_acoustics`` modules.

Rhino must be restarted afterwards: reloading Grasshopper's component libraries
at runtime (``ComponentServer.LoadExternalFiles``) re-registers every plugin
and makes Grasshopper show "Component ID conflict" dialogs.

Inputs: source (str), venv (str), install (bool, connect a button), worker (str, not used: the worker is always
created; kept so that the installer file's inputs stay).
Outputs: log (str), ok (bool).

Rhino quirks to keep in mind when editing this file: Rhino pre-processes the
script text, so do not indent lines inside this docstring, and do not write
script directives (hash + venv/r/requirements) anywhere, not even in comments.
"""

import glob
import os
import re
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
DEFAULT_VENV = "default"
# the Grasshopper plugin's files, in the bundle's gha folder
GHA_FOLDER = "gha"
GHA_FILES = ["CompasAcoustics.gha", "CompasPb.dll", "Google.Protobuf.dll", "Newtonsoft.Json.dll"]


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


def run(cmd, log, env=None):
    log("> " + " ".join('"{}"'.format(c) if " " in c else c for c in cmd))
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    if env is not None:
        kwargs["env"] = env
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, **kwargs)
    for line in process.stdout:
        line = line.rstrip()
        if line:
            log("  " + line)
    if process.wait() != 0:
        raise RuntimeError("Command failed with exit code {}".format(process.returncode))


def query(python, code, target=None):
    kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
    env = dict(os.environ)
    if target:
        env["PYTHONPATH"] = target
    return subprocess.check_output([python, "-c", code], universal_newlines=True, env=env, **kwargs).strip()


def on_ui_thread(func, timeout=60):
    """Run ``func`` on Rhino's UI thread and wait for its result: RhinoCode and Grasshopper are not thread-safe."""
    import Rhino
    import System

    result = {}
    done = threading.Event()

    def wrapper():
        try:
            result["value"] = func()
        except Exception as e:
            result["error"] = e
        finally:
            done.set()

    Rhino.RhinoApp.InvokeOnUiThread(System.Action(wrapper))
    if not done.wait(timeout):
        raise RuntimeError("Timed out waiting for Rhino")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def resolve_venv(venv, log):
    """Folder of the Rhino script environment named ``venv``, created if it does not exist yet."""
    if not re.match(r"^[A-Za-z0-9_.-]+$", venv):
        raise RuntimeError("Invalid venv name '{}': use letters, digits, '.', '_' or '-'.".format(venv))

    def find_or_create():
        from Rhino.Runtime.Code import RhinoCode
        from Rhino.Runtime.Code.Languages import LanguageSpec

        environs = RhinoCode.Languages.QueryLatest(LanguageSpec.Python3).Environs
        environ = environs.OfIdentity(venv)
        created = environ is None
        if created:
            environ = environs.CreateEnviron(venv)
        return str(environ.Location), created

    location, created = on_ui_thread(find_or_create)
    log("{} Rhino script environment '{}': {}".format("Created" if created else "Using", venv, location))
    return location


def install_gha(wheel, log):
    """Copy the plugin of the bundle (its ``gha`` folder, next to ``wheel``) into Grasshopper's Libraries folder.

    The CPython components of earlier versions (``.ghuser`` files in UserObjects) are removed: the plugin replaces them.
    """
    source = os.path.join(os.path.dirname(os.path.abspath(wheel)), GHA_FOLDER)
    missing = [name for name in GHA_FILES if not os.path.isfile(os.path.join(source, name))]
    if missing:
        raise RuntimeError(
            "The Grasshopper plugin is not next to the wheel ({} missing in {}): download and unzip the whole installer bundle, "
            "and leave 'source' empty or point it at the wheel in the bundle.".format(", ".join(missing), source)
        )

    target = os.path.join(Grasshopper.Folders.DefaultAssemblyFolder, PACKAGE)
    if os.path.isdir(target):
        shutil.rmtree(target)
    os.makedirs(target)
    for name in GHA_FILES:
        shutil.copy2(os.path.join(source, name), os.path.join(target, name))
    log("Installed the Grasshopper plugin {} into {}".format(GHA_FILES[0], target))
    if os.name == "nt":
        log("If Grasshopper reports the plugin as blocked, unblock the files in {} (Properties > Unblock).".format(target))

    userobjects = Grasshopper.Folders.DefaultUserObjectFolder
    old = os.path.join(userobjects, PACKAGE)
    removed = 0
    if os.path.isdir(old):
        removed += len(glob.glob(os.path.join(old, "*.ghuser")))
        shutil.rmtree(old)
    # copies of manual installs into the UserObjects root
    for path in glob.glob(os.path.join(userobjects, PACKAGE + ": *.ghuser")):
        os.remove(path)
        removed += 1
    if removed:
        log("Removed {} CPython components of an earlier version from {}".format(removed, userobjects))


def unload_modules():
    names = [n for n in sys.modules if n == PACKAGE or n.startswith(PACKAGE + ".") or n.startswith(PACKAGE + "_ghpython")]
    for name in names:
        del sys.modules[name]
    return len(names)


def run_install(state, source, venv, gh_file, worker):
    def log(message):
        state["log"].append(message)

    try:
        if sys.version_info[0] < 3:
            raise RuntimeError("This installer needs a Python 3 script component (Rhino 8 or newer).")

        python = rhino_python()
        log("Rhino Python {} at {}".format(sys.version.split()[0], python))

        wheel = resolve_source(source, gh_file, log)

        venv = (venv or "").strip() or DEFAULT_VENV
        target = resolve_venv(venv, log)
        # same as Rhino does for a script environment: install all packages into its folder
        pip = [python, "-I", "-m", "pip", "--isolated", "install", "--disable-pip-version-check", "--no-warn-script-location", "--target", target]

        log("")
        log("Installing {} and its dependencies, this can take a few minutes...".format(os.path.basename(wheel)))
        run(pip + ["--upgrade", wheel], log)
        # the version number does not change between pre-release builds, so force a reinstall of the package itself
        run(pip + ["--upgrade", "--force-reinstall", "--no-deps", wheel], log)

        version = query(python, "import {0}; print({0}.__version__)".format(PACKAGE), target)
        log("")
        log("{} {} installed.".format(PACKAGE, version))

        install_gha(wheel, log)

        if (worker or "").strip():
            log("The 'worker' input is not used any more: the worker is always set up.")
        log("")
        log("Setting up the Python 3.12 worker, in which the components run; the first time this takes a while (it downloads PyTorch, several GB)...")
        cmd = [python, "-m", "compas_acoustics.worker.install", "--wheel", wheel, "--hybrid"]
        run(cmd, log, env=dict(os.environ, PYTHONPATH=target))

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
    def RunScript(self, source, venv, install, worker):
        component = ghenv.Component  # noqa: F821
        state = sc.sticky.setdefault(STATE_KEY, {"running": False, "log": [], "ok": None})

        if install and not state["running"]:
            gh_doc = component.OnPingDocument()
            state.update(running=True, log=[], ok=None)
            thread = threading.Thread(target=run_install, args=(state, source, venv, gh_doc.FilePath if gh_doc else None, worker))
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
                "Press the button to install. Leave 'source' empty to use the wheel next to this file, "
                "and 'venv' empty to install into the default script environment.",
            )

        return "\n".join(state["log"]), state["ok"]
