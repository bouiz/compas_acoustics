# Install in Rhino 8 and Grasshopper

The Grasshopper components are a compiled plugin, `CompasAcoustics.gha`, for Rhino 8 on Windows and macOS.
Everything they compute runs in the *worker*, a Python 3.12 environment with `compas_acoustics` that the installer
creates.

## Install

1. Download `compas_acoustics_installer.zip` from the
   [latest release](https://github.com/gramaziokohler/compas_acoustics/releases/latest).
   It holds the wheel, the plugin (`gha/`) and `compas_acoustics_installer.gh`.
2. Unzip it and open `compas_acoustics_installer.gh` in Grasshopper.
3. Press **Install**. When the component says *Restart Rhino*, restart Rhino.

The components are then in the **COMPAS Acoustics** tab. The first installation downloads PyTorch for the worker,
several GB, and takes a few minutes.

## What the installer does

* Copies the plugin into Grasshopper's Libraries folder, in a `compas_acoustics` subfolder, and removes the CPython
  components (`.ghuser`) of earlier versions from UserObjects.
* Installs `compas_acoustics` into a Rhino script environment, for your own Python scripts in Rhino.
* Creates the worker with `uv` under `~/.compas_acoustics/workers/` and records it in
  `~/.compas_acoustics/worker.json`.

Installer options, the inputs of the installer component:

`source`
:   Path of the `.whl` file in the bundle. Empty: the `.whl` file next to the `.gh` file. The plugin must be
    next to it, in `gha/`.

`venv`
:   Name of the Rhino script environment that gets `compas_acoustics` too. Empty: `default`.

`worker`
:   Ignored; the worker is always set up.

On Windows, if Grasshopper reports the plugin as blocked, unblock the files in the Libraries folder
(*Properties > Unblock*).

## The worker

The worker starts with the first component that needs it (a few seconds; half a minute the first time) and stops
with Rhino. Every Rhino session starts its own, on a free port of `127.0.0.1`, and shares a random token with it:
it answers only requests that carry the token, and exits when Rhino ends or dies.

The worker must run the same `compas_acoustics` version and protobuf schema as the plugin; otherwise the components
say to run the installer again. To use another interpreter that has this version of `compas_acoustics`, set the
environment variable `COMPAS_ACOUSTICS_WORKER_PYTHON` to its path. A Python script can use the same worker through
`compas_acoustics.worker.client()`.

The **Worker** component (Util panel) shows the worker's state; every component's right-click menu has
*Open worker log*. See [Troubleshooting](../troubleshooting.md) when the worker does not start.

## Update

Download the installer of the new release and run it as above. It replaces the plugin and the worker's
`compas_acoustics`; restart Rhino afterwards.

## Uninstall

1. Close Rhino.
2. Delete the `compas_acoustics` folder in Grasshopper's Libraries folder (in Grasshopper:
   *File > Special Folders > Components Folder*).
3. Delete `~/.compas_acoustics/`, which holds the worker.
