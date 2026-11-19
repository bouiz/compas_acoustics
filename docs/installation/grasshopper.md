# Install in Rhino 8 and Grasshopper

The Grasshopper components are a compiled plugin, `CompasAcoustics.gha`, for Rhino 8 on Windows and macOS.
Everything they compute runs in the *worker*, a Python 3.12 environment with `compas_acoustics`.

## Install

1. In Rhino, run `PackageManager`, search for `compas_acoustics`, install it and restart Rhino.
2. In Grasshopper, place the **Install Worker** component from the Util panel of the **COMPAS Acoustics** tab.
3. Press **Install** and wait until the component says *installed*. Its `log` output shows the progress.

The first installation downloads the worker's packages, up to several GB, and takes a few minutes. Rhino stays
responsive, and no restart is needed afterwards.

## What Install Worker does

* Fetches `uv` with Rhino's Python, into `~/.compas_acoustics/uv`. Nothing is installed into Rhino's Python.
* Stops running workers.
* Creates the worker with `uv` in `~/.compas_acoustics/worker`, with the newest `compas_acoustics` on PyPI that the
  plugin works with (up to the next breaking release, e.g. `>=0.1.0,<0.2.0`), and records it in
  `~/.compas_acoustics/worker.json`.

If it says that Rhino's Python is not set up, open Rhino's ScriptEditor once, then press **Install** again.

## The worker

The worker starts with the first component that needs it (a few seconds; half a minute the first time) and stops
with Rhino. Every Rhino session starts its own, on a free port of `127.0.0.1`, and shares a random token with it:
it answers only requests that carry the token, and exits when Rhino ends or dies.

The worker must run a `compas_acoustics` version the plugin works with and the plugin's protobuf schema; otherwise the
components say to use Install Worker again. To use another interpreter that has such a `compas_acoustics`, set the
environment variable `COMPAS_ACOUSTICS_WORKER_PYTHON` to its path. A Python script can use the same worker through
`compas_acoustics.worker.client()`.

The **Worker** component (Util panel) shows the worker's state; every component's right-click menu has
*Open worker log*. See [Troubleshooting](../troubleshooting.md) when the worker does not start.

## Update

Update `compas_acoustics` in Rhino's Package Manager and restart Rhino, then press **Install** on the Install Worker
component: it replaces the worker with the newest `compas_acoustics` the plugin works with. Press **Install** again
to get a bugfix release of `compas_acoustics` without a new plugin.

## Uninstall

1. Uninstall `compas_acoustics` in Rhino's Package Manager and close Rhino.
2. Delete `~/.compas_acoustics/`, which holds the worker.
