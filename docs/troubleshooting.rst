********************************************************************************
Troubleshooting
********************************************************************************

Each entry gives the symptom, its usual cause and the fix. When none fits, open the worker log (right-click any
component, **Open worker log**) and `report an issue <https://github.com/gramaziokohler/compas_acoustics/issues>`_
with the log, your operating system and the Rhino version.


The worker does not start
=========================

**Symptom:** components stay red with *The worker at ... could not be started*, *exited while starting* or *did not
answer within ... s*.

**Cause:** the worker's Python environment is missing or broken, often after an interrupted first installation
(it downloads several GB), or something blocks local connections on ``127.0.0.1``.

**Fix:** run the installer again and wait until it says *Restart Rhino*; restart Rhino. Check that a firewall or
security tool allows Rhino's local connections. The log names the Python it tried to start.


"Run the installer again"
=========================

**Symptom:** *is not this version's*, *runs compas_acoustics <version>* or *schema mismatch: run the installer again*.

**Cause:** the plugin and the worker come from different releases, for example after updating only one of them, or
``COMPAS_ACOUSTICS_WORKER_PYTHON`` points at an environment with another version.

**Fix:** run the installer of the release you want, then restart Rhino. If you set
``COMPAS_ACOUSTICS_WORKER_PYTHON``, install the same ``compas_acoustics`` version there or unset it.


Binaural or Ambisonics is missing in Python
===========================================

**Symptom:** ``ca.Binaural`` or ``ca.Ambisonics`` raises ``ImportError: ... needs Python >= 3.11``.

**Cause:** they need Python 3.11 or newer (pyFDN, spharpy and multislope); on 3.9 and 3.10, Rhino 8's Python among
them, ``compas_acoustics`` leaves them out.

**Fix:** use an environment with Python 3.11 or newer. In Grasshopper the Simulate Binaural and Simulate Ambisonics
components work anyway: they run in the worker, which has Python 3.12.


macOS blocks the plugin
=======================

**Symptom:** Grasshopper does not load ``CompasAcoustics.gha``, or macOS says the file cannot be checked.

**Cause:** files downloaded from the internet carry the quarantine attribute.

**Fix:** remove it from the plugin folder (Grasshopper: *File > Special Folders > Components Folder*), then restart
Rhino:

.. code-block:: bash

    xattr -dr com.apple.quarantine "<Components Folder>/compas_acoustics"


Windows blocks the plugin
=========================

**Symptom:** Grasshopper reports the plugin as blocked.

**Fix:** in the Components Folder, right-click each file of ``compas_acoustics``, choose *Properties* and tick
*Unblock*; restart Rhino.


Old components appear next to the new ones
==========================================

**Symptom:** an older version of a component (for example an Audio Player) appears, or Grasshopper reports a
component ID conflict.

**Cause:** a copy installed another way: an old Yak package from the Package Manager, CPython components
(``.ghuser``) of an earlier version in UserObjects, or a developer build folder listed in Grasshopper's settings.

**Fix:** uninstall ``compas_acoustics`` in Rhino's Package Manager if it is listed there, delete
``compas_acoustics`` files from *File > Special Folders > User Object Folder*, remove extra folders listed by Rhino's
``GrasshopperDeveloperSettings`` command, and restart Rhino. The installer
removes the old ``.ghuser`` components itself.


Where the logs are
==================

* The worker's log: right-click any component, **Open worker log**. It is a temporary file named
  ``compas_acoustics_worker_<process>.log``, one per Rhino session.
* The installer's log: the ``log`` output of the installer component.
* The worker's environment: ``~/.compas_acoustics/worker.json`` names its Python.
