********************************************************************************
Contributing to the documentation
********************************************************************************

The documentation is built with Sphinx. Most pages are written by hand; the Grasshopper component reference and the
example pages are generated from the plugin and from ``examples/``, and their images are made offline and committed.
The build checks that those images still match their sources and fails, with instructions, when they do not.


Build the documentation
=======================

.. code-block:: bash

    pip install -e ".[dev]"
    invoke docs

``invoke docs`` first runs ``invoke docs-components`` (which generates the pages and runs the checks), then builds the
HTML into ``dist/docs``. A plain ``sphinx-build`` and the CI build run the same generator, through the
``ghcomponents`` Sphinx extension.


What is generated, and from what
================================

The generator is ``docs/_ext/ghcomponents``. It writes three folders, which git ignores; edit their sources instead.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Generated
     - Sources
   * - ``docs/grasshopper/components/``
     - ``src/compas_acoustics_ghcsharp/CompasAcoustics.Grasshopper/Components/Specs/<CA_name>/metadata.json`` and
       ``icon.png``; the panel order from ``ComponentBase.Subcategories`` in ``ComponentBase.cs``; the GUIDs from
       the C# classes; ``docs/grasshopper/python_equivalents.json``; ``docs/grasshopper/notes/<CA_name>.rst``;
       ``docs/_images/components/<CA_name>.png``
   * - ``docs/python/examples/``
     - ``examples/<name>.py`` (its docstring gives the title and paragraph) and
       ``docs/_images/examples/<name>/`` (``figure_<n>.png``, ``viewer_<n>.png``, ``stdout.txt``)
   * - ``docs/grasshopper/examples/``
     - ``examples/<name>.ghx``, ``docs/grasshopper/walkthroughs/<name>.rst`` and
       ``docs/_images/examples/<name>/`` (``canvas.png``, ``viewport.png``)

``docs/_images/fingerprints.json`` records what each image was made from: the port names of each component, and a
hash of each example file. The checks compare it with the sources.


Make the images
===============

Grasshopper images
    Need Rhino 8 with the current plugin installed (``invoke installer-bundle``, then run the installer). Open
    Rhino 8 and Grasshopper, open ``docs/_tools/capture_grasshopper.py`` in Rhino's ScriptEditor and run it. It
    captures what is missing or out of date (component screenshots, and the canvas and viewport of each ``.ghx``)
    and updates the fingerprints. Set ``ALL = True`` at the top of the script to capture everything.

Python example outputs
    Run ``invoke docs-images``. It runs each example whose output is missing or out of date and saves the figures it
    shows, what it prints, and a capture of each COMPAS viewer it opens (the viewer window opens for a moment, so
    run it on a computer with a display). ``invoke docs-images --all`` runs every example.

Commit ``docs/_images/`` with the change that needed new images.


When the build fails
====================

Each failure names what is wrong and how to fix it. The sections below repeat the fix for each kind.

.. _screenshots:

Component screenshots are out of date
-------------------------------------

A component has no screenshot, or its ports changed since the screenshot was taken (the message names the ports
added or removed). Make the Grasshopper images as above and commit ``docs/_images/``.

.. _example-images:

Grasshopper example images are out of date
------------------------------------------

A ``.ghx`` example is new or changed since its canvas and viewport were captured. Make the Grasshopper images as
above, check that its walkthrough still describes it, and commit ``docs/_images/``.

.. _example-outputs:

Python example outputs are out of date
--------------------------------------

An example script is new or changed since its output was saved. Run ``invoke docs-images`` and commit
``docs/_images/``.

.. _subcategories:

A component is in an unknown panel
----------------------------------

A component's ``subcategory`` in its ``metadata.json`` is not in ``ComponentBase.Subcategories`` in
``src/compas_acoustics_ghcsharp/CompasAcoustics.Grasshopper/Components/ComponentBase.cs``. Add it there, where its
place in the list sets its place in the ribbon and in the docs.

.. _python-equivalents:

A component has no Python equivalent entry
------------------------------------------

Every component needs an entry in ``docs/grasshopper/python_equivalents.json``: the dotted path of the
``compas_acoustics`` class or function it corresponds to, or ``null`` for a component that exists only in
Grasshopper.

.. _example-docstrings:

A Python example has no docstring
---------------------------------

Start the script with a docstring: a one-line title, a blank line, then a paragraph on what it shows. The page takes
its title and introduction from it.

.. _walkthroughs:

A Grasshopper example has no walkthrough
----------------------------------------

Write ``docs/grasshopper/walkthroughs/<name>.rst``: the page title on the first line, a blank line, a paragraph on what
the definition does, a blank line, then one numbered step (``#.``) per group of the definition. Link components with
the ``:ref:`` role and the label ``gh-CA_<name>``.


Add an example
==============

A Python example
    Add ``examples/<name>.py`` with a docstring, run ``invoke docs-images``, and commit the script and
    ``docs/_images/``.

A Grasshopper example
    Save the definition as ``examples/<name>.ghx`` (XML, so that git can diff it), with each stage in a labelled
    group. Write its walkthrough, make the Grasshopper images, and commit the three.

The two lists grow independently. When a ``.py`` and a ``.ghx`` share a name, their pages link to each other.


Component notes and Python equivalents
======================================

To add text to a component's page beyond its description, write ``docs/grasshopper/notes/<CA_name>.rst``; it appears
under *Notes*. The component descriptions themselves live in its ``metadata.json``, which the plugin also shows.


Before a release
================

#. ``invoke docs-images --all``: a library change can alter a figure without changing the script.
#. Run ``capture_grasshopper.py`` with ``ALL = True``, for the same reason.
#. ``invoke docs --rebuild`` and read the changed pages.
#. ``invoke linkcheck``. Until the repository is public and has a release, ``releases/latest`` returns 404.
