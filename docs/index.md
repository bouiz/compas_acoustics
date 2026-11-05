# compas_acoustics

compas_acoustics simulates, analyses and auralises the acoustics of rooms modelled with COMPAS or in Rhino and Grasshopper. It computes impulse responses with the image source method and ray tracing (Mono), and with a hybrid engine of image sources, edge diffraction and a directional feedback delay network (Binaural, Ambisonics); it derives room acoustic parameters (ISO 3382) from them and convolves them with anechoic recordings to listen to a design.

<figure markdown>
![A shoebox room in Rhino with two sources and two receivers](_images/home.png)
<figcaption markdown>The room of [A simple simulation in a shoebox room](grasshopper/examples/simple_simulation_shoebox.md), with its sources and receivers, in Rhino.</figcaption>
</figure>

## Use it in Grasshopper or in Python

<div class="grid cards" markdown>

-   **[Use in Grasshopper](grasshopper/getting_started.md)**

    ---

    Components for Rhino 8 on Windows and macOS: build a room from Rhino geometry, simulate it, read its
    acoustic parameters and listen to it on the canvas. Start with the
    [installation](installation/grasshopper.md).

-   **[Use in Python](python/tutorial.md)**

    ---

    A COMPAS package: script scenes, simulation methods, analyses and auralisations, and run them in batches.
    Start with [Install in Python](installation/python.md).

</div>

New to room acoustics? [Concepts](concepts.md) explains the terms both tracks use.

## Features

* **Simulation methods**: `Mono` (image sources and ray tracing, with pyroomacoustics), `Binaural` and
  `Ambisonics` (a hybrid of image sources, edge diffraction and a directional feedback delay network), all through
  `scene.simulate`.
* **Analysis**: reverberation times, clarity and definition, direct-to-reverberant ratio, lateral fraction, strength
  and STI from an impulse response (`Analyser`).
* **Auralisation**: convolve mono, binaural or ambisonic responses with anechoic recordings (`Auraliser`), and play
  them (`AudioPlayer`).
* **Materials library**: 95 absorbing and scattering materials in categories, searchable from Python and Grasshopper.
* **Grasshopper plugin**: components for Rhino 8 on Windows and macOS that run every computation in a Python worker.

## How to cite

If you use `compas_acoustics` in your work, please cite it:

```bibtex
@software{compas_acoustics,
  title  = {compas\_acoustics: room acoustic simulation, analysis and auralisation for COMPAS},
  author = {
    %% authors: generated from CITATION.cff
  },
  year   = {2025-%% build-year},
  organization = {Gramazio Kohler Research, ETH Zurich},
  url    = {%% repository-code},
}
```
