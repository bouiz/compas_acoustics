import os
import random
import warnings

import Grasshopper

from compas_acoustics import worker
from compas_acoustics_ghpython import MethodGraph
from compas_acoustics_ghpython import ensure_inputs

# First item is the default
TAILS = ["auto", "reference", "ray_tracing", "eyring", "sabine"]
DIFFRACTION = ["none", "udfa", "btms"]
OUTPUTS = ["binaural", "ambisonic", "both"]
# The formats of each 'output', and the graph attribute each is stored under
FORMATS = {"binaural": ["binaural"], "ambisonic": ["ambisonic"], "both": ["ambisonic", "binaural"]}
ATTRIBUTES = {"binaural": "hybrid_binaural", "ambisonic": "hybrid"}


class SimulateHybrid(Grasshopper.Kernel.GH_ScriptInstance):

    def error(self, text):
        ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, text)

    @staticmethod
    def status(graph, formats, tail, sh_order, ism_order, diffraction):
        """What the component says while it is not running: the last result on the graph, or that there is none."""
        by_format = {
            name: [ir for edge in graph.edges() for ir in [graph.edge_attribute(edge, ATTRIBUTES[name])] if ir is not None]
            for name in formats
        }
        runs = [ir for irs in by_format.values() for ir in irs]
        if not runs:
            return "press Run"
        last = runs[0].generation or {}
        models = {None: False, False: False, "none": False, "": False, True: "udfa", "udfa": "udfa", "btms": "btms"}
        changed = (
            not all(by_format.values())
            or (tail not in (None, "", "auto") and tail != last.get("tail"))
            or (sh_order is not None and sh_order != last.get("sh_order"))
            or (ism_order is not None and ism_order != last.get("image_source_order"))
            or (diffraction is not None and models.get(diffraction) != models.get(last.get("diffraction")))
        )
        summary = "tail: {}, order {}".format(last.get("tail"), last.get("sh_order"))
        if changed:
            return "settings changed, press Run ({})".format(summary)
        return "done, {} IR{}, {} (Run to redo)".format(len(runs), "" if len(runs) == 1 else "s", summary)

    @staticmethod
    def graphs(scene, formats):
        """The outputs ``binaural_ir_graph`` and ``ambisonic_ir_graph``: the scene's graph tagged with the format's attribute, None if it is not requested."""
        return tuple(MethodGraph(scene.ir_graph, ATTRIBUTES[name]) if name in formats else None for name in ("binaural", "ambisonic"))

    @staticmethod
    def hrtf_label(given, formats):
        """The HRTF set the binaural output is rendered with, for the message; empty without a binaural output."""
        if "binaural" not in formats:
            return ""
        return ", hrtf {}".format(os.path.basename(given["hrtf"]) if "hrtf" in given else "kemar")

    def run_in_worker(self, scene, given, references, formats):
        settings = {key: value for key, value in given.items() if key != "reference"}
        try:
            _, messages = worker.simulate(scene, "hybrid", settings, references)
        except Exception as e:
            self.error("The worker failed: {}".format(e))
            return
        for message in messages:
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, message)
        ghenv.Component.Message = "{:g}Hz, hybrid (worker), tail: {}, seed {}{}".format(
            scene.sampling_rate, settings.get("tail", "auto"), settings["seed"], self.hrtf_label(settings, formats)
        )
        return self.graphs(scene, formats)

    def RunScript(self, scene, tail, reference, sh_order, ism_order, diffraction, late_envelope, seed, normalize, output, hrtf, run):
        # No value list on hrtf: empty is KEMAR, and a File Path connected there is the user's own SOFA set
        ensure_inputs(ghenv.Component, {"run": None, "tail": TAILS, "diffraction": DIFFRACTION, "output": OUTPUTS})

        if scene is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'scene' failed to collect data."
            )
            return

        output = None if output == "" else output
        if output is not None and output not in FORMATS:
            self.error("output must be one of {}, not {!r}.".format(", ".join(OUTPUTS), output))
            return
        formats = FORMATS[OUTPUTS[0] if output is None else output]

        # Checked before 'run' so that it shows as soon as the tail is chosen, not only after pressing Run
        if tail == "reference" and reference in (None, ""):
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                "tail 'reference' needs a reference: connect the mono_ir_graph of Simulate Pyroomacoustics to 'reference', or choose another tail.",
            )
            ghenv.Component.Message = "connect a reference"
            return self.graphs(scene, formats)

        # The simulation can take minutes and Grasshopper recomputes on every upstream change, so it runs only
        # when 'run' is true (a button pulse). Otherwise the responses already on the scene's graph are passed on.
        if not run:
            ghenv.Component.Message = self.status(scene.ir_graph, formats, tail, sh_order, ism_order, diffraction)
            return self.graphs(scene, formats)

        try:
            from compas_acoustics.simulation_methods import Hybrid
        except ImportError as e:
            if worker.worker_python() is None:
                self.error(
                    "The hybrid method needs Python >= 3.11 (Rhino 8 runs 3.9), where compas_acoustics installs pyFDN, "
                    "or a worker: run the installer with worker 'hybrid'. {}".format(e)
                )
                return
            Hybrid = None

        tail = None if tail in (None, "", "auto") else tail
        diffraction = None if diffraction in (None, "", "none") else diffraction
        references = None
        if reference is not None and hasattr(reference, "edges"):
            # The mono_ir_graph output of Simulate Pyroomacoustics: the wire also orders and refreshes this component
            graph = reference
            reference = {}
            for name in ("pyroomacoustics", "data"):
                for edge in graph.edges():
                    ir = graph.edge_attribute(edge, name)
                    if ir is not None and ir.n_channels == 1:
                        reference.setdefault("{}_{}".format(ir.receiver.guid, ir.source.guid), ir)
                if reference:
                    break
            if not reference:
                self.error("The connected ir_graph holds no mono responses; connect the mono_ir_graph output of Simulate Pyroomacoustics.")
                return
        if reference:
            references = reference if isinstance(reference, dict) else scene.impulse_responses(reference)
            if not references:
                self.error("No responses of method '{}' on the scene; run it first, or leave 'reference' empty.".format(reference))
                return
            reference_rate = next(iter(references.values())).sampling_rate
            if int(reference_rate) != scene.sampling_rate:
                self.error("The reference was simulated at {} Hz, the scene's sampling_rate is {} Hz: simulate both at the same rate.".format(reference_rate, scene.sampling_rate))
                return

        # Unset inputs are left out so that the method's own defaults apply
        given = {
            "reference": reference or None,
            "tail": tail,
            "sh_order": sh_order,
            "image_source_order": ism_order,
            "diffraction": diffraction,
            "late_envelope": late_envelope,
            # The method's own default is a fixed seed; an empty input draws a new one each run
            "seed": random.randrange(2**31) if seed is None else seed,
            "normalize": normalize,
            # Hybrid takes an hrtf only with a binaural output
            "output": None if output is None else (formats[0] if len(formats) == 1 else formats),
            "hrtf": hrtf if "binaural" in formats and hrtf not in ("", None) else None,
        }
        given = {key: value for key, value in given.items() if value is not None}
        if Hybrid is None:
            return self.run_in_worker(scene, given, references, formats)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                method = Hybrid(**given)
                scene.simulate(method)
            except ValueError as e:
                self.error(str(e))
                return
        for warning in caught:
            if not issubclass(warning.category, UserWarning):
                continue
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, str(warning.message))

        ghenv.Component.Message = "{:g}Hz, hybrid, tail: {}, order {}, seed {}{}".format(
            scene.sampling_rate, method.tail, method.sh_order, given["seed"], self.hrtf_label(given, formats)
        )
        return self.graphs(scene, formats)
