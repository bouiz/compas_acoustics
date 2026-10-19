import warnings

import Grasshopper

from compas_acoustics import worker
from compas_acoustics_ghpython import MethodGraph
from compas_acoustics_ghpython import ensure_inputs

# First item is the default
TAILS = ["auto", "reference", "ray_tracing", "eyring", "sabine"]
DIFFRACTION = ["none", "udfa", "btms"]


class SimulateHybrid(Grasshopper.Kernel.GH_ScriptInstance):

    def error(self, text):
        ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, text)

    @staticmethod
    def status(graph, tail, sh_order, image_source_order, diffraction):
        """What the component says while it is not running: the last result on the graph, or that there is none."""
        runs = [ir for edge in graph.edges() for ir in [graph.edge_attribute(edge, "hybrid")] if ir is not None]
        if not runs:
            return "press Run"
        last = runs[0].generation or {}
        models = {None: False, False: False, "none": False, "": False, True: "udfa", "udfa": "udfa", "btms": "btms"}
        changed = (
            (tail not in (None, "", "auto") and tail != last.get("tail"))
            or (sh_order is not None and sh_order != last.get("sh_order"))
            or (image_source_order is not None and image_source_order != last.get("image_source_order"))
            or (diffraction is not None and models.get(diffraction) != models.get(last.get("diffraction")))
        )
        summary = "tail: {}, order {}".format(last.get("tail"), last.get("sh_order"))
        if changed:
            return "settings changed, press Run ({})".format(summary)
        return "done, {} response{}, {} (Run to redo)".format(len(runs), "" if len(runs) == 1 else "s", summary)

    def run_in_worker(self, scene, given, references):
        settings = {key: value for key, value in given.items() if key != "reference"}
        try:
            _, messages = worker.simulate(scene, "hybrid", settings, references)
        except Exception as e:
            self.error("The worker failed: {}".format(e))
            return
        for message in messages:
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, message)
        ghenv.Component.Message = "{:g}Hz, hybrid (worker), tail: {}".format(scene.sampling_rate, settings.get("tail", "auto"))
        return MethodGraph(scene.ir_graph, "hybrid")

    def RunScript(self, scene, tail, reference, sh_order, image_source_order, diffraction, late_envelope, seed, normalize, run):
        ensure_inputs(ghenv.Component, {"run": None, "tail": TAILS, "diffraction": DIFFRACTION})

        if scene is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'scene' failed to collect data."
            )
            return

        # The simulation can take minutes and Grasshopper recomputes on every upstream change, so it runs only
        # when 'run' is true (a button pulse). Otherwise the responses already on the scene's graph are passed on.
        if not run:
            ghenv.Component.Message = self.status(scene.ir_graph, tail, sh_order, image_source_order, diffraction)
            return MethodGraph(scene.ir_graph, "hybrid")

        try:
            from compas_acoustics.simulation_methods import Hybrid
        except ImportError as e:
            if worker.worker_python() is None:
                self.error(
                    "The hybrid method needs Python >= 3.11 (Rhino 8 runs 3.9) and the 'hybrid' extra (pip install compas_acoustics[hybrid]), "
                    "or a worker: run the installer with worker 'hybrid'. {}".format(e)
                )
                return
            Hybrid = None

        tail = None if tail in (None, "", "auto") else tail
        diffraction = None if diffraction in (None, "", "none") else diffraction
        references = None
        if reference is not None and hasattr(reference, "edges"):
            # The ir_graph output of Simulate Pyroomacoustics: the wire also orders and refreshes this component
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
                self.error("The connected ir_graph holds no mono responses; connect the output of Simulate Pyroomacoustics.")
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
            "image_source_order": image_source_order,
            "diffraction": diffraction,
            "late_envelope": late_envelope,
            "seed": seed,
            "normalize": normalize,
        }
        given = {key: value for key, value in given.items() if value is not None}
        if Hybrid is None:
            return self.run_in_worker(scene, given, references)
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

        ghenv.Component.Message = "{:g}Hz, hybrid, tail: {}, order {}".format(
            scene.sampling_rate, method.tail, method.sh_order
        )
        return MethodGraph(scene.ir_graph, "hybrid")
