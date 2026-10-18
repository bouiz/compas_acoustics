# flake8: noqa
import warnings

import Grasshopper
import System
import scriptcontext as sc

from compas_acoustics import worker

# First item is the default
TAILS = ["auto", "reference", "ray_tracing", "eyring", "sabine"]
DIFFRACTION = ["none", "udfa", "btms"]


class SimulateHybrid(Grasshopper.Kernel.GH_ScriptInstance):

    def offer_value_lists(self):
        component = ghenv.Component
        offered_key = "hybrid_value_lists_offered_{}".format(component.InstanceGuid)
        if sc.sticky.get(offered_key):
            return
        sc.sticky[offered_key] = True

        doc = component.OnPingDocument()
        if doc is None:
            return

        # A value list per choice input, and a button for 'run'
        value_lists = {"run": None, "tail": TAILS, "diffraction": DIFFRACTION}
        targets = [
            (index, param.Name, value_lists[param.Name])
            for index, param in enumerate(component.Params.Input)
            if param.Name in value_lists and param.SourceCount == 0
        ]
        if not targets:
            return

        def deferred(gh_doc):
            base = component.Attributes.Pivot
            for index, name, items in targets:
                input_param = component.Params.Input[index]
                if input_param.SourceCount > 0:
                    continue
                if items is None:
                    button = Grasshopper.Kernel.Special.GH_ButtonObject()
                    button.CreateAttributes()
                    button.NickName = "Run"
                    button.Attributes.Pivot = System.Drawing.PointF(base.X - 220, input_param.Attributes.Pivot.Y - 10)
                    gh_doc.AddObject(button, False)
                    input_param.AddSource(button)
                    button.ExpireSolution(True)
                    continue
                value_list = Grasshopper.Kernel.Special.GH_ValueList()
                value_list.CreateAttributes()
                value_list.NickName = name
                value_list.ListItems.Clear()
                for item in items:
                    value_list.ListItems.Add(Grasshopper.Kernel.Special.GH_ValueListItem(item, '"{}"'.format(item)))
                value_list.Attributes.Pivot = System.Drawing.PointF(base.X - 220, input_param.Attributes.Pivot.Y - 10)
                gh_doc.AddObject(value_list, False)
                input_param.AddSource(value_list)
                value_list.ExpireSolution(True)

        doc.ScheduleSolution(1, deferred)

    def error(self, text):
        ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, text)

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
        return scene.ir_graph

    def RunScript(self, scene, run, tail, reference, sh_order, image_source_order, diffraction, late_envelope, seed, normalize):
        self.offer_value_lists()

        if scene is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'scene' failed to collect data."
            )
            return

        # The simulation can take minutes and Grasshopper recomputes on every upstream change, so it runs only
        # when 'run' is true (a button pulse). Otherwise the responses already on the scene's graph are passed on.
        if not run:
            ghenv.Component.Message = "press Run"
            return scene.ir_graph

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
                rirs = scene.simulate(method)
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
        return scene.ir_graph
