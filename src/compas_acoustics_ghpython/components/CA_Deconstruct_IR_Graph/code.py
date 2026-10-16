# flake8: noqa
import Grasshopper
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path
from Rhino.Geometry import Line
import System

class DeconstructIRGraph(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, ir_graph, receiver_idx, method):

        if ir_graph is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'ir_graph' failed to collect data."
            )
            return None, None, None, None

        # Each simulation method stores its responses under its own name; 'data' is the latest mono one
        methods = []
        for edge in ir_graph.edges():
            for name in ir_graph.edge_attributes(edge):
                if name not in methods:
                    methods.append(name)
        if not method:
            # 'data' by default; without it (only ambisonic results, e.g. hybrid), the first method on the graph
            method = "data" if "data" in methods or not methods else methods[0]
        if methods and method not in methods:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                "No responses of method '{}' on the graph; available: {}.".format(method, ", ".join(methods)),
            )
        ghenv.Component.Message = method

        ir_out = DataTree[object]()
        source_out = DataTree[object]()
        receiver_out = DataTree[object]()
        edge_out = DataTree[object]()

        # 1. Set order of receivers
        receiver_list = [
            ir_graph.node_attribute(node, 'data')
            for node in ir_graph.nodes()
            if ir_graph.node_attribute(node, 'type') == 'receiver'
        ]
        receiver_set = {rec.guid for rec in receiver_list}

        # 2. Make ir group as {receiver_0: [src_0, ...], ...}
        source_group = {}
        ir_group = {}

        for u, v in ir_graph.edges():
            ir = ir_graph.edge_attribute((u, v), method)
            if ir is None:
                continue
            data_u = ir_graph.node_attribute(u, 'data')
            data_v = ir_graph.node_attribute(v, 'data')

            if data_u.guid in receiver_set:
                rec, src = data_u, data_v
            elif data_v.guid in receiver_set:
                rec, src = data_v, data_u
            else:
                continue
            
            source_group.setdefault(rec.guid, []).append(src)
            ir_group.setdefault(rec.guid, []).append(ir)

        # 3. Filter receiver, optional
        if receiver_idx is None:
            indices = list(range(len(receiver_list)))
        else:
            indices = receiver_idx  # Already set to list access

        for idx in indices:
            rec = receiver_list[idx]
            path = GH_Path(idx)

            sources = source_group.get(rec.guid, [])
            irs = ir_group.get(rec.guid, [])

            receiver_out.Add(rec, path)
            source_out.AddRange(sources, path)
            ir_out.AddRange(irs, path)
            edge_out.AddRange([Line(*src.position, *rec.position) for src in sources], path)

        return ir_out, source_out, receiver_out, edge_out