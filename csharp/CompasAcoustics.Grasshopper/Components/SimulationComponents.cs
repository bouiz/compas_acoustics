using System;
using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Parameters;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    public sealed class SimulatePyroomacousticsComponent : ComponentBase
    {
        public SimulatePyroomacousticsComponent() : base("CA_Simulate_Ray_Tracing") { }
        protected override string SpecKey => "CA_Simulate_Ray_Tracing";
        public override Guid ComponentGuid => new Guid("afb77a6d-dd01-5224-9ee0-70972dcbdbcf");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) =>
            input && port.Name == "scene" ? new Param_Scene() : !input ? new Param_IrGraph() : null;

        protected override void Solve(IGH_DataAccess access)
        {
            var scene = Item<SceneGoo>(access, "scene");
            if (scene?.Value == null)
            {
                Warn("Input parameter 'scene' failed to collect data.");
                return;
            }
            var rayTracing = Value<bool>(access, "ray_tracing") ?? true;
            var settings = SimulationSettings.Pyroomacoustics(Value<int>(access, "ism_order"), rayTracing, Value<int>(access, "n_rays"),
                Value<bool>(access, "absorb_air"), Value<double>(access, "energy_thres"), Value<double>(access, "time_thres"),
                Value<int>(access, "seed"), Value<bool>(access, "normalize"), out var ignored, out _);
            if (ignored.Length > 0)
                Remark($"Ray tracing is off, so {string.Join(", ", ignored)} are not used.");
            var seed = Value<int>(access, "seed");
            // a seed drawn for this solution is not a change of the inputs
            var keySettings = new Dictionary<string, object?>(settings);
            if (seed == null)
                keySettings.Remove("seed");
            var request = new Background("simulate", new object?[] { scene.Value, "pyroomacoustics", settings, null }, new Dictionary<string, object?>(),
                (scene, settings), AsyncRuns.Key("simulate", new object?[] { scene.Value, "pyroomacoustics", keySettings, null }, new Dictionary<string, object?>()));
            RunInBackground(access, request, (outcome, _) =>
            {
                var (ran, used) = ((SceneGoo, Dictionary<string, object?>))outcome.Context!;
                var result = (IDictionary<string, object?>)outcome.Result!;
                // points outside the room, open rooms, mergeable faces, a slow image source search
                ShowWarnings(result);
                Status(SimulationSettings.PyroomacousticsMessage(ran.SamplingRate, used));
                var responses = (IDictionary<string, object?>)((IDictionary<string, object?>)result["responses"]!)["pyroomacoustics"]!;
                access.SetData(0, IrGraphGoo.FromSimulation(ran, "pyroomacoustics", responses));
            });
        }

        protected override bool RunsInBackground => true;
    }

    public sealed class DeconstructIrGraphComponent : ComponentBase
    {
        public DeconstructIrGraphComponent() : base("CA_Deconstruct_IR_Graph") { }
        protected override string SpecKey => "CA_Deconstruct_IR_Graph";
        public override Guid ComponentGuid => new Guid("c7f804dd-316a-5c87-8ee1-5b392d3ce202");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => (input, port.Name) switch
        {
            (true, "ir_graph") => new Param_IrGraph(),
            (false, "ir") => new Param_ImpulseResponse(),
            (false, "sources") => new Param_Source(),
            (false, "receivers") => new Param_Receiver(),
            (false, "edges") => new Param_Line(),
            _ => null,
        };

        /// <summary>Only the edges: the sources and receivers are drawn by their own components.</summary>
        public override bool PreviewsOutput(IGH_Param output) => output.Name == "edges";

        protected override void Solve(IGH_DataAccess access)
        {
            var graph = Item<IrGraphGoo>(access, "ir_graph");
            if (graph?.Value == null)
            {
                Warn("Input parameter 'ir_graph' failed to collect data.");
                return;
            }
            Message = graph.Value.Method;
            SceneIndices.Apply(this, "source_idx", graph.Value.Sources.Select(x => x.Name));
            SceneIndices.Apply(this, "receiver_idx", graph.Value.Receivers.Select(x => x.Name));
            var branches = IrGraphGrouping.Group(graph.Value, List<int>(access, "source_idx"), List<int>(access, "receiver_idx"), Warn);
            var irs = new DataTree<ImpulseResponseGoo>();
            var sources = new DataTree<SourceGoo>();
            var receivers = new DataTree<ReceiverGoo>();
            var edges = new DataTree<Line>();
            foreach (var branch in branches)
            {
                var path = new GH_Path(branch.ReceiverIndex);
                receivers.Add(branch.Receiver, path);
                sources.AddRange(branch.Sources, path);
                irs.AddRange(branch.Responses, path);
                edges.AddRange(branch.Sources.Select(s => new Line(s.Position, branch.Receiver.Position)), path);
            }
            access.SetDataTree(0, irs);
            access.SetDataTree(1, sources);
            access.SetDataTree(2, receivers);
            access.SetDataTree(3, edges);
        }
    }
}
