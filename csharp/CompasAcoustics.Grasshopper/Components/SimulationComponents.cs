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
    public sealed class SimulateMonoComponent : ComponentBase
    {
        public SimulateMonoComponent() : base("CA_Simulate_Mono") { }
        protected override string SpecKey => "CA_Simulate_Mono";
        public override Guid ComponentGuid => new Guid("afb77a6d-dd01-5224-9ee0-70972dcbdbcf");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) =>
            input && port.Name == "scene" ? new Param_Scene() : !input ? new Param_IrGraph() : null;

        /// <summary>The image source order of the simulations: whole orders from 0 to 10, set when the knob is let go (a run follows).</summary>
        internal static SliderControl IsmOrder => new("ism_order", 0, 10, "", Step: 1, Title: "ISM order");

        public override IReadOnlyList<Control> Controls { get; } = new Control[]
        {
            IsmOrder,
            new ToggleControl("ray_tracing", "Ray tracing", Inline: true), new ToggleControl("absorb_air", "Air", Inline: true),
            new ToggleControl("normalize", "Normalize", Inline: true),
        };

        protected override void Solve(IGH_DataAccess access)
        {
            var scene = Item<SceneGoo>(access, "scene");
            if (scene?.Value == null)
            {
                Warn("Input parameter 'scene' failed to collect data.");
                return;
            }
            var rayTracing = Value<bool>(access, "ray_tracing") ?? true;
            var settings = SimulationSettings.Mono(Value<int>(access, "ism_order"), rayTracing, Value<int>(access, "n_rays"),
                Value<bool>(access, "absorb_air"), Value<double>(access, "energy_thres"), Value<double>(access, "duration"),
                Value<int>(access, "seed"), Value<bool>(access, "normalize"), out var ignored, out _);
            if (ignored.Length > 0)
                Remark($"Ray tracing is off, so {string.Join(", ", ignored)} are not used.");
            var seed = Value<int>(access, "seed");
            // a seed drawn for this solution is not a change of the inputs
            var keySettings = new Dictionary<string, object?>(settings);
            if (seed == null)
                keySettings.Remove("seed");
            var request = new Background("simulate", new object?[] { scene.Value, "mono", settings, null }, new Dictionary<string, object?>(),
                (scene, settings), AsyncRuns.Key("simulate", new object?[] { scene.Value, "mono", keySettings, null }, new Dictionary<string, object?>()));
            RunInBackground(access, request, (outcome, _) =>
            {
                var (ran, used) = ((SceneGoo, Dictionary<string, object?>))outcome.Context!;
                var result = (IDictionary<string, object?>)outcome.Result!;
                // points outside the room, open rooms, mergeable faces, a slow image source search
                ShowWarnings(result);
                Status(SimulationSettings.MonoMessage(ran.SamplingRate, used));
                var responses = (IDictionary<string, object?>)((IDictionary<string, object?>)result["responses"]!)["mono"]!;
                access.SetData(0, IrGraphGoo.FromSimulation(ran, "mono", responses));
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

        public override IReadOnlyList<Control> Controls { get; } = new Control[]
        {
            new DropdownControl("source_idx", new[] { "" }, "", new[] { "All" }, "Source"),
            new DropdownControl("receiver_idx", new[] { "" }, "", new[] { "All" }, "Receiver"),
        };

        /// <summary>Nothing: the sources and receivers are drawn by their own components, the edges are an output to wire on.</summary>
        public override bool PreviewsOutput(IGH_Param output) => false;

        protected override void Solve(IGH_DataAccess access)
        {
            var graph = Item<IrGraphGoo>(access, "ir_graph");
            if (graph?.Value == null)
            {
                Warn("Input parameter 'ir_graph' failed to collect data.");
                return;
            }
            Message = graph.Value.Method;
            SetIndexOptions("source_idx", graph.Value.Sources.Select(x => x.Name));
            SetIndexOptions("receiver_idx", graph.Value.Receivers.Select(x => x.Name));
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
