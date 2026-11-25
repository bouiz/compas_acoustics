using System;
using System.Collections.Generic;
using System.Linq;
using GH_IO.Serialization;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;
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
            new ToggleControl("normalize", "Normalize", Inline: true),
        };

        /// <summary>Whether the definition was saved with the former Air toggle off; air absorption is now always on. Cleared on save.</summary>
        internal bool SavedWithoutAir { get; private set; }

        /// <summary>The energy_thres the definition was saved with, if not the method's 1e-7; the method's is now always used. Cleared on save.</summary>
        internal double? SavedEnergyThreshold { get; private set; }

        /// <summary>
        /// A definition saved before: an n_rays input holding 20000, the port's former default, reads empty (the count from
        /// the room's volume); the former absorb_air and energy_thres inputs, between n_rays and duration, are read and
        /// dropped, so the inputs after them keep their values.
        /// </summary>
        public override bool Read(GH_IReader reader)
        {
            SavedWithoutAir = SavedBoolean(reader, "absorb_air") == false
                || (reader.ItemExists("control_value_absorb_air") && string.Equals(reader.GetString("control_value_absorb_air"), "false", StringComparison.OrdinalIgnoreCase));
            // Grasshopper reads the saved inputs by position: a removed input saved as a port needs its place while they are read
            var removed = new List<IGH_Param>();
            if (SavedInput(reader, "absorb_air"))
                removed.Add(new Param_Boolean { Name = "absorb_air", NickName = "absorb_air", Optional = true });
            if (SavedInput(reader, "energy_thres"))
                removed.Add(new Param_Number { Name = "energy_thres", NickName = "energy_thres", Optional = true });
            foreach (var former in removed)
                Params.RegisterInputParam(former, Params.Input.FindIndex(p => p.Name == "duration"));
            var read = base.Read(reader);
            SavedEnergyThreshold = removed.OfType<Param_Number>().SelectMany(p => p.PersistentData.AllData(true).OfType<GH_Number>())
                .Select(g => (double?)g.Value).FirstOrDefault(v => v != 1e-7);
            foreach (var former in removed)
                Params.UnregisterInputParameter(former, true);
            if (removed.Count > 0)
                Params.OnParametersChanged();
            if (Params.Input.Find(p => p.Name == "n_rays") is Param_Integer port && port.SourceCount == 0
                && port.PersistentData.AllData(true).OfType<GH_Integer>().Select(g => g.Value).SequenceEqual(new[] { 20000 }))
                port.PersistentData.Clear();
            return read;
        }

        public override bool Write(GH_IWriter writer)
        {
            SavedWithoutAir = false;
            SavedEnergyThreshold = null;
            return base.Write(writer);
        }

        protected override void Solve(IGH_DataAccess access)
        {
            if (Required<SceneGoo>(access, "scene") is not { } scene)
                return;
            var rayTracing = Value<bool>(access, "ray_tracing") ?? true;
            var seed = Value<int>(access, "seed");
            var settings = SimulationSettings.Mono(Value<int>(access, "ism_order"), rayTracing, Value<int>(access, "n_rays"),
                Value<double>(access, "duration"), seed, Value<bool>(access, "normalize"), out var ignored);
            if (ignored.Length > 0)
                Remark($"Ray tracing is off, so {string.Join(", ", ignored)} are not used.");
            if (SavedWithoutAir)
                Remark("This definition was saved with Air off. Simulate Mono now always includes air absorption, as Simulate Binaural and Simulate Ambisonics do, so its responses differ from those saved. Save the definition to clear this remark.");
            if (SavedEnergyThreshold is double threshold)
                Remark($"This definition was saved with energy_thres {threshold:g}. Simulate Mono now always uses the method's 1e-7, so its responses differ from those saved. Save the definition to clear this remark.");
            RunInBackground(access, Simulation(scene, "mono", settings, null, (scene, settings)), (outcome, _) =>
            {
                var (ran, used) = ((SceneGoo, Dictionary<string, object?>))outcome.Context!;
                var responses = (IDictionary<string, object?>)((IDictionary<string, object?>)outcome.Result["responses"]!)["mono"]!;
                var graph = IrGraphGoo.FromSimulation(ran, "mono", responses);
                // the seed the run used, the method's own when none was given
                var shown = new Dictionary<string, object?>(used);
                if (graph.Value.Responses.FirstOrDefault()?.Generation is { } generation && generation.TryGetValue("seed", out var ranSeed) && ranSeed != null)
                    shown["seed"] = ranSeed;
                Status(SimulationSettings.MonoMessage(ran.SamplingRate, shown));
                access.SetData(0, graph);
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

        public override IReadOnlyList<Control> Controls { get; } = new Control[] { SceneIndices.SourceDropdown, SceneIndices.ReceiverDropdown };

        /// <summary>Nothing: the sources and receivers are drawn by their own components, the edges are an output to wire on.</summary>
        public override bool PreviewsOutput(IGH_Param output) => false;

        protected override void Solve(IGH_DataAccess access)
        {
            if (Required<IrGraphGoo>(access, "ir_graph") is not { } graph)
                return;
            Status(graph.Value.Method);
            SetIndexOptions("source_idx", graph.Value.Sources.Select(x => x.Name));
            SetIndexOptions("receiver_idx", graph.Value.Receivers.Select(x => x.Name));
            var branches = IrGraphGrouping.Group(graph.Value, Items<int>(access, "source_idx"), Items<int>(access, "receiver_idx"), Warn);
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
