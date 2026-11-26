using System;
using System.Collections.Generic;
using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;

namespace CompasAcoustics.Grasshopper
{
    public sealed class SimulateMonoComponent : ComponentBase
    {
        public SimulateMonoComponent() : base("CA_Simulate_Mono") { }
        protected override string SpecKey => "CA_Simulate_Mono";
        public override Guid ComponentGuid => new Guid("afb77a6d-dd01-5224-9ee0-70972dcbdbcf");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) =>
            input && port.Name == "scene" ? new Param_Scene() : !input ? new Param_ImpulseResponse() : null;

        private static readonly string[] IsmOrders = Enumerable.Range(0, 11).Select(i => i.ToString(System.Globalization.CultureInfo.InvariantCulture)).ToArray();

        /// <summary>The image source order of the simulations: a stop per order from 0 to 10, 3 (the methods' default) until moved.</summary>
        internal static StepSliderControl IsmOrder => new("ism_order", IsmOrders, "3", IsmOrders, "ISM order");

        public override IReadOnlyList<Control> Controls { get; } = new Control[]
        {
            IsmOrder,
        };

        public override float MinBodyWidth => SharedBodyWidth;

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
                Value<double>(access, "duration"), seed, out var ignored);
            if (ignored.Length > 0)
                Remark($"Ray tracing is off, so {string.Join(", ", ignored)} are not used.");
            if (SavedWithoutAir)
                Remark("This definition was saved with Air off. Simulate Mono now always includes air absorption, as Simulate Binaural and Simulate Ambisonics do, so its responses differ from those saved. Save the definition to clear this remark.");
            if (SavedEnergyThreshold is double threshold)
                Remark($"This definition was saved with energy_thres {threshold:g}. Simulate Mono now always uses the method's 1e-7, so its responses differ from those saved. Save the definition to clear this remark.");
            RunInBackground(access, Simulation(scene, "mono", settings, null, scene), (outcome, _) =>
            {
                var ran = (SceneGoo)outcome.Context!;
                var responses = (IDictionary<string, object?>)((IDictionary<string, object?>)outcome.Result["responses"]!)["mono"]!;
                var irs = PairChoice.InSceneOrder(ran, responses);
                Status(MessageText.Run(ran.SamplingRate, irs.Count));
                access.SetDataList(0, irs);
            });
        }

        protected override bool RunsInBackground => true;
    }
}
