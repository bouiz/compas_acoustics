using System;
using System.Collections.Generic;
using CompasAcoustics.Pb;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;

namespace CompasAcoustics.Grasshopper
{
    public sealed class ScatteringSampleComponent : ComponentBase
    {
        internal static readonly string[] Modes = { "as_drawn", "disc" };
        internal static readonly string[] Symmetries = { "none", "rotational" };

        public ScatteringSampleComponent() : base("CA_Scattering_Sample") { }
        protected override string SpecKey => "CA_Scattering_Sample";
        public override Guid ComponentGuid => new Guid("ff1614f6-bb68-5861-aed6-d8d222af8471");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => input ? null : new Param_ScatteringSample();

        public override IReadOnlyList<Control> Controls { get; } = new Control[]
        {
            new DropdownControl("mode", Modes, Modes[0]), new DropdownControl("symmetry", Symmetries, Symmetries[0]),
        };

        protected override void Solve(IGH_DataAccess access)
        {
            var goo = Item<IGH_Goo>(access, "geometry");
            if (goo == null)
            {
                Warn("No geometry provided.");
                return;
            }
            var mesh = LabRequests.MeshOf(goo.ScriptVariable());
            if (mesh == null)
            {
                Warn("Input parameter 'geometry' is not a mesh, Brep, surface or SubD with faces.");
                return;
            }
            var referenceGoo = Item<IGH_Goo>(access, "reference");
            var reference = referenceGoo == null ? null : LabRequests.MeshOf(referenceGoo.ScriptVariable());
            if (referenceGoo != null && reference == null)
            {
                Warn("Input parameter 'reference' is not a mesh, Brep, surface or SubD with faces.");
                return;
            }
            var (args, kwargs, problem) = LabRequests.Sample(mesh, Text(access, "mode") ?? Modes[0], Value<double>(access, "diameter"), reference,
                Value<double>(access, "baseplate"), Text(access, "symmetry"), Text(access, "name"));
            if (problem != null)
            {
                Warn(problem);
                return;
            }
            var sample = (ScatteringSampleData)Worker.Call("make_scattering_sample", args, kwargs)!;
            foreach (var warning in sample.Warnings)
                Warn(warning);
            Status(LabRequests.SampleStatus(sample.Mode, sample.Diameter));
            access.SetData(0, new ScatteringSampleGoo(sample));
        }
    }

    /// <summary>
    /// The virtual scattering lab. The geometric method (seconds) runs when the inputs change; kirchhoff and bem (minutes to hours)
    /// run only when Run is pressed, and a press with unchanged inputs runs them again.
    /// </summary>
    public sealed class ScatteringLabComponent : ComponentBase
    {
        internal static readonly string[] Methods = { "geometric", "kirchhoff", "bem" };
        internal static readonly string[] Presets = { "", "survey", "standard", "fine" };
        internal static readonly string[] PresetLabels = { "Default", "survey (13)", "standard (97)", "fine (864)" };
        internal static readonly string[] BandRanges = { "iso", "extended" };

        public ScatteringLabComponent() : base("CA_Scattering_Lab") { }
        protected override string SpecKey => "CA_Scattering_Lab";
        public override Guid ComponentGuid => new Guid("2a2910fc-7e2b-5734-9c40-d1a082d917f1");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) =>
            input ? (port.Name == "sample" ? new Param_ScatteringSample() : null) : (port.Name == "result" ? new Param_ScatteringResult() : null);

        public override IReadOnlyList<Control> Controls { get; } = new Control[]
        {
            new DropdownControl("method", Methods, Methods[0]),
            new DropdownControl("preset", Presets, "", PresetLabels, "Preset"),
            new DropdownControl("band_range", BandRanges, BandRanges[0], Title: "Band range"),
            new ButtonControl("run", "Run"),
        };

        protected override bool RunsInBackground => true;

        protected override void Solve(IGH_DataAccess access)
        {
            var sample = Item<ScatteringSampleGoo>(access, "sample");
            if (sample?.Value == null)
            {
                Warn("Input parameter 'sample' failed to collect data.");
                return;
            }
            var method = Text(access, "method") ?? Methods[0];
            var (args, kwargs, problem) = LabRequests.Lab(sample.Value, method, Text(access, "preset"), Text(access, "band_range") ?? BandRanges[0],
                Value<int>(access, "n_rays"), Text(access, "cache_dir"));
            if (problem != null)
            {
                Warn(problem);
                return;
            }
            var delivered = false;
            // the wave methods take minutes to hours and Grasshopper recomputes on every upstream change: they start on Run,
            // a press with unchanged inputs runs again; until then the outputs of the last run stay
            var pressed = Value<bool>(access, "run") == true || Pressed("run");
            var decision = RunInBackground(access, new Background("run_scattering_lab", args, kwargs, sample),
                (outcome, _) =>
                {
                    var answer = (IDictionary<string, object?>)outcome.Result!;
                    ShowWarnings(answer);
                    var (result, scattering, octaves, diffusion) = LabResults.Read(answer);
                    Status(LabResults.Summary(result));
                    access.SetData("result", new ScatteringResultGoo(result));
                    access.SetDataList("scattering", scattering);
                    access.SetDataList("diffusion", diffusion);
                    access.SetDataList("octaves", octaves);
                    delivered = true;
                },
                start: LabRequests.StartsWithoutRun(method) || pressed, restart: pressed);
            if (decision.Kind == RunKind.Idle)
            {
                Remark("Press Run: kirchhoff and bem take minutes to hours.");
                if (!delivered)
                    Status("press Run");
            }
        }
    }
}
