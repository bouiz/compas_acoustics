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
}
