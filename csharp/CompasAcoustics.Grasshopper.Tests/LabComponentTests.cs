using System.Linq;
using CompasPb.Data;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>The scattering lab's components (Tasks 5 to 8 of the plan of 2026-10-24): their static, Rhino-free helpers and specs.</summary>
    public class LabComponentTests
    {
        private static MeshData Mesh() => new MeshData { Vertices = { new[] { 0.0, 0.0, 0.0 }, new[] { 1.0, 0.0, 0.0 }, new[] { 0.0, 1.0, 0.1 } }, FaceVertices = { 0u, 1u, 2u }, FaceSizes = { 3u } };

        [Fact]
        public void Sample_DiscWithoutDiameter_IsAProblem()
        {
            var (_, _, problem) = LabRequests.Sample(Mesh(), "disc", null, null, null, null, null);
            Assert.Equal("A disc sample needs 'diameter'.", problem);
        }

        [Fact]
        public void Sample_ReferenceInDiscMode_IsAProblem()
        {
            var (_, _, problem) = LabRequests.Sample(Mesh(), "disc", 1.0, Mesh(), null, null, null);
            Assert.Equal("'reference' is used in as_drawn mode only.", problem);
            Assert.Null(LabRequests.Sample(Mesh(), "as_drawn", null, Mesh(), null, null, null).problem);
            Assert.Null(LabRequests.Sample(Mesh(), "disc", 1.0, null, null, null, null).problem);
        }

        [Fact]
        public void Sample_LeavesUnsetInputsOut()
        {
            var (_, kwargs, _) = LabRequests.Sample(Mesh(), "as_drawn", null, null, null, null, null);
            Assert.Empty(kwargs);
            var (_, set, _) = LabRequests.Sample(Mesh(), "disc", 0.8, null, 0.02, "rotational", "tile");
            Assert.Equal(new[] { "baseplate", "diameter", "name", "symmetry" }, set.Keys.OrderBy(k => k));
            Assert.Equal(0.8, set["diameter"]);
            Assert.Equal(0.02, set["baseplate"]);
            Assert.Equal("rotational", set["symmetry"]);
            Assert.Equal("tile", set["name"]);
            // an empty text is not a value
            Assert.Empty(LabRequests.Sample(Mesh(), "as_drawn", null, null, null, "", "").kwargs);
        }

        [Fact]
        public void Sample_PassesTheMeshAndTheModeAsArguments()
        {
            var mesh = Mesh();
            var reference = Mesh();
            var (args, kwargs, problem) = LabRequests.Sample(mesh, "as_drawn", null, reference, null, null, null);
            Assert.Null(problem);
            Assert.Equal(2, args.Length);
            Assert.Same(mesh, args[0]);
            Assert.Equal("as_drawn", args[1]);
            Assert.Same(reference, kwargs["reference"]);
        }

        [Fact]
        public void SampleStatus_NamesTheModeAndTheDiameter_WhateverTheCulture()
        {
            var culture = System.Threading.Thread.CurrentThread.CurrentCulture;
            try
            {
                System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("de-CH");
                Assert.Equal("disc, d 0.80 m", LabRequests.SampleStatus("disc", 0.8));
            }
            finally
            {
                System.Threading.Thread.CurrentThread.CurrentCulture = culture;
            }
        }

        [Fact]
        public void SampleSpec_HasItsPortsAndDefaults()
        {
            var spec = ComponentSpec.Load("CA_Scattering_Sample");
            Assert.Equal("08 Lab", spec.Subcategory);
            Assert.Equal(global::Grasshopper.Kernel.GH_Exposure.primary, spec.Exposure);
            Assert.Equal(new[] { "geometry", "mode", "diameter", "reference", "baseplate", "symmetry", "name" }, spec.Inputs.Select(p => p.Name));
            Assert.Equal(new[] { "sample" }, spec.Outputs.Select(p => p.Name));
            var inputs = spec.Inputs.ToDictionary(p => p.Name);
            Assert.False(inputs["geometry"].Optional);
            Assert.Equal("geometrybase", inputs["geometry"].TypeHint);
            Assert.All(new[] { "mode", "diameter", "reference", "baseplate", "symmetry", "name" }, n => Assert.True(inputs[n].Optional, n));
            Assert.Equal("as_drawn", inputs["mode"].Default);
            Assert.Equal("none", inputs["symmetry"].Default);
            Assert.Equal(0.01, inputs["baseplate"].Default);
            Assert.Null(inputs["diameter"].Default);
            Assert.Equal("float", inputs["diameter"].TypeHint);
            Assert.Equal("geometrybase", inputs["reference"].TypeHint);
            Assert.Equal("str", inputs["name"].TypeHint);
        }

        [Fact]
        public void ScatteringSampleComponent_HasTheDropdownsOfItsModeAndSymmetry()
        {
            var component = new ScatteringSampleComponent();
            Assert.Equal(new[] { "mode", "symmetry" }, component.Controls.OfType<DropdownControl>().Select(c => c.Port));
            Assert.Equal(new[] { "as_drawn", "disc" }, component.Controls.OfType<DropdownControl>().Single(c => c.Port == "mode").Options);
            Assert.Equal(new[] { "none", "rotational" }, component.Controls.OfType<DropdownControl>().Single(c => c.Port == "symmetry").Options);
            Assert.Equal(new System.Guid("ff1614f6-bb68-5861-aed6-d8d222af8471"), component.ComponentGuid);
            Assert.IsType<Param_ScatteringSample>(component.Params.Output.Single());
        }
    }
}
