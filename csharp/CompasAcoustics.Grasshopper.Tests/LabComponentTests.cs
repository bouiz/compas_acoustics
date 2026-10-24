using System;
using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;
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

        private static ScatteringSampleData LabSample() => InteropTests.Serializer.Unpack<ScatteringSampleData>(InteropTests.Fixture("scattering_sample"))!;
        private static ScatteringResultData LabResult() => InteropTests.Serializer.Unpack<ScatteringResultData>(InteropTests.Fixture("scattering_result"))!;

        /// <summary>The answer of <c>run_scattering_lab</c> as the client reads it: the result as data, the lists as lists of objects.</summary>
        private static Dictionary<string, object?> Answer(ScatteringResultData result) => new()
        {
            ["result"] = result,
            ["scattering_coefficients"] = new List<object> { 0.05, 0.1, 0.2, 0.3, 0.4, 0.44, 0.44 },
            ["octave_bands"] = new List<object> { 125L, 250L, 500L, 1000L, 2000L, 4000L, 8000L },
            ["warnings"] = new List<object>(),
        };

        [Fact]
        public void OnlyGeometric_StartsWithoutRun()
        {
            Assert.True(LabRequests.StartsWithoutRun("geometric"));
            Assert.False(LabRequests.StartsWithoutRun("kirchhoff"));
            Assert.False(LabRequests.StartsWithoutRun("bem"));
        }

        [Fact]
        public void Lab_PutsNRaysInTheGeometricSettings()
        {
            var sample = LabSample();
            var (args, kwargs, problem) = LabRequests.Lab(sample, "geometric", null, "iso", 5000, null);
            Assert.Null(problem);
            Assert.Equal(2, args.Length);
            Assert.Same(sample, args[0]);
            Assert.Equal("geometric", args[1]);
            Assert.Equal("iso", kwargs["band_range"]);
            var settings = Assert.IsType<Dictionary<string, object?>>(kwargs["settings"]);
            Assert.Equal(5000L, Convert.ToInt64(settings["n_rays"]));
            Assert.Single(settings);
        }

        [Fact]
        public void Lab_NRaysWithKirchhoff_IsAProblem()
        {
            Assert.Equal("'n_rays' is used by the geometric method only.", LabRequests.Lab(LabSample(), "kirchhoff", null, "iso", 100, null).problem);
            Assert.Equal("'n_rays' is used by the geometric method only.", LabRequests.Lab(LabSample(), "bem", null, "iso", 100, null).problem);
            Assert.Null(LabRequests.Lab(LabSample(), "kirchhoff", null, "iso", null, null).problem);
        }

        [Fact]
        public void Lab_CacheDirGoesToTheBemSettings()
        {
            var (_, kwargs, problem) = LabRequests.Lab(LabSample(), "bem", "survey", "extended", null, "/tmp/numcalc");
            Assert.Null(problem);
            Assert.Equal("extended", kwargs["band_range"]);
            Assert.Equal("survey", kwargs["preset"]);
            var settings = Assert.IsType<Dictionary<string, object?>>(kwargs["settings"]);
            Assert.Equal("/tmp/numcalc", settings["cache_dir"]);
            Assert.Equal("'cache_dir' is used by bem only.", LabRequests.Lab(LabSample(), "geometric", null, "iso", null, "/tmp/x").problem);
            Assert.Equal("'cache_dir' is used by bem only.", LabRequests.Lab(LabSample(), "kirchhoff", null, "iso", null, "/tmp/x").problem);
        }

        [Fact]
        public void Lab_LeavesAnUnsetPresetOut()
        {
            var (_, kwargs, _) = LabRequests.Lab(LabSample(), "geometric", null, "iso", null, null);
            Assert.Equal(new[] { "band_range" }, kwargs.Keys);
            Assert.DoesNotContain("preset", LabRequests.Lab(LabSample(), "geometric", "", "iso", null, null).kwargs.Keys);
            Assert.Equal("fine", LabRequests.Lab(LabSample(), "geometric", "fine", "iso", null, null).kwargs["preset"]);
            // an empty cache_dir is not a value
            Assert.DoesNotContain("settings", LabRequests.Lab(LabSample(), "bem", null, "iso", null, "").kwargs.Keys);
        }

        [Fact]
        public void Read_TakesTheResultAndItsOctaves()
        {
            var result = LabResult();
            var (read, scattering, octaves, diffusion) = LabResults.Read(Answer(result));
            Assert.Same(result, read);
            Assert.Equal(new[] { 0.05, 0.1, 0.2, 0.3, 0.4, 0.44, 0.44 }, scattering);
            Assert.Equal(new[] { 125, 250, 500, 1000, 2000, 4000, 8000 }, octaves);
            Assert.NotEmpty(diffusion);
            Assert.Equal(result.DOctaves.Values, diffusion);
        }

        [Fact]
        public void Read_WithoutD_GivesAnEmptyDiffusion()
        {
            var result = LabResult().Clone();
            result.DOctaves = null;
            Assert.Empty(LabResults.Read(Answer(result)).diffusion);
        }

        [Fact]
        public void Summary_ShowsMethodAndRange()
        {
            var result = LabResult().Clone();
            result.Method = "geometric";
            result.SOctaves.Clear();
            result.SOctaves.AddRange(new[] { 0.0, 0.2, 0.44 });
            var culture = System.Threading.Thread.CurrentThread.CurrentCulture;
            try
            {
                System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("de-CH");
                Assert.Equal("geometric, s 0.00–0.44", LabResults.Summary(result));
            }
            finally
            {
                System.Threading.Thread.CurrentThread.CurrentCulture = culture;
            }
            result.SOctaves.Clear();
            Assert.Equal("geometric", LabResults.Summary(result));
        }

        [Fact]
        public void LabSpec_HasARunButtonAndGeometricAsDefault()
        {
            var spec = ComponentSpec.Load("CA_Scattering_Lab");
            Assert.Equal("08 Lab", spec.Subcategory);
            Assert.Equal(global::Grasshopper.Kernel.GH_Exposure.primary, spec.Exposure);
            Assert.Equal(new[] { "sample", "method", "preset", "band_range", "n_rays", "cache_dir", "run" }, spec.Inputs.Select(p => p.Name));
            Assert.Equal(new[] { "result", "scattering", "diffusion", "octaves" }, spec.Outputs.Select(p => p.Name));
            var inputs = spec.Inputs.ToDictionary(p => p.Name);
            Assert.False(inputs["sample"].Optional);
            Assert.Equal("ghdoc", inputs["sample"].TypeHint);
            Assert.Equal("geometric", inputs["method"].Default);
            Assert.Equal("iso", inputs["band_range"].Default);
            Assert.All(new[] { "preset", "n_rays", "cache_dir", "run" }, n => Assert.True(inputs[n].Optional, n));
            Assert.Equal("int", inputs["n_rays"].TypeHint);
            Assert.Equal("bool", inputs["run"].TypeHint);
            var component = new ScatteringLabComponent();
            Assert.Equal(new System.Guid("2a2910fc-7e2b-5734-9c40-d1a082d917f1"), component.ComponentGuid);
            var controls = component.Controls;
            Assert.Equal(new[] { "geometric", "kirchhoff", "bem" }, controls.OfType<DropdownControl>().Single(c => c.Port == "method").Options);
            Assert.Equal(new[] { "", "survey", "standard", "fine" }, controls.OfType<DropdownControl>().Single(c => c.Port == "preset").Options);
            Assert.Equal(new[] { "iso", "extended" }, controls.OfType<DropdownControl>().Single(c => c.Port == "band_range").Options);
            Assert.Equal("Run", controls.OfType<ButtonControl>().Single(c => c.Port == "run").Label);
            Assert.IsType<Param_ScatteringResult>(component.Params.Output.First());
            Assert.Equal(new[] { "result", "scattering", "diffusion", "octaves" }, component.Params.Output.Select(p => p.Name));
        }
    }
}
