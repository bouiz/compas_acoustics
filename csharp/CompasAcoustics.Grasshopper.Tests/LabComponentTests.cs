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

        [Fact]
        public void PreviewLabels_NameTheRingAndAzimuthOfEachPreviewSource()
        {
            var result = LabResult();
            var labels = LabRequests.PreviewLabels(result);
            Assert.Equal(result.PolarSourceIndices.Count, labels.Length);
            Assert.NotEmpty(labels);
            for (var i = 0; i < labels.Length; i++)
            {
                var at = result.PolarSourceIndices[i];
                var expected = string.Format(System.Globalization.CultureInfo.InvariantCulture, "\u03b8 {0:0.##}\u00b0, \u03c6 {1:0.##}\u00b0", result.SourceColatitude[at], result.SourceAzimuth[at]);
                Assert.Equal(expected, labels[i]);
            }
            var set = result.Clone();
            set.SourceColatitude.Clear();
            set.SourceColatitude.AddRange(new[] { 0.0, 30.0, 22.5 });
            set.SourceAzimuth.Clear();
            set.SourceAzimuth.AddRange(new[] { 0.0, 0.0, 45.0 });
            set.PolarSourceIndices.Clear();
            set.PolarSourceIndices.AddRange(new[] { 1, 2 });
            var culture = System.Threading.Thread.CurrentThread.CurrentCulture;
            try
            {
                System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("de-CH");
                Assert.Equal(new[] { "\u03b8 30\u00b0, \u03c6 0\u00b0", "\u03b8 22.5\u00b0, \u03c6 45\u00b0" }, LabRequests.PreviewLabels(set));
            }
            finally
            {
                System.Threading.Thread.CurrentThread.CurrentCulture = culture;
            }
        }

        [Fact]
        public void Balloon_PassesPreviewOctaveBodyRadiusAndFloor()
        {
            var result = LabResult();
            var octave = result.OctaveBands[result.OctaveBands.Count - 1];
            var (args, kwargs, problem) = LabRequests.Balloon(result, 0, octave, "reference", 2.0, -20.0);
            Assert.Null(problem);
            Assert.Single(args);
            Assert.Same(result, args[0]);
            Assert.Equal(new[] { "body", "floor_db", "octave", "preview", "radius" }, kwargs.Keys.OrderBy(k => k, StringComparer.Ordinal));
            Assert.Equal(0L, Convert.ToInt64(kwargs["preview"]));
            Assert.Equal((long)octave, Convert.ToInt64(kwargs["octave"]));
            Assert.Equal("reference", kwargs["body"]);
            Assert.Equal(2.0, kwargs["radius"]);
            Assert.Equal(-20.0, kwargs["floor_db"]);
        }

        [Fact]
        public void Balloon_AnOctaveThatWasNotSimulated_IsAProblem()
        {
            var result = LabResult().Clone();
            result.OctaveBands.Clear();
            result.OctaveBands.AddRange(new[] { 125, 250, 500 });
            Assert.Equal("'octave' must be one of 125, 250, 500 Hz", LabRequests.Balloon(result, 0, 1000, "sample", 1.0, -30.0).problem);
            Assert.Null(LabRequests.Balloon(result, 0, 250, "sample", 1.0, -30.0).problem);
        }

        [Fact]
        public void Balloon_APreviewOutOfRange_IsAProblem()
        {
            var result = LabResult();
            var last = result.PolarSourceIndices.Count - 1;
            var octave = result.OctaveBands[0];
            Assert.Equal($"'preview' must be 0 to {last}", LabRequests.Balloon(result, last + 1, octave, "sample", 1.0, -30.0).problem);
            Assert.Equal($"'preview' must be 0 to {last}", LabRequests.Balloon(result, -1, octave, "sample", 1.0, -30.0).problem);
            Assert.Null(LabRequests.Balloon(result, last, octave, "sample", 1.0, -30.0).problem);
        }

        [Fact]
        public void FaceLevels_AreTheMeanOfTheirVertices()
        {
            var levels = new[] { 0.0, -6.0, -12.0, -30.0 };
            var faces = new[] { new[] { 0, 1, 2 }, new[] { 1, 2, 3 }, new[] { 0, 1, 2, 3 } };
            Assert.Equal(new[] { -6.0, -16.0, -12.0 }, LabResults.FaceLevels(levels, faces));
            Assert.Empty(LabResults.FaceLevels(levels, Array.Empty<int[]>()));
        }

        [Fact]
        public void PolarPreviewSpec_HasItsPortsAndDefaults()
        {
            var spec = ComponentSpec.Load("CA_Polar_Preview");
            Assert.Equal("08 Lab", spec.Subcategory);
            Assert.Equal(global::Grasshopper.Kernel.GH_Exposure.primary, spec.Exposure);
            Assert.Equal(new[] { "result", "preview", "octave", "body", "radius", "floor_db", "centre" }, spec.Inputs.Select(p => p.Name));
            Assert.Equal(new[] { "mesh", "levels", "source" }, spec.Outputs.Select(p => p.Name));
            var inputs = spec.Inputs.ToDictionary(p => p.Name);
            Assert.False(inputs["result"].Optional);
            Assert.Equal("ghdoc", inputs["result"].TypeHint);
            Assert.All(new[] { "preview", "octave", "body", "radius", "floor_db", "centre" }, n => Assert.True(inputs[n].Optional, n));
            Assert.Equal("int", inputs["preview"].TypeHint);
            Assert.Equal("int", inputs["octave"].TypeHint);
            Assert.Equal("sample", inputs["body"].Default);
            Assert.Equal(1.0, inputs["radius"].Default);
            Assert.Equal(-30.0, Convert.ToDouble(inputs["floor_db"].Default));
            Assert.Equal("point", inputs["centre"].TypeHint);
            var component = new PolarPreviewComponent();
            Assert.Equal(new System.Guid("caa96e48-6c1b-5a10-bf2f-b8fcb160f04b"), component.ComponentGuid);
            Assert.Equal(new[] { "preview", "octave", "body" }, component.Controls.OfType<DropdownControl>().Select(c => c.Port));
            Assert.Equal(new[] { "sample", "reference" }, component.Controls.OfType<DropdownControl>().Single(c => c.Port == "body").Options);
            Assert.Single(component.Controls.OfType<ToggleControl>());
            Assert.False(component.PreviewsOutput(component.Params.Output[0]));
        }

        [Fact]
        public void ReadBalloon_TakesTheMeshLevelsSourceAndLabel()
        {
            var mesh = Mesh();
            var answer = new Dictionary<string, object?>
            {
                ["mesh"] = mesh,
                ["levels"] = new List<object> { 0.0, -3.0, -30.0 },
                ["source"] = new List<object> { 0.6, 0.0, 1.0 },
                ["label"] = "\u03b8 30\u00b0, \u03c6 0\u00b0, 1000 Hz, sample",
            };
            var (read, levels, source, label) = LabResults.ReadBalloon(answer);
            Assert.Same(mesh, read);
            Assert.Equal(new[] { 0.0, -3.0, -30.0 }, levels);
            Assert.Equal(new[] { 0.6, 0.0, 1.0 }, source);
            Assert.Equal("\u03b8 30\u00b0, \u03c6 0\u00b0, 1000 Hz, sample", label);
        }

        [Fact]
        public void PolarPreview_OffersTheOctavesAndPreviewsOfItsResult_NotAll()
        {
            var component = new PolarPreviewComponent();
            var octave = component.Controls.OfType<DropdownControl>().Single(c => c.Port == "octave");
            Assert.Equal("1000", octave.Default);
            component.SetValueOptions("octave", new[] { 125, 250, 500 }, new[] { "125 Hz", "250 Hz", "500 Hz" });
            component.SetValueOptions("preview", new[] { 0, 1 }, new[] { "\u03b8 0\u00b0, \u03c6 0\u00b0", "\u03b8 30\u00b0, \u03c6 0\u00b0" });
            var (values, labels) = component.OptionsOf(octave);
            Assert.Equal(new[] { "125", "250", "500" }, values);
            Assert.Equal(new[] { "125 Hz", "250 Hz", "500 Hz" }, labels);
            var preview = component.Controls.OfType<DropdownControl>().Single(c => c.Port == "preview");
            Assert.Equal(new[] { "0", "1" }, component.OptionsOf(preview).values);
            // the named values of the integer ports, for the shown input's menu
            var port = (global::Grasshopper.Kernel.Parameters.Param_Integer)component.ControlPort("octave")!;
            Assert.True(port.HasNamedValues);
        }

        private static readonly double[] SevenValues = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };

        [Fact]
        public void Material_WithoutAbsorptionOrMaterial_IsAProblem()
        {
            var (_, _, problem) = LabRequests.Material("panel", new List<double>(), null, null, null);
            Assert.Equal("Connect 'absorption' (1 or 7 values) or a 'material' to take it from.", problem);
        }

        [Fact]
        public void Material_FiveValues_IsAProblem()
        {
            var (_, _, problem) = LabRequests.Material("panel", new List<double> { 0.1, 0.2, 0.3, 0.4, 0.5 }, null, null, null);
            Assert.Equal("'absorption' takes 1 or 7 values (125 Hz to 8 kHz).", problem);
            Assert.Null(LabRequests.Material("panel", new List<double> { 0.3 }, null, null, null).problem);
            Assert.Null(LabRequests.Material("panel", SevenValues, null, null, null).problem);
        }

        [Fact]
        public void Material_TakesTheMaterialWhenAbsorptionIsEmpty()
        {
            var material = new AcousticMaterialData { Name = "plaster" };
            var (args, kwargs, problem) = LabRequests.Material("panel", new List<double>(), material, null, null);
            Assert.Null(problem);
            Assert.Equal(2, args.Length);
            Assert.Equal("panel", args[0]);
            Assert.Same(material, args[1]);
            Assert.Empty(kwargs);
        }

        [Fact]
        public void Material_NumbersWinOverAMaterial_AndTheResultAndCategoryAreKeywords()
        {
            var material = new AcousticMaterialData { Name = "plaster" };
            var result = new ScatteringResultData();
            var (args, kwargs, problem) = LabRequests.Material("panel", SevenValues, material, result, "Absorbers");
            Assert.Null(problem);
            Assert.Equal(SevenValues, args[1]);
            Assert.Equal(new[] { "category", "result" }, kwargs.Keys.OrderBy(k => k));
            Assert.Same(result, kwargs["result"]);
            Assert.Equal("Absorbers", kwargs["category"]);
        }

        [Fact]
        public void LabMaterialSpec_DescribesTheProxy()
        {
            var spec = ComponentSpec.Load("CA_Lab_Material");
            Assert.Equal("08 Lab", spec.Subcategory);
            Assert.Equal(global::Grasshopper.Kernel.GH_Exposure.primary, spec.Exposure);
            Assert.Contains("proxy", spec.Description);
            Assert.Equal(new[] { "name", "absorption", "material", "result", "category" }, spec.Inputs.Select(p => p.Name));
            Assert.Equal(new[] { "material" }, spec.Outputs.Select(p => p.Name));
            var inputs = spec.Inputs.ToDictionary(p => p.Name);
            Assert.False(inputs["name"].Optional);
            Assert.All(new[] { "absorption", "material", "result", "category" }, n => Assert.True(inputs[n].Optional, n));
            Assert.Equal("ghdoc", inputs["material"].TypeHint);
            Assert.Equal("ghdoc", inputs["result"].TypeHint);
            var component = new LabMaterialComponent();
            Assert.Equal(new System.Guid("94143bb0-1ea1-5d3c-9bc9-edbeb8bfeea7"), component.ComponentGuid);
            Assert.Equal(ConstructMaterialComponent.Categories, component.Controls.OfType<DropdownControl>().Single(c => c.Port == "category").Options);
            Assert.IsType<Param_Material>(component.Params.Output.Single());
            Assert.IsType<Param_Material>(component.Params.Input.Single(p => p.Name == "material"));
            Assert.IsType<Param_ScatteringResult>(component.Params.Input.Single(p => p.Name == "result"));
        }
    }
}
