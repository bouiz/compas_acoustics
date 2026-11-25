using System.Globalization;
using System.Linq;
using CompasAcoustics.Pb;
using Rhino.Geometry;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class LabGooTests
    {
        private static ScatteringSampleData Sample() => InteropTests.Serializer.Unpack<ScatteringSampleData>(InteropTests.Fixture("scattering_sample"))!;

        private static ScatteringResultData Result() => InteropTests.Serializer.Unpack<ScatteringResultData>(InteropTests.Fixture("scattering_result"))!;

        private static string F(double value, string format) => value.ToString(format, CultureInfo.InvariantCulture);

        [Fact]
        public void SampleGoo_ToString_NamesModeDiameterDepthAndReference()
        {
            var sample = Sample();
            var goo = new ScatteringSampleGoo(sample);
            var expected = $"Scattering Sample '{sample.Name}', {sample.Mode}, d {F(sample.Diameter, "0.00")} m, h {F(sample.StructuralDepth, "0.000")} m, {sample.ReferenceKind} reference";
            Assert.Equal(expected, goo.ToString());
            Assert.Equal("Scattering Sample", goo.TypeName);
            Assert.StartsWith("Scattering Sample '", goo.ToString());
            Assert.Contains(", d 0.", goo.ToString());
            Assert.Contains(" m, h 0.", goo.ToString());
            Assert.EndsWith(" reference", goo.ToString());
            Assert.Matches(@"^Scattering Sample '.*', \w+, d \d\.\d\d m, h \d\.\d{3} m, \w+ reference$", goo.ToString());
            Assert.NotEmpty(sample.Mode);
            Assert.NotEmpty(sample.ReferenceKind);
        }

        [Fact]
        public void SampleGoo_HasAPreviewBoxAroundTheSample()
        {
            var sample = Sample();
            var goo = new ScatteringSampleGoo(sample);
            var vertices = Enumerable.Range(0, sample.Sample.Vertices.Count / 3)
                .Select(i => new Point3d(sample.Sample.Vertices[3 * i], sample.Sample.Vertices[3 * i + 1], sample.Sample.Vertices[3 * i + 2])).ToArray();
            Assert.NotEmpty(vertices);
            var box = new BoundingBox(vertices);
            Assert.True(goo.ClippingBox.IsValid);
            Assert.Equal(box.Min.X, goo.ClippingBox.Min.X, 9);
            Assert.Equal(box.Min.Y, goo.ClippingBox.Min.Y, 9);
            Assert.Equal(box.Min.Z, goo.ClippingBox.Min.Z, 9);
            Assert.Equal(box.Max.X, goo.ClippingBox.Max.X, 9);
            Assert.Equal(box.Max.Y, goo.ClippingBox.Max.Y, 9);
            Assert.Equal(box.Max.Z, goo.ClippingBox.Max.Z, 9);
            Assert.NotEmpty(goo.Preview!.SampleEdges);
            // the reference's outline: the sides used by one face
            Assert.NotEmpty(goo.Preview.ReferenceOutline);
            // ... the top's outline is a closed loop: every corner is met by two sides, all in one plane
            var corners = goo.Preview.ReferenceOutline.SelectMany(l => new[] { l.From, l.To }).GroupBy(p => (System.Math.Round(p.X, 6), System.Math.Round(p.Y, 6), System.Math.Round(p.Z, 6)));
            Assert.All(corners, c => Assert.Equal(2, c.Count()));
            Assert.False(new ScatteringSampleGoo().ClippingBox.IsValid);
        }

        [Fact]
        public void ResultGoo_ToString_ShowsTheMethodAndTheRangeOfS()
        {
            var result = Result();
            var goo = new ScatteringResultGoo(result);
            var s = result.SOctaves;
            var hz = result.OctaveBands;
            var expected = $"Scattering Result '{result.Name}', {result.Method}, s {F(s.Min(), "0.00")}–{F(s.Max(), "0.00")} ({hz.Min()}–{hz.Max()} Hz), d {F(result.DOctaves.Values.Min(), "0.00")}–{F(result.DOctaves.Values.Max(), "0.00")}";
            Assert.Equal(expected, goo.ToString());
            Assert.Equal("Scattering Result", goo.TypeName);
            Assert.Contains("(125–4000 Hz)", goo.ToString());
            Assert.NotEmpty(goo.Diffusion);
            Assert.Equal(result.DOctaves.Values, goo.Diffusion);
        }

        [Fact]
        public void ResultGoo_WithoutD_HasNoDiffusion()
        {
            var result = Result().Clone();
            result.DOctaves = null;
            var goo = new ScatteringResultGoo(result);
            Assert.Empty(goo.Diffusion);
            Assert.DoesNotContain(", d ", goo.ToString());
            Assert.Contains(", s ", goo.ToString());
        }

        [Fact]
        public void BothGoo_SurviveAWriteAndRead()
        {
            var sample = new ScatteringSampleGoo(Sample());
            var result = new ScatteringResultGoo(Result());
            var chunk = new GH_IO.Serialization.GH_LooseChunk("goo");
            Assert.True(sample.Write(chunk.CreateChunk("sample")));
            Assert.True(result.Write(chunk.CreateChunk("result")));
            var read = new GH_IO.Serialization.GH_LooseChunk("goo");
            read.Deserialize_Binary(chunk.Serialize_Binary());
            var sampleBack = new ScatteringSampleGoo();
            var resultBack = new ScatteringResultGoo();
            Assert.True(sampleBack.Read(read.FindChunk("sample")));
            Assert.True(resultBack.Read(read.FindChunk("result")));
            Assert.Equal(sample.Value, sampleBack.Value);
            Assert.Equal(result.Value, resultBack.Value);
            Assert.Equal(sample.ToString(), sampleBack.ToString());
            Assert.Equal(result.ToString(), resultBack.ToString());
        }

        [Fact]
        public void Parameters_HaveTheirGuids()
        {
            Assert.Equal(new System.Guid("63b9bad1-3835-5f3c-bb45-c2e3b64b96ec"), new Param_ScatteringSample().ComponentGuid);
            Assert.Equal(new System.Guid("6f521d78-33a5-5839-b636-49a3528a1470"), new Param_ScatteringResult().ComponentGuid);
        }
    }
}
