using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;
using CompasPb.Data;
using GH_IO.Serialization;
using Rhino.Geometry;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class GooTests
    {
        private static T Fixture<T>(string name) where T : class, Google.Protobuf.IMessage<T>, new()
        {
            CompasPbRegistration.Ensure();
            return InteropTests.Serializer.Unpack<T>(InteropTests.Fixture(name))!;
        }

        [Fact]
        public void SourceGoo_ReadsNamePositionAndSoundFile()
        {
            var source = new SourceGoo(Fixture<SourceData>("source"));
            Assert.True(source.IsValid);
            Assert.Equal(new Point3d(1, 2, 3), source.Position);
            Assert.Null(source.SoundFile);
            Assert.Equal("Source 'src'", source.ToString());
            Assert.Equal(InteropTests.Guid(1), source.Guid);
        }

        [Fact]
        public void MessageGoo_RoundTripsThroughWriteAndRead()
        {
            var scene = new SceneGoo(Fixture<SceneData>("scene"));
            var chunk = new GH_LooseChunk("goo");
            Assert.True(scene.Write(chunk));
            var bytes = chunk.Serialize_Binary();
            var read = new GH_LooseChunk("goo");
            read.Deserialize_Binary(bytes);
            var back = new SceneGoo();
            Assert.True(back.Read(read));
            Assert.Equal(scene.Value, back.Value);
        }

        [Fact]
        public void MessageGoo_DuplicateIsADeepCopy()
        {
            var source = new SourceGoo(Fixture<SourceData>("source"));
            var copy = (SourceGoo)source.Duplicate();
            copy.Value.Name = "changed";
            Assert.Equal("src", source.Value.Name);
        }

        [Fact]
        public void MessageGoo_CastsFromItsMessageAndToIt()
        {
            var message = Fixture<SourceData>("source");
            var goo = new SourceGoo();
            Assert.True(goo.CastFrom(message));
            Assert.Same(message, goo.Value);
            Assert.False(goo.CastFrom(new ReceiverData()));
            SourceData? target = null;
            Assert.True(goo.CastTo(ref target));
            Assert.Same(message, target);
        }

        [Fact]
        public void SceneGoo_ListsSourcesAndReceiversInOrder()
        {
            var scene = new SceneGoo(Fixture<SceneData>("scene"));
            Assert.Equal(new[] { "src" }, scene.Sources.Select(s => s.Value.Name).ToArray());
            Assert.Equal(new[] { "rec" }, scene.Receivers.Select(r => r.Value.Name).ToArray());
            Assert.Equal(44100, scene.SamplingRate);
            Assert.Equal(new Point3d(5, 3, 1.5), scene.Receivers[0].Position);
        }

        [Fact]
        public void ImpulseResponseGoo_ReadsItsMetadata()
        {
            var ir = new ImpulseResponseGoo(Fixture<ImpulseResponseData>("ir"));
            Assert.Equal(48000, ir.SamplingRate);
            Assert.Equal(2, ir.Channels);
            Assert.Equal(4, ir.Samples);
            Assert.True(ir.Binaural);
            Assert.Equal("test", ir.Method);
            Assert.Equal(InteropTests.Guid(1), ir.SourceGuid);
            Assert.Equal(InteropTests.Guid(2), ir.ReceiverGuid);
            Assert.Equal("IR src -> rec, 2 ch, 48000 Hz", ir.ToString());
        }

        [Fact]
        public void IrGraphGoo_FromSimulation_KeepsSceneOrderAndMapsGuids()
        {
            var scene = new SceneGoo(Fixture<SceneData>("scene"));
            var ir = Fixture<ImpulseResponseData>("ir");
            ir.Source = scene.Value.Sources[0].Clone();
            ir.Receiver = scene.Value.Receivers[0].Clone();
            var key = $"{ir.Receiver.Guid}_{ir.Source.Guid}";
            var graph = IrGraphGoo.FromSimulation(scene, "hybrid_binaural", new Dictionary<string, object?> { [key] = ir });
            Assert.Equal("hybrid_binaural", graph.Value.Method);
            Assert.Same(ir, graph.Value.Responses.Single().Value);
            Assert.Equal("src", graph.Value.Sources.Single().Value.Name);
            Assert.Equal("hybrid_binaural: 1 response", graph.ToString());
        }

        [Fact]
        public void ParameterGoo_ToString_MatchesThePythonPanelText()
        {
            var t30 = new ParameterGoo(new ParameterGoo.Content("T30", "Reverberation Time (T30)", "Frequency (Hz)", "Time (s)", "bands",
                new[] { 125.0, 250.0, 1000.0 }, new[] { new[] { 0.98432, 1.1712, 1.94 } }, 1.7, new string[0]));
            Assert.Equal("T30, Time (s): 125 Hz 0.984, 250 Hz 1.17, 1000 Hz 1.94; overall 1.7", t30.ToString());
            var curve = new ParameterGoo(new ParameterGoo.Content("IR", "Impulse Response", "Time (s)", "Level (dB)", "linear",
                new[] { 0.0, 0.5, 1.25 }, new[] { new[] { 0.0, -3.0, -60.0 } }, null, new string[0]));
            Assert.Equal("Impulse Response: Time (s) 0 to 1.25, 3 points", curve.ToString());
        }

        [Fact]
        public void ParameterGoo_FromDictionary_ReadsTheWorkersShape()
        {
            var parameter = ParameterGoo.FromDictionary(new Dictionary<string, object?>
            {
                ["name"] = "T30", ["title"] = "Reverberation Time (T30)", ["x_unit"] = "Frequency (Hz)", ["y_unit"] = "Time (s)",
                ["x_scale"] = "bands", ["frequencies"] = new List<object?> { 125L, 250.0 }, ["values"] = new List<object?> { new List<object?> { 1.0, 2L } },
                ["overall"] = null, ["channels"] = new List<object?>(),
            });
            Assert.Equal(new[] { 125.0, 250.0 }, parameter.Value.Frequencies);
            Assert.Equal(new[] { 1.0, 2.0 }, parameter.Value.Values[0]);
            Assert.Null(parameter.Value.Overall);
        }

        [Fact]
        public void ReflectionsGoo_CastsToRayFieldGoo()
        {
            var paths = new SpecularReflectionsData { Guid = "g" };
            paths.Paths.Add(new SpecularPathData { Order = 1, SourceIndex = 0, Points = { 0, 0, 1, 3, 0, 1, 5, 3, 1.5 } });
            paths.Sources.Add(0, new SourceData { Name = "src" });
            RayFieldGoo? field = null;
            Assert.True(new ReflectionsGoo(paths).CastTo(ref field));
            Assert.Equal(9, field!.Value.Rays.Single().Points.Count);
            Assert.Equal("src", field.Value.Sources[0].Name);
        }

        [Fact]
        public void ImpulseResponseGoo_ReadsTheGenerationOfItsCurrentValue()
        {
            var ir = new ImpulseResponseGoo(Fixture<ImpulseResponseData>("ir"));
            Assert.True(ir.Binaural);
            Assert.True(ir.CastFrom(new ImpulseResponseData { Shape = { 1 } }));
            Assert.False(ir.Binaural);
            Assert.Null(ir.Method);
        }

        [Fact]
        public void RayFieldGoo_CastsFromReflections()
        {
            var paths = new SpecularReflectionsData();
            paths.Paths.Add(new SpecularPathData { Points = { 0, 0, 1, 1, 0, 1 } });
            var field = new RayFieldGoo();
            Assert.True(field.CastFrom(new ReflectionsGoo(paths)));
            Assert.Single(field.Value.Rays);
        }

        [Theory]
        [InlineData(0.98432, 3, "0.984")]
        [InlineData(1.7, 3, "1.7")]
        [InlineData(1000.0, 6, "1000")]
        [InlineData(1.25, 6, "1.25")]
        [InlineData(0.0, 6, "0")]
        [InlineData(123456789.0, 6, "1.23457e+08")]
        [InlineData(0.00001234, 3, "1.23e-05")]
        [InlineData(-2.5, 3, "-2.5")]
        [InlineData(double.NaN, 3, "nan")]
        public void PythonFormat_G_MatchesPython(double value, int precision, string expected)
        {
            Assert.Equal(expected, PythonFormat.G(value, precision));
        }
    }
}
