using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using CompasAcoustics.Pb;
using CompasPb;
using CompasPb.Data;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>The payloads Python's compas_pb wrote (tests/pb_fixtures/generate.py) read here as typed messages.</summary>
    public class InteropTests
    {
        internal static readonly CompasPbSerializer Serializer = new CompasPbSerializer();

        internal static byte[] Fixture(string name) =>
            Convert.FromBase64String(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", name + ".b64")).Trim());

        internal static string Guid(int n) => new System.Guid(0, 0, 0, new byte[] { 0, 0, 0, 0, 0, 0, 0, (byte)n }).ToString();

        [Fact]
        public void Unpack_SceneFixture_GivesATypedScene()
        {
            CompasPbRegistration.Ensure();
            var scene = Serializer.Unpack<SceneData>(Fixture("scene"))!;
            Assert.Equal(Guid(3), scene.Guid);
            Assert.Equal(44100, scene.SamplingRate);
            Assert.Equal(6, scene.AcousticObjects.Count);
            Assert.Equal("src", scene.Sources.Single().Name);
            Assert.Equal(1.6, scene.Sources[0].Point.Z, 12);
            Assert.Equal(Guid(2), scene.Receivers.Single().Guid);
            var wall = scene.AcousticObjects[0];
            Assert.Equal(AcousticObjectData.GeometryOneofCase.Mesh, wall.GeometryCase);
            Assert.Equal(12, wall.Mesh.Vertices.Count);
            Assert.Equal(new uint[] { 4 }, wall.Mesh.FaceSizes.ToArray());
            Assert.Equal("wall", wall.Material.Name);
            Assert.Equal("boundary", wall.Role);
        }

        [Fact]
        public void Unpack_SourceFixture_KeepsDirectivityAndGuid()
        {
            CompasPbRegistration.Ensure();
            var source = Serializer.Unpack<SourceData>(Fixture("source"))!;
            Assert.Equal(Guid(1), source.Guid);
            Assert.Equal("cardioid", source.Directivity);
            Assert.Equal(new[] { 0.0, 1.0, 0.0 }, new[] { source.Direction.X, source.Direction.Y, source.Direction.Z });
            Assert.False(source.HasSoundFile);
        }

        [Fact]
        public void Unpack_IrFixture_KeepsSamplesShapeAndGeneration()
        {
            CompasPbRegistration.Ensure();
            var ir = Serializer.Unpack<ImpulseResponseData>(Fixture("ir"))!;
            Assert.Equal(new long[] { 4, 2 }, ir.Shape.ToArray());
            Assert.Equal(8, ir.Samples.Count);
            Assert.Equal(0.05, ir.Samples[1], 12);
            Assert.Equal(48000, ir.SamplingRate);
            var generation = (IDictionary<string, object?>)Serializer.UnpackAnyData(ir.Generation)!;
            Assert.Equal("binaural", generation["output"]);
            Assert.Equal("src", ir.Source.Name);
        }

        [Fact]
        public void Unpack_RayFieldFixture_KeepsScatteredRaysAndSources()
        {
            CompasPbRegistration.Ensure();
            var field = Serializer.Unpack<RayFieldData>(Fixture("rayfield"))!;
            var ray = field.Rays.Single();
            Assert.Equal(9, ray.Points.Count);
            var scattered = ray.ScatteredReflections.Single();
            Assert.Equal(1, scattered.OriginReflectionOrder);
            Assert.Equal(3.0, scattered.DistanceOffset);
            Assert.Equal("src", field.Sources[0].Name);
        }

        [Fact]
        public void Unpack_AudioFixture_KeepsSamplesAndFormat()
        {
            CompasPbRegistration.Ensure();
            var audio = Serializer.Unpack<AudioData>(Fixture("audio"))!;
            Assert.Equal(new long[] { 3, 2 }, audio.Shape.ToArray());
            Assert.Equal("binaural", audio.Format);
            Assert.Equal(0.5, audio.Samples[5], 12);
        }

        [Fact]
        public void Unpack_WithoutKnowingTheType_GivesTheMessage()
        {
            CompasPbRegistration.Ensure();
            Assert.IsType<SceneData>(Serializer.Unpack(Fixture("scene")));
        }

        [Fact]
        public void Pack_ASource_RoundTripsAsAMessageOfItsOwnType()
        {
            CompasPbRegistration.Ensure();
            var source = new SourceData { Guid = Guid(42), Name = "s", Point = new PointData { X = 1, Y = 2, Z = 3 }, Directivity = "omnidirectional" };
            var bytes = Serializer.Pack(source);
            var envelope = MessageData.Parser.ParseFrom(bytes);
            Assert.Equal(AnyData.DataOneofCase.Message, envelope.Data.DataCase);
            Assert.EndsWith("/compas_acoustics.pb.SourceData", envelope.Data.Message.TypeUrl);
            Assert.Equal(source, Serializer.Unpack<SourceData>(bytes));
        }

        [Fact]
        public void Pack_ADictionaryOfArgs_ReadsBackItsItems()
        {
            CompasPbRegistration.Ensure();
            var request = new Dictionary<string, object?>
            {
                ["args"] = new List<object?> { new SourceData { Name = "s" }, "pyroomacoustics", 3L, 0.5, null },
                ["kwargs"] = new Dictionary<string, object?>(),
            };
            var back = (IDictionary<string, object?>)Serializer.Unpack(Serializer.Pack(request))!;
            var args = (IList<object?>)back["args"]!;
            Assert.IsType<SourceData>(args[0]);
            Assert.Equal("pyroomacoustics", args[1]);
            Assert.Equal(3L, args[2]);
            Assert.Equal(0.5, args[3]);
            Assert.Null(args[4]);
        }

        [Fact]
        public void Schema_Hash_MatchesTheProtoFiles()
        {
            var root = Path.Combine(RepositoryRoot(), "src", "compas_acoustics", "pb", "protobuf_defs");
            using var sha = SHA256.Create();
            var files = Directory.GetFiles(root, "*.proto", SearchOption.AllDirectories)
                .Select(path => (relative: Path.GetRelativePath(root, path).Replace('\\', '/'), path))
                .OrderBy(file => file.relative, StringComparer.Ordinal);
            var buffer = new MemoryStream();
            foreach (var (relative, path) in files)
            {
                var name = Encoding.UTF8.GetBytes(relative);
                buffer.Write(name, 0, name.Length);
                var bytes = File.ReadAllBytes(path);
                buffer.Write(bytes, 0, bytes.Length);
            }
            var hash = Convert.ToHexString(sha.ComputeHash(buffer.ToArray())).ToLowerInvariant();
            Assert.Equal(Schema.Hash, hash);
        }

        internal static string RepositoryRoot()
        {
            var folder = new DirectoryInfo(AppContext.BaseDirectory);
            while (folder != null && !File.Exists(Path.Combine(folder.FullName, "pyproject.toml")))
                folder = folder.Parent;
            return folder?.FullName ?? throw new DirectoryNotFoundException("The repository root (pyproject.toml) is not above the test output.");
        }
    }
}
