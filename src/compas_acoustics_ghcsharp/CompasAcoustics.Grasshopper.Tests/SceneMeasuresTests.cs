using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using CompasAcoustics.Pb;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>The Acoustic Scene's volume and surface from the worker's room_measures, asked once per scene, and none when the worker does not give them.</summary>
    public class SceneMeasuresTests : IDisposable
    {
        private readonly string _home = Directory.CreateTempSubdirectory("compas_acoustics_home_").FullName;
        private readonly string _python;
        private readonly List<string> _calls = new();

        public SceneMeasuresTests()
        {
            _python = Path.Combine(_home, "python");
            File.WriteAllText(_python, "");
        }

        public void Dispose() => Directory.Delete(_home, true);

        private Worker Answering(Func<HttpResponseMessage> measures) =>
            new(new FakeStarter(), (uri, token) => new WorkerClient(uri, token, new FakeHandler((request, body) =>
            {
                var path = request.RequestUri!.AbsolutePath;
                if (path == "/ping")
                    return FakeHandler.Pong();
                if (path == "/api/info")
                    return FakeHandler.Result(new Dictionary<string, object?> { ["compas_acoustics"] = Versions.Library, ["schema"] = Schema.Hash, ["hybrid"] = true });
                _calls.Add(path);
                return measures();
            })), name => name == Worker.PythonVariable ? _python : null, _home, TimeSpan.Zero);

        private static HttpResponseMessage Measures(double? volume, double? area) =>
            FakeHandler.Result(new Dictionary<string, object?> { ["volume"] = volume, ["area"] = area });

        [Fact]
        public void AClosedRoom_HasTheWorkersVolumeAndArea_AskedOncePerScene()
        {
            var worker = Answering(() => Measures(105.0, 142.0));
            var component = new RoomSceneComponent();
            var scene = new SceneData { Name = "room" };
            Assert.Equal((105.0, 142.0), component.Measures(worker, scene));
            Assert.Equal((105.0, 142.0), component.Measures(worker, new SceneData { Name = "room" }));
            Assert.Equal(new[] { "/api/room_measures" }, _calls);
            component.Measures(worker, new SceneData { Name = "another room" });
            Assert.Equal(2, _calls.Count);
        }

        [Fact]
        public void AnOpenRoom_HasNoVolume() =>
            Assert.Equal((null, 98.0), new RoomSceneComponent().Measures(Answering(() => Measures(null, 98.0)), new SceneData()));

        [Fact]
        public void AWorkerAnsweringNone_GivesNeither() =>
            Assert.Equal((null, null), new RoomSceneComponent().Measures(Answering(() => FakeHandler.Result(null)), new SceneData()));

        [Fact]
        public void AWorkerWithoutRoomMeasures_GivesNone() =>
            Assert.Null(new RoomSceneComponent().Measures(
                Answering(() => FakeHandler.Answer(HttpStatusCode.NotFound, new Dictionary<string, object?> { ["error"] = "room_measures is not part of the API." })),
                new SceneData()));

        [Fact]
        public void NoWorker_GivesNone() =>
            Assert.Null(new RoomSceneComponent().Measures(new Worker(new FakeStarter(), (uri, token) => new WorkerClient(uri, token),
                name => null, _home, TimeSpan.Zero), new SceneData()));
    }
}
