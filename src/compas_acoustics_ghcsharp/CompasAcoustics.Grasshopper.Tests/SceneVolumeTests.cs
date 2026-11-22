using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using CompasAcoustics.Pb;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>The Acoustic Scene's volume from the worker's room_volume, and none when the worker does not give it.</summary>
    [Collection("worker override")]
    public class SceneVolumeTests : IDisposable
    {
        private readonly string _home = Directory.CreateTempSubdirectory("compas_acoustics_home_").FullName;
        private readonly string _python;
        private readonly List<string> _calls = new();

        public SceneVolumeTests()
        {
            _python = Path.Combine(_home, "python");
            File.WriteAllText(_python, "");
        }

        public void Dispose() => Directory.Delete(_home, true);

        private Worker Answering(Func<System.Net.Http.HttpResponseMessage> volume) =>
            new(new FakeStarter(), (uri, token) => new WorkerClient(uri, token, new FakeHandler((request, body) =>
            {
                var path = request.RequestUri!.AbsolutePath;
                if (path == "/ping")
                    return FakeHandler.Pong();
                if (path == "/api/info")
                    return FakeHandler.Result(new Dictionary<string, object?> { ["compas_acoustics"] = Versions.Library, ["schema"] = Schema.Hash, ["hybrid"] = true });
                _calls.Add(path);
                return volume();
            })), name => name == Worker.PythonVariable ? _python : null, _home, TimeSpan.Zero);

        [Fact]
        public void AClosedRoom_HasTheWorkersVolume()
        {
            Assert.Equal(105.0, RoomSceneComponent.RoomVolume(Answering(() => FakeHandler.Result(105.0)), new SceneData()));
            Assert.Equal(new[] { "/api/room_volume" }, _calls);
        }

        [Fact]
        public void AnOpenRoom_HasNoVolume() =>
            Assert.Null(RoomSceneComponent.RoomVolume(Answering(() => FakeHandler.Result(null)), new SceneData()));

        [Fact]
        public void AWorkerWithoutRoomVolume_GivesNoVolume() =>
            Assert.Null(RoomSceneComponent.RoomVolume(
                Answering(() => FakeHandler.Answer(HttpStatusCode.NotFound, new Dictionary<string, object?> { ["error"] = "room_volume is not part of the API." })),
                new SceneData()));

        [Fact]
        public void NoWorker_GivesNoVolume() =>
            Assert.Null(RoomSceneComponent.RoomVolume(new Worker(new FakeStarter(), (uri, token) => new WorkerClient(uri, token),
                name => null, _home, TimeSpan.Zero), new SceneData()));
    }
}
