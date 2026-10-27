using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using CompasAcoustics.Pb;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>The Audio Player's commands to a fake worker.</summary>
    [Collection("worker override")]
    public class AudioPlayerComponentTests : IDisposable
    {
        private readonly List<(string path, IDictionary<string, object?>? payload)> _calls = new();
        private readonly string _home = Directory.CreateTempSubdirectory("compas_acoustics_home_").FullName;
        private bool _fail;

        public AudioPlayerComponentTests()
        {
            var python = Path.Combine(_home, "python");
            File.WriteAllText(python, "");
            Worker.Override = new Worker(new FakeStarter(), (uri, token) => new WorkerClient(uri, token, new FakeHandler(Answer)),
                name => name == Worker.PythonVariable ? python : null, _home, TimeSpan.Zero);
        }

        public void Dispose()
        {
            Worker.Override = null;
            Directory.Delete(_home, true);
        }

        private HttpResponseMessage Answer(HttpRequestMessage request, byte[] body)
        {
            var path = request.RequestUri!.AbsolutePath;
            if (path == "/ping")
                return FakeHandler.Pong();
            if (path == "/api/info")
                return FakeHandler.Result(new Dictionary<string, object?> { ["compas_acoustics"] = Worker.Version, ["schema"] = Schema.Hash, ["hybrid"] = true });
            var payload = body.Length > 0 ? FakeHandler.Serializer.Unpack(body) as IDictionary<string, object?> : null;
            _calls.Add((path, payload));
            if (_fail)
                return FakeHandler.Answer(HttpStatusCode.BadRequest, new Dictionary<string, object?> { ["error"] = "PortAudio error: no device" });
            return FakeHandler.Result(new Dictionary<string, object?> { ["warnings"] = new List<object?>(), ["duration"] = 2.0 });
        }

        private static AudioData Stereo(double seconds = 2.0, int rate = 100)
        {
            var frames = (int)(seconds * rate);
            var audio = new AudioData { SamplingRate = rate, Format = "binaural", Shape = { frames, 2 } };
            audio.Samples.Add(Enumerable.Range(0, frames * 2).Select(i => Math.Sin(i * 0.1)));
            return audio;
        }

        private IDictionary<string, object?> Kwargs(int i) => (IDictionary<string, object?>)_calls[i].payload!["kwargs"]!;

        [Fact]
        public void ADoubleClickOnTheVolume_SetsItBackTo0dB_AndAPlayingSignalGoesOnAtThatLevel()
        {
            var player = new AudioPlayerComponent();
            player.Load(Stereo(), null);
            player.OnGain(-12);
            player.OnPlay();
            player.OnResetVolume();
            Assert.Equal(0.0, player.GainDb);
            Assert.Equal(new[] { "/api/play", "/api/play" }, _calls.Select(c => c.path).ToArray());
            Assert.Equal(0.0, Convert.ToDouble(Kwargs(1)["gain_db"]));
        }

        [Fact]
        public void Normalize_IsOffByDefault_ABodyToggle_SavedWithTheComponent()
        {
            Assert.DoesNotContain("normalize", ComponentSpec.Load("CA_Audio_Player").Inputs.Select(p => p.Name));
            var player = new AudioPlayerComponent();
            Assert.False(player.Normalize);
            player.OnNormalize(true);
            Assert.True(player.Normalize);
            var chunk = new GH_IO.Serialization.GH_LooseChunk("component");
            Assert.True(player.Write(chunk));
            var read = new GH_IO.Serialization.GH_LooseChunk("component");
            read.Deserialize_Binary(chunk.Serialize_Binary());
            var back = new AudioPlayerComponent();
            Assert.True(back.Read(read));
            Assert.True(back.Normalize);
        }

        [Fact]
        public void Play_SendsTheSignalWithGainStartAndNormalize()
        {
            var player = new AudioPlayerComponent();
            player.Load(Stereo(), null);
            player.OnGain(-6);
            player.OnSeek(0.5);
            player.OnPlay();
            Assert.Equal("/api/play", _calls.Single().path);
            Assert.Equal(-6.0, Kwargs(0)["gain_db"]);
            Assert.Equal(0.5, Kwargs(0)["start"]);
            Assert.Equal(false, Kwargs(0)["normalize"]);
            Assert.IsType<AudioData>(((IList<object?>)_calls[0].payload!["args"]!)[0]);
            Assert.Equal(TransportState.Playing, player.Transport.State);
            player.OnStop();
        }

        [Fact]
        public void PauseResumeStop_ReachTheWorker()
        {
            var player = new AudioPlayerComponent();
            player.Load(Stereo(), null);
            player.OnPlay();
            player.OnPause();
            player.OnPlay();
            player.OnStop();
            Assert.Equal(new[] { "/api/play", "/api/pause", "/api/resume", "/api/stop" }, _calls.Select(c => c.path).ToArray());
        }

        [Fact]
        public void OneButton_PlaysPausesAndResumes()
        {
            var player = new AudioPlayerComponent();
            player.Load(Stereo(), null);
            player.OnPlayPause();
            Assert.Equal(TransportState.Playing, player.Transport.State);
            player.OnPlayPause();
            Assert.Equal(TransportState.Paused, player.Transport.State);
            player.OnPlayPause();
            Assert.Equal(new[] { "/api/play", "/api/pause", "/api/resume" }, _calls.Select(c => c.path).ToArray());
        }

        [Fact]
        public void TheChannels_GrowWithTheSignal_AtOneLaneHeight()
        {
            var player = new AudioPlayerComponent();
            Assert.Equal(0, player.WaveHeight);
            player.Load(Stereo(), null);
            Assert.Equal(PlayerLayout.WaveHeight(2, player.LaneHeight), player.WaveHeight);
        }

        [Fact]
        public void AFile_PlaysWholeWithoutPauseOrSeek()
        {
            var player = new AudioPlayerComponent();
            player.Load(Stereo(), "/sounds/viola.wav");
            Assert.False(player.CanSeek);
            player.OnSeek(1.0);
            player.OnPlay();
            player.OnPause();
            Assert.Equal(new[] { "/api/play", "/api/stop" }, _calls.Select(c => c.path).ToArray());
            Assert.Equal("/sounds/viola.wav", ((IList<object?>)_calls[0].payload!["args"]!)[0]);
        }

        [Fact]
        public void AnAmbisonicSignal_IsShownButDoesNotPlay()
        {
            var player = new AudioPlayerComponent();
            var audio = new AudioData { SamplingRate = 100, Format = "ambisonic", Shape = { 10, 4 } };
            audio.Samples.Add(new double[40]);
            player.Load(audio, null);
            Assert.False(player.CanPlay);
            Assert.Contains("4-channel ambisonic signal cannot be played", player.Problem);
            player.OnPlay();
            Assert.Empty(_calls);
        }

        [Fact]
        public void AnotherSignal_StopsWhatPlays()
        {
            var player = new AudioPlayerComponent();
            player.Load(Stereo(), null);
            player.OnPlay();
            player.Load(Stereo(1.0), null);
            Assert.Equal(new[] { "/api/play", "/api/stop" }, _calls.Select(c => c.path).ToArray());
            Assert.Equal(1.0, player.Transport.Duration, 9);
        }

        [Fact]
        public void APlaybackError_StopsTheTransportAndShowsInTheHeader()
        {
            var player = new AudioPlayerComponent();
            player.Load(Stereo(), null);
            _fail = true;
            player.OnPlay();
            Assert.Equal(TransportState.Stopped, player.Transport.State);
            Assert.Equal("The sound could not be played: PortAudio error: no device", player.Problem);
        }

        [Fact]
        public void GainAndSize_AreSavedWithTheDocument()
        {
            var player = new AudioPlayerComponent();
            player.OnGain(-12);
            (player.BodyWidth, player.LaneHeight) = (300, 36);
            var chunk = new GH_IO.Serialization.GH_LooseChunk("component");
            player.Write(chunk);
            var read = new GH_IO.Serialization.GH_LooseChunk("component");
            read.Deserialize_Binary(chunk.Serialize_Binary());
            var back = new AudioPlayerComponent();
            back.Read(read);
            Assert.Equal(-12.0, back.GainDb);
            Assert.Equal(300f, back.BodyWidth);
            Assert.Equal(36f, back.LaneHeight);
        }
    }

    [CollectionDefinition("worker override", DisableParallelization = true)]
    public class WorkerOverrideCollection { }
}
