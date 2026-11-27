using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using CompasAcoustics.Pb;
using Grasshopper.Kernel;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>Simulate Mono runs on Run, as Simulate Binaural and Simulate Ambisonics do, and keeps its responses for the same scene.</summary>
    [Collection("worker override")]
    public class MonoRunTests : IDisposable
    {
        private readonly List<string> _calls = new();
        private readonly string _home = Directory.CreateTempSubdirectory("compas_acoustics_home_").FullName;

        public MonoRunTests()
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
                return FakeHandler.Result(new Dictionary<string, object?> { ["compas_acoustics"] = Versions.Library, ["schema"] = Schema.Hash, ["hybrid"] = true });
            _calls.Add(path);
            var response = new ImpulseResponseData
            {
                Source = new SourceData { Guid = "a-s0", Name = "s0" }, Receiver = new ReceiverData { Guid = "r0", Name = "r0" },
                Shape = { 4 }, SamplingRate = 48000, Samples = { 1.0, 0.5, 0.25, 0.0 },
            };
            return FakeHandler.Result(new Dictionary<string, object?>
            {
                ["responses"] = new Dictionary<string, object?> { ["mono"] = new Dictionary<string, object?> { ["r0_a-s0"] = response } },
            });
        }

        private static FakeAccess Access(string guid = "a", bool run = false, int? ismOrder = null, string? sound = null)
        {
            var source = new SourceData { Guid = $"{guid}-s0", Name = "s0" };
            if (sound != null)
                source.SoundFile = sound;
            var access = new FakeAccess();
            access.Items["scene"] = new SceneGoo(new SceneData
            {
                Guid = guid, SamplingRate = 48000, Sources = { source }, Receivers = { new ReceiverData { Guid = "r0", Name = "r0" } },
            });
            if (ismOrder != null)
                access.Items["ism_order"] = ismOrder.Value;
            if (run)
                access.Items["run"] = true;
            return access;
        }

        private static void Solve(ComponentBase component, IGH_DataAccess access) =>
            typeof(GH_Component).GetMethod("SolveInstance", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.Invoke(component, new object?[] { access });

        private static void WaitForTheRun(ComponentBase component)
        {
            var deadline = DateTime.UtcNow.AddSeconds(5);
            while (component.Runs.AnyRunning && DateTime.UtcNow < deadline)
                System.Threading.Thread.Sleep(10);
            Assert.False(component.Runs.AnyRunning);
        }

        private SimulateMonoComponent Ran()
        {
            var mono = new SimulateMonoComponent();
            Solve(mono, Access(run: true));
            WaitForTheRun(mono);
            var access = Access();
            Solve(mono, access);
            Assert.Equal("simulated 1 response", (mono.FullMessage ?? mono.Message));
            Assert.Single(access.OutputLists["irs"]);
            return mono;
        }

        [Fact]
        public void BeforeRun_CallsNoWorker_AndAsksForRun()
        {
            var mono = new SimulateMonoComponent();
            var access = Access();
            Solve(mono, access);
            Solve(mono, Access(ismOrder: 2));
            Assert.False(mono.Runs.AnyRunning);
            Assert.Empty(_calls);
            Assert.Equal("press Run", (mono.FullMessage ?? mono.Message));
            Assert.Empty(access.OutputLists["irs"]);
        }

        [Fact]
        public void Run_RunsOnce()
        {
            Ran();
            Assert.Equal(new[] { "/api/simulate" }, _calls);
        }

        [Fact]
        public void TheSameSceneAfterARun_KeepsTheResponses_WithoutAWorkerCall()
        {
            var mono = Ran();
            var access = Access("b");
            Solve(mono, access);
            Assert.False(mono.Runs.AnyRunning);
            Assert.Single(_calls);
            Assert.Equal("simulated 1 response", (mono.FullMessage ?? mono.Message));
            Assert.Equal("b-s0", ((ImpulseResponseGoo)Assert.Single(access.OutputLists["irs"])!).SourceGuid);
        }

        [Fact]
        public void AChangedSetting_SaysTheResponsesAreStale_AndAsksForRun()
        {
            var mono = Ran();
            var access = Access(ismOrder: 5);
            Solve(mono, access);
            Assert.False(mono.Runs.AnyRunning);
            Assert.Single(_calls);
            Assert.Equal("settings changed,\nrerun the simulation", (mono.FullMessage ?? mono.Message));
            Assert.Single(mono.RuntimeMessages(GH_RuntimeMessageLevel.Warning));
            Assert.Single(access.OutputLists["irs"]);
        }

        [Fact]
        public void ASoundOnlyChange_KeepsTheResponses_WithTheNewSound()
        {
            var mono = Ran();
            var access = Access("c", sound: "speech");
            Solve(mono, access);
            Assert.Single(_calls);
            Assert.Equal("simulated 1 response", (mono.FullMessage ?? mono.Message));
            var ir = (ImpulseResponseGoo)Assert.Single(access.OutputLists["irs"])!;
            Assert.Equal("speech", ir.Value.Source.SoundFile);
            Assert.Equal("c-s0", ir.SourceGuid);
        }

        [Fact]
        public void AnotherScene_KeepsTheResponses_AndSaysTheModelChanged()
        {
            var mono = Ran();
            var access = Access();
            var scene = (SceneGoo)access.Items["scene"];
            scene.Value.Receivers[0].Name = "moved";
            Solve(mono, access);
            Assert.Single(_calls);
            Assert.Equal("model changed,\nrerun the simulation", (mono.FullMessage ?? mono.Message));
            Assert.Single(mono.RuntimeMessages(GH_RuntimeMessageLevel.Warning));
            var ir = (ImpulseResponseGoo)Assert.Single(access.OutputLists["irs"])!;
            Assert.False(SimulatedScene.Same(ir.SimulatedFor!, scene));
        }

        [Fact]
        public void TheResponsesOfARun_CarryTheSceneTheyWereSimulatedFor()
        {
            var mono = Ran();
            var access = Access("b");
            Solve(mono, access);
            var ir = (ImpulseResponseGoo)Assert.Single(access.OutputLists["irs"])!;
            Assert.True(SimulatedScene.Same(ir.SimulatedFor!, (SceneGoo)access.Items["scene"]));
        }
    }
}
