using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using CompasAcoustics.Pb;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>The Tail of Simulate Binaural and Simulate Ambisonics: always sent, a reference needs Tail Reference, an old Auto reads by the wires.</summary>
    [Collection("worker override")]
    public class HybridTailTests : IDisposable
    {
        private const string Needs = "A reference is connected: set Tail to Reference, or disconnect it.";

        private readonly WorkerCalls _calls = new();
        private readonly string _home = Directory.CreateTempSubdirectory("compas_acoustics_home_").FullName;

        public HybridTailTests()
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
            _calls.Add(path, body);
            var response = new ImpulseResponseData
            {
                Source = new SourceData { Guid = "s0", Name = "s0" }, Receiver = new ReceiverData { Guid = "r0", Name = "r0" },
                Shape = { 4, 2 }, SamplingRate = 48000, Samples = { new double[8] },
            };
            var responses = new Dictionary<string, object?> { ["r0_s0"] = response };
            return FakeHandler.Result(new Dictionary<string, object?> { ["responses"] = new Dictionary<string, object?> { ["binaural"] = responses, ["ambisonics"] = responses } });
        }

        private static void WaitForTheRun(ComponentBase component)
        {
            var deadline = DateTime.UtcNow.AddSeconds(5);
            while (component.Runs.AnyRunning && DateTime.UtcNow < deadline)
                System.Threading.Thread.Sleep(10);
            Assert.False(component.Runs.AnyRunning);
        }

        private static ImpulseResponseGoo Reference() => new(new ImpulseResponseData
        {
            Source = new SourceData { Guid = "s0" }, Receiver = new ReceiverData { Guid = "r0" },
            Shape = { 4, 1 }, SamplingRate = 48000, Samples = { new double[4] },
        });

        private static FakeAccess Access(string? tail, bool referenced, bool run = false, string? lateEnvelope = null)
        {
            var access = new FakeAccess();
            access.Items["scene"] = new SceneGoo(new SceneData
            {
                Guid = "scene", SamplingRate = 48000,
                Sources = { new SourceData { Guid = "s0", Name = "s0" } }, Receivers = { new ReceiverData { Guid = "r0", Name = "r0" } },
            });
            if (tail != null)
                access.Items["tail"] = tail;
            if (lateEnvelope != null)
                access.Items["late_envelope"] = lateEnvelope;
            if (referenced)
                access.Lists["reference"] = new[] { Reference() };
            if (run)
                access.Items["run"] = true;
            return access;
        }

        private static void Solve(ComponentBase component, IGH_DataAccess access) =>
            typeof(GH_Component).GetMethod("SolveInstance", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.Invoke(component, new object?[] { access });

        private IDictionary<string, object?> Settings()
        {
            var call = _calls.AwaitSingle();
            Assert.Equal("/api/simulate", call.path);
            return (IDictionary<string, object?>)((IList<object?>)call.payload!["args"]!)[2]!;
        }

        private static GH_Component RoundTrip(GH_Component component)
        {
            var chunk = new GH_LooseChunk("component");
            Assert.True(component.Write(chunk));
            var read = new GH_LooseChunk("component");
            read.Deserialize_Binary(chunk.Serialize_Binary());
            var back = (GH_Component)Activator.CreateInstance(component.GetType())!;
            Assert.True(back.Read(read));
            return back;
        }

        [Theory]
        [InlineData(null)]
        [InlineData("ray_tracing")]
        [InlineData("sabine")]
        public void AReferenceWithAnotherTail_IsAnError_AndCallsNoWorker(string? tail)
        {
            var component = new SimulateBinauralComponent();
            Solve(component, Access(tail, referenced: true, run: true));
            Assert.Equal(new[] { Needs }, component.RuntimeMessages(GH_RuntimeMessageLevel.Error).ToArray());
            Assert.Empty(_calls);
        }

        [Fact]
        public void TheReferenceError_SetsAStatus_AndAnEmptyOutput()
        {
            var component = new SimulateBinauralComponent();
            var access = Access("eyring", referenced: true, run: true);
            access.OutputLists["irs"] = new List<object?> { "stale" };
            Solve(component, access);
            Assert.Equal(new[] { Needs }, component.RuntimeMessages(GH_RuntimeMessageLevel.Error).ToArray());
            Assert.Equal("set Tail to Reference", component.Message);
            Assert.Empty(access.OutputLists["irs"]);
            Assert.Empty(_calls);
        }

        private const string Stale = "The reference is from an earlier model: rerun Simulate Mono first.";

        private static FakeAccess StaleAccess(bool run)
        {
            var access = Access("reference", referenced: true, run: run);
            var earlier = ((SceneGoo)access.Items["scene"]).Value.Clone();
            earlier.Receivers[0].Name = "moved";
            var reference = Reference();
            reference.SimulatedFor = new SceneGoo(earlier);
            access.Lists["reference"] = new[] { reference };
            return access;
        }

        [Fact]
        public void AReferenceFromAnEarlierModel_Warns_AndCallsNoWorker()
        {
            var component = new SimulateBinauralComponent();
            Solve(component, StaleAccess(run: false));
            Assert.Equal(new[] { Stale }, component.RuntimeMessages(GH_RuntimeMessageLevel.Warning).ToArray());
            Assert.Empty(component.RuntimeMessages(GH_RuntimeMessageLevel.Error));
            Assert.Empty(_calls);
        }

        [Fact]
        public void RunWithAReferenceFromAnEarlierModel_IsAnError_AndCallsNoWorker()
        {
            var component = new SimulateAmbisonicsComponent();
            Solve(component, StaleAccess(run: true));
            Assert.Equal(new[] { Stale }, component.RuntimeMessages(GH_RuntimeMessageLevel.Error).ToArray());
            Assert.Equal("rerun Simulate Mono first", component.Message);
            Assert.Empty(_calls);
        }

        private SimulateBinauralComponent Ran()
        {
            var component = new SimulateBinauralComponent();
            Solve(component, Access("reference", referenced: true, run: true));
            _calls.AwaitSingle();
            WaitForTheRun(component);
            var access = Access("reference", referenced: true);
            Solve(component, access);
            Assert.Equal("simulated 1 response", component.Message);
            Assert.Single(access.OutputLists["irs"]);
            return component;
        }

        [Fact]
        public void AChangedLateEnvelope_AfterARun_SaysTheSettingsChanged_AndKeepsTheResponses()
        {
            var component = Ran();
            var access = Access("reference", referenced: true, lateEnvelope: "directional");
            Solve(component, access);
            Assert.Equal("settings changed,\nrerun the simulation", component.Message);
            Assert.Single(access.OutputLists["irs"]);
        }

        [Fact]
        public void AReferenceRerunWithOtherSamples_AfterARun_SaysTheSettingsChanged()
        {
            var component = Ran();
            var access = Access("reference", referenced: true);
            var rerun = Reference();
            rerun.Value.Samples[1] = 0.5;
            access.Lists["reference"] = new[] { rerun };
            Solve(component, access);
            Assert.Equal("settings changed,\nrerun the simulation", component.Message);
        }

        [Fact]
        public void TailReferenceWithAReference_RunsWithoutAMessage()
        {
            var component = new SimulateAmbisonicsComponent();
            Solve(component, Access("reference", referenced: true, run: true));
            Assert.Equal("reference", Settings()["tail"]);
            Assert.Empty(component.RuntimeMessages(GH_RuntimeMessageLevel.Error));
            Assert.Empty(component.RuntimeMessages(GH_RuntimeMessageLevel.Warning));
        }

        [Fact]
        public void TailReferenceWithoutAReference_Warns_AndCallsNoWorker()
        {
            var component = new SimulateBinauralComponent();
            Solve(component, Access("reference", referenced: false, run: true));
            var warning = Assert.Single(component.RuntimeMessages(GH_RuntimeMessageLevel.Warning));
            Assert.StartsWith("Tail Reference needs a reference: connect the irs of Simulate Mono to 'reference', or choose another tail.", warning);
            Assert.Empty(component.RuntimeMessages(GH_RuntimeMessageLevel.Error));
            Assert.Empty(_calls);
        }

        [Theory]
        [InlineData("eyring", "eyring")]
        [InlineData("sabine", "sabine")]
        [InlineData("auto", "ray_tracing")]
        public void WithoutAReference_TheTailIsSentExplicitly_AnOldAutoAsRayTracing(string? saved, string sent)
        {
            var component = new SimulateBinauralComponent();
            Solve(component, Access(saved, referenced: false, run: true));
            Assert.Equal(sent, Settings()["tail"]);
            Assert.Empty(component.RuntimeMessages(GH_RuntimeMessageLevel.Error));
        }

        [Theory]
        [InlineData(true, "reference")]
        [InlineData(false, "ray_tracing")]
        public void AnOldAutoSavedHidden_ReadsByTheWiresAtSolve(bool referenced, string sent)
        {
            var old = new SimulateBinauralComponent();
            old.ShowControlPorts(false);
            ControlState.Assign(old.ControlPort("tail")!, new[] { "auto" });
            var back = (SimulateBinauralComponent)RoundTrip(old);
            Assert.Equal("auto", back.HiddenValue("tail"));
            ControlState.Press(back, "run");
            Solve(back, Access(null, referenced));
            Assert.Equal(sent, Settings()["tail"]);
            Assert.Empty(back.RuntimeMessages(GH_RuntimeMessageLevel.Error));
        }

        [Theory]
        [InlineData(true, "reference")]
        [InlineData(false, "ray_tracing")]
        public void AnOldAutoSavedAsAShownPort_ReadsByTheWiresAtSolve(bool referenced, string sent)
        {
            var old = new SimulateAmbisonicsComponent();
            ControlState.Assign(old.ControlPort("tail")!, new[] { "auto" });
            var back = (SimulateAmbisonicsComponent)RoundTrip(old);
            Assert.Equal("auto", ControlState.PersistentAll(back.ControlPort("tail")!).Single());
            Solve(back, Access("auto", referenced, run: true));
            Assert.Equal(sent, Settings()["tail"]);
            Assert.Empty(back.RuntimeMessages(GH_RuntimeMessageLevel.Error));
        }

        [Theory]
        [InlineData("ray_tracing")]
        [InlineData("reference")]
        [InlineData("eyring")]
        [InlineData("sabine")]
        public void ATailOfTheNewLayout_RoundTrips(string tail)
        {
            var hybrid = new SimulateBinauralComponent();
            ControlState.Choose(hybrid, hybrid.ControlPort("tail")!, tail);
            var back = RoundTrip(hybrid);
            Assert.Equal((tail, false), ControlState.Of(((SimulateBinauralComponent)back).ControlPort("tail")!, HybridRun.Tails));
            hybrid.ShowControlPorts(false);
            var hidden = (SimulateBinauralComponent)RoundTrip(hybrid);
            Assert.Equal(tail, hidden.HiddenValue("tail"));
        }

        [Fact]
        public void ANewComponent_SendsOmni()
        {
            var component = new SimulateBinauralComponent();
            Solve(component, Access(null, referenced: false, run: true));
            var settings = Settings();
            Assert.Equal("ray_tracing", settings["tail"]);
            Assert.Equal("omni", settings["late_envelope"]);
            Assert.Empty(component.RuntimeMessages(GH_RuntimeMessageLevel.Error));
        }

        [Theory]
        [InlineData("ray_tracing", false, "directional", "directional")]
        [InlineData("reference", true, "directional", "directional")]
        [InlineData("ray_tracing", false, "off", false)]
        [InlineData("ray_tracing", false, "auto", "omni")]
        public void TheLateEnvelopeChosen_IsSentExplicitly_AnOldAutoAsOmni(string tail, bool referenced, string chosen, object sent)
        {
            var component = new SimulateAmbisonicsComponent();
            Solve(component, Access(tail, referenced, run: true, lateEnvelope: chosen));
            Assert.Equal(sent, Settings()["late_envelope"]);
            Assert.True(component.ControlEnabled("late_envelope"));
            Assert.Empty(component.RuntimeMessages(GH_RuntimeMessageLevel.Error));
        }

        [Theory]
        [InlineData("eyring", "directional")]
        [InlineData("eyring", "omni")]
        [InlineData("sabine", "directional")]
        [InlineData("sabine", null)]
        public void AnEyringOrSabineTail_RestsTheLateEnvelope_AndSendsOff(string tail, string? chosen)
        {
            var component = new SimulateBinauralComponent();
            Assert.True(component.ControlEnabled("late_envelope"));
            Solve(component, Access(tail, referenced: false, run: true, lateEnvelope: chosen));
            Assert.False(component.ControlEnabled("late_envelope"));
            Assert.True(component.ControlEnabled("tail"));
            Assert.Equal(false, Settings()["late_envelope"]);
            Assert.Empty(component.RuntimeMessages(GH_RuntimeMessageLevel.Error));
        }

        [Fact]
        public void TheLateEnvelope_WorksAgainWhenTheTailChangesBack()
        {
            var component = new SimulateBinauralComponent();
            Solve(component, Access("eyring", referenced: false));
            Assert.False(component.ControlEnabled("late_envelope"));
            Solve(component, Access("ray_tracing", referenced: false));
            Assert.True(component.ControlEnabled("late_envelope"));
        }

        [Fact]
        public void ANewComponent_ShowsRayTracing()
        {
            var hybrid = new SimulateAmbisonicsComponent();
            Assert.Equal(("ray_tracing", false), ControlState.Of(hybrid.ControlPort("tail")!, hybrid.Controls.OfType<DropdownControl>().Single(c => c.Port == "tail").Options));
        }
    }
}
