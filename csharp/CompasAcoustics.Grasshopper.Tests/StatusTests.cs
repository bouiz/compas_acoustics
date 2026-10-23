using System.Linq;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class StatusTests
    {
        [Fact]
        public void Status_TrimsTo40CharactersAndKeepsTheFullText()
        {
            Assert.Equal(("done, 2 IRs", null), MessageText.Status("done, 2 IRs"));
            var full = "44100Hz, hybrid, tail: reference, order 1, seed 123, hrtf kemar";
            var (shown, kept) = MessageText.Status(full);
            Assert.Equal(40, shown.Length);
            Assert.EndsWith("...", shown);
            Assert.StartsWith(shown[..^3], full);
            Assert.Equal(full, kept);
            Assert.Equal(("short", "a longer explanation"), MessageText.Status("short", "a longer explanation"));
        }

        [Fact]
        public void SceneMessage_Format()
        {
            Assert.Equal("6 objects, 1 source, 2 receivers, 48 kHz, V 105 m3", MessageText.Scene(6, 1, 2, 48000, 105.2));
            Assert.Equal("1 object, 2 sources, 1 receiver, 44.1 kHz, V 0 m3", MessageText.Scene(1, 2, 1, 44100, 0));
        }

        [Fact]
        public void WorkerLine_SaysTheStateAndHowLongItRuns()
        {
            Assert.Equal("not configured", MessageText.WorkerLine(WorkerState.NotConfigured, null));
            Assert.Equal("starting", MessageText.WorkerLine(WorkerState.Starting, null));
            Assert.Equal("stopped", MessageText.WorkerLine(WorkerState.Stopped, null));
            Assert.Equal("running 12 min", MessageText.WorkerLine(WorkerState.Running, 12.7 * 60));
            Assert.Equal("running 0 min", MessageText.WorkerLine(WorkerState.Running, 5));
        }

        [Fact]
        public void Specs_HaveAWorkerComponent()
        {
            var spec = ComponentSpec.Load("CA_Worker");
            Assert.Equal("Worker", spec.Name);
            Assert.Equal("06 Util", spec.Subcategory);
            Assert.Empty(spec.Inputs);
            Assert.Equal(new[] { "python", "version", "running" }, spec.Outputs.Select(p => p.Name));
            Assert.Equal(22, ComponentSpec.Names().Count());
        }
    }
}
