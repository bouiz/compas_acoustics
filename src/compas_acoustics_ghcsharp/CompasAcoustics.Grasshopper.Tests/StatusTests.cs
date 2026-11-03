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
        public void AcousticObjectMessage_ShowsItsSurface()
        {
            Assert.Equal("Total area: 12.3 m2", MessageText.Object(12.34));
            Assert.Equal("Total area: 0.6 m2", MessageText.Object(0.6));
            Assert.Equal("Total area: 245 m2", MessageText.Object(245.04));
        }

        [Fact]
        public void SceneMessage_ShowsTheVolumeAndSurface_TheTooltipWhatItHolds()
        {
            // one line each: the strip draws a line break
            var (shown, full) = MessageText.Scene(105.2, 142, 6, 1, 2, 48000);
            Assert.Equal("Volume: 105 m3\nTotal area: 142 m2", shown);
            Assert.Equal("Volume: 105 m3\nTotal area: 142 m2\n6 objects, 1 source, 2 receivers, 48 kHz", full);
            Assert.Equal("Volume: 1786 m3\nTotal area: 1279 m2", MessageText.Scene(1786.4, 1279.2, 1, 2, 1, 44100).shown);
            // the strip's limit is a line's: two short lines are not cut
            Assert.Equal(("Volume: 1786 m3\nTotal area: 1279 m2", null), MessageText.Status("Volume: 1786 m3\nTotal area: 1279 m2"));
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
            Assert.Equal("Util", spec.Subcategory);
            Assert.Empty(spec.Inputs);
            Assert.Equal(new[] { "python", "version", "running" }, spec.Outputs.Select(p => p.Name));
            Assert.Equal(23, ComponentSpec.Names().Count());
        }
    }
}
