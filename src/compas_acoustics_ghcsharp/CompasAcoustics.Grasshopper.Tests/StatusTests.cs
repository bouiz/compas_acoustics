using System.Linq;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class StatusTests
    {
        [Fact]
        public void Status_KeepsTheFullText()
        {
            Assert.Equal(("done, 2 IRs", null), MessageText.Status("done, 2 IRs"));
            var full = "44.1 kHz, binaural, tail reference, seed 1234567890, hrtf ku100";
            var (shown, kept) = MessageText.Status(full);
            Assert.StartsWith(shown[..^3], full);
            Assert.Equal(full, kept);
            Assert.Equal(("short", "a longer explanation"), MessageText.Status("short", "a longer explanation"));
        }

        [Fact]
        public void SimulatedMessage_CountsTheResponses()
        {
            Assert.Equal("simulated 1 response", MessageText.Simulated(1));
            Assert.Equal("simulated 4 responses", MessageText.Simulated(4));
        }

        [Fact]
        public void Specs_HaveAWorkerComponent()
        {
            var spec = ComponentSpec.Load("CA_Worker");
            Assert.Equal("Worker", spec.Name);
            Assert.Equal("Util", spec.Subcategory);
            Assert.Empty(spec.Inputs);
            Assert.Equal(new[] { "python", "version", "running" }, spec.Outputs.Select(p => p.Name));
            Assert.Equal(24, EmbeddedSpecs.Names().Count());
        }
    }
}
