using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>Choices without an input: Auralise's level on its body, Analyse IR's time scale on its time output's menu.</summary>
    public class PortlessChoiceTests
    {
        private static GH_LooseChunk Saved(GH_Component component)
        {
            var chunk = new GH_LooseChunk("component");
            Assert.True(component.Write(chunk));
            var read = new GH_LooseChunk("component");
            read.Deserialize_Binary(chunk.Serialize_Binary());
            return read;
        }

        [Fact]
        public void AuraliseLevel_IsAChoiceOfTheComponent_SavedWithIt()
        {
            var auralise = new AuraliseComponent();
            // a dropdown: relative, the default, first
            Assert.Equal(new[] { "relative", "individual", "none" }, auralise.Controls.OfType<DropdownControl>().Single(c => c.Port == "level").Options);
            Assert.Equal("relative", auralise.Choice("level"));
            // level is the component's own; the other dropdowns set inputs
            Assert.Equal(new[] { "source_idx", "receiver_idx", "sound" }, auralise.ControlPorts);
            auralise.SetChoice("level", "none");
            var back = new AuraliseComponent();
            Assert.True(back.Read(Saved(auralise)));
            Assert.Equal("none", back.Choice("level"));
        }

        [Fact]
        public void AnalyseIrTimeScale_IsAChoiceOfItsTimeOutput()
        {
            var analyse = new AnalyseIrComponent();
            Assert.Equal("db", analyse.Choice("time_scale"));
            var time = Assert.IsType<Param_TimeParameter>(analyse.Params.Output.Single(p => p.Name == "time"));
            Assert.Equal(new[] { "db", "linear" }, Param_TimeParameter.Options);
            analyse.SetChoice("time_scale", "linear");
            var back = new AnalyseIrComponent();
            Assert.True(back.Read(Saved(analyse)));
            Assert.Equal("linear", back.Choice("time_scale"));
        }

        [Fact]
        public void AChoiceOutsideItsOptions_IsRefused()
        {
            var auralise = new AuraliseComponent();
            auralise.SetChoice("level", "loud");
            Assert.Equal("relative", auralise.Choice("level"));
        }
    }
}
