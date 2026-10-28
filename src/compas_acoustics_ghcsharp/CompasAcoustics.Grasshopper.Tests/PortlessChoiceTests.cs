using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>Choices without an input: Auralise's level on its body, Analyse IR's time scale on its time output's menu.</summary>
    public class PortlessChoiceTests
    {
        private static string[] Names(System.Collections.Generic.IEnumerable<PortSpec> ports) => ports.Select(p => p.Name).ToArray();

        private static GH_LooseChunk Saved(GH_Component component)
        {
            var chunk = new GH_LooseChunk("component");
            Assert.True(component.Write(chunk));
            var read = new GH_LooseChunk("component");
            read.Deserialize_Binary(chunk.Serialize_Binary());
            return read;
        }

        [Fact]
        public void TheRemovedPorts_AreGoneFromTheSpecs()
        {
            Assert.Equal(new[] { "ir" }, Names(ComponentSpec.Load("CA_Analyse_IR").Inputs));
            Assert.DoesNotContain("level", Names(ComponentSpec.Load("CA_Auralise").Inputs));
            Assert.Equal(new[] { "audio" }, Names(ComponentSpec.Load("CA_Audio_Player").Inputs));
            Assert.Equal(new[] { "particles" }, Names(ComponentSpec.Load("CA_Visualise_Billiard").Outputs));
            Assert.DoesNotContain("role", Names(ComponentSpec.Load("CA_Acoustic_Object").Inputs));
            Assert.DoesNotContain("select", Names(ComponentSpec.Load("CA_Deconstruct_IR").Inputs));
            Assert.True(new DeconstructParameterComponent().IsChoice("select"));
        }

        [Fact]
        public void AuraliseLevel_IsAChoiceOfTheComponent_SavedWithIt()
        {
            var auralise = new AuraliseComponent();
            // none, relative, individual from left to right; relative the default
            Assert.Equal(new[] { "none", "relative", "individual" }, auralise.Controls.OfType<SwitchControl>().Single(c => c.Port == "level").Options);
            Assert.Equal("relative", auralise.Choice("level"));
            // level is the component's own; mix_sources an input its toggle sets
            Assert.Equal(new[] { "mix_sources" }, auralise.ControlPorts);
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
            Assert.Equal(new[] { "db", "linear" }, time.Options);
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
