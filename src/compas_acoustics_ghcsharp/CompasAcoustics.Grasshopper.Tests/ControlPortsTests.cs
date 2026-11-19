using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>"Show control inputs": the inputs of the body's switches and dropdowns, hidden unless asked for.</summary>
    public class ControlPortsTests
    {
        // the test runner reads these signatures before WinFormsResolver is registered: no plugin type in them
        private static string[] Inputs(GH_Component c) => c.Params.Input.Select(p => p.Name).ToArray();

        private static GH_Component RoundTrip(GH_Component component)
        {
            var chunk = new GH_LooseChunk("component");
            Assert.True(component.Write(chunk));
            var read = new GH_LooseChunk("component");
            read.Deserialize_Binary(chunk.Serialize_Binary());
            var back = (GH_Component)System.Activator.CreateInstance(component.GetType())!;
            Assert.True(back.Read(read));
            return back;
        }

        [Fact]
        public void TheControlPorts_AreTheInputsOfTheBodysControls_RunIncluded()
        {
            // run stays an input for loops and scripts that set it, hidden like the others
            Assert.Equal(new[] { "ism_order", "tail", "diffraction", "late_envelope", "normalize", "hrtf", "run" }, new SimulateBinauralComponent().ControlPorts);
            Assert.Equal(new[] { "ism_order", "tail", "diffraction", "late_envelope", "normalize", "run" }, new SimulateAmbisonicsComponent().ControlPorts);
            Assert.Equal(new[] { "time", "speed", "play" }, new VisualiseBilliardComponent().ControlPorts);
            Assert.Empty(new AcousticObjectComponent().ControlPorts);
            Assert.True(new AcousticObjectComponent().IsChoice("role"));
            Assert.Empty(new WorkerComponent().ControlPorts);
        }

        [Fact]
        public void TheSource_ChoosesItsDirectivityOnTheBody()
        {
            var source = new SourceComponent();
            var dropdown = source.Controls.OfType<DropdownControl>().Single();
            Assert.Equal("directivity", dropdown.Port);
            Assert.Equal(new[] { "omnidirectional", "cardioid", "supercardioid", "hypercardioid", "bidirectional" }, dropdown.Options);
            // no input: a choice of the component; the sound file's button sets an input
            Assert.Equal(new[] { "sound_file" }, source.ControlPorts);
            Assert.True(source.IsChoice("directivity"));
            Assert.DoesNotContain("directivity", ComponentSpec.Load("CA_Source").Inputs.Select(p => p.Name));
        }

        [Fact]
        public void TheSource_TakesItsLevelAsAnOptionalListAfterTheSoundFile()
        {
            var inputs = ComponentSpec.Load("CA_Source").Inputs;
            Assert.Equal(new[] { "position", "name", "direction", "sound_file", "spl" }, inputs.Select(p => p.Name));
            var spl = inputs.Last();
            Assert.True(spl.Optional);
            Assert.Equal(global::Grasshopper.Kernel.GH_ParamAccess.list, spl.Access);
        }

        [Fact]
        public void APreviewOffByDefault_IsOffInADocumentWithoutTheKey_AndKeepsASavedValue()
        {
            // without the key: Grasshopper's own Hidden (false) only
            var old = new GH_LooseChunk("component");
            new AcousticObjectComponent().Write(old);
            old.RemoveItem(ComponentBase.PreviewKey);
            var read = new GH_LooseChunk("component");
            read.Deserialize_Binary(old.Serialize_Binary());
            var back = new AcousticObjectComponent();
            Assert.True(back.Read(read));
            Assert.True(back.Hidden);
            // turned on and saved: on
            var shown = new AcousticObjectComponent { Hidden = false };
            var chunk = new GH_LooseChunk("component");
            shown.Write(chunk);
            var again = new GH_LooseChunk("component");
            again.Deserialize_Binary(chunk.Serialize_Binary());
            var on = new AcousticObjectComponent();
            Assert.True(on.Read(again));
            Assert.False(on.Hidden);
            // a component without the default keeps Grasshopper's
            var source = new GH_LooseChunk("component");
            new SourceComponent().Write(source);
            source.RemoveItem(ComponentBase.PreviewKey);
            var sourceRead = new GH_LooseChunk("component");
            sourceRead.Deserialize_Binary(source.Serialize_Binary());
            var src = new SourceComponent();
            Assert.True(src.Read(sourceRead));
            Assert.False(src.Hidden);
        }

        [Fact]
        public void HidingThem_RemovesThePorts_AndShowingPutsThemBackInTheirPlace()
        {
            var hybrid = new SimulateAmbisonicsComponent();
            var all = Inputs(hybrid);
            hybrid.ShowControlPorts(false);
            Assert.False(hybrid.ControlPortsShown);
            Assert.Equal(all.Where(n => !hybrid.ControlPorts.Contains(n)), Inputs(hybrid));
            Assert.Equal(new[] { "scene", "reference", "seed" }, Inputs(hybrid));
            Assert.NotNull(hybrid.ControlPort("run"));
            Assert.NotNull(hybrid.ControlPort("tail"));
            hybrid.ShowControlPorts(true);
            Assert.Equal(all, Inputs(hybrid));
        }

        [Fact]
        public void AHiddenPort_KeepsTheValueTheBodyChooses()
        {
            var hybrid = new SimulateAmbisonicsComponent();
            hybrid.ShowControlPorts(false);
            ControlState.Choose(hybrid, hybrid.ControlPort("tail")!, "eyring");
            Assert.Equal("eyring", hybrid.HiddenValue("tail"));
            Assert.Null(hybrid.HiddenValue("diffraction"));
            Assert.Null(hybrid.HiddenValue("scene"));
        }

        [Fact]
        public void AWiredPort_StaysWhenTheOthersHide()
        {
            var hybrid = new SimulateAmbisonicsComponent();
            hybrid.ControlPort("tail")!.AddSource(new Param_String());
            hybrid.ShowControlPorts(false);
            Assert.Contains("tail", Inputs(hybrid));
            Assert.DoesNotContain("diffraction", Inputs(hybrid));
        }

        [Fact]
        public void TheHiddenPortsAndTheirValues_AreSavedWithTheDocument()
        {
            var hybrid = new SimulateAmbisonicsComponent();
            hybrid.ShowControlPorts(false);
            ControlState.Choose(hybrid, hybrid.ControlPort("diffraction")!, "udfa");
            var back = (SimulateHybridComponent)RoundTrip(hybrid);
            Assert.False(back.ControlPortsShown);
            Assert.Equal(Inputs(hybrid), Inputs(back));
            Assert.Equal("udfa", back.HiddenValue("diffraction"));
        }

        [Fact]
        public void AComponentSavedWithItsPortsShown_ReadsThemShown()
        {
            var hybrid = new SimulateAmbisonicsComponent();
            ControlState.Choose(hybrid, hybrid.ControlPort("tail")!, "sabine");
            var back = (SimulateHybridComponent)RoundTrip(hybrid);
            Assert.True(back.ControlPortsShown);
            Assert.Equal(Inputs(hybrid), Inputs(back));
            Assert.Equal(("sabine", false), ControlState.Of(back.ControlPort("tail")!, HybridRun.Tails));
        }
    }
}
