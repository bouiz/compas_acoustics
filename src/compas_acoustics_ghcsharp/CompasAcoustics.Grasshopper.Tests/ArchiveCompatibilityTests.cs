using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>Saved definitions read back without errors; a late_envelope saved as a boolean reads as a mode.</summary>
    public class ArchiveCompatibilityTests
    {
        // the test runner reads these signatures before WinFormsResolver is registered: no plugin type in them
        private static (GH_Archive archive, bool read) SaveThenRead(GH_Component saved, GH_Component fresh)
        {
            // as Grasshopper does on the canvas: a component and its parameters write their chunks with their attributes
            foreach (var c in new[] { saved, fresh })
            {
                c.CreateAttributes();
                foreach (var p in c.Params)
                    if (p.Attributes == null)
                        p.CreateAttributes();
            }
            var archive = new GH_Archive();
            Assert.True(saved.Write(archive.CreateTopLevelNode("component")));
            var back = new GH_Archive();
            back.Deserialize_Binary(archive.Serialize_Binary());
            var read = fresh.Read(back.GetRootNode.FindChunk("component"));
            return (back, read);
        }

        private static string[] Errors(GH_Archive archive) =>
            (archive.Messages ?? new System.Collections.Generic.List<GH_Message>()).Where(m => m.Type != GH_Message_Type.info).Select(m => m.Message).ToArray();

        private static SimulateAmbisonicsComponent WithALateEnvelopeBoolean(bool on)
        {
            // a boolean input in late_envelope's place
            var old = new SimulateAmbisonicsComponent();
            var index = old.Params.Input.FindIndex(p => p.Name == "late_envelope");
            old.Params.UnregisterInputParameter(old.Params.Input[index]);
            var toggle = new Param_Boolean { Name = "late_envelope", NickName = "late_envelope", Optional = true };
            toggle.PersistentData.Append(new GH_Boolean(on));
            old.Params.RegisterInputParam(toggle, index);
            return old;
        }

        [Theory]
        [InlineData(true, "directional")]
        [InlineData(false, false)]
        public void ASimulateAmbisonicsSavedWithALateEnvelopeBoolean_KeepsItsChoice(bool on, object expected)
        {
            var fresh = new SimulateAmbisonicsComponent();
            var (archive, read) = SaveThenRead(WithALateEnvelopeBoolean(on), fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            var value = ControlState.PersistentAll(fresh.ControlPort("late_envelope")!).Single();
            Assert.Contains(value, HybridRun.LateEnvelopes);
            Assert.Equal(expected, HybridRun.LateEnvelope(value));
        }

        [Theory]
        [InlineData(true, "directional")]
        [InlineData(false, false)]
        public void ASimulateAmbisonicsSavedWithALateEnvelopeBooleanHidden_KeepsItsChoice(bool on, object expected)
        {
            var old = WithALateEnvelopeBoolean(on);
            old.ShowControlPorts(false);
            var fresh = new SimulateAmbisonicsComponent();
            var (archive, read) = SaveThenRead(old, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            var value = fresh.HiddenValue("late_envelope");
            Assert.Contains(value, HybridRun.LateEnvelopes);
            Assert.Equal(expected, HybridRun.LateEnvelope(value));
        }

        [Fact]
        public void ASavedSimulateMono_IsFoundByItsGuid_AndReadsItsPortsByName()
        {
            var mono = new SimulateMonoComponent();
            Assert.Equal(new System.Guid("afb77a6d-dd01-5224-9ee0-70972dcbdbcf"), mono.ComponentGuid);
            Assert.Equal(("Simulate Mono", "SimMono"), (mono.Name, mono.NickName));
            Assert.Equal(new[] { "scene", "ism_order", "ray_tracing", "n_rays", "absorb_air", "energy_thres", "duration", "seed", "normalize" },
                mono.Params.Input.Select(p => p.Name));
            Assert.Equal(new[] { "mono_ir_graph" }, mono.Params.Output.Select(p => p.Name));
            var (archive, read) = SaveThenRead(new SimulateMonoComponent(), new SimulateMonoComponent());
            Assert.True(read);
            Assert.Empty(Errors(archive));
        }

        [Theory]
        [InlineData(20000, new int[0])]
        [InlineData(30000, new[] { 30000 })]
        public void ASimulateMonoSavedWithTheFormerRayDefault_TakesTheCountFromTheRoom(int saved, int[] expected)
        {
            var old = new SimulateMonoComponent();
            var port = (Param_Integer)old.Params.Input.Single(p => p.Name == "n_rays");
            port.PersistentData.Clear();
            port.SetPersistentData(new GH_Integer(saved));
            var fresh = new SimulateMonoComponent();
            var (archive, read) = SaveThenRead(old, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            var back = (Param_Integer)fresh.Params.Input.Single(p => p.Name == "n_rays");
            Assert.Equal(expected, back.PersistentData.AllData(true).OfType<GH_Integer>().Select(g => g.Value));
        }

        [Fact]
        public void AComponentSavedNow_ReadsWithoutErrors()
        {
            var (archive, read) = SaveThenRead(new SimulateBinauralComponent(), new SimulateBinauralComponent());
            Assert.True(read);
            Assert.Empty(Errors(archive));
        }
    }
}
