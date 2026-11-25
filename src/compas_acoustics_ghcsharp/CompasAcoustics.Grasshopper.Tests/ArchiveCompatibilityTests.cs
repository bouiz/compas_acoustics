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
            Assert.Equal(new[] { "scene", "ism_order", "ray_tracing", "n_rays", "duration", "seed" },
                mono.Params.Input.Select(p => p.Name));
            Assert.Equal(new[] { "irs" }, mono.Params.Output.Select(p => p.Name));
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

        /// <summary>A Simulate Mono as saved before absorb_air and energy_thres were removed: both between n_rays and duration.</summary>
        private static SimulateMonoComponent MonoWithAirAndEnergyInputs(bool air, double energy)
        {
            var old = new SimulateMonoComponent();
            var port = new Param_Boolean { Name = "absorb_air", NickName = "absorb_air", Optional = true };
            port.SetPersistentData(new GH_Boolean(air));
            old.Params.RegisterInputParam(port, old.Params.Input.FindIndex(p => p.Name == "duration"));
            var threshold = new Param_Number { Name = "energy_thres", NickName = "energy_thres", Optional = true };
            threshold.SetPersistentData(new GH_Number(energy));
            old.Params.RegisterInputParam(threshold, old.Params.Input.FindIndex(p => p.Name == "duration"));
            ((Param_Number)old.Params.Input.Single(p => p.Name == "duration")).SetPersistentData(new GH_Number(2.5));
            var seed = (Param_Integer)old.Params.Input.Single(p => p.Name == "seed");
            seed.PersistentData.Clear();
            seed.SetPersistentData(new GH_Integer(5));
            return old;
        }

        private static object? First(GH_Component component, string name) => ControlState.Persistent(component.Params.Input.Single(p => p.Name == name));

        [Theory]
        [InlineData(false, 1e-7, true, null)]
        [InlineData(true, 1e-7, false, null)]
        [InlineData(true, 1e-6, false, 1e-6)]
        public void ASimulateMonoSavedWithItsAirAndEnergyInputsShown_ReadsTheLaterInputsInPlace_AndSaysWhatTheyChanged(bool air, double energy,
            bool withoutAir, double? threshold)
        {
            var fresh = new SimulateMonoComponent();
            var (archive, read) = SaveThenRead(MonoWithAirAndEnergyInputs(air, energy), fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.DoesNotContain("absorb_air", fresh.Params.Input.Select(p => p.Name));
            Assert.DoesNotContain("energy_thres", fresh.Params.Input.Select(p => p.Name));
            Assert.Equal(2.5, First(fresh, "duration"));
            Assert.Equal(5, First(fresh, "seed"));
            Assert.Equal(withoutAir, fresh.SavedWithoutAir);
            Assert.Equal(threshold, fresh.SavedEnergyThreshold);
        }

        /// <summary>A simulation as saved before Normalize was removed: a boolean input, off, in front of <paramref name="before"/> (at the end for null), and a seed of 7.</summary>
        private static T WithNormalizeOff<T>(T old, string? before) where T : GH_Component
        {
            var port = new Param_Boolean { Name = "normalize", NickName = "normalize", Optional = true };
            port.SetPersistentData(new GH_Boolean(false));
            if (before == null)
                old.Params.RegisterInputParam(port);
            else
                old.Params.RegisterInputParam(port, old.Params.Input.FindIndex(p => p.Name == before));
            var seed = (Param_Integer)old.Params.Input.Single(p => p.Name == "seed");
            seed.PersistentData.Clear();
            seed.SetPersistentData(new GH_Integer(7));
            return old;
        }

        [Fact]
        public void ASimulateMonoSavedWithNormalizeOff_ReadsWithoutIt_AndKeepsTheSeed()
        {
            var fresh = new SimulateMonoComponent();
            var (archive, read) = SaveThenRead(WithNormalizeOff(new SimulateMonoComponent(), null), fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.DoesNotContain("normalize", fresh.Params.Input.Select(p => p.Name));
            Assert.Equal(7, First(fresh, "seed"));
            Assert.Empty(fresh.RuntimeMessages(GH_RuntimeMessageLevel.Remark));
            Assert.False(fresh.SavedWithoutAir);
            Assert.Null(fresh.SavedEnergyThreshold);
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void ASimulateBinauralSavedWithNormalizeOff_ReadsWithoutIt_AndKeepsTheInputsAfterIt(bool hidden)
        {
            var old = WithNormalizeOff(new SimulateBinauralComponent(), "hrtf");
            ((Param_String)old.Params.Input.Single(p => p.Name == "hrtf")).SetPersistentData(new GH_String("/sets/my_head.sofa"));
            if (hidden)
                old.ShowControlPorts(false);
            var fresh = new SimulateBinauralComponent();
            var (archive, read) = SaveThenRead(old, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.DoesNotContain("normalize", fresh.Params.Input.Select(p => p.Name));
            Assert.Equal(7, First(fresh, "seed"));
            Assert.Equal("/sets/my_head.sofa", ControlState.Persistent(fresh.ControlPort("hrtf")!));
            Assert.Empty(fresh.RuntimeMessages(GH_RuntimeMessageLevel.Remark));
        }

        [Fact]
        public void ASimulateBinauralSavedWithNormalizeHidden_ReadsWithoutIt_AndKeepsTheHrtf()
        {
            var old = new SimulateBinauralComponent();
            ControlState.Choose(old, old.ControlPort("hrtf")!, "/sets/my_head.sofa");
            old.ShowControlPorts(false);
            old.CreateAttributes();
            foreach (var p in old.Params)
                if (p.Attributes == null)
                    p.CreateAttributes();
            var archive = new GH_Archive();
            var node = archive.CreateTopLevelNode("component");
            Assert.True(old.Write(node));
            node.RemoveItem("control_inputs_hidden");
            node.SetString("control_inputs_hidden", string.Join("\n", old.ControlPorts).Replace("hrtf", "normalize\nhrtf"));
            node.SetString("control_value_normalize", "False");
            var back = new GH_Archive();
            back.Deserialize_Binary(archive.Serialize_Binary());
            var fresh = new SimulateBinauralComponent();
            fresh.CreateAttributes();
            Assert.True(fresh.Read(back.GetRootNode.FindChunk("component")));
            Assert.Empty(Errors(back));
            Assert.DoesNotContain("normalize", fresh.Params.Input.Select(p => p.Name));
            Assert.Equal("/sets/my_head.sofa", ControlState.Persistent(fresh.ControlPort("hrtf")!));
            Assert.Empty(fresh.RuntimeMessages(GH_RuntimeMessageLevel.Remark));
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void ASimulateAmbisonicsSavedWithNormalizeOff_ReadsWithoutIt_AndKeepsTheInputsBeforeIt(bool hidden)
        {
            var old = WithNormalizeOff(new SimulateAmbisonicsComponent(), "run");
            if (hidden)
                old.ShowControlPorts(false);
            var fresh = new SimulateAmbisonicsComponent();
            var (archive, read) = SaveThenRead(old, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.DoesNotContain("normalize", fresh.Params.Input.Select(p => p.Name));
            Assert.Equal(7, First(fresh, "seed"));
            Assert.Empty(fresh.RuntimeMessages(GH_RuntimeMessageLevel.Remark));
        }

        [Fact]
        public void ASimulateMonoSavedWithItsTogglesHidden_ShowsRayTracingWithItsValue_AndSaysAirWasOff()
        {
            var saved = new SimulateMonoComponent();
            saved.CreateAttributes();
            foreach (var p in saved.Params)
                if (p.Attributes == null)
                    p.CreateAttributes();
            saved.ShowControlPorts(false);
            // as the former body saved it: ray_tracing and absorb_air hidden with the body's values, not saved as ports
            saved.Params.UnregisterInputParameter(saved.Params.Input.Single(p => p.Name == "ray_tracing"), true);
            var threshold = new Param_Number { Name = "energy_thres", NickName = "energy_thres", Optional = true };
            threshold.SetPersistentData(new GH_Number(1e-7));
            saved.Params.RegisterInputParam(threshold, saved.Params.Input.FindIndex(p => p.Name == "duration"));
            var archive = new GH_Archive();
            var node = archive.CreateTopLevelNode("component");
            Assert.True(saved.Write(node));
            node.RemoveItem("control_inputs_hidden");
            node.SetString("control_inputs_hidden", "ism_order\nray_tracing\nabsorb_air\nnormalize");
            node.SetString("control_value_ray_tracing", "False");
            node.SetString("control_value_absorb_air", "False");
            var back = new GH_Archive();
            back.Deserialize_Binary(archive.Serialize_Binary());
            var fresh = new SimulateMonoComponent();
            fresh.CreateAttributes();
            Assert.True(fresh.Read(back.GetRootNode.FindChunk("component")));
            Assert.Empty(Errors(back));
            Assert.Equal(new[] { "scene", "ray_tracing", "n_rays", "duration", "seed" }, fresh.Params.Input.Select(p => p.Name));
            Assert.Equal(false, ControlState.Persistent(fresh.Params.Input.Single(p => p.Name == "ray_tracing")));
            Assert.Equal(new[] { "ism_order" }, fresh.ControlPorts.Where(n => fresh.Params.Input.All(p => p.Name != n)));
            Assert.True(fresh.SavedWithoutAir);
        }

        [Fact]
        public void AComponentSavedByAnEarlierRelease_TakesTheCurrentDescriptions()
        {
            var old = new SimulateMonoComponent();
            old.ShowControlPorts(false);
            old.Description = "old component text";
            foreach (var p in old.Params.Input.Concat(old.Params.Output))
                p.Description = "old port text";
            var fresh = new SimulateMonoComponent();
            var (archive, read) = SaveThenRead(old, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            var spec = ComponentSpec.Load("CA_Simulate_Mono");
            Assert.Equal(spec.Description, fresh.Description);
            foreach (var p in fresh.Params.Input.Concat(fresh.Params.Output))
                Assert.Equal(spec.Inputs.Concat(spec.Outputs).Single(s => s.Name == p.Name).Description, p.Description);
            fresh.ShowControlPorts(true);
            Assert.Equal(spec.Inputs.Single(s => s.Name == "ism_order").Description, fresh.Params.Input.Single(p => p.Name == "ism_order").Description);
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
