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

        /// <summary>A simulation as saved before it output its responses: its one output named <paramref name="output"/>, with an old description.</summary>
        private static T WithTheIrGraphOutput<T>(T old, string output) where T : GH_Component
        {
            var port = old.Params.Output.Single();
            port.Name = output;
            port.NickName = output;
            port.Description = "old port text";
            return old;
        }

        public static TheoryData<string, string> IrGraphOutputs => new()
        {
            { "CA_Simulate_Mono", "mono_ir_graph" },
            { "CA_Simulate_Binaural", "binaural_ir_graph" },
            { "CA_Simulate_Ambisonics", "ambisonic_ir_graph" },
        };

        private static GH_Component Simulation(string key) => key switch
        {
            "CA_Simulate_Mono" => new SimulateMonoComponent(),
            "CA_Simulate_Binaural" => new SimulateBinauralComponent(),
            _ => new SimulateAmbisonicsComponent(),
        };

        [Theory]
        [MemberData(nameof(IrGraphOutputs))]
        public void ASimulationSavedWithItsIrGraphOutput_ReadsItAsIrs(string key, string output)
        {
            var fresh = Simulation(key);
            var (archive, read) = SaveThenRead(WithTheIrGraphOutput(Simulation(key), output), fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            var port = Assert.Single(fresh.Params.Output);
            Assert.Equal(("irs", "irs"), (port.Name, port.NickName));
            Assert.Equal(ComponentSpec.Load(key).Outputs.Single().Description, port.Description);
            Assert.Equal(0, fresh.Params.IndexOfOutputParam("irs"));
        }

        [Fact]
        public void ASimulationSavedWithItsOutputRenamedOnTheCanvas_KeepsTheCanvasName()
        {
            var old = WithTheIrGraphOutput(new SimulateBinauralComponent(), "binaural_ir_graph");
            old.Params.Output[0].NickName = "left ears";
            var fresh = new SimulateBinauralComponent();
            var (_, read) = SaveThenRead(old, fresh);
            Assert.True(read);
            Assert.Equal(("irs", "left ears"), (fresh.Params.Output[0].Name, fresh.Params.Output[0].NickName));
        }

        [Fact]
        public void AnAnalyseSavedWithTheSingleIrInput_ReadsItAsIrs_AndTheNewInputsFollowHidden()
        {
            var old = new AnalyseIrComponent();
            foreach (var p in old.Params.Input.Skip(1).ToList())
                old.Params.UnregisterInputParameter(p);
            old.Params.Input[0].Name = "ir";
            old.Params.Input[0].NickName = "ir";
            old.Params.Input[0].Access = GH_ParamAccess.item;
            old.CreateAttributes();
            foreach (var p in old.Params)
                if (p.Attributes == null)
                    p.CreateAttributes();
            var archive = new GH_Archive();
            var node = archive.CreateTopLevelNode("component");
            Assert.True(old.Write(node));
            // the former Analyse had no controls: it saved no control inputs
            node.RemoveItem("control_inputs_shown");
            node.RemoveItem("control_inputs_hidden");
            var back = new GH_Archive();
            back.Deserialize_Binary(archive.Serialize_Binary());
            var fresh = new AnalyseIrComponent();
            fresh.CreateAttributes();
            Assert.True(fresh.Read(back.GetRootNode.FindChunk("component")));
            Assert.Empty(Errors(back));
            Assert.Equal(new[] { "irs" }, fresh.Params.Input.Select(p => p.Name).ToArray());
            Assert.False(fresh.ControlPortsShown);
            foreach (var name in new[] { "source_idx", "receiver_idx" })
            {
                var port = fresh.ControlPort(name)!;
                Assert.True(fresh.IsHidden(port));
                Assert.Empty(ControlState.PersistentAll(port));
            }
            Assert.Equal(GH_ParamAccess.tree, fresh.Params.Input[0].Access);
            Assert.Empty(fresh.RuntimeMessages(GH_RuntimeMessageLevel.Remark));
        }

        [Fact]
        public void AnAnalyseSavedWithItsControlPortsHidden_KeepsTheChosenIndicesAndTheLayout()
        {
            var saved = new AnalyseIrComponent();
            saved.ShowControlPorts(false);
            ControlState.Choose(saved, saved.ControlPort("source_idx")!, "1");
            ControlState.Choose(saved, saved.ControlPort("receiver_idx")!, "2");
            var fresh = new AnalyseIrComponent();
            var (archive, read) = SaveThenRead(saved, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.False(fresh.ControlPortsShown);
            Assert.Equal(new[] { "irs" }, fresh.Params.Input.Select(p => p.Name).ToArray());
            Assert.Equal("1", fresh.HiddenValue("source_idx"));
            Assert.Equal("2", fresh.HiddenValue("receiver_idx"));
        }

        [Fact]
        public void AnAnalyseSavedWithOneControlPortWired_KeepsTheHiddenOnesValueAndTheWiredPort()
        {
            var saved = new AnalyseIrComponent();
            saved.Params.Input.Single(p => p.Name == "receiver_idx").AddSource(new Param_Integer());
            saved.ShowControlPorts(false);
            ControlState.Choose(saved, saved.ControlPort("source_idx")!, "1");
            Assert.Equal(new[] { "irs", "receiver_idx" }, saved.Params.Input.Select(p => p.Name).ToArray());
            var fresh = new AnalyseIrComponent();
            var (archive, read) = SaveThenRead(saved, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.Equal(new[] { "irs", "receiver_idx" }, fresh.Params.Input.Select(p => p.Name).ToArray());
            Assert.Equal("1", fresh.HiddenValue("source_idx"));
            Assert.Null(fresh.HiddenValue("receiver_idx"));
        }

        [Fact]
        public void APlot2DSavedBeforeItChoseTheParameter_ReadsWithT30Chosen_AndKeepsItsSize()
        {
            var old = new Plot2DComponent { ChartSize = new System.Drawing.SizeF(420, 240) };
            old.Params.Input[0].Access = GH_ParamAccess.list;
            var fresh = new Plot2DComponent();
            var (archive, read) = SaveThenRead(old, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.Equal(new[] { "data" }, fresh.Params.Input.Select(p => p.Name).ToArray());
            Assert.Equal(GH_ParamAccess.tree, fresh.Params.Input[0].Access);
            Assert.Equal(new System.Drawing.SizeF(420, 240), fresh.ChartSize);
            Assert.Equal("T30", fresh.Choice("select"));
            Assert.Empty(fresh.RuntimeMessages(GH_RuntimeMessageLevel.Remark));
        }

        [Theory]
        [InlineData("all")]
        [InlineData("EDT")]
        public void APlot2DSavedWithAnotherChoice_ReadsTheChoice(string choice)
        {
            var saved = new Plot2DComponent();
            saved.SetChoice("select", choice);
            var fresh = new Plot2DComponent();
            var (archive, read) = SaveThenRead(saved, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.Equal(choice, fresh.Choice("select"));
        }

        [Fact]
        public void APlot2DSavedWithLFC_ReadsIt_AndWarnsThatTheDataDoNotHaveIt()
        {
            var saved = new Plot2DComponent();
            typeof(ComponentBase).GetField("_choices", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
                .SetValue(saved, new System.Collections.Generic.Dictionary<string, string> { ["select"] = "LFC" });
            var fresh = new Plot2DComponent();
            var (archive, read) = SaveThenRead(saved, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.Equal("LFC", fresh.Choice("select"));
            var warnings = new System.Collections.Generic.List<string>();
            PlotSelection.Select(new[] { new ParameterGoo.Content("T30", "T30", "Frequency (Hz)", "Time (s)", "bands", new[] { 125.0 }, new[] { new[] { 1.0 } }, 1.0, System.Array.Empty<string>()) }, fresh.Choice("select"), warnings.Add);
            Assert.Equal(new[] { "LFC is not in the connected data." }, warnings);
        }

        [Fact]
        public void APlot2DSavedWithAParameterChosen_ReadsTheChoice()
        {
            var saved = new Plot2DComponent();
            saved.SetChoice("select", "T30");
            var fresh = new Plot2DComponent();
            var (archive, read) = SaveThenRead(saved, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.Equal("T30", fresh.Choice("select"));
        }

        /// <summary>An Auralise as saved before it chose pairs: irs, anechoic_sounds wired to a path, mix_sources true (shown, or hidden behind its toggle).</summary>
        private static GH_Archive OldAuralise(bool mixHidden, string? level = null)
        {
            var old = new AuraliseComponent();
            foreach (var p in old.Params.Input.Skip(1).ToList())
                old.Params.UnregisterInputParameter(p);
            var sounds = new Param_GenericObject { Name = "anechoic_sounds", NickName = "anechoic_sounds", Optional = true, Access = GH_ParamAccess.tree };
            var path = new Param_String();
            path.SetPersistentData(new GH_String("/sounds/viola.wav"));
            sounds.AddSource(path);
            old.Params.RegisterInputParam(sounds);
            if (!mixHidden)
            {
                var mix = new Param_Boolean { Name = "mix_sources", NickName = "mix_sources", Optional = true };
                mix.SetPersistentData(new GH_Boolean(true));
                old.Params.RegisterInputParam(mix);
            }
            old.CreateAttributes();
            foreach (var p in old.Params)
                if (p.Attributes == null)
                    p.CreateAttributes();
            var archive = new GH_Archive();
            var node = archive.CreateTopLevelNode("component");
            Assert.True(old.Write(node));
            node.RemoveItem("control_inputs_shown");
            node.RemoveItem("control_inputs_hidden");
            node.SetBoolean("control_inputs_shown", !mixHidden);
            node.SetString("control_inputs_hidden", mixHidden ? "mix_sources" : "");
            if (mixHidden)
                node.SetString("control_value_mix_sources", "True");
            // as the level switch saved its choice
            if (level != null)
                node.SetString("choice_level", level);
            var back = new GH_Archive();
            back.Deserialize_Binary(archive.Serialize_Binary());
            return back;
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void Auralise_SavedWithAnechoicSoundsAndMixSources_ReadsWithoutThem(bool mixHidden)
        {
            var back = OldAuralise(mixHidden);
            var fresh = new AuraliseComponent();
            fresh.CreateAttributes();
            Assert.True(fresh.Read(back.GetRootNode.FindChunk("component")));
            Assert.Empty(Errors(back));
            Assert.Equal(!mixHidden, fresh.ControlPortsShown);
            Assert.Equal(mixHidden ? new[] { "irs" } : new[] { "irs", "source_idx", "receiver_idx", "sound" }, fresh.Params.Input.Select(p => p.Name).ToArray());
            Assert.Equal(GH_ParamAccess.tree, fresh.Params.Input[0].Access);
            foreach (var name in new[] { "source_idx", "receiver_idx", "sound" })
            {
                var port = fresh.ControlPort(name)!;
                Assert.Equal(0, port.SourceCount);
                Assert.Empty(ControlState.PersistentAll(port));
                Assert.Equal(ComponentSpec.Load("CA_Auralise").Inputs.Single(p => p.Name == name).Description, port.Description);
            }
            // no level saved: the default, none
            Assert.Equal("none", fresh.Choice("level"));
            Assert.Empty(fresh.RuntimeMessages(GH_RuntimeMessageLevel.Remark));
        }

        [Theory]
        [InlineData("relative")]
        [InlineData("individual")]
        [InlineData("none")]
        public void AnAuraliseSavedWithItsLevelSwitch_ReadsTheLevelIntoTheDropdown(string level)
        {
            var back = OldAuralise(false, level);
            var fresh = new AuraliseComponent();
            fresh.CreateAttributes();
            Assert.True(fresh.Read(back.GetRootNode.FindChunk("component")));
            Assert.Empty(Errors(back));
            Assert.IsType<DropdownControl>(fresh.Controls.Single(c => c.Port == "level"));
            Assert.Equal(level, fresh.Choice("level"));
        }

        [Fact]
        public void AnAuraliseSavedWithItsSamplingRateOutput_ReadsWithOnlyTheSignals()
        {
            var saved = new AuraliseComponent();
            saved.Params.RegisterOutputParam(new Param_Integer { Name = "sampling_rate", NickName = "sampling_rate" });
            var fresh = new AuraliseComponent();
            var (archive, read) = SaveThenRead(saved, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.Equal(new[] { "signals" }, fresh.Params.Output.Select(p => p.Name).ToArray());
            Assert.Equal(new[] { "signals" }, ComponentSpec.Load("CA_Auralise").Outputs.Select(p => p.Name).ToArray());
        }

        [Fact]
        public void AnAuraliseSavedWithItsControlPortsHidden_KeepsTheChosenPairsAndSound()
        {
            var saved = new AuraliseComponent();
            saved.ShowControlPorts(false);
            ControlState.Choose(saved, saved.ControlPort("source_idx")!, "1");
            ControlState.Choose(saved, saved.ControlPort("receiver_idx")!, "0");
            ControlState.Choose(saved, saved.ControlPort("sound")!, "/sounds/viola.wav");
            saved.SetChoice("level", "individual");
            var fresh = new AuraliseComponent();
            var (archive, read) = SaveThenRead(saved, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.False(fresh.ControlPortsShown);
            Assert.Equal(new[] { "irs" }, fresh.Params.Input.Select(p => p.Name).ToArray());
            Assert.Equal("1", fresh.HiddenValue("source_idx"));
            Assert.Equal("0", fresh.HiddenValue("receiver_idx"));
            Assert.Equal("/sounds/viola.wav", fresh.HiddenValue("sound"));
            Assert.Equal("individual", fresh.Choice("level"));
        }

        [Fact]
        public void AnAuraliseSavedWithOneControlPortWired_KeepsTheHiddenOnesValuesAndTheWiredPort()
        {
            var saved = new AuraliseComponent();
            saved.Params.Input.Single(p => p.Name == "source_idx").AddSource(new Param_Integer());
            saved.ShowControlPorts(false);
            ControlState.Choose(saved, saved.ControlPort("receiver_idx")!, "1");
            ControlState.Choose(saved, saved.ControlPort("sound")!, "/sounds/viola.wav");
            Assert.Equal(new[] { "irs", "source_idx" }, saved.Params.Input.Select(p => p.Name).ToArray());
            var fresh = new AuraliseComponent();
            var (archive, read) = SaveThenRead(saved, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.Equal(new[] { "irs", "source_idx" }, fresh.Params.Input.Select(p => p.Name).ToArray());
            Assert.Null(fresh.HiddenValue("source_idx"));
            Assert.Equal("1", fresh.HiddenValue("receiver_idx"));
            Assert.Equal("/sounds/viola.wav", fresh.HiddenValue("sound"));
        }

        [Theory]
        [InlineData(false, false)]
        [InlineData(false, true)]
        [InlineData(true, false)]
        [InlineData(true, true)]
        public void AnAuraliseSavedWithALoadedSound_ReadsTheChoiceAndTheLoadedEntryBack(bool hidden, bool loadedChosen)
        {
            var saved = new AuraliseComponent();
            if (hidden)
                saved.ShowControlPorts(false);
            var file = System.IO.Path.Combine("takes", "bsn_tele_44.wav");
            saved.OpenSound = _ => file;
            saved.OnDropdownOption("sound", "Load...");
            if (!loadedChosen)
                ControlState.Choose(saved, saved.ControlPort("sound")!, "viola");
            var fresh = new AuraliseComponent();
            var (archive, read) = SaveThenRead(saved, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.Equal(!hidden, fresh.ControlPortsShown);
            Assert.Equal(loadedChosen ? file : "viola", ControlState.Persistent(fresh.ControlPort("sound")!));
            Assert.Equal(file, fresh.Loaded);
            Assert.Equal(new[] { "female_speech", "male_speech", "viola", "guitar", "castanets", file, "Load..." }, fresh.OptionsOf(AuraliseComponent.SoundDropdown).values);
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void AnAuraliseSavedWithFemaleSpeechChosen_KeepsIt_OneSavedWithNoneHasNone(bool chosen)
        {
            var saved = new AuraliseComponent();
            if (chosen)
                ControlState.Choose(saved, saved.ControlPort("sound")!, "female_speech");
            var fresh = new AuraliseComponent();
            var (archive, read) = SaveThenRead(saved, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.Equal(chosen ? "female_speech" : null, ControlState.Persistent(fresh.ControlPort("sound")!));
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void AnAuraliseSavedWithTheAnechoicSoundButton_ReadsItsFileAsTheLoadedEntryChosen(bool hidden)
        {
            // the button kept the file as the port's value, shown or hidden (control_value_sound), and nothing else
            var saved = new AuraliseComponent();
            if (hidden)
                saved.ShowControlPorts(false);
            var file = System.IO.Path.Combine("sounds", "bsn_tele_44.wav");
            ControlState.Choose(saved, saved.ControlPort("sound")!, file);
            saved.CreateAttributes();
            foreach (var p in saved.Params)
                if (p.Attributes == null)
                    p.CreateAttributes();
            var archive = new GH_Archive();
            var node = archive.CreateTopLevelNode("component");
            Assert.True(saved.Write(node));
            node.RemoveItem("anechoic_sound_loaded");
            var back = new GH_Archive();
            back.Deserialize_Binary(archive.Serialize_Binary());
            Assert.Equal(hidden, back.GetRootNode.FindChunk("component").ItemExists("control_value_sound"));
            var fresh = new AuraliseComponent();
            fresh.CreateAttributes();
            Assert.True(fresh.Read(back.GetRootNode.FindChunk("component")));
            Assert.Empty(Errors(back));
            Assert.Equal(!hidden, fresh.ControlPortsShown);
            Assert.Equal(file, ControlState.Persistent(fresh.ControlPort("sound")!));
            Assert.Equal(file, fresh.Loaded);
            Assert.Equal(new[] { "Female speech", "Male speech", "Viola", "Guitar", "Castanets", "bsn_tele_44.wav", "Load..." }, fresh.OptionsOf(AuraliseComponent.SoundDropdown).labels);
        }

        /// <summary>A Source as saved with a sound_file input between direction and spl (shown with its path, or hidden behind its button) and spl 90.</summary>
        private static GH_Archive OldSource(bool soundHidden)
        {
            var old = new SourceComponent();
            foreach (var port in old.Params.Input.Where(p => p.Name == "sound_file").ToList())
                old.Params.UnregisterInputParameter(port);
            var spl = old.Params.Input.Single(p => p.Name == "spl");
            ControlState.Assign(spl, new[] { "90" });
            if (!soundHidden)
            {
                var sound = new Param_String { Name = "sound_file", NickName = "sound_file", Optional = true };
                sound.SetPersistentData(new GH_String("/a.wav"));
                old.Params.UnregisterInputParameter(spl, false);
                old.Params.RegisterInputParam(sound);
                old.Params.RegisterInputParam(spl);
            }
            old.SetChoice("directivity", "cardioid");
            old.CreateAttributes();
            foreach (var p in old.Params)
                if (p.Attributes == null)
                    p.CreateAttributes();
            var archive = new GH_Archive();
            var node = archive.CreateTopLevelNode("component");
            Assert.True(old.Write(node));
            node.RemoveItem("control_inputs_shown");
            node.RemoveItem("control_inputs_hidden");
            node.SetBoolean("control_inputs_shown", !soundHidden);
            node.SetString("control_inputs_hidden", soundHidden ? "sound_file" : "");
            if (soundHidden)
                node.SetString("control_value_sound_file", "/a.wav");
            var back = new GH_Archive();
            back.Deserialize_Binary(archive.Serialize_Binary());
            return back;
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void Source_SavedWithASoundFile_ReadsWithoutIt(bool soundHidden)
        {
            var back = OldSource(soundHidden);
            var fresh = new SourceComponent();
            fresh.CreateAttributes();
            Assert.True(fresh.Read(back.GetRootNode.FindChunk("component")));
            Assert.Empty(Errors(back));
            Assert.Equal(new[] { "position", "name", "direction", "spl" }, fresh.Params.Input.Select(p => p.Name).ToArray());
            Assert.Equal(new[] { "90" }, ControlState.PersistentAll(fresh.Params.Input.Single(p => p.Name == "spl")));
            Assert.Equal("cardioid", fresh.Choice("directivity"));
        }

        [Fact]
        public void ASourceSavedNow_KeepsItsDirectivityAndLevel()
        {
            var saved = new SourceComponent();
            saved.SetChoice("directivity", "hypercardioid");
            ControlState.Assign(saved.Params.Input.Single(p => p.Name == "spl"), new[] { "70" });
            var fresh = new SourceComponent();
            var (archive, read) = SaveThenRead(saved, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.Equal(new[] { "position", "name", "direction", "spl" }, fresh.Params.Input.Select(p => p.Name).ToArray());
            Assert.Equal("hypercardioid", fresh.Choice("directivity"));
            Assert.Equal(new[] { "70" }, ControlState.PersistentAll(fresh.Params.Input.Single(p => p.Name == "spl")));
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
