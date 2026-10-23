using System;
using System.Drawing;
using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>The body controls of the plan of 2026-10-23: typed inputs, toggles, dropdowns with labels, sliders, file buttons.</summary>
    public class BodyControlsTests
    {
        // the test runner reads these signatures before WinFormsResolver is registered: no plugin type in them
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

        private static readonly string[] Orders = Enumerable.Range(0, 11).Select(i => i.ToString()).ToArray();

        [Fact]
        public void AnIntegerOrBoolInput_TakesTheBodysValue_AndEmptyClearsIt()
        {
            var mono = new SimulateMonoComponent();
            var ism = mono.ControlPort("ism_order")!;
            ControlState.Choose(mono, ism, "8");
            Assert.Equal(8, ControlState.Persistent(ism));
            ControlState.Choose(mono, ism, "");
            Assert.Null(ControlState.Persistent(ism));
            var tracing = mono.ControlPort("ray_tracing")!;
            ControlState.Choose(mono, tracing, "False");
            Assert.Equal(false, ControlState.Persistent(tracing));
        }

        [Fact]
        public void AHiddenInputOfAnyType_KeepsItsValue_ForTheSolveAndTheFile()
        {
            var mono = new SimulateMonoComponent();
            mono.ShowControlPorts(false);
            ControlState.Choose(mono, mono.ControlPort("ism_order")!, "8");
            ControlState.Choose(mono, mono.ControlPort("absorb_air")!, "False");
            Assert.Equal(8, mono.HiddenAs<int>("ism_order"));
            Assert.False(mono.HiddenAs<bool>("absorb_air"));
            // untouched: the method's own default, as an unconnected input
            Assert.Null(mono.HiddenAs<bool>("normalize"));
            var back = (ComponentBase)RoundTrip(mono);
            Assert.Equal("8", back.HiddenValue("ism_order"));
            Assert.Equal(8, back.HiddenAs<int>("ism_order"));
            Assert.False(back.HiddenAs<bool>("absorb_air"));
        }

        [Fact]
        public void AnInputLeftAtItsDeclaredDefault_IsNotGiven()
        {
            // the ISM order shows 3, and 3 is the method's default: nothing is sent, the messages name no change
            var mono = new SimulateMonoComponent();
            mono.ShowControlPorts(false);
            Assert.Null(mono.HiddenAs<int>("ism_order"));
            Assert.Equal(("3", false), ControlState.Of(mono.ControlPort("ism_order")!, Orders, "3"));
        }

        [Fact]
        public void InlineToggles_ShareOneRow_EachWithItsOwnBox()
        {
            var controls = new Control[]
            {
                new DropdownControl("ism_order", Orders, Title: "ISM order"),
                new ToggleControl("ray_tracing", "Ray tracing", Inline: true),
                new ToggleControl("absorb_air", "Air", Inline: true),
                new ToggleControl("normalize", "Normalize", Inline: true),
                new ButtonControl("run", "Run"),
            };
            var boxes = ControlLayout.Arrange(controls, new RectangleF(0, 0, 150, 0));
            var row = boxes.Skip(1).Take(3).ToList();
            Assert.All(row, b => Assert.Equal(row[0].Bounds.Top, b.Bounds.Top));
            Assert.All(row, b => Assert.Equal(row[0].Bounds.Width, b.Bounds.Width, 3));
            Assert.True(row[0].Bounds.Right <= row[1].Bounds.Left && row[1].Bounds.Right <= row[2].Bounds.Left);
            Assert.Equal(150 - 2 * ControlLayout.Margin, row[2].Bounds.Right - row[0].Bounds.Left, 3);
            Assert.True(boxes[0].Bounds.Bottom <= row[0].Bounds.Top && row[0].Bounds.Bottom <= boxes[4].Bounds.Top);
            Assert.Equal(boxes[4].Bounds.Bottom + ControlLayout.Margin, ControlLayout.Height(controls), 3);
            Assert.Equal("ISM order", ControlLayout.Title(controls[0]));
        }

        [Fact]
        public void AToggleOnAnInput_ShowsItsDefaultUntilChosen()
        {
            var mono = new SimulateMonoComponent();
            var tracing = mono.Controls.OfType<ToggleControl>().Single(t => t.Port == "ray_tracing");
            Assert.Equal((true, false), ControlState.Toggle(mono.ControlPort("ray_tracing")!, tracing.Default));
            ControlState.Choose(mono, mono.ControlPort("ray_tracing")!, "False");
            Assert.Equal((false, false), ControlState.Toggle(mono.ControlPort("ray_tracing")!, tracing.Default));
            var auralise = new AuraliseComponent();
            var mix = auralise.Controls.OfType<ToggleControl>().Single(t => t.Port == "mix_sources");
            Assert.Equal((false, false), ControlState.Toggle(auralise.ControlPort("mix_sources")!, mix.Default));
        }

        [Fact]
        public void EveryToggle_StartsAtItsInputsDefault()
        {
            var defaults = new (ComponentBase component, string port, bool on)[]
            {
                (new SimulateMonoComponent(), "ray_tracing", true), (new SimulateMonoComponent(), "absorb_air", true),
                (new SimulateMonoComponent(), "normalize", true),
                (new SimulateBinauralComponent(), "late_envelope", true), (new SimulateBinauralComponent(), "normalize", true),
                (new SimulateAmbisonicsComponent(), "late_envelope", true), (new SimulateAmbisonicsComponent(), "normalize", true),
                (new AuraliseComponent(), "mix_sources", false), (new VisualiseRaysComponent(), "scattering", false),
            };
            foreach (var (component, port, on) in defaults)
                Assert.Equal(on, component.Controls.OfType<ToggleControl>().Single(t => t.Port == port).Default);
        }

        [Fact]
        public void ATooltip_GivesTheInputsDescription_AndItsDefault()
        {
            var mono = new SimulateMonoComponent();
            Assert.Equal(("ism_order", "Highest image source (ISM) order. With ray tracing, the ray tracer takes over after it. (default 3)"), mono.ControlTooltip("ism_order"));
            // a description that names its default already keeps its words
            Assert.Equal(ComponentSpec.Load("CA_Simulate_Mono").Inputs.Single(p => p.Name == "ray_tracing").Description, mono.ControlTooltip("ray_tracing").text);
            Assert.EndsWith("(default 48 kHz)", new RoomSceneComponent().ControlTooltip("sampling_rate").text);
            Assert.Equal(("level", "none, relative or individual (default relative)"), new AuraliseComponent().ControlTooltip("level"));
            Assert.EndsWith("(default on)", new VisualiseRaysComponent().ControlTooltip("preview").text);
        }

        [Fact]
        public void TheIsmOrder_IsADropdownFrom0To10_On_EverySimulation()
        {
            foreach (var simulation in new ComponentBase[] { new SimulateMonoComponent(), new SimulateBinauralComponent(), new SimulateAmbisonicsComponent() })
            {
                var ism = simulation.Controls.OfType<DropdownControl>().Single(c => c.Port == "ism_order");
                Assert.Equal(Orders, ism.Options);
                Assert.Equal("ISM order", ControlLayout.Title(ism));
                Assert.Equal(3, ComponentSpec.Load(simulation is SimulateMonoComponent ? "CA_Simulate_Mono" : simulation is SimulateBinauralComponent ? "CA_Simulate_Binaural" : "CA_Simulate_Ambisonics")
                    .Inputs.Single(p => p.Name == "ism_order").Default);
            }
        }

        [Fact]
        public void TheControlPorts_FollowTheBody()
        {
            Assert.Equal(new[] { "ism_order", "ray_tracing", "absorb_air", "normalize" }, new SimulateMonoComponent().ControlPorts);
            Assert.Equal(new[] { "tail", "diffraction", "ism_order", "late_envelope", "normalize", "hrtf", "run" }, new SimulateBinauralComponent().ControlPorts);
            Assert.Equal(new[] { "tail", "diffraction", "ism_order", "sh_order", "late_envelope", "normalize", "run" }, new SimulateAmbisonicsComponent().ControlPorts);
            Assert.Equal(new[] { "sampling_rate" }, new RoomSceneComponent().ControlPorts);
            Assert.Equal(new[] { "mix_sources" }, new AuraliseComponent().ControlPorts);
            Assert.Equal(new[] { "source_idx", "scattering", "order" }, new VisualiseRaysComponent().ControlPorts);
            Assert.Equal(new[] { "source_idx", "receiver_idx", "order" }, new VisualiseReflectionsComponent().ControlPorts);
            Assert.Equal(new[] { "time", "speed", "play" }, new VisualiseBilliardComponent().ControlPorts);
            Assert.Equal(new[] { "band", "source_idx" }, new VisualiseEnergyMapComponent().ControlPorts);
            Assert.Equal(new[] { "source_idx", "receiver_idx" }, new DeconstructIrGraphComponent().ControlPorts);
            Assert.Equal(new[] { "sound_file" }, new SourceComponent().ControlPorts);
            Assert.Equal(new[] { "category" }, new ConstructMaterialComponent().ControlPorts);
        }

        [Fact]
        public void TheSamplingRate_IsASwitchOfThreeRates_ShownInKilohertz()
        {
            var rate = new RoomSceneComponent().Controls.OfType<SwitchControl>().Single();
            Assert.Equal(new[] { "44100", "48000", "96000" }, rate.Options);
            Assert.Equal(new[] { "44.1 kHz", "48 kHz", "96 kHz" }, rate.Options.Select(o => ControlLayout.Label(rate, o)));
            Assert.Equal("48000", rate.Default);
        }

        [Fact]
        public void Ambisonics_HasTheShOrderSwitch_AndBinauralTheHrtfButton()
        {
            var ambisonics = new SimulateAmbisonicsComponent();
            var sh = ambisonics.Controls.OfType<SwitchControl>().Single();
            Assert.Equal(("sh_order", "1"), (sh.Port, sh.Default));
            Assert.Equal(new[] { "1", "2", "3" }, sh.Options);
            Assert.DoesNotContain(ambisonics.Controls, c => c.Port == "hrtf");
            var binaural = new SimulateBinauralComponent();
            Assert.Empty(binaural.Controls.OfType<SwitchControl>());
            Assert.True(binaural.HandlesButton("hrtf"));
            Assert.Equal("HRTF: KEMAR", binaural.ButtonLabel("hrtf"));
            ControlState.Choose(binaural, binaural.ControlPort("hrtf")!, "/sets/my_head.sofa");
            Assert.Equal("HRTF: my_head.sofa", binaural.ButtonLabel("hrtf"));
            // Run last, after the controls
            Assert.Equal("run", binaural.Controls[^1].Port);
            Assert.Equal("run", ambisonics.Controls[^1].Port);
        }

        [Fact]
        public void TheSourcesSoundFile_IsAFileButtonNamingTheFile()
        {
            var source = new SourceComponent();
            Assert.True(source.HandlesButton("sound_file"));
            Assert.Equal("Sound file...", source.ButtonLabel("sound_file"));
            ControlState.Choose(source, source.ControlPort("sound_file")!, "/sounds/viola.wav");
            Assert.Equal("Sound: viola.wav", source.ButtonLabel("sound_file"));
        }

        [Fact]
        public void IndexDropdowns_OfferAll_AndTheNamesOfTheScene()
        {
            var rays = new VisualiseRaysComponent();
            var dropdown = rays.Controls.OfType<DropdownControl>().Single(c => c.Port == "source_idx");
            Assert.Equal("Source", ControlLayout.Title(dropdown));
            Assert.Equal(new[] { "" }, rays.OptionsOf(dropdown).values);
            Assert.Equal(new[] { "All" }, rays.OptionsOf(dropdown).labels);
            rays.SetIndexOptions("source_idx", new[] { "lectern", "piano" });
            Assert.Equal(new[] { "", "0", "1" }, rays.OptionsOf(dropdown).values);
            Assert.Equal(new[] { "All", "0: lectern", "1: piano" }, rays.OptionsOf(dropdown).labels);
            var graph = new DeconstructIrGraphComponent();
            Assert.Equal(new[] { "Source", "Receiver" }, graph.Controls.OfType<DropdownControl>().Select(ControlLayout.Title));
        }

        [Fact]
        public void TheEnergyMap_HasPreviewBandAndSource_NoPlayback()
        {
            var map = new VisualiseEnergyMapComponent();
            Assert.DoesNotContain(map.Controls, c => c is ButtonControl);
            var band = map.Controls.OfType<DropdownControl>().Single(c => c.Port == "band");
            Assert.Equal(new[] { "", "0", "1", "2", "3", "4", "5", "6" }, band.Options);
            Assert.Equal(new[] { "A-weighted", "125 Hz", "250 Hz", "500 Hz", "1 kHz", "2 kHz", "4 kHz", "8 kHz" }, band.Options.Select(o => ControlLayout.Label(band, o)));
            Assert.DoesNotContain("Play", ComponentSpec.Load("CA_Visualise_Energy_Map").Description);
        }

        [Fact]
        public void TheReflectionOrders_AreSeveralSegmentsLitAtOnce()
        {
            var reflections = new VisualiseReflectionsComponent();
            var orders = reflections.Controls.OfType<MultiSwitchControl>().Single();
            Assert.Equal(new[] { "0", "1", "2", "3", "4" }, orders.Options);
            var port = reflections.ControlPort("order")!;
            // empty: the first order, as the input says
            Assert.Equal(new[] { "1" }, ControlState.Members(port, orders.Default));
            ControlState.ToggleMember(reflections, port, "2", orders.Default);
            Assert.Equal(new[] { "1", "2" }, ControlState.Members(port, orders.Default));
            ControlState.ToggleMember(reflections, port, "1", orders.Default);
            Assert.Equal(new[] { "2" }, ControlState.Members(port, orders.Default));
            reflections.ShowControlPorts(false);
            var back = (ComponentBase)RoundTrip(reflections);
            Assert.Equal(new[] { "2" }, ControlState.Members(back.ControlPort("order")!, orders.Default));
        }

        [Fact]
        public void TheRaysDrawnOrder_IsASliderOfWholeOrders_UpToTheTracedOrder()
        {
            var rays = new VisualiseRaysComponent();
            var slider = rays.Controls.OfType<SliderControl>().Single();
            Assert.Equal(("order", 1.0), (slider.Port, slider.Step));
            Assert.Equal((1.0, 10.0), rays.SliderRange(slider));
            Assert.Equal(4.0, ComponentUi.SliderValue(slider, 1, 10, new RectangleF(0, 0, 90, 20), 32));
        }

        [Fact]
        public void TheBilliard_ScrubsTimeOnASlider_AndPlaysAtAFactorOfItsDefaultSpeed()
        {
            var billiard = new VisualiseBilliardComponent();
            var time = billiard.Controls.OfType<SliderControl>().Single();
            Assert.Equal(("time", "ms"), (time.Port, time.Unit));
            var speed = billiard.Controls.OfType<DropdownControl>().Single();
            Assert.Equal(new[] { "0.25", "0.5", "1", "2", "4" }, speed.Options);
            Assert.Equal("1", speed.Default);
            Assert.Equal("2x", ControlLayout.Label(speed, "2"));
            // a factor of the whole field in 10 s
            Assert.Equal(300.0, VisualiseBilliardComponent.Rate(1.0, 3000));
            Assert.Equal(600.0, VisualiseBilliardComponent.Rate(2.0, 3000));
            Assert.Equal(300.0, VisualiseBilliardComponent.Rate(null, 3000));
        }

        [Fact]
        public void ConstructMaterial_ChoosesItsCategoryFromTheLibrarysCategories()
        {
            var category = new ConstructMaterialComponent().Controls.OfType<DropdownControl>().Single();
            Assert.Equal("category", category.Port);
            Assert.Equal(13, category.Options.Length);
            Assert.Equal("Other", category.Default);
            Assert.Contains("Stone & Tiles", category.Options);
        }

        [Fact]
        public void TheAudioPlayer_HasNoSamplingRateInput()
        {
            Assert.Equal(new[] { "audio" }, ComponentSpec.Load("CA_Audio_Player").Inputs.Select(p => p.Name));
        }
    }
}
