using System;
using System.Drawing;
using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>The body controls: typed inputs, toggles, dropdowns with labels, sliders, file buttons.</summary>
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
            var rays = new VisualiseRaysComponent();
            var scattering = rays.ControlPort("scattering")!;
            ControlState.Choose(rays, scattering, "True");
            Assert.Equal(true, ControlState.Persistent(scattering));
        }

        [Fact]
        public void AShownInput_LetsGoOfWhatItCollected_WhenTheBodySetsIt()
        {
            // expiring the component clears only its outputs: the input must expire itself, or the solve reads the old value
            var reflections = new VisualiseReflectionsComponent();
            var order = reflections.ControlPort("order")!;
            ControlState.Choose(reflections, order, "1");
            order.CollectData();
            Assert.Equal(1, order.VolatileDataCount);
            ControlState.ToggleMember(reflections, order, "3", new[] { "1" });
            Assert.Equal(0, order.VolatileDataCount);
            order.CollectData();
            Assert.Equal(new[] { "1", "3" }, order.VolatileData.AllData(true).Select(g => g.ToString()));
        }

        [Fact]
        public void AHiddenInputOfAnyType_KeepsItsValue_ForTheSolveAndTheFile()
        {
            var mono = new SimulateMonoComponent();
            mono.ShowControlPorts(false);
            ControlState.Choose(mono, mono.ControlPort("ism_order")!, "8");
            var rays = new VisualiseRaysComponent();
            rays.ShowControlPorts(false);
            // untouched: the method's own default, as an unconnected input
            Assert.Null(rays.HiddenAs<bool>("scattering"));
            ControlState.Choose(rays, rays.ControlPort("scattering")!, "True");
            Assert.Equal(8, mono.HiddenAs<int>("ism_order"));
            Assert.True(rays.HiddenAs<bool>("scattering"));
            var back = (ComponentBase)RoundTrip(mono);
            Assert.Equal("8", back.HiddenValue("ism_order"));
            Assert.Equal(8, back.HiddenAs<int>("ism_order"));
            Assert.True(((ComponentBase)RoundTrip(rays)).HiddenAs<bool>("scattering"));
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
        public void AToggleOnAnInput_ShowsItsDefaultUntilChosen()
        {
            var mono = new SimulateMonoComponent();
            var rays = new VisualiseRaysComponent();
            var scattering = rays.Controls.OfType<ToggleControl>().Single(t => t.Port == "scattering");
            Assert.Equal((false, false), ControlState.Toggle(rays.ControlPort("scattering")!, scattering.Default));
            ControlState.Choose(rays, rays.ControlPort("scattering")!, "True");
            Assert.Equal((true, false), ControlState.Toggle(rays.ControlPort("scattering")!, scattering.Default));
        }

        [Fact]
        public void ATooltip_GivesTheInputsDescription_AndItsDefault()
        {
            var mono = new SimulateMonoComponent();
            Assert.Equal(("ism_order", "Highest image source (ISM) order. With ray tracing, the ray tracer takes over after it. (default 3)"), mono.ControlTooltip("ism_order"));
            // a description that names its default already keeps its words
            var (title, text) = new SimulateBinauralComponent().ControlTooltip("diffraction");
            Assert.Equal("diffraction", title);
            Assert.Equal(ComponentSpec.Load("CA_Simulate_Binaural").Inputs.Single(p => p.Name == "diffraction").Description, text);
            Assert.EndsWith("(default 48 kHz)", new RoomSceneComponent().ControlTooltip("sampling_rate").text);
            Assert.Equal(("level", "None, Relative or Individual (default None)"), new AuraliseComponent().ControlTooltip("level"));
            Assert.EndsWith("(default on)", new VisualiseRaysComponent().ControlTooltip("show").text);
        }

        [Fact]
        public void TheIsmOrder_IsAStepSliderOfTheOrdersFrom0To10_OnEverySimulation()
        {
            foreach (var simulation in new ComponentBase[] { new SimulateMonoComponent(), new SimulateBinauralComponent(), new SimulateAmbisonicsComponent() })
            {
                var ism = simulation.Controls.OfType<StepSliderControl>().Single(c => c.Port == "ism_order");
                Assert.Equal(Orders, ism.Options);
                Assert.Equal(Orders, ism.Labels);
                Assert.Equal(("3", "ISM order"), (ism.Default, ControlLayout.Title(ism)));
                Assert.DoesNotContain(simulation.Controls, c => c is SliderControl { Port: "ism_order" } or DropdownControl { Port: "ism_order" });
                // it shows the input's own value, 3 until moved
                Assert.Equal(3, ControlState.Persistent(simulation.ControlPort("ism_order")!));
                Assert.Equal(("3", false), ComponentUi.StepValue(simulation, ism));
                var row = new RectangleF(0, 0, 120, StepSlider.Height);
                Assert.Equal("7", StepSlider.Nearest(row, StepSlider.StopX(row, 7, ism) + 2, ism, simulation.OfferedOptionsOf("ism_order")));
                Assert.Equal(3, ComponentSpec.Load(simulation is SimulateMonoComponent ? "CA_Simulate_Mono" : simulation is SimulateBinauralComponent ? "CA_Simulate_Binaural" : "CA_Simulate_Ambisonics")
                    .Inputs.Single(p => p.Name == "ism_order").Default);
            }
        }

        [Fact]
        public void SimulateMono_HasTheHybridsBody_AndRayTracingAsAnInput()
        {
            var mono = new SimulateMonoComponent();
            Assert.Equal(new Control[] { SimulateMonoComponent.IsmOrder, new ButtonControl("run", "Run") }, mono.Controls);
            Assert.DoesNotContain("absorb_air", ComponentSpec.Load("CA_Simulate_Mono").Inputs.Select(p => p.Name));
            mono.ShowControlPorts(false);
            Assert.Contains("ray_tracing", mono.Params.Input.Select(p => p.Name));
        }

        [Fact]
        public void TheControlPorts_FollowTheBody()
        {
            Assert.Equal(new[] { "ism_order", "run" }, new SimulateMonoComponent().ControlPorts);
            Assert.Equal(new[] { "ism_order", "tail", "diffraction", "late_envelope", "hrtf", "run" }, new SimulateBinauralComponent().ControlPorts);
            Assert.Equal(new[] { "ism_order", "tail", "diffraction", "late_envelope", "run" }, new SimulateAmbisonicsComponent().ControlPorts);
            Assert.Empty(new RoomSceneComponent().ControlPorts);
            Assert.Equal(new[] { "source_idx", "receiver_idx", "sound" }, new AuraliseComponent().ControlPorts);
            Assert.Equal(new[] { "source_idx", "scattering", "order" }, new VisualiseRaysComponent().ControlPorts);
            Assert.Equal(new[] { "source_idx", "receiver_idx", "order" }, new VisualiseReflectionsComponent().ControlPorts);
            Assert.Equal(new[] { "time", "speed", "play" }, new VisualiseBilliardComponent().ControlPorts);
            Assert.Equal(new[] { "frequency_bands", "source_idx", "run" }, new VisualiseEnergyMapComponent().ControlPorts);
            Assert.Equal(new[] { "sound" }, new SourceComponent().ControlPorts);
            Assert.Equal(new[] { "category" }, new ConstructMaterialComponent().ControlPorts);
            Assert.Empty(new AcousticObjectComponent().ControlPorts);
            Assert.True(new AcousticObjectComponent().IsChoice("role"));
            Assert.Empty(new WorkerComponent().ControlPorts);
        }

        [Fact]
        public void TheSamplingRate_IsADropdownOfThreeRates_TheScenesOwnChoice()
        {
            var scene = new RoomSceneComponent();
            var rate = scene.Controls.OfType<DropdownControl>().Single();
            Assert.Equal(new[] { "44100", "48000", "96000" }, rate.Options);
            Assert.Equal(new[] { "44.1 kHz", "48 kHz", "96 kHz" }, rate.Options.Select(o => ControlLayout.Label(rate, o)));
            Assert.Equal("48000", rate.Default);
            // not an input: the component holds it, and saves it
            Assert.DoesNotContain("sampling_rate", ComponentSpec.Load("CA_Model").Inputs.Select(p => p.Name));
            Assert.True(scene.IsChoice("sampling_rate"));
            Assert.Equal("48000", scene.Choice("sampling_rate"));
            scene.SetChoice("sampling_rate", "96000");
            Assert.Equal("96000", ((RoomSceneComponent)RoundTrip(scene)).Choice("sampling_rate"));
        }

        [Fact]
        public void Ambisonics_HasTheAmbisonicOrderSlider_AndBinauralTheHrtfButton()
        {
            var ambisonics = new SimulateAmbisonicsComponent();
            var sh = ambisonics.Controls.OfType<StepSliderControl>().Single(c => c.Port == "sh_order");
            Assert.Equal(("sh_order", "1", "Ambisonic order"), (sh.Port, sh.Default, ControlLayout.Title(sh)));
            Assert.Equal(new[] { "1", "2", "3", "4", "5", "6", "7" }, sh.Options);
            Assert.Equal(sh.Options, sh.Labels);
            Assert.DoesNotContain(ambisonics.Controls, c => c is DropdownControl { Port: "sh_order" });
            // the component's own choice, not a hidden input
            Assert.DoesNotContain("sh_order", ComponentSpec.Load("CA_Simulate_Ambisonics").Inputs.Select(p => p.Name));
            Assert.Null(ambisonics.ControlPort("sh_order"));
            Assert.True(ambisonics.IsChoice("sh_order"));
            Assert.Equal("1", ambisonics.Choice("sh_order"));
            Assert.Equal(("1", false), ComponentUi.StepValue(ambisonics, sh));
            Assert.True(ComponentUi.Pressable(ambisonics, sh, null));
            Assert.Equal(("sh_order", "1, 2, 3, 4, 5, 6 or 7 (default 1)"), ambisonics.ControlTooltip("sh_order"));
            // a stop let go on sets the choice, saved with the definition
            ComponentUi.ChooseStop(ambisonics, sh, "4");
            Assert.Equal(("4", false), ComponentUi.StepValue(ambisonics, sh));
            ambisonics.SetChoice("sh_order", "5");
            Assert.Equal("5", ((SimulateHybridComponent)RoundTrip(ambisonics)).Choice("sh_order"));
            Assert.Empty(ambisonics.Controls.OfType<SwitchControl>());
            Assert.DoesNotContain(ambisonics.Controls, c => c.Port == "hrtf");
            var binaural = new SimulateBinauralComponent();
            Assert.Empty(binaural.Controls.OfType<SwitchControl>());
            Assert.True(binaural.HandlesButton("hrtf"));
            Assert.Equal("HRTF: KU100", binaural.ButtonLabel("hrtf"));
            ControlState.Choose(binaural, binaural.ControlPort("hrtf")!, "/sets/my_head.sofa");
            Assert.Equal("HRTF: my_head.sofa", binaural.ButtonLabel("hrtf"));
            // Run last, set apart from the controls above it
            Assert.Equal("run", binaural.Controls[^1].Port);
            Assert.Equal("run", ambisonics.Controls[^1].Port);
            foreach (var hybrid in new ComponentBase[] { binaural, ambisonics })
            {
                var boxes = ControlLayout.Arrange(hybrid.Controls, new RectangleF(0, 0, 150, 0));
                Assert.Equal(ControlLayout.ButtonGap, boxes[^1].Bounds.Top - boxes[^2].Bounds.Bottom, 3);
            }
        }

        [Fact]
        public void TheLateEnvelope_IsADropdownOfItsModes_OmniFirst()
        {
            foreach (var component in new ComponentBase[] { new SimulateBinauralComponent(), new SimulateAmbisonicsComponent() })
            {
                var dropdown = component.Controls.OfType<DropdownControl>().Single(d => d.Port == "late_envelope");
                Assert.Equal(new[] { "omni", "directional", "off" }, dropdown.Options);
                Assert.Equal(new[] { "Omni", "Directional", "Off" }, dropdown.Options.Select(o => ControlLayout.Label(dropdown, o)));
                Assert.Equal(("omni", false), ControlState.Of(component.ControlPort("late_envelope")!, dropdown.Options));
                Assert.Equal("Late envelope", ControlLayout.Title(dropdown));
                Assert.IsType<global::Grasshopper.Kernel.Parameters.Param_String>(component.ControlPort("late_envelope"));
            }
        }

        [Fact]
        public void TheHybridBodies_StartWithTheOrders_ThenTheTail()
        {
            Assert.Equal(new[] { "ism_order", "sh_order", "tail", "diffraction", "late_envelope", "run" },
                new SimulateAmbisonicsComponent().Controls.Select(c => c.Port));
            Assert.Equal(new[] { "ism_order", "tail", "diffraction", "late_envelope", "hrtf", "run" },
                new SimulateBinauralComponent().Controls.Select(c => c.Port));
        }

        [Fact]
        public void TogglesAndButtons_StandAsFarFromTheControlAbove_AsATitleDoes()
        {
            // Binaural: Diffraction, the toggles, HRTF..., Run, each the same gap below the one above
            var binaural = new SimulateBinauralComponent();
            var boxes = ControlLayout.Arrange(binaural.Controls, new RectangleF(0, 0, 150, 0));
            RectangleF Of(string port) => boxes.Single(b => b.Control.Port == port).Bounds;
            // Late envelope is a dropdown below Diffraction, as Diffraction is below Tail; then the toggle, HRTF..., Run
            Assert.Equal(Of("diffraction").Top - Of("tail").Bottom, Of("late_envelope").Top - Of("diffraction").Bottom, 3);
            Assert.Equal(ControlLayout.ButtonGap, Of("hrtf").Top - Of("late_envelope").Bottom, 3);
            Assert.Equal(ControlLayout.ButtonGap, Of("run").Top - Of("hrtf").Bottom, 3);
            // the gap a title leaves: the margin and the empty top of its row
            Assert.Equal(10f, ControlLayout.ButtonGap);
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
        }

        [Fact]
        public void TheEnergyMap_HasPreviewBandSourceAndRun_NoPlayback()
        {
            var map = new VisualiseEnergyMapComponent();
            Assert.Equal(new ButtonControl("run", "Run"), map.Controls.OfType<ButtonControl>().Single());
            Assert.Same(map.Controls.Last(), map.Controls.OfType<ButtonControl>().Single());
            var band = map.Controls.OfType<DropdownControl>().Single(c => c.Port == "frequency_bands");
            Assert.Equal("Frequency bands", ControlLayout.Title(band));
            Assert.Equal(new[] { "", "0", "1", "2", "3", "4", "5", "6" }, band.Options);
            Assert.Equal(new[] { "Full (A-weighted)", "125 Hz", "250 Hz", "500 Hz", "1 kHz", "2 kHz", "4 kHz", "8 kHz" }, band.Options.Select(o => ControlLayout.Label(band, o)));
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
        public void TheRaysDrawnOrder_IsAStepSliderOfTheOrders_1ToTheTracedOne_AndAll()
        {
            var rays = new VisualiseRaysComponent();
            var declared = rays.Controls.OfType<StepSliderControl>().Single();
            Assert.DoesNotContain(rays.Controls, c => c is SliderControl);
            Assert.Equal(("order", "", "Order"), (declared.Port, declared.Default, ControlLayout.Title(declared)));
            var stops = rays.StepStopsOf(declared);
            Assert.Equal(new[] { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "" }, stops.Options);
            var four = VisualiseRaysComponent.OrderStops(4);
            Assert.Equal(new[] { "1", "2", "3", "4", "" }, four.Options);
            Assert.Equal(new[] { "1", "2", "3", "4", "All" }, four.Labels);
            Assert.Equal(("order", ""), (four.Port, four.Default));
            Assert.Equal(("", false), ControlState.Of(rays.ControlPort("order")!, four.Options, four.Default));
            Assert.Equal(("order", "1, 2, 3 or 4 (default All)"), (rays.ControlTooltip("order").title, "1, 2, 3 or 4 (default All)"));
        }

        [Theory]
        [InlineData(1)]
        [InlineData(12)]
        public void TheRaysOrderStops_LabelEveryStop_UpTo12Orders(int maxOrder)
        {
            var stops = VisualiseRaysComponent.OrderStops(maxOrder);
            Assert.Equal(maxOrder + 1, stops.Options.Length);
            Assert.Equal(Enumerable.Range(1, maxOrder).Select(i => i.ToString()).Append("All"), stops.Labels);
        }

        [Theory]
        [InlineData(13, 2)]
        [InlineData(24, 2)]
        [InlineData(30, 3)]
        public void TheRaysOrderStops_ThinTheLabels_BeyondTwelveOrders_KeepingFirstAndAll(int maxOrder, int every)
        {
            var stops = VisualiseRaysComponent.OrderStops(maxOrder);
            Assert.Equal(Enumerable.Range(1, maxOrder).Select(i => i.ToString()).Append(""), stops.Options);
            var labelled = stops.Labels!.Select((l, i) => (l, n: i + 1)).Where(x => x.l.Length > 0 && x.l != "All").Select(x => x.n).ToList();
            Assert.Equal(1, labelled[0]);
            Assert.All(labelled, n => Assert.Equal(0, (n - 1) % every));
            Assert.True(labelled.Count <= 12);
            Assert.True(maxOrder - labelled[^1] >= 1);
            Assert.Equal("All", stops.Labels![^1]);
            Assert.All(stops.Labels.Select((l, i) => (l, i)).Where(x => x.l.Length > 0 && x.l != "All"), x => Assert.Equal((x.i + 1).ToString(), x.l));
        }

        [Fact]
        public void ChoosingAnOrderStop_SetsTheOrderInput_AndAllClearsIt()
        {
            var rays = new VisualiseRaysComponent();
            rays.ShowControlPorts(false);
            var stops = VisualiseRaysComponent.OrderStops(4);
            Assert.Null(rays.HiddenAs<int>("order"));
            ComponentUi.ChooseStop(rays, stops, "2");
            Assert.Equal(("2", false), ComponentUi.StepValue(rays, stops));
            Assert.Equal(2, rays.HiddenAs<int>("order"));
            ComponentUi.ChooseStop(rays, stops, "");
            Assert.Null(rays.HiddenAs<int>("order"));
            Assert.Equal(("", false), ComponentUi.StepValue(rays, stops));
        }

        [Fact]
        public void AStepSlidersTrack_SpansTheRow_LeavingRoomForTheEndLabelsAndTheKnob()
        {
            var ism = SimulateMonoComponent.IsmOrder;
            var row = new RectangleF(0, 0, 120, StepSlider.Height);
            var room = Math.Max(StepSlider.CharWidth * 2 / 2, StepSlider.DotRadius) + 1;
            Assert.InRange(StepSlider.StopX(row, 0, ism), room - 0.01f, room);
            Assert.InRange(row.Right - StepSlider.StopX(row, 10, ism), room - 0.01f, room);
            Assert.True(StepSlider.StopX(row, 0, ism) < StepSlider.StopX(row, 1, ism));
            Assert.True(room <= 7);
            var rays = VisualiseRaysComponent.OrderStops(4);
            Assert.InRange(row.Right - StepSlider.StopX(row, 4, rays), 0, Math.Max(StepSlider.CharWidth * 3 / 2, StepSlider.DotRadius) + 1);
            Assert.Equal("", StepSlider.Nearest(row, row.Right, rays, null));
        }

        [Fact]
        public void AnOrderBeyondTheTracedOnes_ShowsTheLastOrder()
        {
            var row = new RectangleF(0, 0, 120, StepSlider.Height);
            var stops = VisualiseRaysComponent.OrderStops(4);
            Assert.Equal("4", StepSlider.Shown(row, "7", stops, null));
            Assert.Equal("", StepSlider.Shown(row, "", stops, null));
            Assert.Equal("3", StepSlider.Shown(row, "3", stops, null));
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

        [Theory]
        [InlineData(typeof(VisualiseRaysComponent))]
        [InlineData(typeof(VisualiseBilliardComponent))]
        public void PlayAndRewind_AreIconButtonsSideBySide_SetApartFromTheControlsAbove(Type type)
        {
            var component = (ComponentBase)Activator.CreateInstance(type)!;
            var buttons = component.Controls.OfType<ButtonControl>().Where(b => b.Port is "play" or "rewind").ToList();
            Assert.Equal(new[] { "play", "rewind" }, buttons.Select(b => b.Port));
            Assert.All(buttons, b => Assert.True(b.Inline && b.Icon));
            var boxes = ControlLayout.Arrange(component.Controls, new RectangleF(0, 0, 150, 0));
            var (play, rewind) = (boxes.Single(b => b.Control.Port == "play"), boxes.Single(b => b.Control.Port == "rewind"));
            Assert.Equal(play.Bounds.Top, rewind.Bounds.Top);
            Assert.Equal(play.Bounds.Width, rewind.Bounds.Width, 3);
            Assert.True(play.Bounds.Right <= rewind.Bounds.Left);
            var above = boxes[boxes.IndexOf(play) - 1];
            Assert.Equal(ControlLayout.ButtonGap, play.Bounds.Top - above.Bounds.Bottom, 3);
        }

        [Fact]
        public void AnIconButton_DrawsTheGlyphOfWhatItReads()
        {
            // Play reads Stop while playing: its glyph follows
            Assert.Equal(Glyph.Play, ControlLayout.GlyphOf("Play"));
            Assert.Equal(Glyph.Stop, ControlLayout.GlyphOf("Stop"));
            Assert.Equal(Glyph.Rewind, ControlLayout.GlyphOf("Rewind"));
            Assert.Equal(Glyph.None, ControlLayout.GlyphOf("Save"));
            Assert.Equal(("Rewind", ""), new VisualiseRaysComponent().ControlTooltip("rewind"));
        }

        [Fact]
        public void TheBilliardsTime_FollowsTheKnobWhileItIsDragged()
        {
            var billiard = new VisualiseBilliardComponent();
            var time = billiard.Controls.OfType<SliderControl>().Single();
            Assert.True(time.Live);
            // each step of the drag sets the input and lets go of what it collected, without an undo step of its own
            var port = billiard.ControlPort("time")!;
            ControlState.Scrub(billiard, port, "12.5");
            port.CollectData();
            ControlState.Scrub(billiard, port, "40");
            Assert.Equal(0, port.VolatileDataCount);
            Assert.Equal(40.0, ControlState.Persistent(port));
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
        public void TheHrtfSetChosenOnTheBody_IsUsedWhileItsInputIsHidden()
        {
            var binaural = new SimulateBinauralComponent();
            binaural.ShowControlPorts(false);
            Assert.DoesNotContain(binaural.Params.Input, p => p.Name == "hrtf");
            Assert.Null(binaural.Hrtf(null!));
            ControlState.Choose(binaural, binaural.ControlPort("hrtf")!, "/sets/my_head.sofa");
            // a hidden input is read from the body, without the solution's data access
            Assert.Equal("/sets/my_head.sofa", binaural.Hrtf(null!));
            Assert.Null(new SimulateAmbisonicsComponent().Hrtf(null!));
        }
    }
}
