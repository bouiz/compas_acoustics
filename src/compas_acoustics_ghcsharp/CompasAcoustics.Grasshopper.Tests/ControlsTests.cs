using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class ControlsTests
    {
        private static readonly RectangleF Body = new(100, 200, 120, 0);

        [Fact]
        public void Arrange_GivesOneRowPerControlAndEqualSegments()
        {
            var controls = new Control[]
            {
                new SwitchControl("output", new[] { "binaural", "ambisonic", "both" }),
                new DropdownControl("tail", new[] { "auto", "reference" }),
                new ButtonControl("run", "Run"),
            };
            var boxes = ControlLayout.Arrange(controls, Body);
            Assert.Equal(3, boxes.Count);
            Assert.True(boxes[0].Bounds.Bottom <= boxes[1].Bounds.Top && boxes[1].Bounds.Bottom <= boxes[2].Bounds.Top);
            Assert.All(boxes, b => Assert.InRange(b.Bounds.Left, Body.Left, Body.Left + ControlLayout.Margin));
            var segments = boxes[0].Segments;
            Assert.Equal(3, segments.Length);
            Assert.Equal(segments[0].Width, segments[2].Width, 3);
            Assert.Equal(boxes[0].Bounds.Width, segments.Sum(s => s.Width), 3);
            Assert.Empty(boxes[1].Segments);
            // a switch and a dropdown have their title above them; a button carries its own label
            Assert.True(boxes[0].Title.Bottom <= boxes[0].Bounds.Top && boxes[0].Title.Height == ControlLayout.TitleHeight);
            Assert.True(boxes[0].Bounds.Bottom <= boxes[1].Title.Top);
            Assert.True(boxes[2].Title.IsEmpty);
        }

        [Fact]
        public void Height_IsRowsTimesRowHeightPlusMargins()
        {
            var controls = new Control[] { new ButtonControl("run", "Run"), new SwitchControl("level", new[] { "a", "b" }) };
            Assert.Equal(2 * ControlLayout.RowHeight + ControlLayout.TitleHeight + ControlLayout.TitleGap + 3 * ControlLayout.Margin, ControlLayout.Height(controls));
            Assert.Equal(0, ControlLayout.Height(new Control[0]));
        }

        private static readonly Control[] Hybrid =
        {
            new DropdownControl("tail", new[] { "auto", "reference" }),
            new SwitchControl("output", new[] { "binaural", "ambisonic", "both" }),
            new ButtonControl("run", "Run"),
        };

        [Fact]
        public void ATitle_SitsALittleAboveItsControl()
        {
            var boxes = ControlLayout.Arrange(Hybrid, Body);
            Assert.True(ControlLayout.TitleGap >= 2);
            Assert.All(boxes.Take(2), b => Assert.Equal(ControlLayout.TitleGap, b.Bounds.Top - b.Title.Bottom, 3));
        }

        [Fact]
        public void AButtonAfterATitledControl_StandsApartAsTheBodyFromThePorts()
        {
            var boxes = ControlLayout.Arrange(Hybrid, Body);
            Assert.True(ControlLayout.ButtonGap > ControlLayout.Margin);
            Assert.Equal(ControlLayout.ButtonGap, boxes[2].Bounds.Top - boxes[1].Bounds.Bottom, 3);
            // between two titled controls the usual margin
            Assert.Equal(ControlLayout.Margin, boxes[1].Title.Top - boxes[0].Bounds.Bottom, 3);
            Assert.Equal(boxes[2].Bounds.Bottom + ControlLayout.Margin - Body.Top, ControlLayout.Height(Hybrid), 3);
        }

        [Fact]
        public void HitTest_FindsTheControlAndSegmentUnderAPoint()
        {
            var controls = new Control[] { new SwitchControl("output", new[] { "binaural", "ambisonic", "both" }) };
            var boxes = ControlLayout.Arrange(controls, Body);
            var middle = boxes[0].Segments[1];
            var hit = ControlLayout.HitTest(boxes, new PointF(middle.X + middle.Width / 2, middle.Y + middle.Height / 2));
            Assert.NotNull(hit);
            Assert.Equal("output", hit!.Value.box.Control.Port);
            Assert.Equal(1, hit.Value.segment);
            Assert.Null(ControlLayout.HitTest(boxes, new PointF(0, 0)));
        }

        [Fact]
        public void Resolve_WiredValueWins()
        {
            Assert.Equal(("ambisonic", true), ControlState.Resolve(1, "ambisonic", "binaural", new[] { "binaural", "ambisonic", "both" }));
        }

        [Fact]
        public void Resolve_UnwiredUsesThePersistentValueOrTheFirstOption()
        {
            var options = new[] { "binaural", "ambisonic", "both" };
            Assert.Equal(("both", false), ControlState.Resolve(0, null, "both", options));
            Assert.Equal(("binaural", false), ControlState.Resolve(0, null, null, options));
            Assert.Equal(("binaural", false), ControlState.Resolve(0, null, "", options));
        }

        [Fact]
        public void Of_ReadsThePortsPersistentData()
        {
            var port = new Param_String();
            port.PersistentData.Append(new GH_String("linear"));
            Assert.Equal(("linear", false), ControlState.Of(port, new[] { "db", "linear" }));
        }

        [Fact]
        public void Choice_RoundTripsThroughThePortsPersistentData()
        {
            var binaural = new SimulateBinauralComponent();
            var port = binaural.Params.Input.Single(p => p.Name == "tail");
            ControlState.Choose(binaural, port, "eyring");
            var chunk = new GH_LooseChunk("component");
            Assert.True(binaural.Write(chunk));
            var read = new GH_LooseChunk("component");
            read.Deserialize_Binary(chunk.Serialize_Binary());
            var back = new SimulateBinauralComponent();
            Assert.True(back.Read(read));
            Assert.Equal(("eyring", false), ControlState.Of(back.Params.Input.Single(p => p.Name == "tail"), HybridRun.Tails));
        }

        [Fact]
        public void Press_IsReadOnceThenCleared()
        {
            var hybrid = new SimulateAmbisonicsComponent();
            ControlState.Press(hybrid, "run");
            Assert.True(hybrid.Pressed("run"));
            Assert.False(hybrid.Pressed("play"));
            hybrid.ClearPresses();
            Assert.False(hybrid.Pressed("run"));
        }

        public static IEnumerable<object[]> FormerInputObjects() => new[]
        {
            new object[] { typeof(SimulateBinauralComponent), new[] { "tail", "diffraction", "run" } },
            new object[] { typeof(SimulateAmbisonicsComponent), new[] { "tail", "diffraction", "run" } },
            new object[] { typeof(AuraliseComponent), new[] { "level" } },
            new object[] { typeof(DeconstructParameterComponent), new[] { "select" } },
            new object[] { typeof(AcousticObjectComponent), new[] { "role" } },
            new object[] { typeof(VisualiseBilliardComponent), new[] { "play" } },
        };

        [Theory]
        [MemberData(nameof(FormerInputObjects))]
        public void Components_DeclareControlsForEveryFormerValueListAndButton(System.Type type, string[] ports)
        {
            var component = (ComponentBase)System.Activator.CreateInstance(type)!;
            Assert.Equal(ports, component.Controls.Select(c => c.Port).Where(ports.Contains).ToArray());
            // a switch or dropdown sets an input, or a choice the component holds (Auralise's level, Acoustic Object's role);
            // a button may stand alone, pressed for one solution (Cancel, Billiard's Stop)
            Assert.All(component.Controls.Where(c => c is not ButtonControl),
                c => Assert.True(component.Params.Input.Any(p => p.Name == c.Port) || component.IsChoice(c.Port), c.Port));
        }

        [Fact]
        public void ADropdown_ShowsItsChoice_UnderItsTitle()
        {
            // the dropdown shows its value; its input's name is the title above it
            Assert.Equal("auto", ControlLayout.Text(new DropdownControl("tail", HybridRun.Tails), "auto"));
            Assert.Equal("Run", ControlLayout.Text(new ButtonControl("run", "Run"), "Run"));
            Assert.Equal("Tail", ControlLayout.Title(new DropdownControl("tail", HybridRun.Tails)));
            Assert.Equal("Role", ControlLayout.Title(new SwitchControl("role", new[] { "boundary", "obstacle" })));
            Assert.Equal("Time scale", ControlLayout.Title(new SwitchControl("time_scale", new[] { "db" })));
        }

        [Fact]
        public void AControlsTooltip_IsItsInputsDescription()
        {
            var hybrid = new SimulateBinauralComponent();
            var (title, text) = hybrid.ControlTooltip("diffraction");
            Assert.Equal("diffraction", title);
            Assert.Equal(ComponentSpec.Load("CA_Simulate_Binaural").Inputs.Single(p => p.Name == "diffraction").Description, text);
            // a choice without an input: its options, and its default
            Assert.Equal(("level", "none, relative or individual (default relative)"), new AuraliseComponent().ControlTooltip("level"));
        }

        [Fact]
        public void ChoiceControls_OfferTheComponentsOptions()
        {
            foreach (var hybrid in new SimulateHybridComponent[] { new SimulateBinauralComponent(), new SimulateAmbisonicsComponent() })
            {
                Assert.Empty(hybrid.Controls.OfType<SwitchControl>());
                Assert.Equal(HybridRun.Tails, hybrid.Controls.OfType<DropdownControl>().Single(c => c.Port == "tail").Options);
                Assert.Equal(HybridRun.Diffraction, hybrid.Controls.OfType<DropdownControl>().Single(c => c.Port == "diffraction").Options);
            }
        }
    }
}
