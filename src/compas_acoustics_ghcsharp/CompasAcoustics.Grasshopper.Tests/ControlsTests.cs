using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Grasshopper.Kernel;
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
                new SwitchControl("level", new[] { "none", "relative", "individual" }),
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

        private static readonly Control[] TitledThenButton =
        {
            new DropdownControl("tail", new[] { "auto", "reference" }),
            new SwitchControl("level", new[] { "none", "relative", "individual" }),
            new ButtonControl("run", "Run"),
        };

        [Fact]
        public void AButtonAfterATitledControl_StandsApartAsTheBodyFromThePorts()
        {
            var boxes = ControlLayout.Arrange(TitledThenButton, Body);
            Assert.True(ControlLayout.TitleGap >= 2);
            Assert.All(boxes.Take(2), b => Assert.Equal(ControlLayout.TitleGap, b.Bounds.Top - b.Title.Bottom, 3));
            Assert.True(ControlLayout.ButtonGap > ControlLayout.Margin);
            Assert.Equal(ControlLayout.ButtonGap, boxes[2].Bounds.Top - boxes[1].Bounds.Bottom, 3);
            // between two titled controls the usual margin
            Assert.Equal(ControlLayout.Margin, boxes[1].Title.Top - boxes[0].Bounds.Bottom, 3);
            Assert.Equal(boxes[2].Bounds.Bottom + ControlLayout.Margin - Body.Top, ControlLayout.Height(TitledThenButton), 3);
        }

        [Fact]
        public void HitTest_FindsTheControlAndSegmentUnderAPoint()
        {
            var controls = new Control[] { new SwitchControl("level", new[] { "none", "relative", "individual" }) };
            var boxes = ControlLayout.Arrange(controls, Body);
            var middle = boxes[0].Segments[1];
            var hit = ControlLayout.HitTest(boxes, new PointF(middle.X + middle.Width / 2, middle.Y + middle.Height / 2));
            Assert.NotNull(hit);
            Assert.Equal("level", hit!.Value.box.Control.Port);
            Assert.Equal(1, hit.Value.segment);
            Assert.Null(ControlLayout.HitTest(boxes, new PointF(0, 0)));
        }

        [Fact]
        public void Resolve_WiredValueWins()
        {
            Assert.Equal(("relative", true), ControlState.Resolve(1, "relative", "none", new[] { "none", "relative", "individual" }));
        }

        [Fact]
        public void Resolve_UnwiredUsesThePersistentValueOrTheFirstOption()
        {
            var options = new[] { "none", "relative", "individual" };
            Assert.Equal(("individual", false), ControlState.Resolve(0, null, "individual", options));
            Assert.Equal(("none", false), ControlState.Resolve(0, null, null, options));
            Assert.Equal(("none", false), ControlState.Resolve(0, null, "", options));
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

        public static IEnumerable<object[]> ControlledInputs() => new[]
        {
            new object[] { typeof(SimulateBinauralComponent), new[] { "tail", "diffraction", "run" } },
            new object[] { typeof(SimulateAmbisonicsComponent), new[] { "tail", "diffraction", "run" } },
            new object[] { typeof(AuraliseComponent), new[] { "level" } },
            new object[] { typeof(Plot2DComponent), new[] { "select" } },
            new object[] { typeof(AcousticObjectComponent), new[] { "role" } },
            new object[] { typeof(VisualiseBilliardComponent), new[] { "play" } },
        };

        [Theory]
        [MemberData(nameof(ControlledInputs))]
        public void Components_DeclareTheirBodyControls(System.Type type, string[] ports)
        {
            var component = (ComponentBase)System.Activator.CreateInstance(type)!;
            Assert.Equal(ports, component.Controls.Select(c => c.Port).Where(ports.Contains).ToArray());
            // a switch or dropdown sets an input, or a choice the component holds (Auralise's level, Acoustic Object's role);
            // a button may stand alone, pressed for one solution (Cancel, Billiard's Stop)
            Assert.All(component.Controls.Where(c => c is not ButtonControl),
                c => Assert.True(component.Params.Input.Any(p => p.Name == c.Port) || component.IsChoice(c.Port), c.Port));
        }

        [Fact]
        public void ADropdownsLabels_StartWithACapital_AndItsValuesKeepTheirForm()
        {
            var tail = new DropdownControl("tail", new[] { "auto", "ray_tracing" });
            var diffraction = new DropdownControl("diffraction", new[] { "none", "udfa", "btms" });
            Assert.Equal("Ray tracing", ControlLayout.Label(tail, "ray_tracing"));
            Assert.Equal(new[] { "None", "UDFA", "BTMS" }, diffraction.Options.Select(o => ControlLayout.Label(diffraction, o)));
            Assert.Equal(new[] { "none", "udfa", "btms" }, diffraction.Options);
            Assert.Equal("c:/x.wav", ControlLayout.Label(tail, "c:/x.wav"));
            Assert.Equal(("level", "Boundary"), ("level", ControlLayout.Label(new SwitchControl("role", new[] { "boundary", "obstacle" }), "boundary")));
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
