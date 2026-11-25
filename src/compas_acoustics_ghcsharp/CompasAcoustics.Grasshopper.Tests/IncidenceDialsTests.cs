using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>Polar Preview's incidence dials: which directions they offer, which tick a click picks, and the direction it makes.</summary>
    public class IncidenceDialsTests
    {
        private static readonly RectangleF Row = new(0, 0, 170, IncidenceDials.Height);

        /// <summary>The directions of a survey result: the normal, then 30° and 60° at every 60° of azimuth.</summary>
        private static List<Incidence> Survey()
        {
            var directions = new List<Incidence> { new(0, 0, 0) };
            foreach (var theta in new[] { 30.0, 60.0 })
                for (var phi = 0; phi < 360; phi += 60)
                    directions.Add(new Incidence(directions.Count, theta, phi));
            return directions;
        }

        [Fact]
        public void Rings_AreTheColatitudes_AndARingItsAzimuths()
        {
            var survey = Survey();
            Assert.Equal(new[] { 0.0, 30.0, 60.0 }, IncidenceDials.Rings(survey));
            Assert.Equal(new[] { 0.0, 60.0, 120.0, 180.0, 240.0, 300.0 }, IncidenceDials.Ring(survey, 30).Select(d => d.Phi));
            Assert.Single(IncidenceDials.Ring(survey, 0));
        }

        [Fact]
        public void AClickOnATick_PicksIt_AndNothingElsewhere()
        {
            var survey = Survey();
            var current = survey[3]; // θ 30°, φ 120°
            Assert.Equal((30.0, (double?)null), IncidenceDials.Hit(Row, IncidenceDials.SectionTick(Row, 30), survey, current));
            Assert.Equal((60.0, (double?)null), IncidenceDials.Hit(Row, IncidenceDials.SectionTick(Row, 60), survey, current));
            Assert.Equal(((double?)null, 240.0), IncidenceDials.Hit(Row, IncidenceDials.PlanTick(Row, 240), survey, current));
            // a few pixels off still picks the tick; the middle of the dials picks nothing
            var near = IncidenceDials.SectionTick(Row, 0);
            Assert.Equal((0.0, (double?)null), IncidenceDials.Hit(Row, new PointF(near.X + 3, near.Y + 2), survey, current));
            Assert.Equal(((double?)null, (double?)null), IncidenceDials.Hit(Row, new PointF(Row.Width / 2, Row.Height - 2), survey, current));
        }

        [Fact]
        public void AtTheNormal_ThePlanDialPicksNothing()
        {
            var survey = Survey();
            Assert.Equal(((double?)null, (double?)null), IncidenceDials.Hit(Row, IncidenceDials.PlanTick(Row, 60), survey, survey[0]));
        }

        [Fact]
        public void AColatitude_KeepsTheAzimuth_AndAnAzimuthMovesOnTheRing()
        {
            var survey = Survey();
            var current = survey[3]; // θ 30°, φ 120°
            Assert.Equal(new Incidence(9, 60, 120), IncidenceDials.Choose(survey, current, 60, null, null));
            Assert.Equal(new Incidence(5, 30, 240), IncidenceDials.Choose(survey, current, null, 240, null));
            // the normal has one direction; back off it, the azimuth chosen before it returns
            Assert.Equal(survey[0], IncidenceDials.Choose(survey, current, 0, null, null));
            Assert.Equal(new Incidence(3, 30, 120), IncidenceDials.Choose(survey, survey[0], 30, null, 120));
        }

        [Fact]
        public void AnAzimuthTheRingDoesNotHave_GivesTheNearest()
        {
            // a standard result has every 30°, survey every 60°: φ 320° on a survey ring is 300°, across 0° if need be
            var survey = Survey();
            Assert.Equal(300.0, IncidenceDials.Choose(survey, survey[0], 30, null, 320)!.Phi);
            Assert.Equal(0.0, IncidenceDials.Choose(survey, survey[0], 30, null, 20)!.Phi);
            // halfway between two (90° between 60° and 120°), the lower one
            Assert.Equal(60.0, IncidenceDials.Choose(survey, survey[0], 30, null, 90)!.Phi);
            Assert.Null(IncidenceDials.Choose(new List<Incidence>(), null, 30, null, null));
        }

        [Fact]
        public void TheDials_TakeTheirOwnRow_AndWidenTheComponent()
        {
            var controls = new Control[] { new IncidenceControl("preview", "Incidence sound"), new DropdownControl("octave", new[] { "1000" }) };
            var boxes = ControlLayout.Arrange(controls, new RectangleF(0, 0, 180, 0));
            Assert.Equal(IncidenceDials.Height, boxes[0].Bounds.Height);
            Assert.Equal("Incidence sound", ControlLayout.Title(controls[0]));
            Assert.True(boxes[1].Bounds.Top > boxes[0].Bounds.Bottom);
            Assert.Equal(IncidenceDials.MinWidth, ControlLayout.MinWidth(controls));
            Assert.Equal(0, ControlLayout.MinWidth(new Control[] { controls[1] }));
        }

        private static readonly string[] Octaves = { "125", "250", "500", "1000", "2000", "4000", "8000" };
        private static readonly string[] Simulated = { "125", "250", "500", "1000", "2000", "4000" }; // the ISO range: no 8000 Hz

        [Fact]
        public void TheOctaveSlider_StopsOnlyAtSimulatedOctaves()
        {
            var row = new RectangleF(0, 0, 170, StepSlider.Height);
            Assert.Equal("1000", StepSlider.Nearest(row, StepSlider.StopX(row, 3, 7) + 4, Octaves, Simulated));
            // a drag to the far right stops at 4000 Hz: 8000 Hz was not simulated
            Assert.Equal("4000", StepSlider.Nearest(row, row.Right + 20, Octaves, Simulated));
            Assert.Equal("8000", StepSlider.Nearest(row, row.Right, Octaves, null));
            // a definition saved at 8000 Hz shows the knob at 4000 Hz
            Assert.Equal("4000", StepSlider.Shown(row, "8000", Octaves, Simulated));
            Assert.Equal("2000", StepSlider.Shown(row, "2000", Octaves, Simulated));
            Assert.True(StepSlider.StopX(row, 0, 7) < StepSlider.StopX(row, 6, 7));
        }

        [Fact]
        public void PolarPreview_DrawsTheNearestSimulatedOctave_UnlessAWireSetsIt()
        {
            var result = new CompasAcoustics.Pb.ScatteringResultData();
            result.OctaveBands.AddRange(new[] { 125, 250, 500, 1000, 2000, 4000 });
            Assert.Equal(4000, LabRequests.Octave(8000, result, wired: false));
            Assert.Equal(1000, LabRequests.Octave(1000, result, wired: false));
            Assert.Equal(8000, LabRequests.Octave(8000, result, wired: true)); // Balloon then names the octaves there are
        }

        [Fact]
        public void TheOctaveSlider_TakesItsOwnRow()
        {
            var controls = new Control[] { new StepSliderControl("octave", Octaves, "1000", null, "Octave") };
            Assert.Equal(StepSlider.Height, ControlLayout.Arrange(controls, new RectangleF(0, 0, 180, 0))[0].Bounds.Height);
            Assert.Equal("Octave", ControlLayout.Title(controls[0]));
        }

        [Fact]
        public void AResultsDirections_AreItsPreviewsIndicesAndAngles()
        {
            var result = InteropTests.Serializer.Unpack<CompasAcoustics.Pb.ScatteringResultData>(InteropTests.Fixture("scattering_result"))!;
            var directions = LabRequests.Incidences(result);
            Assert.Equal(Enumerable.Range(0, result.PolarSourceIndices.Count), directions.Select(d => d.Index));
            var last = result.PolarSourceIndices[^1];
            Assert.Equal((result.SourceColatitude[last], result.SourceAzimuth[last]), (directions[^1].Theta, directions[^1].Phi));
            Assert.Empty(new PolarPreviewComponent().IncidencesOf("preview")); // until it has solved
        }
    }
}
