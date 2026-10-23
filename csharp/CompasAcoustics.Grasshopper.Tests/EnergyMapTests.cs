using System.Drawing;
using System.Linq;
using Rhino.Geometry;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class EnergyMapTests
    {
        [Fact]
        public void TheScale_RunsFromBlueAtTheLowestToRedAtTheHighest()
        {
            Assert.Equal(EnergyMapColours.Of(0.0, 0.0, 1.0), EnergyMapColours.Of(-5.0, 0.0, 1.0));
            var low = EnergyMapColours.Of(50.0, 50.0, 70.0);
            var high = EnergyMapColours.Of(70.0, 50.0, 70.0);
            Assert.True(low.B > low.R && low.B > low.G);
            Assert.True(high.R > high.B && high.R > high.G);
            Assert.Equal(high, EnergyMapColours.Of(90.0, 50.0, 70.0));
        }

        [Fact]
        public void AFaceOutsideTheRoom_IsGrey_AndBeforeAnyEnergy_IsTheLowestColour()
        {
            Assert.Equal(EnergyMapColours.Outside, EnergyMapColours.Of(double.NaN, 50.0, 70.0));
            Assert.Equal(EnergyMapColours.Of(50.0, 50.0, 70.0), EnergyMapColours.Of(double.NegativeInfinity, 50.0, 70.0));
        }

        [Fact]
        public void TheRange_IsTheFiniteValuesOrTheGivenInterval()
        {
            var values = new[] { double.NaN, 55.0, 70.5, double.NegativeInfinity, 61.0 };
            Assert.Equal((55.0, 70.5), EnergyMapColours.Range(values, null));
            Assert.Equal((40.0, 80.0), EnergyMapColours.Range(values, new Interval(80.0, 40.0)));
            Assert.Equal((0.0, 0.0), EnergyMapColours.Range(new[] { double.NaN }, null));
        }

        [Fact]
        public void TheFaceCentres_AreTheMeansOfTheirVertices()
        {
            var vertices = new[] { new Point3d(0, 0, 1), new Point3d(2, 0, 1), new Point3d(2, 2, 1), new Point3d(0, 2, 1), new Point3d(4, 0, 1) };
            var faces = new[] { new[] { 0, 1, 2, 3 }, new[] { 1, 4, 2 } };
            var centres = EnergyMapColours.FaceCentres(vertices, faces);
            Assert.Equal(new Point3d(1, 1, 1), centres[0]);
            Assert.Equal(new Point3d(8 / 3.0, 2 / 3.0, 1), centres[1]);
        }

        [Fact]
        public void TheComponent_TakesAMeshAndColoursIt()
        {
            var spec = ComponentSpec.Load("CA_Visualise_Energy_Map");
            Assert.Equal(new[] { "scene", "mesh", "source_idx", "n_rays", "image_source_order", "band", "range", "seed" }, spec.Inputs.Select(p => p.Name));
            Assert.Equal(new[] { "mesh", "values", "colours" }, spec.Outputs.Select(p => p.Name));
            Assert.Equal("05 Display", spec.Subcategory);
        }
    }
}
