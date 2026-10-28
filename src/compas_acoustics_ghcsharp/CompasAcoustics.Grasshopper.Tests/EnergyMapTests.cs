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

        private static double Area(Point3d[] v, int[][] faces) => faces.Sum(f => Enumerable.Range(1, f.Length - 2)
            .Sum(i => Vector3d.CrossProduct(v[f[i]] - v[f[0]], v[f[i + 1]] - v[f[0]]).Length / 2));

        private static double LongestEdge(Point3d[] v, int[][] faces) =>
            faces.SelectMany(f => f.Select((c, i) => v[c].DistanceTo(v[f[(i + 1) % f.Length]]))).Max();

        [Fact]
        public void AQuad_IsSplitIntoAGrid_NoEdgeLongerThanTheSpacing()
        {
            var v = new[] { new Point3d(0, 0, 1.2), new Point3d(2, 0, 1.2), new Point3d(2, 1, 1.2), new Point3d(0, 1, 1.2) };
            var (corners, faces) = EnergyMapColours.Subdivide(v, new[] { new[] { 0, 1, 2, 3 } }, 0.5);
            Assert.Equal(8, faces.Length);
            Assert.All(faces, f => Assert.Equal(4, f.Length));
            Assert.True(LongestEdge(corners, faces) <= 0.5 + 1e-9);
            Assert.Equal(2.0, Area(corners, faces), 9);
            Assert.All(corners, c => Assert.Equal(1.2, c.Z, 9));
        }

        [Fact]
        public void ATriangle_IsSplitIntoTriangles_NoEdgeLongerThanTheSpacing()
        {
            var v = new[] { new Point3d(0, 0, 0), new Point3d(1, 0, 0), new Point3d(0, 1, 0) };
            var (corners, faces) = EnergyMapColours.Subdivide(v, new[] { new[] { 0, 1, 2 } }, 0.25);
            // the hypotenuse, 1.41, takes 6 steps: 36 triangles
            Assert.Equal(36, faces.Length);
            Assert.All(faces, f => Assert.Equal(3, f.Length));
            Assert.True(LongestEdge(corners, faces) <= 0.25 + 1e-9);
            Assert.Equal(0.5, Area(corners, faces), 9);
        }

        [Fact]
        public void TheMap_IsLiftedOffItsSurface_UpwardOrAlongAWallsNormal()
        {
            // drawn on the floor it came from, it would flicker with it: a centimetre above, whichever way the face is wound
            var floorDown = new[] { new Point3d(0, 0, 0), new Point3d(0, 1, 0), new Point3d(1, 1, 0), new Point3d(1, 0, 0) };
            var floorUp = floorDown.Reverse().ToArray();
            var wall = new[] { new Point3d(0, 0, 0), new Point3d(1, 0, 0), new Point3d(1, 0, 1), new Point3d(0, 0, 1) };
            Assert.Equal(new Vector3d(0, 0, EnergyMapColours.Lift), EnergyMapColours.Offset(floorDown));
            Assert.Equal(new Vector3d(0, 0, EnergyMapColours.Lift), EnergyMapColours.Offset(floorUp));
            Assert.Equal(0.01, EnergyMapColours.Lift);
            var side = EnergyMapColours.Offset(wall);
            Assert.Equal(EnergyMapColours.Lift, side.Length, 9);
            Assert.Equal(0.0, side.Z, 9);
        }

        [Fact]
        public void AFaceSmallerThanTheSpacing_StaysWhole()
        {
            var v = new[] { new Point3d(0, 0, 0), new Point3d(0.3, 0, 0), new Point3d(0.3, 0.3, 0), new Point3d(0, 0.3, 0) };
            var (corners, faces) = EnergyMapColours.Subdivide(v, new[] { new[] { 0, 1, 2, 3 } }, 0.5);
            Assert.Single(faces);
            Assert.Equal(v, faces[0].Select(i => corners[i]));
        }

        [Fact]
        public void TheComponent_TakesASurfaceAndItsSpacing_AndChoosesTheRestItself()
        {
            // the rays, the image source order and the seed are fixed: a map has no setting a user should tune but its grid
            var spec = ComponentSpec.Load("CA_Visualise_Energy_Map");
            Assert.Equal(new[] { "scene", "mesh", "spacing", "source_idx", "frequency_bands", "range" }, spec.Inputs.Select(p => p.Name));
            // a fine map: the rays keep their 1 m grid, the image sources and the interpolation take the points
            Assert.Equal(0.25, spec.Inputs.Single(p => p.Name == "spacing").Default);
            Assert.Equal(new[] { "mesh", "values", "colours" }, spec.Outputs.Select(p => p.Name));
            Assert.Equal("05 Display", spec.Subcategory);
        }
    }
}
