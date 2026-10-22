using System.Linq;
using CompasAcoustics.Pb;
using CompasPb.Data;
using Rhino.Geometry;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class RayGeometryTests
    {
        private static RayFieldData Field()
        {
            CompasPbRegistration.Ensure();
            return InteropTests.Serializer.Unpack<RayFieldData>(InteropTests.Fixture("rayfield"))!;
        }

        [Fact]
        public void UpTo_CutsARayWhereTheSoundHasTravelled()
        {
            var ray = new RayGeometry.Ray(new[] { new Point3d(0, 0, 0), new Point3d(1, 0, 0), new Point3d(1, 1, 0) }, 0, 0, false);
            Assert.Equal(new[] { new Point3d(0, 0, 0), new Point3d(1, 0, 0), new Point3d(1, 0.5, 0) }, RayGeometry.UpTo(ray, 1.5).Points);
            Assert.Equal(ray.Points, RayGeometry.UpTo(ray, 10).Points);
            Assert.Empty(RayGeometry.UpTo(ray, -1).Points);
            // a scattered ray emerges at its offset: nothing of it before
            var scattered = ray with { DistanceOffset = 2.0, Scattered = true, BirthOrder = 1 };
            Assert.Empty(RayGeometry.UpTo(scattered, 1.9).Points);
            var cut = RayGeometry.UpTo(scattered, 2.5);
            Assert.Equal(new[] { new Point3d(0, 0, 0), new Point3d(0.5, 0, 0) }, cut.Points);
            Assert.Equal((2.0, 1, true), (cut.DistanceOffset, cut.BirthOrder, cut.Scattered));
        }

        [Fact]
        public void FromField_GivesSpecularThenScatteredRays()
        {
            var rays = RayGeometry.FromField(Field());
            Assert.Equal(2, rays.Count);
            Assert.False(rays[0].Scattered);
            Assert.True(rays[1].Scattered);
            Assert.Equal(new Point3d(3, 4, 1), rays[0].Points[^1]);
            Assert.Single(RayGeometry.Specular(Field()));
        }

        [Fact]
        public void CumulativeDistances_StartAtTheOffset()
        {
            var rays = RayGeometry.FromField(Field());
            Assert.Equal(new[] { 0.0, 3.0, 7.0 }, rays[0].CumulativeDistances);
            Assert.Equal(new[] { 3.0, 5.0 }, rays[1].CumulativeDistances);
            Assert.Equal(7.0, rays[0].TotalDistance);
        }

        [Fact]
        public void Segments_CountOrdersFromTheBirthOrder()
        {
            var rays = RayGeometry.FromField(Field());
            Assert.Equal(new[] { 0, 1 }, RayGeometry.Segments(rays[0]).Select(s => s.order).ToArray());
            Assert.Equal(new[] { 1 }, RayGeometry.Segments(rays[1]).Select(s => s.order).ToArray());
            Assert.Equal(new Line(new Point3d(3, 0, 1), new Point3d(3, 0, 3)), RayGeometry.Segments(rays[1]).Single().segment);
        }

        [Fact]
        public void TravelTime_IsTotalDistanceOverSpeedInMs()
        {
            Assert.Equal(7.0 / 343.0 * 1000.0, RayGeometry.FromField(Field())[0].TravelTimeMs(343.0), 9);
        }

        [Fact]
        public void FromReflections_GivesTheirOrders()
        {
            var found = new SpecularReflectionsData();
            found.Paths.Add(new SpecularPathData { Order = 2, Points = { 0, 0, 1, 3, 0, 1, 3, 4, 1, 5, 3, 1.5 } });
            var path = RayGeometry.FromReflections(found).Single();
            Assert.Equal(2, path.Order);
            Assert.Equal(4, path.Ray.Points.Length);
        }

        [Fact]
        public void ToPolyline_HasTheRaysPoints() => Assert.Equal(3, RayGeometry.ToPolyline(RayGeometry.FromField(Field())[0]).Count);
    }
}
