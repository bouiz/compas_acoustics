using System.Linq;
using CompasPb.Data;
using Rhino.Geometry;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class MeshConversionTests
    {
        [Fact]
        public void ToMeshData_WritesFlatVerticesAndCsrFaces()
        {
            var vertices = new[] { new Point3d(0, 0, 0), new Point3d(1, 0, 0), new Point3d(1, 1, 0), new Point3d(0, 1, 0), new Point3d(2, 0, 0) };
            var faces = new[] { new MeshFace(0, 1, 2, 3), new MeshFace(1, 4, 2) };
            var mesh = MeshConversion.ToMeshData(vertices, faces);
            Assert.Equal(15, mesh.Vertices.Count);
            Assert.Equal(new double[] { 1, 1, 0 }, mesh.Vertices.Skip(6).Take(3).ToArray());
            Assert.Equal(new uint[] { 4, 3 }, mesh.FaceSizes.ToArray());
            Assert.Equal(new uint[] { 0, 1, 2, 3, 1, 4, 2 }, mesh.FaceVertices.ToArray());
        }

        [Fact]
        public void ToPolylineData_WritesTriplets()
        {
            var polyline = new Polyline(new[] { new Point3d(0, 0, 0), new Point3d(1, 2, 3) });
            Assert.Equal(new double[] { 0, 0, 0, 1, 2, 3 }, MeshConversion.ToPolylineData(polyline).Points.ToArray());
        }

        [Fact]
        public void ToFrameData_AndBack_KeepsTheAxes()
        {
            // Plane's constructors call Rhino's native code, which tests do not have
            var plane = Plane.WorldXY;
            plane.Origin = new Point3d(1, 2, 3);
            plane.XAxis = new Vector3d(0, 1, 0);
            plane.YAxis = new Vector3d(-1, 0, 0);
            plane.ZAxis = new Vector3d(0, 0, 1);
            var frame = MeshConversion.ToFrameData(plane);
            Assert.Equal(new[] { 1.0, 2.0, 3.0 }, new[] { frame.Point.X, frame.Point.Y, frame.Point.Z });
            Assert.Equal(new[] { 0.0, 1.0, 0.0 }, new[] { frame.Xaxis.X, frame.Xaxis.Y, frame.Xaxis.Z });
            var back = MeshConversion.ToPlane(frame);
            Assert.Equal(plane.Origin, back.Origin);
            Assert.Equal(plane.XAxis, back.XAxis);
            Assert.Equal(plane.ZAxis, back.ZAxis);
        }

        [Fact]
        public void Xyz_IsAListOfThreeNumbers()
        {
            Assert.Equal(new object?[] { 1.0, 2.0, 3.0 }, MeshConversion.Xyz(new Point3d(1, 2, 3)).ToArray());
            Assert.Equal(new object?[] { 0.0, 0.0, 1.0 }, MeshConversion.Xyz(Vector3d.ZAxis).ToArray());
        }
    }
}
