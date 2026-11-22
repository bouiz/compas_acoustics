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
        public void Xyz_IsAListOfThreeNumbers()
        {
            Assert.Equal(new object?[] { 1.0, 2.0, 3.0 }, MeshConversion.Xyz(new Point3d(1, 2, 3)).ToArray());
            Assert.Equal(new object?[] { 0.0, 0.0, 1.0 }, MeshConversion.Xyz(Vector3d.ZAxis).ToArray());
        }
    }
}
