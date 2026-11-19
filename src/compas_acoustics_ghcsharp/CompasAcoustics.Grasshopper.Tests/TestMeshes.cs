using System.Collections.Generic;
using CompasPb.Data;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>Meshes the tests build scenes from.</summary>
    internal static class TestMeshes
    {
        internal static MeshData Quad(params double[][] corners)
        {
            var mesh = new MeshData();
            foreach (var corner in corners)
                mesh.Vertices.Add(corner);
            mesh.FaceVertices.Add(new uint[] { 0, 1, 2, 3 });
            mesh.FaceSizes.Add(4);
            return mesh;
        }

        internal static IEnumerable<MeshData> ShoeboxWalls(double x, double y, double z)
        {
            double[] P(double a, double b, double c) => new[] { a, b, c };
            yield return Quad(P(0, 0, 0), P(0, y, 0), P(x, y, 0), P(x, 0, 0));
            yield return Quad(P(0, 0, z), P(x, 0, z), P(x, y, z), P(0, y, z));
            yield return Quad(P(0, 0, 0), P(x, 0, 0), P(x, 0, z), P(0, 0, z));
            yield return Quad(P(x, 0, 0), P(x, y, 0), P(x, y, z), P(x, 0, z));
            yield return Quad(P(x, y, 0), P(0, y, 0), P(0, y, z), P(x, y, z));
            yield return Quad(P(0, y, 0), P(0, 0, 0), P(0, 0, z), P(0, y, z));
        }
    }
}
