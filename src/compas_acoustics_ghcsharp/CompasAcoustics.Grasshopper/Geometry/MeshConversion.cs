using System.Collections.Generic;
using System.Linq;
using CompasPb.Data;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>Rhino geometry as compas_pb messages and worker arguments, and points back.</summary>
    public static class MeshConversion
    {
        /// <summary>A mesh as compas_pb's MeshData: flat x, y, z triplets and its faces in CSR form (indices, then sizes).</summary>
        public static MeshData ToMeshData(IReadOnlyList<Point3d> vertices, IEnumerable<MeshFace> faces)
        {
            var mesh = new MeshData();
            foreach (var v in vertices)
                mesh.Vertices.Add(new[] { v.X, v.Y, v.Z });
            foreach (var face in faces)
            {
                var corners = Corners(face);
                mesh.FaceVertices.Add(corners.Select(i => (uint)i));
                mesh.FaceSizes.Add((uint)corners.Length);
            }
            return mesh;
        }

        public static MeshData ToMeshData(Mesh mesh) => ToMeshData(mesh.Vertices.ToPoint3dArray(), mesh.Faces);

        /// <summary>The vertex indices of a face: four for a quad, three for a triangle.</summary>
        public static int[] Corners(MeshFace face) => face.IsQuad ? new[] { face.A, face.B, face.C, face.D } : new[] { face.A, face.B, face.C };

        public static PolylineData ToPolylineData(Polyline polyline)
        {
            var data = new PolylineData();
            foreach (var p in polyline)
                data.Points.Add(new[] { p.X, p.Y, p.Z });
            return data;
        }

        /// <summary>A point or vector as the <c>[x, y, z]</c> list the worker takes.</summary>
        /// <summary>The frame of ``plane``: its origin and its X and Y axes.</summary>
        public static FrameData ToFrameData(Plane plane) => new FrameData
        {
            Point = new PointData { X = plane.Origin.X, Y = plane.Origin.Y, Z = plane.Origin.Z },
            Xaxis = new VectorData { X = plane.XAxis.X, Y = plane.XAxis.Y, Z = plane.XAxis.Z },
            Yaxis = new VectorData { X = plane.YAxis.X, Y = plane.YAxis.Y, Z = plane.YAxis.Z },
        };

        public static List<object?> Xyz(Point3d p) => new() { p.X, p.Y, p.Z };

        public static List<object?> Xyz(Vector3d v) => new() { v.X, v.Y, v.Z };

        public static Point3d[] Points(IList<double> flat)
        {
            var points = new Point3d[flat.Count / 3];
            for (var i = 0; i < points.Length; i++)
                points[i] = new Point3d(flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]);
            return points;
        }
    }
}
