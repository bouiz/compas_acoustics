using System.Collections.Generic;
using System.Linq;
using CompasPb.Data;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>Rhino geometry as compas_pb messages, and back.</summary>
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
                var corners = face.IsQuad ? new[] { face.A, face.B, face.C, face.D } : new[] { face.A, face.B, face.C };
                mesh.FaceVertices.Add(corners.Select(i => (uint)i));
                mesh.FaceSizes.Add((uint)corners.Length);
            }
            return mesh;
        }

        public static MeshData ToMeshData(Mesh mesh) => ToMeshData(mesh.Vertices.ToPoint3dArray(), mesh.Faces);

        public static PolylineData ToPolylineData(Polyline polyline)
        {
            var data = new PolylineData();
            foreach (var p in polyline)
                data.Points.Add(new[] { p.X, p.Y, p.Z });
            return data;
        }

        public static FrameData ToFrameData(Plane plane) => new FrameData
        {
            Point = new PointData { X = plane.Origin.X, Y = plane.Origin.Y, Z = plane.Origin.Z },
            Xaxis = new VectorData { X = plane.XAxis.X, Y = plane.XAxis.Y, Z = plane.XAxis.Z },
            Yaxis = new VectorData { X = plane.YAxis.X, Y = plane.YAxis.Y, Z = plane.YAxis.Z },
        };

        /// <summary>The plane of an orthonormal frame (the worker's are), built without Rhino's native code.</summary>
        public static Plane ToPlane(FrameData frame)
        {
            var x = Unit(new Vector3d(frame.Xaxis.X, frame.Xaxis.Y, frame.Xaxis.Z));
            var y = Unit(new Vector3d(frame.Yaxis.X, frame.Yaxis.Y, frame.Yaxis.Z));
            var plane = Plane.WorldXY;
            plane.Origin = new Point3d(frame.Point.X, frame.Point.Y, frame.Point.Z);
            plane.XAxis = x;
            plane.YAxis = y;
            plane.ZAxis = Unit(Vector3d.CrossProduct(x, y));
            return plane;
        }

        private static Vector3d Unit(Vector3d v)
        {
            var length = System.Math.Sqrt(v.X * v.X + v.Y * v.Y + v.Z * v.Z);
            return length > 0 ? new Vector3d(v.X / length, v.Y / length, v.Z / length) : v;
        }

        /// <summary>A point or vector as the ``[x, y, z]`` list the worker takes.</summary>
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
