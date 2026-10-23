using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Linq;
using CompasAcoustics.Pb;
using CompasPb.Data;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>A source: a loudspeaker, a cube turned with it and a driver on the face toward its direction; its arrow and directivity lobe.</summary>
    public sealed record SourcePreview(Point3d Centre, Line Arrow, Polyline Lobe, string Label, Line[] Cube, Point3d Driver, Vector3d Facing, double DriverRadius);

    /// <summary>A receiver: a head, a sphere with its ears on the sides, its facing and up (a source is a loudspeaker).</summary>
    public sealed record ReceiverPreview(Point3d Centre, Line Facing, Line Up, Point3d LeftEar, Point3d RightEar, Vector3d EarAxis, string Label);

    /// <summary>One acoustic object: its mesh (or its polyline), the edges drawn of it, whether it is an obstacle.</summary>
    public sealed record ObjectPreview(Point3d[] Vertices, int[][] Faces, Polyline? Polyline, bool Obstacle, Line[] Edges)
    {
        /// <summary>The surface of its faces (0 for a polyline).</summary>
        public double Area => Faces.Sum(face => Enumerable.Range(1, Math.Max(face.Length - 2, 0))
            .Sum(i => Vector3d.CrossProduct(Vertices[face[i]] - Vertices[face[0]], Vertices[face[i + 1]] - Vertices[face[0]]).Length / 2));
    }

    /// <summary>A scene: its objects, and the room's volume and surface (its boundary, obstacles left out) for the Scene component's message.</summary>
    public sealed record ScenePreview(IReadOnlyList<ObjectPreview> Objects, BoundingBox Box, double Volume, double Area);

    /// <summary>What the goo draw in the viewport, as plain geometry (the drawing itself is in <see cref="PreviewDrawing"/>).</summary>
    public static class PreviewGeometry
    {
        public const double SourceRadius = 0.15;
        public const double SourceArrow = 0.5;
        public const double LobeRadius = 0.3;
        public const int LobePoints = 36;
        public const double HeadRadius = 0.1;
        public const double EarRadius = 0.03;
        public const double FacingArrow = 0.4;
        public const double UpTick = 0.15;

        /// <summary>``a`` of the first-order pattern ``|a + (1 - a) cos t|``.</summary>
        public static double PatternWeight(string? directivity) => directivity switch
        {
            "cardioid" => 0.5,
            "hypercardioid" => 0.25,
            "supercardioid" => 0.366,
            "bidirectional" => 0.0,
            _ => 1.0,
        };

        // Vector3d.Unitize and IsTiny are native RhinoCommon, which the tests do not load
        private static Vector3d Unit(Vector3d v, Vector3d fallback) => !v.IsValid || v.Length < 1e-12 ? fallback : v / v.Length;

        public static SourcePreview Source(SourceData source)
        {
            var centre = Points.Of(source.Point);
            var direction = Unit(Points.Of(source.Direction), Vector3d.XAxis);
            // the lobe lies in the plane of the direction and world z (world x when the source faces up or down)
            var side = Vector3d.ZAxis - (Vector3d.ZAxis * direction) * direction;
            if (side.Length < 1e-9)
                side = Vector3d.XAxis - (Vector3d.XAxis * direction) * direction;
            side /= side.Length;
            var a = PatternWeight(source.Directivity);
            var lobe = new Polyline(LobePoints + 1);
            for (var i = 0; i < LobePoints; i++)
            {
                var t = 2 * Math.PI * i / LobePoints;
                var r = LobeRadius * Math.Abs(a + (1 - a) * Math.Cos(t));
                lobe.Add(centre + r * (Math.Cos(t) * direction + Math.Sin(t) * side));
            }
            lobe.Add(lobe[0]);
            // the cube's up is world z, or x for a source facing up or down
            var up = Vector3d.ZAxis - (Vector3d.ZAxis * direction) * direction;
            up = up.Length < 1e-9 ? Vector3d.XAxis - (Vector3d.XAxis * direction) * direction : up;
            up /= up.Length;
            var left = Vector3d.CrossProduct(up, direction);
            left /= left.Length;
            return new SourcePreview(centre, new Line(centre, centre + SourceArrow * direction), lobe, source.Name,
                Cube(centre, direction, left, up, SourceRadius), centre + SourceRadius * direction, direction, 0.6 * SourceRadius);
        }

        public static ReceiverPreview Receiver(ReceiverData receiver)
        {
            var centre = Points.Of(receiver.Point);
            var facing = Unit(Points.Of(receiver.Direction), Vector3d.XAxis);
            var up = Unit(Points.Of(receiver.Up), Vector3d.ZAxis);
            up -= (up * facing) * facing;
            up = Unit(up, Vector3d.ZAxis);
            // the frame's y axis: z cross x, to the listener's left
            var y = Unit(Vector3d.CrossProduct(up, facing), Vector3d.YAxis);
            return new ReceiverPreview(centre, new Line(centre, centre + FacingArrow * facing), new Line(centre, centre + UpTick * up),
                centre + HeadRadius * y, centre - HeadRadius * y, y, receiver.Name);
        }

        /// <summary>The 12 edges of a cube of half-size ``h`` around ``centre``, along the axes ``x``, ``y``, ``z``.</summary>
        private static Line[] Cube(Point3d centre, Vector3d x, Vector3d y, Vector3d z, double h)
        {
            Point3d Corner(int i) => centre + h * ((i & 1) == 0 ? -1 : 1) * x + h * ((i & 2) == 0 ? -1 : 1) * y + h * ((i & 4) == 0 ? -1 : 1) * z;
            var edges = new List<Line>();
            for (var i = 0; i < 8; i++)
                foreach (var bit in new[] { 1, 2, 4 })
                    if ((i & bit) == 0)
                        edges.Add(new Line(Corner(i), Corner(i | bit)));
            return edges.ToArray();
        }

        private static (Point3d[] vertices, int[][] faces) Mesh(MeshData mesh)
        {
            var vertices = MeshConversion.Points(mesh.Vertices);
            var faces = new int[mesh.FaceSizes.Count][];
            var at = 0;
            for (var f = 0; f < faces.Length; f++)
            {
                var size = (int)mesh.FaceSizes[f];
                faces[f] = mesh.FaceVertices.Skip(at).Take(size).Select(i => (int)i).ToArray();
                at += size;
            }
            return (vertices, faces);
        }

        /// <summary>One acoustic object: its mesh or polyline, whether it is an obstacle.</summary>
        public static ObjectPreview Object(AcousticObjectData o)
        {
            var obstacle = o.Role == "obstacle";
            if (o.GeometryCase == AcousticObjectData.GeometryOneofCase.Mesh)
            {
                var (vertices, faces) = Mesh(o.Mesh);
                return new ObjectPreview(vertices, faces, null, obstacle, Edges(vertices, faces));
            }
            var polyline = o.GeometryCase == AcousticObjectData.GeometryOneofCase.Polyline ? new Polyline(MeshConversion.Points(o.Polyline.Points)) : null;
            return new ObjectPreview(Array.Empty<Point3d>(), Array.Empty<int[]>(), polyline, obstacle, Array.Empty<Line>());
        }

        /// <summary>Two faces meeting at less than this angle (cosine of 1 degree) are one surface: their side is not an edge.</summary>
        private const double Coplanar = 0.99985;

        /// <summary>
        /// The edges of a mesh as Rhino draws a surface's: the sides of one face only (the outline) and the folds between two,
        /// not the sides shared by faces of one plane (the triangulation). Corners are matched by position to the micrometre,
        /// as a Rhino mesh's faces often have their own copies of them.
        /// </summary>
        public static Line[] Edges(Point3d[] vertices, int[][] faces)
        {
            (long, long, long) Key(Point3d p) => ((long)Math.Round(p.X * 1e6), (long)Math.Round(p.Y * 1e6), (long)Math.Round(p.Z * 1e6));
            var sides = new Dictionary<((long, long, long), (long, long, long)), (Line line, List<Vector3d> normals)>();
            foreach (var face in faces.Where(f => f.Length >= 3))
            {
                // Newell's normal, robust for a polygon of any size
                var normal = Vector3d.Zero;
                for (var i = 0; i < face.Length; i++)
                {
                    Point3d a = vertices[face[i]], b = vertices[face[(i + 1) % face.Length]];
                    normal += new Vector3d((a.Y - b.Y) * (a.Z + b.Z), (a.Z - b.Z) * (a.X + b.X), (a.X - b.X) * (a.Y + b.Y));
                }
                normal = normal.Length > 0 ? normal / normal.Length : normal;
                for (var i = 0; i < face.Length; i++)
                {
                    Point3d a = vertices[face[i]], b = vertices[face[(i + 1) % face.Length]];
                    var (ka, kb) = (Key(a), Key(b));
                    if (ka == kb)
                        continue;
                    var key = ka.CompareTo(kb) < 0 ? (ka, kb) : (kb, ka);
                    if (!sides.TryGetValue(key, out var side))
                        sides[key] = side = (new Line(a, b), new List<Vector3d>());
                    side.normals.Add(normal);
                }
            }
            return sides.Values.Where(s => s.normals.Count != 2 || Math.Abs(s.normals[0] * s.normals[1]) < Coplanar).Select(s => s.line).ToArray();
        }

        /// <summary>The scene's objects (its sources and receivers are drawn by their own components), and the room's volume
        /// and surface from its boundary.</summary>
        public static ScenePreview Scene(SceneData scene)
        {
            var objects = scene.AcousticObjects.Select(Object).ToList();
            var box = BoundingBox.Empty;
            double volume = 0;
            foreach (var o in objects)
            {
                foreach (var v in o.Polyline ?? (IEnumerable<Point3d>)o.Vertices)
                    box.Union(v);
                if (o.Obstacle)
                    continue;
                foreach (var face in o.Faces)
                    for (var i = 1; i + 1 < face.Length; i++)
                    {
                        Point3d p0 = o.Vertices[face[0]], p1 = o.Vertices[face[i]], p2 = o.Vertices[face[i + 1]];
                        // divergence theorem: the signed volume of the tetrahedron with the origin
                        volume += new Vector3d(p0) * Vector3d.CrossProduct(new Vector3d(p1), new Vector3d(p2)) / 6;
                    }
            }
            return new ScenePreview(objects, box, Math.Abs(volume), objects.Where(o => !o.Obstacle).Sum(o => o.Area));
        }
    }
}
