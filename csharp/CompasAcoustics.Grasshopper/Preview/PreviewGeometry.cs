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
    public sealed record SourcePreview(Point3d Centre, Line Arrow, Polyline Lobe, string Label);

    /// <summary>A receiver: a cube turned with it (a source is a sphere), its ears on the cube's sides, its facing and up.</summary>
    public sealed record ReceiverPreview(Point3d Centre, Line Facing, Line Up, Point3d LeftEar, Point3d RightEar, Vector3d EarAxis, string Label, Line[] Cube);

    /// <summary>One acoustic object: its mesh (or its polyline), its shade, whether it is an obstacle.</summary>
    public sealed record ObjectPreview(Point3d[] Vertices, int[][] Faces, Polyline? Polyline, Color? Shade, bool Obstacle);

    public sealed record ScenePreview(IReadOnlyList<ObjectPreview> Objects, IReadOnlyList<SourcePreview> Sources, IReadOnlyList<ReceiverPreview> Receivers,
        string Label, Point3d LabelAt, double Volume, double Area, BoundingBox Box);

    public sealed record PairPreview(Line Line, string Label);

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
            return new SourcePreview(centre, new Line(centre, centre + SourceArrow * direction), lobe, source.Name);
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
                centre + HeadRadius * y, centre - HeadRadius * y, y, receiver.Name, Cube(centre, facing, y, up, HeadRadius));
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

        /// <summary>One acoustic object: its mesh or polyline, its shade, whether it is an obstacle.</summary>
        public static ObjectPreview Object(AcousticObjectData o)
        {
            var obstacle = o.Role == "obstacle";
            var shade = MaterialShade.Of(o.Material);
            if (o.GeometryCase == AcousticObjectData.GeometryOneofCase.Mesh)
            {
                var (vertices, faces) = Mesh(o.Mesh);
                return new ObjectPreview(vertices, faces, null, shade, obstacle);
            }
            var polyline = o.GeometryCase == AcousticObjectData.GeometryOneofCase.Polyline ? new Polyline(MeshConversion.Points(o.Polyline.Points)) : null;
            return new ObjectPreview(Array.Empty<Point3d>(), Array.Empty<int[]>(), polyline, shade, obstacle);
        }

        public static ScenePreview Scene(SceneData scene)
        {
            var objects = new List<ObjectPreview>();
            var box = BoundingBox.Empty;
            double volume = 0, area = 0;
            var boundary = BoundingBox.Empty;
            foreach (var o in scene.AcousticObjects)
            {
                var preview = Object(o);
                var obstacle = preview.Obstacle;
                objects.Add(preview);
                if (o.GeometryCase == AcousticObjectData.GeometryOneofCase.Mesh)
                {
                    var (vertices, faces) = (preview.Vertices, preview.Faces);
                    foreach (var v in vertices)
                        box.Union(v);
                    if (obstacle)
                        continue;
                    foreach (var v in vertices)
                        boundary.Union(v);
                    foreach (var face in faces)
                        for (var i = 1; i + 1 < face.Length; i++)
                        {
                            Point3d p0 = vertices[face[0]], p1 = vertices[face[i]], p2 = vertices[face[i + 1]];
                            var cross = Vector3d.CrossProduct(p1 - p0, p2 - p0);
                            area += cross.Length / 2;
                            // divergence theorem: the signed volume of the tetrahedron with the origin
                            volume += new Vector3d(p0) * Vector3d.CrossProduct(new Vector3d(p1), new Vector3d(p2)) / 6;
                        }
                }
                else if (o.GeometryCase == AcousticObjectData.GeometryOneofCase.Polyline)
                {
                    foreach (var v in preview.Polyline!)
                    {
                        box.Union(v);
                        if (!obstacle)
                            boundary.Union(v);
                    }
                }
            }
            var sources = scene.Sources.Select(Source).ToList();
            var receivers = scene.Receivers.Select(Receiver).ToList();
            foreach (var s in sources)
                box.Union(new BoundingBox(s.Centre - new Vector3d(SourceArrow, SourceArrow, SourceArrow), s.Centre + new Vector3d(SourceArrow, SourceArrow, SourceArrow)));
            foreach (var r in receivers)
                box.Union(new BoundingBox(r.Centre - new Vector3d(FacingArrow, FacingArrow, FacingArrow), r.Centre + new Vector3d(FacingArrow, FacingArrow, FacingArrow)));
            volume = Math.Abs(volume);
            var label = string.Format(CultureInfo.InvariantCulture, "{0}, V {1:0} m3, S {2:0} m2", scene.Name, volume, area);
            var at = boundary.IsValid ? boundary.Center : box.IsValid ? box.Center : Point3d.Origin;
            return new ScenePreview(objects, sources, receivers, label, at, volume, area, box);
        }

        /// <summary>A line from the source to the receiver of each response.</summary>
        public static List<PairPreview> Graph(IrGraphGoo.Content graph) =>
            graph.Responses.Where(r => r.Value?.Source != null && r.Value.Receiver != null)
                .Select(r => new PairPreview(new Line(Points.Of(r.Value.Source.Point), Points.Of(r.Value.Receiver.Point)), $"{r.SourceName} -> {r.ReceiverName}"))
                .ToList();
    }

    /// <summary>The grey of a material in the scene preview: light for absorption 0, dark for 1, by the mean absorption.</summary>
    public static class MaterialShade
    {
        public const int Light = 230;
        public const int Dark = 60;

        public static Color? Of(AcousticMaterialData? material)
        {
            if (material == null || material.AbsorptionCoefficients.Count == 0)
                return null;
            var mean = Math.Clamp(material.AbsorptionCoefficients.Average(), 0, 1);
            var grey = (int)Math.Round(Light - (Light - Dark) * mean);
            return Color.FromArgb(grey, grey, grey);
        }
    }
}
