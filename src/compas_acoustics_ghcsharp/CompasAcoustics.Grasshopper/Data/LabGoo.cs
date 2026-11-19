using System;
using System.Globalization;
using System.Linq;
using CompasAcoustics.Pb;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>What the scattering sample draws: its mesh's edges, and the outline of its reference.</summary>
    public sealed class ScatteringSamplePreview
    {
        public ScatteringSamplePreview(Line[] sampleEdges, Line[] referenceOutline, BoundingBox box)
        {
            SampleEdges = sampleEdges;
            ReferenceOutline = referenceOutline;
            Box = box;
        }

        public Line[] SampleEdges { get; }

        /// <summary>The sides of the reference used by one face only.</summary>
        public Line[] ReferenceOutline { get; }

        /// <summary>The sample's box.</summary>
        public BoundingBox Box { get; }

        public static ScatteringSamplePreview Of(ScatteringSampleData data)
        {
            var (vertices, faces) = data.Sample == null ? (Array.Empty<Point3d>(), Array.Empty<int[]>()) : PreviewGeometry.MeshOf(data.Sample);
            var box = vertices.Length == 0 ? BoundingBox.Empty : new BoundingBox(vertices);
            return new ScatteringSamplePreview(PreviewGeometry.Edges(vertices, faces), Outline(data.Reference), box);
        }

        /// <summary>Whether the face's normal (Newell's, for a polygon of any size) has a positive z.</summary>
        private static bool FacesUp(Point3d[] vertices, int[] face)
        {
            double z = 0;
            for (var i = 0; i < face.Length; i++)
            {
                Point3d a = vertices[face[i]], b = vertices[face[(i + 1) % face.Length]];
                z += (a.X - b.X) * (a.Y + b.Y);
            }
            return z > 1e-12;
        }

        /// <summary>
        /// The outline of the reference's top: the sides used by one face of those facing up (the reference is a closed body,
        /// its walls and bottom are not drawn). Corners are matched by position to the micrometre.
        /// </summary>
        private static Line[] Outline(global::CompasPb.Data.MeshData? mesh)
        {
            if (mesh == null)
                return Array.Empty<Line>();
            var (vertices, faces) = PreviewGeometry.MeshOf(mesh);
            (long, long, long) Key(Point3d p) => ((long)Math.Round(p.X * 1e6), (long)Math.Round(p.Y * 1e6), (long)Math.Round(p.Z * 1e6));
            var sides = new System.Collections.Generic.Dictionary<((long, long, long), (long, long, long)), (Line line, int count)>();
            foreach (var face in faces.Where(f => f.Length >= 3 && FacesUp(vertices, f)))
                for (var i = 0; i < face.Length; i++)
                {
                    Point3d a = vertices[face[i]], b = vertices[face[(i + 1) % face.Length]];
                    var (ka, kb) = (Key(a), Key(b));
                    if (ka == kb)
                        continue;
                    var key = ka.CompareTo(kb) < 0 ? (ka, kb) : (kb, ka);
                    sides[key] = sides.TryGetValue(key, out var side) ? (side.line, side.count + 1) : (new Line(a, b), 1);
                }
            return sides.Values.Where(s => s.count == 1).Select(s => s.line).ToArray();
        }
    }

    public sealed class ScatteringSampleGoo : MessageGoo<ScatteringSampleData>, IOwnPreview
    {
        private (ScatteringSampleData of, ScatteringSamplePreview preview)? _preview;

        /// <summary>The preview geometry, built once for each message.</summary>
        public ScatteringSamplePreview? Preview => Value == null ? null : (_preview is { } p && ReferenceEquals(p.of, Value) ? p : (_preview = (Value, ScatteringSamplePreview.Of(Value))).Value).preview;

        public BoundingBox ClippingBox => Preview?.Box ?? BoundingBox.Empty;

        public void DrawWires(Rhino.Display.DisplayPipeline pipeline, System.Drawing.Color color, int thickness)
        {
            if (Preview is not { } preview)
                return;
            pipeline.DrawLines(preview.SampleEdges, color, thickness);
            pipeline.DrawLines(preview.ReferenceOutline, PreviewDrawing.ObstacleEdge, thickness);
        }

        public ScatteringSampleGoo() { }
        public ScatteringSampleGoo(ScatteringSampleData value) : base(value) { }
        public override string TypeName => "Scattering Sample";
        public override string TypeDescription => "A compas_acoustics ScatteringSample: a surface sample and its reference, for the scattering lab.";
        public override string Name => Value?.Name ?? "";

        public override string ToString() => Value == null
            ? base.ToString()
            : string.Format(CultureInfo.InvariantCulture, "{0} '{1}', {2}, d {3:0.00} m, h {4:0.000} m, {5} reference",
                TypeName, Value.Name, Value.Mode, Value.Diameter, Value.StructuralDepth, Value.ReferenceKind);

        protected override MessageGoo<ScatteringSampleData> Create(ScatteringSampleData? value) => value == null ? new ScatteringSampleGoo() : new ScatteringSampleGoo(value);
    }

    public sealed class ScatteringResultGoo : MessageGoo<ScatteringResultData>
    {
        public ScatteringResultGoo() { }
        public ScatteringResultGoo(ScatteringResultData value) : base(value) { }
        public override string TypeName => "Scattering Result";
        public override string TypeDescription => "A compas_acoustics ScatteringResult: scattering (and diffusion) coefficients of a sample.";
        public override string Name => Value?.Name ?? "";

        /// <summary>The diffusion coefficient of each octave band; empty when the result has none.</summary>
        public double[] Diffusion => Value?.DOctaves == null ? Array.Empty<double>() : Value.DOctaves.Values.ToArray();

        public override string ToString()
        {
            if (Value == null)
                return base.ToString();
            var text = string.Format(CultureInfo.InvariantCulture, "{0} '{1}', {2}", TypeName, Value.Name, Value.Method);
            if (Value.SOctaves.Count > 0)
            {
                text += string.Format(CultureInfo.InvariantCulture, ", s {0:0.00}–{1:0.00}", Value.SOctaves.Min(), Value.SOctaves.Max());
                if (Value.OctaveBands.Count > 0)
                    text += string.Format(CultureInfo.InvariantCulture, " ({0}–{1} Hz)", Value.OctaveBands.Min(), Value.OctaveBands.Max());
            }
            var d = Diffusion;
            if (d.Length > 0)
                text += string.Format(CultureInfo.InvariantCulture, ", d {0:0.00}–{1:0.00}", d.Min(), d.Max());
            return text;
        }

        protected override MessageGoo<ScatteringResultData> Create(ScatteringResultData? value) => value == null ? new ScatteringResultGoo() : new ScatteringResultGoo(value);
    }
}
