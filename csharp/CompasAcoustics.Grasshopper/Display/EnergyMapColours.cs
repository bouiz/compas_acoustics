using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The colours and points of Visualise Energy Map: a level per mesh face, from blue (lowest) to red (highest).</summary>
    public static class EnergyMapColours
    {
        /// <summary>A face without a level: outside the room or inside an obstacle.</summary>
        public static readonly Color Outside = Color.FromArgb(160, 160, 160);

        /// <summary>The colour of ``value`` on the scale from ``low`` (blue) to ``high`` (red); -inf (no energy yet) is the lowest.</summary>
        public static Color Of(double value, double low, double high)
        {
            if (double.IsNaN(value))
                return Outside;
            var t = high > low ? Math.Min(Math.Max((value - low) / (high - low), 0.0), 1.0) : 1.0;
            return Hsv(240.0 * (1.0 - t), 0.85, 0.95);
        }

        /// <summary>The scale's ends: the given interval, else the lowest and highest finite value (0, 0 when there is none).</summary>
        public static (double low, double high) Range(IEnumerable<double> values, Interval? given)
        {
            if (given is Interval interval)
                return (interval.Min, interval.Max);
            var finite = values.Where(double.IsFinite).ToArray();
            return finite.Length == 0 ? (0.0, 0.0) : (finite.Min(), finite.Max());
        }

        /// <summary>The centre of each face, the mean of its vertices.</summary>
        public static Point3d[] FaceCentres(IReadOnlyList<Point3d> vertices, IEnumerable<int[]> faces) =>
            faces.Select(face =>
            {
                var sum = Point3d.Origin;
                foreach (var v in face)
                    sum += vertices[v];
                return sum / face.Length;
            }).ToArray();

        /// <summary>
        /// The faces split into grids until no edge is longer than ``spacing``: a quad into quads, a triangle into triangles,
        /// so the map has a point every ``spacing`` metres whatever the mesh it was given (a plane of one face).
        /// </summary>
        public static (Point3d[] corners, int[][] faces) Subdivide(IReadOnlyList<Point3d> vertices, IReadOnlyList<int[]> faces, double spacing)
        {
            var corners = new List<Point3d>();
            var split = new List<int[]>();
            int Steps(params double[] lengths) => Math.Max(1, (int)Math.Ceiling(lengths.Max() / spacing - 1e-9));
            int Add(Point3d p)
            {
                corners.Add(p);
                return corners.Count - 1;
            }
            foreach (var face in faces)
            {
                var v = face.Select(i => vertices[i]).ToArray();
                if (v.Length == 4)
                {
                    var (a, b, c, d) = (v[0], v[1], v[2], v[3]);
                    var nu = Steps(a.DistanceTo(b), d.DistanceTo(c));
                    var nv = Steps(a.DistanceTo(d), b.DistanceTo(c));
                    var grid = new int[nu + 1, nv + 1];
                    for (var i = 0; i <= nu; i++)
                        for (var j = 0; j <= nv; j++)
                        {
                            double u = (double)i / nu, w = (double)j / nv;
                            // bilinear: the face's own shape, flat or not
                            var p = (1 - u) * (1 - w) * a + u * (1 - w) * b + u * w * c + (1 - u) * w * d;
                            grid[i, j] = Add(new Point3d(p.X, p.Y, p.Z));
                        }
                    for (var i = 0; i < nu; i++)
                        for (var j = 0; j < nv; j++)
                            split.Add(new[] { grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1] });
                }
                else if (v.Length == 3)
                {
                    var (a, b, c) = (v[0], v[1], v[2]);
                    var n = Steps(a.DistanceTo(b), b.DistanceTo(c), c.DistanceTo(a));
                    var grid = new int[n + 1, n + 1];
                    for (var i = 0; i <= n; i++)
                        for (var j = 0; i + j <= n; j++)
                            grid[i, j] = Add(a + (b - a) * ((double)i / n) + (c - a) * ((double)j / n));
                    for (var i = 0; i < n; i++)
                        for (var j = 0; i + j < n; j++)
                        {
                            split.Add(new[] { grid[i, j], grid[i + 1, j], grid[i, j + 1] });
                            if (i + j < n - 1)
                                split.Add(new[] { grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1] });
                        }
                }
            }
            return (corners.ToArray(), split.ToArray());
        }

        /// <summary>How far the map is drawn off its surface, in metres: enough not to flicker with the surface it came from.</summary>
        public const double Lift = 0.01;

        /// <summary>A face's lift: along its normal, turned upward on a floor or ceiling (|z| over 0.5), as wound on a wall.</summary>
        public static Vector3d Offset(IReadOnlyList<Point3d> face)
        {
            // Newell's normal
            var normal = Vector3d.Zero;
            for (var i = 0; i < face.Count; i++)
            {
                Point3d a = face[i], b = face[(i + 1) % face.Count];
                normal += new Vector3d((a.Y - b.Y) * (a.Z + b.Z), (a.Z - b.Z) * (a.X + b.X), (a.X - b.X) * (a.Y + b.Y));
            }
            var length = normal.Length;
            if (length < 1e-12)
                return new Vector3d(0, 0, Lift);
            normal /= length;
            if (Math.Abs(normal.Z) > 0.5 && normal.Z < 0)
                normal = -normal;
            return normal * Lift;
        }

        private static Color Hsv(double hue, double saturation, double value)
        {
            var c = value * saturation;
            var x = c * (1 - Math.Abs(hue / 60.0 % 2 - 1));
            var m = value - c;
            var (r, g, b) = hue switch
            {
                < 60 => (c, x, 0.0),
                < 120 => (x, c, 0.0),
                < 180 => (0.0, c, x),
                < 240 => (0.0, x, c),
                < 300 => (x, 0.0, c),
                _ => (c, 0.0, x),
            };
            return Color.FromArgb((int)Math.Round((r + m) * 255), (int)Math.Round((g + m) * 255), (int)Math.Round((b + m) * 255));
        }
    }
}
