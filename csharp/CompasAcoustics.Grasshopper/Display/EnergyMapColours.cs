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

        /// <summary>The frame shown at ``timeMs``: the last whose start (``times``, in seconds) it has reached.</summary>
        public static int FrameAt(IReadOnlyList<double> times, double timeMs)
        {
            var index = 0;
            for (var i = 0; i < times.Count; i++)
                if (times[i] * 1000.0 <= timeMs + 1e-9)
                    index = i;
            return index;
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
