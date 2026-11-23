using System;
using System.Collections.Generic;
using System.Linq;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>Where the sound of each specular ray is at a time, for Visualise Billiard.</summary>
    public static class Particles
    {
        private const double Tolerance = 1e-9;

        public static double DurationMs(IEnumerable<RayGeometry.Ray> rays, double speedOfSound) =>
            rays.Select(r => r.TravelTimeMs(speedOfSound)).DefaultIfEmpty(0.0).Max();

        /// <summary>
        /// The position of every ray at <paramref name="timeMs"/> (clamped to the rays' duration, <paramref name="durationMs"/>
        /// when known), with the number of reflections it has passed. A ray that has ended or not emerged yet gives none; a
        /// particle on a wall is reflected only once it moves on.
        /// </summary>
        public static List<(Point3d point, int order)> At(IReadOnlyList<RayGeometry.Ray> rays, double timeMs, double speedOfSound, double? durationMs = null)
        {
            timeMs = Math.Min(Math.Max(timeMs, 0.0), durationMs ?? DurationMs(rays, speedOfSound));
            var distance = speedOfSound * timeMs / 1000.0;
            var particles = new List<(Point3d, int)>();
            foreach (var ray in rays)
            {
                if (ray.Points.Length == 0 || ray.TravelTimeMs(speedOfSound) < timeMs)
                    continue;
                var cumulative = ray.CumulativeDistances;
                if (distance < cumulative[0])
                    continue;
                particles.Add((PointAt(ray.Points, cumulative, distance), cumulative.Skip(1).Count(reached => reached < distance - Tolerance)));
            }
            return particles;
        }

        private static Point3d PointAt(Point3d[] points, double[] cumulative, double distance)
        {
            if (distance <= cumulative[0])
                return points[0];
            for (var i = 1; i < points.Length; i++)
            {
                if (cumulative[i] <= distance)
                    continue;
                var segment = cumulative[i] - cumulative[i - 1];
                return RayGeometry.Lerp(points[i - 1], points[i], segment > 0 ? (distance - cumulative[i - 1]) / segment : 0.0);
            }
            return points[^1];
        }
    }
}
