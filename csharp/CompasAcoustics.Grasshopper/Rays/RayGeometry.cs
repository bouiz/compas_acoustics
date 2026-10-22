using System;
using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The rays of a ray field or the paths of specular reflections, as geometry: what the display components draw.</summary>
    public static class RayGeometry
    {
        /// <summary>A traced ray. ``BirthOrder`` is the number of reflections before its first point (a scattered ray's origin order).</summary>
        public sealed record Ray(Point3d[] Points, double DistanceOffset, int BirthOrder, bool Scattered)
        {
            public double[] CumulativeDistances
            {
                get
                {
                    var distances = new double[Points.Length];
                    var accumulated = DistanceOffset;
                    for (var i = 0; i < Points.Length; i++)
                    {
                        if (i > 0)
                            accumulated += Distance(Points[i - 1], Points[i]);
                        distances[i] = accumulated;
                    }
                    return distances;
                }
            }

            public double TotalDistance => Points.Length == 0 ? DistanceOffset : CumulativeDistances[^1];

            public double TravelTimeMs(double speedOfSound) => TotalDistance / speedOfSound * 1000.0;
        }

        public sealed record Path(Ray Ray, int Order);

        internal static double Distance(Point3d a, Point3d b)
        {
            var (x, y, z) = (b.X - a.X, b.Y - a.Y, b.Z - a.Z);
            return Math.Sqrt(x * x + y * y + z * z);
        }

        private static Ray Of(IList<double> points, double offset, int birthOrder, bool scattered) =>
            new(MeshConversion.Points(points), offset, birthOrder, scattered);

        /// <summary>The specular rays of the field, then the scattered rays of each.</summary>
        public static List<Ray> FromField(RayFieldData field) =>
            Specular(field).Concat(field.Rays.SelectMany(r => r.ScatteredReflections)
                .Select(s => Of(s.Points, s.DistanceOffset, s.OriginReflectionOrder, true))).ToList();

        public static List<Ray> Specular(RayFieldData field) => field.Rays.Select(r => Of(r.Points, r.DistanceOffset, 0, false)).ToList();

        public static List<Path> FromReflections(SpecularReflectionsData found) =>
            found.Paths.Select(p => new Path(Of(p.Points, p.DistanceOffset, 0, false), p.Order)).ToList();

        /// <summary>Each segment of a ray with the reflection order it travels after.</summary>
        public static IEnumerable<(int order, Line segment)> Segments(Ray ray)
        {
            for (var k = 0; k + 1 < ray.Points.Length; k++)
                yield return (ray.BirthOrder + k, new Line(ray.Points[k], ray.Points[k + 1]));
        }

        public static Polyline ToPolyline(Ray ray) => new(ray.Points);

        /// <summary>The part of ``ray`` sound has travelled after ``distance`` metres: nothing before it emerges, all after it ends.</summary>
        public static Ray UpTo(Ray ray, double distance)
        {
            if (ray.Points.Length == 0 || distance < ray.DistanceOffset)
                return ray with { Points = Array.Empty<Point3d>() };
            var d = ray.CumulativeDistances;
            var points = new List<Point3d> { ray.Points[0] };
            for (var i = 1; i < ray.Points.Length; i++)
            {
                if (d[i] <= distance)
                {
                    points.Add(ray.Points[i]);
                    continue;
                }
                var t = (distance - d[i - 1]) / Math.Max(d[i] - d[i - 1], 1e-12);
                var (a, b) = (ray.Points[i - 1], ray.Points[i]);
                points.Add(new Point3d(a.X + t * (b.X - a.X), a.Y + t * (b.Y - a.Y), a.Z + t * (b.Z - a.Z)));
                break;
            }
            return ray with { Points = points.ToArray() };
        }
    }
}
