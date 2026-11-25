using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.Linq;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>A direction a result holds polar levels for: its index (the value of Polar Preview's preview input), colatitude and azimuth in degrees.</summary>
    public sealed record Incidence(int Index, double Theta, double Phi);

    /// <summary>
    /// The incidence direction as two dials side by side: a section through the sample for the colatitude θ (the sound from the left,
    /// 0° the normal), a plan for the azimuth φ (counterclockwise from +X, as Rhino's Top view). Only the directions a result has are
    /// ticks; an arrow runs from the chosen one to the sample, the way the sound travels.
    /// </summary>
    public static class IncidenceDials
    {
        /// <summary>The height of the dials' row.</summary>
        public const float Height = 84;

        /// <summary>The narrowest component the two dials fit in.</summary>
        public const float MinWidth = 176;

        private const float SectionRadius = 30;
        private const float PlanRadius = 24;
        private const float Reach = 9; // px around a tick's outer end that picks it
        private const double Same = 1e-6;

        private static readonly Color Chosen = Color.FromArgb(31, 98, 208);
        private static readonly Color Off = Color.FromArgb(150, 150, 150);
        private static readonly Color Faint = Color.FromArgb(170, 170, 170);

        /// <summary>The centre of the section dial (the sample's top, where its arc starts and ends) and of the plan dial.</summary>
        public static (PointF section, PointF plan) Centres(RectangleF row)
        {
            var half = row.Width / 2;
            return (new PointF(row.Left + half / 2 + 4, row.Top + 58), new PointF(row.Left + half + half / 2, row.Top + 34));
        }

        /// <summary>The colatitudes the result has, ascending: one ring each.</summary>
        public static double[] Rings(IReadOnlyList<Incidence> incidences) =>
            incidences.Select(i => i.Theta).OrderBy(t => t).Aggregate(new List<double>(), (rings, t) =>
            {
                if (rings.Count == 0 || Math.Abs(rings[^1] - t) > Same)
                    rings.Add(t);
                return rings;
            }).ToArray();

        /// <summary>The directions on the ring of colatitude <paramref name="theta"/>, by azimuth.</summary>
        public static Incidence[] Ring(IReadOnlyList<Incidence> incidences, double theta) =>
            incidences.Where(i => Math.Abs(i.Theta - theta) <= Same).OrderBy(i => i.Phi).ToArray();

        /// <summary>The outer end of a colatitude's tick on the section dial: θ from the normal, on the sound's side (the left).</summary>
        public static PointF SectionTick(RectangleF row, double theta, float beyond = 3)
        {
            var (c, _) = Centres(row);
            var a = theta * Math.PI / 180;
            var r = SectionRadius + beyond;
            return new PointF(c.X - (float)(r * Math.Sin(a)), c.Y - (float)(r * Math.Cos(a)));
        }

        /// <summary>The outer end of an azimuth's tick on the plan dial: counterclockwise from +X, with the screen's y down.</summary>
        public static PointF PlanTick(RectangleF row, double phi, float beyond = 3)
        {
            var (_, c) = Centres(row);
            var a = phi * Math.PI / 180;
            var r = PlanRadius + beyond;
            return new PointF(c.X + (float)(r * Math.Cos(a)), c.Y - (float)(r * Math.Sin(a)));
        }

        /// <summary>
        /// The tick under <paramref name="point"/>: a colatitude of the section dial, or an azimuth on the plan dial of the ring of
        /// <paramref name="current"/> (none when that ring has one direction); both null away from every tick.
        /// </summary>
        public static (double? theta, double? phi) Hit(RectangleF row, PointF point, IReadOnlyList<Incidence> incidences, Incidence? current)
        {
            static float Distance(PointF a, PointF b) => (float)Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));
            var best = (distance: Reach, theta: (double?)null, phi: (double?)null);
            foreach (var theta in Rings(incidences))
            {
                var d = Distance(point, SectionTick(row, theta));
                if (d < best.distance)
                    best = (d, theta, null);
            }
            var ring = current == null ? Array.Empty<Incidence>() : Ring(incidences, current.Theta);
            if (ring.Length > 1)
                foreach (var direction in ring)
                {
                    var d = Distance(point, PlanTick(row, direction.Phi));
                    if (d < best.distance)
                        best = (d, null, direction.Phi);
                }
            return (best.theta, best.phi);
        }

        private static double AzimuthGap(double a, double b)
        {
            var d = Math.Abs(a - b) % 360;
            return Math.Min(d, 360 - d);
        }

        /// <summary>
        /// The direction a pick makes: a colatitude keeps the azimuth (<paramref name="remembered"/>, the last one chosen on a ring
        /// of several, else the current one), or the nearest the new ring has; an azimuth moves on the current ring. Null when the
        /// result has no direction.
        /// </summary>
        public static Incidence? Choose(IReadOnlyList<Incidence> incidences, Incidence? current, double? theta, double? phi, double? remembered)
        {
            if (incidences.Count == 0)
                return null;
            var ringTheta = theta ?? current?.Theta ?? Rings(incidences)[0];
            var ring = Ring(incidences, ringTheta);
            if (ring.Length == 0)
                return current;
            var wanted = phi ?? remembered ?? current?.Phi ?? 0;
            return ring.OrderBy(i => AzimuthGap(i.Phi, wanted)).First();
        }

        private static string Degrees(double value) => string.Create(CultureInfo.InvariantCulture, $"{value:0.##}°");

        private static void Arrow(Graphics g, Color colour, PointF from, PointF to)
        {
            using var pen = new Pen(colour, 2) { StartCap = LineCap.Round };
            var a = Math.Atan2(to.Y - from.Y, to.X - from.X);
            const float head = 6;
            g.DrawLine(pen, from, to);
            using var brush = new SolidBrush(colour);
            g.FillPolygon(brush, new[]
            {
                to,
                new PointF(to.X - head * (float)Math.Cos(a - 0.45), to.Y - head * (float)Math.Sin(a - 0.45)),
                new PointF(to.X - head * (float)Math.Cos(a + 0.45), to.Y - head * (float)Math.Sin(a + 0.45)),
            });
        }

        private static PointF Towards(PointF centre, PointF tick, float distance)
        {
            var dx = tick.X - centre.X;
            var dy = tick.Y - centre.Y;
            var length = (float)Math.Sqrt(dx * dx + dy * dy);
            return new PointF(centre.X + dx / length * distance, centre.Y + dy / length * distance);
        }

        /// <summary>Draw both dials in <paramref name="row"/>; grey when a wire sets the input, empty ticks until the component has a result.</summary>
        public static void Draw(Graphics g, RectangleF row, IReadOnlyList<Incidence> incidences, Incidence? current, bool wired)
        {
            var state = g.SmoothingMode;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            var (section, plan) = Centres(row);
            var pick = wired ? Off : Chosen;
            using var font = new Font(FontFamily.GenericSansSerif, 5.5f);
            using var bold = new Font(FontFamily.GenericSansSerif, 8.5f, FontStyle.Bold);
            using var centred = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center };
            using var line = new Pen(Color.Black, 1.2f);
            using var faint = new Pen(Faint, 0.8f);

            // the section: the arc from horizon to horizon, a 15° scale, the sample as a slab under its centre
            g.DrawArc(line, section.X - SectionRadius, section.Y - SectionRadius, 2 * SectionRadius, 2 * SectionRadius, 180, 180);
            for (var t = -90; t <= 90; t += 15)
                g.DrawLine(faint, SectionTick(row, t, -3), SectionTick(row, t, 3));
            g.FillRectangle(Brushes.White, section.X - 18, section.Y + 2, 36, 5);
            g.DrawRectangle(line, section.X - 18, section.Y + 2, 36, 5);
            var rings = Rings(incidences);
            var dense = rings.Length > 4;
            foreach (var theta in rings)
            {
                var chosen = current != null && Math.Abs(current.Theta - theta) <= Same;
                // the chosen tick long and coloured, the others short and thin
                using var pen = new Pen(chosen ? pick : Color.Black, chosen ? 2.4f : 0.9f) { StartCap = LineCap.Round, EndCap = LineCap.Round };
                g.DrawLine(pen, SectionTick(row, theta, chosen ? -5 : -3), SectionTick(row, theta, chosen ? 6 : 3));
                if (!dense || Math.Abs(theta % 30) < 0.5 || Math.Abs(theta % 30 - 30) < 0.5)
                    g.DrawString(Degrees(theta), font, Brushes.DimGray, SectionTick(row, theta, 14), centred);
                if (chosen)
                    Arrow(g, pick, SectionTick(row, theta, -3), Towards(section, SectionTick(row, theta), 6));
            }

            // the plan: a circle, a 15° scale, the sample as a square; the azimuths of the chosen ring
            var ring = current == null ? Array.Empty<Incidence>() : Ring(incidences, current.Theta);
            var normal = ring.Length <= 1;
            using var circle = new Pen(normal ? Off : Color.Black, 1.2f);
            g.DrawEllipse(circle, plan.X - PlanRadius, plan.Y - PlanRadius, 2 * PlanRadius, 2 * PlanRadius);
            for (var p = 0; p < 360; p += 15)
                g.DrawLine(faint, PlanTick(row, p, -2), PlanTick(row, p, 2));
            g.FillRectangle(Brushes.White, plan.X - 8, plan.Y - 8, 16, 16);
            g.DrawRectangle(circle, plan.X - 8, plan.Y - 8, 16, 16);
            if (normal)
            {
                using var dot = new SolidBrush(current == null ? Off : pick);
                g.FillEllipse(dot, plan.X - 2.5f, plan.Y - 2.5f, 5, 5);
            }
            else
                foreach (var direction in ring)
                {
                    var chosen = Math.Abs(direction.Phi - current!.Phi) <= Same;
                    using var pen = new Pen(chosen ? pick : Color.Black, chosen ? 2.4f : 0.9f) { StartCap = LineCap.Round, EndCap = LineCap.Round };
                    g.DrawLine(pen, PlanTick(row, direction.Phi, chosen ? -5 : -3), PlanTick(row, direction.Phi, chosen ? 6 : 3));
                    if (chosen)
                        Arrow(g, pick, PlanTick(row, direction.Phi, -3), Towards(plan, PlanTick(row, direction.Phi), 11));
                }

            // the chosen angles under the dials
            var labels = row.Top + Height - 8;
            g.DrawString(current == null ? "θ" : $"θ {Degrees(current.Theta)}", bold, Brushes.Black, new PointF(section.X, labels), centred);
            g.DrawString(normal ? "φ –" : $"φ {Degrees(current!.Phi)}", bold, normal ? Brushes.DimGray : Brushes.Black, new PointF(plan.X, labels), centred);
            g.SmoothingMode = state;
        }
    }
}
