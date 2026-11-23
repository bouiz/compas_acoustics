using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Linq;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The material charts: absorption and scattering side by side per band; a click sets a band's value.</summary>
    public static class MaterialChart
    {
        /// <summary>The colours of the absorption and the scattering bars, on the body and in the picker.</summary>
        public static readonly (int r, int g, int b) AbsorptionRgb = (60, 60, 60), ScatteringRgb = (110, 165, 220);

        /// <summary>The octave bands from 125 Hz, as the charts label them: 125, 250, 500, 1k ... 8k.</summary>
        public static string Band(int index)
        {
            var hz = 125 * Math.Pow(2, index);
            return hz >= 1000 ? (hz / 1000).ToString("0.#", CultureInfo.InvariantCulture) + "k" : hz.ToString("0", CultureInfo.InvariantCulture);
        }

        /// <summary>The compact chart of a material: its plot area, the bands below, 0 and 1 beside.</summary>
        public sealed record StripChart(RectangleF Plot, (float x, string text)[] Bands, (float y, string text)[] Values);

        /// <summary>The compact chart in <paramref name="area"/>, <paramref name="labelWidth"/> the width of the value labels on its left.</summary>
        public static StripChart Strip(IReadOnlyList<double> absorption, IReadOnlyList<double>? scattering, RectangleF area, float labelWidth)
        {
            var plot = ChartLayout.CompactPlot(area, 2 + labelWidth + ChartLayout.TickGap);
            var bands = Math.Max(Math.Max(absorption.Count, scattering?.Count ?? 0), 1);
            var step = plot.Width / bands;
            return new StripChart(plot,
                Enumerable.Range(0, bands).Select(i => (plot.Left + step * (i + 0.5f), Band(i))).ToArray(),
                new[] { (plot.Bottom, "0"), (plot.Top, "1") });
        }

        /// <summary>A typed coefficient: a number (a comma as the decimal mark too), within 0 and 1, to 0.01; null for anything else.</summary>
        public static double? Parse(string? text)
        {
            var t = (text ?? "").Trim().Replace(',', '.');
            if (t.Length == 0 || !double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out var v)
                || !double.IsFinite(v))
                return null;
            return Math.Round(Math.Clamp(v, 0, 1), 2);
        }

        /// <summary>A coefficient as its box shows it.</summary>
        public static string Format(double value) => value.ToString("0.00", CultureInfo.InvariantCulture);

        /// <summary>The coefficient axis's ticks, 0 to 1 by 0.2.</summary>
        public static readonly double[] Ticks = { 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };

        /// <summary>The band and the value (0 at the bottom, 1 at the top, to 0.01) under <paramref name="point"/>.</summary>
        public static (int band, double value) Hit(RectangleF plot, int bands, PointF point)
        {
            var band = (int)Math.Clamp(Math.Floor((point.X - plot.Left) / (plot.Width / Math.Max(bands, 1))), 0, Math.Max(bands - 1, 0));
            var value = Math.Round(Math.Clamp((plot.Bottom - point.Y) / Math.Max(plot.Height, 1), 0, 1), 2);
            return (band, value);
        }

        /// <summary>Per band, the absorption bar and beside it the scattering bar, from 0 to 1 of the area's height.</summary>
        public static (RectangleF[] absorption, RectangleF[] scattering) PairedBars(IReadOnlyList<double> absorption, IReadOnlyList<double>? scattering, RectangleF area)
        {
            var bands = Math.Max(absorption.Count, scattering?.Count ?? 0);
            if (bands == 0)
                return (Array.Empty<RectangleF>(), Array.Empty<RectangleF>());
            var slot = area.Width / bands;
            var width = Math.Max((slot - 2) / 2, 1);
            RectangleF Bar(double v, int i, int side)
            {
                var h = (float)Math.Clamp(v, 0, 1) * area.Height;
                return new RectangleF(area.Left + i * slot + 1 + side * width, area.Bottom - h, width, h);
            }
            var a = absorption.Select((v, i) => Bar(v, i, 0)).ToArray();
            var s = scattering?.Select((v, i) => Bar(v, i, 1)).ToArray() ?? Array.Empty<RectangleF>();
            return (a, s);
        }
    }
}
