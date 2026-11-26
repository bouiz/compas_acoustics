using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Linq;

namespace CompasAcoustics.Grasshopper
{
    public sealed record ChartBar(RectangleF Bar, string Label, double Frequency, double Value);

    public sealed record XTick(float X, string Text);

    public sealed record YTick(float Y, string Text);

    /// <summary>
    /// A chart of one acoustic parameter in canvas pixels (y down): a line through a dot per band (<c>Dots</c>), or a line per
    /// channel for a time or frequency response. <c>Bars</c> place each band, for the line and the readout; <c>Data</c> holds
    /// the values of each line's points.
    /// </summary>
    public sealed record Chart(string Title, RectangleF Plot, IReadOnlyList<ChartBar> Bars, IReadOnlyList<PointF[]> Lines,
        IReadOnlyList<(double x, double y)[]> Data, IReadOnlyList<XTick> XTicks, IReadOnlyList<YTick> YTicks, string XTitle, string YTitle,
        string? Overall, float? ZeroY, double YLow, double YHigh, IReadOnlyList<string> Channels, bool Dots = false);

    /// <summary>The layout of the charts on the canvas (Plot 2D): margins, ticks and ranges.</summary>
    public static class ChartLayout
    {
        /// <summary>The margins around the plot: the y title and the y ticks on the left, the title above, the x ticks and title below.</summary>
        public const float Left = 46, Right = 10, Top = 18, Bottom = 26;

        /// <summary>The left margin that leaves the y title as far from the area's edge as the plot from its right edge.</summary>
        public static float LeftMargin(float ticksWidth, float titleHeight) => Right + titleHeight + TitleGap + ticksWidth + TickGap;

        /// <summary>The gap between the y axis and its tick labels, and between those and the y title.</summary>
        public const float TickGap = 2, TitleGap = 4;

        /// <summary>
        /// Where the y title goes, reading upwards: <see cref="TitleGap"/> left of the tick labels (<paramref name="ticksWidth"/>
        /// wide), centred on the plot, inside <paramref name="area"/>. <paramref name="text"/> is the size of the title written
        /// level; the box is that size turned a quarter.
        /// </summary>
        public static RectangleF YTitleBox(RectangleF area, RectangleF plot, SizeF text, float ticksWidth)
        {
            var height = Math.Min(text.Width, area.Height - 4);
            var top = Math.Clamp(plot.Top + plot.Height / 2 - height / 2, area.Top + 2, area.Bottom - 2 - height);
            var right = Math.Max(plot.Left - ticksWidth - TickGap - TitleGap, area.Left + 2 + text.Height);
            return new RectangleF(right - text.Height, top, text.Height, height);
        }

        /// <summary>The pixels of a <paramref name="width"/> x <paramref name="height"/> image (row by row) turned a quarter turn counter-clockwise.</summary>
        public static int[] TurnUpward(int width, int height, int[] pixels)
        {
            var turned = new int[pixels.Length];
            for (var y = 0; y < height; y++)
                for (var x = 0; x < width; x++)
                    turned[(width - 1 - x) * height + y] = pixels[y * width + x];
            return turned;
        }

        private static readonly double[] FrequencyAxis = { 20.0, 20000.0 };
        private static readonly double[] FrequencyTicks = { 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000 };
        private const float BarFill = 0.6f;

        /// <summary>
        /// The chart of <paramref name="p"/> in <paramref name="area"/>. <paramref name="left"/> replaces the default left
        /// margin. <paramref name="compact"/>: a strip, without titles and with three value labels.
        /// </summary>
        public static Chart Build(ParameterGoo.Content p, RectangleF area, float? left = null, bool compact = false)
        {
            var margin = left ?? (compact ? 20 : Left);
            var plot = compact ? CompactPlot(area, margin)
                : new RectangleF(area.Left + margin, area.Top + Top, Math.Max(area.Width - margin - Right, 1), Math.Max(area.Height - Top - Bottom, 1));
            var overall = Overall(p);
            var chart = p.XScale == "bands" ? BandChart(p, plot, overall) : Curve(p, plot, overall);
            if (!compact)
                return chart;
            var ends = new[] { chart.YLow, (chart.YLow + chart.YHigh) / 2, chart.YHigh };
            return chart with { YTicks = ends.Select(v => new YTick(Y(chart, v), v.ToString("G3", CultureInfo.InvariantCulture))).ToList() };
        }

        /// <summary>The height of a body with a compact chart (Material Library), and of its header row.</summary>
        public const float StripHeight = 60, StripHeader = 11;

        /// <summary>The overall value as a chart shows it; null without one.</summary>
        private static string? Overall(ParameterGoo.Content p) => p.Overall is double o ? $"overall {PythonFormat.Repr(o)}" : null;

        /// <summary>Where a body of <see cref="StripHeight"/> draws its compact chart: below its header row.</summary>
        public static RectangleF StripArea(RectangleF body) =>
            new(body.Left + 3, body.Top + StripHeader, body.Width - 6, body.Height - StripHeader - 2);

        /// <summary>The compact chart's margins: a little room above, the band labels below.</summary>
        public const float CompactTop = 4, CompactRight = 6, CompactBottom = 10;

        /// <summary>The plot of a compact chart in <paramref name="area"/>, <paramref name="left"/> the margin of its value labels.</summary>
        public static RectangleF CompactPlot(RectangleF area, float left) =>
            new(area.Left + left, area.Top + CompactTop, Math.Max(area.Width - left - CompactRight, 1), Math.Max(area.Height - CompactTop - CompactBottom, 1));

        /// <summary>The reverberation times, whose axis starts at 0.</summary>
        private static readonly HashSet<string> ReverberationTimes = new() { "EDT", "T20", "T30" };

        /// <summary>
        /// The y range a parameter always has, whatever its values: STI is an index and LF is a fraction from 0 to
        /// 1, a reverberation time starts at 0.
        /// </summary>
        private static double[]? FixedTicks(ParameterGoo.Content p, out bool integral)
        {
            integral = false;
            if (p.Name is "STI" or "LF")
                return Ticks.Linear(0, 1, out integral);
            var values = p.Values.Length > 0 ? p.Values[0].Where(double.IsFinite).ToArray() : Array.Empty<double>();
            if (ReverberationTimes.Contains(p.Name) && values.Length > 0 && values.Max() > 0)
                return Ticks.Linear(0, values.Max(), out integral);
            return null;
        }

        public static float Y(Chart chart, double value) => Y(chart.Plot, chart.YLow, chart.YHigh, value);

        private static float Y(RectangleF plot, double low, double high, double value) =>
            plot.Bottom - (float)Ticks.Remap(value, low, high, 0, plot.Height);

        private static (List<YTick> ticks, double low, double high, float? zero) YAxis(RectangleF plot, double[] ticks, bool integral)
        {
            var (low, high) = (ticks.Min(), ticks.Max());
            if (high <= low)
                (low, high) = (low - 1, high + 1);
            var list = ticks.Select(t => new YTick(Y(plot, low, high, t), Ticks.Label(t, integral))).ToList();
            float? zero = low < 0 && 0 < high ? Y(plot, low, high, 0) : null;
            return (list, low, high, zero);
        }

        private static Chart BandChart(ParameterGoo.Content p, RectangleF plot, string? overall)
        {
            var values = p.Values.Length > 0 ? p.Values[0] : Array.Empty<double>();
            var count = Math.Min(values.Length, p.Frequencies.Length);
            var integral = false;
            var ticks = FixedTicks(p, out integral) ?? (values.Length > 0 ? Ticks.ForValues(values, out integral) : new[] { 0.0, 1.0 });
            var (yTicks, low, high, zero) = YAxis(plot, ticks, integral);
            var step = plot.Width / Math.Max(count, 1);
            var floor = zero ?? plot.Bottom;
            var bars = Enumerable.Range(0, count).Select(i =>
            {
                var centre = plot.Left + step * (i + 0.5f);
                var y = Y(plot, low, high, values[i]);
                var width = step * BarFill;
                return new ChartBar(new RectangleF(centre - width / 2, Math.Min(y, floor), width, Math.Abs(floor - y)), PythonFormat.G(p.Frequencies[i]),
                    p.Frequencies[i], values[i]);
            }).ToList();
            var xTicks = bars.Select(b => new XTick(b.Bar.Left + b.Bar.Width / 2, b.Label)).ToList();
            var line = bars.Select(b => new PointF(b.Bar.Left + b.Bar.Width / 2, Y(plot, low, high, b.Value))).ToArray();
            var data = bars.Select(b => (b.Frequency, b.Value)).ToArray();
            return new Chart(p.Title, plot, bars, new[] { line }, new[] { data }, xTicks, yTicks, p.XUnit, p.YUnit, overall, zero, low, high,
                p.Channels, Dots: true);
        }

        /// <summary>
        /// Linear from nice ticks around the data (time), or logarithmic over 20 Hz to 20 kHz (frequency); a signed curve (an
        /// amplitude) gets a range symmetric about zero, a level in dB runs from its floor to its peak. Each line keeps the
        /// extremes of each pixel column.
        /// </summary>
        private static Chart Curve(ParameterGoo.Content p, RectangleF plot, string? overall)
        {
            var xs = p.Frequencies;
            Func<double, double> position;
            List<(double value, string label)> ticks;
            if (p.XScale == "log")
            {
                var (low, high) = (FrequencyAxis[0], FrequencyAxis[1]);
                ticks = FrequencyTicks.Where(f => low <= f && f <= high).Select(f => (f, PythonFormat.G(f))).ToList();
                position = v => (Math.Log10(v) - Math.Log10(low)) / (Math.Log10(high) - Math.Log10(low));
            }
            else
            {
                var values = xs.Length > 0 ? Ticks.Linear(xs[0], xs[^1]) : new[] { 0.0, 1.0 };
                ticks = values.Select(t => (t, PythonFormat.G(t))).ToList();
                var (low, high) = (values[0], values[^1]);
                position = v => high > low ? (v - low) / (high - low) : 0;
            }
            var positions = xs.Select(position).ToArray();
            var kept = p.Values.Select(channel => Ticks.MinMaxIndices(positions, channel, Math.Max((int)plot.Width, 1))).ToList();
            var shown = p.Values.Zip(kept, (channel, keep) => keep.Select(i => channel[i])).SelectMany(v => v).DefaultIfEmpty(0).ToList();
            double[] yValues;
            bool integral;
            if (shown.Min() < 0 && 0 < shown.Max())
            {
                var bound = Math.Max(-shown.Min(), shown.Max());
                yValues = Ticks.Linear(-bound, bound, out integral);
            }
            else
                yValues = Ticks.Linear(shown.Min(), shown.Max(), out integral);
            var (yTicks, yLow, yHigh, zero) = YAxis(plot, yValues, integral);
            float X(double pos) => plot.Left + (float)pos * plot.Width;
            var lines = p.Values.Zip(kept, (channel, keep) => keep.Select(i => new PointF(X(positions[i]), Y(plot, yLow, yHigh, channel[i]))).ToArray()).ToList();
            var data = p.Values.Zip(kept, (channel, keep) => keep.Select(i => (xs[i], channel[i])).ToArray()).ToList();
            var xTicks = ticks.Select(t => new XTick(X(position(t.value)), t.label)).ToList();
            return new Chart(p.Title, plot, Array.Empty<ChartBar>(), lines, data, xTicks, yTicks, p.XUnit, p.YUnit, overall, zero, yLow, yHigh, p.Channels);
        }

        private static string Number(double v) => v.ToString("G4", CultureInfo.InvariantCulture);

        /// <summary>What lies under the mouse: the nearest bar's band and value, or the curve's point; null outside the plot.</summary>
        public static string? Readout(Chart chart, PointF mouse)
        {
            if (mouse.X < chart.Plot.Left || mouse.X > chart.Plot.Right || mouse.Y < chart.Plot.Top - 4 || mouse.Y > chart.Plot.Bottom + 4)
                return null;
            if (chart.Bars.Count > 0)
            {
                var bar = chart.Bars.OrderBy(b => Math.Abs(b.Bar.Left + b.Bar.Width / 2 - mouse.X)).First();
                return $"{bar.Label} Hz: {PythonFormat.Repr(bar.Value)}";
            }
            if (chart.Lines.Count == 0 || chart.Lines[0].Length == 0)
                return null;
            var line = chart.Lines[0];
            var k = Enumerable.Range(0, line.Length).OrderBy(i => Math.Abs(line[i].X - mouse.X)).First();
            var x = chart.Data[0][k].x;
            if (chart.Lines.Count == 1)
                return $"{Number(x)} {chart.XTitle}, {Number(chart.Data[0][k].y)} {chart.YTitle}";
            var values = chart.Data.Select((d, c) =>
            {
                var near = d.OrderBy(point => Math.Abs(point.x - x)).First();
                return $"{(c < chart.Channels.Count ? chart.Channels[c] : $"channel {c}")} {Number(near.y)}";
            });
            return $"{Number(x)} {chart.XTitle}, {string.Join(", ", values)} {chart.YTitle}";
        }
    }
}
