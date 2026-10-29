using System;
using System.Collections.Generic;
using System.Linq;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>Axis ticks and curve decimation of Plot 2D.</summary>
    public static class Ticks
    {
        /// <summary>A curve is drawn with the minimum and maximum of each of this many bins along the x axis.</summary>
        public const int CurveBins = 1000;

        public static double OrderOfMagnitude(double value) => Math.Pow(10, Math.Floor(Math.Log10(value)));

        public static double Remap(double value, double fromMin, double fromMax, double toMin, double toMax) =>
            toMin + (toMax - toMin) * ((value - fromMin) / (fromMax - fromMin));

        private static bool IsClose(double a, double b) => Math.Abs(a - b) <= Math.Max(1e-9 * Math.Max(Math.Abs(a), Math.Abs(b)), 0.0);

        /// <summary>Heckbert's label placement: a "nice" step that cuts ``range`` in about ``segments`` steps.</summary>
        public static double NiceStep(double range, int segments = 5)
        {
            var raw = range / segments;
            var magnitude = OrderOfMagnitude(raw);
            var significand = raw / magnitude;
            if (significand < 1.5) return 1.0 * magnitude;
            if (significand < 3.0) return 2.0 * magnitude;
            if (significand < 7.0) return 5.0 * magnitude;
            return 10.0 * magnitude;
        }

        private static int Decimals(double step) => Math.Max(0, -(int)Math.Floor(Math.Log10(step)));

        /// <summary>Nice ticks from at or below ``low`` to at or above ``high``.</summary>
        public static double[] Linear(double low, double high) => Linear(low, high, out _);

        /// <param name="integral">Whether the ticks are whole numbers, which Python labelled as ints.</param>
        public static double[] Linear(double low, double high, out bool integral)
        {
            integral = false;
            if (IsClose(high, low))
                return new[] { low };
            var step = NiceStep(high - low);
            var decimals = Decimals(step);
            integral = decimals == 0;
            var first = (int)Math.Floor(low / step + 1e-9);
            var last = (int)Math.Ceiling(high / step - 1e-9);
            return Enumerable.Range(first, last - first + 1).Select(i => Math.Round(i * step, decimals, MidpointRounding.ToEven)).ToArray();
        }

        /// <summary>The y ticks of a bands plot (calculate_ticks): around the values, a step below a minimum that would lie on the axis.</summary>
        public static double[] ForValues(IReadOnlyList<double> values) => ForValues(values, out _);

        public static double[] ForValues(IReadOnlyList<double> values, out bool integral)
        {
            integral = false;
            if (values.Count == 0)
                return new[] { 0.0 };
            var rawMin = values.Min();
            var rawMax = values.Max();
            var range = rawMax - rawMin;
            if (IsClose(range, 0.0))
                range = !IsClose(rawMax, 0.0) ? Math.Abs(rawMax) : 1.0;
            var magnitude = OrderOfMagnitude(range);
            var approxMax = Math.Ceiling(rawMax / magnitude) * magnitude;
            var approxMin = Math.Floor(rawMin / magnitude) * magnitude;
            var graphRange = approxMax - approxMin;
            if (IsClose(graphRange, 0.0))
                return new[] { approxMin };
            var step = NiceStep(graphRange);
            var niceMin = Math.Floor(rawMin / step) * step;
            if (IsClose(niceMin, rawMin))
                // the curve would sit on the x axis line: drop the axis a step below
                niceMin -= step;
            var niceMax = Math.Ceiling(rawMax / step) * step;
            var decimals = step < 1.0 ? Decimals(step) : 0;
            integral = decimals == 0;
            var ticks = new List<double>();
            for (var tick = niceMin; tick <= niceMax + step * 0.001; tick += step)
                ticks.Add(Math.Round(tick, decimals, MidpointRounding.ToEven));
            return ticks.ToArray();
        }

        /// <summary>
        /// The indices of the minimum and maximum of ``values`` in each of ``bins`` along ``positions`` (0 to 1), in order:
        /// drawn through these, a curve of many more points than the chart is wide looks as it does at full resolution.
        /// </summary>
        public static int[] MinMaxIndices(IReadOnlyList<double> positions, IReadOnlyList<double> values, int bins = CurveBins)
        {
            if (values.Count <= 2 * bins)
                return Enumerable.Range(0, values.Count).ToArray();
            var lowest = new int[bins];
            var highest = new int[bins];
            Array.Fill(lowest, -1);
            Array.Fill(highest, -1);
            for (var i = 0; i < values.Count; i++)
            {
                var bin = Math.Clamp((int)(positions[i] * bins), 0, bins - 1);
                if (lowest[bin] < 0 || values[i] < values[lowest[bin]])
                    lowest[bin] = i;
                if (highest[bin] < 0 || values[i] >= values[highest[bin]])
                    highest[bin] = i;
            }
            return lowest.Concat(highest).Where(i => i >= 0).Distinct().OrderBy(i => i).ToArray();
        }

        /// <summary>A tick as Python's ``str`` printed it: an int for whole-number ticks, else the float.</summary>
        public static string Label(double tick, bool integral) =>
            integral ? ((long)Math.Round(tick)).ToString(System.Globalization.CultureInfo.InvariantCulture) : PythonFormat.Repr(tick);
    }
}
