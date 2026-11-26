using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// A horizontal slider of fixed stops, one per option, labelled under the track (Polar Preview's octaves). A stop the component
    /// does not offer now (an octave the result did not simulate) is grey, and the knob never stops there.
    /// </summary>
    public static class StepSlider
    {
        /// <summary>The height of the slider's row: the track and its stops, the labels under them.</summary>
        public const float Height = 30;

        /// <summary>The radius of the knob, which a stop at the track's end must leave room for.</summary>
        public const float DotRadius = 5;

        /// <summary>The width a label character takes at the labels' size, to leave room for the end labels.</summary>
        public const float CharWidth = 4.5f;

        private const float EdgeMargin = 3;
        private static readonly Color Knob = Color.FromArgb(31, 98, 208);
        private static readonly Color Off = Color.FromArgb(165, 165, 165);

        /// <summary>The y of the track in a row.</summary>
        public static float TrackY(RectangleF row) => row.Top + 9;

        /// <summary>The text under stop <paramref name="i"/>: its label, else its option.</summary>
        private static string LabelOf(StepSliderControl slider, int i) => slider.Labels?[i] ?? slider.Options[i];

        /// <summary>How far the track's ends stand from the row's edges: half the wider end label, the knob's radius and a margin.</summary>
        public static float Inset(StepSliderControl slider)
        {
            var n = slider.Options.Length;
            return Math.Max(LabelOf(slider, 0).Length, LabelOf(slider, n - 1).Length) * CharWidth / 2 + DotRadius + EdgeMargin;
        }

        /// <summary>The x of stop <paramref name="i"/>, evenly along the track.</summary>
        public static float StopX(RectangleF row, int i, StepSliderControl slider)
        {
            var count = slider.Options.Length;
            var inset = Inset(slider);
            return count <= 1 ? row.Left + row.Width / 2 : row.Left + inset + i * (row.Width - 2 * inset) / (count - 1);
        }

        /// <summary>
        /// The option of the offered stop nearest <paramref name="x"/>: a click or drag never lands on a stop that is not offered;
        /// null when none is.
        /// </summary>
        public static string? Nearest(RectangleF row, float x, StepSliderControl slider, IReadOnlyCollection<string>? offered)
        {
            var options = slider.Options;
            string? best = null;
            var distance = float.MaxValue;
            for (var i = 0; i < options.Length; i++)
            {
                if (offered != null && !offered.Contains(options[i]))
                    continue;
                var d = Math.Abs(StopX(row, i, slider) - x);
                if (d < distance)
                    (best, distance) = (options[i], d);
            }
            return best;
        }

        /// <summary>
        /// Where the knob stands for <paramref name="value"/>: its stop when that is offered, else the nearest offered stop (a definition
        /// saved at 8000 Hz shows 4000 Hz for a result simulated to 5000 Hz, and solves for it).
        /// </summary>
        public static string? Shown(RectangleF row, string value, StepSliderControl slider, IReadOnlyCollection<string>? offered)
        {
            var options = slider.Options;
            var i = Array.IndexOf(options, value);
            if (i >= 0 && (offered == null || offered.Contains(value)))
                return value;
            if (i >= 0)
                return Nearest(row, StopX(row, i, slider), slider, offered);
            if (double.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var number))
            {
                var near = options.Select((o, k) => (o, k, v: double.TryParse(o, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var d) ? d : double.NaN))
                    .Where(x => !double.IsNaN(x.v) && (offered == null || offered.Contains(x.o))).OrderBy(x => Math.Abs(x.v - number)).FirstOrDefault();
                if (near.o != null)
                    return near.o;
            }
            return Nearest(row, row.Left, slider, offered);
        }

        /// <summary>Draw the slider: the track, a stop per option (grey when not offered), the labels, and the knob at <paramref name="value"/>.</summary>
        public static void Draw(Graphics g, RectangleF row, StepSliderControl slider, string? value, IReadOnlyCollection<string>? offered, bool wired)
        {
            var state = g.SmoothingMode;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            var y = TrackY(row);
            var options = slider.Options;
            using var track = new Pen(Color.FromArgb(120, 120, 120), 2);
            g.DrawLine(track, StopX(row, 0, slider), y, StopX(row, options.Length - 1, slider), y);
            using var font = new Font(FontFamily.GenericSansSerif, 6f);
            using var centred = new StringFormat { Alignment = StringAlignment.Center };
            for (var i = 0; i < options.Length; i++)
            {
                var x = StopX(row, i, slider);
                var on = offered == null || offered.Contains(options[i]);
                using var stop = new Pen(on ? Color.Black : Off, 1);
                g.DrawLine(stop, x, y - 4, x, y + 4);
                using var label = new SolidBrush(on ? Color.Black : Off);
                g.DrawString(LabelOf(slider, i), font, label, x, y + 6, centred);
            }
            var at = value == null ? -1 : Array.IndexOf(options, value);
            if (at >= 0)
            {
                using var knob = new SolidBrush(wired ? Color.Gray : Knob);
                var x = StopX(row, at, slider);
                g.FillEllipse(knob, x - DotRadius, y - DotRadius, 2 * DotRadius, 2 * DotRadius);
            }
            g.SmoothingMode = state;
        }
    }
}
