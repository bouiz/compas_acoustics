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

        private const float Inset = 10;
        private static readonly Color Knob = Color.FromArgb(31, 98, 208);
        private static readonly Color Off = Color.FromArgb(165, 165, 165);

        /// <summary>The y of the track in a row.</summary>
        public static float TrackY(RectangleF row) => row.Top + 9;

        /// <summary>The x of stop <paramref name="i"/> of <paramref name="count"/>, evenly along the track.</summary>
        public static float StopX(RectangleF row, int i, int count) =>
            count <= 1 ? row.Left + row.Width / 2 : row.Left + Inset + i * (row.Width - 2 * Inset) / (count - 1);

        /// <summary>
        /// The option of the offered stop nearest <paramref name="x"/>: a click or drag never lands on a stop that is not offered;
        /// null when none is.
        /// </summary>
        public static string? Nearest(RectangleF row, float x, IReadOnlyList<string> options, IReadOnlyCollection<string>? offered)
        {
            string? best = null;
            var distance = float.MaxValue;
            for (var i = 0; i < options.Count; i++)
            {
                if (offered != null && !offered.Contains(options[i]))
                    continue;
                var d = Math.Abs(StopX(row, i, options.Count) - x);
                if (d < distance)
                    (best, distance) = (options[i], d);
            }
            return best;
        }

        /// <summary>
        /// Where the knob stands for <paramref name="value"/>: its stop when that is offered, else the nearest offered stop (a definition
        /// saved at 8000 Hz shows 4000 Hz for a result simulated to 5000 Hz, and solves for it).
        /// </summary>
        public static string? Shown(RectangleF row, string value, IReadOnlyList<string> options, IReadOnlyCollection<string>? offered)
        {
            var i = options.ToList().IndexOf(value);
            if (i >= 0 && (offered == null || offered.Contains(value)))
                return value;
            return i >= 0 ? Nearest(row, StopX(row, i, options.Count), options, offered) : Nearest(row, row.Left, options, offered);
        }

        /// <summary>Draw the slider: the track, a stop per option (grey when not offered), the labels, and the knob at <paramref name="value"/>.</summary>
        public static void Draw(Graphics g, RectangleF row, StepSliderControl slider, string? value, IReadOnlyCollection<string>? offered, bool wired)
        {
            var state = g.SmoothingMode;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            var y = TrackY(row);
            var options = slider.Options;
            using var track = new Pen(Color.FromArgb(120, 120, 120), 2);
            g.DrawLine(track, StopX(row, 0, options.Length), y, StopX(row, options.Length - 1, options.Length), y);
            using var font = new Font(FontFamily.GenericSansSerif, 6f);
            using var centred = new StringFormat { Alignment = StringAlignment.Center };
            for (var i = 0; i < options.Length; i++)
            {
                var x = StopX(row, i, options.Length);
                var on = offered == null || offered.Contains(options[i]);
                using var stop = new Pen(on ? Color.Black : Off, 1);
                g.DrawLine(stop, x, y - 4, x, y + 4);
                using var label = new SolidBrush(on ? Color.Black : Off);
                g.DrawString(slider.Labels?[i] ?? options[i], font, label, x, y + 6, centred);
            }
            var at = value == null ? -1 : Array.IndexOf(options, value);
            if (at >= 0)
            {
                using var knob = new SolidBrush(wired ? Color.Gray : Knob);
                var x = StopX(row, at, options.Length);
                g.FillEllipse(knob, x - 5, y - 5, 10, 10);
            }
            g.SmoothingMode = state;
        }
    }
}
