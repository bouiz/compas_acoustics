using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Linq;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The Audio Player's picture of a signal: a min/max envelope per pixel column and channel, the lanes and the time ruler.</summary>
    public static class Waveform
    {
        /// <param name="samples">Sample-major, <paramref name="channels"/> per frame, as Audio messages hold them.</param>
        public static (float min, float max)[][] Envelope(IReadOnlyList<double> samples, int channels, int columns)
        {
            channels = Math.Max(channels, 1);
            columns = Math.Max(columns, 1);
            var frames = samples.Count / channels;
            var envelope = new (float, float)[channels][];
            for (var c = 0; c < channels; c++)
            {
                envelope[c] = new (float, float)[columns];
                if (frames == 0)
                    continue;
                for (var col = 0; col < columns; col++)
                {
                    var first = (int)((long)col * frames / columns);
                    var last = Math.Max(first + 1, (int)((long)(col + 1) * frames / columns));
                    var min = double.MaxValue;
                    var max = double.MinValue;
                    for (var f = first; f < last && f < frames; f++)
                    {
                        var v = samples[f * channels + c];
                        if (v < min) min = v;
                        if (v > max) max = v;
                    }
                    envelope[c][col] = ((float)min, (float)max);
                }
            }
            return envelope;
        }

        /// <summary>The gap between two channels' lanes.</summary>
        public const float LaneGap = 2;

        public static RectangleF[] Lanes(RectangleF area, int channels, float gap = LaneGap)
        {
            channels = Math.Max(channels, 1);
            var height = (area.Height - gap * (channels - 1)) / channels;
            return Enumerable.Range(0, channels).Select(i => new RectangleF(area.Left, area.Top + i * (height + gap), area.Width, height)).ToArray();
        }

        public static (float x, string label)[] Ticks(double durationSeconds, float width)
        {
            if (durationSeconds <= 0)
                return new[] { (0f, "0") };
            return Grasshopper.Ticks.Linear(0, durationSeconds).Where(t => t <= durationSeconds + 1e-9)
                .Select(t => ((float)(t / durationSeconds * width), PythonFormat.G(t))).ToArray();
        }

        public static string Header(string format, int channels, int samplingRate, double durationSeconds, double peakDbfs) =>
            string.Format(CultureInfo.InvariantCulture, "{0}, {1} ch, {2} kHz, {3:0.00} s, peak {4:0.0} dBFS",
                format, channels, PythonFormat.G(samplingRate / 1000.0, 4), durationSeconds, peakDbfs);
    }

    /// <summary>Where the parts of the Audio Player's body are.</summary>
    public sealed record PlayerBoxes(RectangleF Header, RectangleF Lanes, RectangleF Ruler, RectangleF PlayPause, RectangleF Stop,
        RectangleF Normalize, RectangleF Volume, RectangleF VolumeLabel);

    /// <summary>What a press on the Audio Player lands on; None moves the component, as a press on its icon does.</summary>
    public enum PlayerHit { None, PlayPause, Stop, Normalize, Volume, Lanes }

    public static class PlayerLayout
    {
        public const float Margin = 4;
        public const float HeaderHeight = 12;
        public const float RulerHeight = 10;
        public const float TransportHeight = 24;
        public const float ButtonWidth = 34;
        /// <summary>The Normalize button's tooltip: what it does, with its default.</summary>
        public static readonly (string title, string text) NormalizeTooltip = ("normalize",
            "Whether to scale the signal to full level before playing (default off). Leave it off to hear auralised signals at their rendered level.");

        /// <summary>Between the Play/Stop row and the Normalize row.</summary>
        public const float RowGap = 3;
        /// <summary>The row of the Volume label, above the slider: a control title's, in the font of the input names.</summary>
        public const float VolumeLabelHeight = ControlLayout.TitleHeight;

        /// <summary>The transport beside the icon, top right: Play/Pause and Stop, Normalize below them, the volume to their right.</summary>
        public const float ControlsHeight = Margin + TransportHeight + RowGap + TransportHeight + Margin;

        /// <summary>The channels' height below the inputs, for <paramref name="channels"/> lanes of <paramref name="laneHeight"/> each: more channels, a taller component.</summary>
        public static float WaveHeight(int channels, float laneHeight)
        {
            channels = Math.Max(channels, 1);
            var lanes = channels * laneHeight + (channels - 1) * Waveform.LaneGap;
            return HeaderHeight + 2 + lanes + RulerHeight + Margin;
        }

        /// <summary>The boxes: the transport in <paramref name="controls"/>; the header, lanes and ruler across <paramref name="wave"/>, none when it is empty (no signal).</summary>
        public static PlayerBoxes Arrange(RectangleF controls, RectangleF wave)
        {
            var left = controls.Left + Margin;
            var top = controls.Top + Margin;
            RectangleF Button(int i) => new(left + i * (ButtonWidth + 2), top, ButtonWidth, TransportHeight);
            var normalize = new RectangleF(left, top + TransportHeight + RowGap, 2 * ButtonWidth + 2, TransportHeight);
            // the label and the slider, centred on the two rows of buttons
            var volumeLeft = normalize.Right + 6;
            var labelTop = (top + normalize.Bottom) / 2 - (VolumeLabelHeight + TransportHeight) / 2;
            var volumeLabel = new RectangleF(volumeLeft, labelTop, Math.Max(controls.Right - Margin - volumeLeft, 20), VolumeLabelHeight);
            var volume = new RectangleF(volumeLeft, volumeLabel.Bottom, volumeLabel.Width, TransportHeight);
            if (wave.IsEmpty)
                return new PlayerBoxes(RectangleF.Empty, RectangleF.Empty, RectangleF.Empty, Button(0), Button(1), normalize, volume, volumeLabel);
            var width = wave.Width - 2 * Margin;
            var header = new RectangleF(wave.Left + Margin, wave.Top, width, HeaderHeight);
            var ruler = new RectangleF(wave.Left + Margin, wave.Bottom - Margin - RulerHeight, width, RulerHeight);
            var lanes = new RectangleF(wave.Left + Margin, header.Bottom + 2, width, Math.Max(ruler.Top - header.Bottom - 2, 10));
            return new PlayerBoxes(header, lanes, ruler, Button(0), Button(1), normalize, volume, volumeLabel);
        }

        /// <summary>The control under <paramref name="point"/>: a button, the volume, the lanes (a seek); None for anything else.</summary>
        public static PlayerHit Hit(PlayerBoxes boxes, PointF point) =>
            boxes.PlayPause.Contains(point) ? PlayerHit.PlayPause
            : boxes.Stop.Contains(point) ? PlayerHit.Stop
            : boxes.Normalize.Contains(point) ? PlayerHit.Normalize
            : boxes.Volume.Contains(point) ? PlayerHit.Volume
            : !boxes.Lanes.IsEmpty && boxes.Lanes.Contains(point) ? PlayerHit.Lanes
            : PlayerHit.None;

        public static double SeekTime(RectangleF lanes, float x, double duration) =>
            Math.Clamp((x - lanes.Left) / Math.Max(lanes.Width, 1), 0, 1) * duration;

        public static float PlayheadX(RectangleF lanes, double position, double duration) =>
            duration <= 0 ? lanes.Left : lanes.Left + (float)(Math.Clamp(position / duration, 0, 1) * lanes.Width);
    }
}
