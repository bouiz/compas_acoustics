using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text.Json;
using CompasAcoustics.Pb;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// What the Material Library gives: a library material as it is (``Library`` set, not ``Edited``), one whose
    /// coefficients the picker changed, or a custom one (``Library`` null). Saved with the document.
    /// </summary>
    public sealed record MaterialChoice(string? Library, string Name, string Category, double[] Absorption, double[]? Scattering, string Origin,
        bool Edited = false)
    {
        /// <summary>The octave bands from 125 Hz to 8 kHz.</summary>
        public const int Bands = 7;

        /// <summary>The scattering simulations use where a material has none.</summary>
        public const double DefaultScattering = 0.05;

        public const string CustomCategory = "Other";

        public bool IsCustom => Library == null;

        public bool AsInTheLibrary => Library != null && !Edited;

        public bool HasNoAbsorption => Absorption.All(a => a == 0);

        public static MaterialChoice NewCustom() => new(null, "custom", CustomCategory, new double[Bands], new double[Bands], "custom");

        public static MaterialChoice Of(AcousticMaterialData material) =>
            new(material.Name, material.Name, material.Category, material.AbsorptionCoefficients.ToArray(),
                material.ScatteringCoefficients.Count > 0 ? material.ScatteringCoefficients.ToArray() : null, material.DataOrigin);

        /// <summary>A library material known only by its name (a document saved before the picker edited coefficients).</summary>
        public static MaterialChoice Named(string name) => new(name, name, "", Array.Empty<double>(), null, "");

        /// <summary>The choice with band ``band`` of its absorption (or scattering) set to ``value``, within 0 and 1.</summary>
        public MaterialChoice WithBand(bool scattering, int band, double value)
        {
            value = Math.Round(Math.Clamp(value, 0, 1), 2);
            var origin = IsCustom || Edited ? Origin : string.IsNullOrEmpty(Origin) ? "modified" : $"{Origin}, modified";
            if (scattering)
            {
                var s = (Scattering ?? Enumerable.Repeat(DefaultScattering, Math.Max(Absorption.Length, Bands)).ToArray()).ToArray();
                if (band >= 0 && band < s.Length)
                    s[band] = value;
                return this with { Scattering = s, Origin = origin, Edited = true };
            }
            var a = Absorption.ToArray();
            if (band >= 0 && band < a.Length)
                a[band] = value;
            return this with { Absorption = a, Origin = origin, Edited = true };
        }

        public string Write() => JsonSerializer.Serialize(this);

        public static MaterialChoice? Read(string text)
        {
            try
            {
                return JsonSerializer.Deserialize<MaterialChoice>(text);
            }
            catch (JsonException)
            {
                return null;
            }
        }

        /// <summary>
        /// The message under the component (the default scattering of a material without its own) and the warning of a
        /// custom material whose absorption was left at 0 in every band.
        /// </summary>
        public static (string? message, string? warning) Notes(MaterialChoice choice, bool absorptionWired, AcousticMaterialData result)
        {
            var message = result.ScatteringCoefficients.Count == 0 ? "scattering 0.05 (default)" : null;
            var warning = choice.IsCustom && !absorptionWired && choice.HasNoAbsorption
                ? "The custom material has no absorption: every band is 0, a surface that reflects all sound. Set its absorption in the picker " +
                  "(double-click the component) or connect 'absorption'."
                : null;
            return (message, warning);
        }
    }

    /// <summary>The material charts: absorption and scattering side by side per band; a click sets a band's value.</summary>
    public static class MaterialChart
    {
        /// <summary>The octave bands from 125 Hz, as the charts label them: 125, 250, 500, 1k ... 8k.</summary>
        public static string Band(int index)
        {
            var hz = 125 * Math.Pow(2, index);
            return hz >= 1000 ? (hz / 1000).ToString("0.#", System.Globalization.CultureInfo.InvariantCulture) + "k"
                : hz.ToString("0", System.Globalization.CultureInfo.InvariantCulture);
        }

        /// <summary>The compact chart of a material: a line per coefficient through a dot per band, the bands below, 0 and 1 beside.</summary>
        public sealed record StripChart(RectangleF Plot, PointF[] Absorption, PointF[] Scattering, (float x, string text)[] Bands, (float y, string text)[] Values);

        /// <summary>The compact chart in ``area``, ``labelWidth`` the width of the value labels on its left.</summary>
        public static StripChart Strip(IReadOnlyList<double> absorption, IReadOnlyList<double>? scattering, RectangleF area, float labelWidth)
        {
            var left = 2 + labelWidth + ChartLayout.TickGap;
            var plot = new RectangleF(area.Left + left, area.Top + ChartLayout.CompactTop, Math.Max(area.Width - left - ChartLayout.CompactRight, 1),
                Math.Max(area.Height - ChartLayout.CompactTop - ChartLayout.CompactBottom, 1));
            var bands = Math.Max(Math.Max(absorption.Count, scattering?.Count ?? 0), 1);
            var step = plot.Width / bands;
            PointF Point(double v, int i) => new(plot.Left + step * (i + 0.5f), plot.Bottom - (float)Math.Clamp(v, 0, 1) * plot.Height);
            return new StripChart(plot, absorption.Select((v, i) => Point(v, i)).ToArray(), scattering?.Select((v, i) => Point(v, i)).ToArray() ?? Array.Empty<PointF>(),
                Enumerable.Range(0, bands).Select(i => (plot.Left + step * (i + 0.5f), Band(i))).ToArray(),
                new[] { (plot.Bottom, "0"), (plot.Top, "1") });
        }

        /// <summary>A typed coefficient: a number (a comma as the decimal mark too), within 0 and 1, to 0.01; null for anything else.</summary>
        public static double? Parse(string? text)
        {
            var t = (text ?? "").Trim().Replace(',', '.');
            if (t.Length == 0 || !double.TryParse(t, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v)
                || !double.IsFinite(v))
                return null;
            return Math.Round(Math.Clamp(v, 0, 1), 2);
        }

        /// <summary>A coefficient as its box shows it.</summary>
        public static string Format(double value) => value.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture);

        /// <summary>The coefficient axis's ticks, 0 to 1 by 0.2.</summary>
        public static readonly double[] Ticks = { 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };

        /// <summary>The band and the value (0 at the bottom, 1 at the top, to 0.01) under ``point``.</summary>
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
