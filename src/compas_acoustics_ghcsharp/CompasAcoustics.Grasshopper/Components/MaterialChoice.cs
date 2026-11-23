using System;
using System.Linq;
using System.Text.Json;
using CompasAcoustics.Pb;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// What the Material Library gives: a library material as it is (<c>Library</c> set, not <c>Edited</c>), one whose
    /// coefficients the picker changed, or a custom one (<c>Library</c> null). Saved with the document.
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

        /// <summary>The choice with band <paramref name="band"/> of its absorption (or scattering) set to <paramref name="value"/>, within 0 and 1.</summary>
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
            var message = result.ScatteringCoefficients.Count == 0 ? $"scattering {MaterialChart.Format(DefaultScattering)} (default)" : null;
            var warning = choice.IsCustom && !absorptionWired && choice.HasNoAbsorption
                ? "The custom material has no absorption: every band is 0, a surface that reflects all sound. Set its absorption in the picker " +
                  "(double-click the component) or connect 'absorption'."
                : null;
            return (message, warning);
        }
    }
}
