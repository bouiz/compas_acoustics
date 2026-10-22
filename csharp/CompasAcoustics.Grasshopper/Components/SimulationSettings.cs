using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The settings Simulate Pyroomacoustics passes to the method, and its message.</summary>
    public static class SimulationSettings
    {
        // inputs that only apply with ray tracing, by the method's names
        private static readonly string[] RayTracingOnly = { "n_rays", "energy_thres", "duration", "seed" };

        /// <summary>
        /// Unset inputs are left out so that the method's own defaults apply; air absorption is always stated (default on).
        /// Without ray tracing its inputs are dropped and named in ``ignored``; with it, an empty seed draws a new one.
        /// </summary>
        public static Dictionary<string, object?> Pyroomacoustics(int? ismOrder, bool rayTracing, int? nRays, bool? absorbAir, double? energyThres,
            double? timeThres, int? seed, bool? normalize, out string[] ignored, out int drawnSeed)
        {
            var given = new Dictionary<string, object?>
            {
                ["image_source_order"] = ismOrder is int order ? (long)order : null,
                ["n_rays"] = nRays is int rays ? (long)rays : null,
                ["air_absorption"] = absorbAir ?? true,
                ["energy_thres"] = energyThres,
                ["duration"] = timeThres,
                ["seed"] = seed is int s ? (long)s : null,
                ["normalize"] = normalize,
            };
            var settings = given.Where(kv => kv.Value != null).ToDictionary(kv => kv.Key, kv => kv.Value);
            ignored = Array.Empty<string>();
            drawnSeed = seed ?? 0;
            if (!rayTracing)
            {
                ignored = RayTracingOnly.Where(settings.ContainsKey).ToArray();
                foreach (var key in RayTracingOnly)
                    settings.Remove(key);
            }
            else if (!settings.ContainsKey("seed"))
            {
                // the method's own default is a fixed seed; an empty input draws a new one each run
                drawnSeed = Random.Shared.Next(int.MaxValue);
                settings["seed"] = (long)drawnSeed;
            }
            settings["ray_tracing"] = rayTracing;
            return settings;
        }

        public static string PyroomacousticsMessage(int samplingRate, IReadOnlyDictionary<string, object?> settings)
        {
            var rayTracing = settings.TryGetValue("ray_tracing", out var rt) && rt is true;
            string Get(string key, string fallback) => settings.TryGetValue(key, out var v) && v != null ? Show(v) : fallback;
            return string.Format(CultureInfo.InvariantCulture, "{0}Hz, {1}, order={2}, rays={3}, air={4}, seed={5}",
                samplingRate, rayTracing ? "ISM + ray tracing" : "ISM", Get("image_source_order", "default"),
                rayTracing ? Get("n_rays", "default") : "-", Get("air_absorption", "default"), rayTracing ? Get("seed", "-") : "-");
        }

        /// <summary>A value as Python's ``str`` shows it.</summary>
        internal static string Show(object value) => value switch
        {
            bool b => b ? "True" : "False",
            double d => PythonFormat.G(d, 17) is var g && g.Contains('.') || g.Contains('e') ? g : g + ".0",
            IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
            _ => value.ToString() ?? "",
        };
    }
}
