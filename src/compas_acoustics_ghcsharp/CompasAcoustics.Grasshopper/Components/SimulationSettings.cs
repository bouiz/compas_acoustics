using System;
using System.Collections.Generic;
using System.Linq;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The settings Simulate Mono passes to the mono method (pyroomacoustics), and its message.</summary>
    public static class SimulationSettings
    {
        // inputs that only apply with ray tracing, by the method's names
        private static readonly string[] RayTracingOnly = { "n_rays", "duration", "seed" };

        /// <summary>
        /// Unset inputs are left out so that the method's own defaults apply; air absorption is always on, as in the hybrid methods.
        /// Without ray tracing its inputs are dropped and named in <paramref name="ignored"/>.
        /// </summary>
        public static Dictionary<string, object?> Mono(int? ismOrder, bool rayTracing, int? nRays, double? duration, int? seed, bool? normalize,
            out string[] ignored)
        {
            var given = new Dictionary<string, object?>
            {
                ["image_source_order"] = ismOrder is int order ? (long)order : null,
                ["n_rays"] = nRays is int rays ? (long)rays : null,
                ["air_absorption"] = true,
                ["duration"] = duration,
                ["seed"] = seed is int s ? (long)s : null,
                ["normalize"] = normalize,
            };
            var settings = given.Where(kv => kv.Value != null).ToDictionary(kv => kv.Key, kv => kv.Value);
            ignored = Array.Empty<string>();
            if (!rayTracing)
            {
                ignored = RayTracingOnly.Where(settings.ContainsKey).ToArray();
                foreach (var key in RayTracingOnly)
                    settings.Remove(key);
            }
            settings["ray_tracing"] = rayTracing;
            return settings;
        }

        /// <summary>The message of a run: the rate, the method, the image source order, and with ray tracing the rays and seed.</summary>
        public static string MonoMessage(int samplingRate, IReadOnlyDictionary<string, object?> settings)
        {
            var rayTracing = settings.TryGetValue("ray_tracing", out var rt) && rt is true;
            string Get(string key) => settings.TryGetValue(key, out var v) && v != null ? MessageText.Show(v) : "default";
            var rays = rayTracing ? $", rays {Get("n_rays")}, seed {Get("seed")}" : "";
            return $"{MessageText.Rate(samplingRate)}, {(rayTracing ? "ISM + ray tracing" : "ISM")}, ISM order {Get("image_source_order")}{rays}";
        }
    }
}
