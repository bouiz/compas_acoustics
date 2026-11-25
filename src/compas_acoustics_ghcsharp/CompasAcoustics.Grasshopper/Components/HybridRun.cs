using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using CompasAcoustics.Pb;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>What Simulate Binaural and Simulate Ambisonics pass to their method, and what they say while they do not run.</summary>
    public static class HybridRun
    {
        // first item is the default
        public static readonly string[] Tails = { "auto", "reference", "ray_tracing", "eyring", "sabine" };
        public static readonly string[] Diffraction = { "none", "udfa", "btms" };
        public static readonly string[] LateEnvelopes = { "auto", "omni", "directional", "off" };

        /// <summary>
        /// What the methods take for the Late envelope choice: null for auto (the method's own default by tail), the mode, or
        /// false for off. A boolean reads as directional (true) or off (false).
        /// </summary>
        public static object? LateEnvelope(string? choice) => choice?.Trim().ToLowerInvariant() switch
        {
            null or "" or "auto" => null,
            "omni" => "omni",
            "directional" or "true" => "directional",
            "off" or "false" => false,
            _ => choice,
        };

        /// <summary>
        /// Whether the method traces rays itself, so that n_rays applies: the reference of a ray-tracing tail (auto
        /// without a reference), or the directional histograms of a given reference.
        /// </summary>
        public static bool TracesRays(string? tail, string? lateEnvelope, bool referenced)
        {
            var resolved = string.IsNullOrEmpty(tail) || tail == "auto" ? (referenced ? "reference" : "ray_tracing") : tail;
            return resolved == "ray_tracing" || (resolved == "reference" && Equals(LateEnvelope(lateEnvelope), "directional"));
        }

        /// <summary>The responses of the last run, for the scene they were simulated for, and the run's description.</summary>
        public sealed record LastResult(string SceneGuid, IReadOnlyList<ImpulseResponseGoo> Responses, IReadOnlyDictionary<string, object?> Generation);

        /// <summary>
        /// The settings both methods take: unset inputs are left out so that the method's own defaults apply, the seed's
        /// included. <paramref name="references"/> are the mono responses of the reference by pair key; the reference must hold
        /// one for every pair of the scene's <paramref name="sources"/> and <paramref name="receivers"/>.
        /// </summary>
        public static Dictionary<string, object?> Settings(string? tail, int? ismOrder, string? diffraction, string? lateEnvelope, int? seed,
            bool? normalize, IReadOnlyList<ImpulseResponseGoo>? reference, int sceneSamplingRate, IReadOnlyList<SourceGoo> sources,
            IReadOnlyList<ReceiverGoo> receivers, out Dictionary<string, ImpulseResponseData>? references)
        {
            references = null;
            if (reference != null)
            {
                // the irs of Simulate Mono; the wire also orders and refreshes this component
                var keyed = new Dictionary<string, ImpulseResponseData>();
                foreach (var ir in reference.Where(ir => ir.Channels == 1))
                    keyed.TryAdd($"{ir.ReceiverGuid}_{ir.SourceGuid}", ir.Value);
                references = keyed;
                if (keyed.Count == 0)
                    throw new ArgumentException("The connected reference holds no mono responses; connect the irs output of Simulate Mono.");
                var rate = keyed.Values.First().SamplingRate;
                if (rate != sceneSamplingRate)
                    throw new ArgumentException($"The reference was simulated at {rate} Hz, the scene's sampling_rate is {sceneSamplingRate} Hz: simulate both at the same rate.");
                var missing = (from receiver in receivers
                               from source in sources
                               where !keyed.ContainsKey($"{receiver.Guid}_{source.Guid}")
                               select $"no reference for source '{source.Name}' and receiver '{receiver.Name}'").ToList();
                if (missing.Count > 0)
                    throw new ArgumentException($"The connected reference is incomplete: {string.Join(", ", missing)}; connect the irs of Simulate Mono simulated for this scene.");
            }
            var given = new Dictionary<string, object?>
            {
                ["tail"] = string.IsNullOrEmpty(tail) || tail == "auto" ? null : tail,
                ["image_source_order"] = ismOrder is int ism ? (long)ism : null,
                ["diffraction"] = string.IsNullOrEmpty(diffraction) || diffraction == "none" ? null : diffraction,
                ["late_envelope"] = LateEnvelope(lateEnvelope),
                ["seed"] = seed is int s ? (long)s : null,
                ["normalize"] = normalize,
            };
            return given.Where(kv => kv.Value != null).ToDictionary(kv => kv.Key, kv => kv.Value);
        }

        private static object? Model(object? diffraction) => diffraction switch
        {
            null or false or "none" or "" => false,
            true or "udfa" => "udfa",
            "btms" => "btms",
            _ => diffraction,
        };

        private static bool Differs(int? given, object? last) => given is int g && !(last is IConvertible c && Convert.ToInt64(c, CultureInfo.InvariantCulture) == g);

        /// <summary>
        /// What the component says while it is not running: the last result for this scene, or that there is none; and
        /// whether the settings differ from that result's.
        /// </summary>
        public static (string text, bool changed) Status(LastResult? last, string sceneGuid, string? tail, int? shOrder, int? ismOrder, string? diffraction)
        {
            if (last == null || last.SceneGuid != sceneGuid || last.Responses.Count == 0)
                return ("press Run", false);
            var runs = last.Responses.Count;
            object? Last(string key) => last.Generation.TryGetValue(key, out var value) ? value : null;
            var changed = (!string.IsNullOrEmpty(tail) && tail != "auto" && tail != Last("tail") as string)
                          || Differs(shOrder, Last("sh_order"))
                          || Differs(ismOrder, Last("image_source_order"))
                          || (diffraction != null && !Equals(Model(diffraction), Model(Last("diffraction"))));
            var summary = $"tail {MessageText.Show(Last("tail"))}" + (Last("sh_order") is { } order ? $", SH order {MessageText.Show(order)}" : "");
            return (changed ? $"settings changed, press Run ({summary})" : $"done, {runs} IR{(runs == 1 ? "" : "s")}, {summary} (Run to redo)", changed);
        }
    }
}
