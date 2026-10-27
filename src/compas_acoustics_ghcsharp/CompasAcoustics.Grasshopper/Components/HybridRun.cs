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

        /// <summary>The responses of the last run, for the scene they were simulated for, and the run's description.</summary>
        public sealed record LastResult(string SceneGuid, IrGraphGoo Graph, IReadOnlyDictionary<string, object?> Generation);

        /// <summary>
        /// The settings both methods take: unset inputs are left out so that the method's own defaults apply, an empty seed
        /// draws a new one. ``references`` are the mono responses of a reference graph by pair key.
        /// </summary>
        public static Dictionary<string, object?> Settings(string? tail, int? ismOrder, string? diffraction, bool? lateEnvelope, int? seed,
            bool? normalize, IrGraphGoo? reference, int sceneSamplingRate, out Dictionary<string, ImpulseResponseData>? references)
        {
            references = null;
            if (reference?.Value != null)
            {
                // the mono_ir_graph of Simulate Mono; the wire also orders and refreshes this component
                references = new Dictionary<string, ImpulseResponseData>();
                foreach (var ir in reference.Value.Responses.Where(ir => ir.Channels == 1))
                    references.TryAdd($"{ir.ReceiverGuid}_{ir.SourceGuid}", ir.Value);
                if (references.Count == 0)
                    throw new ArgumentException("The connected ir_graph holds no mono responses; connect the mono_ir_graph output of Simulate Mono.");
                var rate = references.Values.First().SamplingRate;
                if (rate != sceneSamplingRate)
                    throw new ArgumentException($"The reference was simulated at {rate} Hz, the scene's sampling_rate is {sceneSamplingRate} Hz: simulate both at the same rate.");
            }
            var given = new Dictionary<string, object?>
            {
                ["tail"] = string.IsNullOrEmpty(tail) || tail == "auto" ? null : tail,
                ["image_source_order"] = ismOrder is int ism ? (long)ism : null,
                ["diffraction"] = string.IsNullOrEmpty(diffraction) || diffraction == "none" ? null : diffraction,
                ["late_envelope"] = lateEnvelope,
                // the method's own default is a fixed seed; an empty input draws a new one each run
                ["seed"] = (long)(seed ?? Random.Shared.Next(int.MaxValue)),
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

        private static string Show(object? value) => value switch
        {
            null => "None",
            bool b => b ? "True" : "False",
            IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
            _ => value.ToString() ?? "",
        };

        private static bool Differs(int? given, object? last) => given is int g && !(last is IConvertible c && Convert.ToInt64(c, CultureInfo.InvariantCulture) == g);

        /// <summary>What the component says while it is not running: the last result for this scene, or that there is none.</summary>
        public static string Status(LastResult? last, string sceneGuid, string? tail, int? shOrder, int? ismOrder, string? diffraction)
        {
            if (last == null || last.SceneGuid != sceneGuid || last.Graph.Value.Responses.Count == 0)
                return "press Run";
            var runs = last.Graph.Value.Responses.Count;
            object? Last(string key) => last.Generation.TryGetValue(key, out var value) ? value : null;
            var changed = (!string.IsNullOrEmpty(tail) && tail != "auto" && tail != Last("tail") as string)
                          || Differs(shOrder, Last("sh_order"))
                          || Differs(ismOrder, Last("image_source_order"))
                          || (diffraction != null && !Equals(Model(diffraction), Model(Last("diffraction"))));
            var summary = $"tail: {Show(Last("tail"))}" + (Last("sh_order") is { } order ? $", order {Show(order)}" : "");
            return changed ? $"settings changed, press Run ({summary})" : $"done, {runs} IR{(runs == 1 ? "" : "s")}, {summary} (Run to redo)";
        }
    }
}
