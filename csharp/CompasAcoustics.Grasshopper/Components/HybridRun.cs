using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using CompasAcoustics.Pb;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>What Simulate Hybrid passes to the method, and what it says while it does not run.</summary>
    public static class HybridRun
    {
        // first item is the default
        public static readonly string[] Tails = { "auto", "reference", "ray_tracing", "eyring", "sabine" };
        public static readonly string[] Diffraction = { "none", "udfa", "btms" };
        public static readonly string[] Outputs = { "binaural", "ambisonic", "both" };

        /// <summary>The graph attribute each format is stored under.</summary>
        public static readonly IReadOnlyDictionary<string, string> Attributes = new Dictionary<string, string>
        {
            ["binaural"] = "hybrid_binaural",
            ["ambisonic"] = "hybrid",
        };

        /// <summary>The responses of the last run, for the scene they were simulated for, and the run's description.</summary>
        public sealed record LastResult(string SceneGuid, IReadOnlyDictionary<string, IrGraphGoo> Graphs, IReadOnlyDictionary<string, object?> Generation);

        /// <summary>The formats of an ``output``: binaural (default), ambisonic, or both (ambisonic, then binaural).</summary>
        public static string[] Formats(string? output) => string.IsNullOrEmpty(output) ? new[] { "binaural" } : output switch
        {
            "binaural" => new[] { "binaural" },
            "ambisonic" => new[] { "ambisonic" },
            "both" => new[] { "ambisonic", "binaural" },
            _ => throw new ArgumentException($"output must be one of {string.Join(", ", Outputs)}, not '{output}'."),
        };

        /// <summary>
        /// The method's settings: unset inputs are left out so that its own defaults apply, an empty seed draws a new one,
        /// ``hrtf`` only with a binaural output. ``references`` are the mono responses of a reference graph by pair key.
        /// </summary>
        public static Dictionary<string, object?> Settings(string? tail, int? shOrder, int? ismOrder, string? diffraction, bool? lateEnvelope, int? seed,
            bool? normalize, string? output, string? hrtf, IrGraphGoo? reference, int sceneSamplingRate, out Dictionary<string, ImpulseResponseData>? references)
        {
            var formats = Formats(output);
            references = null;
            if (reference?.Value != null)
            {
                // the mono_ir_graph of Simulate Pyroomacoustics; the wire also orders and refreshes this component
                references = new Dictionary<string, ImpulseResponseData>();
                foreach (var ir in reference.Value.Responses.Where(ir => ir.Channels == 1))
                    references.TryAdd($"{ir.ReceiverGuid}_{ir.SourceGuid}", ir.Value);
                if (references.Count == 0)
                    throw new ArgumentException("The connected ir_graph holds no mono responses; connect the mono_ir_graph output of Simulate Pyroomacoustics.");
                var rate = references.Values.First().SamplingRate;
                if (rate != sceneSamplingRate)
                    throw new ArgumentException($"The reference was simulated at {rate} Hz, the scene's sampling_rate is {sceneSamplingRate} Hz: simulate both at the same rate.");
            }
            var given = new Dictionary<string, object?>
            {
                ["tail"] = string.IsNullOrEmpty(tail) || tail == "auto" ? null : tail,
                ["sh_order"] = shOrder is int sh ? (long)sh : null,
                ["image_source_order"] = ismOrder is int ism ? (long)ism : null,
                ["diffraction"] = string.IsNullOrEmpty(diffraction) || diffraction == "none" ? null : diffraction,
                ["late_envelope"] = lateEnvelope,
                // the method's own default is a fixed seed; an empty input draws a new one each run
                ["seed"] = (long)(seed ?? Random.Shared.Next(int.MaxValue)),
                ["normalize"] = normalize,
                ["output"] = string.IsNullOrEmpty(output) ? null : formats.Length == 1 ? formats[0] : formats.Cast<object?>().ToList(),
                ["hrtf"] = formats.Contains("binaural") && !string.IsNullOrEmpty(hrtf) ? hrtf : null,
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
        public static string Status(LastResult? last, string sceneGuid, string[] formats, string? tail, int? shOrder, int? ismOrder, string? diffraction)
        {
            if (last == null || last.SceneGuid != sceneGuid)
                return "press Run";
            var byFormat = formats.ToDictionary(f => f, f => last.Graphs.TryGetValue(f, out var graph) ? graph.Value.Responses.Count : 0);
            var runs = byFormat.Values.Sum();
            if (runs == 0)
                return "press Run";
            var generation = last.Generation;
            object? Last(string key) => generation.TryGetValue(key, out var value) ? value : null;
            var changed = byFormat.Values.Any(n => n == 0)
                          || (!string.IsNullOrEmpty(tail) && tail != "auto" && tail != Last("tail") as string)
                          || Differs(shOrder, Last("sh_order"))
                          || Differs(ismOrder, Last("image_source_order"))
                          || (diffraction != null && !Equals(Model(diffraction), Model(Last("diffraction"))));
            var summary = $"tail: {Show(Last("tail"))}, order {Show(Last("sh_order"))}";
            return changed ? $"settings changed, press Run ({summary})" : $"done, {runs} IR{(runs == 1 ? "" : "s")}, {summary} (Run to redo)";
        }

        /// <summary>The HRTF set the binaural output is rendered with, for the message; empty without a binaural output.</summary>
        public static string HrtfLabel(string? hrtf, string[] formats) =>
            !formats.Contains("binaural") ? "" : $", hrtf {(string.IsNullOrEmpty(hrtf) ? "kemar" : Path.GetFileName(hrtf))}";
    }
}
