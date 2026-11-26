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
        public static readonly string[] Tails = { "ray_tracing", "reference", "eyring", "sabine" };
        public static readonly string[] Diffraction = { "none", "udfa", "btms" };
        public static readonly string[] LateEnvelopes = { "omni", "directional", "off" };

        /// <summary>
        /// What the methods take for the Late envelope choice: the mode, or false for off. A boolean reads as directional
        /// (true) or off (false); an unset or unknown choice, a saved Auto among them, is omni.
        /// </summary>
        public static object LateEnvelope(string? choice) => choice?.Trim().ToLowerInvariant() switch
        {
            "directional" or "true" => "directional",
            "off" or "false" => false,
            _ => "omni",
        };

        /// <summary>Whether the Late envelope rests: an Eyring or Sabine tail has no reference for it to follow, and the methods refuse a mode with one.</summary>
        public static bool LateEnvelopeRests(string tail) => tail is "eyring" or "sabine";

        /// <summary>
        /// The tail to send: the chosen one, ray tracing when none is set, and, for a definition saved with the former Auto
        /// (resolved here, where the wires are known), the reference when one is connected, else ray tracing.
        /// </summary>
        public static string Tail(string? choice, bool referenced) => string.IsNullOrEmpty(choice) ? "ray_tracing"
            : choice == "auto" ? (referenced ? "reference" : "ray_tracing")
            : choice;

        public const string ReferenceNeedsItsTail = "A reference is connected: set Tail to Reference, or disconnect it.";

        public const string TailNeedsAReference = "Tail Reference needs a reference: connect the irs of Simulate Mono to 'reference', or choose another tail.";

        /// <summary>What is wrong with <paramref name="tail"/> and the reference's wire: an error text for a reference with another tail, a warning text for Tail Reference without one; null when they agree.</summary>
        public static string? TailProblem(string tail, bool referenced) =>
            referenced && tail != "reference" ? ReferenceNeedsItsTail : !referenced && tail == "reference" ? TailNeedsAReference : null;

        /// <summary>
        /// Whether the method traces rays itself, so that n_rays applies: the reference of a ray-tracing tail, or the
        /// directional histograms of a given reference.
        /// </summary>
        public static bool TracesRays(string tail, string? lateEnvelope) =>
            tail == "ray_tracing" || (tail == "reference" && Equals(LateEnvelope(lateEnvelope), "directional"));

        /// <summary>The responses of the last run, the scene they were simulated for, and the run's description.</summary>
        public sealed record LastResult(SceneGoo Scene, IReadOnlyList<ImpulseResponseGoo> Responses, IReadOnlyDictionary<string, object?> Generation);

        /// <summary>The responses of the last run when they hold for <paramref name="scene"/>, with its sources' current sounds; else none.</summary>
        public static IReadOnlyList<ImpulseResponseGoo> Passed(LastResult? last, SceneGoo scene) =>
            last != null && SimulatedScene.Same(last.Scene, scene) ? SimulatedScene.Current(last.Responses, last.Scene, scene) : Array.Empty<ImpulseResponseGoo>();

        /// <summary>
        /// The settings both methods take: unset inputs are left out so that the method's own defaults apply, the seed's
        /// included, but the tail and the late envelope are always sent. <paramref name="references"/> are the mono responses of the reference by pair key; the reference must hold
        /// one for every pair of the scene's <paramref name="sources"/> and <paramref name="receivers"/>.
        /// </summary>
        public static Dictionary<string, object?> Settings(string? tail, int? ismOrder, string? diffraction, string? lateEnvelope, int? seed,
            IReadOnlyList<ImpulseResponseGoo>? reference, int sceneSamplingRate, IReadOnlyList<SourceGoo> sources,
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
            var sentTail = Tail(tail, reference != null);
            var given = new Dictionary<string, object?>
            {
                ["tail"] = sentTail,
                ["image_source_order"] = ismOrder is int ism ? (long)ism : null,
                ["diffraction"] = string.IsNullOrEmpty(diffraction) || diffraction == "none" ? null : diffraction,
                ["late_envelope"] = LateEnvelopeRests(sentTail) ? false : LateEnvelope(lateEnvelope),
                ["seed"] = seed is int s ? (long)s : null,
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
        public static (string text, bool changed) Status(LastResult? last, SceneGoo scene, string? tail, int? shOrder, int? ismOrder, string? diffraction)
        {
            if (last == null || !SimulatedScene.Same(last.Scene, scene) || last.Responses.Count == 0)
                return ("press Run", false);
            object? Last(string key) => last.Generation.TryGetValue(key, out var value) ? value : null;
            var changed = (!string.IsNullOrEmpty(tail) && tail != Last("tail") as string)
                          || Differs(shOrder, Last("sh_order"))
                          || Differs(ismOrder, Last("image_source_order"))
                          || (diffraction != null && !Equals(Model(diffraction), Model(Last("diffraction"))));
            var summary = MessageText.Run(last.Responses[0].SamplingRate, last.Responses.Count);
            return (changed ? $"settings changed, press Run ({summary})" : $"done, {summary} (Run to redo)", changed);
        }
    }
}
