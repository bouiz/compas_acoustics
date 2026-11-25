using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using CompasAcoustics.Pb;
using Grasshopper.Kernel.Data;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>One response with the numbers of its receiver and source.</summary>
    public sealed record Pair(int Receiver, int Source, ImpulseResponseGoo Response);

    /// <summary>The source-receiver pairs of connected responses: numbered by first appearance, chosen by index.</summary>
    public static class PairChoice
    {
        /// <summary>The worker's responses sorted by their receiver's index in the scene, then their source's.</summary>
        public static List<ImpulseResponseGoo> InSceneOrder(SceneGoo scene, IDictionary<string, object?> responses)
        {
            var receivers = scene.Receivers.Select((r, i) => (r.Guid, i)).ToDictionary(t => t.Guid, t => t.i);
            var sources = scene.Sources.Select((s, i) => (s.Guid, i)).ToDictionary(t => t.Guid, t => t.i);
            return responses.Values.OfType<ImpulseResponseData>()
                .Select(ir => new ImpulseResponseGoo(ir))
                .OrderBy(ir => receivers.TryGetValue(ir.ReceiverGuid, out var r) ? r : int.MaxValue)
                .ThenBy(ir => sources.TryGetValue(ir.SourceGuid, out var s) ? s : int.MaxValue)
                .ToList();
        }

        /// <summary>The n-th distinct source (receiver) GUID in order of first appearance is source (receiver) n.</summary>
        public static List<Pair> Number(IEnumerable<ImpulseResponseGoo> irs)
        {
            var sources = new Dictionary<string, int>();
            var receivers = new Dictionary<string, int>();
            var seen = new HashSet<(int, int)>();
            var pairs = new List<Pair>();
            foreach (var ir in irs)
            {
                if (ir == null || !ir.IsValid)
                    continue;
                if (!sources.TryGetValue(ir.SourceGuid, out var s))
                    sources[ir.SourceGuid] = s = sources.Count;
                if (!receivers.TryGetValue(ir.ReceiverGuid, out var r))
                    receivers[ir.ReceiverGuid] = r = receivers.Count;
                if (!seen.Add((r, s)))
                    throw new ArgumentException($"Two responses for source '{ir.SourceName}' and receiver '{ir.ReceiverName}': connect the irs of one Simulate component.");
                pairs.Add(new Pair(r, s, ir));
            }
            return pairs;
        }

        /// <summary>The names of the sources and receivers by index.</summary>
        public static (List<string> Sources, List<string> Receivers) Names(IReadOnlyList<Pair> pairs) =>
            (NamesBy(pairs, p => p.Source, p => p.Response.SourceName), NamesBy(pairs, p => p.Receiver, p => p.Response.ReceiverName));

        private static List<string> NamesBy(IReadOnlyList<Pair> pairs, Func<Pair, int> index, Func<Pair, string> name)
        {
            var names = new List<string>();
            foreach (var pair in pairs.OrderBy(index))
                if (index(pair) == names.Count)
                    names.Add(name(pair));
            return names;
        }

        /// <param name="sourceIdx">Sources to keep; null or empty keeps all.</param>
        /// <param name="receiverIdx">Receivers to keep; null or empty keeps all.</param>
        public static List<Pair> Choose(IReadOnlyList<Pair> pairs, IReadOnlyList<int>? sourceIdx, IReadOnlyList<int>? receiverIdx, Action<string> warn)
        {
            var sources = pairs.Count == 0 ? 0 : pairs.Max(p => p.Source) + 1;
            var receivers = pairs.Count == 0 ? 0 : pairs.Max(p => p.Receiver) + 1;
            var keepSources = Keep("source", sourceIdx, sources, warn);
            var keepReceivers = Keep("receiver", receiverIdx, receivers, warn);
            var chosen = pairs
                .Where(p => (keepSources == null || keepSources.Contains(p.Source)) && (keepReceivers == null || keepReceivers.Contains(p.Receiver)))
                .OrderBy(p => p.Receiver).ThenBy(p => p.Source)
                .ToList();
            if (chosen.Count == 0)
                warn("No pair chosen.");
            return chosen;
        }

        private static HashSet<int>? Keep(string kind, IReadOnlyList<int>? indices, int count, Action<string> warn)
        {
            if (indices == null || indices.Count == 0)
                return null;
            var wrong = indices.Where(i => i < 0 || i >= count).ToList();
            if (wrong.Count > 0)
                warn($"No {kind} at index {string.Join(", ", wrong.Select(i => i.ToString(CultureInfo.InvariantCulture)))}; the responses have {count} {kind}(s), 0 to {count - 1}.");
            return indices.Where(i => i >= 0 && i < count).ToHashSet();
        }

        public static GH_Path PathOf(Pair p) => new(p.Receiver, p.Source);

        public static string LabelOf(Pair p) => $"{p.Receiver}_{p.Source}";
    }
}
