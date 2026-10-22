using System;
using System.Collections.Generic;
using System.Linq;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>One receiver's branch of Deconstruct IR Graph: the receiver, its sources and their responses.</summary>
    public sealed record IrBranch(int ReceiverIndex, ReceiverGoo Receiver, IReadOnlyList<SourceGoo> Sources, IReadOnlyList<ImpulseResponseGoo> Responses);

    /// <summary>The responses of an IR graph per receiver, filtered by source and receiver index, as Deconstruct IR Graph gives them.</summary>
    public static class IrGraphGrouping
    {
        private static string Message(string kind, int index, int count) =>
            $"No {kind} at index {index}; the graph has {count} {kind}(s), 0 to {count - 1}.";

        /// <param name="sourceIdx">Sources to keep, by index over the whole graph; null or empty keeps all.</param>
        /// <param name="receiverIdx">Receivers to give a branch, in this order; null or empty gives every receiver its branch.</param>
        public static List<IrBranch> Group(IrGraphGoo.Content graph, IReadOnlyList<int>? sourceIdx, IReadOnlyList<int>? receiverIdx, Action<string> warn)
        {
            HashSet<string>? keep = null;
            if (sourceIdx != null && sourceIdx.Count > 0)
            {
                var wrong = sourceIdx.Where(i => i < 0 || i >= graph.Sources.Count).ToList();
                if (wrong.Count > 0)
                    warn($"No source at index {string.Join(", ", wrong)}; the graph has {graph.Sources.Count} source(s), 0 to {graph.Sources.Count - 1}.");
                keep = sourceIdx.Where(i => i >= 0 && i < graph.Sources.Count).Select(i => graph.Sources[i].Guid).ToHashSet();
            }
            var indices = receiverIdx != null && receiverIdx.Count > 0 ? receiverIdx : Enumerable.Range(0, graph.Receivers.Count).ToList();
            var branches = new List<IrBranch>();
            foreach (var index in indices)
            {
                if (index < 0 || index >= graph.Receivers.Count)
                {
                    warn(Message("receiver", index, graph.Receivers.Count));
                    continue;
                }
                var receiver = graph.Receivers[index];
                var responses = graph.Responses
                    .Where(ir => ir.ReceiverGuid == receiver.Guid && (keep == null || keep.Contains(ir.SourceGuid)))
                    .ToList();
                var sources = responses
                    .Select(ir => graph.Sources.FirstOrDefault(s => s.Guid == ir.SourceGuid) ?? new SourceGoo(ir.Value.Source))
                    .ToList();
                branches.Add(new IrBranch(index, receiver, sources, responses));
            }
            return branches;
        }
    }
}
