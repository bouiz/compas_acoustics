using System;
using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// What the simulations see of a scene: everything but the sources' sounds, which only Auralise takes, and the guids,
    /// which the worker draws anew whenever a component solves again, even with the same inputs (a recompute, a sound
    /// chosen). Such a solution keeps the simulations' results; the responses they pass on carry each source's current
    /// sound and the current guids of their source and receiver.
    /// </summary>
    public static class SimulatedScene
    {
        private static SourceData Simulated(SourceData source)
        {
            var clone = source.Clone();
            clone.Guid = "";
            clone.ClearSoundFile();
            return clone;
        }

        private static ReceiverData Simulated(ReceiverData receiver)
        {
            var clone = receiver.Clone();
            clone.Guid = "";
            return clone;
        }

        private static SceneData Simulated(SceneData scene)
        {
            var clone = scene.Clone();
            clone.Guid = "";
            for (var i = 0; i < clone.Sources.Count; i++)
                clone.Sources[i] = Simulated(clone.Sources[i]);
            for (var i = 0; i < clone.Receivers.Count; i++)
                clone.Receivers[i] = Simulated(clone.Receivers[i]);
            foreach (var obj in clone.AcousticObjects)
            {
                obj.Guid = "";
                if (obj.Material != null)
                    obj.Material.Guid = "";
                for (var i = 0; i < obj.Sources.Count; i++)
                    obj.Sources[i] = Simulated(obj.Sources[i]);
            }
            return clone;
        }

        /// <summary>Whether the simulations give <paramref name="a"/> and <paramref name="b"/> the same responses.</summary>
        public static bool Same(SceneGoo a, SceneGoo b) => Simulated(a.Value).Equals(Simulated(b.Value));

        /// <summary>
        /// The key of the worker's <c>simulate</c>: a hash of the call with the scene as the simulations see it and each
        /// reference, in the scene's receiver and source order, without the guids of its source and receiver and its source's sound.
        /// </summary>
        public static string Key(SceneGoo scene, string method, Dictionary<string, object?> settings, IReadOnlyDictionary<string, ImpulseResponseData>? references)
        {
            object? keyed = null;
            if (references != null)
            {
                var sources = scene.Value.Sources.Select(s => s.Guid).ToList();
                var receivers = scene.Value.Receivers.Select(r => r.Guid).ToList();
                keyed = references.Values
                    .OrderBy(ir => receivers.IndexOf(ir.Receiver?.Guid ?? ""))
                    .ThenBy(ir => sources.IndexOf(ir.Source?.Guid ?? ""))
                    .Select(ir =>
                    {
                        var clone = ir.Clone();
                        if (clone.Source != null)
                            clone.Source = Simulated(clone.Source);
                        if (clone.Receiver != null)
                            clone.Receiver = Simulated(clone.Receiver);
                        return (object?)clone;
                    })
                    .ToList();
            }
            return BackgroundRuns.Key("simulate", new object?[] { Simulated(scene.Value), method, settings, keyed }, new Dictionary<string, object?>());
        }

        /// <summary>
        /// <paramref name="irs"/>, simulated for <paramref name="ran"/>, with the guid and sound of the source and the guid
        /// of the receiver of <paramref name="scene"/> in the same place; unchanged when the simulations see the two scenes apart.
        /// </summary>
        public static List<ImpulseResponseGoo> Current(IReadOnlyList<ImpulseResponseGoo> irs, SceneGoo ran, SceneGoo scene)
        {
            if (!Same(ran, scene))
                return irs.ToList();
            var sources = new Dictionary<string, SourceData>();
            for (var i = 0; i < ran.Value.Sources.Count; i++)
                sources.TryAdd(ran.Value.Sources[i].Guid, scene.Value.Sources[i]);
            var receivers = new Dictionary<string, string>();
            for (var i = 0; i < ran.Value.Receivers.Count; i++)
                receivers.TryAdd(ran.Value.Receivers[i].Guid, scene.Value.Receivers[i].Guid);
            return irs.Select(ir =>
            {
                if (ir?.Value?.Source is not { } old || !sources.TryGetValue(old.Guid, out var source))
                    return ir!;
                var receiver = ir.Value.Receiver is { } r && receivers.TryGetValue(r.Guid, out var now) ? now : null;
                if (old.Guid == source.Guid && old.HasSoundFile == source.HasSoundFile && old.SoundFile == source.SoundFile
                    && (receiver == null || receiver == ir.Value.Receiver!.Guid))
                    return ir;
                var clone = ir.Value.Clone();
                clone.Source.Guid = source.Guid;
                if (source.HasSoundFile)
                    clone.Source.SoundFile = source.SoundFile;
                else
                    clone.Source.ClearSoundFile();
                if (receiver != null)
                    clone.Receiver.Guid = receiver;
                return new ImpulseResponseGoo(clone) { SimulatedFor = ir.SimulatedFor };
            }).ToList();
        }
    }
}
