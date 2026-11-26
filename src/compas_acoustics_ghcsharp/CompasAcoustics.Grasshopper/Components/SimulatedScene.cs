using System;
using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// What the simulations see of a scene: everything but the sources' sounds, which only Auralise takes, and the guids of
    /// the scene and its sources, drawn anew in every solution of Model and Source. Choosing a source's sound makes both
    /// anew, yet the simulations keep their results; the responses they pass on carry each source's current sound.
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

        private static SceneData Simulated(SceneData scene)
        {
            var clone = scene.Clone();
            clone.Guid = "";
            for (var i = 0; i < clone.Sources.Count; i++)
                clone.Sources[i] = Simulated(clone.Sources[i]);
            return clone;
        }

        /// <summary>Whether the simulations give <paramref name="a"/> and <paramref name="b"/> the same responses.</summary>
        public static bool Same(SceneGoo a, SceneGoo b) => Simulated(a.Value).Equals(Simulated(b.Value));

        /// <summary>
        /// The key of the worker's <c>simulate</c>: a hash of the call with the scene as the simulations see it and each
        /// reference, in the scene's source order, without its source's guid and sound.
        /// </summary>
        public static string Key(SceneGoo scene, string method, Dictionary<string, object?> settings, IReadOnlyDictionary<string, ImpulseResponseData>? references)
        {
            object? keyed = null;
            if (references != null)
            {
                var order = scene.Value.Sources.Select(s => s.Guid).ToList();
                keyed = references.Values
                    .OrderBy(ir => ir.Receiver?.Guid ?? "", StringComparer.Ordinal)
                    .ThenBy(ir => order.IndexOf(ir.Source?.Guid ?? ""))
                    .Select(ir =>
                    {
                        var clone = ir.Clone();
                        if (clone.Source != null)
                            clone.Source = Simulated(clone.Source);
                        return (object?)clone;
                    })
                    .ToList();
            }
            return BackgroundRuns.Key("simulate", new object?[] { Simulated(scene.Value), method, settings, keyed }, new Dictionary<string, object?>());
        }

        /// <summary>
        /// <paramref name="irs"/>, simulated for <paramref name="ran"/>, with the guid and sound of the source of
        /// <paramref name="scene"/> in the same place; unchanged when the simulations see the two scenes apart.
        /// </summary>
        public static List<ImpulseResponseGoo> Current(IReadOnlyList<ImpulseResponseGoo> irs, SceneGoo ran, SceneGoo scene)
        {
            if (!Same(ran, scene))
                return irs.ToList();
            var now = new Dictionary<string, SourceData>();
            for (var i = 0; i < ran.Value.Sources.Count; i++)
                now.TryAdd(ran.Value.Sources[i].Guid, scene.Value.Sources[i]);
            return irs.Select(ir =>
            {
                if (ir?.Value?.Source is not { } old || !now.TryGetValue(old.Guid, out var source)
                    || (old.Guid == source.Guid && old.HasSoundFile == source.HasSoundFile && old.SoundFile == source.SoundFile))
                    return ir!;
                var clone = ir.Value.Clone();
                clone.Source.Guid = source.Guid;
                if (source.HasSoundFile)
                    clone.Source.SoundFile = source.SoundFile;
                else
                    clone.Source.ClearSoundFile();
                return new ImpulseResponseGoo(clone);
            }).ToList();
        }
    }
}
