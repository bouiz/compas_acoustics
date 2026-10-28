using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;
using GH_IO.Serialization;
using Google.Protobuf;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// The responses of one simulation, with the scene's sources and receivers: what Simulate Mono, Simulate Binaural and
    /// Simulate Ambisonics output and Deconstruct IR Graph takes. ``Method`` names the graph attribute the responses belong to:
    /// ``mono``, ``binaural`` or ``ambisonics``.
    /// </summary>
    public sealed class IrGraphGoo : GH_Goo<IrGraphGoo.Content>
    {
        public sealed record Content(string Method, IReadOnlyList<SourceGoo> Sources, IReadOnlyList<ReceiverGoo> Receivers,
            IReadOnlyList<ImpulseResponseGoo> Responses);

        public IrGraphGoo() { }

        public IrGraphGoo(Content value) { Value = value; }

        /// <summary>The graph of ``attribute`` from the ``responses`` the worker's simulate returned for it, keyed ``receiverGuid_sourceGuid``.</summary>
        public static IrGraphGoo FromSimulation(SceneGoo scene, string attribute, IDictionary<string, object?> responses) =>
            new IrGraphGoo(new Content(attribute, scene.Sources, scene.Receivers,
                responses.Values.OfType<ImpulseResponseData>().Select(ir => new ImpulseResponseGoo(ir)).ToList()));

        public override bool IsValid => Value != null;
        public override string TypeName => "IRGraph";
        public override string TypeDescription => "The impulse responses of a simulation output, per source and receiver.";

        public override string ToString() =>
            Value == null ? "Null IRGraph" : $"{Value.Method}: {Value.Responses.Count} response{(Value.Responses.Count == 1 ? "" : "s")}";

        public override IGH_Goo Duplicate() => new IrGraphGoo(Value);

        public override bool Write(GH_IWriter writer)
        {
            if (Value == null)
                return true;
            writer.SetString("method", Value.Method);
            Save(writer, "source", Value.Sources.Select(s => (IMessage)s.Value).ToList());
            Save(writer, "receiver", Value.Receivers.Select(r => (IMessage)r.Value).ToList());
            Save(writer, "response", Value.Responses.Select(r => (IMessage)r.Value).ToList());
            return true;
        }

        public override bool Read(GH_IReader reader)
        {
            if (!reader.ItemExists("method"))
                return true;
            Value = new Content(reader.GetString("method"),
                Load<SourceData>(reader, "source").Select(m => new SourceGoo(m)).ToList(),
                Load<ReceiverData>(reader, "receiver").Select(m => new ReceiverGoo(m)).ToList(),
                Load<ImpulseResponseData>(reader, "response").Select(m => new ImpulseResponseGoo(m)).ToList());
            return true;
        }

        private static void Save(GH_IWriter writer, string name, IReadOnlyList<IMessage> messages)
        {
            writer.SetInt32(name + "_count", messages.Count);
            for (var i = 0; i < messages.Count; i++)
                writer.SetByteArray(name, i, messages[i].ToByteArray());
        }

        private static IEnumerable<T> Load<T>(GH_IReader reader, string name) where T : IMessage<T>, new()
        {
            var count = reader.ItemExists(name + "_count") ? reader.GetInt32(name + "_count") : 0;
            for (var i = 0; i < count; i++)
            {
                var message = new T();
                message.MergeFrom(reader.GetByteArray(name, i));
                yield return message;
            }
        }
    }
}
