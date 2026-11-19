using System;
using Google.Protobuf;
using Grasshopper.Kernel.Types;
using GH_IO.Serialization;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// Grasshopper data holding one compas_acoustics message, as the worker sends and takes it. Saved with the
    /// document as the message's protobuf bytes.
    /// </summary>
    public abstract class MessageGoo<TMessage> : GH_Goo<TMessage> where TMessage : class, IMessage<TMessage>, new()
    {
        private const string Key = "pb";

        protected MessageGoo()
        {
        }

        protected MessageGoo(TMessage value)
        {
            Value = value;
        }

        public override bool IsValid => Value != null;

        public override string IsValidWhyNot => IsValid ? string.Empty : $"No {TypeName}.";

        /// <summary>The name the panel shows; empty when the message has none.</summary>
        public abstract string Name { get; }

        public override string ToString() => Value == null ? $"Null {TypeName}" : $"{TypeName} '{Name}'";

        protected abstract MessageGoo<TMessage> Create(TMessage? value);

        public override IGH_Goo Duplicate() => Create(Value?.Clone());

        public override bool CastFrom(object source)
        {
            switch (source)
            {
                case TMessage message:
                    Value = message;
                    return true;
                case MessageGoo<TMessage> goo:
                    Value = goo.Value;
                    return true;
                case GH_ObjectWrapper wrapper when wrapper.Value is TMessage wrapped:
                    Value = wrapped;
                    return true;
                default:
                    return false;
            }
        }

        public override bool CastTo<Q>(ref Q target)
        {
            if (Value != null && typeof(Q).IsAssignableFrom(typeof(TMessage)))
            {
                target = (Q)(object)Value;
                return true;
            }
            return false;
        }

        public override bool Write(GH_IWriter writer)
        {
            if (Value != null)
                writer.SetByteArray(Key, Value.ToByteArray());
            return true;
        }

        public override bool Read(GH_IReader reader)
        {
            if (!reader.ItemExists(Key))
            {
                Value = null!;
                return true;
            }
            var message = new TMessage();
            message.MergeFrom(reader.GetByteArray(Key));
            Value = message;
            return true;
        }
    }
}
