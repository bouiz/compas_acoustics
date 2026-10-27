using System.Collections.Generic;
using CompasAcoustics.Pb;
using CompasPb;

namespace CompasAcoustics.Grasshopper
{
    public sealed class ImpulseResponseGoo : MessageGoo<ImpulseResponseData>
    {
        private static readonly CompasPbSerializer Serializer = new CompasPbSerializer();
        private IDictionary<string, object?>? _generation;
        private ImpulseResponseData? _generationOf;

        public ImpulseResponseGoo() { }
        public ImpulseResponseGoo(ImpulseResponseData value) : base(value) { }
        public override string TypeName => "ImpulseResponse";
        public override string TypeDescription => "A compas_acoustics ImpulseResponse, at its raw level.";
        public override string Name => Value?.Name ?? "";
        public int SamplingRate => Value?.SamplingRate ?? 0;
        public int Samples => Value == null || Value.Shape.Count == 0 ? 0 : (int)Value.Shape[0];
        public int Channels => Value == null ? 0 : Value.Shape.Count > 1 ? (int)Value.Shape[1] : 1;
        public string SourceGuid => Value?.Source?.Guid ?? "";
        public string ReceiverGuid => Value?.Receiver?.Guid ?? "";
        public string SourceName => Value?.Source?.Name ?? "";
        public string ReceiverName => Value?.Receiver?.Name ?? "";

        /// <summary>The method's description of the run, e.g. ``method``, ``output``, ``seed``.</summary>
        public IDictionary<string, object?> Generation
        {
            get
            {
                // the value can be replaced (CastFrom, Read): the cache belongs to the message it was read from
                if (_generation == null || !ReferenceEquals(_generationOf, Value))
                {
                    _generationOf = Value;
                    CompasPbRegistration.Ensure();
                    _generation = Value?.Generation != null && Serializer.UnpackAnyData(Value.Generation) is IDictionary<string, object?> dict
                        ? dict
                        : new Dictionary<string, object?>();
                }
                return _generation;
            }
        }

        public string? Method => Generation.TryGetValue("method", out var method) ? method as string : null;
        /// <summary>``mono``, ``binaural`` (two channels, no ambisonic order) or ``ambisonic``, as ``ImpulseResponse.format``.</summary>
        public string Format => Value == null ? "" : Channels == 1 ? "mono" : Channels == 2 && Value?.HasAmbisonicOrder != true ? "binaural" : "ambisonic";

        public override string ToString() => Value == null ? "Null ImpulseResponse" : $"IR {SourceName} -> {ReceiverName}, {Channels} ch, {SamplingRate} Hz";

        protected override MessageGoo<ImpulseResponseData> Create(ImpulseResponseData? value) => value == null ? new ImpulseResponseGoo() : new ImpulseResponseGoo(value);
    }

    public sealed class AudioGoo : MessageGoo<AudioData>
    {
        public AudioGoo() { }
        public AudioGoo(AudioData value) : base(value) { }
        public override string TypeName => "Audio";
        public override string TypeDescription => "A rendered signal: Auralise gives it, the Audio Player plays it.";
        public override string Name => Value?.Format ?? "";
        public int SamplingRate => Value?.SamplingRate ?? 0;
        public int Channels => Value == null ? 0 : Value.Shape.Count > 1 ? (int)Value.Shape[1] : 1;
        public string Format => Value?.Format ?? "";
        public override string ToString() => Value == null ? "Null Audio" : $"Audio, {Format}, {Channels} ch, {SamplingRate} Hz, {(Value.Shape.Count > 0 ? Value.Shape[0] : 0)} samples";
        protected override MessageGoo<AudioData> Create(AudioData? value) => value == null ? new AudioGoo() : new AudioGoo(value);
    }

    public sealed class RayFieldGoo : MessageGoo<RayFieldData>
    {
        public RayFieldGoo() { }
        public RayFieldGoo(RayFieldData value) : base(value) { }
        public override string TypeName => "RayField";
        public override string TypeDescription => "Traced acoustic rays, with their scattered rays.";
        public override string Name => Value != null && Value.HasName ? Value.Name : "";
        public override string ToString() => Value == null ? "Null RayField" : $"RayField, {Value.Rays.Count} rays";
        protected override MessageGoo<RayFieldData> Create(RayFieldData? value) => value == null ? new RayFieldGoo() : new RayFieldGoo(value);

        public override bool CastFrom(object source)
        {
            // specular reflections play in the billiard too
            if (source is ReflectionsGoo reflections && reflections.Value != null)
            {
                Value = reflections.ToRayField();
                return true;
            }
            return base.CastFrom(source);
        }
    }

    public sealed class ReflectionsGoo : MessageGoo<SpecularReflectionsData>
    {
        public ReflectionsGoo() { }
        public ReflectionsGoo(SpecularReflectionsData value) : base(value) { }
        public override string TypeName => "SpecularReflections";
        public override string TypeDescription => "Specular reflection paths from the sources over the walls to the receivers.";
        public override string Name => Value != null && Value.HasName ? Value.Name : "";
        public override string ToString() => Value == null ? "Null SpecularReflections" : $"SpecularReflections, {Value.Paths.Count} paths";
        protected override MessageGoo<SpecularReflectionsData> Create(SpecularReflectionsData? value) => value == null ? new ReflectionsGoo() : new ReflectionsGoo(value);

        /// <summary>The paths as a ray field, which the billiard plays.</summary>
        public RayFieldData ToRayField()
        {
            var field = new RayFieldData { Guid = Value.Guid };
            foreach (var path in Value.Paths)
            {
                var ray = new RayData { Guid = path.Guid, Direction = path.Direction, DistanceOffset = path.DistanceOffset };
                ray.Points.Add(path.Points);
                if (path.HasSourceIndex) ray.SourceIndex = path.SourceIndex;
                if (path.HasRayIndex) ray.RayIndex = path.RayIndex;
                field.Rays.Add(ray);
            }
            field.Sources.Add(Value.Sources);
            return field;
        }

        public override bool CastTo<Q>(ref Q target)
        {
            if (Value != null && typeof(Q).IsAssignableFrom(typeof(RayFieldGoo)))
            {
                target = (Q)(object)new RayFieldGoo(ToRayField());
                return true;
            }
            return base.CastTo(ref target);
        }
    }
}
