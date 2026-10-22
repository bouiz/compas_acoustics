using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;
using CompasPb.Data;
using Grasshopper.Kernel;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    internal static class Points
    {
        public static Point3d Of(PointData? point) => point == null ? Point3d.Unset : new Point3d(point.X, point.Y, point.Z);

        public static Vector3d Of(VectorData? vector) => vector == null ? Vector3d.Unset : new Vector3d(vector.X, vector.Y, vector.Z);
    }

    public sealed class MaterialGoo : MessageGoo<AcousticMaterialData>
    {
        public MaterialGoo() { }
        public MaterialGoo(AcousticMaterialData value) : base(value) { }
        public override string TypeName => "Material";
        public override string TypeDescription => "A compas_acoustics AcousticMaterial.";
        public override string Name => Value?.Name ?? "";

        public override string ToString()
        {
            if (Value == null)
                return base.ToString();
            var parts = new List<string> { Value.Name };
            if (!string.IsNullOrEmpty(Value.Category))
                parts.Add(Value.Category);
            if (Value.AbsorptionCoefficients.Count > 0)
                parts.Add(string.Format(System.Globalization.CultureInfo.InvariantCulture, "mean absorption {0:0.00}", Value.AbsorptionCoefficients.Average()));
            return string.Join(", ", parts);
        }

        protected override MessageGoo<AcousticMaterialData> Create(AcousticMaterialData? value) => value == null ? new MaterialGoo() : new MaterialGoo(value);
    }

    public sealed class SourceGoo : MessageGoo<SourceData>, IOwnPreview
    {
        private (SourceData of, SourcePreview preview)? _preview;

        public SourcePreview? Preview => Value == null ? null : (_preview is { } p && ReferenceEquals(p.of, Value) ? p : (_preview = (Value, PreviewGeometry.Source(Value))).Value).preview;

        public BoundingBox ClippingBox => Preview == null ? BoundingBox.Empty : PreviewDrawing.Box(Preview.Centre, PreviewGeometry.SourceArrow);

        public void DrawWires(Rhino.Display.DisplayPipeline pipeline, System.Drawing.Color color, int thickness)
        {
            if (Preview != null)
                PreviewDrawing.Source(pipeline, Preview, color, thickness);
        }

        public SourceGoo() { }
        public SourceGoo(SourceData value) : base(value) { }
        public override string TypeName => "Source";
        public override string TypeDescription => "A compas_acoustics Source.";
        public override string Name => Value?.Name ?? "";
        public string Guid => Value?.Guid ?? "";
        public Point3d Position => Points.Of(Value?.Point);
        public Vector3d Direction => Points.Of(Value?.Direction);
        public string? SoundFile => Value != null && Value.HasSoundFile ? Value.SoundFile : null;
        protected override MessageGoo<SourceData> Create(SourceData? value) => value == null ? new SourceGoo() : new SourceGoo(value);
    }

    public sealed class ReceiverGoo : MessageGoo<ReceiverData>, IOwnPreview
    {
        private (ReceiverData of, ReceiverPreview preview)? _preview;

        public ReceiverPreview? Preview => Value == null ? null : (_preview is { } p && ReferenceEquals(p.of, Value) ? p : (_preview = (Value, PreviewGeometry.Receiver(Value))).Value).preview;

        public BoundingBox ClippingBox => Preview == null ? BoundingBox.Empty : PreviewDrawing.Box(Preview.Centre, PreviewGeometry.FacingArrow);

        public void DrawWires(Rhino.Display.DisplayPipeline pipeline, System.Drawing.Color color, int thickness)
        {
            if (Preview != null)
                PreviewDrawing.Receiver(pipeline, Preview, color, thickness);
        }

        public ReceiverGoo() { }
        public ReceiverGoo(ReceiverData value) : base(value) { }
        public override string TypeName => "Receiver";
        public override string TypeDescription => "A compas_acoustics Receiver.";
        public override string Name => Value?.Name ?? "";
        public string Guid => Value?.Guid ?? "";
        public Point3d Position => Points.Of(Value?.Point);
        protected override MessageGoo<ReceiverData> Create(ReceiverData? value) => value == null ? new ReceiverGoo() : new ReceiverGoo(value);
    }

    /// <summary>
    /// Model data drawn only from its own parameter, under the preview toggle of the component that makes it: not
    /// IGH_PreviewData, which Grasshopper's generic parameters (Merge, Data) would draw too, past that toggle. Surfaces only
    /// as edges: a mesh would cover the Rhino surface it comes from.
    /// </summary>
    public interface IOwnPreview
    {
        BoundingBox ClippingBox { get; }

        void DrawWires(Rhino.Display.DisplayPipeline pipeline, System.Drawing.Color color, int thickness);
    }

    public sealed class AcousticObjectGoo : MessageGoo<AcousticObjectData>, IOwnPreview
    {
        private (AcousticObjectData of, ObjectPreview preview)? _preview;

        /// <summary>The object's surface, built once for each message.</summary>
        public ObjectPreview? Preview => Value == null ? null : (_preview is { } p && ReferenceEquals(p.of, Value) ? p : (_preview = (Value, PreviewGeometry.Object(Value))).Value).preview;

        public BoundingBox ClippingBox
        {
            get
            {
                var box = BoundingBox.Empty;
                if (Preview is { } preview)
                    foreach (var v in preview.Polyline ?? (IEnumerable<Point3d>)preview.Vertices)
                        box.Union(v);
                return box;
            }
        }

        public void DrawWires(Rhino.Display.DisplayPipeline pipeline, System.Drawing.Color color, int thickness)
        {
            if (Preview is { } preview)
                PreviewDrawing.ObjectWires(pipeline, new[] { preview }, color, thickness);
        }


        public AcousticObjectGoo() { }
        public AcousticObjectGoo(AcousticObjectData value) : base(value) { }
        public override string TypeName => "AcousticObject";
        public override string TypeDescription => "A compas_acoustics AcousticObject: a surface of the room or an obstacle, with its material.";
        public override string Name => Value != null && Value.HasName ? Value.Name : Value?.Role ?? "";
        protected override MessageGoo<AcousticObjectData> Create(AcousticObjectData? value) => value == null ? new AcousticObjectGoo() : new AcousticObjectGoo(value);
    }

    public sealed class SceneGoo : MessageGoo<SceneData>, IOwnPreview
    {
        private (SceneData of, ScenePreview preview)? _preview;

        /// <summary>The preview geometry, built once for each message.</summary>
        public ScenePreview? Preview => Value == null ? null : (_preview is { } p && ReferenceEquals(p.of, Value) ? p : (_preview = (Value, PreviewGeometry.Scene(Value))).Value).preview;

        public BoundingBox ClippingBox => Preview?.Box ?? BoundingBox.Empty;

        public void DrawWires(Rhino.Display.DisplayPipeline pipeline, System.Drawing.Color color, int thickness)
        {
            if (Preview != null)
                PreviewDrawing.SceneWires(pipeline, Preview, color, thickness);
        }


        public SceneGoo() { }
        public SceneGoo(SceneData value) : base(value) { }
        public override string TypeName => "Scene";
        public override string TypeDescription => "A compas_acoustics Scene: acoustic objects, sources and receivers.";
        public override string Name => Value?.Name ?? "";
        public int SamplingRate => Value?.SamplingRate ?? 0;
        public IReadOnlyList<SourceGoo> Sources => Value?.Sources.Select(s => new SourceGoo(s)).ToList() ?? new List<SourceGoo>();
        public IReadOnlyList<ReceiverGoo> Receivers => Value?.Receivers.Select(r => new ReceiverGoo(r)).ToList() ?? new List<ReceiverGoo>();
        protected override MessageGoo<SceneData> Create(SceneData? value) => value == null ? new SceneGoo() : new SceneGoo(value);
    }
}
