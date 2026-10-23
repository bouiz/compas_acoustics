using System;
using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;
using CompasPb.Data;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    public sealed class SourceComponent : ComponentBase
    {
        public SourceComponent() : base("CA_Source") { }
        protected override string SpecKey => "CA_Source";
        public override Guid ComponentGuid => new Guid("c2b3794e-54b9-5ba2-8d99-2e10bef8be08");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => input ? null : new Param_Source();

        /// <summary>The source's directivity patterns, the first the default.</summary>
        internal static readonly string[] Directivities = { "omnidirectional", "cardioid", "supercardioid", "hypercardioid", "bidirectional" };

        public override IReadOnlyList<Control> Controls { get; } = new Control[] { new DropdownControl("directivity", Directivities) };

        protected override void Solve(IGH_DataAccess access)
        {
            var position = Value<Point3d>(access, "position") ?? Point3d.Origin;
            var direction = Value<Vector3d>(access, "direction") ?? Vector3d.XAxis;
            var spl = List<double>(access, "spl");
            var source = (SourceData)Worker.Call("make_source", Text(access, "name"), MeshConversion.Xyz(position), MeshConversion.Xyz(direction),
                Text(access, "directivity"), Text(access, "sound_file"), spl.Count == 0 ? null : spl.Cast<object?>().ToList())!;
            access.SetData(0, new SourceGoo(source));
        }
    }

    public sealed class ReceiverComponent : ComponentBase
    {
        public ReceiverComponent() : base("CA_Receiver") { }
        protected override string SpecKey => "CA_Receiver";
        public override Guid ComponentGuid => new Guid("3a875c99-7e37-5257-a83d-1265489c24df");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => input ? null : new Param_Receiver();

        protected override void Solve(IGH_DataAccess access)
        {
            var position = Value<Point3d>(access, "position");
            var direction = Value<Vector3d>(access, "direction");
            // upright: the receiver's up stays +z, it faces 'direction'
            var kwargs = new Dictionary<string, object?>
            {
                ["position"] = position is Point3d p ? MeshConversion.Xyz(p) : null,
                ["direction"] = direction is Vector3d d ? MeshConversion.Xyz(d) : null,
            };
            var result = (IDictionary<string, object?>)Worker.Call("make_receiver", new object?[] { Text(access, "name") }, kwargs)!;
            access.SetData(0, new ReceiverGoo((ReceiverData)result["receiver"]!));
        }
    }

    public sealed class AcousticObjectComponent : ComponentBase
    {
        internal static readonly string[] Roles = { "boundary", "obstacle" };

        public AcousticObjectComponent() : base("CA_Acoustic_Object") { }
        protected override string SpecKey => "CA_Acoustic_Object";
        public override Guid ComponentGuid => new Guid("125fc178-5b46-56d2-9ba7-6f2db3a78e46");

        public override bool PreviewOffByDefault => true;

        protected override IGH_Param? CreateParam(PortSpec port, bool input) =>
            input && port.Name == "material" ? new Param_Material() : !input ? new Param_AcousticObject() : null;

        public override IReadOnlyList<Control> Controls { get; } = new Control[] { new SwitchControl("role", Roles) };

        /// <summary>The geometry message of a Rhino mesh, polyline or polyline curve; null for anything else.</summary>
        internal static object? Geometry(object? geometry) => geometry switch
        {
            Mesh mesh => MeshConversion.ToMeshData(mesh),
            Polyline polyline => MeshConversion.ToPolylineData(polyline),
            Curve curve when curve.TryGetPolyline(out var polyline) => MeshConversion.ToPolylineData(polyline),
            _ => null,
        };

        protected override void Solve(IGH_DataAccess access)
        {
            var goo = Item<IGH_Goo>(access, "geometry");
            if (goo == null)
            {
                Warn("No geometry provided.");
                return;
            }
            var material = Item<MaterialGoo>(access, "material");
            if (material?.Value == null)
            {
                Warn("No material provided.");
                return;
            }
            var geometry = Geometry(goo.ScriptVariable());
            if (geometry == null)
            {
                Error("Unsupported geometry type.");
                return;
            }
            var obj = (AcousticObjectData)Worker.Call("make_acoustic_object", geometry, material.Value, Text(access, "name"),
                Text(access, "role") ?? Roles[0])!;
            var result = new AcousticObjectGoo(obj);
            if (result.Preview is { Faces.Length: > 0 } preview)
                Status(MessageText.Object(preview.Area));
            access.SetData(0, result);
        }
    }

    public sealed class RoomSceneComponent : ComponentBase
    {
        public RoomSceneComponent() : base("CA_Model") { }
        protected override string SpecKey => "CA_Model";
        public override Guid ComponentGuid => new Guid("63ac4b6f-5c00-50a5-973f-6337bd1f2b11");

        public override bool PreviewOffByDefault => true;

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => (input, port.Name) switch
        {
            (true, "objects") => new Param_AcousticObject(),
            (true, "sources") => new Param_Source(),
            (true, "receivers") => new Param_Receiver(),
            (false, _) => new Param_Scene(),
            _ => null,
        };

        protected override void Solve(IGH_DataAccess access)
        {
            var given = List<AcousticObjectGoo>(access, "objects");
            var sources = List<SourceGoo>(access, "sources").Where(s => s?.Value != null).Select(s => (object?)s.Value).ToList();
            var receivers = List<ReceiverGoo>(access, "receivers").Where(r => r?.Value != null).Select(r => (object?)r.Value).ToList();
            if (given.Count == 0)
            {
                Warn("No objects provided.");
                return;
            }
            if (sources.Count == 0)
            {
                Warn("No sources provided.");
                return;
            }
            if (receivers.Count == 0)
            {
                Warn("No receivers provided.");
                return;
            }
            // an upstream component that failed passes nothing; leave such items out instead of failing the scene
            var objects = given.Where(o => o?.Value != null).Select(o => (object?)o.Value).ToList();
            if (objects.Count == 0)
            {
                Warn($"No acoustic objects: the {given.Count} object(s) given are empty or invalid.");
                return;
            }
            if (objects.Count < given.Count)
                Warn($"{given.Count - objects.Count} of the {given.Count} objects are not acoustic objects (empty or invalid) and are left out.");
            var rate = Value<int>(access, "sampling_rate");
            var scene = (SceneData)Worker.Call("make_scene", objects, sources, receivers, Text(access, "name"), rate is int r ? (long)r : null)!;
            var goo = new SceneGoo(scene);
            var room = goo.Preview;
            var (shown, full) = MessageText.Scene(room?.Volume ?? 0, room?.Area ?? 0,
                scene.AcousticObjects.Count, scene.Sources.Count, scene.Receivers.Count, scene.SamplingRate);
            Status(shown, full);
            access.SetData(0, goo);
        }
    }
}
