using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using CompasAcoustics.Pb;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>A component that draws its own geometry in the colour of each reflection order, besides the previews of its outputs.</summary>
    public abstract class DisplayComponent : ComponentBase
    {
        protected BoundingBox Box = BoundingBox.Empty;

        protected DisplayComponent(string key) : base(key) { }

        /// <summary>The drawing on and off on the body: the component's preview, as its right-click menu sets it.</summary>
        protected static readonly ToggleControl PreviewSwitch = new("preview", "Preview");

        /// <summary>Not the outputs: the component draws its rays, paths or particles itself (Rays' curves would show every ray before Play).</summary>
        public override bool PreviewsOutput(IGH_Param output) => false;

        /// <summary>What the Play button reads: Stop while the playback moves.</summary>
        public static string PlayLabel(Playback playback) => playback.Running ? "Stop" : "Play";

        public override string Choice(string name) => name == PreviewSwitch.Port ? (Hidden ? "off" : "on") : base.Choice(name);

        public override void SetChoice(string name, string value)
        {
            if (name != PreviewSwitch.Port)
            {
                base.SetChoice(name, value);
                return;
            }
            var hide = value == "off";
            if (hide == Hidden)
                return;
            var document = OnPingDocument();
            if (document != null)
                RecordUndoEvent($"Preview {value}");
            Hidden = hide;
            if (document != null)
            {
                ExpirePreview(true);
                global::Grasshopper.Instances.RedrawCanvas();
            }
        }

        protected static BoundingBox BoxOf(IEnumerable<Point3d> points)
        {
            var box = BoundingBox.Empty;
            foreach (var point in points)
                box.Union(point);
            return box;
        }

        public override BoundingBox ClippingBox
        {
            get
            {
                var box = base.ClippingBox;
                box.Union(Box);
                return box;
            }
        }

        protected override void BeforeSolveInstance()
        {
            Box = BoundingBox.Empty;
            Clear();
            base.BeforeSolveInstance();
        }

        protected abstract void Clear();
    }

    public sealed class VisualiseReflectionsComponent : DisplayComponent
    {
        private const int PathThickness = 2;
        private readonly List<(Polyline path, Color color)> _drawn = new();

        public VisualiseReflectionsComponent() : base("CA_Visualise_Reflections") { }
        protected override string SpecKey => "CA_Visualise_Reflections";
        public override Guid ComponentGuid => new Guid("11077914-3d8f-5a76-a0f9-07250add684e");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => (input, port.Name) switch
        {
            (true, "scene") => new Param_Scene(),
            (false, "reflections") => new Param_Reflections(),
            _ => null,
        };

        protected override void Clear() => _drawn.Clear();

        protected override bool RunsInBackground => true;

        public override IReadOnlyList<Control> Controls { get; } = new Control[] { PreviewSwitch };

        protected override void Solve(IGH_DataAccess access)
        {
            var scene = Item<SceneGoo>(access, "scene");
            if (scene?.Value == null)
            {
                Warn("Input parameter 'scene' failed to collect data.");
                return;
            }
            SceneIndices.Apply(this, "source_idx", scene.Value.Sources.Select(x => x.Name));
            SceneIndices.Apply(this, "receiver_idx", scene.Value.Receivers.Select(x => x.Name));
            var orders = List<int>(access, "order").Select(o => (object?)(long)o).ToList();
            var source = Value<int>(access, "source_idx");
            var receiver = Value<int>(access, "receiver_idx");
            var kwargs = new Dictionary<string, object?> { ["source_index"] = source is int s ? (long)s : null, ["receiver_index"] = receiver is int r ? (long)r : null };
            RunInBackground(access, new Background("specular_reflections", new object?[] { scene.Value, orders }, kwargs), (outcome, _) =>
            {
                var result = (IDictionary<string, object?>)outcome.Result!;
                // orders left out, indices out of range, sources and receivers outside the room
                ShowWarnings(result);
                var found = (SpecularReflectionsData)result["reflections"]!;
                var paths = RayGeometry.FromReflections(found);
                // drawn here, not output as geometry: Grasshopper would preview polylines in one colour on top
                _drawn.AddRange(paths.Select(p => (RayGeometry.ToPolyline(p.Ray), OrderColors.Of(p.Order))));
                Box = BoxOf(paths.SelectMany(p => p.Ray.Points));
                var counts = ((IEnumerable<object?>)result["counts"]!).Cast<IList<object?>>().Select(c => $"order {c[0]}: {c[1]}").ToList();
                Message = counts.Count > 0 ? string.Join(", ", counts) : "no paths";
                access.SetData("reflections", new ReflectionsGoo(found));
                access.SetDataList("order", paths.Select(p => p.Order));
                access.SetDataList("time", paths.Select(p => p.Ray.TravelTimeMs(Worker.SpeedOfSound)));
            });
        }

        public override void DrawViewportWires(IGH_PreviewArgs args)
        {
            base.DrawViewportWires(args);
            foreach (var (path, color) in _drawn)
                args.Display.DrawPolyline(path, color, PathThickness);
        }
    }

    public sealed class VisualiseRaysComponent : DisplayComponent
    {
        private const int LineThickness = 1;
        private const int FrameMs = 33;
        private readonly Dictionary<int, List<Line>> _drawn = new();
        private readonly Playback _playback = new(() => System.Diagnostics.Stopwatch.GetTimestamp() / (double)System.Diagnostics.Stopwatch.Frequency);

        public override IReadOnlyList<Control> Controls { get; } = new Control[]
        {
            PreviewSwitch, new ButtonControl("play", "Play"), new ButtonControl("rewind", "Rewind"),
        };

        public VisualiseRaysComponent() : base("CA_Visualise_Rays") { }

        public override string? ButtonLabel(string port) => port == "play" ? PlayLabel(_playback) : null;

        /// <summary>How far the rays are drawn: nothing until Play, then as far as sound has travelled in ``timeMs``.</summary>
        public static double DrawnDistance(bool active, double timeMs, double speedOfSound) => active ? timeMs / 1000.0 * speedOfSound : 0.0;
        protected override string SpecKey => "CA_Visualise_Rays";
        public override Guid ComponentGuid => new Guid("e693b3da-9ac2-56bb-a810-2910e8e4af8e");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => (input, port.Name) switch
        {
            (true, "scene") => new Param_Scene(),
            (false, "ray_field") => new Param_RayField(),
            (false, _) => new Param_Curve(),
            _ => null,
        };

        protected override void Clear() => _drawn.Clear();

        protected override bool RunsInBackground => true;

        protected override void Solve(IGH_DataAccess access)
        {
            var scene = Item<SceneGoo>(access, "scene");
            if (scene?.Value == null)
            {
                Warn("Input parameter 'scene' failed to collect data.");
                return;
            }
            SceneIndices.Apply(this, "source_idx", scene.Value.Sources.Select(x => x.Name));
            var maxOrder = Value<int>(access, "max_order") ?? 10;
            var order = Value<int>(access, "order");
            var direction = Value<Vector3d>(access, "direction");
            var kwargs = new Dictionary<string, object?>
            {
                ["source_index"] = Value<int>(access, "source_idx") is int s ? (long)s : null,
                ["n_rays"] = (long)(Value<int>(access, "n_rays") ?? 500),
                ["max_order"] = (long)maxOrder,
                ["direction"] = direction is Vector3d d ? MeshConversion.Xyz(d) : null,
                ["angle"] = Value<double>(access, "angle"),
                ["scattering"] = Value<bool>(access, "scattering") ?? false,
                ["seed"] = Value<int>(access, "seed") is int seed ? (long)seed : null,
                ["order"] = order is int o ? (long)o : null,
            };
            RunInBackground(access, new Background("ray_field", new object?[] { scene.Value }, kwargs, order ?? maxOrder), (outcome, _) =>
            {
                var result = (IDictionary<string, object?>)outcome.Result!;
                // sources outside the room
                ShowWarnings(result);
                var field = (RayFieldData)result["field"]!;
                var shown = (RayFieldData)result["shown"]!;
                Message = $"{field.Rays.Count} rays, order={outcome.Context}, seed={result["seed"]}";
                var rays = RayGeometry.FromField(shown);
                // nothing until Play: all rays at once are a blur; Play draws them as far as sound has travelled, Stop holds them
                var speedOfSound = Worker.SpeedOfSound;
                // one button: Play, and Stop while playing (it holds the rays where they are)
                var play = Pressed("play");
                if (play && _playback.Running)
                {
                    _playback.Stop();
                    play = false;
                }
                if (Pressed("rewind"))
                    _playback.Rewind();
                var frame = _playback.Update(null, play, null, rays.Select(r => r.TravelTimeMs(speedOfSound)).DefaultIfEmpty(0).Max());
                if (frame.Animating)
                    OnPingDocument()?.ScheduleSolution(FrameMs, document => ExpireSolution(false));
                var distance = DrawnDistance(_playback.Active, frame.T, speedOfSound);
                Message = _playback.Active ? frame.Message : $"{field.Rays.Count} rays, press Play";
                foreach (var ray in rays)
                    foreach (var (k, line) in RayGeometry.Segments(RayGeometry.UpTo(ray, distance)))
                    {
                        if (!_drawn.TryGetValue(k, out var lines))
                            _drawn[k] = lines = new List<Line>();
                        lines.Add(line);
                    }
                Box = BoxOf(rays.SelectMany(r => r.Points));
                access.SetData("ray_field", new RayFieldGoo(field));
                access.SetDataList("rays", rays.Where(r => !r.Scattered).Select(r => new PolylineCurve(r.Points)));
                access.SetDataList("scattered", rays.Where(r => r.Scattered).Select(r => new PolylineCurve(r.Points)));
            });
        }

        public override void DrawViewportWires(IGH_PreviewArgs args)
        {
            base.DrawViewportWires(args);
            foreach (var (k, lines) in _drawn)
                args.Display.DrawLines(lines, OrderColors.Of(k), LineThickness);
        }
    }
}

namespace CompasAcoustics.Grasshopper
{
    public sealed class VisualiseBilliardComponent : DisplayComponent
    {
        private const int FrameMs = 33;
        private const float ParticleRadius = 4;
        private readonly Dictionary<int, List<Point3d>> _drawn = new();
        private readonly Playback _playback = new(() => System.Diagnostics.Stopwatch.GetTimestamp() / (double)System.Diagnostics.Stopwatch.Frequency);

        public VisualiseBilliardComponent() : base("CA_Visualise_Billiard") { }
        protected override string SpecKey => "CA_Visualise_Billiard";
        public override Guid ComponentGuid => new Guid("1dd7e727-b816-578f-8eaa-253a9a153829");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => input && port.Name == "ray_field" ? new Param_RayField() : null;

        public override IReadOnlyList<Control> Controls { get; } = new Control[]
        {
            PreviewSwitch, new ButtonControl("play", "Play"), new ButtonControl("rewind", "Rewind"),
        };

        public override string? ButtonLabel(string port) => port == "play" ? PlayLabel(_playback) : null;

        protected override void Clear() => _drawn.Clear();

        protected override void Solve(IGH_DataAccess access)
        {
            var field = Item<RayFieldGoo>(access, "ray_field");
            if (field?.Value == null || field.Value.Rays.Count == 0)
            {
                Message = "";
                Warn(field?.Value == null ? "Input parameter 'ray_field' failed to collect data." : "The ray field has no rays.");
                return;
            }
            Worker.EnsureStarted();
            var speedOfSound = Worker.SpeedOfSound;
            var rays = RayGeometry.Specular(field.Value);
            // one button: Play, and Stop while playing (it holds the particles where they are)
            var pressed = Pressed("play");
            if (pressed && _playback.Running)
            {
                _playback.Stop();
                pressed = false;
            }
            if (Pressed("rewind"))
                _playback.Rewind();
            var frame = _playback.Update(Value<double>(access, "time"), Value<bool>(access, "play") == true || pressed, Value<double>(access, "speed"),
                Particles.DurationMs(rays, speedOfSound));
            if (frame.Remark != null)
                Remark(frame.Remark);
            if (frame.Animating)
                OnPingDocument()?.ScheduleSolution(FrameMs, document => ExpireSolution(false));
            Message = frame.Message;
            var particles = Particles.At(rays, frame.T, speedOfSound);
            foreach (var (point, order) in particles)
            {
                if (!_drawn.TryGetValue(order, out var points))
                    _drawn[order] = points = new List<Point3d>();
                points.Add(point);
            }
            // the whole field, not the particles: the box stays still while they move
            Box = BoxOf(rays.SelectMany(r => r.Points));
            access.SetDataList("particles", particles.Select(p => p.point));
        }

        public override void DrawViewportWires(IGH_PreviewArgs args)
        {
            base.DrawViewportWires(args);
            foreach (var (order, points) in _drawn)
                args.Display.DrawPoints(points, Rhino.Display.PointStyle.RoundSimple, ParticleRadius, OrderColors.Of(order));
        }
    }
}
