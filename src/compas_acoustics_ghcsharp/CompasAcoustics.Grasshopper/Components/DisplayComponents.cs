using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using CompasAcoustics.Pb;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>A component that draws its own geometry, besides the previews of its outputs.</summary>
    public abstract class DisplayComponent : ComponentBase
    {
        /// <summary>The time between two frames of a playback.</summary>
        protected const int FrameMs = 33;

        protected BoundingBox Box = BoundingBox.Empty;

        protected DisplayComponent(string key) : base(key) { }

        /// <summary>
        /// The drawing on and off on the body: the component's preview, as its right-click menu sets it. Its port is no input's name: a
        /// control on an input's name sets that input (Polar Preview's preview, its source index, took every click of a switch named so).
        /// Hidden holds its state, which Grasshopper saves, so the name is saved nowhere.
        /// </summary>
        protected static readonly ToggleControl PreviewSwitch = new("show", "Preview");

        /// <summary>The playback speed on the body, a factor of the default (see <see cref="Rate"/>).</summary>
        protected static readonly DropdownControl SpeedDropdown = new("speed", new[] { "0.25", "0.5", "1", "2", "4" }, "1", new[] { "0.25x", "0.5x", "1x", "2x", "4x" });

        /// <summary>The playback speed, in ms of sound per second: <paramref name="factor"/> (default 1) of the whole field in 10 s.</summary>
        public static double Rate(double? factor, double durationMs) => (factor ?? 1.0) * durationMs / 10.0;

        /// <summary>Not the outputs: the component draws its geometry itself.</summary>
        public override bool PreviewsOutput(IGH_Param output) => false;

        /// <summary>What the Play button reads, and so the glyph it draws: Pause while the playback moves.</summary>
        public static string PlayLabel(Playback playback) => playback.Running ? "Pause" : "Play";

        /// <summary>A playback on the clock of this process.</summary>
        protected static Playback NewPlayback() => new(() => System.Diagnostics.Stopwatch.GetTimestamp() / (double)System.Diagnostics.Stopwatch.Frequency);

        /// <summary>
        /// The Play and Rewind buttons of <paramref name="playback"/>: Play pauses a moving playback (it holds where it is;
        /// Play goes on), Rewind goes back to the start. Whether Play starts or resumes the playback.
        /// </summary>
        protected bool PressPlayback(Playback playback)
        {
            var play = Pressed("play");
            if (play && playback.Running)
            {
                playback.Pause();
                play = false;
            }
            if (Pressed("rewind"))
                playback.Rewind();
            return play;
        }

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
        public override float MinBodyWidth => VisualiseBodyWidth;
        public override Guid ComponentGuid => new Guid("11077914-3d8f-5a76-a0f9-07250add684e");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => (input, port.Name) switch
        {
            (true, "scene") => new Param_Scene(),
            (false, "reflections") => new Param_Reflections(),
            _ => null,
        };

        protected override void Clear() => _drawn.Clear();

        protected override bool RunsInBackground => true;

        public override IReadOnlyList<Control> Controls { get; } = new Control[]
        {
            PreviewSwitch, SceneIndices.SourceDropdown, SceneIndices.ReceiverDropdown,
            new MultiSwitchControl("order", new[] { "0", "1", "2", "3", "4" }, new[] { "1" }),
        };

        protected override void Solve(IGH_DataAccess access)
        {
            if (Required<SceneGoo>(access, "scene") is not { } scene)
                return;
            SetIndexOptions("source_idx", scene.Value.Sources.Select(x => x.Name));
            SetIndexOptions("receiver_idx", scene.Value.Receivers.Select(x => x.Name));
            var orders = Items<int>(access, "order").Select(o => (object?)(long)o).ToList();
            var source = Value<int>(access, "source_idx");
            var receiver = Value<int>(access, "receiver_idx");
            var kwargs = new Dictionary<string, object?> { ["source_index"] = source is int s ? (long)s : null, ["receiver_index"] = receiver is int r ? (long)r : null };
            RunInBackground(access, new Background("specular_reflections", new object?[] { scene.Value, orders }, kwargs), (outcome, _) =>
            {
                var result = outcome.Result;
                var found = (SpecularReflectionsData)result["reflections"]!;
                var paths = RayGeometry.FromReflections(found);
                // drawn here, not output as geometry: Grasshopper would preview polylines in one colour on top
                _drawn.AddRange(paths.Select(p => (RayGeometry.ToPolyline(p.Ray), OrderColors.Of(p.Order))));
                Box = BoxOf(paths.SelectMany(p => p.Ray.Points));
                var counts = ((IEnumerable<object?>)result["counts"]!).Cast<IList<object?>>().Select(c => $"order {c[0]}: {c[1]}").ToList();
                Status(counts.Count > 0 ? string.Join(", ", counts) : "no paths");
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

    /// <summary>
    /// Visualise Rays and Visualise Billiard: rays traced from the scene's sources (the worker's ray_field) and played on the
    /// body, with Play and Rewind, a time slider and Speed. Nothing is drawn before Play or a time.
    /// </summary>
    public abstract class TracedPlaybackComponent : DisplayComponent
    {
        private readonly Playback _playback = NewPlayback();
        private int _maxOrder = 10;
        private double _durationMs;
        private double? _shownMs;

        protected TracedPlaybackComponent(string key) : base(key) { }

        protected static readonly SliderControl TimeSlider = new("time", 0, 1000, "ms", Step: 0.1, Live: true);
        protected static readonly ButtonControl PlayButton = new("play", "Play", Inline: true, Icon: true);
        protected static readonly ButtonControl RewindButton = new("rewind", "Rewind", Inline: true, Icon: true);

        /// <summary>The input of the number of rays traced per source.</summary>
        protected abstract string CountInput { get; }

        /// <summary>What the status names the traced rays before Play.</summary>
        protected abstract string CountName { get; }

        /// <summary>Whether the rays also scatter at the reflections (the scattering and seed inputs).</summary>
        protected virtual bool Scatters => false;

        /// <summary>The traced order: the order slider chooses what is drawn, so it is hidden with the control inputs.</summary>
        protected override IEnumerable<string> HiddenInputs => new[] { "max_order" };

        public override (double min, double max) SliderRange(SliderControl slider) => slider.Port == "time" ? (0, Math.Round(_durationMs, 1)) : base.SliderRange(slider);

        /// <summary>The time slider follows the frame shown, as playback moves it.</summary>
        public override double? SliderShown(SliderControl slider) => slider.Port == "time" ? _shownMs : null;

        public override string? ButtonLabel(string port) => port == "play" ? PlayLabel(_playback) : null;

        /// <summary>
        /// The drawn order's stops: 1 to <paramref name="maxOrder"/>, and All (the empty input, every traced order) last and the default.
        /// Beyond 12 orders only every k-th stop is labelled, from the first, so the labels stay apart.
        /// </summary>
        internal static StepSliderControl OrderStops(int maxOrder)
        {
            var count = Math.Max(maxOrder, 1);
            var every = (count + 11) / 12;
            var orders = Enumerable.Range(1, count).Select(n => n.ToString(System.Globalization.CultureInfo.InvariantCulture)).ToArray();
            var labels = orders.Select((o, i) => every == 1 || ((i % every == 0) && i + 1 < count) ? o : "").Append("All").ToArray();
            return new StepSliderControl("order", orders.Append("").ToArray(), "", labels, "Order");
        }

        /// <summary>The order slider's stops follow the traced order.</summary>
        public override StepSliderControl StepStopsOf(StepSliderControl step) => step.Port == "order" ? OrderStops(_maxOrder) : base.StepStopsOf(step);

        /// <summary>How far the rays are drawn: nothing until Play or a time, then as far as sound has travelled in <paramref name="timeMs"/>.</summary>
        public static double DrawnDistance(bool active, double timeMs, double speedOfSound) => active ? timeMs / 1000.0 * speedOfSound : 0.0;

        public override float MinBodyWidth => VisualiseBodyWidth;

        protected override bool RunsInBackground => true;

        /// <summary>The rays of the shown field that are played, converted once per field.</summary>
        protected abstract List<RayGeometry.Ray> Played(RayFieldData shown);

        /// <summary>Draw and output the frame at <paramref name="timeMs"/>; <paramref name="drawn"/> is false before Play or a time.</summary>
        protected abstract void Show(IGH_DataAccess access, RayFieldData field, List<RayGeometry.Ray> rays, double timeMs, bool drawn, double speedOfSound);

        protected override void Solve(IGH_DataAccess access)
        {
            if (Required<SceneGoo>(access, "scene") is not { } scene)
                return;
            SetIndexOptions("source_idx", scene.Value.Sources.Select(x => x.Name));
            var maxOrder = Value<int>(access, "max_order") ?? 10;
            _maxOrder = maxOrder;
            var order = Value<int>(access, "order");
            var direction = Value<Vector3d>(access, "direction");
            var kwargs = new Dictionary<string, object?>
            {
                ["source_index"] = Value<int>(access, "source_idx") is int s ? (long)s : null,
                ["n_rays"] = Value<int>(access, CountInput) is int n ? (long)n : null,
                ["max_order"] = (long)maxOrder,
                ["direction"] = direction is Vector3d d ? MeshConversion.Xyz(d) : null,
                ["angle"] = Value<double>(access, "angle"),
                ["scattering"] = Scatters && (Value<bool>(access, "scattering") ?? false),
                ["seed"] = Scatters && Value<int>(access, "seed") is int seed ? (long)seed : null,
                ["order"] = order is int o ? (long)o : null,
            };
            RunInBackground(access, new Background("ray_field", new object?[] { scene.Value }, kwargs), (outcome, _) =>
            {
                var field = (RayFieldData)outcome.Result["field"]!;
                var rays = Played((RayFieldData)outcome.Result["shown"]!);
                // nothing until Play or a time: all rays at once are a blur; they are drawn as far as sound has travelled
                var speedOfSound = Worker.SpeedOfSound;
                var pressed = PressPlayback(_playback);
                // Play on the body plays the whole field: the time the slider scrubbed to is let go. A shown input has
                // collected it for this solve already, so it is passed as none now and collected again on the next frame.
                var letGo = pressed && ControlPort("time") is { SourceCount: 0 };
                if (letGo)
                    ControlState.Assign(ControlPort("time")!, System.Array.Empty<string>());
                _durationMs = Particles.DurationMs(rays, speedOfSound);
                var time = letGo ? null : Value<double>(access, "time");
                var frame = _playback.Update(time, pressed, Rate(Value<double>(access, "speed"), _durationMs), _durationMs);
                _shownMs = frame.T;
                if (frame.Remark != null)
                    Remark(frame.Remark);
                if (frame.Animating || letGo)
                    OnPingDocument()?.ScheduleSolution(FrameMs, document =>
                    {
                        if (letGo && ControlPort("time") is { } port && !IsHidden(port))
                            port.ExpireSolution(false);
                        ExpireSolution(false);
                    });
                var drawn = _playback.Active || time != null;
                Status(drawn ? frame.Message : $"{field.Rays.Count} {CountName}, press Play");
                // the whole field, not the frame: the box stays still while the rays grow
                Box = BoxOf(rays.SelectMany(r => r.Points));
                Show(access, field, rays, frame.T, drawn, speedOfSound);
            });
        }
    }

    public sealed class VisualiseRaysComponent : TracedPlaybackComponent
    {
        private const int LineThickness = 1;
        private readonly Dictionary<int, List<Line>> _drawn = new();
        // the shown field's rays and curves, the same for every frame of a playback
        private (RayFieldData shown, List<RayGeometry.Ray> rays, PolylineCurve[] specular, PolylineCurve[] scattered)? _converted;

        public VisualiseRaysComponent() : base("CA_Visualise_Rays") { }

        public override IReadOnlyList<Control> Controls { get; } = new Control[]
        {
            PreviewSwitch, SceneIndices.SourceDropdown, new ToggleControl("scattering", "Scattering", Default: false), OrderStops(10),
            TimeSlider, SpeedDropdown, PlayButton, RewindButton,
        };

        protected override string CountInput => "n_rays";
        protected override string CountName => "rays";
        protected override bool Scatters => true;

        /// <summary>The seed of the scattered rays, rarely set: hidden with the control inputs too.</summary>
        protected override IEnumerable<string> HiddenInputs => base.HiddenInputs.Append("seed");

        /// <summary>
        /// A definition saved before the rays had their time and speed inputs: those are set aside while Grasshopper reads the
        /// saved ones by position, and come back after it, hidden when the definition saved its control inputs hidden.
        /// </summary>
        public override bool Read(GH_IReader reader)
        {
            var added = new[] { "time", "speed" }.Where(n => !SavedControlPort(reader, n)).Select(n => Params.Input.Single(p => p.Name == n)).ToList();
            if (added.Count == 0)
                return base.Read(reader);
            foreach (var port in added)
                Params.UnregisterInputParameter(port, false);
            var read = base.Read(reader);
            foreach (var port in added)
                Params.RegisterInputParam(port);
            Params.OnParametersChanged();
            HideNewControlPortsAsRead(added.Select(p => p.Name));
            return read;
        }

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

        protected override List<RayGeometry.Ray> Played(RayFieldData shown)
        {
            if (_converted is not { } converted || !ReferenceEquals(converted.shown, shown))
            {
                var all = RayGeometry.FromField(shown);
                _converted = converted = (shown, all, all.Where(r => !r.Scattered).Select(r => new PolylineCurve(r.Points)).ToArray(),
                    all.Where(r => r.Scattered).Select(r => new PolylineCurve(r.Points)).ToArray());
            }
            return converted.rays;
        }

        protected override void Show(IGH_DataAccess access, RayFieldData field, List<RayGeometry.Ray> rays, double timeMs, bool drawn, double speedOfSound)
        {
            var distance = DrawnDistance(drawn, timeMs, speedOfSound);
            foreach (var ray in rays)
                foreach (var (k, line) in RayGeometry.Segments(RayGeometry.UpTo(ray, distance)))
                {
                    if (!_drawn.TryGetValue(k, out var lines))
                        _drawn[k] = lines = new List<Line>();
                    lines.Add(line);
                }
            access.SetData("ray_field", new RayFieldGoo(field));
            access.SetDataList("rays", _converted!.Value.specular);
            access.SetDataList("scattered", _converted!.Value.scattered);
        }

        public override void DrawViewportWires(IGH_PreviewArgs args)
        {
            base.DrawViewportWires(args);
            foreach (var (k, lines) in _drawn)
                args.Display.DrawLines(lines, OrderColors.Of(k), LineThickness);
        }
    }

    /// <summary>The rays traced from the scene's sources as balls, like a 3D billiard: the same tracing and playback as Visualise Rays, without scattering.</summary>
    public sealed class VisualiseBilliardComponent : TracedPlaybackComponent
    {
        private const float ParticleRadius = 4;
        private readonly Dictionary<int, List<Point3d>> _drawn = new();
        // the shown field's rays, the same for every frame of a playback
        private (RayFieldData shown, List<RayGeometry.Ray> rays)? _converted;

        public VisualiseBilliardComponent() : base("CA_Visualise_Billiard") { }

        public override IReadOnlyList<Control> Controls { get; } = new Control[]
        {
            PreviewSwitch, SceneIndices.SourceDropdown, OrderStops(10), TimeSlider, SpeedDropdown, PlayButton, RewindButton,
        };

        protected override string CountInput => "n_balls";
        protected override string CountName => "balls";

        protected override string SpecKey => "CA_Visualise_Billiard";
        public override Guid ComponentGuid => new Guid("1dd7e727-b816-578f-8eaa-253a9a153829");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => input && port.Name == "scene" ? new Param_Scene() : null;

        /// <summary>
        /// A definition saved when the billiard played a ray field (the inputs ray_field, time, play and speed): those inputs
        /// hold their places while Grasshopper reads the saved ones by position and go after it, with their wires; the billiard
        /// then has the inputs of a new one, its control inputs hidden, and needs a scene.
        /// </summary>
        public override bool Read(GH_IReader reader)
        {
            if (!SavedInput(reader, "ray_field"))
                return base.Read(reader);
            var inputs = Params.Input.ToList();
            foreach (var port in inputs)
                Params.UnregisterInputParameter(port, false);
            var former = new List<IGH_Param>();
            foreach (var (name, port) in new (string, IGH_Param)[] { ("ray_field", new Param_RayField()), ("time", new Param_Number()), ("play", new Param_Boolean()), ("speed", new Param_Number()) })
                if (SavedInput(reader, name))
                {
                    port.Name = port.NickName = name;
                    port.Optional = true;
                    Params.RegisterInputParam(port);
                    former.Add(port);
                }
            var read = base.Read(reader);
            foreach (var port in former)
                Params.UnregisterInputParameter(port, true);
            foreach (var port in inputs)
                Params.RegisterInputParam(port);
            Params.OnParametersChanged();
            HideControlPortsAsNew();
            return read;
        }

        protected override void Clear() => _drawn.Clear();

        protected override List<RayGeometry.Ray> Played(RayFieldData shown)
        {
            if (_converted is not { } converted || !ReferenceEquals(converted.shown, shown))
                _converted = converted = (shown, RayGeometry.Specular(shown));
            return converted.rays;
        }

        protected override void Show(IGH_DataAccess access, RayFieldData field, List<RayGeometry.Ray> rays, double timeMs, bool drawn, double speedOfSound)
        {
            if (!drawn)
                return;
            var particles = Particles.At(rays, timeMs, speedOfSound, Particles.DurationMs(rays, speedOfSound));
            foreach (var (point, order) in particles)
            {
                if (!_drawn.TryGetValue(order, out var points))
                    _drawn[order] = points = new List<Point3d>();
                points.Add(point);
            }
            access.SetDataList("particles", particles.Select(p => p.point));
        }

        public override void DrawViewportWires(IGH_PreviewArgs args)
        {
            base.DrawViewportWires(args);
            foreach (var (order, points) in _drawn)
                args.Display.DrawPoints(points, Rhino.Display.PointStyle.RoundSimple, ParticleRadius, OrderColors.Of(order));
        }
    }

    /// <summary>A mesh coloured by the sound pressure level of the scene's sources at its face centres (the worker's energy_map).</summary>
    public sealed class VisualiseEnergyMapComponent : DisplayComponent
    {
        private static readonly string[] Bands = { "125 Hz", "250 Hz", "500 Hz", "1 kHz", "2 kHz", "4 kHz", "8 kHz" };
        private Mesh? _drawn;

        public VisualiseEnergyMapComponent() : base("CA_Visualise_Energy_Map") { }
        protected override string SpecKey => "CA_Visualise_Energy_Map";
        public override float MinBodyWidth => VisualiseBodyWidth;
        public override Guid ComponentGuid => new Guid("a0fc70ee-6c93-5bb3-b30d-315b8e4530cf");

        // a static map of the levels: no playback controls
        public override IReadOnlyList<Control> Controls { get; } = new Control[]
        {
            PreviewSwitch,
            new DropdownControl("frequency_bands", new[] { "", "0", "1", "2", "3", "4", "5", "6" }, "", new[] { "Full (A-weighted)" }.Concat(Bands).ToArray()),
            SceneIndices.SourceDropdown,
            new ButtonControl("run", "Run"),
        };

        /// <summary>
        /// Why the map waits, while it does: it runs only on a press, since it takes seconds and shares the worker with the
        /// components that run on every change of the scene. Changed inputs keep the last map drawn.
        /// </summary>
        public static (string text, bool changed)? Waiting(RunDecision decision) => decision.Kind != RunKind.Idle ? null
            : decision.Result == null ? ("press Run", false) : ("inputs changed, press Run", true);

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => (input, port.Name) switch
        {
            (true, "scene") => new Param_Scene(),
            (true, "mesh") => new Param_Geometry(),
            (true, "range") => new Param_Interval(),
            (false, "mesh") => new Param_Mesh(),
            (false, "values") => new Param_Number(),
            (false, "colours") => new Param_Colour(),
            _ => null,
        };

        protected override void Clear() => _drawn = null;

        protected override bool RunsInBackground => true;

        protected override void Solve(IGH_DataAccess access)
        {
            if (Required<SceneGoo>(access, "scene") is not { } scene)
                return;
            var spacing = Value<double>(access, "spacing") ?? 0.25;
            if (spacing <= 0)
            {
                Warn("Input parameter 'spacing' must be positive.");
                return;
            }
            var mesh = SurfaceMesh(Item<IGH_Goo>(access, "mesh")?.ScriptVariable(), spacing);
            if (mesh == null || mesh.Faces.Count == 0)
            {
                Warn("Input parameter 'mesh' failed to collect data: connect a mesh, a surface or a Brep.");
                return;
            }
            SetIndexOptions("source_idx", scene.Value.Sources.Select(x => x.Name));
            var band = Value<int>(access, "frequency_bands");
            if (band is int b && (b < 0 || b >= Bands.Length))
            {
                Warn($"Input parameter 'frequency_bands' must be 0 to {Bands.Length - 1}; empty for the full, A-weighted level.");
                return;
            }
            var range = Value<Interval>(access, "range");
            // a point every 'spacing': the faces of the mesh given (a plane of one face) split into grids
            var (corners, faceCorners) = EnergyMapColours.Subdivide(mesh.Vertices.ToPoint3dArray(),
                mesh.Faces.Select(MeshConversion.Corners).ToArray(), spacing);
            var centres = EnergyMapColours.FaceCentres(corners, faceCorners);
            // the image source order, the rays and their seed are the worker's: the same scene gives the same map
            var kwargs = new Dictionary<string, object?> { ["source_index"] = Value<int>(access, "source_idx") is int s ? (long)s : null };
            var points = centres.Select(c => (object?)MeshConversion.Xyz(c)).ToList();
            // the mesh travels with its result: while a changed mesh runs, the last map is drawn on the mesh it was made for
            var pressed = Value<bool>(access, "run") == true || Pressed("run");
            var decision = RunInBackground(access, new Background("energy_map", new object?[] { scene.Value, points }, kwargs, (mesh, corners, faceCorners)), (outcome, _) =>
            {
                var result = outcome.Result;
                var (_, vertices, faces) = ((Mesh, Point3d[], int[][]))outcome.Context!;
                var level = band is int index
                    ? Rows(result["spl"]).Select(row => row[index]).ToArray()
                    : Numbers(result["spl_a"]);
                var scale = EnergyMapColours.Range(level, range);
                var label = band is int i ? Bands[i] : "dB(A)";
                var colours = level.Select(v => EnergyMapColours.Of(v, scale.low, scale.high)).ToArray();
                _drawn = EnergyMapColours.Coloured(vertices, faces, colours);
                Box = BoxOf(vertices);
                Status(string.Format(System.Globalization.CultureInfo.InvariantCulture, "{0} {1:0.0} to {2:0.0} dB", label, scale.low, scale.high),
                    string.Format(System.Globalization.CultureInfo.InvariantCulture, "{0} {1:0.0} to {2:0.0} dB\n{3} points{4}",
                        label, scale.low, scale.high, faces.Length,
                        result.TryGetValue("n_rays", out var rays) && rays != null ? $", {rays} rays per source" : ""));
                access.SetData("mesh", _drawn);
                access.SetDataList("values", level);
                access.SetDataList("colours", colours);
            }, start: pressed);
            if (Waiting(decision) is { } waiting)
            {
                Status(waiting.text);
                if (waiting.changed)
                    Warn("The inputs differ from the last map: press Run to update it.");
            }
        }

        /// <summary>The analysis surface as a mesh: a mesh as it is, a surface or Brep meshed with faces no larger than <paramref name="spacing"/>.</summary>
        private static Mesh? SurfaceMesh(object? geometry, double spacing)
        {
            Brep? brep = geometry switch
            {
                Brep b => b,
                Surface surface => surface.ToBrep(),
                _ => null,
            };
            if (geometry is Mesh given)
                return given;
            if (brep == null)
                return null;
            var parameters = new MeshingParameters { MaximumEdgeLength = spacing, MinimumEdgeLength = spacing / 10, JaggedSeams = true, SimplePlanes = false };
            var meshes = Mesh.CreateFromBrep(brep, parameters);
            if (meshes == null || meshes.Length == 0)
                return null;
            var joined = new Mesh();
            joined.Append(meshes);
            return joined;
        }

        private static double[] Numbers(object? list) => ((IEnumerable<object?>)list!).Select(v => Convert.ToDouble(v, System.Globalization.CultureInfo.InvariantCulture)).ToArray();

        private static double[][] Rows(object? list) => ((IEnumerable<object?>)list!).Select(Numbers).ToArray();

        public override void DrawViewportMeshes(IGH_PreviewArgs args)
        {
            base.DrawViewportMeshes(args);
            if (_drawn != null)
                args.Display.DrawMeshFalseColors(_drawn);
        }
    }
}
