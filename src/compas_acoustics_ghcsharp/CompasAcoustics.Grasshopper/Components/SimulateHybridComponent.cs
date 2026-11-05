using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CompasAcoustics.Pb;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// Simulate Binaural and Simulate Ambisonics: the hybrid engine in the worker, started by Run. A subclass names its method
    /// and its output; the inputs only one of them has (``sh_order``, ``hrtf``) are read where the component has them.
    /// </summary>
    public abstract class SimulateHybridComponent : ComponentBase
    {
        private HybridRun.LastResult? _last;

        protected SimulateHybridComponent(string key) : base(key) { }

        /// <summary>The worker's method name, which is also the graph attribute of its responses.</summary>
        protected abstract string Method { get; }

        /// <summary>The one output, an IR graph of the method's responses.</summary>
        protected abstract string Output { get; }

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => (input, port.Name) switch
        {
            (true, "scene") => new Param_Scene(),
            (true, "reference") => new Param_IrGraph(),
            (false, _) => new Param_IrGraph(),
            _ => null,
        };

        private IReadOnlyList<Control>? _controls;

        /// <summary>ISM order, the method's own controls, Tail, Diffraction, the toggles, the method's own buttons, Run.</summary>
        public override IReadOnlyList<Control> Controls => _controls ??= new Control[] { SimulateMonoComponent.IsmOrder }
            .Concat(MethodControls)
            .Append(new DropdownControl("tail", HybridRun.Tails))
            .Append(new DropdownControl("diffraction", HybridRun.Diffraction))
            .Append(new ToggleControl("late_envelope", "Late envelope", Inline: true))
            .Append(new ToggleControl("normalize", "Normalize", Inline: true))
            .Concat(MethodButtons)
            .Append(new ButtonControl("run", "Run"))
            .ToList();

        /// <summary>The controls of the method's own inputs, after the ISM order (Ambisonics' SH order).</summary>
        protected virtual IEnumerable<Control> MethodControls => Array.Empty<Control>();

        /// <summary>The buttons of the method's own inputs, before Run (Binaural's HRTF set).</summary>
        protected virtual IEnumerable<Control> MethodButtons => Array.Empty<Control>();

        /// <summary>The ambisonic order the method renders; null for one without (Binaural).</summary>
        protected virtual int? ShOrder => null;

        private bool Has(string port) => Params.Input.Exists(p => p.Name == port);

        private void SetGraph(IGH_DataAccess access, string sceneGuid) =>
            access.SetData(Output, _last != null && _last.SceneGuid == sceneGuid ? _last.Graph : null);

        protected override void Solve(IGH_DataAccess access)
        {
            var scene = Item<SceneGoo>(access, "scene");
            if (scene?.Value == null)
            {
                Warn("Input parameter 'scene' failed to collect data.");
                return;
            }
            var tail = Text(access, "tail");
            var reference = Item<IrGraphGoo>(access, "reference");
            // checked before 'run' so that it shows as soon as the tail is chosen, not only after pressing Run
            if (tail == "reference" && reference == null)
            {
                Warn("tail 'reference' needs a reference: connect the mono_ir_graph of Simulate Mono to 'reference', or choose another tail.");
                Message = "connect a reference";
                SetGraph(access, scene.Value.Guid);
                return;
            }
            var shOrder = ShOrder;
            var hrtf = Has("hrtf") ? Text(access, "hrtf") : null;
            var ismOrder = Value<int>(access, "ism_order");
            var diffraction = Text(access, "diffraction");
            var seed = Value<int>(access, "seed");
            Dictionary<string, object?> settings;
            Dictionary<string, ImpulseResponseData>? references;
            try
            {
                settings = HybridRun.Settings(tail, ismOrder, diffraction, Value<bool>(access, "late_envelope"), seed,
                    Value<bool>(access, "normalize"), reference, scene.SamplingRate, out references);
            }
            catch (ArgumentException e)
            {
                Error(e.Message);
                return;
            }
            if (shOrder is int sh)
                settings["sh_order"] = (long)sh;
            if (hrtf != null)
                settings["hrtf"] = hrtf;
            var refs = references?.ToDictionary(kv => kv.Key, kv => (object?)kv.Value);
            var args = new object?[] { scene.Value, Method, settings, refs };
            // a seed drawn for this solution is not a change of the inputs
            var keySettings = new Dictionary<string, object?>(settings);
            if (seed == null)
                keySettings.Remove("seed");
            var key = AsyncRuns.Key("simulate", new object?[] { scene.Value, Method, keySettings, refs }, new Dictionary<string, object?>());
            var request = new Background("simulate", args, new Dictionary<string, object?>(), new Context(scene, settings["seed"], hrtf), key);
            var shown = false;
            void Deliver(Outcome outcome, bool fresh)
            {
                var result = (IDictionary<string, object?>)outcome.Result!;
                ShowWarnings(result);
                if (fresh)
                    Keep((Context)outcome.Context!, result);
                ShowStatus(fresh ? RunMessage((Context)outcome.Context!) : HybridRun.Status(_last, scene.Value.Guid, tail, shOrder, ismOrder, diffraction));
                SetGraph(access, scene.Value.Guid);
                shown = true;
            }
            // the simulation can take minutes and Grasshopper recomputes on every change of the inputs, so it starts only
            // when 'run' is true (a button pulse); otherwise the responses of the last run for this scene are passed on
            var pressed = Value<bool>(access, "run") == true || Pressed("run");
            var decision = RunInBackground(access, request, Deliver, start: pressed, restart: pressed);
            if (decision.Kind == RunKind.Idle)
            {
                if (!shown)
                {
                    ShowStatus(HybridRun.Status(_last, scene.Value.Guid, tail, shOrder, ismOrder, diffraction));
                    SetGraph(access, scene.Value.Guid);
                }
                if (decision.Interrupted)
                    ShowStatus("settings changed, press Run");
            }
            else if (!shown && decision.Kind != RunKind.Failed)
                SetGraph(access, scene.Value.Guid);
        }

        /// <summary>The message; while the settings differ from the last result, a warning colours the body.</summary>
        private void ShowStatus(string text)
        {
            Status(text);
            if (text.StartsWith("settings changed"))
                Warn("The settings differ from the last run: press Run to update the responses.");
        }

        private sealed record Context(SceneGoo Scene, object? Seed, string? Hrtf);

        protected override bool RunsInBackground => true;

        protected override async Task<object?> Call(Background request, string callId, CancellationToken cancel)
        {
            await Task.Run(Worker.EnsureStarted, cancel).ConfigureAwait(false);
            if (!Worker.Hybrid)
                throw new WorkerException($"{Name} needs a worker with pyFDN: use Install Worker.");
            return await base.Call(request, callId, cancel).ConfigureAwait(false);
        }

        private void Keep(Context context, IDictionary<string, object?> result)
        {
            var responses = (IDictionary<string, object?>)((IDictionary<string, object?>)result["responses"]!)[Method]!;
            var graph = IrGraphGoo.FromSimulation(context.Scene, Method, responses);
            var first = graph.Value.Responses.FirstOrDefault()?.Generation ?? new Dictionary<string, object?>();
            _last = new HybridRun.LastResult(context.Scene.Value.Guid, graph, new Dictionary<string, object?>(first));
        }

        private string RunMessage(Context context)
        {
            var generation = _last?.Generation ?? new Dictionary<string, object?>();
            generation.TryGetValue("tail", out var tail);
            var order = generation.TryGetValue("sh_order", out var sh) ? $", order {sh}" : "";
            var set = Has("hrtf") ? $", hrtf {(string.IsNullOrEmpty(context.Hrtf) ? "kemar" : Path.GetFileName(context.Hrtf))}" : "";
            return $"{context.Scene.SamplingRate}Hz, {Method}, tail: {tail ?? "auto"}{order}, seed {context.Seed}{set}";
        }
    }

    public sealed class SimulateBinauralComponent : SimulateHybridComponent
    {
        public SimulateBinauralComponent() : base("CA_Simulate_Binaural") { }
        protected override string SpecKey => "CA_Simulate_Binaural";
        public override Guid ComponentGuid => new Guid("2c46c2f3-7b8b-43fd-ba7b-b249681d829b");
        protected override string Method => "binaural";
        protected override string Output => "binaural_ir_graph";

        // empty is KEMAR; the button chooses the user's own SOFA set, and a File Path wired to hrtf wins
        protected override IEnumerable<Control> MethodButtons => new Control[] { new ButtonControl("hrtf", "HRTF...") };

        public override bool HandlesButton(string port) => port == "hrtf";

        public override void OnButton(string port)
        {
            if (port == "hrtf")
                FileButtons.Choose(this, port, "HRTF set of the binaural responses", "SOFA HRTF set", ".sofa");
        }

        public override string? ButtonLabel(string port) => port == "hrtf" ? FileButtons.Label(this, port, "HRTF: ", "HRTF: KEMAR") : null;

        protected override void AppendMenu(System.Windows.Forms.ToolStripDropDown menu) => FileButtons.AppendClear(this, menu, "hrtf", "Use KEMAR");
    }

    public sealed class SimulateAmbisonicsComponent : SimulateHybridComponent
    {
        public SimulateAmbisonicsComponent() : base("CA_Simulate_Ambisonics") { }
        protected override string SpecKey => "CA_Simulate_Ambisonics";
        public override Guid ComponentGuid => new Guid("5da5e5fc-80d6-4a8a-9170-91d3d3a3f94d");
        protected override string Method => "ambisonics";
        protected override string Output => "ambisonic_ir_graph";

        // the component's own choice, not an input: 1 (the method's default) to 7
        protected override IEnumerable<Control> MethodControls => new Control[]
        {
            new DropdownControl("sh_order", new[] { "1", "2", "3", "4", "5", "6", "7" }, "1", Title: "SH order"),
        };

        protected override int? ShOrder => int.Parse(Choice("sh_order"), System.Globalization.CultureInfo.InvariantCulture);
    }
}
