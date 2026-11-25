using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CompasAcoustics.Pb;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// Simulate Binaural and Simulate Ambisonics: the hybrid method in the worker, started by Run. A subclass names its method
    /// and its output, and gives the settings only it has.
    /// </summary>
    public abstract class SimulateHybridComponent : ComponentBase
    {
        private HybridRun.LastResult? _last;

        protected SimulateHybridComponent(string key) : base(key) { }

        /// <summary>A late_envelope input that holds a boolean reads as a mode: true as directional, false as off.</summary>
        public override bool Read(GH_IReader reader)
        {
            // before base.Read: a boolean does not survive Grasshopper's read into a text input
            var toggle = SavedBoolean(reader, "late_envelope");
            var read = base.Read(reader);
            if (ControlPort("late_envelope") is Param_String port)
            {
                var value = ControlState.PersistentAll(port).FirstOrDefault();
                var mode = string.IsNullOrEmpty(value) ? toggle switch { true => "directional", false => "off", _ => null }
                    : string.Equals(value, "true", StringComparison.OrdinalIgnoreCase) ? "directional"
                    : string.Equals(value, "false", StringComparison.OrdinalIgnoreCase) ? "off"
                    : null;
                if (mode != null)
                {
                    port.PersistentData.Clear();
                    port.PersistentData.Append(new GH_String(mode));
                }
            }
            return read;
        }

        /// <summary>The worker's method name, which is also the key of its responses in the worker's result.</summary>
        protected abstract string Method { get; }

        /// <summary>The one output, the method's responses.</summary>
        protected abstract string Output { get; }

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => (input, port.Name) switch
        {
            (true, "scene") => new Param_Scene(),
            (true, "reference") => new Param_ImpulseResponse(),
            (false, _) => new Param_ImpulseResponse(),
            _ => null,
        };

        private IReadOnlyList<Control>? _controls;

        /// <summary>ISM order, the method's own controls, Tail, Diffraction, the toggles, the method's own buttons, Run.</summary>
        public override IReadOnlyList<Control> Controls => _controls ??= new Control[] { SimulateMonoComponent.IsmOrder }
            .Concat(MethodControls)
            .Append(new DropdownControl("tail", HybridRun.Tails))
            .Append(new DropdownControl("diffraction", HybridRun.Diffraction))
            .Append(new DropdownControl("late_envelope", HybridRun.LateEnvelopes))
            .Append(new ToggleControl("normalize", "Normalize", Inline: true))
            .Concat(MethodButtons)
            .Append(new ButtonControl("run", "Run"))
            .ToList();

        /// <summary>The controls of the method's own inputs, after the ISM order.</summary>
        protected virtual IEnumerable<Control> MethodControls => Array.Empty<Control>();

        /// <summary>The buttons of the method's own inputs, before Run.</summary>
        protected virtual IEnumerable<Control> MethodButtons => Array.Empty<Control>();

        /// <summary>The ambisonic order the method renders; null for a method without one.</summary>
        protected virtual int? ShOrder => null;

        /// <summary>Whether the method renders with an HRTF set.</summary>
        protected virtual bool TakesHrtf => false;

        /// <summary>The HRTF set chosen for the method, whether its input is shown or hidden; null for the default or a method without one.</summary>
        internal string? Hrtf(IGH_DataAccess access) => TakesHrtf ? Text(access, "hrtf") : null;

        /// <summary>The HRTF set as the message names it: the file's name, or ku100 for the default.</summary>
        internal static string HrtfLabel(string? hrtf) => string.IsNullOrEmpty(hrtf) ? "ku100" : Path.GetFileName(hrtf);

        private void SetResponses(IGH_DataAccess access, string sceneGuid) =>
            access.SetDataList(Output, _last != null && _last.SceneGuid == sceneGuid ? _last.Responses : Array.Empty<ImpulseResponseGoo>());

        protected override void Solve(IGH_DataAccess access)
        {
            if (Required<SceneGoo>(access, "scene") is not { } scene)
                return;
            var tail = Text(access, "tail");
            var references = Items<ImpulseResponseGoo>(access, "reference").Where(ir => ir != null).ToList();
            var reference = references.Count == 0 ? null : references;
            // checked before 'run' so that it shows as soon as the tail is chosen, not only after pressing Run
            if (tail == "reference" && reference == null)
            {
                Warn("tail 'reference' needs a reference: connect the irs of Simulate Mono to 'reference', or choose another tail.");
                Status("connect a reference");
                SetResponses(access, scene.Value.Guid);
                return;
            }
            var shOrder = ShOrder;
            var hrtf = Hrtf(access);
            var ismOrder = Value<int>(access, "ism_order");
            var diffraction = Text(access, "diffraction");
            var seed = Value<int>(access, "seed");
            Dictionary<string, object?> settings;
            Dictionary<string, ImpulseResponseData>? refs;
            try
            {
                settings = HybridRun.Settings(tail, ismOrder, diffraction, Text(access, "late_envelope"), seed,
                    Value<bool>(access, "normalize"), reference, scene.SamplingRate, scene.Sources, scene.Receivers, out refs);
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
            if (Value<int>(access, "n_rays") is int rays)
            {
                if (HybridRun.TracesRays(tail, Text(access, "late_envelope"), reference != null))
                    settings["n_rays"] = (long)rays;
                else
                    Remark("n_rays is not used: the method traces no rays itself (an Eyring or Sabine tail, or a given reference without a directional late envelope). Set the reference's rays on Simulate Mono.");
            }
            var given = refs?.ToDictionary(kv => kv.Key, kv => (object?)kv.Value);
            var request = Simulation(scene, Method, settings, given, new Context(scene, hrtf));
            var shown = false;
            void Deliver(Outcome outcome, bool fresh)
            {
                if (fresh)
                    Keep((Context)outcome.Context!, outcome.Result);
                ShowStatus(fresh ? (RunMessage((Context)outcome.Context!), false) : HybridRun.Status(_last, scene.Value.Guid, tail, shOrder, ismOrder, diffraction));
                SetResponses(access, scene.Value.Guid);
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
                    SetResponses(access, scene.Value.Guid);
                }
                if (decision.Interrupted)
                    ShowStatus(("settings changed, press Run", true));
            }
            else if (!shown && decision.Kind != RunKind.Failed)
                SetResponses(access, scene.Value.Guid);
        }

        /// <summary>The message; while the settings differ from the last result, a warning colours the body.</summary>
        private void ShowStatus((string text, bool changed) status)
        {
            Status(status.text);
            if (status.changed)
                Warn("The settings differ from the last run: press Run to update the responses.");
        }

        private sealed record Context(SceneGoo Scene, string? Hrtf);

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
            var irs = PairChoice.InSceneOrder(context.Scene, responses);
            var first = irs.FirstOrDefault()?.Generation ?? new Dictionary<string, object?>();
            _last = new HybridRun.LastResult(context.Scene.Value.Guid, irs, new Dictionary<string, object?>(first));
        }

        private string RunMessage(Context context)
        {
            var generation = _last?.Generation ?? new Dictionary<string, object?>();
            generation.TryGetValue("tail", out var tail);
            // the seed the run used, the method's own when none was given
            generation.TryGetValue("seed", out var seed);
            var order = generation.TryGetValue("sh_order", out var sh) ? $", SH order {MessageText.Show(sh)}" : "";
            var set = TakesHrtf ? $", hrtf {HrtfLabel(context.Hrtf)}" : "";
            return $"{MessageText.Rate(context.Scene.SamplingRate)}, {Method}, tail {tail ?? "auto"}{order}, seed {MessageText.Show(seed)}{set}";
        }
    }

    public sealed class SimulateBinauralComponent : SimulateHybridComponent
    {
        public SimulateBinauralComponent() : base("CA_Simulate_Binaural") { }
        protected override string SpecKey => "CA_Simulate_Binaural";
        public override Guid ComponentGuid => new Guid("2c46c2f3-7b8b-43fd-ba7b-b249681d829b");
        protected override string Method => "binaural";
        protected override string Output => "irs";
        protected override bool TakesHrtf => true;

        // empty is the KU 100; the button chooses the user's own SOFA set, and a File Path wired to hrtf wins
        protected override IEnumerable<Control> MethodButtons => new Control[] { new ButtonControl("hrtf", "HRTF...") };

        public override bool HandlesButton(string port) => port == "hrtf";

        public override void OnButton(string port)
        {
            if (port == "hrtf")
                FileButtons.Choose(this, port, "HRTF set of the binaural responses", "SOFA HRTF set", ".sofa");
        }

        public override string? ButtonLabel(string port) => port == "hrtf" ? FileButtons.Label(this, port, "HRTF: ", "HRTF: KU100") : null;

        protected override void AppendMenu(System.Windows.Forms.ToolStripDropDown menu) => FileButtons.AppendClear(this, menu, "hrtf", "Use KU100");
    }

    public sealed class SimulateAmbisonicsComponent : SimulateHybridComponent
    {
        public SimulateAmbisonicsComponent() : base("CA_Simulate_Ambisonics") { }
        protected override string SpecKey => "CA_Simulate_Ambisonics";
        public override Guid ComponentGuid => new Guid("5da5e5fc-80d6-4a8a-9170-91d3d3a3f94d");
        protected override string Method => "ambisonics";
        protected override string Output => "irs";

        // the component's own choice, not an input: 1 (the method's default) to 7
        protected override IEnumerable<Control> MethodControls => new Control[]
        {
            new DropdownControl("sh_order", new[] { "1", "2", "3", "4", "5", "6", "7" }, "1", Title: "SH order"),
        };

        protected override int? ShOrder => int.Parse(Choice("sh_order"), System.Globalization.CultureInfo.InvariantCulture);
    }
}
