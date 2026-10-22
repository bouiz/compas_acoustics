using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CompasAcoustics.Pb;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    public sealed class SimulateHybridComponent : ComponentBase
    {
        private HybridRun.LastResult? _last;

        public SimulateHybridComponent() : base("CA_Simulate_Hybrid") { }
        protected override string SpecKey => "CA_Simulate_Hybrid";
        public override Guid ComponentGuid => new Guid("875b3eae-1993-5cbb-b521-11f4cdcf4e78");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => (input, port.Name) switch
        {
            (true, "scene") => new Param_Scene(),
            (true, "reference") => new Param_IrGraph(),
            (false, _) => new Param_IrGraph(),
            _ => null,
        };

        // no control on hrtf: empty is KEMAR, and a File Path connected there is the user's own SOFA set
        public override IReadOnlyList<Control> Controls { get; } = new Control[]
        {
            new DropdownControl("tail", HybridRun.Tails),
            new DropdownControl("diffraction", HybridRun.Diffraction),
            new SwitchControl("output", HybridRun.Outputs),
            new ButtonControl("run", "Run"),
        };

        private void SetGraphs(IGH_DataAccess access, string sceneGuid, string[] formats)
        {
            var graphs = _last != null && _last.SceneGuid == sceneGuid ? _last.Graphs : new Dictionary<string, IrGraphGoo>();
            access.SetData("binaural_ir_graph", formats.Contains("binaural") && graphs.TryGetValue("binaural", out var b) ? b : null);
            access.SetData("ambisonic_ir_graph", formats.Contains("ambisonic") && graphs.TryGetValue("ambisonic", out var a) ? a : null);
        }

        protected override void Solve(IGH_DataAccess access)
        {
            var scene = Item<SceneGoo>(access, "scene");
            if (scene?.Value == null)
            {
                Warn("Input parameter 'scene' failed to collect data.");
                return;
            }
            var output = Text(access, "output");
            string[] formats;
            try
            {
                formats = HybridRun.Formats(output);
            }
            catch (ArgumentException e)
            {
                Error(e.Message);
                return;
            }
            var tail = Text(access, "tail");
            var reference = Item<IrGraphGoo>(access, "reference");
            // checked before 'run' so that it shows as soon as the tail is chosen, not only after pressing Run
            if (tail == "reference" && reference == null)
            {
                Warn("tail 'reference' needs a reference: connect the mono_ir_graph of Simulate Pyroomacoustics to 'reference', or choose another tail.");
                Message = "connect a reference";
                SetGraphs(access, scene.Value.Guid, formats);
                return;
            }
            var shOrder = Value<int>(access, "sh_order");
            var ismOrder = Value<int>(access, "ism_order");
            var diffraction = Text(access, "diffraction");
            var hrtf = Text(access, "hrtf");
            var seed = Value<int>(access, "seed");
            Dictionary<string, object?> settings;
            Dictionary<string, ImpulseResponseData>? references;
            try
            {
                settings = HybridRun.Settings(tail, shOrder, ismOrder, diffraction, Value<bool>(access, "late_envelope"), seed,
                    Value<bool>(access, "normalize"), output, hrtf, reference, scene.SamplingRate, out references);
            }
            catch (ArgumentException e)
            {
                Error(e.Message);
                return;
            }
            var refs = references?.ToDictionary(kv => kv.Key, kv => (object?)kv.Value);
            var args = new object?[] { scene.Value, "hybrid", settings, refs };
            // a seed drawn for this solution is not a change of the inputs
            var keySettings = new Dictionary<string, object?>(settings);
            if (seed == null)
                keySettings.Remove("seed");
            var key = AsyncRuns.Key("simulate", new object?[] { scene.Value, "hybrid", keySettings, refs }, new Dictionary<string, object?>());
            var request = new Background("simulate", args, new Dictionary<string, object?>(), new Context(scene, formats, settings["seed"], hrtf), key);
            var shown = false;
            void Deliver(Outcome outcome, bool fresh)
            {
                var result = (IDictionary<string, object?>)outcome.Result!;
                ShowWarnings(result);
                if (fresh)
                    Keep((Context)outcome.Context!, result);
                ShowStatus(fresh ? RunMessage((Context)outcome.Context!) : HybridRun.Status(_last, scene.Value.Guid, formats, tail, shOrder, ismOrder, diffraction));
                SetGraphs(access, scene.Value.Guid, formats);
                shown = true;
            }
            // the simulation can take minutes and Grasshopper recomputes on every upstream change, so it starts only when
            // 'run' is true (a button pulse); otherwise the responses of the last run for this scene are passed on
            var pressed = Value<bool>(access, "run") == true || Pressed("run");
            var decision = RunInBackground(access, request, Deliver, start: pressed, restart: pressed);
            if (decision.Kind == RunKind.Idle)
            {
                if (!shown)
                {
                    ShowStatus(HybridRun.Status(_last, scene.Value.Guid, formats, tail, shOrder, ismOrder, diffraction));
                    SetGraphs(access, scene.Value.Guid, formats);
                }
                if (decision.Interrupted)
                    ShowStatus("settings changed, press Run");
            }
            else if (!shown && decision.Kind != RunKind.Failed)
                SetGraphs(access, scene.Value.Guid, formats);
        }

        /// <summary>The message; while the settings differ from the last result, a warning colours the body.</summary>
        private void ShowStatus(string text)
        {
            Status(text);
            if (text.StartsWith("settings changed"))
                Warn("The settings differ from the last run: press Run to update the responses.");
        }

        private sealed record Context(SceneGoo Scene, string[] Formats, object? Seed, string? Hrtf);

        protected override bool RunsInBackground => true;

        protected override async Task<object?> Call(Background request, string callId, CancellationToken cancel)
        {
            await Task.Run(Worker.EnsureStarted, cancel).ConfigureAwait(false);
            if (!Worker.Hybrid)
                throw new WorkerException("The hybrid method needs a worker with pyFDN: run the installer.");
            return await base.Call(request, callId, cancel).ConfigureAwait(false);
        }

        private void Keep(Context context, IDictionary<string, object?> result)
        {
            var responses = (IDictionary<string, object?>)result["responses"]!;
            var graphs = new Dictionary<string, IrGraphGoo>();
            foreach (var format in context.Formats)
                if (responses.TryGetValue(HybridRun.Attributes[format], out var irs) && irs is IDictionary<string, object?> byKey)
                    graphs[format] = IrGraphGoo.FromSimulation(context.Scene, HybridRun.Attributes[format], byKey);
            var first = graphs.Values.SelectMany(g => g.Value.Responses).FirstOrDefault();
            _last = new HybridRun.LastResult(context.Scene.Value.Guid, graphs, new Dictionary<string, object?>(first?.Generation ?? new Dictionary<string, object?>()));
        }

        private string RunMessage(Context context)
        {
            var generation = _last?.Generation ?? new Dictionary<string, object?>();
            generation.TryGetValue("tail", out var resolvedTail);
            generation.TryGetValue("sh_order", out var resolvedOrder);
            return $"{context.Scene.SamplingRate}Hz, hybrid, tail: {resolvedTail ?? "auto"}, order {resolvedOrder}, seed {context.Seed}{HybridRun.HrtfLabel(context.Hrtf, context.Formats)}";
        }
    }
}
