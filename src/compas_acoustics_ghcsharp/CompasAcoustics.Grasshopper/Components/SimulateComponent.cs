using System;
using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// Simulate Mono, Simulate Binaural and Simulate Ambisonics: the method in the worker, started by Run. The responses of
    /// the last run are passed on while the scene is the same, and the message says whether the settings changed since.
    /// </summary>
    public abstract class SimulateComponent : ComponentBase
    {
        private readonly Dictionary<int, LastResult> _last = new();

        protected SimulateComponent(string key) : base(key) { }

        /// <summary>The worker's method name, which is also the key of its responses in the worker's result.</summary>
        protected abstract string Method { get; }

        /// <summary>The one output, the method's responses.</summary>
        protected abstract string Output { get; }

        protected override bool RunsInBackground => true;

        /// <summary>The responses of the last run, the scene they were simulated for, the run's description and the key of its call.</summary>
        public sealed record LastResult(SceneGoo Scene, IReadOnlyList<ImpulseResponseGoo> Responses, IReadOnlyDictionary<string, object?> Generation, string? Key = null);

        /// <summary>The responses of the last run when they hold for <paramref name="scene"/>, with its sources' current sounds; else none.</summary>
        public static IReadOnlyList<ImpulseResponseGoo> Passed(LastResult? last, SceneGoo scene) =>
            last != null && SimulatedScene.Same(last.Scene, scene) ? SimulatedScene.Current(last.Responses, last.Scene, scene) : Array.Empty<ImpulseResponseGoo>();

        /// <summary>
        /// What the component says while it is not running: the last result for this scene, or that there is none; and
        /// whether the settings differ from that result's, as <paramref name="changed"/> tells.
        /// </summary>
        public static (string text, bool changed) Status(LastResult? last, SceneGoo scene, Func<LastResult, bool> changed)
        {
            if (last == null || !SimulatedScene.Same(last.Scene, scene) || last.Responses.Count == 0)
                return ("press Run", false);
            var differs = changed(last);
            var summary = MessageText.Run(last.Responses[0].SamplingRate, last.Responses.Count);
            return (differs ? $"settings changed, press Run ({summary})" : $"done, {summary} (Run to redo)", differs);
        }

        /// <summary>The method's call for <paramref name="scene"/>.</summary>
        protected Background Request(SceneGoo scene, Dictionary<string, object?> settings, IReadOnlyDictionary<string, ImpulseResponseData>? references) =>
            Simulation(scene, Method, settings, references, scene);

        /// <summary>Pass on the responses of the last run when they hold for <paramref name="scene"/>, else none.</summary>
        protected void SetResponses(IGH_DataAccess access, SceneGoo scene) =>
            access.SetDataList(Output, Passed(_last.GetValueOrDefault(access.Iteration), scene));

        /// <summary>
        /// Start <paramref name="request"/> when Run is pressed or 'run' is true; otherwise pass on the responses of the last
        /// run for <paramref name="scene"/>, and say whether the settings differ from that run's (<paramref name="changed"/>).
        /// </summary>
        protected void Simulate(IGH_DataAccess access, SceneGoo scene, Background request, Func<LastResult, bool> changed)
        {
            var iteration = access.Iteration;
            var shown = false;
            void Deliver(Outcome outcome, bool fresh)
            {
                if (fresh)
                    Keep(iteration, (SceneGoo)outcome.Context!, outcome.Result, request.Key);
                ShowStatus(fresh ? (MessageText.Run(((SceneGoo)outcome.Context!).SamplingRate, _last[iteration].Responses.Count), false)
                    : Status(_last.GetValueOrDefault(iteration), scene, changed));
                SetResponses(access, scene);
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
                    ShowStatus(Status(_last.GetValueOrDefault(iteration), scene, changed));
                    SetResponses(access, scene);
                }
                if (decision.Interrupted)
                    ShowStatus(("settings changed, press Run", true));
            }
            else if (!shown && decision.Kind != RunKind.Failed)
                SetResponses(access, scene);
        }

        /// <summary>The message; while the settings differ from the last result, a warning colours the body.</summary>
        private void ShowStatus((string text, bool changed) status)
        {
            Status(status.text);
            if (status.changed)
                Warn("The settings differ from the last run: press Run to update the responses.");
        }

        private void Keep(int iteration, SceneGoo scene, IDictionary<string, object?> result, string? key)
        {
            var responses = (IDictionary<string, object?>)((IDictionary<string, object?>)result["responses"]!)[Method]!;
            var irs = PairChoice.InSceneOrder(scene, responses);
            var first = irs.FirstOrDefault()?.Generation ?? new Dictionary<string, object?>();
            _last[iteration] = new LastResult(scene, irs, new Dictionary<string, object?>(first), key);
        }
    }
}
