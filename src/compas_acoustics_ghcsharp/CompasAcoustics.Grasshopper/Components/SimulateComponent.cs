using System;
using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// Simulate Mono, Simulate Binaural and Simulate Ambisonics: the method in the worker, started by Run. The responses of
    /// the last run are passed on until the next run, and the message says whether the model or the settings changed since.
    /// </summary>
    public abstract class SimulateComponent : ComponentBase
    {
        /// <summary>The message while the scene differs from the last run's, on two lines to show in full on the body's width.</summary>
        public const string ModelChanged = "model changed,\nrerun the simulation";

        /// <summary>The message while the settings differ from the last run's.</summary>
        public const string SettingsChanged = "settings changed,\nrerun the simulation";

        private readonly Dictionary<int, LastResult> _last = new();

        protected SimulateComponent(string key) : base(key) { }

        /// <summary>The worker's method name, which is also the key of its responses in the worker's result.</summary>
        protected abstract string Method { get; }

        /// <summary>The one output, the method's responses.</summary>
        protected abstract string Output { get; }

        protected override bool RunsInBackground => true;

        /// <summary>The responses of the last run, the scene they were simulated for, the run's description and the key of its call.</summary>
        public sealed record LastResult(SceneGoo Scene, IReadOnlyList<ImpulseResponseGoo> Responses, IReadOnlyDictionary<string, object?> Generation, string? Key = null);

        /// <summary>The responses of the last run, with the sources' current sounds when they hold for <paramref name="scene"/>; none before a run.</summary>
        public static IReadOnlyList<ImpulseResponseGoo> Passed(LastResult? last, SceneGoo scene) =>
            last == null ? Array.Empty<ImpulseResponseGoo>() : SimulatedScene.Current(last.Responses, last.Scene, scene);

        /// <summary>
        /// What the component says while it is not running: how many responses it simulated while they hold, else that there
        /// are none or that the model or the settings (as <paramref name="changed"/> tells) differ from the last run's.
        /// </summary>
        public static (string text, bool changed) Status(LastResult? last, SceneGoo scene, Func<LastResult, bool> changed)
        {
            if (last == null || last.Responses.Count == 0)
                return ("press Run", false);
            if (!SimulatedScene.Same(last.Scene, scene))
                return (ModelChanged, true);
            return changed(last) ? (SettingsChanged, true) : (MessageText.Simulated(last.Responses.Count), false);
        }

        /// <summary>The method's call for <paramref name="scene"/>.</summary>
        protected Background Request(SceneGoo scene, Dictionary<string, object?> settings, IReadOnlyDictionary<string, ImpulseResponseData>? references) =>
            Simulation(scene, Method, settings, references, scene);

        /// <summary>Pass on the responses of the last run.</summary>
        protected void SetResponses(IGH_DataAccess access, SceneGoo scene) =>
            access.SetDataList(Output, Passed(_last.GetValueOrDefault(access.Iteration), scene));

        /// <summary>Say what the last run was and pass on its responses, without starting a run.</summary>
        protected void Hold(IGH_DataAccess access, SceneGoo scene)
        {
            ShowStatus(Status(_last.GetValueOrDefault(access.Iteration), scene, _ => false));
            SetResponses(access, scene);
        }

        /// <summary>
        /// Start <paramref name="request"/> when Run is pressed or 'run' is true; otherwise pass on the responses of the last
        /// run, and say whether the model or the call (every setting and reference) differs from that run's.
        /// </summary>
        protected void Simulate(IGH_DataAccess access, SceneGoo scene, Background request)
        {
            var iteration = access.Iteration;
            bool Changed(LastResult last) => last.Key != request.Key;
            (string text, bool changed)? shown = null;
            void Deliver(Outcome outcome, bool fresh)
            {
                if (fresh)
                    Keep(iteration, (SceneGoo)outcome.Context!, outcome.Result, request.Key);
                shown = fresh ? (MessageText.Simulated(_last[iteration].Responses.Count), false) : Status(_last.GetValueOrDefault(iteration), scene, Changed);
                ShowStatus(shown.Value);
                SetResponses(access, scene);
            }
            // the simulation can take minutes and Grasshopper recomputes on every change of the inputs, so it starts only
            // when 'run' is true (a button pulse); otherwise the responses of the last run are passed on
            var pressed = Value<bool>(access, "run") == true || Pressed("run");
            var decision = RunInBackground(access, request, Deliver, start: pressed, restart: pressed);
            if (decision.Kind == RunKind.Idle)
            {
                if (shown == null)
                {
                    shown = Status(_last.GetValueOrDefault(iteration), scene, Changed);
                    ShowStatus(shown.Value);
                    SetResponses(access, scene);
                }
                // a run cancelled by a change: say so unless the status already says what changed
                if (decision.Interrupted && !shown.Value.changed)
                    ShowStatus((SettingsChanged, true));
            }
            else if (shown == null && decision.Kind != RunKind.Failed)
                SetResponses(access, scene);
        }

        /// <summary>The message; while the model or the settings differ from the last result, a warning colours the body.</summary>
        private void ShowStatus((string text, bool changed) status)
        {
            Status(status.text);
            if (status.changed)
                Warn(status.text == ModelChanged
                    ? "The model changed since the last run: press Run to update the responses."
                    : "The settings differ from the last run: press Run to update the responses.");
        }

        private void Keep(int iteration, SceneGoo scene, IDictionary<string, object?> result, string? key)
        {
            var responses = (IDictionary<string, object?>)((IDictionary<string, object?>)result["responses"]!)[Method]!;
            var irs = PairChoice.InSceneOrder(scene, responses);
            foreach (var ir in irs)
                ir.SimulatedFor = scene;
            var first = irs.FirstOrDefault()?.Generation ?? new Dictionary<string, object?>();
            _last[iteration] = new LastResult(scene, irs, new Dictionary<string, object?>(first), key);
        }
    }
}
