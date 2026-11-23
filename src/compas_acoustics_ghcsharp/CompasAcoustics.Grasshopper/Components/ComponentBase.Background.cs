using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// Worker calls on a background thread: the last outputs stay while a job runs, the body shows Cancel, and the message
    /// counts the time and shows the worker's stage. A job only stores its result and schedules a solution, in which the
    /// outputs are set (Grasshopper's rule).
    /// </summary>
    public abstract partial class ComponentBase
    {
        public const string CancelPort = "cancel";
        private static readonly ButtonControl CancelButton = new(CancelPort, "Cancel");
        private const int TickMs = 1000;

        private BackgroundRuns? _runs;
        private Timer? _ticker;

        /// <summary>Whether the component runs its worker calls in the background (its body has room for Cancel).</summary>
        protected virtual bool RunsInBackground => false;

        internal BackgroundRuns Runs => _runs ??= new BackgroundRuns();

        /// <summary>The controls the body shows: the component's own, and Cancel while a job runs.</summary>
        public IReadOnlyList<Control> BodyControls => _runs?.AnyRunning == true ? Controls.Append(CancelButton).ToList() : Controls;

        /// <summary>
        /// A call to run in the background. <c>Context</c> travels with the result; <c>Key</c> replaces the hash of the call
        /// when the call holds a value drawn anew in each solution.
        /// </summary>
        protected sealed record Background(string Function, object?[] Args, IDictionary<string, object?> Kwargs, object? Context = null, string? Key = null);

        /// <summary>A result of the worker, with the context of its request.</summary>
        public sealed record Outcome(object? Context, IDictionary<string, object?> Result);

        /// <summary>
        /// Run <paramref name="request"/> in the background, or keep its job or its result: after the result's warnings,
        /// <c>deliver(outcome, fresh)</c> sets the outputs of a result (the last one while a job runs or after a cancel).
        /// Starts only when <paramref name="start"/>; <paramref name="restart"/> starts again for inputs that have a result.
        /// </summary>
        protected RunDecision RunInBackground(IGH_DataAccess access, Background request, Action<Outcome, bool> deliver, bool start = true, bool restart = false)
        {
            var key = request.Key ?? BackgroundRuns.Key(request.Function, request.Args, request.Kwargs);
            var wasRunning = Runs.AnyRunning;
            var decision = Runs.Next(access.Iteration, key, start, restart);
            if (decision.Kind == RunKind.Start)
            {
                var job = new WorkerJob(async (id, cancel) =>
                        (object?)new Outcome(request.Context, (IDictionary<string, object?>)(await Call(request, id, cancel).ConfigureAwait(false))!),
                    id => Worker.Progress(id));
                job.Completed += OnJobCompleted;
                Runs.Started(access.Iteration, key, job);
                job.Start();
                StartTicker();
            }
            var delivered = decision.Kind == RunKind.Deliver ? (Outcome)decision.Result!
                : decision.Kind != RunKind.Failed ? decision.Result as Outcome : null;
            if (delivered != null)
            {
                ShowWarnings(delivered.Result);
                deliver(delivered, decision.Kind == RunKind.Deliver && decision.Fresh);
            }
            switch (decision.Kind)
            {
                case RunKind.Start:
                case RunKind.Running:
                    Status(RunningMessage());
                    break;
                case RunKind.Cancelled:
                    Status("cancelled");
                    break;
                case RunKind.Failed:
                    Message = null;
                    Error(decision.Message!);
                    break;
            }
            if (Runs.AnyRunning != wasRunning)
                Attributes?.ExpireLayout();
            return decision;
        }

        /// <summary>
        /// The worker's <c>simulate</c> of <paramref name="method"/>. A seed drawn for this solution is left out of the key:
        /// it is not a change of the inputs.
        /// </summary>
        protected static Background Simulation(SceneGoo scene, string method, Dictionary<string, object?> settings, object? references,
            bool seedGiven, object? context)
        {
            var keySettings = new Dictionary<string, object?>(settings);
            if (!seedGiven)
                keySettings.Remove("seed");
            var none = new Dictionary<string, object?>();
            return new Background("simulate", new object?[] { scene.Value, method, settings, references }, none, context,
                BackgroundRuns.Key("simulate", new object?[] { scene.Value, method, keySettings, references }, none));
        }

        /// <summary>The call itself; a component may check the worker first.</summary>
        protected virtual Task<object?> Call(Background request, string callId, CancellationToken cancel) =>
            Worker.CallAsync(request.Function, callId, request.Args, request.Kwargs, cancel);

        private string RunningMessage()
        {
            if (Worker.Starting)
                return "starting worker...";
            var jobs = _runs?.Running ?? Array.Empty<WorkerJob>();
            if (jobs.Count == 0)
                return BackgroundRuns.RunningText(0, null);
            var longest = jobs.OrderByDescending(j => j.Elapsed).First();
            var text = BackgroundRuns.RunningText(longest.Elapsed, jobs.Select(j => j.ProgressText).FirstOrDefault(p => p != null));
            return jobs.Count > 1 ? $"{text} ({jobs.Count} runs)" : text;
        }

        private void OnJobCompleted(WorkerJob job)
        {
            // a cancel already solved the component again
            if (job.State == JobState.Cancelled)
                return;
            Rhino.RhinoApp.InvokeOnUiThread(new Action(() => OnPingDocument()?.ScheduleSolution(1, _ => ExpireSolution(false))));
        }

        private void StartTicker()
        {
            lock (Runs)
                _ticker ??= new Timer(Tick, null, TickMs, Timeout.Infinite);
        }

        /// <summary>Once a second while a job runs: ask the worker for each job's stage, then repaint the message (no solution).</summary>
        private void Tick(object? state)
        {
            var jobs = _runs?.Running ?? Array.Empty<WorkerJob>();
            foreach (var job in jobs)
                job.Poll();
            lock (Runs)
            {
                if (jobs.Count == 0)
                {
                    _ticker?.Dispose();
                    _ticker = null;
                    return;
                }
                _ticker?.Change(TickMs, Timeout.Infinite);
            }
            Rhino.RhinoApp.InvokeOnUiThread(new Action(() =>
            {
                if (_runs?.AnyRunning != true)
                    return;
                Status(RunningMessage());
                Attributes?.ExpireLayout();
                global::Grasshopper.Instances.ActiveCanvas?.Invalidate();
            }));
        }

        public override void RemovedFromDocument(GH_Document document)
        {
            _runs?.CancelAll();
            base.RemovedFromDocument(document);
        }

        public override void DocumentContextChanged(GH_Document document, GH_DocumentContext context)
        {
            if (context == GH_DocumentContext.Close || context == GH_DocumentContext.Unloaded)
                _runs?.CancelAll();
            base.DocumentContextChanged(document, context);
        }
    }
}
