using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// Worker calls that run on a background thread: the canvas stays responsive, the last outputs stay while a job runs,
    /// the body shows Cancel, and the message counts the time and shows the worker's stage. The job only stores its result
    /// and schedules a solution; outputs are set inside the solution (Grasshopper's rule).
    /// </summary>
    public abstract partial class ComponentBase
    {
        public const string CancelPort = "cancel";
        private static readonly ButtonControl CancelButton = new(CancelPort, "Cancel");
        private const int TickMs = 1000;

        private AsyncRuns? _runs;
        private Timer? _ticker;

        /// <summary>Whether the component runs its worker calls in the background (its body has room for Cancel).</summary>
        protected virtual bool RunsInBackground => false;

        internal AsyncRuns Runs => _runs ??= new AsyncRuns();

        /// <summary>The controls the body shows: the component's own, and Cancel while a job runs.</summary>
        public IReadOnlyList<Control> BodyControls => _runs?.AnyRunning == true ? Controls.Append(CancelButton).ToList() : Controls;

        /// <summary>
        /// A call to run in the background. ``Context`` travels with the result (e.g. the scene the responses belong to);
        /// ``Key`` replaces the hash of the call when the call holds a value drawn anew in each solution (a random seed).
        /// </summary>
        protected sealed record Background(string Function, object?[] Args, IDictionary<string, object?> Kwargs, object? Context = null, string? Key = null);

        public sealed record Outcome(object? Context, object? Result);

        /// <summary>
        /// Run ``request`` in the background, or keep its job or its result: ``deliver(outcome, fresh)`` sets the outputs of a
        /// result (the last one while a job runs or after a cancel). Starts only when ``start``; ``restart`` starts again for
        /// inputs that have a result.
        /// </summary>
        protected RunDecision RunInBackground(IGH_DataAccess access, Background request, Action<Outcome, bool> deliver, bool start = true, bool restart = false)
        {
            var key = request.Key ?? AsyncRuns.Key(request.Function, request.Args, request.Kwargs);
            var wasRunning = Runs.AnyRunning;
            var decision = Runs.Next(access.Iteration, key, start, restart);
            if (decision.Kind == RunKind.Start)
            {
                var job = new WorkerJob(async (id, cancel) => (object?)new Outcome(request.Context, await Call(request, id, cancel).ConfigureAwait(false)),
                    id => Worker.Progress(id));
                job.Completed += OnJobCompleted;
                Runs.Started(access.Iteration, key, job);
                job.Start();
                StartTicker();
            }
            if (decision.Kind == RunKind.Deliver)
                deliver((Outcome)decision.Result!, decision.Fresh);
            else if (decision.Kind != RunKind.Failed && decision.Result is Outcome previous)
                deliver(previous, false);
            switch (decision.Kind)
            {
                case RunKind.Start:
                case RunKind.Running:
                    Status(RunningMessage());
                    break;
                case RunKind.Cancelled:
                    Message = "cancelled";
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

        /// <summary>The call itself; Simulate Binaural and Simulate Ambisonics first check that the worker has pyFDN.</summary>
        protected virtual Task<object?> Call(Background request, string callId, CancellationToken cancel) =>
            Worker.CallAsync(request.Function, callId, request.Args, request.Kwargs, cancel);

        private string RunningMessage()
        {
            if (Worker.Starting)
                return "starting worker...";
            var jobs = _runs?.Running ?? Array.Empty<WorkerJob>();
            if (jobs.Count == 0)
                return AsyncRuns.RunningText(0, null);
            var longest = jobs.OrderByDescending(j => j.Elapsed).First();
            var text = AsyncRuns.RunningText(longest.Elapsed, jobs.Select(j => j.ProgressText).FirstOrDefault(p => p != null));
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
