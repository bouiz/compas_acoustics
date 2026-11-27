using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;

namespace CompasAcoustics.Grasshopper
{
    public enum JobState
    {
        Idle,
        Running,
        Done,
        Failed,
        Cancelled,
    }

    /// <summary>
    /// One worker call on a background thread: its call id, its state, its result or error, how long it ran, and the stage
    /// the worker last reported. A cancelled job keeps no result, even one that arrives afterwards.
    /// </summary>
    public sealed class WorkerJob
    {
        private readonly Func<string, CancellationToken, Task<object?>> _run;
        private readonly Func<string, IDictionary<string, object?>?> _progress;
        private readonly Func<double> _now;
        private readonly CancellationTokenSource _cancel = new();
        private readonly object _lock = new();
        private double _started;
        private double? _ended;

        public WorkerJob(Func<string, CancellationToken, Task<object?>> run, Func<string, IDictionary<string, object?>?> progress, Func<double>? now = null)
        {
            _run = run;
            _progress = progress;
            _now = now ?? (() => Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency);
        }

        public string CallId { get; } = Guid.NewGuid().ToString("N");

        public JobState State { get; private set; } = JobState.Idle;

        public object? Result { get; private set; }

        public Exception? Error { get; private set; }

        /// <summary>Seconds since the start; frozen when the job ends.</summary>
        public double Elapsed => (_ended ?? _now()) - _started;

        /// <summary>The worker's last report, e.g. <c>ray tracing 3/4</c>; null before the first.</summary>
        public string? ProgressText { get; private set; }

        /// <summary>Raised on a thread-pool thread when the call ends, also after <see cref="Cancel"/>.</summary>
        public event Action<WorkerJob>? Completed;

        public void Start()
        {
            lock (_lock)
            {
                if (State != JobState.Idle)
                    return;
                State = JobState.Running;
                _started = _now();
            }
            Task<object?> task;
            try
            {
                task = _run(CallId, _cancel.Token);
            }
            catch (Exception e)
            {
                task = Task.FromException<object?>(e);
            }
            task.ContinueWith(Finish, TaskScheduler.Default);
        }

        private void Finish(Task<object?> task)
        {
            lock (_lock)
            {
                if (State == JobState.Running)
                {
                    _ended = _now();
                    var error = task.Exception?.InnerException;
                    if (task.IsCanceled || error is OperationCanceledException)
                        State = JobState.Cancelled;
                    else if (error != null)
                        (State, Error) = (JobState.Failed, error);
                    else
                        (State, Result) = (JobState.Done, task.Result);
                }
            }
            Completed?.Invoke(this);
        }

        /// <summary>Stop the job: the worker is asked to stop the call at its next stage, and whatever it answers is dropped.</summary>
        public void Cancel()
        {
            lock (_lock)
            {
                if (State != JobState.Running)
                    return;
                State = JobState.Cancelled;
                _ended = _now();
            }
            _cancel.Cancel();
        }

        /// <summary>Ask the worker where the call is (a timer does, once a second).</summary>
        public void Poll()
        {
            if (State != JobState.Running)
                return;
            var text = Describe(_progress(CallId));
            if (text != null)
                ProgressText = text;
        }

        /// <summary>The hybrid's rendering steps report as <c>simulating: &lt;step&gt;</c>, <c>done</c> the pairs past the step.</summary>
        private const string RenderingStep = "simulating: ";

        /// <summary>Above this many, the counts are an amount (rays), shown as done; at or below it, items, shown as the current one.</summary>
        private const long ItemsAtMost = 100;

        private static readonly IReadOnlyDictionary<string, string> StepLabels = new Dictionary<string, string>
        {
            ["late reverberation"] = "late reverberation (FDN)",
        };

        /// <summary>What a stage's amounts count: rays traced, reflections found (the search reports them with no total).</summary>
        private static readonly IReadOnlyDictionary<string, string> AmountUnits = new Dictionary<string, string>
        {
            ["ray tracing"] = "rays",
            ["image sources"] = "reflections",
        };

        /// <summary>
        /// A report as text: <c>stage</c>, <c>stage 3</c> or <c>stage 3/4</c> for items (<c>done</c> counts the finished ones, so the
        /// current one is <c>done + 1</c>), <c>ray tracing 6,200/20,000 rays</c> for an amount, <c>image sources, 1,234 reflections</c>
        /// for an amount found so far (no total), and
        /// <c>late reverberation (FDN), pair 3/4</c> for a rendering step of the hybrid.
        /// </summary>
        public static string? Describe(IDictionary<string, object?>? report)
        {
            if (report == null || !report.TryGetValue("stage", out var stage) || stage is not string text)
                return null;
            var step = text.StartsWith(RenderingStep, StringComparison.Ordinal) ? text.Substring(RenderingStep.Length) : null;
            var label = step == null ? text : StepLabels.TryGetValue(step, out var named) ? named : step;
            if (!report.TryGetValue("done", out var done) || done == null)
                return label;
            var count = Convert.ToInt64(done);
            long? total = report.TryGetValue("of", out var of) && of != null ? Convert.ToInt64(of) : null;
            var unit = AmountUnits.TryGetValue(text, out var counts) ? counts : null;
            if (total == null && unit != null)
                return string.Format(CultureInfo.InvariantCulture, "{0}, {1:N0} {2}", label, count, unit);
            if (total > ItemsAtMost)
                return string.Format(CultureInfo.InvariantCulture, "{0} {1:N0}/{2:N0}{3}", label, count, total, unit == null ? "" : " " + unit);
            var current = total is long n ? Math.Min(count + 1, n) : count + 1;
            var counted = total is long all ? $"{current}/{all}" : $"{current}";
            return step == null ? $"{label} {counted}" : $"{label}, pair {counted}";
        }
    }
}
