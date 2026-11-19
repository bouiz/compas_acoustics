using System;
using System.Collections.Generic;
using System.Diagnostics;
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

        /// <summary>A job that calls ``function`` of the worker.</summary>
        public static WorkerJob Of(Worker worker, string function, object?[] args, IDictionary<string, object?> kwargs) =>
            new((id, cancel) => worker.CallAsync(function, id, args, kwargs, cancel), worker.Progress);

        public string CallId { get; } = Guid.NewGuid().ToString("N");

        public JobState State { get; private set; } = JobState.Idle;

        public object? Result { get; private set; }

        public Exception? Error { get; private set; }

        /// <summary>Seconds since the start; frozen when the job ends.</summary>
        public double Elapsed => (_ended ?? _now()) - _started;

        /// <summary>The worker's last report, e.g. ``ray tracing 3/4``; null before the first.</summary>
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

        /// <summary>A report as text: ``stage``, ``stage 3`` or ``stage 3/4`` (``done`` counts the finished items).</summary>
        public static string? Describe(IDictionary<string, object?>? report)
        {
            if (report == null || !report.TryGetValue("stage", out var stage) || stage is not string text)
                return null;
            if (!report.TryGetValue("done", out var done) || done == null)
                return text;
            var current = Convert.ToInt64(done) + 1;
            return report.TryGetValue("of", out var of) && of != null ? $"{text} {current}/{Convert.ToInt64(of)}" : $"{text} {current}";
        }
    }
}
