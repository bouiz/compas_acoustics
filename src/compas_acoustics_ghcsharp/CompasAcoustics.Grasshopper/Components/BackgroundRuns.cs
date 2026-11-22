using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;

namespace CompasAcoustics.Grasshopper
{
    public enum RunKind
    {
        /// <summary>Nothing to run: the component waits for a start.</summary>
        Idle,
        /// <summary>Start a job for these inputs; <c>Result</c> is the last result, shown meanwhile.</summary>
        Start,
        /// <summary>The job for these inputs still runs; <c>Result</c> is the last result.</summary>
        Running,
        /// <summary><c>Result</c> is the result for these inputs.</summary>
        Deliver,
        /// <summary>The job for these inputs was cancelled; <c>Result</c> is the last result.</summary>
        Cancelled,
        /// <summary>The job for these inputs failed with <c>Message</c>.</summary>
        Failed,
    }

    /// <summary>
    /// What a solution does with one iteration's request. <c>Fresh</c>: the result arrived since the last solution.
    /// <c>Interrupted</c>: a running job was cancelled because the inputs changed.
    /// </summary>
    public sealed record RunDecision(RunKind Kind, object? Result = null, string? Message = null, bool Fresh = false, bool Interrupted = false);

    /// <summary>
    /// The background jobs of one component, one per iteration, keyed by a hash of the request: the same inputs keep their
    /// job or its result, changed inputs cancel it and start another.
    /// </summary>
    public sealed class BackgroundRuns
    {
        private sealed class Entry
        {
            public string? Key;
            public WorkerJob? Job;
            public string? ResultKey;
            public object? Result;
            public bool Cancelled;
        }

        private readonly Dictionary<int, Entry> _entries = new();
        private readonly object _lock = new();

        public bool AnyRunning => Running.Count > 0;

        public IReadOnlyList<WorkerJob> Running
        {
            get
            {
                lock (_lock)
                    return _entries.Values.Select(e => e.Job).Where(j => j?.State == JobState.Running).Cast<WorkerJob>().ToList();
            }
        }

        /// <param name="start">Whether a request without a result starts a job.</param>
        /// <param name="restart">Start a job even when these inputs have a result or were cancelled.</param>
        public RunDecision Next(int iteration, string key, bool start, bool restart = false)
        {
            lock (_lock)
            {
                if (!_entries.TryGetValue(iteration, out var e))
                    _entries[iteration] = e = new Entry();
                var interrupted = false;
                if (e.Job is { } job)
                {
                    if (job.State == JobState.Running && e.Key == key)
                        return new RunDecision(RunKind.Running, e.Result);
                    e.Job = null;
                    if (job.State == JobState.Running)
                    {
                        job.Cancel();
                        interrupted = true;
                    }
                    else if (e.Key == key)
                    {
                        switch (job.State)
                        {
                            case JobState.Done:
                                (e.ResultKey, e.Result) = (key, job.Result);
                                return new RunDecision(RunKind.Deliver, job.Result, Fresh: true);
                            case JobState.Failed:
                                return new RunDecision(RunKind.Failed, e.Result, ComponentBase.ErrorText(job.Error!));
                            case JobState.Cancelled:
                                e.Cancelled = true;
                                return new RunDecision(RunKind.Cancelled, e.Result);
                        }
                    }
                }
                if (!restart && e.Key == key && e.Cancelled)
                    return new RunDecision(RunKind.Cancelled, e.Result);
                if (!restart && e.ResultKey == key)
                    return new RunDecision(RunKind.Deliver, e.Result);
                if (!start)
                    return new RunDecision(RunKind.Idle, e.Result, Interrupted: interrupted);
                e.Cancelled = false;
                return new RunDecision(RunKind.Start, e.Result);
            }
        }

        /// <summary>The job started for <paramref name="key"/> (after <see cref="Next"/> said Start).</summary>
        public void Started(int iteration, string key, WorkerJob job)
        {
            lock (_lock)
            {
                if (!_entries.TryGetValue(iteration, out var e))
                    _entries[iteration] = e = new Entry();
                (e.Key, e.Job, e.Cancelled) = (key, job, false);
            }
        }

        /// <summary>Cancel every running job; their inputs stay cancelled until they change.</summary>
        public void CancelAll()
        {
            lock (_lock)
                foreach (var e in _entries.Values)
                    if (e.Job?.State == JobState.Running)
                    {
                        e.Job.Cancel();
                        e.Job = null;
                        e.Cancelled = true;
                    }
        }

        /// <summary>A hash of the call <c>function(*args, **kwargs)</c> as it goes to the worker.</summary>
        public static string Key(string function, object?[] args, IDictionary<string, object?> kwargs)
        {
            var payload = new Dictionary<string, object?>
            {
                ["function"] = function,
                ["args"] = new List<object?>(args),
                ["kwargs"] = new Dictionary<string, object?>(kwargs),
            };
            return Convert.ToHexString(SHA256.HashData(WorkerClient.Serializer.Pack(payload)));
        }

        /// <summary><c>running m:ss</c>, and the worker's last report.</summary>
        public static string RunningText(double elapsed, string? progress)
        {
            var seconds = (int)Math.Floor(Math.Max(0, elapsed));
            var text = $"running {seconds / 60}:{seconds % 60:00}";
            return progress == null ? text : $"{text}, {progress}";
        }
    }
}
