using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class WorkerJobTests
    {
        private double _now;

        private WorkerJob Make(Func<string, CancellationToken, Task<object?>> run, Func<string, IDictionary<string, object?>?>? progress = null) =>
            new(run, progress ?? (_ => null), () => _now);

        [Fact]
        public void Job_CompletesWithTheResultAndElapsed()
        {
            var tcs = new TaskCompletionSource<object?>();
            var job = Make((id, ct) => tcs.Task);
            var completed = new ManualResetEventSlim();
            job.Completed += _ => completed.Set();
            job.Start();
            Assert.Equal(JobState.Running, job.State);
            _now = 3.5;
            tcs.SetResult("answer");
            Assert.True(completed.Wait(2000));
            Assert.Equal(JobState.Done, job.State);
            Assert.Equal("answer", job.Result);
            _now = 10;
            Assert.Equal(3.5, job.Elapsed, 9);
        }

        [Fact]
        public void Job_Cancel_CancelsTheTokenAndEndsCancelled()
        {
            CancellationToken token = default;
            var job = Make((id, ct) => { token = ct; return Task.Delay(Timeout.Infinite, ct).ContinueWith<object?>(t => throw new OperationCanceledException()); });
            job.Start();
            job.Cancel();
            Assert.True(token.IsCancellationRequested);
            Assert.Equal(JobState.Cancelled, job.State);
        }

        [Fact]
        public void Cancel_DiscardsALateResult()
        {
            var tcs = new TaskCompletionSource<object?>();
            var job = Make((id, ct) => tcs.Task);
            var completed = new ManualResetEventSlim();
            job.Completed += _ => completed.Set();
            job.Start();
            job.Cancel();
            tcs.SetResult("late");
            completed.Wait(2000);
            Assert.Equal(JobState.Cancelled, job.State);
            Assert.Null(job.Result);
        }

        [Fact]
        public void Job_FailsWithTheWorkersError()
        {
            var job = Make((id, ct) => Task.FromException<object?>(new WorkerException("ValueError: bad", 400)));
            var completed = new ManualResetEventSlim();
            job.Completed += _ => completed.Set();
            job.Start();
            Assert.True(completed.Wait(2000));
            Assert.Equal(JobState.Failed, job.State);
            Assert.Equal("ValueError: bad", job.Error!.Message);
        }

        [Fact]
        public void Job_PollsProgressAsText()
        {
            var tcs = new TaskCompletionSource<object?>();
            string? asked = null;
            var job = Make((id, ct) => tcs.Task, id =>
            {
                asked = id;
                return new Dictionary<string, object?> { ["stage"] = "ray tracing", ["done"] = 2L, ["of"] = 4L };
            });
            job.Start();
            job.Poll();
            Assert.Equal(job.CallId, asked);
            Assert.Equal("ray tracing 3/4", job.ProgressText);
            Assert.Equal("references", WorkerJob.Describe(new Dictionary<string, object?> { ["stage"] = "references", ["done"] = null, ["of"] = null }));
            Assert.Equal("rendering 6", WorkerJob.Describe(new Dictionary<string, object?> { ["stage"] = "rendering", ["done"] = 5L, ["of"] = null }));
            Assert.Null(WorkerJob.Describe(null));
        }

        [Fact]
        public void RunningText_ShowsMinutesSecondsAndProgress()
        {
            Assert.Equal("running 0:42", AsyncRuns.RunningText(42.4, null));
            Assert.Equal("running 1:12, ray tracing 3/4", AsyncRuns.RunningText(72, "ray tracing 3/4"));
        }
    }

    public class AsyncRunsTests
    {
        private static WorkerJob Job(TaskCompletionSource<object?> tcs) => new((id, ct) => tcs.Task, _ => null, () => 0);

        private static void Finish(WorkerJob job, TaskCompletionSource<object?> tcs, object? result)
        {
            var done = new ManualResetEventSlim();
            job.Completed += _ => done.Set();
            tcs.SetResult(result);
            done.Wait(2000);
        }

        [Fact]
        public void FirstSolve_Starts_ThenKeepsRunningForTheSameInputs()
        {
            var runs = new AsyncRuns();
            Assert.Equal(RunKind.Start, runs.Next(0, "a", true).Kind);
            var tcs = new TaskCompletionSource<object?>();
            var job = Job(tcs);
            runs.Started(0, "a", job);
            job.Start();
            Assert.Equal(RunKind.Running, runs.Next(0, "a", true).Kind);
            Assert.True(runs.AnyRunning);
        }

        [Fact]
        public void AFinishedRun_IsDeliveredAndKept()
        {
            var runs = new AsyncRuns();
            var tcs = new TaskCompletionSource<object?>();
            var job = Job(tcs);
            runs.Next(0, "a", true);
            runs.Started(0, "a", job);
            job.Start();
            Finish(job, tcs, "result");
            var d = runs.Next(0, "a", true);
            Assert.Equal((RunKind.Deliver, "result"), (d.Kind, d.Result));
            // a later solve with the same inputs delivers the kept result without a new run
            Assert.Equal((RunKind.Deliver, "result"), (runs.Next(0, "a", true).Kind, runs.Next(0, "a", true).Result));
            Assert.False(runs.AnyRunning);
        }

        [Fact]
        public void ChangedInputs_CancelTheRunAndStartAgainWithThePreviousResult()
        {
            var runs = new AsyncRuns();
            var first = new TaskCompletionSource<object?>();
            var job = Job(first);
            runs.Next(0, "a", true);
            runs.Started(0, "a", job);
            job.Start();
            Finish(job, first, "old");
            runs.Next(0, "a", true);
            var second = new TaskCompletionSource<object?>();
            var running = Job(second);
            Assert.Equal((RunKind.Start, "old"), (runs.Next(0, "b", true).Kind, runs.Next(0, "b", true).Result));
            runs.Started(0, "b", running);
            running.Start();
            var d = runs.Next(0, "c", true);
            Assert.Equal(JobState.Cancelled, running.State);
            Assert.Equal((RunKind.Start, "old"), (d.Kind, d.Result));
        }

        [Fact]
        public void WithoutLeaveToStart_ChangedInputsAreIdle()
        {
            var runs = new AsyncRuns();
            Assert.Equal(RunKind.Idle, runs.Next(0, "a", false).Kind);
        }

        [Fact]
        public void ACancelledRun_StaysCancelledUntilTheInputsChange()
        {
            var runs = new AsyncRuns();
            var tcs = new TaskCompletionSource<object?>();
            var job = Job(tcs);
            runs.Next(0, "a", true);
            runs.Started(0, "a", job);
            job.Start();
            runs.CancelAll();
            Assert.Equal(RunKind.Cancelled, runs.Next(0, "a", true).Kind);
            Assert.Equal(RunKind.Cancelled, runs.Next(0, "a", true).Kind);
            Assert.Equal(RunKind.Start, runs.Next(0, "b", true).Kind);
        }

        [Fact]
        public void AFailedRun_GivesItsMessageOnce()
        {
            var runs = new AsyncRuns();
            var job = new WorkerJob((id, ct) => Task.FromException<object?>(new WorkerException("ValueError: bad", 400)), _ => null, () => 0);
            var done = new ManualResetEventSlim();
            job.Completed += _ => done.Set();
            runs.Next(0, "a", true);
            runs.Started(0, "a", job);
            job.Start();
            done.Wait(2000);
            var d = runs.Next(0, "a", true);
            Assert.Equal((RunKind.Failed, "ValueError: bad"), (d.Kind, d.Message));
        }

        [Fact]
        public void Hybrid_ChangeWhileRunning_CancelsAndSaysInterrupted()
        {
            var runs = new AsyncRuns();
            var job = Job(new TaskCompletionSource<object?>());
            runs.Next(0, "a", true);
            runs.Started(0, "a", job);
            job.Start();
            var d = runs.Next(0, "b", false);
            Assert.Equal((RunKind.Idle, true), (d.Kind, d.Interrupted));
            Assert.Equal(JobState.Cancelled, job.State);
            Assert.False(runs.Next(0, "b", false).Interrupted);
        }

        [Fact]
        public void Restart_StartsAgainForInputsWithAResult_ButDeliversAFreshResultFirst()
        {
            var runs = new AsyncRuns();
            var tcs = new TaskCompletionSource<object?>();
            var job = Job(tcs);
            runs.Next(0, "a", true, restart: true);
            runs.Started(0, "a", job);
            job.Start();
            Finish(job, tcs, "result");
            // a held Run toggle: the solution the result schedules delivers it instead of starting over
            var d = runs.Next(0, "a", true, restart: true);
            Assert.Equal((RunKind.Deliver, "result", true), (d.Kind, d.Result, d.Fresh));
            Assert.False(runs.Next(0, "a", false).Fresh);
            Assert.Equal((RunKind.Start, "result"), (runs.Next(0, "a", true, restart: true).Kind, runs.Next(0, "a", true, restart: true).Result));
        }

        [Fact]
        public void Iterations_RunApart()
        {
            var runs = new AsyncRuns();
            Assert.Equal(RunKind.Start, runs.Next(0, "a", true).Kind);
            runs.Started(0, "a", Job(new TaskCompletionSource<object?>()));
            Assert.Equal(RunKind.Start, runs.Next(1, "a", true).Kind);
        }

        [Fact]
        public void Key_IsTheSameForTheSameRequestAndDiffersOtherwise()
        {
            var a = AsyncRuns.Key("simulate", new object?[] { "x", 1L }, new Dictionary<string, object?> { ["seed"] = null });
            var b = AsyncRuns.Key("simulate", new object?[] { "x", 1L }, new Dictionary<string, object?> { ["seed"] = null });
            var c = AsyncRuns.Key("simulate", new object?[] { "x", 2L }, new Dictionary<string, object?> { ["seed"] = null });
            Assert.Equal(a, b);
            Assert.NotEqual(a, c);
        }
    }

    /// <summary>Worker.CallAsync: the call id travels, and a cancellation reaches the worker.</summary>
    public class CallAsyncTests : IDisposable
    {
        private readonly string _home = Directory.CreateTempSubdirectory("compas_acoustics_home_").FullName;
        private readonly List<string> _paths = new();
        private readonly List<string?> _ids = new();
        private readonly ManualResetEventSlim _release = new();

        public void Dispose() => Directory.Delete(_home, true);

        private Worker Make()
        {
            var python = Path.Combine(_home, "python");
            File.WriteAllText(python, "");
            return new Worker(new FakeStarter(), (uri, token) => new WorkerClient(uri, token, new FakeHandler((request, body) =>
            {
                var path = request.RequestUri!.AbsolutePath;
                lock (_paths)
                {
                    _paths.Add(path);
                    _ids.Add(request.Headers.TryGetValues("X-Call-Id", out var v) ? v.Single() : null);
                }
                if (path == "/ping") return FakeHandler.Pong();
                if (path == "/api/info")
                    return FakeHandler.Result(new Dictionary<string, object?> { ["compas_acoustics"] = Worker.Version, ["schema"] = Schema.Hash, ["hybrid"] = true });
                if (path.StartsWith("/cancel/"))
                {
                    _release.Set();
                    return FakeHandler.Result(true);
                }
                if (path.StartsWith("/progress/"))
                    return FakeHandler.Result(new Dictionary<string, object?> { ["stage"] = "x", ["done"] = 0L, ["of"] = 2L });
                _release.Wait(5000);
                return _release.IsSet && path == "/api/simulate"
                    ? FakeHandler.Answer((HttpStatusCode)499, new Dictionary<string, object?> { ["error"] = "cancelled" })
                    : FakeHandler.Result("done");
            })), name => name == Worker.PythonVariable ? python : null, _home, TimeSpan.Zero);
        }

        [Fact]
        public void CallAsync_SendsACallId_AndCancelReachesTheWorker()
        {
            var worker = Make();
            using var cts = new CancellationTokenSource();
            var task = worker.CallAsync("simulate", "job-7", new object?[0], new Dictionary<string, object?>(), cts.Token);
            SpinWait.SpinUntil(() => { lock (_paths) return _paths.Contains("/api/simulate"); }, 5000);
            Assert.Equal(new Dictionary<string, object?> { ["stage"] = "x", ["done"] = 0L, ["of"] = 2L }, worker.Progress("job-7"));
            cts.Cancel();
            Assert.Throws<AggregateException>(() => task.Wait(5000));
            Assert.True(task.IsCanceled || task.Exception!.InnerException is OperationCanceledException);
            lock (_paths)
            {
                Assert.Contains("/cancel/job-7", _paths);
                Assert.Equal("job-7", _ids[_paths.IndexOf("/api/simulate")]);
            }
        }
    }
}
