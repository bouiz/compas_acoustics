using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class WorkerTests : IDisposable
    {
        private readonly string _home = Directory.CreateTempSubdirectory("compas_acoustics_home_").FullName;
        private readonly string _python;
        private readonly FakeStarter _starter = new();
        private string _version = Worker.Version;
        private string _schema = Schema.Hash;
        private int _pingsBeforeUp;
        private int _foreignPongs;
        private int _shutdowns;
        private int _refusals;

        public WorkerTests()
        {
            _python = Path.Combine(_home, "python");
            File.WriteAllText(_python, "");
        }

        public void Dispose() => Directory.Delete(_home, true);

        private HttpResponseMessage Answer(HttpRequestMessage request, byte[] body)
        {
            var path = request.RequestUri!.AbsolutePath;
            if (path == "/ping")
            {
                if (_pingsBeforeUp-- > 0) throw new HttpRequestException("refused");
                if (_foreignPongs-- > 0)
                    return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("pong 0123456789abcdef") };
                return FakeHandler.Pong();
            }
            if (path == "/shutdown")
            {
                _shutdowns++;
                return new HttpResponseMessage(HttpStatusCode.OK);
            }
            if (path == "/api/info")
                return FakeHandler.Result(new Dictionary<string, object?>
                {
                    ["compas_acoustics"] = _version, ["schema"] = _schema, ["speed_of_sound"] = 343.0, ["hybrid"] = true,
                });
            if (_refusals-- > 0) throw new HttpRequestException("Connection refused");
            return FakeHandler.Result("answered");
        }

        private Worker Make(Func<string, string?>? getEnv = null) =>
            new Worker(_starter, (uri, token) => new WorkerClient(uri, token, new FakeHandler(Answer)),
                getEnv ?? (name => name == "COMPAS_ACOUSTICS_WORKER_PYTHON" ? _python : null), _home, TimeSpan.Zero);

        [Fact]
        public void AnotherWorkerOnThePort_StartsThisOneOnAnother()
        {
            _foreignPongs = 1;
            var worker = Make();
            worker.EnsureStarted();
            Assert.Equal(WorkerState.Running, worker.State);
            Assert.Equal(2, _starter.Processes.Count);
            Assert.True(_starter.Processes[0].Killed);
            Assert.False(_starter.Processes[1].Killed);
        }

        [Fact]
        public void AnotherWorkerOnEveryPort_Fails()
        {
            _foreignPongs = 3;
            var error = Assert.Throws<WorkerException>(() => Make().EnsureStarted());
            Assert.Contains("Another worker answered", error.Message);
            Assert.All(_starter.Processes, p => Assert.True(p.Killed));
        }

        [Fact]
        public void TheIdentityIsTheTokensHashAsTheWorkerSendsIt()
        {
            // python -c "import hashlib; print(hashlib.sha256(b'abc').hexdigest()[:16])"
            Assert.Equal("ba7816bf8f01cfea", WorkerClient.Identity("abc"));
        }

        [Fact]
        public void WorkerState_FollowsStartAndStop()
        {
            Assert.Equal(WorkerState.NotConfigured, Make(_ => null).State);
            var worker = Make();
            var changes = new List<WorkerState>();
            worker.StateChanged += () => changes.Add(worker.State);
            Assert.Equal(WorkerState.Stopped, worker.State);
            Assert.Null(worker.StartedAt);
            worker.EnsureStarted();
            Assert.Equal(WorkerState.Running, worker.State);
            Assert.NotNull(worker.StartedAt);
            Assert.Equal(_starter.Processes.Single().LogPath, worker.LogPath);
            worker.Restart();
            Assert.Equal(WorkerState.Running, worker.State);
            Assert.Equal(2, _starter.Processes.Count);
            worker.Stop();
            Assert.Equal(WorkerState.Stopped, worker.State);
            // the log of the last worker stays readable after it stopped
            Assert.Equal(_starter.Processes[1].LogPath, worker.LogPath);
            Assert.Equal(new[] { WorkerState.Running, WorkerState.Stopped, WorkerState.Running, WorkerState.Stopped },
                changes.Where(c => c != WorkerState.Starting).ToArray());
            Assert.Contains(WorkerState.Starting, changes);
        }

        [Fact]
        public void WorkerState_KeepsTheLastStartError()
        {
            _schema = "other";
            var worker = Make();
            Assert.Throws<WorkerException>(() => worker.EnsureStarted());
            Assert.Equal(WorkerState.Stopped, worker.State);
            Assert.Contains("run the installer again", worker.LastError);
        }

        [Fact]
        public void ConfiguredPython_PrefersTheEnvironmentVariable()
        {
            Directory.CreateDirectory(Path.Combine(_home, ".compas_acoustics"));
            File.WriteAllText(Path.Combine(_home, ".compas_acoustics", "worker.json"), "{\"python\": \"/elsewhere\"}");
            Assert.Equal(_python, Worker.ConfiguredPython(name => name == "COMPAS_ACOUSTICS_WORKER_PYTHON" ? _python : null, _home));
        }

        [Fact]
        public void ConfiguredPython_ReadsWorkerJson()
        {
            Directory.CreateDirectory(Path.Combine(_home, ".compas_acoustics"));
            File.WriteAllText(Path.Combine(_home, ".compas_acoustics", "worker.json"), "{\"python\": \"" + _python.Replace("\\", "\\\\") + "\", \"hybrid\": true}");
            Assert.Equal(_python, Worker.ConfiguredPython(name => null, _home));
        }

        [Fact]
        public void ConfiguredPython_IsNullWithoutAConfigOrWithAMissingFile()
        {
            Assert.Null(Worker.ConfiguredPython(name => null, _home));
            Assert.Null(Worker.ConfiguredPython(name => "/no/such/python", _home));
        }

        [Fact]
        public void Start_WithoutConfiguredPython_ThrowsWorkerException()
        {
            var error = Assert.Throws<WorkerException>(() => Make(name => null).Call("make_scene"));
            Assert.Equal("No worker: run the compas_acoustics installer.", error.Message);
            Assert.Empty(_starter.Started);
        }

        [Fact]
        public void Start_PassesTokenParentAndNoPythonPath()
        {
            Environment.SetEnvironmentVariable("PYTHONPATH", "/rhino/site-envs");
            try
            {
                var worker = Make();
                worker.Call("make_scene");
                var (python, args, env) = _starter.Started.Single();
                Assert.Equal(_python, python);
                Assert.Equal(new[] { "-m", "compas_acoustics.worker.service", "--port", worker.Port.ToString() }, args);
                Assert.Matches("^[0-9a-f]{32}$", env["COMPAS_ACOUSTICS_WORKER_TOKEN"]);
                Assert.Equal(Environment.ProcessId.ToString(), env["COMPAS_ACOUSTICS_WORKER_PARENT"]);
                Assert.True(env.ContainsKey("PYTHONPATH") && env["PYTHONPATH"] == null);
                Assert.True(env.ContainsKey("PYTHONHOME") && env["PYTHONHOME"] == null);
                Assert.Equal(343.0, worker.SpeedOfSound);
                Assert.True(worker.Hybrid);
            }
            finally
            {
                Environment.SetEnvironmentVariable("PYTHONPATH", null);
            }
        }

        [Fact]
        public void Start_WaitsForAColdWorker()
        {
            _pingsBeforeUp = 5;
            Assert.Equal("answered", Make().Call("make_scene"));
            Assert.True(Worker.ConnectionAttempts * 0.1 >= 60);
        }

        [Fact]
        public void Start_WhenTheProcessExits_SaysSoWithItsOutput()
        {
            _pingsBeforeUp = int.MaxValue;
            var worker = new Worker(new ExitingStarter(), (uri, token) => new WorkerClient(uri, token, new FakeHandler(Answer)),
                name => name == "COMPAS_ACOUSTICS_WORKER_PYTHON" ? _python : null, _home, TimeSpan.Zero);
            var error = Assert.Throws<WorkerException>(() => worker.Call("make_scene"));
            Assert.Contains("exited while starting", error.Message);
            Assert.Contains("ModuleNotFoundError: compas_acoustics", error.Message);
        }

        private sealed class ExitingStarter : IProcessStarter
        {
            public IWorkerProcess Start(string python, IReadOnlyList<string> args, IDictionary<string, string?> env) =>
                new FakeProcess { Exited = true, Log = "ModuleNotFoundError: compas_acoustics" };
        }

        [Fact]
        public void Start_WithOtherVersionOrSchema_StopsAndNamesBoth()
        {
            _version = "0.0.1";
            var error = Assert.Throws<WorkerException>(() => Make().Call("make_scene"));
            Assert.Contains("0.0.1", error.Message);
            Assert.Contains(Worker.Version, error.Message);
            Assert.Contains("run the installer again", error.Message);
            Assert.Equal(1, _shutdowns);
            _version = Worker.Version;
            _schema = "0000";
            error = Assert.Throws<WorkerException>(() => Make().Call("make_scene"));
            Assert.Contains("0000", error.Message);
        }

        [Fact]
        public void Call_AfterConnectionRefused_RestartsOnce()
        {
            var worker = Make();
            worker.Call("make_scene");
            _refusals = 1;
            Assert.Equal("answered", worker.Call("make_scene"));
            Assert.Equal(2, _starter.Started.Count);
            Assert.True(_starter.Processes[0].Killed || _shutdowns > 0);
        }

        [Fact]
        public void Call_RefusedTwice_Throws()
        {
            var worker = Make();
            worker.Call("make_scene");
            _refusals = 2;
            Assert.Throws<HttpRequestException>(() => worker.Call("make_scene"));
        }

        [Fact]
        public void Stop_ShutsTheWorkerDownAndForgetsItsRecord()
        {
            var worker = Make();
            worker.Call("make_scene");
            var records = Path.Combine(_home, ".compas_acoustics", "workers");
            Directory.CreateDirectory(records);
            File.WriteAllText(Path.Combine(records, "1.json"), "{\"pid\": 1, \"port\": " + worker.Port + "}");
            File.WriteAllText(Path.Combine(records, "2.json"), "{\"pid\": 2, \"port\": 1}");
            worker.Stop();
            Assert.Equal(1, _shutdowns);
            Assert.Equal(new[] { "2.json" }, Directory.GetFiles(records).Select(Path.GetFileName).ToArray());
            worker.Stop();
            Assert.Equal(1, _shutdowns);
        }

        [Fact]
        public void EnsureStarted_StartsOnceAndReadsTheCapabilities()
        {
            var worker = Make();
            worker.EnsureStarted();
            worker.EnsureStarted();
            Assert.Single(_starter.Started);
            Assert.True(worker.Hybrid);
        }

        [Fact]
        public void Start_OfAWorkerThatAnswersButIsNotThisVersions_FailsAtOnceAndStopsIt()
        {
            // an earlier worker (compas.rpc's XML-RPC server) answers the ping with an error status
            var worker = new Worker(_starter, (uri, token) => new WorkerClient(uri, token, new FakeHandler((request, body) =>
                    new HttpResponseMessage(HttpStatusCode.NotImplemented))),
                name => name == "COMPAS_ACOUSTICS_WORKER_PYTHON" ? _python : null, _home, TimeSpan.FromSeconds(10));
            var started = System.Diagnostics.Stopwatch.StartNew();
            var error = Assert.Throws<WorkerException>(() => worker.Call("make_scene"));
            Assert.True(started.Elapsed < TimeSpan.FromSeconds(5));
            Assert.Contains("run the installer again", error.Message);
            Assert.True(_starter.Processes.Single().Killed);
        }

        [Fact]
        public void Start_WhenInfoFails_StopsTheProcess()
        {
            var worker = new Worker(_starter, (uri, token) => new WorkerClient(uri, token, new FakeHandler((request, body) =>
                    request.RequestUri!.AbsolutePath == "/ping" ? FakeHandler.Pong()
                    : FakeHandler.Answer(HttpStatusCode.NotFound, new Dictionary<string, object?> { ["error"] = "info is not part of the API." }))),
                name => name == "COMPAS_ACOUSTICS_WORKER_PYTHON" ? _python : null, _home, TimeSpan.Zero);
            Assert.Throws<WorkerException>(() => worker.Call("make_scene"));
            Assert.True(_starter.Processes.Single().Killed);
            Assert.False(worker.Running);
        }

        [Fact]
        public void Version_IsThePackageVersion()
        {
            Assert.Equal("0.1.0", Worker.Version);
        }
    }
}
