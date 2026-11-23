using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace CompasAcoustics.Grasshopper
{
    public enum WorkerState
    {
        NotConfigured,
        Stopped,
        Starting,
        Running,
    }

    /// <summary>
    /// The compas_acoustics worker of this Rhino session: a Python with compas_acoustics, started on the first call on a
    /// free port of 127.0.0.1 with a random token, checked to run a compas_acoustics this plugin works with and the same
    /// message schema, restarted once when it stops answering, and stopped with Rhino (it also exits when Rhino's process is
    /// gone).
    /// </summary>
    public sealed class Worker
    {
        public const string PythonVariable = "COMPAS_ACOUSTICS_WORKER_PYTHON";
        public const string TokenVariable = "COMPAS_ACOUSTICS_WORKER_TOKEN";
        public const string ParentVariable = "COMPAS_ACOUSTICS_WORKER_PARENT";
        public const string Service = "compas_acoustics.worker.service";

        /// <summary>Pings, one ping interval apart, before giving up: the first start of a fresh environment imports uncached.</summary>
        public const int ConnectionAttempts = 1200;

        /// <summary>
        /// How many calls of <see cref="CallAsync"/> the worker runs at once; the others wait their turn. Half the cores, at
        /// most four: a simulation renders on threads of its own and may hold a few GB.
        /// </summary>
        public static readonly int Slots = Math.Clamp(Environment.ProcessorCount / 2, 1, 4);

        private static readonly Lazy<Worker> _instance = new(() => new Worker(
            new ProcessStarter(), (uri, token) => new WorkerClient(uri, token), Environment.GetEnvironmentVariable,
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), TimeSpan.FromMilliseconds(100)));

        private readonly IProcessStarter _starter;
        private readonly Func<Uri, string, WorkerClient> _clientFactory;
        private readonly Func<string, string?> _getEnv;
        private readonly string _home;
        private readonly TimeSpan _pingInterval;
        private readonly object _lock = new();
        private readonly SemaphoreSlim _slots = new(Slots);
        private WorkerClient? _client;
        private IWorkerProcess? _process;
        private bool _rejected;

        internal Worker(IProcessStarter starter, Func<Uri, string, WorkerClient> clientFactory, Func<string, string?> getEnv, string home,
            TimeSpan pingInterval)
        {
            _starter = starter;
            _clientFactory = clientFactory;
            _getEnv = getEnv;
            _home = home;
            _pingInterval = pingInterval;
        }

        /// <summary>The worker the components use: this session's, or one a test puts in its place.</summary>
        public static Worker Instance => Override ?? _instance.Value;

        /// <summary>A worker the components use instead of this session's (tests).</summary>
        internal static Worker? Override { get; set; }

        /// <summary>Whether <see cref="Instance"/> was ever asked for, so that closing Rhino does not start one.</summary>
        public static bool Created => _instance.IsValueCreated;

        /// <summary>The folder of the worker's environment, its record and the records of running workers, in <paramref name="home"/>.</summary>
        internal static string Folder(string home) => Path.Combine(home, ".compas_acoustics");

        /// <summary>The worker's interpreter: $COMPAS_ACOUSTICS_WORKER_PYTHON, else the one ~/.compas_acoustics/worker.json names; null if neither exists.</summary>
        public static string? ConfiguredPython(Func<string, string?> getEnv, string home)
        {
            var python = getEnv(PythonVariable);
            var config = Path.Combine(Folder(home), "worker.json");
            if (string.IsNullOrEmpty(python) && File.Exists(config))
            {
                try
                {
                    using var json = JsonDocument.Parse(File.ReadAllText(config));
                    if (json.RootElement.TryGetProperty("python", out var value))
                        python = value.GetString();
                }
                catch (JsonException)
                {
                }
            }
            return !string.IsNullOrEmpty(python) && File.Exists(python) ? python : null;
        }

        // the state

        /// <summary>Raised when the worker starts to start, has started or failed to, and has stopped (on the thread that did it).</summary>
        public event Action? StateChanged;

        public WorkerState State =>
            Starting ? WorkerState.Starting : _client != null ? WorkerState.Running : Python == null ? WorkerState.NotConfigured : WorkerState.Stopped;

        /// <summary>Whether the worker is being started (the first call of a session waits for it).</summary>
        public bool Starting { get; private set; }

        /// <summary>The configured interpreter; null when no worker is installed.</summary>
        public string? Python => ConfiguredPython(_getEnv, _home);

        /// <summary>The compas_acoustics version of the running worker; null before it first answered.</summary>
        public string? RunningVersion { get; private set; }

        public int Port { get; private set; }

        public double SpeedOfSound { get; private set; } = 343.0;

        /// <summary>Whether the worker can run the binaural and ambisonics methods (its Python has pyFDN).</summary>
        public bool Hybrid { get; private set; }

        /// <summary>The log of the running worker, or of the last one; null before the first start.</summary>
        public string? LogPath { get; private set; }

        public DateTime? StartedAt { get; private set; }

        /// <summary>Why the last start failed; null after a start that succeeded.</summary>
        public string? LastError { get; private set; }

        /// <summary>Whether the last start failed because the worker's compas_acoustics or message schema does not fit the plugin's.</summary>
        public bool Incompatible { get; private set; }

        // the calls

        public object? Call(string function, params object?[] args) => Call(function, args, new Dictionary<string, object?>());

        /// <summary>Call <paramref name="function"/> of compas_acoustics.worker.api, starting the worker first if needed.</summary>
        public object? Call(string function, object?[] args, IDictionary<string, object?> kwargs) =>
            Call(function, args, kwargs, null, CancellationToken.None);

        private object? Call(string function, object?[] args, IDictionary<string, object?> kwargs, string? callId, CancellationToken cancel)
        {
            WorkerClient client;
            lock (_lock)
                client = _client ?? Start();
            cancel.ThrowIfCancellationRequested();
            try
            {
                return client.Call(function, args, kwargs, callId);
            }
            catch (HttpRequestException)
            {
                // the worker is gone: start another, once
                lock (_lock)
                {
                    if (ReferenceEquals(_client, client))
                        StopLocked();
                    client = _client ?? Start();
                }
                cancel.ThrowIfCancellationRequested();
                return client.Call(function, args, kwargs, callId);
            }
        }

        /// <summary>
        /// Call <paramref name="function"/> on a background thread as the call <paramref name="callId"/>. Cancelling
        /// <paramref name="cancel"/> asks the worker to stop the call at its next stage; the task then ends cancelled.
        /// </summary>
        public Task<object?> CallAsync(string function, string callId, object?[] args, IDictionary<string, object?> kwargs, CancellationToken cancel) =>
            Task.Run(async () =>
            {
                await _slots.WaitAsync(cancel).ConfigureAwait(false);
                try
                {
                    using var registration = cancel.Register(() => CancelCall(callId));
                    try
                    {
                        return Call(function, args, kwargs, callId, cancel);
                    }
                    catch (WorkerException e) when (e.Status == 499 || cancel.IsCancellationRequested)
                    {
                        throw new OperationCanceledException(cancel);
                    }
                }
                finally
                {
                    _slots.Release();
                }
            }, CancellationToken.None);

        private void CancelCall(string callId)
        {
            try
            {
                _client?.Cancel(callId);
            }
            catch (Exception e) when (WorkerClient.IsGone(e) || e is WorkerException)
            {
            }
        }

        /// <summary>The last stage the call <paramref name="callId"/> reported; null when it is unknown, finished, or no worker runs.</summary>
        public IDictionary<string, object?>? Progress(string callId)
        {
            try
            {
                return _client?.Progress(callId);
            }
            catch (Exception e) when (WorkerClient.IsGone(e) || e is WorkerException)
            {
                return null;
            }
        }

        // the start

        /// <summary>Start the worker if it does not run, e.g. to read <see cref="Hybrid"/>.</summary>
        public void EnsureStarted()
        {
            lock (_lock)
                _ = _client ?? Start();
        }

        /// <summary>Stop the worker and start another.</summary>
        public void Restart()
        {
            lock (_lock)
            {
                StopLocked();
                _ = Start();
            }
        }

        private WorkerClient Start()
        {
            Starting = true;
            StateChanged?.Invoke();
            try
            {
                var client = StartProcess();
                (LastError, Incompatible, StartedAt) = (null, false, DateTime.Now);
                return client;
            }
            catch (WorkerException e)
            {
                (LastError, Incompatible, _rejected) = (e.Message, _rejected, false);
                throw;
            }
            finally
            {
                Starting = false;
                StateChanged?.Invoke();
            }
        }

        private WorkerClient StartProcess()
        {
            var python = ConfiguredPython(_getEnv, _home) ?? throw new WorkerException("No worker: use the Install Worker component.");
            // the port is free when probed, but another worker can take it before this one binds it
            for (var port = 0; port < 3; port++)
            {
                var client = StartProcess(python);
                if (client != null)
                    return client;
            }
            throw new WorkerException($"Another worker answered on each of 3 ports chosen for the worker at {python}.");
        }

        /// <summary>A worker of <paramref name="python"/> on a free port; null when another worker answers on that port.</summary>
        private WorkerClient? StartProcess(string python)
        {
            var token = Convert.ToHexString(RandomNumberGenerator.GetBytes(16)).ToLowerInvariant();
            var port = FreePort();
            var env = new Dictionary<string, string?>
            {
                // Rhino's Python paths would load its 3.9 packages into the worker
                ["PYTHONPATH"] = null,
                ["PYTHONHOME"] = null,
                [TokenVariable] = token,
                [ParentVariable] = Environment.ProcessId.ToString(),
            };
            var process = _starter.Start(python, new[] { "-m", Service, "--port", port.ToString() }, env);
            LogPath = process.LogPath ?? LogPath;
            var client = _clientFactory(new Uri($"http://127.0.0.1:{port}"), token);
            if (!AwaitPing(client, process, python, token))
                return null;
            Accept(client, process, python, port);
            return client;
        }

        /// <summary>Ping until the worker answers; false when another worker answers on its port.</summary>
        private bool AwaitPing(WorkerClient client, IWorkerProcess process, string python, string token)
        {
            for (var attempt = 0; ; attempt++)
            {
                var (status, identity) = client.PingAnswer();
                if (status == 200 && identity != null && identity != WorkerClient.Identity(token))
                {
                    process.Kill();
                    return false;
                }
                if (status == 200)
                    return true;
                if (status != null)
                {
                    // something answers, but not a compas_acoustics worker's service
                    process.Kill();
                    throw new WorkerException($"The worker at {python} does not answer as a compas_acoustics worker (it answers HTTP {status}): use Install Worker again.");
                }
                if (process.HasExited)
                    throw new WorkerException($"The worker at {python} exited while starting:\n{process.Output}");
                if (attempt >= ConnectionAttempts)
                {
                    process.Kill();
                    throw new WorkerException($"The worker at {python} did not answer within {ConnectionAttempts * _pingInterval.TotalSeconds:0} s.");
                }
                if (_pingInterval > TimeSpan.Zero)
                    Thread.Sleep(_pingInterval);
            }
        }

        /// <summary>Take the answering worker as this session's once its version and schema fit the plugin's; else stop it.</summary>
        private void Accept(WorkerClient client, IWorkerProcess process, string python, int port)
        {
            IDictionary<string, object?> info;
            try
            {
                info = (IDictionary<string, object?>)client.Call("info")!;
            }
            catch (HttpRequestException)
            {
                var exited = process.HasExited;
                Stop(client, process, port);
                throw new WorkerException($"The worker at {python} answered, then dropped its first call " +
                                          $"({(exited ? "it had exited" : "it was still running")}):\n{process.Output}");
            }
            catch (Exception)
            {
                Stop(client, process, port);
                throw;
            }
            var version = info.TryGetValue("compas_acoustics", out var v) ? v as string : null;
            var schema = info.TryGetValue("schema", out var s) ? s as string : null;
            if (!Versions.Compatible(version) || schema != Schema.Hash)
            {
                Stop(client, process, port);
                _rejected = true;
                throw new WorkerException($"The worker at {python} runs compas_acoustics {version ?? "?"} (schema {Short(schema)}), the Grasshopper plugin {Versions.Plugin} " +
                                          $"needs compas_acoustics {Versions.Requirement} (schema {Short(Schema.Hash)}): use Install Worker again.");
            }
            RunningVersion = version;
            SpeedOfSound = info.TryGetValue("speed_of_sound", out var c) && c is double speed ? speed : SpeedOfSound;
            Hybrid = info.TryGetValue("hybrid", out var h) && h is bool hybrid && hybrid;
            Port = port;
            _process = process;
            _client = client;
        }

        private static string Short(string? hash) => hash == null ? "none" : hash.Length > 12 ? hash.Substring(0, 12) : hash;

        private static int FreePort()
        {
            var probe = new TcpListener(IPAddress.Loopback, 0);
            probe.Start();
            var port = ((IPEndPoint)probe.LocalEndpoint).Port;
            probe.Stop();
            return port;
        }

        // the stop

        /// <summary>Stop the worker, if one runs.</summary>
        public void Stop()
        {
            lock (_lock)
                StopLocked();
        }

        /// <summary>Stop this session's worker and ask every other recorded one to shut down, as Install Worker replaces their files.</summary>
        public void StopAll()
        {
            Stop();
            foreach (var record in Records())
            {
                try
                {
                    using var json = JsonDocument.Parse(File.ReadAllText(record));
                    var entry = json.RootElement;
                    _clientFactory(new Uri($"http://127.0.0.1:{entry.GetProperty("port").GetInt32()}"), entry.GetProperty("token").GetString() ?? "").Shutdown();
                    SpinWait.SpinUntil(() => !File.Exists(record), TimeSpan.FromSeconds(10));
                }
                catch (Exception e) when (WorkerClient.IsGone(e) || e is JsonException || e is KeyNotFoundException || e is InvalidOperationException)
                {
                    // the worker is gone already
                }
                File.Delete(record);
            }
        }

        private void StopLocked()
        {
            if (_client == null)
                return;
            var (client, process, port) = (_client, _process, Port);
            _client = null;
            _process = null;
            StartedAt = null;
            Stop(client, process, port);
            StateChanged?.Invoke();
        }

        private void Stop(WorkerClient client, IWorkerProcess? process, int port)
        {
            try
            {
                client.Shutdown();
            }
            catch (Exception e) when (WorkerClient.IsGone(e))
            {
            }
            if (process != null && !process.WaitForExit(TimeSpan.FromSeconds(2)))
                process.Kill();
            Forget(port);
        }

        /// <summary>The records the running workers keep of their port and token.</summary>
        private string[] Records()
        {
            var folder = Path.Combine(Folder(_home), "workers");
            return Directory.Exists(folder) ? Directory.GetFiles(folder, "*.json") : Array.Empty<string>();
        }

        /// <summary>Remove the record a killed worker could not remove itself.</summary>
        private void Forget(int port)
        {
            foreach (var record in Records())
            {
                try
                {
                    using var json = JsonDocument.Parse(File.ReadAllText(record));
                    if (json.RootElement.TryGetProperty("port", out var value) && value.GetInt32() == port)
                        File.Delete(record);
                }
                catch (Exception e) when (e is IOException || e is JsonException || e is InvalidOperationException || e is UnauthorizedAccessException)
                {
                }
            }
        }
    }
}
