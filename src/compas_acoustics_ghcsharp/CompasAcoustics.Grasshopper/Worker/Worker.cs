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
    /// free port of 127.0.0.1 with a random token, checked to run the same compas_acoustics and schema as this plugin,
    /// restarted once when it stops answering, and stopped with Rhino (it also exits when Rhino's process is gone).
    /// </summary>
    public sealed class Worker
    {
        public const string PythonVariable = "COMPAS_ACOUSTICS_WORKER_PYTHON";
        public const string TokenVariable = "COMPAS_ACOUSTICS_WORKER_TOKEN";
        public const string ParentVariable = "COMPAS_ACOUSTICS_WORKER_PARENT";
        public const string Service = "compas_acoustics.worker.service";
        /// <summary>Pings, 0.1 s apart, before giving up: the first start of a fresh environment imports uncached.</summary>
        public const int ConnectionAttempts = 1200;

        private static readonly Lazy<Worker> _instance = new(() => new Worker(
            new ProcessStarter(), (uri, token) => new WorkerClient(uri, token), Environment.GetEnvironmentVariable,
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), TimeSpan.FromMilliseconds(100)));

        private readonly IProcessStarter _starter;
        private readonly Func<Uri, string, WorkerClient> _clientFactory;
        private readonly Func<string, string?> _getEnv;
        private readonly string _home;
        private readonly TimeSpan _pingInterval;
        private readonly object _lock = new();
        private WorkerClient? _client;
        private IWorkerProcess? _process;

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

        /// <summary>The plugin's version, which the worker's compas_acoustics must have.</summary>
        public static string Version => typeof(Worker).Assembly.GetName().Version!.ToString(3);

        public int Port { get; private set; }

        public double SpeedOfSound { get; private set; } = 343.0;

        /// <summary>Whether the worker can run the hybrid method (pyFDN).</summary>
        public bool Hybrid { get; private set; }

        public bool Running => _client != null;

        /// <summary>The worker's interpreter: $COMPAS_ACOUSTICS_WORKER_PYTHON, else the one ~/.compas_acoustics/worker.json names; null if neither exists.</summary>
        public static string? ConfiguredPython(Func<string, string?> getEnv, string home)
        {
            var python = getEnv(PythonVariable);
            var config = Path.Combine(home, ".compas_acoustics", "worker.json");
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

        public object? Call(string function, params object?[] args) => Call(function, args, new Dictionary<string, object?>());

        /// <summary>Call ``function`` of compas_acoustics.worker.api, starting the worker first if needed.</summary>
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
                // the worker is gone (Rhino slept, it crashed): start another, once
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

        /// <summary>How many calls of <see cref="CallAsync"/> the worker runs at once; the others wait their turn.</summary>
        public static readonly int Slots = Math.Clamp(Environment.ProcessorCount / 2, 1, 4);

        private readonly SemaphoreSlim _slots = new(Slots);

        /// <summary>
        /// Call ``function`` on a background thread as the call ``callId``. Cancelling ``cancel`` asks the worker to stop the call
        /// at its next stage; the task then ends cancelled.
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
            catch (Exception e) when (e is HttpRequestException || e is OperationCanceledException || e is IOException || e is WorkerException)
            {
            }
        }

        /// <summary>The last stage the call ``callId`` reported; null when it is unknown, finished, or no worker runs.</summary>
        public IDictionary<string, object?>? Progress(string callId)
        {
            try
            {
                return _client?.Progress(callId);
            }
            catch (Exception e) when (e is HttpRequestException || e is OperationCanceledException || e is IOException || e is WorkerException)
            {
                return null;
            }
        }

        /// <summary>Whether the worker is being started (its first call of a session waits for it, about 30 s on a cold start).</summary>
        public bool Starting { get; private set; }

        public T Call<T>(string function, params object?[] args) => (T)Call(function, args)!;

        /// <summary>Start the worker if it does not run, e.g. to read <see cref="Hybrid"/>.</summary>
        public void EnsureStarted()
        {
            lock (_lock)
                _ = _client ?? Start();
        }

        private WorkerClient Start()
        {
            Starting = true;
            StateChanged?.Invoke();
            try
            {
                var client = StartProcess();
                (LastError, StartedAt) = (null, DateTime.Now);
                return client;
            }
            catch (WorkerException e)
            {
                LastError = e.Message;
                throw;
            }
            finally
            {
                Starting = false;
                StateChanged?.Invoke();
            }
        }

        /// <summary>Raised when the worker starts to start, has started or failed to, and has stopped (on the thread that did it).</summary>
        public event Action? StateChanged;

        public WorkerState State =>
            Starting ? WorkerState.Starting : _client != null ? WorkerState.Running : Python == null ? WorkerState.NotConfigured : WorkerState.Stopped;

        /// <summary>The configured interpreter; null when the installer has not run.</summary>
        public string? Python => ConfiguredPython(_getEnv, _home);

        /// <summary>The log of the running worker, or of the last one; null before the first start.</summary>
        public string? LogPath { get; private set; }

        public DateTime? StartedAt { get; private set; }

        /// <summary>Why the last start failed; null after a start that succeeded.</summary>
        public string? LastError { get; private set; }

        /// <summary>Stop the worker and start another.</summary>
        public void Restart()
        {
            lock (_lock)
            {
                StopLocked();
                _ = Start();
            }
        }

        private WorkerClient StartProcess()
        {
            var python = ConfiguredPython(_getEnv, _home) ?? throw new WorkerException("No worker: run the compas_acoustics installer.");
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
            for (var attempt = 0; ; attempt++)
            {
                var status = client.PingStatus();
                if (status == 200)
                    break;
                if (status != null)
                {
                    // something answers, but not this worker's service: an earlier version (compas.rpc's XML-RPC server)
                    process.Kill();
                    throw new WorkerException($"The worker at {python} is not this version's (it answers HTTP {status}): run the installer again.");
                }
                if (process.HasExited)
                    throw new WorkerException($"The worker at {python} exited while starting:\n{process.Output}");
                if (attempt >= ConnectionAttempts)
                {
                    process.Kill();
                    throw new WorkerException($"The worker at {python} did not answer within {ConnectionAttempts / 10} s.");
                }
                if (_pingInterval > TimeSpan.Zero)
                    Thread.Sleep(_pingInterval);
            }
            IDictionary<string, object?> info;
            try
            {
                info = (IDictionary<string, object?>)client.Call("info")!;
            }
            catch (HttpRequestException e)
            {
                var exited = process.HasExited;
                Stop(client, process, port);
                throw new WorkerException($"The worker at {python} answered, then dropped its first call " +
                                          $"({(exited ? "it had exited" : "it was still running")}):\n{process.Output}", details: e.Message);
            }
            catch (Exception)
            {
                Stop(client, process, port);
                throw;
            }
            var version = info.TryGetValue("compas_acoustics", out var v) ? v as string : null;
            var schema = info.TryGetValue("schema", out var s) ? s as string : null;
            if (version != Version || schema != Schema.Hash)
            {
                Stop(client, process, port);
                throw new WorkerException($"The worker at {python} runs compas_acoustics {version ?? "?"} (schema {Short(schema)}), the Grasshopper plugin is " +
                                          $"{Version} (schema {Short(Schema.Hash)}): run the installer again.");
            }
            SpeedOfSound = info.TryGetValue("speed_of_sound", out var c) && c is double speed ? speed : SpeedOfSound;
            Hybrid = info.TryGetValue("hybrid", out var h) && h is bool hybrid && hybrid;
            Port = port;
            _process = process;
            _client = client;
            return client;
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

        /// <summary>Stop the worker, if one runs.</summary>
        public void Stop()
        {
            lock (_lock)
                StopLocked();
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
            catch (Exception e) when (e is HttpRequestException || e is OperationCanceledException || e is IOException)
            {
            }
            if (process != null && !process.WaitForExit(TimeSpan.FromSeconds(2)))
                process.Kill();
            Forget(port);
        }

        /// <summary>Remove the record a killed worker could not remove itself.</summary>
        private void Forget(int port)
        {
            var folder = Path.Combine(_home, ".compas_acoustics", "workers");
            if (!Directory.Exists(folder))
                return;
            foreach (var record in Directory.GetFiles(folder, "*.json"))
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
