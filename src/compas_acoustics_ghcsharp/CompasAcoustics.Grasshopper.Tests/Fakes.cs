using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using CompasPb;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>The keys of the component specs embedded in the plugin, in ordinal order.</summary>
    internal static class EmbeddedSpecs
    {
        public static IEnumerable<string> Names() =>
            typeof(ComponentSpec).Assembly.GetManifestResourceNames()
                .Where(n => n.StartsWith("Specs/") && n.EndsWith("metadata.json"))
                .Select(n => n.Substring("Specs/".Length).Split('/', '\\')[0])
                .OrderBy(n => n, StringComparer.Ordinal);
    }

    /// <summary>An HttpMessageHandler that answers with a function, for the client without a worker.</summary>
    internal sealed class FakeHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, byte[], HttpResponseMessage> _answer;
        public readonly List<(HttpRequestMessage request, byte[] body)> Seen = new();

        public FakeHandler(Func<HttpRequestMessage, byte[], HttpResponseMessage> answer) => _answer = answer;

        protected override HttpResponseMessage Send(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var body = request.Content?.ReadAsByteArrayAsync(cancellationToken).GetAwaiter().GetResult() ?? Array.Empty<byte>();
            Seen.Add((request, body));
            return _answer(request, body);
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(Send(request, cancellationToken));

        internal static readonly CompasPbSerializer Serializer = new CompasPbSerializer();

        public static HttpResponseMessage Answer(HttpStatusCode status, object payload) =>
            new HttpResponseMessage(status) { Content = new ByteArrayContent(Serializer.Pack(payload)) };

        public static HttpResponseMessage Result(object? result) =>
            Answer(HttpStatusCode.OK, new Dictionary<string, object?> { ["result"] = result });

        public static HttpResponseMessage Pong() => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("pong") };
    }

    /// <summary>The calls a fake worker received, recorded under one lock so a test on another thread sees a call with its payload.</summary>
    internal sealed class WorkerCalls : IEnumerable<string>
    {
        private readonly List<(string path, IDictionary<string, object?>? payload)> _list = new();

        public void Add(string path, byte[] body)
        {
            var payload = body.Length > 0 ? FakeHandler.Serializer.Unpack(body) as IDictionary<string, object?> : null;
            lock (_list)
            {
                _list.Add((path, payload));
                Monitor.PulseAll(_list);
            }
        }

        public (string path, IDictionary<string, object?>? payload) AwaitSingle()
        {
            var deadline = DateTime.UtcNow.AddSeconds(5);
            lock (_list)
            {
                while (_list.Count == 0 && Monitor.Wait(_list, TimeSpan.FromSeconds(Math.Max(0, (deadline - DateTime.UtcNow).TotalSeconds)))) { }
                return _list.Single();
            }
        }

        public IEnumerator<string> GetEnumerator()
        {
            lock (_list)
                return _list.Select(c => c.path).ToList().GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    internal sealed class FakeProcess : IWorkerProcess
    {
        public bool Exited;
        public bool Killed;
        public string Log = "";
        public bool HasExited => Exited || Killed;
        public string Output => Log;
        public string? LogPath { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"fake_worker_{Guid.NewGuid():N}.log");
        public void Kill() => Killed = true;
        public bool WaitForExit(TimeSpan timeout) => HasExited;
    }

    internal sealed class FakeStarter : IProcessStarter
    {
        public readonly List<(string python, IReadOnlyList<string> args, IDictionary<string, string?> env)> Started = new();
        public readonly List<FakeProcess> Processes = new();

        public IWorkerProcess Start(string python, IReadOnlyList<string> args, IDictionary<string, string?> env)
        {
            Started.Add((python, args, env));
            var process = new FakeProcess();
            Processes.Add(process);
            return process;
        }
    }
}
