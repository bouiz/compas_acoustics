using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using CompasPb;

namespace CompasAcoustics.Grasshopper.Tests
{
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
