using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using CompasPb;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// The HTTP client of one running worker (compas_acoustics.worker.service): ``POST /api/&lt;function&gt;`` with the
    /// compas_pb payload ``{"args": [...], "kwargs": {...}}`` and the bearer token, answered with ``{"result": ...}`` or
    /// ``{"error": text}``.
    /// </summary>
    public sealed class WorkerClient
    {
        public const string ContentType = "application/x-protobuf";
        private static readonly TimeSpan PingTimeout = TimeSpan.FromSeconds(2);
        internal static readonly CompasPbSerializer Serializer = new CompasPbSerializer();

        private readonly Uri _base;
        private readonly string _token;
        private readonly HttpClient _http;

        public WorkerClient(Uri baseAddress, string token, HttpMessageHandler? handler = null)
        {
            CompasPbRegistration.Ensure();
            _base = baseAddress;
            _token = token;
            // a simulation may take minutes
            _http = new HttpClient(handler ?? new SocketsHttpHandler { UseProxy = false }) { Timeout = Timeout.InfiniteTimeSpan };
        }

        public const string CallIdHeader = "X-Call-Id";

        public object? Call(string function, params object?[] args) => Call(function, args, new Dictionary<string, object?>());

        /// <summary>Call ``function`` of the worker API; its result, as CompasPb unpacks it.</summary>
        /// <param name="callId">The id under which <see cref="Cancel"/> and <see cref="Progress"/> find the call.</param>
        /// <exception cref="WorkerException">The worker answered with an error (status 499: the call was cancelled).</exception>
        /// <exception cref="HttpRequestException">No worker answered.</exception>
        public object? Call(string function, object?[] args, IDictionary<string, object?> kwargs, string? callId = null)
        {
            var payload = new Dictionary<string, object?>
            {
                ["args"] = new List<object?>(args),
                ["kwargs"] = new Dictionary<string, object?>(kwargs),
            };
            using var request = new HttpRequestMessage(HttpMethod.Post, new Uri(_base, "/api/" + function))
            {
                Content = new ByteArrayContent(Serializer.Pack(payload)),
            };
            request.Content.Headers.ContentType = new MediaTypeHeaderValue(ContentType);
            if (callId != null)
                request.Headers.Add(CallIdHeader, callId);
            return Result(Send(request, CancellationToken.None));
        }

        /// <summary>Ask the call sent with ``callId`` to stop at its next stage; false when the worker knows no such call.</summary>
        public bool Cancel(string callId)
        {
            using var cancel = new CancellationTokenSource(PingTimeout);
            using var request = new HttpRequestMessage(HttpMethod.Post, new Uri(_base, "/cancel/" + Uri.EscapeDataString(callId)))
            {
                Content = new ByteArrayContent(Array.Empty<byte>()),
            };
            return Result(Send(request, cancel.Token)) is true;
        }

        /// <summary>The last stage the call ``callId`` reported (``stage``, ``done``, ``of``); null when it is unknown or finished.</summary>
        public IDictionary<string, object?>? Progress(string callId)
        {
            using var cancel = new CancellationTokenSource(PingTimeout);
            using var request = new HttpRequestMessage(HttpMethod.Get, new Uri(_base, "/progress/" + Uri.EscapeDataString(callId)));
            var (status, answer) = Send(request, cancel.Token);
            return status == 404 ? null : Result((status, answer)) as IDictionary<string, object?>;
        }

        private (int status, IDictionary<string, object?>? answer) Send(HttpRequestMessage request, CancellationToken cancel)
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _token);
            using var response = _http.Send(request, cancel);
            var body = response.Content.ReadAsByteArrayAsync(cancel).GetAwaiter().GetResult();
            return ((int)response.StatusCode, body.Length > 0 ? Serializer.Unpack(body) as IDictionary<string, object?> : null);
        }

        private static object? Result((int status, IDictionary<string, object?>? answer) sent)
        {
            var (status, answer) = sent;
            if (status != 200)
                throw WorkerException.FromAnswer(status, answer != null && answer.TryGetValue("error", out var error) && error is string text ? text : $"HTTP {status}");
            if (answer == null || !answer.TryGetValue("result", out var result))
                throw new WorkerException("The worker's answer has no result.", status);
            return result;
        }

        public bool Ping() => PingStatus() == 200;

        /// <summary>The HTTP status ``GET /ping`` answers with; null when nothing answers.</summary>
        public int? PingStatus() => PingAnswer().status;

        /// <summary>What a worker started with ``token`` answers ``GET /ping`` with after ``pong ``.</summary>
        public static string Identity(string token) =>
            Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(token))).Substring(0, 16).ToLowerInvariant();

        /// <summary>The status of ``GET /ping`` and the identity after ``pong``; null when nothing answers, or the answer names none.</summary>
        public (int? status, string? identity) PingAnswer()
        {
            try
            {
                using var cancel = new CancellationTokenSource(PingTimeout);
                using var request = new HttpRequestMessage(HttpMethod.Get, new Uri(_base, "/ping"));
                using var response = _http.Send(request, cancel.Token);
                var body = response.Content.ReadAsStringAsync(cancel.Token).GetAwaiter().GetResult();
                return ((int)response.StatusCode, body.StartsWith("pong ") ? body.Substring(5).Trim() : null);
            }
            catch (Exception e) when (e is HttpRequestException || e is OperationCanceledException || e is System.IO.IOException)
            {
                return (null, null);
            }
        }

        public void Shutdown()
        {
            using var cancel = new CancellationTokenSource(PingTimeout);
            using var request = new HttpRequestMessage(HttpMethod.Post, new Uri(_base, "/shutdown")) { Content = new ByteArrayContent(Array.Empty<byte>()) };
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _token);
            using var response = _http.Send(request, cancel.Token);
        }
    }
}
