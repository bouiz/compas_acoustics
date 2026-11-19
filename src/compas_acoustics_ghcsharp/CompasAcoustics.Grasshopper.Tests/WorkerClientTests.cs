using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using CompasAcoustics.Pb;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class WorkerClientTests
    {
        private static WorkerClient Client(FakeHandler handler) => new WorkerClient(new Uri("http://127.0.0.1:5000"), "secret", handler);

        [Fact]
        public void Call_SendsAProtobufEnvelopeWithArgsAndTheBearerToken()
        {
            var handler = new FakeHandler((request, body) => FakeHandler.Result(1L));
            Client(handler).Call("make_source", "s", 3L);
            var (request, sent) = handler.Seen.Single();
            Assert.Equal(HttpMethod.Post, request.Method);
            Assert.Equal("/api/make_source", request.RequestUri!.AbsolutePath);
            Assert.Equal("Bearer secret", request.Headers.Authorization!.ToString());
            Assert.Equal("application/x-protobuf", request.Content!.Headers.ContentType!.MediaType);
            var payload = (IDictionary<string, object?>)FakeHandler.Serializer.Unpack(sent)!;
            Assert.Equal(new object?[] { "s", 3L }, ((IList<object?>)payload["args"]!).ToArray());
            Assert.Empty((IDictionary<string, object?>)payload["kwargs"]!);
        }

        [Fact]
        public void Call_SendsKeywordArguments()
        {
            var handler = new FakeHandler((request, body) => FakeHandler.Result(null));
            Client(handler).Call("make_receiver", new object?[] { "r" }, new Dictionary<string, object?> { ["frame"] = null, ["radius"] = 2.0 });
            var payload = (IDictionary<string, object?>)FakeHandler.Serializer.Unpack(handler.Seen.Single().body)!;
            var kwargs = (IDictionary<string, object?>)payload["kwargs"]!;
            Assert.Equal(2.0, kwargs["radius"]);
            Assert.True(kwargs.ContainsKey("frame"));
        }

        [Fact]
        public void Call_ReturnsTheResultOfTheAnswer()
        {
            var source = new SourceData { Name = "s" };
            var handler = new FakeHandler((request, body) => FakeHandler.Result(new Dictionary<string, object?> { ["source"] = source }));
            var result = (IDictionary<string, object?>)Client(handler).Call("x")!;
            Assert.Equal(source, result["source"]);
        }

        [Fact]
        public void Call_ReturnsAnEmptyOrZeroResult()
        {
            var handler = new FakeHandler((request, body) => FakeHandler.Result(0L));
            Assert.Equal(0L, Client(handler).Call("x"));
        }

        [Fact]
        public void Call_ThrowsWorkerExceptionWithTheErrorText()
        {
            var handler = new FakeHandler((request, body) =>
                FakeHandler.Answer(HttpStatusCode.BadRequest, new Dictionary<string, object?> { ["error"] = "ValueError: bad" }));
            var error = Assert.Throws<WorkerException>(() => Client(handler).Call("x"));
            Assert.Equal("ValueError: bad", error.Message);
            Assert.Equal(400, error.Status);
        }

        [Fact]
        public void Call_TakesTheLastLineOfATraceback()
        {
            var traceback = "Traceback (most recent call last):\n  File \"api.py\", line 3, in info\n    raise RuntimeError(\"boom\")\nRuntimeError: boom\n";
            var handler = new FakeHandler((request, body) =>
                FakeHandler.Answer(HttpStatusCode.InternalServerError, new Dictionary<string, object?> { ["error"] = traceback }));
            var error = Assert.Throws<WorkerException>(() => Client(handler).Call("x"));
            Assert.Equal("RuntimeError: boom", error.Message);
            Assert.Equal(traceback, error.Details);
        }

        [Fact]
        public void Ping_IsAGetWithoutTheToken()
        {
            var handler = new FakeHandler((request, body) => FakeHandler.Pong());
            Assert.True(Client(handler).Ping());
            var (request, _) = handler.Seen.Single();
            Assert.Equal(HttpMethod.Get, request.Method);
            Assert.Equal("/ping", request.RequestUri!.AbsolutePath);
            Assert.Null(request.Headers.Authorization);
        }

        [Fact]
        public void Ping_IsFalseWhenNothingAnswers()
        {
            var handler = new FakeHandler((request, body) => throw new HttpRequestException("refused"));
            Assert.False(Client(handler).Ping());
        }

        [Fact]
        public void Shutdown_PostsWithTheToken()
        {
            var handler = new FakeHandler((request, body) => new HttpResponseMessage(HttpStatusCode.OK));
            Client(handler).Shutdown();
            var (request, _) = handler.Seen.Single();
            Assert.Equal("/shutdown", request.RequestUri!.AbsolutePath);
            Assert.Equal("Bearer secret", request.Headers.Authorization!.ToString());
        }
    }
}
