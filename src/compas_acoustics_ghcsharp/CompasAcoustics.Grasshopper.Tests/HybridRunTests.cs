using System;
using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class HybridRunTests
    {
        private static ImpulseResponseGoo Response(string source, string receiver, int channels = 1, int rate = 48000) =>
            new ImpulseResponseGoo(new ImpulseResponseData
            {
                Source = new SourceData { Guid = source }, Receiver = new ReceiverData { Guid = receiver },
                Shape = { 4, channels }, SamplingRate = rate, Samples = { new double[4 * channels] },
            });

        private static List<ImpulseResponseGoo> Reference(params ImpulseResponseGoo[] responses) => responses.ToList();

        private static SourceGoo Source(string guid, string? name = null) => new(new SourceData { Guid = guid, Name = name ?? guid });

        private static ReceiverGoo Receiver(string guid, string? name = null) => new(new ReceiverData { Guid = guid, Name = name ?? guid });

        private static readonly IReadOnlyList<SourceGoo> NoSources = new List<SourceGoo>();
        private static readonly IReadOnlyList<ReceiverGoo> NoReceivers = new List<ReceiverGoo>();

        [Fact]
        public void Settings_LeavesUnsetInputsOut_TheSeedIncluded()
        {
            var settings = HybridRun.Settings(null, null, null, null, null, null, 48000, NoSources, NoReceivers, out var references);
            Assert.Empty(settings);
            Assert.Null(references);
            settings = HybridRun.Settings("auto", null, "none", null, 3, null, 48000, NoSources, NoReceivers, out _);
            Assert.Equal(new Dictionary<string, object?> { ["seed"] = 3L }, settings);
        }

        [Fact]
        public void Settings_PassesTheEngineInputsUnderTheMethodsNames_AndNoFormatOfTheirOwn()
        {
            var settings = HybridRun.Settings("eyring", 4, "udfa", "off", 7, null, 48000, NoSources, NoReceivers, out _);
            Assert.Equal(new Dictionary<string, object?>
            {
                ["tail"] = "eyring", ["image_source_order"] = 4L, ["diffraction"] = "udfa", ["late_envelope"] = false, ["seed"] = 7L,
            }, settings);
        }

        [Fact]
        public void Settings_SendNoNormalize()
        {
            var settings = HybridRun.Settings("eyring", 4, "udfa", "off", 7, null, 48000, NoSources, NoReceivers, out _);
            Assert.False(settings.ContainsKey("normalize"));
        }

        [Theory]
        [InlineData("omni", "omni")]
        [InlineData("directional", "directional")]
        [InlineData("off", false)]
        [InlineData("True", "directional")]
        [InlineData("true", "directional")]
        [InlineData("False", false)]
        public void Settings_PassesTheLateEnvelopeMode_AndReadsABoolean(string choice, object expected)
        {
            var settings = HybridRun.Settings(null, null, null, choice, 1, null, 48000, NoSources, NoReceivers, out _);
            Assert.Equal(expected, settings["late_envelope"]);
        }

        [Theory]
        [InlineData(null)]
        [InlineData("auto")]
        [InlineData("")]
        public void Settings_LeavesAnAutoLateEnvelopeToTheMethod(string? choice)
        {
            var settings = HybridRun.Settings(null, null, null, choice, 1, null, 48000, NoSources, NoReceivers, out _);
            Assert.False(settings.ContainsKey("late_envelope"));
        }

        [Fact]
        public void Settings_TakesTheMonoResponsesOfTheReferenceByKey()
        {
            var reference = Reference(Response("s0", "r0"), Response("s1", "r0", channels: 4));
            HybridRun.Settings("reference", null, null, null, 1, reference, 48000, NoSources, NoReceivers, out var references);
            Assert.Equal(new[] { "r0_s0" }, references!.Keys.ToArray());
        }

        [Fact]
        public void Settings_ReferenceIsAListOfMonoResponses()
        {
            var reference = Reference(Response("s0", "r0"), Response("s1", "r0"), Response("s2", "r0", channels: 2));
            HybridRun.Settings("reference", null, null, null, 1, reference, 48000,
                new[] { Source("s0"), Source("s1") }, new[] { Receiver("r0") }, out var references);
            Assert.Equal(new[] { "r0_s0", "r0_s1" }, references!.Keys.OrderBy(k => k).ToArray());
        }

        [Fact]
        public void Settings_ReferenceMissingAPair_Throws()
        {
            var reference = Reference(Response("a", "rec"));
            var error = Assert.Throws<ArgumentException>(() => HybridRun.Settings("reference", null, null, null, 1, reference, 48000,
                new[] { Source("a"), Source("b") }, new[] { Receiver("rec") }, out _));
            Assert.Contains("no reference for source 'b' and receiver 'rec'", error.Message);
            Assert.DoesNotContain("source 'a'", error.Message);
            Assert.Contains("connect the irs of Simulate Mono", error.Message);
        }

        [Fact]
        public void Settings_ReferenceMissingTwoPairs_NamesEach()
        {
            var error = Assert.Throws<ArgumentException>(() => HybridRun.Settings("reference", null, null, null, 1,
                Reference(Response("a", "r1")), 48000,
                new[] { Source("a"), Source("b") }, new[] { Receiver("r1"), Receiver("r2") }, out _));
            Assert.Contains("source 'b' and receiver 'r1'", error.Message);
            Assert.Contains("source 'a' and receiver 'r2'", error.Message);
            Assert.Contains("source 'b' and receiver 'r2'", error.Message);
        }

        [Fact]
        public void Settings_NoMonoResponse_Throws()
        {
            var error = Assert.Throws<ArgumentException>(() =>
                HybridRun.Settings(null, null, null, null, 1, Reference(Response("s", "r", channels: 2)), 48000, NoSources, NoReceivers, out _));
            Assert.Equal("The connected reference holds no mono responses; connect the irs output of Simulate Mono.", error.Message);
        }

        [Fact]
        public void Settings_RejectsAReferenceAtAnotherRate()
        {
            var error = Assert.Throws<ArgumentException>(() =>
                HybridRun.Settings(null, null, null, null, 1, Reference(Response("s", "r", rate: 44100)), 48000, NoSources, NoReceivers, out _));
            Assert.Equal("The reference was simulated at 44100 Hz, the scene's sampling_rate is 48000 Hz: simulate both at the same rate.", error.Message);
        }

        [Theory]
        [InlineData(null, null, false, true)]
        [InlineData("ray_tracing", "omni", false, true)]
        [InlineData(null, "directional", true, true)]
        [InlineData("reference", "directional", true, true)]
        [InlineData(null, null, true, false)]
        [InlineData("reference", "omni", true, false)]
        [InlineData(null, "off", true, false)]
        [InlineData("eyring", null, false, false)]
        [InlineData("sabine", null, false, false)]
        public void NRays_GoesToTheMethodOnlyWhenItTracesRaysItself(string? tail, string? lateEnvelope, bool referenced, bool used) =>
            Assert.Equal(used, HybridRun.TracesRays(tail, lateEnvelope, referenced));

        [Fact]
        public void TheHybridComponents_TakeTheRaysTheyTrace()
        {
            foreach (var key in new[] { "CA_Simulate_Binaural", "CA_Simulate_Ambisonics" })
            {
                var names = ComponentSpec.Load(key).Inputs.Select(p => p.Name).ToList();
                Assert.Equal(names.IndexOf("late_envelope") + 1, names.IndexOf("n_rays"));
            }
        }

        private static HybridRun.LastResult Last(string sceneGuid, int count = 1, string tail = "reference", long? shOrder = 1, long ismOrder = 3,
            object? diffraction = null)
        {
            var generation = new Dictionary<string, object?> { ["tail"] = tail, ["image_source_order"] = ismOrder, ["diffraction"] = diffraction };
            if (shOrder != null)
                generation["sh_order"] = shOrder;
            return new HybridRun.LastResult(sceneGuid, Reference(Enumerable.Range(0, count).Select(i => Response($"s{i}", "r")).ToArray()), generation);
        }

        [Fact]
        public void Status_WithoutLastResult_SaysPressRun()
        {
            Assert.Equal(("press Run", false), HybridRun.Status(null, "scene", null, null, null, null));
        }

        [Fact]
        public void Status_ForAnotherScene_SaysPressRun()
        {
            Assert.Equal(("press Run", false), HybridRun.Status(Last("old"), "new", null, null, null, null));
        }

        [Fact]
        public void Status_SaysDoneWithTheRateAndCount()
        {
            Assert.Equal(("done, 48 kHz, 2 responses (Run to redo)", false), HybridRun.Status(Last("scene", count: 2), "scene", null, null, null, null));
            Assert.Equal(("done, 48 kHz, 1 response (Run to redo)", false), HybridRun.Status(Last("scene"), "scene", "auto", 1, 3, "none"));
        }

        [Fact]
        public void Status_SaysSettingsChanged()
        {
            var last = Last("scene");
            var changed = ("settings changed, press Run (48 kHz, 1 response)", true);
            Assert.Equal(changed, HybridRun.Status(last, "scene", null, null, 4, null));
            Assert.Equal(changed, HybridRun.Status(last, "scene", "eyring", null, null, null));
            Assert.Equal(changed, HybridRun.Status(last, "scene", null, 2, null, null));
            Assert.Equal(changed, HybridRun.Status(last, "scene", null, null, null, "udfa"));
            Assert.Equal(("done, 48 kHz, 1 response (Run to redo)", false), HybridRun.Status(Last("scene", diffraction: "udfa"), "scene", null, null, null, "udfa"));
        }
    }
}
