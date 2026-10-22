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

        private static IrGraphGoo Reference(params ImpulseResponseGoo[] responses) =>
            new IrGraphGoo(new IrGraphGoo.Content("pyroomacoustics", new List<SourceGoo>(), new List<ReceiverGoo>(), responses));

        [Fact]
        public void Formats_DefaultIsBinaural_BothIsAmbisonicThenBinaural()
        {
            Assert.Equal(new[] { "binaural" }, HybridRun.Formats(null));
            Assert.Equal(new[] { "binaural" }, HybridRun.Formats(""));
            Assert.Equal(new[] { "ambisonic" }, HybridRun.Formats("ambisonic"));
            Assert.Equal(new[] { "ambisonic", "binaural" }, HybridRun.Formats("both"));
            var error = Assert.Throws<ArgumentException>(() => HybridRun.Formats("stereo"));
            Assert.Equal("output must be one of binaural, ambisonic, both, not 'stereo'.", error.Message);
        }

        [Fact]
        public void Settings_LeavesUnsetInputsOutAndDrawsASeed()
        {
            var settings = HybridRun.Settings(null, null, null, null, null, null, null, null, null, null, 48000, out var references);
            Assert.Equal(new[] { "seed" }, settings.Keys.ToArray());
            Assert.IsType<long>(settings["seed"]);
            Assert.Null(references);
            settings = HybridRun.Settings("auto", null, null, "none", null, 3, null, null, null, null, 48000, out _);
            Assert.Equal(new Dictionary<string, object?> { ["seed"] = 3L }, settings);
        }

        [Fact]
        public void Settings_PassesGivenInputsUnderTheMethodsNames()
        {
            var settings = HybridRun.Settings("eyring", 2, 4, "udfa", false, 7, false, "both", null, null, 48000, out _);
            Assert.Equal("eyring", settings["tail"]);
            Assert.Equal(2L, settings["sh_order"]);
            Assert.Equal(4L, settings["image_source_order"]);
            Assert.Equal("udfa", settings["diffraction"]);
            Assert.Equal(false, settings["late_envelope"]);
            Assert.Equal(false, settings["normalize"]);
            Assert.Equal(new object?[] { "ambisonic", "binaural" }, ((IEnumerable<object?>)settings["output"]!).ToArray());
            Assert.Equal("binaural", HybridRun.Settings(null, null, null, null, null, 1, null, "binaural", null, null, 48000, out _)["output"]);
        }

        [Fact]
        public void Settings_PassesHrtfOnlyWithABinauralOutput()
        {
            Assert.Equal("/h.sofa", HybridRun.Settings(null, null, null, null, null, 1, null, null, "/h.sofa", null, 48000, out _)["hrtf"]);
            Assert.False(HybridRun.Settings(null, null, null, null, null, 1, null, "ambisonic", "/h.sofa", null, 48000, out _).ContainsKey("hrtf"));
        }

        [Fact]
        public void Settings_TakesTheMonoResponsesOfTheReferenceByKey()
        {
            var reference = Reference(Response("s0", "r0"), Response("s1", "r0", channels: 4));
            HybridRun.Settings("reference", null, null, null, null, 1, null, null, null, reference, 48000, out var references);
            Assert.Equal(new[] { "r0_s0" }, references!.Keys.ToArray());
        }

        [Fact]
        public void Settings_RejectsAReferenceWithoutMonoResponses()
        {
            var error = Assert.Throws<ArgumentException>(() =>
                HybridRun.Settings(null, null, null, null, null, 1, null, null, null, Reference(Response("s", "r", channels: 2)), 48000, out _));
            Assert.Equal("The connected ir_graph holds no mono responses; connect the mono_ir_graph output of Simulate Pyroomacoustics.", error.Message);
        }

        [Fact]
        public void Settings_RejectsAReferenceAtAnotherRate()
        {
            var error = Assert.Throws<ArgumentException>(() =>
                HybridRun.Settings(null, null, null, null, null, 1, null, null, null, Reference(Response("s", "r", rate: 44100)), 48000, out _));
            Assert.Equal("The reference was simulated at 44100 Hz, the scene's sampling_rate is 48000 Hz: simulate both at the same rate.", error.Message);
        }

        private static HybridRun.LastResult Last(string sceneGuid, string[] formats, int count = 1, string tail = "reference", long shOrder = 1, long ismOrder = 3,
            object? diffraction = null)
        {
            var generation = new Dictionary<string, object?>
            {
                ["tail"] = tail, ["sh_order"] = shOrder, ["image_source_order"] = ismOrder, ["diffraction"] = diffraction,
            };
            var graphs = formats.ToDictionary(f => f, f => Reference(Enumerable.Range(0, count).Select(i => Response($"s{i}", "r")).ToArray()));
            return new HybridRun.LastResult(sceneGuid, graphs, generation);
        }

        [Fact]
        public void Status_WithoutLastResult_SaysPressRun()
        {
            Assert.Equal("press Run", HybridRun.Status(null, "scene", new[] { "binaural" }, null, null, null, null));
        }

        [Fact]
        public void Status_ForAnotherScene_SaysPressRun()
        {
            Assert.Equal("press Run", HybridRun.Status(Last("old", new[] { "binaural" }), "new", new[] { "binaural" }, null, null, null, null));
        }

        [Fact]
        public void Status_SaysDoneWithTheCountAndSummary()
        {
            var last = Last("scene", new[] { "binaural", "ambisonic" });
            Assert.Equal("done, 2 IRs, tail: reference, order 1 (Run to redo)",
                HybridRun.Status(last, "scene", new[] { "ambisonic", "binaural" }, null, null, null, null));
            Assert.Equal("done, 1 IR, tail: reference, order 1 (Run to redo)",
                HybridRun.Status(last, "scene", new[] { "binaural" }, "auto", 1, 3, "none"));
        }

        [Fact]
        public void Status_SaysSettingsChanged()
        {
            var last = Last("scene", new[] { "binaural" });
            const string changed = "settings changed, press Run (tail: reference, order 1)";
            Assert.Equal(changed, HybridRun.Status(last, "scene", new[] { "binaural" }, null, null, 4, null));
            Assert.Equal(changed, HybridRun.Status(last, "scene", new[] { "binaural" }, "eyring", null, null, null));
            Assert.Equal(changed, HybridRun.Status(last, "scene", new[] { "binaural" }, null, 2, null, null));
            Assert.Equal(changed, HybridRun.Status(last, "scene", new[] { "binaural" }, null, null, null, "udfa"));
            Assert.Equal("done, 1 IR, tail: reference, order 1 (Run to redo)",
                HybridRun.Status(Last("scene", new[] { "binaural" }, diffraction: "udfa"), "scene", new[] { "binaural" }, null, null, null, "udfa"));
        }

        [Fact]
        public void Status_SaysWhenARequestedFormatHasNoResult()
        {
            Assert.Equal("settings changed, press Run (tail: reference, order 1)",
                HybridRun.Status(Last("scene", new[] { "binaural" }), "scene", new[] { "ambisonic", "binaural" }, null, null, null, null));
        }

        [Fact]
        public void HrtfLabel_NamesTheSetOfABinauralOutput()
        {
            Assert.Equal(", hrtf kemar", HybridRun.HrtfLabel(null, new[] { "binaural" }));
            Assert.Equal(", hrtf mine.sofa", HybridRun.HrtfLabel("/data/mine.sofa", new[] { "ambisonic", "binaural" }));
            Assert.Equal("", HybridRun.HrtfLabel("/data/mine.sofa", new[] { "ambisonic" }));
        }

        [Fact]
        public void Attributes_AreTheGraphAttributesOfTheFormats()
        {
            Assert.Equal("hybrid_binaural", HybridRun.Attributes["binaural"]);
            Assert.Equal("hybrid", HybridRun.Attributes["ambisonic"]);
        }
    }
}
