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
            new IrGraphGoo(new IrGraphGoo.Content("mono", new List<SourceGoo>(), new List<ReceiverGoo>(), responses));

        [Fact]
        public void Settings_LeavesUnsetInputsOutAndDrawsASeed()
        {
            var settings = HybridRun.Settings(null, null, null, null, null, null, null, 48000, out var references);
            Assert.Equal(new[] { "seed" }, settings.Keys.ToArray());
            Assert.IsType<long>(settings["seed"]);
            Assert.Null(references);
            settings = HybridRun.Settings("auto", null, "none", null, 3, null, null, 48000, out _);
            Assert.Equal(new Dictionary<string, object?> { ["seed"] = 3L }, settings);
        }

        [Fact]
        public void Settings_PassesTheEngineInputsUnderTheMethodsNames_AndNoFormatOfTheirOwn()
        {
            var settings = HybridRun.Settings("eyring", 4, "udfa", false, 7, false, null, 48000, out _);
            Assert.Equal(new Dictionary<string, object?>
            {
                ["tail"] = "eyring", ["image_source_order"] = 4L, ["diffraction"] = "udfa", ["late_envelope"] = false, ["seed"] = 7L, ["normalize"] = false,
            }, settings);
        }

        [Fact]
        public void Settings_TakesTheMonoResponsesOfTheReferenceByKey()
        {
            var reference = Reference(Response("s0", "r0"), Response("s1", "r0", channels: 4));
            HybridRun.Settings("reference", null, null, null, 1, null, reference, 48000, out var references);
            Assert.Equal(new[] { "r0_s0" }, references!.Keys.ToArray());
        }

        [Fact]
        public void Settings_RejectsAReferenceWithoutMonoResponses()
        {
            var error = Assert.Throws<ArgumentException>(() =>
                HybridRun.Settings(null, null, null, null, 1, null, Reference(Response("s", "r", channels: 2)), 48000, out _));
            Assert.Equal("The connected ir_graph holds no mono responses; connect the mono_ir_graph output of Simulate Mono.", error.Message);
        }

        [Fact]
        public void Settings_RejectsAReferenceAtAnotherRate()
        {
            var error = Assert.Throws<ArgumentException>(() =>
                HybridRun.Settings(null, null, null, null, 1, null, Reference(Response("s", "r", rate: 44100)), 48000, out _));
            Assert.Equal("The reference was simulated at 44100 Hz, the scene's sampling_rate is 48000 Hz: simulate both at the same rate.", error.Message);
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
            Assert.Equal("press Run", HybridRun.Status(null, "scene", null, null, null, null));
        }

        [Fact]
        public void Status_ForAnotherScene_SaysPressRun()
        {
            Assert.Equal("press Run", HybridRun.Status(Last("old"), "new", null, null, null, null));
        }

        [Fact]
        public void Status_SaysDoneWithTheCountAndSummary_TheOrderOnlyWhereThereIsOne()
        {
            Assert.Equal("done, 2 IRs, tail: reference, order 1 (Run to redo)", HybridRun.Status(Last("scene", count: 2), "scene", null, null, null, null));
            Assert.Equal("done, 1 IR, tail: reference, order 1 (Run to redo)", HybridRun.Status(Last("scene"), "scene", "auto", 1, 3, "none"));
            Assert.Equal("done, 1 IR, tail: reference (Run to redo)", HybridRun.Status(Last("scene", shOrder: null), "scene", null, null, null, null));
        }

        [Fact]
        public void Status_SaysSettingsChanged()
        {
            var last = Last("scene");
            const string changed = "settings changed, press Run (tail: reference, order 1)";
            Assert.Equal(changed, HybridRun.Status(last, "scene", null, null, 4, null));
            Assert.Equal(changed, HybridRun.Status(last, "scene", "eyring", null, null, null));
            Assert.Equal(changed, HybridRun.Status(last, "scene", null, 2, null, null));
            Assert.Equal(changed, HybridRun.Status(last, "scene", null, null, null, "udfa"));
            Assert.Equal("done, 1 IR, tail: reference, order 1 (Run to redo)", HybridRun.Status(Last("scene", diffraction: "udfa"), "scene", null, null, null, "udfa"));
        }
    }
}
