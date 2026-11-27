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
        public void Settings_LeavesUnsetInputsOut_TheSeedIncluded_AndAlwaysSendsATailAndALateEnvelope()
        {
            var settings = HybridRun.Settings(null, null, null, null, null, null, 48000, NoSources, NoReceivers, out var references);
            Assert.Equal(new Dictionary<string, object?> { ["tail"] = "ray_tracing", ["late_envelope"] = "omni" }, settings);
            Assert.Null(references);
            settings = HybridRun.Settings("ray_tracing", null, "none", null, 3, null, 48000, NoSources, NoReceivers, out _);
            Assert.Equal(new Dictionary<string, object?> { ["tail"] = "ray_tracing", ["late_envelope"] = "omni", ["seed"] = 3L }, settings);
        }

        [Theory]
        [InlineData(null, false, "ray_tracing")]
        [InlineData("", true, "ray_tracing")]
        [InlineData("auto", true, "reference")]
        [InlineData("auto", false, "ray_tracing")]
        [InlineData("reference", false, "reference")]
        [InlineData("reference", true, "reference")]
        [InlineData("ray_tracing", true, "ray_tracing")]
        [InlineData("eyring", true, "eyring")]
        [InlineData("sabine", false, "sabine")]
        public void Tail_ReadsAnOldAutoByTheReference_AndKeepsAnyOtherTail(string? saved, bool referenced, string expected) =>
            Assert.Equal(expected, HybridRun.Tail(saved, referenced));

        [Fact]
        public void Tails_AreRayTracingFirst_AndHaveNoAuto()
        {
            Assert.Equal(new[] { "ray_tracing", "reference", "eyring", "sabine" }, HybridRun.Tails);
            Assert.Equal(new[] { "ray_tracing", "reference", "eyring", "sabine" }, new SimulateBinauralComponent().Controls.OfType<DropdownControl>().Single(c => c.Port == "tail").Options);
        }

        [Theory]
        [InlineData("ray_tracing", true, true)]
        [InlineData("eyring", true, true)]
        [InlineData("sabine", true, true)]
        [InlineData("reference", true, false)]
        [InlineData("reference", false, true)]
        [InlineData("ray_tracing", false, false)]
        public void TailProblem_IsAnErrorForAReferenceWithAnotherTail_AndAWarningForTailReferenceWithoutOne(string tail, bool referenced, bool problem) =>
            Assert.Equal(problem, HybridRun.TailProblem(tail, referenced) != null);

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

        [Fact]
        public void LateEnvelopes_AreOmniFirst_AndHaveNoAuto() =>
            Assert.Equal(new[] { "omni", "directional", "off" }, HybridRun.LateEnvelopes);

        [Theory]
        [InlineData("ray_tracing", "omni", "omni")]
        [InlineData("ray_tracing", "directional", "directional")]
        [InlineData("ray_tracing", "off", false)]
        [InlineData("ray_tracing", "True", "directional")]
        [InlineData("ray_tracing", "False", false)]
        [InlineData("reference", "omni", "omni")]
        [InlineData("reference", "directional", "directional")]
        [InlineData("reference", "off", false)]
        public void Settings_PassesTheLateEnvelopeMode_AndReadsABoolean(string tail, string choice, object expected)
        {
            var settings = HybridRun.Settings(tail, null, null, choice, 1, null, 48000, NoSources, NoReceivers, out _);
            Assert.Equal(expected, settings["late_envelope"]);
        }

        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("auto")]
        [InlineData("nonsense")]
        public void Settings_SendsOmniForAnUnsetOrUnknownLateEnvelope(string? choice)
        {
            var settings = HybridRun.Settings(null, null, null, choice, 1, null, 48000, NoSources, NoReceivers, out _);
            Assert.Equal("omni", settings["late_envelope"]);
        }

        [Theory]
        [InlineData("eyring", null)]
        [InlineData("eyring", "omni")]
        [InlineData("eyring", "directional")]
        [InlineData("sabine", "omni")]
        [InlineData("sabine", "directional")]
        [InlineData("sabine", "off")]
        public void Settings_SendLateEnvelopeOffForAnEyringOrSabineTail_WhateverIsChosen(string tail, string? choice)
        {
            var settings = HybridRun.Settings(tail, null, null, choice, 1, null, 48000, NoSources, NoReceivers, out _);
            Assert.Equal(false, settings["late_envelope"]);
        }

        [Theory]
        [InlineData("ray_tracing", false)]
        [InlineData("reference", false)]
        [InlineData("eyring", true)]
        [InlineData("sabine", true)]
        public void TheLateEnvelopeRests_ForAnEyringOrSabineTail(string tail, bool rests) =>
            Assert.Equal(rests, HybridRun.LateEnvelopeRests(tail));

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
        [InlineData("ray_tracing", null, true)]
        [InlineData("ray_tracing", "omni", true)]
        [InlineData("ray_tracing", "off", true)]
        [InlineData("ray_tracing", "directional", true)]
        [InlineData("reference", "directional", true)]
        [InlineData("reference", "auto", false)]
        [InlineData("reference", null, false)]
        [InlineData("reference", "omni", false)]
        [InlineData("reference", "off", false)]
        [InlineData("eyring", null, false)]
        [InlineData("eyring", "directional", false)]
        [InlineData("sabine", null, false)]
        public void NRays_GoesToTheMethodOnlyWhenItTracesRaysItself(string tail, string? lateEnvelope, bool used) =>
            Assert.Equal(used, HybridRun.TracesRays(tail, lateEnvelope));

        [Fact]
        public void TheHybridComponents_TakeTheRaysTheyTrace()
        {
            foreach (var key in new[] { "CA_Simulate_Binaural", "CA_Simulate_Ambisonics" })
            {
                var names = ComponentSpec.Load(key).Inputs.Select(p => p.Name).ToList();
                Assert.Equal(names.IndexOf("late_envelope") + 1, names.IndexOf("n_rays"));
            }
        }

        private static SceneData SceneOf(string guid, int count = 1, double x = 1, string? sound = null)
        {
            var scene = new SceneData { Guid = guid, Name = "room", SamplingRate = 48000, Receivers = { new ReceiverData { Guid = "r", Name = "r" } } };
            for (var i = 0; i < count; i++)
            {
                var source = new SourceData { Guid = $"{guid}-s{i}", Name = $"s{i}", Point = new CompasPb.Data.PointData { X = x + i, Y = 2, Z = 1 } };
                if (sound != null)
                    source.SoundFile = sound;
                scene.Sources.Add(source);
            }
            return scene;
        }

        private static SceneGoo Scene(string guid, int count = 1, double x = 1, string? sound = null) => new(SceneOf(guid, count, x, sound));

        private static SimulateComponent.LastResult Last(string sceneGuid, int count = 1)
        {
            var scene = Scene(sceneGuid, count);
            var responses = scene.Value.Sources.Select(source =>
            {
                var ir = Response(source.Guid, "r");
                ir.Value.Source = source.Clone();
                return ir;
            }).ToArray();
            return new SimulateComponent.LastResult(scene, Reference(responses), new Dictionary<string, object?>());
        }

        private static (string text, bool changed) Status(SimulateComponent.LastResult? last, SceneGoo scene, bool settingsChanged = false) =>
            SimulateComponent.Status(last, scene, _ => settingsChanged);

        [Fact]
        public void Status_WithoutLastResult_SaysPressRun()
        {
            Assert.Equal(("press Run", false), Status(null, Scene("scene")));
        }

        [Fact]
        public void Status_ForAnotherScene_SaysTheModelChanged()
        {
            Assert.Equal(("model changed,\nrerun the simulation", true), Status(Last("old"), Scene("new", x: 5)));
            Assert.Equal(("model changed,\nrerun the simulation", true), Status(Last("old"), Scene("new", x: 5), settingsChanged: true));
        }

        [Fact]
        public void Status_ForResponsesThatHold_SaysHowManyWereSimulated()
        {
            Assert.Equal(("simulated 2 responses", false), Status(Last("scene", count: 2), Scene("scene", count: 2)));
            Assert.Equal(("simulated 1 response", false), Status(Last("scene"), Scene("scene")));
        }

        [Fact]
        public void Status_SaysSettingsChanged()
        {
            Assert.Equal(("settings changed,\nrerun the simulation", true), Status(Last("scene"), Scene("scene"), settingsChanged: true));
        }

        [Fact]
        public void Status_ForASoundOnlyChange_SaysHowManyWereSimulated_AndAPositionChangeSaysTheModelChanged()
        {
            var last = Last("scene", count: 2);
            Assert.Equal(("simulated 2 responses", false), Status(last, Scene("again", count: 2, sound: "speech")));
            Assert.Equal(("model changed,\nrerun the simulation", true), Status(last, Scene("again", count: 2, x: 3, sound: "speech")));
        }

        [Fact]
        public void Passed_ForASoundOnlyChange_KeepsTheLastResponses_WithTheSourcesCurrentSounds()
        {
            var last = Last("scene", count: 2);
            var scene = Scene("again", count: 2, sound: "speech");
            var passed = SimulateComponent.Passed(last, scene);
            Assert.Equal(2, passed.Count);
            Assert.Equal(new[] { "again-s0", "again-s1" }, passed.Select(ir => ir.SourceGuid));
            Assert.All(passed, ir => Assert.Equal("speech", ir.Value.Source.SoundFile));
            Assert.Equal(last.Responses.Select(ir => ir.Value.Samples.ToArray()), passed.Select(ir => ir.Value.Samples.ToArray()));
            Assert.All(last.Responses, ir => Assert.False(ir.Value.Source.HasSoundFile));
            Assert.Empty(SimulateComponent.Passed(null, scene));
        }

        [Fact]
        public void Passed_ForAnotherModel_KeepsTheLastResponsesAsTheyWere()
        {
            var last = Last("scene", count: 2);
            var passed = SimulateComponent.Passed(last, Scene("again", count: 2, x: 3, sound: "speech"));
            Assert.Equal(last.Responses, passed);
        }

        [Fact]
        public void TheSceneOfAResponse_SurvivesDuplicateCastAndASoundChange()
        {
            var scene = Scene("scene");
            var ir = Response("scene-s0", "r");
            ir.Value.Source = scene.Value.Sources[0].Clone();
            ir.SimulatedFor = scene;
            Assert.Same(scene, ((ImpulseResponseGoo)ir.Duplicate()).SimulatedFor);
            var cast = new ImpulseResponseGoo();
            Assert.True(cast.CastFrom(ir));
            Assert.Same(scene, cast.SimulatedFor);
            var passed = SimulatedScene.Current(new[] { ir }, scene, Scene("again", sound: "speech"));
            Assert.Same(scene, Assert.Single(passed).SimulatedFor);
        }

        [Fact]
        public void AReference_IsStale_OnlyWhenItWasSimulatedForAnotherModel()
        {
            var ir = Response("scene-s0", "r");
            Assert.False(HybridRun.Stale(new[] { ir }, Scene("scene")));
            ir.SimulatedFor = Scene("other");
            Assert.False(HybridRun.Stale(new[] { ir }, Scene("scene")));
            Assert.False(HybridRun.Stale(new[] { ir }, Scene("scene", sound: "speech")));
            Assert.True(HybridRun.Stale(new[] { ir }, Scene("scene", x: 5)));
        }
    }
}
