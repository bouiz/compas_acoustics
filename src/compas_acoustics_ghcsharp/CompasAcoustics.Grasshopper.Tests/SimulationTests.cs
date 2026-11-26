using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;
using CompasPb.Data;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class SimulationSettingsTests
    {
        [Fact]
        public void Mono_LeavesUnsetInputsOut_AndAlwaysAbsorbsInAir()
        {
            var settings = SimulationSettings.Mono(null, true, null, null, 7, out var ignored);
            Assert.Equal(new Dictionary<string, object?> { ["ray_tracing"] = true, ["air_absorption"] = true, ["seed"] = 7L }, settings);
            Assert.Empty(ignored);
        }

        [Fact]
        public void Mono_PassesGivenInputsUnderTheMethodsNames()
        {
            var settings = SimulationSettings.Mono(2, true, 1000, 1.5, 3, out _);
            Assert.Equal(2L, settings["image_source_order"]);
            Assert.Equal(1000L, settings["n_rays"]);
            Assert.Equal(true, settings["air_absorption"]);
            Assert.False(settings.ContainsKey("energy_thres"));
            Assert.Equal(1.5, settings["duration"]);
        }

        [Fact]
        public void Mono_SendsNoNormalize()
        {
            Assert.False(SimulationSettings.Mono(2, true, 1000, 1.5, 3, out _).ContainsKey("normalize"));
        }

        [Fact]
        public void Mono_WithoutRayTracing_DropsTheRayInputsAndNamesThem()
        {
            var settings = SimulationSettings.Mono(1, false, 500, null, 3, out var ignored);
            Assert.Equal(new[] { "n_rays", "seed" }, ignored);
            Assert.False(settings.ContainsKey("n_rays") || settings.ContainsKey("seed"));
            Assert.Equal(false, settings["ray_tracing"]);
        }

        [Fact]
        public void Mono_WithoutASeed_LeavesItToTheMethod()
        {
            Assert.False(SimulationSettings.Mono(null, true, null, null, null, out _).ContainsKey("seed"));
        }
    }

    public class SimulatedSceneTests
    {
        private static SceneGoo Scene(string guid, double x = 1, string? sound = null)
        {
            var source = new SourceData { Guid = $"{guid}-s", Name = "s", Point = new PointData { X = x, Y = 2, Z = 1 }, Directivity = "omnidirectional" };
            if (sound != null)
                source.SoundFile = sound;
            return new SceneGoo(new SceneData
            {
                Guid = guid, Name = "room", SamplingRate = 48000, Sources = { source },
                Receivers = { new ReceiverData { Guid = "r", Name = "r", Point = new PointData { X = 4, Y = 2, Z = 1 } } },
            });
        }

        private static readonly Dictionary<string, object?> Settings = new() { ["ism_order"] = 3L };

        private static ImpulseResponseData Response(SceneGoo scene) => new()
        {
            Guid = "ir", Source = scene.Value.Sources[0].Clone(), Receiver = scene.Value.Receivers[0].Clone(),
            Shape = { 4 }, SamplingRate = 48000, Samples = { 1.0, 0.5, 0.25, 0.0 },
        };

        private static Dictionary<string, ImpulseResponseData> References(SceneGoo scene) =>
            new() { [$"r_{scene.Value.Sources[0].Guid}"] = Response(scene) };

        [Fact]
        public void Key_OfTwoScenesDifferingOnlyInASourcesSound_IsTheSame()
        {
            var key = SimulatedScene.Key(Scene("a"), "mono", Settings, null);
            Assert.Equal(key, SimulatedScene.Key(Scene("b", sound: "speech"), "mono", Settings, null));
            Assert.Equal(key, SimulatedScene.Key(Scene("c", sound: "piano"), "mono", Settings, null));
        }

        [Fact]
        public void Key_OfTwoScenesDifferingInASourcesPosition_Differs()
        {
            Assert.NotEqual(SimulatedScene.Key(Scene("a"), "mono", Settings, null), SimulatedScene.Key(Scene("a", x: 2), "mono", Settings, null));
            Assert.NotEqual(SimulatedScene.Key(Scene("a"), "mono", Settings, null), SimulatedScene.Key(Scene("a"), "binaural", Settings, null));
        }

        [Fact]
        public void Key_OfAHybrid_IgnoresTheSoundOfItsReference_ButNotItsSamples()
        {
            var before = Scene("a");
            var after = Scene("b", sound: "speech");
            var key = SimulatedScene.Key(before, "binaural", Settings, References(before));
            Assert.Equal(key, SimulatedScene.Key(after, "binaural", Settings, References(after)));
            var other = References(after);
            other.Values.First().Samples[1] = 0.4;
            Assert.NotEqual(key, SimulatedScene.Key(after, "binaural", Settings, other));
        }

        private static SceneGoo Solved(string guid, double absorption = 0.1)
        {
            var scene = Scene(guid);
            scene.Value.Receivers[0].Guid = $"{guid}-r";
            scene.Value.AcousticObjects.Add(new AcousticObjectData
            {
                Guid = $"{guid}-o", Name = "wall", Role = "boundary", Sources = { new SourceData { Guid = $"{guid}-os", Name = "os" } },
                Material = new AcousticMaterialData { Guid = $"{guid}-m", Name = "plaster", AbsorptionCoefficients = { absorption } },
            });
            return scene;
        }

        [Fact]
        public void Same_IgnoresTheGuidsAComponentDrawsAnewOnEverySolution()
        {
            Assert.True(SimulatedScene.Same(Solved("a"), Solved("b")));
            Assert.False(SimulatedScene.Same(Solved("a"), Solved("b", absorption: 0.2)));
        }

        [Fact]
        public void Current_GivesTheResponsesTheReceiversCurrentGuid()
        {
            var ran = Solved("a");
            var scene = Solved("b");
            var current = SimulatedScene.Current(new[] { new ImpulseResponseGoo(Response(ran)) }, ran, scene).Single();
            Assert.Equal("b-r", current.ReceiverGuid);
            Assert.Equal("b-s", current.SourceGuid);
        }

        [Fact]
        public void Key_OfAHybrid_IgnoresTheGuidsOfItsReferencesReceiver()
        {
            var before = Solved("a");
            var after = Solved("b");
            Assert.Equal(SimulatedScene.Key(before, "binaural", Settings, new Dictionary<string, ImpulseResponseData> { ["x"] = Response(before) }),
                SimulatedScene.Key(after, "binaural", Settings, new Dictionary<string, ImpulseResponseData> { ["x"] = Response(after) }));
        }

        [Fact]
        public void Current_GivesTheResponsesTheSourcesCurrentSound_AndLeavesTheRunsOwn()
        {
            var ran = Scene("a");
            var ir = new ImpulseResponseGoo(Response(ran));
            var scene = Scene("b", sound: "speech");
            var current = SimulatedScene.Current(new[] { ir }, ran, scene).Single();
            Assert.Equal("speech", current.Value.Source.SoundFile);
            Assert.Equal("b-s", current.SourceGuid);
            Assert.Equal(ir.Value.Samples, current.Value.Samples);
            Assert.False(ir.Value.Source.HasSoundFile);
            Assert.Equal("a-s", ir.SourceGuid);
            var silent = SimulatedScene.Current(new[] { current }, scene, Scene("c")).Single();
            Assert.False(silent.Value.Source.HasSoundFile);
            Assert.Same(ir, SimulatedScene.Current(new[] { ir }, ran, ran).Single());
        }
    }
}
