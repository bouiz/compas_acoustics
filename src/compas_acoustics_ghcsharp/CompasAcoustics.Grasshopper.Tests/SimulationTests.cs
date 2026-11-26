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
}
