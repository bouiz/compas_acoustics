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
        public void Pyroomacoustics_LeavesUnsetInputsOutButAlwaysStatesAirAbsorption()
        {
            var settings = SimulationSettings.Pyroomacoustics(null, true, null, null, null, null, 7, null, out var ignored, out var seed);
            Assert.Equal(new Dictionary<string, object?> { ["ray_tracing"] = true, ["air_absorption"] = true, ["seed"] = 7L }, settings);
            Assert.Empty(ignored);
            Assert.Equal(7, seed);
        }

        [Fact]
        public void Pyroomacoustics_PassesGivenInputsUnderTheMethodsNames()
        {
            var settings = SimulationSettings.Pyroomacoustics(2, true, 1000, false, 1e-6, 1.5, 3, false, out _, out _);
            Assert.Equal(2L, settings["image_source_order"]);
            Assert.Equal(1000L, settings["n_rays"]);
            Assert.Equal(false, settings["air_absorption"]);
            Assert.Equal(1e-6, settings["energy_thres"]);
            Assert.Equal(1.5, settings["duration"]);
            Assert.Equal(false, settings["normalize"]);
        }

        [Fact]
        public void Pyroomacoustics_WithoutRayTracing_DropsTheRayInputsAndNamesThem()
        {
            var settings = SimulationSettings.Pyroomacoustics(1, false, 500, null, null, null, 3, null, out var ignored, out _);
            Assert.Equal(new[] { "n_rays", "seed" }, ignored);
            Assert.False(settings.ContainsKey("n_rays") || settings.ContainsKey("seed"));
            Assert.Equal(false, settings["ray_tracing"]);
        }

        [Fact]
        public void Pyroomacoustics_WithRayTracingAndNoSeed_DrawsOne()
        {
            var a = SimulationSettings.Pyroomacoustics(null, true, null, null, null, null, null, null, out _, out var first);
            SimulationSettings.Pyroomacoustics(null, true, null, null, null, null, null, null, out _, out var second);
            Assert.Equal((long)first, a["seed"]);
            Assert.InRange(first, 0, int.MaxValue);
            Assert.NotEqual(first, second);
        }

        [Fact]
        public void Pyroomacoustics_Message_ShowsRateMethodOrderRaysAirAndSeed()
        {
            var settings = SimulationSettings.Pyroomacoustics(null, true, null, null, null, null, 5, null, out _, out _);
            Assert.Equal("48000Hz, ISM + ray tracing, order=default, rays=default, air=True, seed=5", SimulationSettings.PyroomacousticsMessage(48000, settings));
            settings = SimulationSettings.Pyroomacoustics(2, false, null, false, null, null, null, null, out _, out _);
            Assert.Equal("44100Hz, ISM, order=2, rays=-, air=False, seed=-", SimulationSettings.PyroomacousticsMessage(44100, settings));
        }
    }

    public class IrGraphGroupingTests
    {
        private static IrGraphGoo.Content Graph(int sources, int receivers)
        {
            var s = Enumerable.Range(0, sources).Select(i => new SourceGoo(new SourceData { Guid = $"s{i}", Name = $"S{i}", Point = new PointData { X = i } })).ToList();
            var r = Enumerable.Range(0, receivers).Select(i => new ReceiverGoo(new ReceiverData { Guid = $"r{i}", Name = $"R{i}", Point = new PointData { Y = i } })).ToList();
            var irs = (from source in s
                       from receiver in r
                       select new ImpulseResponseGoo(new ImpulseResponseData
                       {
                           Name = $"{source.Name}{receiver.Name}", Source = source.Value, Receiver = receiver.Value, Shape = { 1 }, Samples = { 0 },
                       })).ToList();
            return new IrGraphGoo.Content("pyroomacoustics", s, r, irs);
        }

        [Fact]
        public void Grouping_PutsEachReceiversResponsesInItsBranch()
        {
            var warnings = new List<string>();
            var grouped = IrGraphGrouping.Group(Graph(2, 2), null, null, warnings.Add);
            Assert.Equal(new[] { 0, 1 }, grouped.Select(b => b.ReceiverIndex).ToArray());
            Assert.Equal(new[] { "S0R1", "S1R1" }, grouped[1].Responses.Select(ir => ir.Value.Name).ToArray());
            Assert.Equal(new[] { "S0", "S1" }, grouped[1].Sources.Select(s => s.Value.Name).ToArray());
            Assert.Empty(warnings);
        }

        [Fact]
        public void Grouping_FiltersBySourceAndReceiverIndex()
        {
            var grouped = IrGraphGrouping.Group(Graph(3, 2), new[] { 2 }, new[] { 1 }, _ => { });
            var branch = Assert.Single(grouped);
            Assert.Equal(1, branch.ReceiverIndex);
            Assert.Equal("S2R1", Assert.Single(branch.Responses).Value.Name);
        }

        [Fact]
        public void Grouping_WarnsForAnIndexOutOfRange()
        {
            var warnings = new List<string>();
            var grouped = IrGraphGrouping.Group(Graph(2, 1), new[] { 5, 0 }, new[] { 3, 0 }, warnings.Add);
            Assert.Contains("No source at index 5; the graph has 2 source(s), 0 to 1.", warnings);
            Assert.Contains("No receiver at index 3; the graph has 1 receiver(s), 0 to 0.", warnings);
            Assert.Equal("S0R0", Assert.Single(Assert.Single(grouped).Responses).Value.Name);
        }

        [Fact]
        public void Grouping_CopiesNoSamples()
        {
            var graph = Graph(2, 2);
            var grouped = IrGraphGrouping.Group(graph, null, null, _ => { });
            Assert.Same(graph.Responses[1].Value, grouped[1].Responses[0].Value);
        }
    }
}
