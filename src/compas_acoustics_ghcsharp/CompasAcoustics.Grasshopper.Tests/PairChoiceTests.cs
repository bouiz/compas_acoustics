using System;
using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;
using Grasshopper.Kernel.Data;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class PairChoiceTests
    {
        private static ImpulseResponseData Data(string receiver, string source) => new()
        {
            Name = $"{receiver}_{source}",
            Shape = { 1 },
            Samples = { 1 },
            Source = new SourceData { Guid = source, Name = $"name {source}" },
            Receiver = new ReceiverData { Guid = receiver, Name = $"name {receiver}" },
        };

        private static ImpulseResponseGoo Ir(string receiver, string source) => new(Data(receiver, source));

        private static List<Pair> Pairs(params (int receiver, int source)[] ids) =>
            ids.Select(p => new Pair(p.receiver, p.source, Ir($"r{p.receiver}", $"s{p.source}"))).ToList();

        private static (int, int)[] Ids(IEnumerable<Pair> pairs) => pairs.Select(p => (p.Receiver, p.Source)).ToArray();

        [Fact]
        public void Number_CountsSourcesAndReceiversByFirstAppearance()
        {
            var irs = new[] { Ir("r0", "s0"), Ir("r0", "s1"), Ir("r1", "s0"), Ir("r1", "s1") };
            var pairs = PairChoice.Number(irs);
            Assert.Equal(new[] { (0, 0), (0, 1), (1, 0), (1, 1) }, Ids(pairs));
            var reversed = PairChoice.Number(Enumerable.Reverse(irs));
            Assert.Equal(new[] { (0, 0), (0, 1), (1, 0), (1, 1) }, Ids(reversed));
            Assert.Equal("r1", reversed[0].Response.ReceiverGuid);
            Assert.Equal("s1", reversed[0].Response.SourceGuid);
        }

        [Fact]
        public void Number_TwoResponsesForOnePair_Throws()
        {
            var ex = Assert.Throws<ArgumentException>(() => PairChoice.Number(new[] { Ir("r0", "s0"), Ir("r0", "s0") }));
            Assert.Equal("Two responses for source 'name s0' and receiver 'name r0': connect the irs of one Simulate component.", ex.Message);
        }

        [Fact]
        public void Number_SkipsNullAndInvalid()
        {
            var pairs = PairChoice.Number(new ImpulseResponseGoo?[] { null, new ImpulseResponseGoo(), Ir("r0", "s0") }!);
            Assert.Single(pairs);
        }

        [Fact]
        public void Names_ByIndex()
        {
            var (sources, receivers) = PairChoice.Names(Pairs((0, 0), (0, 1), (1, 0), (1, 1)));
            Assert.Equal(new[] { "name s0", "name s1" }, sources);
            Assert.Equal(new[] { "name r0", "name r1" }, receivers);
        }

        [Fact]
        public void Choose_EmptyIndicesGiveEveryPair()
        {
            var pairs = Pairs((0, 0), (0, 1), (1, 0), (1, 1));
            Assert.Equal(4, PairChoice.Choose(pairs, null, null, _ => { }).Count);
            Assert.Equal(4, PairChoice.Choose(pairs, new List<int>(), new List<int>(), _ => { }).Count);
        }

        [Fact]
        public void Choose_IsEveryChosenSourceWithEveryChosenReceiver()
        {
            var pairs = Pairs((0, 0), (0, 1), (1, 0), (1, 1));
            var chosen = PairChoice.Choose(pairs, new[] { 1 }, new[] { 1, 0 }, _ => { });
            Assert.Equal(new[] { (0, 1), (1, 1) }, Ids(chosen));
        }

        [Fact]
        public void Choose_OutOfRangeWarnsAndSkips()
        {
            var pairs = Pairs((0, 0), (0, 1), (1, 0), (1, 1));
            var warnings = new List<string>();
            var none = PairChoice.Choose(pairs, new[] { 3 }, null, warnings.Add);
            Assert.Empty(none);
            Assert.Equal(new[] { "No source at index 3; the responses have 2 source(s), 0 to 1.", "No pair chosen." }, warnings);

            warnings.Clear();
            var some = PairChoice.Choose(pairs, new[] { 0, 3 }, null, warnings.Add);
            Assert.Equal(new[] { (0, 0), (1, 0) }, Ids(some));
            Assert.Equal(new[] { "No source at index 3; the responses have 2 source(s), 0 to 1." }, warnings);
        }

        [Fact]
        public void Choose_OutOfRangeReceiverWarns()
        {
            var warnings = new List<string>();
            PairChoice.Choose(Pairs((0, 0), (1, 0)), null, new[] { -1 }, warnings.Add);
            Assert.Equal("No receiver at index -1; the responses have 2 receiver(s), 0 to 1.", warnings[0]);
        }

        [Fact]
        public void InSceneOrder_SortsByReceiverThenSource()
        {
            var scene = new SceneGoo(new SceneData
            {
                Sources = { new SourceData { Guid = "s0" }, new SourceData { Guid = "s1" } },
                Receivers = { new ReceiverData { Guid = "r0" }, new ReceiverData { Guid = "r1" } },
            });
            var responses = new Dictionary<string, object?>
            {
                ["r1_s0"] = Data("r1", "s0"),
                ["r0_s1"] = Data("r0", "s1"),
                ["r0_s0"] = Data("r0", "s0"),
            };
            var ordered = PairChoice.InSceneOrder(scene, responses);
            Assert.Equal(new[] { "r0_s0", "r0_s1", "r1_s0" }, ordered.Select(ir => ir.Name).ToArray());
        }

        [Fact]
        public void PathAndLabel()
        {
            var pair = new Pair(2, 1, Ir("r2", "s1"));
            Assert.Equal("{2;1}", PairChoice.PathOf(pair).ToString());
            Assert.Equal("2_1", PairChoice.LabelOf(pair));
        }
    }
}
