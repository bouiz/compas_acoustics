using System;
using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class AnalyseIrTests
    {
        private static ImpulseResponseGoo Ir(string receiver, string source) => new(new ImpulseResponseData
        {
            Name = $"{receiver}_{source}",
            Shape = { 1 },
            Samples = { 1 },
            Source = new SourceData { Guid = source, Name = source },
            Receiver = new ReceiverData { Guid = receiver, Name = receiver },
        });

        [Fact]
        public void Analyse_HasTheSourceAndReceiverDropdowns()
        {
            var analyse = new AnalyseIrComponent();
            Assert.Contains(SceneIndices.SourceDropdown, analyse.Controls);
            Assert.Contains(SceneIndices.ReceiverDropdown, analyse.Controls);
            var spec = ComponentSpec.Load("CA_Analyse_IR");
            Assert.Equal(new[] { "irs", "source_idx", "receiver_idx" }, spec.Inputs.Select(i => i.Name).ToArray());
            Assert.Equal(GH_ParamAccess.tree, spec.Inputs[0].Access);
            Assert.All(spec.Inputs.Skip(1), i => Assert.True(i.Optional));
            Assert.All(spec.Inputs.Skip(1), i => Assert.Equal(GH_ParamAccess.list, i.Access));
            Assert.Equal(new[] { "parameters", "time", "frequency" }, spec.Outputs.Select(o => o.Name).ToArray());
        }

        [Fact]
        public void Analyse_IndexDropdowns_OfferTheNamesOfTheResponses()
        {
            var analyse = new AnalyseIrComponent();
            var (sources, receivers) = PairChoice.Names(PairChoice.Number(new[] { Ir("rec", "a"), Ir("rec", "b") }));
            analyse.SetIndexOptions("source_idx", sources);
            analyse.SetIndexOptions("receiver_idx", receivers);
            Assert.Equal(new[] { "All", "0: a", "1: b" }, analyse.OptionsOf(SceneIndices.SourceDropdown).labels);
            Assert.Equal(new[] { "All", "0: rec" }, analyse.OptionsOf(SceneIndices.ReceiverDropdown).labels);
        }

        [Fact]
        public void Analyse_PathsAndTitlesNameThePair()
        {
            var pairs = PairChoice.Number(new[] { Ir("rec", "a"), Ir("rec", "b") });
            var branches = AnalyseIrComponent.Branches(pairs);
            Assert.Equal(new GH_Path(0, 0), branches[0].Path);
            Assert.Equal(" - a to rec", branches[0].TitleSuffix);
            Assert.Equal(new GH_Path(0, 1), branches[1].Path);
            Assert.Equal(" - b to rec", branches[1].TitleSuffix);
        }

        [Fact]
        public void Analyse_NoResponses_WarnsMissing()
        {
            var analyse = new AnalyseIrComponent();
            Assert.Null(analyse.Numbered(new ImpulseResponseGoo?[] { null, new ImpulseResponseGoo() }));
            Assert.Equal(new[] { "Input parameter 'irs' failed to collect data." }, analyse.RuntimeMessages(GH_RuntimeMessageLevel.Warning).ToArray());
            Assert.Empty(analyse.RuntimeMessages(GH_RuntimeMessageLevel.Error));
        }

        [Fact]
        public void Analyse_TwoResponsesForOnePair_IsAnError()
        {
            var analyse = new AnalyseIrComponent();
            Assert.Null(analyse.Numbered(new[] { Ir("rec", "a"), Ir("rec", "a") }));
            Assert.Equal(new[] { "Two responses for source 'a' and receiver 'rec': connect the irs of one Simulate component." },
                analyse.RuntimeMessages(GH_RuntimeMessageLevel.Error).ToArray());
            Assert.Empty(analyse.RuntimeMessages(GH_RuntimeMessageLevel.Warning));
        }

        [Fact]
        public void TheCallKey_ChangesWithTheBranchesTheResultsGoTo()
        {
            var a = Ir("rec", "a");
            var b = Ir("rec", "b");
            var args = new object?[] { new List<object?> { b.Value }, "db" };
            var viaB = AnalyseIrComponent.Branches(new[] { PairChoice.Number(new[] { a, b })[1] });
            var alone = AnalyseIrComponent.Branches(PairChoice.Number(new[] { b }));
            Assert.NotEqual(AnalyseIrComponent.CallKey(args, viaB), AnalyseIrComponent.CallKey(args, alone));
            Assert.Equal(AnalyseIrComponent.CallKey(args, alone), AnalyseIrComponent.CallKey(args, alone));
        }

        [Fact]
        public void Titled_AppendsTheSuffixToTheTitleAndKeepsTheName()
        {
            var parameter = new ParameterGoo.Content("T30", "Reverberation Time (T30)", "Frequency (Hz)", "Time (s)", "bands", new[] { 125.0 }, new[] { new[] { 1.0 } }, 1.0, Array.Empty<string>());
            var titled = AnalyseIrComponent.Titled(new ParameterGoo(parameter), " - a to rec").Value;
            Assert.Equal("Reverberation Time (T30) - a to rec", titled.Title);
            Assert.Equal("T30", titled.Name);
            Assert.Equal(parameter.Values, titled.Values);
        }
    }
}
