using System;
using System.Collections.Generic;
using System.Linq;
using GH_IO.Serialization;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;

namespace CompasAcoustics.Grasshopper
{
    public sealed class AnalyseIrComponent : ComponentBase
    {
        public AnalyseIrComponent() : base("CA_Analyse_IR") { }
        protected override string SpecKey => "CA_Analyse_IR";
        public override Guid ComponentGuid => new Guid("410e7b44-f239-5e1e-ba74-5884e208b73e");

        public override float MinBodyWidth => SceneBodyWidth;

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => (input, port.Name) switch
        {
            (true, "irs") => new Param_ImpulseResponse(),
            (false, "time") => new Param_TimeParameter { Owner = this },
            (false, _) => new Param_Parameter(),
            _ => null,
        };

        protected override IReadOnlyDictionary<string, string[]> MenuChoices { get; } = new Dictionary<string, string[]> { [Param_TimeParameter.Choice] = Param_TimeParameter.Options };

        public override IReadOnlyList<Control> Controls { get; } = new Control[] { SceneIndices.SourceDropdown, SceneIndices.ReceiverDropdown };

        /// <summary>
        /// A definition saved with the single ir input, now irs (it had no controls): it keeps its place and takes the name and
        /// tree access of irs, and the inputs it did not save are set aside while Grasshopper reads the saved ones by
        /// position, so that none is reported missing, and come back after it, empty and hidden, as on a new Analyse.
        /// </summary>
        public override bool Read(GH_IReader reader)
        {
            if (!SavedInput(reader, "ir"))
                return base.Read(reader);
            var saved = 0;
            while (reader.ChunkExists("param_input", saved))
                saved++;
            var absent = Params.Input.Skip(saved).ToList();
            var positions = absent.Select(p => Params.Input.IndexOf(p)).ToList();
            foreach (var port in absent)
                Params.UnregisterInputParameter(port, false);
            var read = base.Read(reader);
            if (Params.Input[0].Name == "ir")
            {
                var port = Params.Input[0];
                port.Name = "irs";
                if (port.NickName == "ir")
                    port.NickName = "irs";
                port.Access = GH_ParamAccess.tree;
            }
            for (var i = 0; i < absent.Count; i++)
                Params.RegisterInputParam(absent[i], positions[i]);
            HideControlPortsAsNew();
            return read;
        }

        /// <summary>The branch of each pair, {receiver;source}, and the suffix its parameters' titles take.</summary>
        public static List<(GH_Path Path, string TitleSuffix)> Branches(IReadOnlyList<Pair> chosen) =>
            chosen.Select(p => (PairChoice.PathOf(p), $" - {p.Response.SourceName} to {p.Response.ReceiverName}")).ToList();

        /// <summary>The parameter with <paramref name="suffix"/> after its title; its name stays what Plot 2D selects by.</summary>
        public static ParameterGoo Titled(ParameterGoo parameter, string suffix) => new(parameter.Value with { Title = parameter.Value.Title + suffix });

        /// <summary>The key of a call: the worker's arguments and the branches the results go to, which a kept result must not take from an earlier call.</summary>
        public static string CallKey(object?[] args, IReadOnlyList<(GH_Path Path, string TitleSuffix)> branches) =>
            BackgroundRuns.Key("analysis", args, new Dictionary<string, object?>()) + string.Concat(branches.Select(b => $"|{b.Path}{b.TitleSuffix}"));

        protected override void Solve(IGH_DataAccess access)
        {
            if (ChosenPairs(access) is not { Chosen: var chosen })
                return;
            var timeScale = Choice(Param_TimeParameter.Choice);
            var branches = Branches(chosen);
            var args = new object?[] { chosen.Select(p => (object?)p.Response.Value).ToList(), timeScale };
            RunInBackground(access, new Background("analysis", args, new Dictionary<string, object?>(), branches, CallKey(args, branches)), (outcome, _) =>
            {
                var result = outcome.Result;
                var paths = (List<(GH_Path Path, string TitleSuffix)>)outcome.Context!;
                Status(timeScale == "db" ? "time in dB" : "time linear");
                var parameters = new DataTree<ParameterGoo>();
                var time = new DataTree<ParameterGoo>();
                var frequency = new DataTree<ParameterGoo>();
                var analyses = ((IEnumerable<object?>)result["analyses"]!).Cast<IDictionary<string, object?>>().ToList();
                for (var i = 0; i < analyses.Count; i++)
                {
                    var (path, suffix) = paths[i];
                    parameters.AddRange(((IEnumerable<object?>)analyses[i]["parameters"]!).Cast<IDictionary<string, object?>>().Select(p => Titled(ParameterGoo.FromDictionary(p), suffix)), path);
                    time.Add(Titled(ParameterGoo.FromDictionary((IDictionary<string, object?>)analyses[i]["time"]!), suffix), path);
                    frequency.Add(Titled(ParameterGoo.FromDictionary((IDictionary<string, object?>)analyses[i]["frequency"]!), suffix), path);
                }
                access.SetDataTree(0, parameters);
                access.SetDataTree(1, time);
                access.SetDataTree(2, frequency);
            });
        }

        protected override bool RunsInBackground => true;
    }
}
