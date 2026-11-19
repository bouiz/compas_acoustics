using System;
using System.Collections.Generic;
using System.Linq;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>Which parameters Deconstruct Parameter gives: all, or the one named by ``select``.</summary>
    public static class DeconstructParameter
    {
        public static readonly string[] Names = { "EDT", "T20", "T30", "C50", "C80", "D50", "DRR", "Ts", "STI", "G", "SPL", "LF", "LFC" };

        // first item is the default
        public static readonly string[] Options = new[] { "all" }.Concat(Names).ToArray();

        public sealed record Selection(IReadOnlyList<ParameterGoo.Content> Parameters, bool Single);

        public static Selection Select(IReadOnlyList<ParameterGoo.Content> parameters, string? select, Action<string> warn)
        {
            var chosen = parameters.ToList();
            if (!string.IsNullOrEmpty(select) && select != "all")
            {
                chosen = chosen.Where(p => p.Name == select).ToList();
                if (chosen.Count == 0)
                    warn($"The parameters connected do not include '{select}'.");
            }
            return new Selection(chosen, chosen.Count == 1);
        }
    }

    /// <summary>The number of channels of a sound, as Check Channel counts it.</summary>
    public static class SoundChannels
    {
        /// <param name="fileChannels">The channels of an audio file, by path (the worker reads it).</param>
        public static int Of(object sound, Func<string, int> fileChannels) => sound switch
        {
            ImpulseResponseGoo ir => ir.Channels,
            AudioGoo audio => audio.Channels,
            SourceGoo source => source.SoundFile is string file && file.Length > 0 ? fileChannels(file) : throw new ArgumentException("The source has no sound file."),
            GH_String path => fileChannels(path.Value),
            string path => fileChannels(path),
            IGH_Goo goo => throw new ArgumentException($"Cannot read the channels of a {goo.TypeName}."),
            _ => throw new ArgumentException($"Cannot read the channels of a {sound.GetType().Name}."),
        };
    }

    public sealed class AnalyseIrComponent : ComponentBase
    {
        private static readonly string[] TimeScales = { "db", "linear" };

        public AnalyseIrComponent() : base("CA_Analyse_IR") { }
        protected override string SpecKey => "CA_Analyse_IR";
        public override Guid ComponentGuid => new Guid("410e7b44-f239-5e1e-ba74-5884e208b73e");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => (input, port.Name) switch
        {
            (true, "ir") => new Param_ImpulseResponse(),
            (false, "time") => new Param_TimeParameter { Owner = this },
            (false, _) => new Param_Parameter(),
            _ => null,
        };

        protected override IReadOnlyDictionary<string, string[]> MenuChoices { get; } = new Dictionary<string, string[]> { [Param_TimeParameter.Choice] = TimeScales };

        protected override void Solve(IGH_DataAccess access)
        {
            var ir = Item<ImpulseResponseGoo>(access, "ir");
            if (ir?.Value == null)
            {
                Warn("Input parameter 'ir' failed to collect data.");
                return;
            }
            var timeScale = Choice(Param_TimeParameter.Choice);
            RunInBackground(access, new Background("analysis", new object?[] { ir.Value, timeScale }, new Dictionary<string, object?>()), (outcome, _) =>
            {
                var result = (IDictionary<string, object?>)outcome.Result!;
                ShowWarnings(result);
                Status(timeScale == "db" ? "time in dB" : "time linear");
                var parameters = ((IEnumerable<object?>)result["parameters"]!).Cast<IDictionary<string, object?>>().Select(ParameterGoo.FromDictionary).ToList();
                access.SetDataList("parameters", parameters);
                access.SetData("time", ParameterGoo.FromDictionary((IDictionary<string, object?>)result["time"]!));
                access.SetData("frequency", ParameterGoo.FromDictionary((IDictionary<string, object?>)result["frequency"]!));
            });
        }

        protected override bool RunsInBackground => true;
    }

    public sealed class DeconstructParameterComponent : ComponentBase
    {
        public DeconstructParameterComponent() : base("CA_Deconstruct_IR") { }
        protected override string SpecKey => "CA_Deconstruct_IR";
        public override Guid ComponentGuid => new Guid("a6756aac-cd13-55a6-9f3e-068a5bd97765");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => (input, port.Name) switch
        {
            (true, "parameters") => new Param_Parameter(),
            (false, "parameter") => new Param_Parameter(),
            _ => null,
        };

        public override IReadOnlyList<Control> Controls { get; } = new Control[] { new DropdownControl("select", DeconstructParameter.Options) };

        public override void CreateAttributes() => m_attributes = new ParameterStripUi(this);

        /// <summary>The one parameter per band this component holds, drawn as a chart on its body; null otherwise.</summary>
        public ParameterGoo.Content? Strip { get; private set; }

        protected override void BeforeSolveInstance()
        {
            Strip = null;
            base.BeforeSolveInstance();
        }

        protected override void AfterSolveInstance()
        {
            base.AfterSolveInstance();
            Attributes?.ExpireLayout();
        }

        protected override void Solve(IGH_DataAccess access)
        {
            var given = List<ParameterGoo>(access, "parameters").Where(p => p?.Value != null).ToList();
            if (given.Count == 0)
            {
                Warn("Input parameter 'parameters' failed to collect data.");
                return;
            }
            var selection = DeconstructParameter.Select(given.Select(p => p.Value).ToList(), Text(access, "select"), Warn);
            var chosen = selection.Parameters;
            if (chosen.Count == 0)
                return;
            if (selection.Single)
            {
                var p = chosen[0];
                if (access.Iteration == 0 && p.XScale == "bands")
                    Strip = p;
                access.SetData("parameter", new ParameterGoo(p));
                access.SetData("name", p.Title);
                if (p.Values.Length == 1)
                    access.SetDataList("values", p.Values[0]);
                else
                {
                    // a curve of several channels: a branch per channel
                    var rows = new DataTree<double>();
                    for (var i = 0; i < p.Values.Length; i++)
                        rows.AddRange(p.Values[i], new GH_Path(i));
                    access.SetDataTree(2, rows);
                }
                access.SetData("overall", p.Overall);
                access.SetDataList("frequencies", p.Frequencies);
                return;
            }
            // every parameter in its own branch, so that they stay apart; they share their frequency bands
            var values = new DataTree<double>();
            for (var i = 0; i < chosen.Count; i++)
                values.AddRange(chosen[i].Values.Length > 0 ? chosen[i].Values[0] : Array.Empty<double>(), new GH_Path(i));
            access.SetDataList("parameter", chosen.Select(p => new ParameterGoo(p)));
            access.SetDataList("name", chosen.Select(p => p.Title));
            access.SetDataTree(2, values);
            access.SetDataList("overall", chosen.Select(p => p.Overall));
            access.SetDataList("frequencies", chosen[0].Frequencies);
        }
    }

    public sealed class CheckChannelComponent : ComponentBase
    {
        public CheckChannelComponent() : base("CA_Channel") { }
        protected override string SpecKey => "CA_Channel";
        public override Guid ComponentGuid => new Guid("23803429-885f-5cdd-9480-09282cd8a303");

        protected override void Solve(IGH_DataAccess access)
        {
            var sound = Item<IGH_Goo>(access, "sound");
            if (sound == null || (sound is GH_String text && string.IsNullOrEmpty(text.Value)))
            {
                Warn("Input parameter 'sound' failed to collect data.");
                Message = null;
                return;
            }
            int channels;
            try
            {
                channels = SoundChannels.Of(sound, path => Convert.ToInt32(((IDictionary<string, object?>)Worker.Call("sound_info", path)!)["channels"]));
            }
            catch (ArgumentException e)
            {
                Error(e.Message);
                Message = null;
                return;
            }
            Message = $"channels: {channels}";
            access.SetData(0, channels);
        }
    }
}
