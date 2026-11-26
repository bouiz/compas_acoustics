using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>Which of the data Plot 2D charts: every parameter, or the one named by <c>select</c>; the time and frequency curves always.</summary>
    public static class PlotSelection
    {
        public static readonly string[] Names = { "EDT", "T20", "T30", "C50", "C80", "D50", "DRR", "Ts", "STI", "G", "SPL", "LF" };

        public static readonly string[] Titles =
        {
            "Early Decay Time", "Reverberation Time", "Reverberation Time", "Clarity", "Clarity", "Definition", "Direct-to-Reverberant Ratio",
            "Center Time", "Speech Transmission Index", "Sound Strength", "Sound Pressure Level", "Lateral Energy Fraction",
        };

        /// <summary>The sentence under the dropdown for each choice ("all" for All); edit the wording here.</summary>
        public static readonly IReadOnlyDictionary<string, string> Descriptions = new Dictionary<string, string>
        {
            ["all"] = "Every parameter the connected responses have.",
            ["EDT"] = "Early decay time: the time for a 60 dB decay, from the 0 to -10 dB slope of the Schroeder decay curve (ISO 3382-1).",
            ["T20"] = "Reverberation time: the time for a 60 dB decay, extrapolated from the -5 to -25 dB range of the decay curve (ISO 3382-1).",
            ["T30"] = "Reverberation time: the time for a 60 dB decay, extrapolated from the -5 to -35 dB range of the decay curve (ISO 3382-1).",
            ["C50"] = "Clarity for speech: the energy of the first 50 ms over the later energy, in dB (ISO 3382-1).",
            ["C80"] = "Clarity for music: the energy of the first 80 ms over the later energy, in dB (ISO 3382-1).",
            ["D50"] = "Definition: the share of the energy arriving in the first 50 ms, in percent (ISO 3382-1).",
            ["Ts"] = "Centre time: the centre of gravity of the squared impulse response, in seconds after the direct sound (ISO 3382-1).",
            ["DRR"] = "Direct-to-reverberant ratio: the energy of the direct sound over that of the reflected sound, in dB; a main cue for how far away a source sounds.",
            ["STI"] = "Speech transmission index: how intelligible speech is at the receiver, from 0 (unintelligible) to 1 (perfect) (IEC 60268-16).",
            ["G"] = "Strength: the level relative to the same source's free-field level at 10 m, in dB (ISO 3382-1).",
            ["SPL"] = "Sound pressure level: from the source's level and G; the single number is the A-weighted total.",
            ["LF"] = "Early lateral energy fraction: the share of the early sound that arrives from the sides; higher values give a wider apparent source (ISO 3382-1). Needs an ambisonic response.",
        };

        public static string Description(string choice) => Descriptions.TryGetValue(choice, out var text) ? text : "";

        public static readonly string[] Options = new[] { "all" }.Concat(Names).ToArray();

        public static readonly string[] Labels = new[] { "All" }.Concat(Names).ToArray();

        public static readonly string[] MenuLabels = new[] { "All" }.Concat(Names.Select((n, i) => $"{n} ({Titles[i]})")).ToArray();

        public static bool IsCurve(ParameterGoo.Content p) => p.XScale != "bands";

        public static IReadOnlyList<ParameterGoo.Content> Select(IReadOnlyList<ParameterGoo.Content> data, string? select, Action<string> warn)
        {
            if (string.IsNullOrEmpty(select) || select == "all")
                return data;
            var chosen = data.Where(p => IsCurve(p) || p.Name == select).ToList();
            if (data.Any(p => !IsCurve(p)) && chosen.All(IsCurve))
                warn(select == "LF"
                    ? "LF is not in the connected data: it needs an ambisonic response (Simulate Ambisonics)."
                    : $"{select} is not in the connected data.");
            return chosen;
        }

        /// <summary>"All" and the band parameters among <paramref name="present"/>, in <see cref="Names"/> order.</summary>
        public static (string[] values, string[] labels) OptionsFor(IReadOnlyCollection<string> present)
        {
            var names = Names.Where(present.Contains).ToArray();
            return (new[] { "all" }.Concat(names).ToArray(), new[] { "All" }.Concat(names).ToArray());
        }
    }

    /// <summary>Plot 2D: a chart per parameter on the wire, the one chosen on its body or all, drawn on its body (no geometry in Rhino).</summary>
    public sealed class Plot2DComponent : ComponentBase
    {
        private readonly List<ParameterGoo.Content> _charts = new();

        public Plot2DComponent() : base("CA_Plot_2D") { }
        protected override string SpecKey => "CA_Plot_2D";
        public override Guid ComponentGuid => new Guid("ddca21b5-bcc8-5806-85f8-fc0a45ced2e5");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => input && port.Name == "data" ? new Param_Parameter() : null;

        public override IReadOnlyList<Control> Controls { get; } = new Control[] { new DropdownControl("select", PlotSelection.Options, "T30", PlotSelection.Labels, "Select parameter", PlotSelection.MenuLabels) };

        public override void CreateAttributes() => m_attributes = new Plot2DUi(this);

        /// <summary>The parameters of the last solution, a chart each.</summary>
        public IReadOnlyList<ParameterGoo.Content> Charts => _charts;

        private bool _hasParameter;
        private readonly HashSet<string> _connected = new();

        /// <summary>"All" and the parameters of the connected data; every parameter while the data hold none.</summary>
        public override (string[] values, string[] labels) OptionsOf(DropdownControl dropdown) =>
            dropdown.Port == "select" && _hasParameter ? PlotSelection.OptionsFor(_connected) : base.OptionsOf(dropdown);

        protected override bool IsOption(string name, string value) => name == "select" ? value.Length > 0 : base.IsOption(name, value);

        /// <summary>The parameter's dropdown rests while the data hold no parameter: only curves, which are charted whichever is chosen.</summary>
        public override bool ControlEnabled(string port) => port != "select" || _hasParameter;

        /// <summary>The size of one chart, saved with the component.</summary>
        public SizeF ChartSize { get; set; } = new(360, 200);

        protected override void BeforeSolveInstance()
        {
            _charts.Clear();
            _hasParameter = false;
            _connected.Clear();
            base.BeforeSolveInstance();
        }

        protected override void Solve(IGH_DataAccess access)
        {
            access.GetDataTree("data", out GH_Structure<ParameterGoo> tree);
            var data = (tree?.FlattenData() ?? new List<ParameterGoo>()).Where(p => p?.Value != null).Select(p => p.Value).ToList();
            if (data.Count == 0)
            {
                WarnMissing("data");
                return;
            }
            _hasParameter = !data.All(PlotSelection.IsCurve);
            _connected.UnionWith(data.Where(p => !PlotSelection.IsCurve(p)).Select(p => p.Name));
            _charts.AddRange(PlotSelection.Select(data, Text(access, "select"), Warn));
        }

        protected override void AfterSolveInstance()
        {
            base.AfterSolveInstance();
            Attributes?.ExpireLayout();
        }

        public override bool Write(GH_IWriter writer)
        {
            writer.SetDouble("chart_width", ChartSize.Width);
            writer.SetDouble("chart_height", ChartSize.Height);
            return base.Write(writer);
        }

        public override bool Read(GH_IReader reader)
        {
            if (reader.ItemExists("chart_width") && reader.ItemExists("chart_height"))
                ChartSize = new SizeF((float)reader.GetDouble("chart_width"), (float)reader.GetDouble("chart_height"));
            return base.Read(reader);
        }
    }
}
