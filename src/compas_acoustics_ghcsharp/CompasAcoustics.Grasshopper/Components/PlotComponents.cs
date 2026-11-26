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
                warn($"The parameters connected do not include '{select}'.");
            return chosen;
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

        /// <summary>The parameter's dropdown rests while the data hold no parameter: only curves, which are charted whichever is chosen.</summary>
        public override bool ControlEnabled(string port) => port != "select" || _hasParameter;

        /// <summary>The size of one chart, saved with the component.</summary>
        public SizeF ChartSize { get; set; } = new(360, 200);

        protected override void BeforeSolveInstance()
        {
            _charts.Clear();
            _hasParameter = false;
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
