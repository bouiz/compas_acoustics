using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>Plot 2D: a chart per parameter on the wire, drawn on its body (no geometry in Rhino).</summary>
    public sealed class Plot2DComponent : ComponentBase
    {
        private readonly List<ParameterGoo.Content> _charts = new();

        public Plot2DComponent() : base("CA_Plot_2D") { }
        protected override string SpecKey => "CA_Plot_2D";
        public override Guid ComponentGuid => new Guid("ddca21b5-bcc8-5806-85f8-fc0a45ced2e5");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => input && port.Name == "data" ? new Param_Parameter() : null;

        public override void CreateAttributes() => m_attributes = new Plot2DUi(this);

        /// <summary>The parameters of the last solution, a chart each.</summary>
        public IReadOnlyList<ParameterGoo.Content> Charts => _charts;

        /// <summary>The size of one chart, saved with the component.</summary>
        public SizeF ChartSize { get; set; } = new(360, 200);

        protected override void BeforeSolveInstance()
        {
            _charts.Clear();
            base.BeforeSolveInstance();
        }

        protected override void Solve(IGH_DataAccess access)
        {
            var data = List<ParameterGoo>(access, "data").Where(p => p?.Value != null).ToList();
            if (data.Count == 0)
            {
                Warn("Input parameter 'data' failed to collect data.");
                return;
            }
            _charts.AddRange(data.Select(p => p.Value));
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
