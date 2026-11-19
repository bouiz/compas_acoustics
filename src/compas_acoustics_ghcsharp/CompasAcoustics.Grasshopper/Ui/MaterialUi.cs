using System;
using System.Drawing;
using System.Linq;
using CompasAcoustics.Pb;
using Grasshopper.GUI;
using Grasshopper.GUI.Canvas;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>A component that shows a material on its body: Material Library and Construct Material.</summary>
    public interface IMaterialBody
    {
        AcousticMaterialData? Shown { get; }

        string Caption { get; }

        /// <summary>Open the picker; null when the body does not pick (Construct Material).</summary>
        Action? Pick { get; }
    }

    /// <summary>The body below the ports: the material's name and its absorption and scattering per band as paired bars (0 to 1), as in the picker.</summary>
    public sealed class MaterialUi : ComponentUi
    {
        private const float Height = ChartLayout.StripHeight;
        private static readonly Color AbsorptionColor = Color.FromArgb(70, 70, 70);
        private static readonly Color ScatteringColor = Color.FromArgb(110, 165, 220);

        public MaterialUi(ComponentBase owner) : base(owner) { }

        private IMaterialBody Body => (IMaterialBody)Owner;

        protected override float ExtraHeight => Height;

        protected override void RenderExtra(GH_Canvas canvas, Graphics g)
        {
            var area = ExtraArea;
            using var font = new Font(FontFamily.GenericSansSerif, 6f);
            var caption = new RectangleF(area.Left + 3, area.Top + 1, area.Width - 6, 10);
            using var format = new StringFormat { Trimming = StringTrimming.EllipsisCharacter, FormatFlags = StringFormatFlags.NoWrap };
            g.DrawString(Body.Caption, font, Body.Shown == null ? Brushes.Gray : Brushes.Black, caption, format);
            var strip = ChartLayout.StripArea(area);
            g.FillRectangle(Brushes.White, strip);
            if (Body.Shown is not { } material)
                return;
            // the bands below, 0 and 1 beside
            var labelWidth = Math.Max(GH_FontServer.StringWidth("0", ChartPainter.Tiny), GH_FontServer.StringWidth("1", ChartPainter.Tiny));
            var chart = MaterialChart.Strip(material.AbsorptionCoefficients.ToList(),
                material.ScatteringCoefficients.Count > 0 ? material.ScatteringCoefficients.ToList() : null, strip, labelWidth);
            var plot = chart.Plot;
            using (var grid = new Pen(Color.FromArgb(230, 230, 230)))
                foreach (var (y, text) in chart.Values)
                {
                    g.DrawLine(grid, plot.Left, y, plot.Right, y);
                    var size = g.MeasureString(text, ChartPainter.Tiny);
                    g.DrawString(text, ChartPainter.Tiny, Brushes.DimGray, plot.Left - size.Width - ChartLayout.TickGap + 2, y - size.Height / 2);
                }
            foreach (var (x, text) in chart.Bands)
            {
                var size = g.MeasureString(text, ChartPainter.Tiny);
                g.DrawString(text, ChartPainter.Tiny, Brushes.DimGray, x - size.Width / 2, plot.Bottom + 1);
            }
            g.DrawRectangle(Pens.Gray, plot.X, plot.Y, plot.Width, plot.Height);
            // the picker's paired bars, absorption then scattering per band
            var (absorption, scattering) = MaterialChart.PairedBars(material.AbsorptionCoefficients.ToList(),
                material.ScatteringCoefficients.Count > 0 ? material.ScatteringCoefficients.ToList() : null, plot);
            foreach (var (bars, color) in new[] { (absorption, AbsorptionColor), (scattering, ScatteringColor) })
            {
                using var fill = new SolidBrush(color);
                foreach (var r in bars.Where(r => r.Height > 0))
                    g.FillRectangle(fill, r);
            }
        }

        protected override GH_ObjectResponse? RespondToExtraMouseDown(GH_Canvas sender, GH_CanvasMouseEvent e)
        {
            if (Body.Pick is not { } pick || !ExtraArea.Contains(e.CanvasLocation))
                return null;
            pick();
            return GH_ObjectResponse.Handled;
        }
    }
}
