using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using Grasshopper.GUI;
using Grasshopper.GUI.Canvas;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>Draws charts with GDI+: bars, lines per channel, ticks, titles.</summary>
    internal static class ChartPainter
    {
        private static readonly Color[] ChannelColors =
        {
            Color.FromArgb(31, 119, 180), Color.FromArgb(214, 39, 40), Color.FromArgb(44, 160, 44), Color.FromArgb(148, 103, 189),
        };

        private const float DotRadius = 2.5f;
        private const int Supersample = 4;
        private static readonly Dictionary<string, Bitmap> Upward = new();

        /// <summary>
        /// ``text`` reading upwards, as an image four times its size: the Mac draws text under a rotated transform at odd
        /// angles and places, so the text is drawn level and its pixels are turned.
        /// </summary>
        private static Bitmap UpwardText(string text, Font font)
        {
            lock (Upward)
            {
                if (Upward.TryGetValue(text, out var known))
                    return known;
                using var large = new Font(font.FontFamily, font.Size * Supersample, font.Style);
                SizeF size;
                using (var probe = new Bitmap(1, 1))
                using (var measure = Graphics.FromImage(probe))
                    size = measure.MeasureString(text, large);
                var image = new Bitmap(Math.Max((int)Math.Ceiling(size.Width), 1), Math.Max((int)Math.Ceiling(size.Height), 1));
                using (var g = Graphics.FromImage(image))
                {
                    g.Clear(Color.Transparent);
                    g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;
                    g.DrawString(text, large, Brushes.Black, 0, 0);
                }
                var pixels = new int[image.Width * image.Height];
                for (var y = 0; y < image.Height; y++)
                    for (var x = 0; x < image.Width; x++)
                        pixels[y * image.Width + x] = image.GetPixel(x, y).ToArgb();
                var turnedPixels = ChartLayout.TurnUpward(image.Width, image.Height, pixels);
                var turned = new Bitmap(image.Height, image.Width);
                for (var y = 0; y < turned.Height; y++)
                    for (var x = 0; x < turned.Width; x++)
                        turned.SetPixel(x, y, Color.FromArgb(turnedPixels[y * turned.Width + x]));
                image.Dispose();
                image = turned;
                if (Upward.Count > 64)
                {
                    foreach (var old in Upward.Values)
                        old.Dispose();
                    Upward.Clear();
                }
                Upward[text] = image;
                return image;
            }
        }

        /// <summary>The font of the ticks and the axis titles.</summary>
        public static readonly Font Small = new(FontFamily.GenericSansSerif, 6f);

        /// <summary>The chart in ``area``, its left margin fitted to the y title and the tick labels.</summary>
        public static Chart Fitted(ParameterGoo.Content content, RectangleF area)
        {
            var first = ChartLayout.Build(content, area);
            var left = ChartLayout.LeftMargin(TicksWidth(first), TitleSize(first.YTitle).Height);
            return ChartLayout.Build(content, area, left: left);
        }

        public static float TicksWidth(Chart chart) => chart.YTicks.Select(t => GH_FontServer.StringWidth(t.Text, Small)).DefaultIfEmpty(0).Max();

        public static SizeF TitleSize(string title) => title.Length == 0 ? SizeF.Empty : GH_FontServer.MeasureString(title, Small);

        public static void Draw(Graphics g, Chart chart, RectangleF area)
        {
            var font = Small;
            using var bold = new Font(FontFamily.GenericSansSerif, 7f, FontStyle.Bold);
            g.FillRectangle(Brushes.White, area);
            g.DrawString(chart.Title, bold, Brushes.Black, area.Left + ChartLayout.Left, area.Top + 3);
            if (chart.Overall != null)
            {
                var size = g.MeasureString(chart.Overall, font);
                g.DrawString(chart.Overall, font, Brushes.DimGray, area.Right - ChartLayout.Right - size.Width, area.Top + 4);
            }
            var plot = chart.Plot;
            using (var grid = new Pen(Color.FromArgb(230, 230, 230)))
                foreach (var tick in chart.YTicks)
                {
                    g.DrawLine(grid, plot.Left, tick.Y, plot.Right, tick.Y);
                    var size = g.MeasureString(tick.Text, font);
                    g.DrawString(tick.Text, font, Brushes.DimGray, plot.Left - size.Width - ChartLayout.TickGap, tick.Y - size.Height / 2);
                }
            foreach (var tick in chart.XTicks)
            {
                g.DrawLine(Pens.Gray, tick.X, plot.Bottom, tick.X, plot.Bottom + 3);
                var size = g.MeasureString(tick.Text, font);
                g.DrawString(tick.Text, font, Brushes.DimGray, tick.X - size.Width / 2, plot.Bottom + 3);
            }
            g.DrawRectangle(Pens.Gray, plot.X, plot.Y, plot.Width, plot.Height);
            if (chart.ZeroY is float zero)
                g.DrawLine(Pens.Gray, plot.Left, zero, plot.Right, zero);
            var state = g.SmoothingMode;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            for (var c = 0; c < chart.Lines.Count; c++)
            {
                var color = ChannelColors[c % ChannelColors.Length];
                if (chart.Lines[c].Length > 1)
                {
                    using var pen = new Pen(color, chart.Dots ? 1.5f : 1);
                    g.DrawLines(pen, chart.Lines[c]);
                }
                if (chart.Dots)
                    using (var dot = new SolidBrush(color))
                        foreach (var point in chart.Lines[c])
                            g.FillEllipse(dot, point.X - DotRadius, point.Y - DotRadius, 2 * DotRadius, 2 * DotRadius);
            }
            g.SmoothingMode = state;
            var x = g.MeasureString(chart.XTitle, font);
            g.DrawString(chart.XTitle, font, Brushes.Black, plot.Left + plot.Width / 2 - x.Width / 2, area.Bottom - x.Height - 1);
            if (chart.YTitle.Length > 0)
                g.DrawImage(UpwardText(chart.YTitle, font), ChartLayout.YTitleBox(area, plot, TitleSize(chart.YTitle), TicksWidth(chart)));
            if (chart.Lines.Count > 1)
                for (var c = 0; c < chart.Lines.Count && c < chart.Channels.Count; c++)
                    using (var brush = new SolidBrush(ChannelColors[c % ChannelColors.Length]))
                        g.DrawString(chart.Channels[c], font, brush, plot.Right - 40, plot.Top + 2 + 8 * c);
        }

        /// <summary>The font of the compact chart's labels.</summary>
        public static readonly Font Tiny = new(FontFamily.GenericSansSerif, 5f);

        /// <summary>The compact chart in ``area``, its left margin fitted to its three value labels.</summary>
        public static Chart FittedCompact(ParameterGoo.Content content, RectangleF area)
        {
            var first = ChartLayout.Build(content, area, compact: true);
            var width = first.YTicks.Select(t => GH_FontServer.StringWidth(t.Text, Tiny)).DefaultIfEmpty(0).Max();
            return ChartLayout.Build(content, area, compact: true, left: 2 + width + ChartLayout.TickGap);
        }

        /// <summary>Deconstruct Parameter's strip: a line through a dot per band, the bands below, three values beside.</summary>
        public static void DrawCompact(Graphics g, Chart chart)
        {
            var plot = chart.Plot;
            using (var grid = new Pen(Color.FromArgb(230, 230, 230)))
                foreach (var tick in chart.YTicks)
                {
                    g.DrawLine(grid, plot.Left, tick.Y, plot.Right, tick.Y);
                    var size = g.MeasureString(tick.Text, Tiny);
                    g.DrawString(tick.Text, Tiny, Brushes.DimGray, plot.Left - size.Width - ChartLayout.TickGap + 2, tick.Y - size.Height / 2);
                }
            foreach (var tick in chart.XTicks)
            {
                var size = g.MeasureString(tick.Text, Tiny);
                g.DrawString(tick.Text, Tiny, Brushes.DimGray, tick.X - size.Width / 2, plot.Bottom + 1);
            }
            g.DrawRectangle(Pens.Gray, plot.X, plot.Y, plot.Width, plot.Height);
            var state = g.SmoothingMode;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            var color = ChannelColors[0];
            foreach (var line in chart.Lines)
            {
                if (line.Length > 1)
                    using (var pen = new Pen(color, 1.2f))
                        g.DrawLines(pen, line);
                using var dot = new SolidBrush(color);
                foreach (var point in line)
                    g.FillEllipse(dot, point.X - 1.6f, point.Y - 1.6f, 3.2f, 3.2f);
            }
            g.SmoothingMode = state;
        }
    }

    /// <summary>Plot 2D's body: one chart per parameter, stacked, resized by the corner grip, values in the tooltip.</summary>
    public sealed class Plot2DUi : ComponentUi
    {
        private const float Grip = 8;
        private const float MinWidth = 160;
        private readonly Dictionary<(ParameterGoo.Content, RectangleF), Chart> _cache = new();
        private bool _resizing;

        public Plot2DUi(Plot2DComponent owner) : base(owner) { }

        private Plot2DComponent Plot => (Plot2DComponent)Owner;

        private int Count => Math.Max(Plot.Charts.Count, 1);

        protected override float ExtraHeight => Plot2DLayout.BodyHeight(Plot.Charts.Count, Plot.ChartSize);

        protected override void Layout()
        {
            base.Layout();
            var b = Bounds;
            Bounds = new RectangleF(b.X, b.Y, Math.Max(b.Width, Plot.ChartSize.Width), b.Height);
        }

        private RectangleF Body => new(Bounds.X, Bounds.Bottom - ExtraHeight, Bounds.Width, ExtraHeight);

        private RectangleF Slot(int i) => Plot2DLayout.Slot(Body, i, Plot.ChartSize);

        private bool _saving;

        private Chart ChartOf(int i)
        {
            var area = Slot(i);
            var key = (Plot.Charts[i], area);
            if (!_cache.TryGetValue(key, out var chart))
            {
                if (_cache.Count > 64)
                    _cache.Clear();
                _cache[key] = chart = ChartPainter.Fitted(Plot.Charts[i], area);
            }
            return chart;
        }

        protected override void RenderExtra(GH_Canvas canvas, Graphics g)
        {
            if (Plot.Charts.Count == 0)
            {
                using var font = new Font(FontFamily.GenericSansSerif, 7f);
                g.FillRectangle(Brushes.White, Slot(0));
                g.DrawString("connect a parameter, or the time or frequency output of Analyse IR", font, Brushes.Gray, Slot(0));
            }
            for (var i = 0; i < Plot.Charts.Count; i++)
                ChartPainter.Draw(g, ChartOf(i), Slot(i));
            var save = Plot2DLayout.SaveButton(Body);
            using (var button = GH_Capsule.CreateTextCapsule(Rectangle.Round(save), Rectangle.Round(save),
                       Plot.Charts.Count == 0 ? GH_Palette.Hidden : _saving ? GH_Palette.Grey : GH_Palette.Black, "Save", 2, 0))
                button.Render(g, Selected, Owner.Locked, false);
            var body = Body;
            g.DrawLine(Pens.Gray, body.Right - Grip, body.Bottom - 2, body.Right - 2, body.Bottom - Grip);
            g.DrawLine(Pens.Gray, body.Right - Grip / 2, body.Bottom - 2, body.Right - 2, body.Bottom - Grip / 2);
        }

        protected override GH_ObjectResponse? RespondToExtraMouseDown(GH_Canvas sender, GH_CanvasMouseEvent e)
        {
            var body = Body;
            var p = e.CanvasLocation;
            if (Plot.Charts.Count > 0 && Plot2DLayout.SaveButton(body).Contains(p))
            {
                _saving = true;
                sender.Invalidate();
                return GH_ObjectResponse.Capture;
            }
            if (!body.Contains(p) || p.X < body.Right - Grip * 1.5f || p.Y < body.Bottom - Grip * 1.5f)
                return null;
            _resizing = true;
            return GH_ObjectResponse.Capture;
        }

        public override GH_ObjectResponse RespondToMouseMove(GH_Canvas sender, GH_CanvasMouseEvent e)
        {
            if (!_resizing)
                return base.RespondToMouseMove(sender, e);
            // the width follows the mouse, the height keeps the chart's aspect ratio: a resize does not distort the charts
            Plot.ChartSize = Plot2DLayout.Resize(Plot.ChartSize, e.CanvasLocation.X - Bounds.X, MinWidth);
            ExpireLayout();
            sender.Invalidate();
            return GH_ObjectResponse.Handled;
        }

        public override GH_ObjectResponse RespondToMouseUp(GH_Canvas sender, GH_CanvasMouseEvent e)
        {
            if (_saving)
            {
                _saving = false;
                sender.Invalidate();
                if (Plot2DLayout.SaveButton(Body).Contains(e.CanvasLocation))
                    SaveCharts();
                return GH_ObjectResponse.Release;
            }
            if (!_resizing)
                return base.RespondToMouseUp(sender, e);
            _resizing = false;
            return GH_ObjectResponse.Release;
        }

        private const int SaveScale = 3;

        /// <summary>Save the charts, stacked as on the body, as one PNG at three times their size, where a dialog asks.</summary>
        private void SaveCharts()
        {
            if (Plot.Charts.Count == 0)
                return;
            var dialog = new Eto.Forms.SaveFileDialog { Title = "Save plot", FileName = Plot2DLayout.FileName(Plot.Charts[0].Title) };
            dialog.Filters.Add(new Eto.Forms.FileFilter("PNG image", ".png"));
            if (dialog.ShowDialog(Rhino.UI.RhinoEtoApp.MainWindow) != Eto.Forms.DialogResult.Ok || string.IsNullOrEmpty(dialog.FileName))
                return;
            var path = dialog.FileName.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ? dialog.FileName : dialog.FileName + ".png";
            var size = Plot.ChartSize;
            var page = new RectangleF(0, 0, size.Width, Plot.Charts.Count * size.Height + Plot2DLayout.SaveRow);
            using var image = new Bitmap((int)Math.Ceiling(size.Width * SaveScale), (int)Math.Ceiling(Plot.Charts.Count * size.Height * SaveScale));
            using (var g = Graphics.FromImage(image))
            {
                g.Clear(Color.White);
                g.ScaleTransform(SaveScale, SaveScale);
                for (var i = 0; i < Plot.Charts.Count; i++)
                {
                    var area = Plot2DLayout.Slot(page, i, size);
                    ChartPainter.Draw(g, ChartPainter.Fitted(Plot.Charts[i], area), area);
                }
            }
            image.Save(path, System.Drawing.Imaging.ImageFormat.Png);
        }

        private (Chart chart, string? readout)? Under(PointF point)
        {
            for (var i = 0; i < Plot.Charts.Count; i++)
                if (Slot(i).Contains(point))
                {
                    var chart = ChartOf(i);
                    return (chart, ChartLayout.Readout(chart, point));
                }
            return null;
        }

        public override bool IsTooltipRegion(PointF canvasPoint) => Under(canvasPoint)?.readout != null || base.IsTooltipRegion(canvasPoint);

        public override void SetupTooltip(PointF canvasPoint, GH_TooltipDisplayEventArgs e)
        {
            if (Under(canvasPoint) is { readout: { } readout } under)
            {
                e.Title = under.chart.Title;
                e.Text = readout;
                e.Description = "";
                return;
            }
            base.SetupTooltip(canvasPoint, e);
        }
    }

    /// <summary>Deconstruct Parameter's body: its controls, and a compact chart when it holds one parameter per band.</summary>
    public sealed class ParameterStripUi : ComponentUi
    {
        private const float Height = ChartLayout.StripHeight;

        public ParameterStripUi(DeconstructParameterComponent owner) : base(owner) { }

        private DeconstructParameterComponent Component => (DeconstructParameterComponent)Owner;

        protected override float ExtraHeight => Component.Strip != null ? Height : 0;

        protected override void RenderExtra(GH_Canvas canvas, Graphics g)
        {
            if (Component.Strip is not { } p)
                return;
            var area = ExtraArea;
            using var font = new Font(FontFamily.GenericSansSerif, 6f);
            var header = ChartLayout.StripHeaderText(p);
            g.DrawString(header, font, Brushes.Black, area.Left + 3, area.Top + 1);
            var strip = ChartLayout.StripArea(area);
            g.FillRectangle(Brushes.White, strip);
            ChartPainter.DrawCompact(g, ChartPainter.FittedCompact(p, strip));
        }
    }
}
