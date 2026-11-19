using System;
using System.Linq;
using System.Drawing;
using Grasshopper.Kernel;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class TicksTests
    {
        [Fact]
        public void NiceStep_OfARangeOf7_Is1_AndOf12_Is2() { Assert.Equal(1.0, Ticks.NiceStep(7.0)); Assert.Equal(2.0, Ticks.NiceStep(12.0)); }

        [Fact]
        public void Linear_From0To0_37_Is0To0_4ByTenths() =>
            Assert.Equal(new[] { 0.0, 0.1, 0.2, 0.3, 0.4 }, Ticks.Linear(0, 0.37));

        [Fact]
        public void Linear_OfAnEmptyRange_IsItsValue() => Assert.Equal(new[] { 3.0 }, Ticks.Linear(3, 3));

        [Fact]
        public void ForValues_DropsTheAxisAStepBelowAMinimumThatLiesOnATick()
        {
            Assert.Equal(new[] { 0.5, 1.0, 1.5, 2.0, 2.5 }, Ticks.ForValues(new[] { 0.98, 1.17, 1.69, 1.94, 2.16, 1.77, 1.15 }));
            Assert.Equal(new[] { -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0 }, Ticks.ForValues(new[] { -1.0, 2.0 }));
        }

        [Fact]
        public void MinMaxIndices_KeepsEveryPointOfAShortCurve() =>
            Assert.Equal(Enumerable.Range(0, 10).ToArray(), Ticks.MinMaxIndices(Enumerable.Range(0, 10).Select(i => i / 9.0).ToArray(), new double[10], 1000));

        [Fact]
        public void MinMaxIndices_KeepsTheExtremesOfEachBinInOrder()
        {
            var positions = Enumerable.Range(0, 100).Select(i => i / 100.0).ToArray();
            var values = positions.Select((p, i) => i == 37 ? 9.0 : Math.Sin(i)).ToArray();
            var kept = Ticks.MinMaxIndices(positions, values, 10);
            Assert.Contains(37, kept);
            Assert.True(kept.Length <= 20);
            Assert.Equal(kept.OrderBy(i => i).ToArray(), kept);
        }

        [Theory]
        [InlineData(0.98, "0.98")]
        [InlineData(1.0, "1.0")]
        [InlineData(0.1 + 0.2, "0.30000000000000004")]
        [InlineData(1e-05, "1e-05")]
        [InlineData(-2.5, "-2.5")]
        public void Repr_IsPythonsStr(double value, string expected) => Assert.Equal(expected, PythonFormat.Repr(value));
    }

    public class ChartLayoutTests
    {
        private static readonly RectangleF Area = new(0, 0, 300, 160);

        private static ParameterGoo.Content Bands(string name, double[] bands, double[] values, double? overall, string yUnit = "Time (s)") =>
            new(name, name, "Frequency (Hz)", yUnit, "bands", bands, new[] { values }, overall, Array.Empty<string>());

        private static readonly double[] Octaves = { 125, 250, 500, 1000, 2000, 4000, 8000 };
        private static readonly double[] T30 = { 0.98, 1.17, 1.69, 1.94, 2.16, 1.77, 1.15 };

        [Fact]
        public void BandsAreDotsJoinedByALine_HalfABandFromEachSide()
        {
            var chart = ChartLayout.Build(Bands("T30", Octaves, T30, 1.81), Area);
            Assert.True(Area.Contains(chart.Plot));
            // the y title reads upwards left of the y axis, inside the area, centred on the plot, clear of the tick labels
            var title = ChartLayout.YTitleBox(Area, chart.Plot, new SizeF(60, 10), 16);
            Assert.True(Area.Contains(title));
            Assert.Equal((10f, 60f), (title.Width, title.Height));
            Assert.Equal(chart.Plot.Top + chart.Plot.Height / 2, title.Top + title.Height / 2, 3);
            // next to the tick labels, as the x title is below its labels
            Assert.Equal(chart.Plot.Left - 16 - ChartLayout.TickGap - ChartLayout.TitleGap, title.Right, 3);
            // a title longer than the area is cut to it; wide tick labels push it to the area's edge, not out of it
            Assert.True(Area.Contains(ChartLayout.YTitleBox(Area, chart.Plot, new SizeF(400, 10), 16)));
            Assert.True(Area.Contains(ChartLayout.YTitleBox(Area, chart.Plot, new SizeF(60, 10), 80)));
            var step = chart.Plot.Width / 7;
            Assert.Equal(Enumerable.Range(0, 7).Select(i => (double)(chart.Plot.Left + step * (i + 0.5f))), chart.Bars.Select(b => (double)(b.Bar.Left + b.Bar.Width / 2)), new Close(1e-3));
            Assert.Equal(new[] { "125", "250", "500", "1000", "2000", "4000", "8000" }, chart.Bars.Select(b => b.Label));
            // a dot per band at its value, joined by one line
            var line = Assert.Single(chart.Lines);
            Assert.Equal(7, line.Length);
            Assert.Equal(chart.Bars.Select(b => (double)(b.Bar.Left + b.Bar.Width / 2)), line.Select(p => (double)p.X), new Close(1e-3));
            Assert.Equal(T30.Select(v => (double)ChartLayout.Y(chart, v)), line.Select(p => (double)p.Y), new Close(1e-3));
            Assert.True(chart.Dots);
            Assert.Equal("T30", chart.Title);
            Assert.Equal("overall 1.81", chart.Overall);
            // a reverberation time starts at 0; a bar reaches its value from the bottom of the plot
            Assert.Equal(new[] { "0.0", "0.5", "1.0", "1.5", "2.0", "2.5" }, chart.YTicks.Select(t => t.Text));
            Assert.Equal(0.0, chart.YLow);
            foreach (var name in new[] { "EDT", "T20" })
                Assert.Equal(0.0, ChartLayout.Build(Bands(name, Octaves, T30, null), Area).YLow);
            // another parameter keeps the range around its values
            Assert.Equal(0.5, ChartLayout.Build(Bands("Ts", Octaves, T30, null), Area).YLow);
            var y = ChartLayout.Y(chart, 1.94);
            Assert.Equal(y, chart.Bars[3].Bar.Top, 3);
            Assert.Equal(chart.Plot.Bottom, chart.Bars[3].Bar.Bottom, 3);
        }

        [Fact]
        public void TheLeftMargin_IsTheRightOne_PlusTheYTitleAndTheTickLabels()
        {
            var margin = ChartLayout.LeftMargin(ticksWidth: 14, titleHeight: 9);
            Assert.Equal(ChartLayout.Right + 9 + ChartLayout.TitleGap + 14 + ChartLayout.TickGap, margin, 3);
            var chart = ChartLayout.Build(Bands("T30", Octaves, T30, 1.81), Area, left: margin);
            Assert.Equal(Area.Left + margin, chart.Plot.Left, 3);
            // the y title then sits as far from the panel's left edge as the plot from its right edge
            Assert.Equal(Area.Left + ChartLayout.Right, ChartLayout.YTitleBox(Area, chart.Plot, new SizeF(60, 9), 14).Left, 3);
            Assert.Equal(Area.Right - ChartLayout.Right, chart.Plot.Right, 3);
        }

        [Fact]
        public void TheBody_StacksTheChartsAboveASaveButton()
        {
            var size = new SizeF(360, 200);
            var body = new RectangleF(0, 50, 360, Plot2DLayout.BodyHeight(2, size));
            Assert.Equal(2 * 200 + Plot2DLayout.SaveRow, body.Height, 3);
            var second = Plot2DLayout.Slot(body, 1, size);
            Assert.True(second.Top >= 50 + 200 && second.Bottom <= 50 + 400);
            var save = Plot2DLayout.SaveButton(body);
            Assert.True(save.Top >= second.Bottom && save.Bottom <= body.Bottom && save.Left < body.Left + 60);
            Assert.Equal(1 * 200 + Plot2DLayout.SaveRow, Plot2DLayout.BodyHeight(0, size), 3);
        }

        [Theory]
        [InlineData("Reverberation Time (T30)", "Reverberation Time (T30).png")]
        [InlineData("Impulse Response: left/right", "Impulse Response left right.png")]
        [InlineData("", "plot.png")]
        public void ASavedPlot_IsNamedAfterItsTitle(string title, string file) => Assert.Equal(file, Plot2DLayout.FileName(title));

        [Fact]
        public void Resizing_KeepsTheChartsAspectRatio()
        {
            Assert.Equal(new SizeF(540, 300), Plot2DLayout.Resize(new SizeF(360, 200), 540, minWidth: 160));
            var smallest = Plot2DLayout.Resize(new SizeF(360, 200), 50, minWidth: 160);
            Assert.Equal(160f, smallest.Width);
            Assert.Equal(160f * 200 / 360, smallest.Height, 3);
        }

        [Fact]
        public void STI_RunsFromZeroToOne()
        {
            var sti = new ParameterGoo.Content("STI", "Speech Transmission Index (STI)", "Frequency (Hz)", "STI", "bands", Octaves,
                new[] { new[] { 0.55, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7 } }, 0.63, Array.Empty<string>());
            var chart = ChartLayout.Build(sti, Area);
            Assert.Equal((0.0, 1.0), (chart.YLow, chart.YHigh));
            Assert.Equal(chart.Plot.Bottom, ChartLayout.Y(chart, 0), 3);
        }

        [Theory]
        [InlineData("LF")]
        [InlineData("LFC")]
        public void LateralFractions_RunFromZeroToOne(string name)
        {
            var lf = new ParameterGoo.Content(name, name, "Frequency (Hz)", "Fraction (0-1)", "bands", Octaves,
                new[] { new[] { 0.15, 0.18, 0.2, 0.22, 0.2, 0.19, 0.18 } }, 0.19, Array.Empty<string>());
            var chart = ChartLayout.Build(lf, Area);
            Assert.Equal((0.0, 1.0), (chart.YLow, chart.YHigh));
        }

        [Fact]
        public void TheStripsHeader_IsOnlyTheOverallValue()
        {
            Assert.Equal("overall 0.88", ChartLayout.StripHeaderText(Bands("T30", Octaves, T30, 0.88)));
            Assert.Equal("", ChartLayout.StripHeaderText(Bands("T30", Octaves, T30, null)));
        }

        [Fact]
        public void TheCompactChart_IsALineWithBandsBelowAndThreeValuesBeside()
        {
            var strip = new RectangleF(0, 0, 160, 50);
            var chart = ChartLayout.Build(Bands("T30", Octaves, T30, 1.81), strip, compact: true, left: 20);
            Assert.Equal(new[] { "0", "1.25", "2.5" }, chart.YTicks.Select(t => t.Text));
            Assert.Equal(new[] { "125", "250", "500", "1000", "2000", "4000", "8000" }, chart.XTicks.Select(t => t.Text));
            Assert.Equal(7, Assert.Single(chart.Lines).Length);
            Assert.True(chart.Dots);
            Assert.True(strip.Contains(chart.Plot) && chart.Plot.Bottom + ChartLayout.CompactBottom <= strip.Bottom + 0.01f);
            var sti = new ParameterGoo.Content("STI", "STI", "Frequency (Hz)", "STI", "bands", Octaves,
                new[] { new[] { 0.55, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7 } }, 0.63, Array.Empty<string>());
            Assert.Equal(new[] { "0", "0.5", "1" }, ChartLayout.Build(sti, strip, compact: true, left: 20).YTicks.Select(t => t.Text));
        }

        [Fact]
        public void TurningUpward_PutsTheTextsStartAtTheBottom()
        {
            // 3 x 2 pixels, a b c / d e f: turned a quarter counter-clockwise, 2 x 3: c f / b e / a d
            var turned = ChartLayout.TurnUpward(3, 2, new[] { 1, 2, 3, 4, 5, 6 });
            Assert.Equal(new[] { 3, 6, 2, 5, 1, 4 }, turned);
        }

        [Fact]
        public void ATimeResponse_IsACurvePerChannel_SymmetricAboutZero()
        {
            var random = new Random(0);
            double Normal() => Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
            var times = Enumerable.Range(0, 96000).Select(i => i / 48000.0).ToArray();
            var left = times.Select(_ => Normal()).ToArray();
            left[12345] = 9.0;
            var right = times.Select(_ => 0.5 * Normal()).ToArray();
            var curve = new ParameterGoo.Content("IR", "Impulse Response (left, right)", "Time (s)", "Amplitude", "linear", times, new[] { left, right }, null,
                new[] { "left", "right" });
            var chart = ChartLayout.Build(curve, Area);
            Assert.Equal(2, chart.Lines.Count);
            Assert.Empty(chart.Bars);
            Assert.False(chart.Dots);
            Assert.All(chart.Lines, l => Assert.True(l.Length <= 2 * (int)chart.Plot.Width + 2));
            Assert.Equal(new[] { "0", "0.5", "1", "1.5", "2" }, chart.XTicks.Select(t => t.Text));
            Assert.Equal(chart.Plot.Right, chart.XTicks[^1].X, 3);
            // the decimated curve keeps its peak; an amplitude is symmetric about zero, the zero line in the middle
            Assert.Equal(ChartLayout.Y(chart, 9.0), chart.Lines[0].Min(p => p.Y), 3);
            var ticks = chart.YTicks.Select(t => double.Parse(t.Text, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
            Assert.Equal(Enumerable.Reverse(ticks).Select(t => -t).ToArray(), ticks);
            Assert.Equal(chart.Plot.Top + chart.Plot.Height / 2, chart.ZeroY!.Value, 3);
        }

        [Fact]
        public void ALevelInDb_RunsFromItsFloorAtTheBottomToItsPeakAtTheTop()
        {
            var random = new Random(0);
            var raw = Enumerable.Range(0, 4800).Select(i => (random.NextDouble() - 0.5) * Math.Exp(-i / 500.0)).ToArray();
            var peak = raw.Max(Math.Abs);
            var level = raw.Select(v => 20 * Math.Log10(Math.Max(Math.Abs(v) / peak, 1e-5))).ToArray();
            var curve = new ParameterGoo.Content("IR", "Impulse Response", "Time (s)", "Level (dB)", "linear",
                Enumerable.Range(0, 4800).Select(i => i / 48000.0).ToArray(), new[] { level }, null, Array.Empty<string>());
            var chart = ChartLayout.Build(curve, Area);
            Assert.Equal(new[] { "-100", "-80", "-60", "-40", "-20", "0" }, chart.YTicks.Select(t => t.Text));
            var line = Assert.Single(chart.Lines);
            Assert.Equal(chart.Plot.Top, line.Min(p => p.Y), 3);
            Assert.Equal(chart.Plot.Bottom, line.Max(p => p.Y), 3);
            Assert.Null(chart.ZeroY);
        }

        [Fact]
        public void AFrequencyResponse_IsOnALogAxis()
        {
            var frequencies = Enumerable.Range(0, 500).Select(i => 20 * Math.Pow(1000, i / 499.0)).ToArray();
            var curve = new ParameterGoo.Content("FR", "Frequency Response, 1/3 octave", "Frequency (Hz)", "Level (dB)", "log", frequencies,
                new[] { frequencies.Select(f => -Math.Log10(f)).ToArray() }, null, Array.Empty<string>());
            var chart = ChartLayout.Build(curve, Area);
            Assert.Equal(new[] { "50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000" }, chart.XTicks.Select(t => t.Text));
            Assert.Equal(chart.Plot.Left + chart.Plot.Width * Math.Log10(1000.0 / 20) / 3, chart.XTicks[4].X, 3);
            var line = Assert.Single(chart.Lines);
            Assert.Equal(chart.Plot.Left, line[0].X, 3);
            Assert.Equal(chart.Plot.Right, line[^1].X, 3);
            Assert.Equal("Frequency (Hz)", chart.XTitle);
            Assert.Equal("Level (dB)", chart.YTitle);
        }

        [Fact]
        public void Readout_NamesTheNearestBar()
        {
            var chart = ChartLayout.Build(Bands("T30", Octaves, T30, 1.81), Area);
            var bar = chart.Bars[3].Bar;
            Assert.Equal("1000 Hz: 1.94", ChartLayout.Readout(chart, new PointF(bar.Left + bar.Width / 2 + 2, chart.Plot.Top + 5)));
            Assert.Null(ChartLayout.Readout(chart, new PointF(chart.Plot.Left - 20, chart.Plot.Top + 5)));
        }

        [Fact]
        public void Readout_OfACurve_GivesItsPointUnderTheMouse()
        {
            var times = Enumerable.Range(0, 11).Select(i => i / 10.0).ToArray();
            var curve = new ParameterGoo.Content("IR", "Impulse Response", "Time (s)", "Amplitude", "linear", times, new[] { times.Select(t => 2 * t).ToArray() },
                null, Array.Empty<string>());
            var chart = ChartLayout.Build(curve, Area);
            var x = chart.Plot.Left + chart.Plot.Width * 0.5f;
            Assert.Equal("0.5 Time (s), 1 Amplitude", ChartLayout.Readout(chart, new PointF(x + 1, chart.Plot.Top)));
        }

        [Fact]
        public void Plot2D_OnlyTakesData()
        {
            var plot = ComponentSpec.Load("CA_Plot_2D");
            Assert.Equal(new[] { "data" }, plot.Inputs.Select(p => p.Name));
            Assert.Equal(GH_ParamAccess.list, plot.Inputs[0].Access);
            Assert.Empty(plot.Outputs);
        }

        private sealed class Close : System.Collections.Generic.IEqualityComparer<double>
        {
            private readonly double _tolerance;
            public Close(double tolerance = 1e-9) => _tolerance = tolerance;
            public bool Equals(double a, double b) => Math.Abs(a - b) < _tolerance;
            public int GetHashCode(double v) => 0;
        }
    }
}
