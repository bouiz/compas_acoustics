using System.Drawing;
using System.Linq;
using CompasAcoustics.Pb;
using Grasshopper.Kernel;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>The Material Library's choice: a library material as it is, one with edited coefficients, or a custom one.</summary>
    public class MaterialChoiceTests
    {
        private static AcousticMaterialData Plaster()
        {
            var m = new AcousticMaterialData { Name = "acoustical_plaster_25mm", Category = "Plaster & Gypsum", DataOrigin = "pyroomacoustics" };
            m.AbsorptionCoefficients.Add(new[] { 0.17, 0.36, 0.66, 0.65, 0.62, 0.68, 0.68 });
            return m;
        }

        [Fact]
        public void ACustomMaterial_StartsWithNoAbsorptionOrScatteringInEveryBand()
        {
            var custom = MaterialChoice.NewCustom();
            Assert.True(custom.IsCustom);
            Assert.Equal(new double[MaterialChoice.Bands], custom.Absorption);
            Assert.Equal(new double[MaterialChoice.Bands], custom.Scattering);
            Assert.Equal(("custom", "Other", "custom"), (custom.Name, custom.Category, custom.Origin));
            Assert.True(custom.HasNoAbsorption);
        }

        [Fact]
        public void ALibraryMaterial_IsUsedAsItIs_UntilABandIsSet()
        {
            var plaster = MaterialChoice.Of(Plaster());
            Assert.True(plaster.AsInTheLibrary);
            Assert.Null(plaster.Scattering);
            var edited = plaster.WithBand(scattering: false, band: 2, value: 1.4);
            Assert.False(edited.AsInTheLibrary);
            Assert.Equal(1.0, edited.Absorption[2]);
            Assert.Equal(0.66, plaster.Absorption[2]);
            Assert.Equal("pyroomacoustics, modified", edited.Origin);
            // a material without scattering gets the simulations' 0.05 in the other bands once one is set
            var scattered = plaster.WithBand(scattering: true, band: 0, value: 0.3);
            Assert.Equal(new[] { 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05 }, scattered.Scattering);
        }

        [Fact]
        public void AChoice_IsSavedAndReadBack()
        {
            var choice = MaterialChoice.NewCustom().WithBand(false, 3, 0.5) with { Name = "panel", Category = "Absorbers" };
            var back = MaterialChoice.Read(choice.Write())!;
            Assert.Equal((choice.Library, choice.Name, choice.Category, choice.Origin), (back.Library, back.Name, back.Category, back.Origin));
            Assert.Equal(choice.Absorption, back.Absorption);
            Assert.Equal(choice.Scattering, back.Scattering);
            Assert.Equal(choice.Edited, back.Edited);
            Assert.Null(MaterialChoice.Read("not json"));
            // a document saved with only the library's name
            Assert.True(MaterialChoice.Named("gypsum_board").AsInTheLibrary);
        }

        [Fact]
        public void TheNotes_SayWhenScatteringIsTheDefault_AndWhenACustomMaterialHasNoAbsorption()
        {
            var plaster = Plaster();
            Assert.Equal(("scattering 0.05 (default)", null), MaterialChoice.Notes(MaterialChoice.Of(plaster), absorptionWired: false, plaster));
            var custom = MaterialChoice.NewCustom();
            var result = new AcousticMaterialData { Name = "custom" };
            result.AbsorptionCoefficients.Add(new double[7]);
            result.ScatteringCoefficients.Add(new double[7]);
            var (message, warning) = MaterialChoice.Notes(custom, absorptionWired: false, result);
            Assert.Null(message);
            Assert.Contains("absorption", warning);
            Assert.Null(MaterialChoice.Notes(custom, absorptionWired: true, result).warning);
            Assert.Null(MaterialChoice.Notes(custom.WithBand(false, 0, 0.2), absorptionWired: false, result).warning);
        }

        [Fact]
        public void TheChart_MapsAPointToABandAndAValue()
        {
            var plot = new RectangleF(10, 20, 70, 100);
            Assert.Equal((0, 1.0), MaterialChart.Hit(plot, 7, new PointF(11, 10)));
            Assert.Equal((3, 0.5), MaterialChart.Hit(plot, 7, new PointF(45, 70)));
            Assert.Equal((6, 0.0), MaterialChart.Hit(plot, 7, new PointF(500, 300)));
            var (absorption, scattering) = MaterialChart.PairedBars(new[] { 0.5, 1.0 }, new[] { 0.25, 0.0 }, new RectangleF(0, 0, 40, 20));
            Assert.Equal((2, 2), (absorption.Length, scattering.Length));
            Assert.True(absorption[0].Right <= scattering[0].Left && scattering[0].Right <= absorption[1].Left);
            Assert.Equal((10f, 5f), (absorption[0].Height, scattering[0].Height));
            Assert.Empty(MaterialChart.PairedBars(new[] { 0.5 }, null, new RectangleF(0, 0, 40, 20)).scattering);
        }

        [Fact]
        public void TheCoefficientChart_HasTicksEveryTenthsPair() =>
            Assert.Equal(new[] { "0.0", "0.2", "0.4", "0.6", "0.8", "1.0" }, MaterialChart.Ticks.Select(t => t.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture)));

        [Fact]
        public void TheStrip_IsALinePerCoefficient_WithTheBandsBelowAndZeroAndOneBeside()
        {
            var area = new RectangleF(0, 0, 140, 50);
            var strip = MaterialChart.Strip(new[] { 0.0, 0.5, 1.0 }, new[] { 0.2, 0.2, 0.2 }, area, labelWidth: 6);
            Assert.True(area.Contains(strip.Plot));
            Assert.Equal(new[] { strip.Plot.Bottom, strip.Plot.Top + strip.Plot.Height / 2, strip.Plot.Top }, strip.Absorption.Select(p => p.Y));
            Assert.Equal(3, strip.Scattering.Length);
            var step = strip.Plot.Width / 3;
            Assert.Equal(strip.Plot.Left + step / 2, strip.Absorption[0].X, 3);
            Assert.Equal(new[] { "125", "250", "500" }, strip.Bands.Select(b => b.text));
            Assert.Equal(new[] { "0", "1" }, strip.Values.Select(v => v.text));
            Assert.Equal((strip.Plot.Bottom, strip.Plot.Top), (strip.Values[0].y, strip.Values[1].y));
            Assert.Empty(MaterialChart.Strip(new[] { 0.5 }, null, area, 6).Scattering);
        }

        [Fact]
        public void TheMaterialStrip_HasDeconstructParametersHeightAndArea()
        {
            // both bodies draw their chart in ChartLayout.StripArea of a ChartLayout.StripHeight body
            var body = new RectangleF(10, 100, 120, ChartLayout.StripHeight);
            var chart = ChartLayout.StripArea(body);
            Assert.Equal(ChartLayout.StripHeight - ChartLayout.StripHeader - 2, chart.Height, 3);
            Assert.True(body.Contains(chart) && chart.Top == body.Top + ChartLayout.StripHeader);
        }

        [Theory]
        [InlineData("0.3", 0.3)]
        [InlineData("0,3", 0.3)]
        [InlineData(".35", 0.35)]
        [InlineData(" 1 ", 1.0)]
        [InlineData("1.7", 1.0)]
        [InlineData("-0.2", 0.0)]
        [InlineData("0.333", 0.33)]
        public void ATypedCoefficient_IsReadWithinZeroAndOne(string text, double value) => Assert.Equal(value, MaterialChart.Parse(text));

        [Theory]
        [InlineData("")]
        [InlineData("abc")]
        [InlineData("0.3.1")]
        public void AnythingElse_IsNoValue(string text) => Assert.Null(MaterialChart.Parse(text));

        [Fact]
        public void ABoxShowsItsValueToTwoDecimals() =>
            Assert.Equal(new[] { "0.00", "0.35", "1.00" }, new[] { 0.0, 0.35, 1.0 }.Select(MaterialChart.Format));

        [Fact]
        public void TheComponent_TakesAbsorptionAndScatteringLists()
        {
            var inputs = ComponentSpec.Load("CA_Material_Library").Inputs;
            Assert.Equal(new[] { "absorption", "scattering" }, inputs.Select(p => p.Name));
            Assert.All(inputs, p => Assert.True(p.Optional && p.Access == GH_ParamAccess.list));
        }
    }
}
