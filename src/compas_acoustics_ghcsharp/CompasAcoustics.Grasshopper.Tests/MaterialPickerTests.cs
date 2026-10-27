using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using CompasAcoustics.Pb;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class MaterialPickerTests
    {
        private static readonly List<(string category, IReadOnlyList<string> names)> Library = new()
        {
            ("Wood", new[] { "Oak floor", "Plywood panel" }),
            ("Glass", new[] { "Window pane" }),
            ("Fabric", new[] { "Heavy curtain", "Wood wool board" }),
        };

        [Fact]
        public void Filter_IsCaseInsensitiveAndDropsEmptyCategories()
        {
            var found = MaterialSearch.Filter(Library, "WOOD");
            Assert.Equal(new[] { "Wood", "Fabric" }, found.Select(g => g.category));
            Assert.Equal(new[] { "Plywood panel" }, found[0].names);
            Assert.Equal(new[] { "Wood wool board" }, found[1].names);
            var withO = MaterialSearch.Filter(Library, "o");
            Assert.Equal(new[] { "Wood", "Glass", "Fabric" }, withO.Select(g => g.category));
            Assert.Equal(new[] { "Wood wool board" }, withO[2].names);
        }

        [Fact]
        public void Filter_EmptyQueryKeepsAll()
        {
            Assert.Equal(Library.Select(g => g.category), MaterialSearch.Filter(Library, "").Select(g => g.category));
            Assert.Equal(Library.Select(g => g.category), MaterialSearch.Filter(Library, "  ").Select(g => g.category));
        }

        [Fact]
        public void MaterialGoo_ToString_HasMeanAbsorption()
        {
            var material = new AcousticMaterialData { Name = "Oak floor", Category = "Wood" };
            material.AbsorptionCoefficients.Add(new[] { 0.1, 0.2, 0.3, 0.68 });
            Assert.Equal("Oak floor, Wood, mean absorption 0.32", new MaterialGoo(material).ToString());
            Assert.Equal("Null Material", new MaterialGoo().ToString());
        }
    }
}
