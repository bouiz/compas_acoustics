using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CompasAcoustics.Pb;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class AuraliseGroupsTests
    {
        private static ImpulseResponseGoo Ir(string name) => new(new ImpulseResponseData { Name = name, Shape = { 1 }, Samples = { 1 } });

        private static GH_Structure<ImpulseResponseGoo> Irs(params (GH_Path path, string[] names)[] branches)
        {
            var tree = new GH_Structure<ImpulseResponseGoo>();
            foreach (var (path, names) in branches)
                foreach (var name in names)
                    tree.Append(Ir(name), path);
            return tree;
        }

        [Fact]
        public void FromTrees_PairsEachBranchWithItsSounds()
        {
            var irs = Irs((new GH_Path(0, 1), new[] { "a", "b" }), (new GH_Path(1), new[] { "c" }));
            var sounds = new GH_Structure<IGH_Goo>();
            sounds.Append(new GH_String("/a.wav"), new GH_Path(0, 1));
            sounds.Append(new GH_String("/b.wav"), new GH_Path(0, 1));
            sounds.Append(new GH_String("/c.wav"), new GH_Path(1));
            var groups = AuraliseGroups.FromTrees(irs, sounds, out var paths);
            Assert.Equal(2, groups.Count);
            Assert.Equal("{0;1}", groups[0]["branch"]);
            Assert.Equal(new object?[] { "/a.wav", "/b.wav" }, ((IEnumerable<object?>)groups[0]["sounds"]!).ToArray());
            Assert.Equal(new[] { "a", "b" }, ((IEnumerable<object?>)groups[0]["irs"]!).Cast<ImpulseResponseData>().Select(i => i.Name).ToArray());
            Assert.Equal(new GH_Path(1), paths["{1}"]);
        }

        [Fact]
        public void FromTrees_WithoutSounds_GivesNullSounds()
        {
            var groups = AuraliseGroups.FromTrees(Irs((new GH_Path(0), new[] { "a", "b" })), null, out _);
            Assert.Equal(new object?[] { null, null }, ((IEnumerable<object?>)groups.Single()["sounds"]!).ToArray());
            groups = AuraliseGroups.FromTrees(Irs((new GH_Path(0), new[] { "a" })), new GH_Structure<IGH_Goo>(), out _);
            Assert.Equal(new object?[] { null }, ((IEnumerable<object?>)groups.Single()["sounds"]!).ToArray());
        }

        [Fact]
        public void FromTrees_BranchCountMismatch_Throws()
        {
            var sounds = new GH_Structure<IGH_Goo>();
            sounds.Append(new GH_String("/a.wav"), new GH_Path(0));
            sounds.Append(new GH_String("/b.wav"), new GH_Path(1));
            var error = Assert.Throws<ArgumentException>(() => AuraliseGroups.FromTrees(Irs((new GH_Path(0), new[] { "a" })), sounds, out _));
            Assert.Equal("DataTree branch count mismatch.", error.Message);
        }

        [Fact]
        public void FromTrees_BranchLengthMismatch_Throws()
        {
            var sounds = new GH_Structure<IGH_Goo>();
            sounds.Append(new GH_String("/a.wav"), new GH_Path(0));
            var error = Assert.Throws<ArgumentException>(() => AuraliseGroups.FromTrees(Irs((new GH_Path(0), new[] { "a", "b" })), sounds, out _));
            Assert.Equal("DataTree branch length mismatch.", error.Message);
        }

        [Fact]
        public void FromTrees_TakesASourcesSoundFile()
        {
            var sounds = new GH_Structure<IGH_Goo>();
            sounds.Append(new SourceGoo(new SourceData { SoundFile = "/s.wav" }), new GH_Path(0));
            var groups = AuraliseGroups.FromTrees(Irs((new GH_Path(0), new[] { "a" })), sounds, out _);
            Assert.Equal(new object?[] { "/s.wav" }, ((IEnumerable<object?>)groups.Single()["sounds"]!).ToArray());
        }

        [Fact]
        public void Message_NamesFormatsLevelAndPeak()
        {
            Assert.Equal("binaural + mono, relative, peak -1.0 dBFS", AuraliseGroups.Message(new[] { "mono", "binaural", "mono" }, "relative", -1.0));
            Assert.Equal("Output peaks at +2.5 dBFS and will clip on playback. Use level 'relative' or 'individual'.", AuraliseGroups.ClipWarning(2.5));
        }

        [Theory]
        [InlineData("/renders/take.wav", "/renders", "take")]
        [InlineData("/renders/take", "/renders", "take")]
        [InlineData("/renders/take.WAV", "/renders", "take")]
        public void ASavedFile_GivesTheFolderAndTheNameOfTheSignals(string path, string folder, string name) =>
            Assert.Equal((folder.Replace('/', Path.DirectorySeparatorChar), name), AuraliseGroups.SaveTarget(path));

        [Fact]
        public void Auralise_SavesWithAButton()
        {
            Assert.Equal(new[] { "irs", "anechoic_sounds", "mix_sources" }, ComponentSpec.Load("CA_Auralise").Inputs.Select(p => p.Name));
            var auralise = new AuraliseComponent();
            Assert.Contains(auralise.Controls, c => c is ButtonControl { Port: "save", Label: "Save" });
            Assert.True(auralise.HandlesButton("save"));
            Assert.False(auralise.HandlesButton("run"));
        }
    }
}
