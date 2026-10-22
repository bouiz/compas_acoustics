using System;
using System.Collections.Generic;
using System.Linq;
using CompasAcoustics.Pb;
using Grasshopper.Kernel.Types;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class DeconstructParameterTests
    {
        private static ParameterGoo.Content P(string name) =>
            new(name, name + " title", "Frequency (Hz)", "Time (s)", "bands", new[] { 125.0, 250.0 }, new[] { new[] { 1.0, 2.0 } }, 1.5, Array.Empty<string>());

        [Fact]
        public void Select_AllGivesEveryParameterNotSingle()
        {
            var warnings = new List<string>();
            var selection = DeconstructParameter.Select(new[] { P("T30"), P("C80") }, null, warnings.Add);
            Assert.Equal(new[] { "T30", "C80" }, selection.Parameters.Select(p => p.Name).ToArray());
            Assert.False(selection.Single);
            Assert.False(DeconstructParameter.Select(new[] { P("T30"), P("C80") }, "all", warnings.Add).Single);
            Assert.Empty(warnings);
        }

        [Fact]
        public void Select_OneParameterAloneIsSingle()
        {
            Assert.True(DeconstructParameter.Select(new[] { P("T30") }, null, _ => { }).Single);
        }

        [Fact]
        public void Select_OneNameGivesASingle()
        {
            var selection = DeconstructParameter.Select(new[] { P("T30"), P("C80") }, "C80", _ => { });
            Assert.True(selection.Single);
            Assert.Equal("C80", selection.Parameters.Single().Name);
        }

        [Fact]
        public void Select_MissingNameWarnsAndGivesNone()
        {
            var warnings = new List<string>();
            Assert.Empty(DeconstructParameter.Select(new[] { P("C80") }, "T30", warnings.Add).Parameters);
            Assert.Equal(new[] { "The parameters connected do not include 'T30'." }, warnings);
        }

        [Fact]
        public void Options_AreAllThenEveryParameterName()
        {
            Assert.Equal(new[] { "all", "EDT", "T20", "T30", "C50", "C80", "D50", "DRR", "Ts", "STI", "G" }, DeconstructParameter.Options);
        }
    }

    public class SoundChannelsTests
    {
        private static int NoFile(string path) => throw new InvalidOperationException("no file expected");

        [Fact]
        public void Channels_OfAResponseAndOfAudioAreTheirShapes()
        {
            var ir = new ImpulseResponseGoo(new ImpulseResponseData { Shape = { 10, 4 } });
            Assert.Equal(4, SoundChannels.Of(ir, NoFile));
            Assert.Equal(1, SoundChannels.Of(new ImpulseResponseGoo(new ImpulseResponseData { Shape = { 10 } }), NoFile));
            Assert.Equal(2, SoundChannels.Of(new AudioGoo(new AudioData { Shape = { 3, 2 } }), NoFile));
        }

        [Fact]
        public void Channels_OfAPathOrASourceReadTheFile()
        {
            Assert.Equal(3, SoundChannels.Of(new GH_String("/a.flac"), path => path == "/a.flac" ? 3 : 0));
            Assert.Equal(1, SoundChannels.Of(new SourceGoo(new SourceData { SoundFile = "/v.wav" }), path => path == "/v.wav" ? 1 : 0));
        }

        [Fact]
        public void Channels_OfASourceWithoutASoundFileIsAnError()
        {
            var error = Assert.Throws<ArgumentException>(() => SoundChannels.Of(new SourceGoo(new SourceData()), NoFile));
            Assert.Equal("The source has no sound file.", error.Message);
        }

        [Fact]
        public void Channels_OfSomethingElseIsAnError()
        {
            var error = Assert.Throws<ArgumentException>(() => SoundChannels.Of(new GH_Integer(3), NoFile));
            Assert.Equal("Cannot read the channels of a Integer.", error.Message);
        }
    }
}
