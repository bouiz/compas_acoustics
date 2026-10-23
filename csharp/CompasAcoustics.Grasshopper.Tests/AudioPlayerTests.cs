using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using CompasAcoustics.Pb;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class WaveformTests
    {
        [Fact]
        public void Envelope_HasMinAndMaxPerColumnPerChannel()
        {
            // 1000 stereo frames: left ramps 0..999, right is minus that
            var samples = Enumerable.Range(0, 1000).SelectMany(i => new double[] { i, -i }).ToArray();
            var envelope = Waveform.Envelope(samples, 2, 10);
            Assert.Equal(2, envelope.Length);
            Assert.Equal(10, envelope[0].Length);
            Assert.Equal((300f, 399f), envelope[0][3]);
            Assert.Equal((-399f, -300f), envelope[1][3]);
        }

        [Fact]
        public void Envelope_OfFewerFramesThanColumns_UsesOneFramePerColumn()
        {
            var envelope = Waveform.Envelope(new double[] { 0.5, -0.5 }, 1, 4);
            Assert.Equal(new[] { (0.5f, 0.5f), (0.5f, 0.5f), (-0.5f, -0.5f), (-0.5f, -0.5f) }, envelope[0]);
        }

        [Fact]
        public void Envelope_OfNothingIsFlat()
        {
            Assert.All(Waveform.Envelope(Array.Empty<double>(), 2, 3), lane => Assert.All(lane, c => Assert.Equal((0f, 0f), c)));
        }

        [Fact]
        public void Lanes_SplitTheAreaEvenly()
        {
            var lanes = Waveform.Lanes(new RectangleF(0, 0, 100, 62), 2);
            Assert.Equal(2, lanes.Length);
            Assert.Equal(lanes[0].Height, lanes[1].Height, 3);
            Assert.True(lanes[0].Bottom <= lanes[1].Top);
            Assert.Equal(62, lanes[1].Bottom, 3);
        }

        [Fact]
        public void Ticks_RunFromZeroToTheDurationAcrossTheWidth()
        {
            var ticks = Waveform.Ticks(2.0, 200);
            Assert.Equal(new[] { "0", "0.5", "1", "1.5", "2" }, ticks.Select(t => t.label).ToArray());
            Assert.Equal(0f, ticks[0].x);
            Assert.Equal(200f, ticks[^1].x, 3);
        }

        [Fact]
        public void Header_NamesFormatChannelsRateDurationAndPeak()
        {
            Assert.Equal("binaural, 2 ch, 48 kHz, 2.50 s, peak -1.0 dBFS", Waveform.Header("binaural", 2, 48000, 2.5, 10 * Math.Log10(Math.Pow(10, -0.1))));
        }
    }

    public class PlayerLayoutTests
    {
        // the transport beside the icon, top right; the channels below, across the whole component
        private static readonly RectangleF Controls = new(200, 100, 240, PlayerLayout.ControlsHeight);

        [Fact]
        public void Arrange_PutsTheTransportTopRight_AndTheChannelsFullWidthBelow()
        {
            var wave = new RectangleF(10, Controls.Bottom, 430, PlayerLayout.WaveHeight(2, 44));
            var boxes = PlayerLayout.Arrange(Controls, wave);
            Assert.All(new[] { boxes.PlayPause, boxes.Stop, boxes.Normalize, boxes.Volume, boxes.VolumeLabel }, b => Assert.True(Controls.Contains(b)));
            Assert.True(boxes.PlayPause.Right <= boxes.Stop.Left && boxes.Stop.Right <= boxes.Normalize.Left && boxes.Normalize.Right <= boxes.Volume.Left);
            Assert.True(boxes.VolumeLabel.Bottom <= boxes.Volume.Top && boxes.VolumeLabel.Left == boxes.Volume.Left);
            Assert.True(boxes.Header.Top >= wave.Top && boxes.Header.Bottom <= boxes.Lanes.Top && boxes.Lanes.Bottom <= boxes.Ruler.Top && boxes.Ruler.Bottom <= wave.Bottom);
            Assert.Equal(wave.Left + PlayerLayout.Margin, boxes.Lanes.Left, 3);
            Assert.Equal(wave.Width - 2 * PlayerLayout.Margin, boxes.Lanes.Width, 3);
            Assert.Equal(boxes.Lanes.Width, boxes.Ruler.Width, 3);
        }

        [Fact]
        public void WithoutASignal_OnlyTheButtonsAndTheVolumeShow()
        {
            var boxes = PlayerLayout.Arrange(Controls, RectangleF.Empty);
            Assert.True(boxes.Lanes.IsEmpty && boxes.Header.IsEmpty && boxes.Ruler.IsEmpty);
            Assert.True(Controls.Contains(boxes.PlayPause) && Controls.Contains(boxes.Volume) && Controls.Contains(boxes.VolumeLabel));
            // larger buttons than the 26 by 18 they were
            Assert.True(boxes.PlayPause.Width >= 32 && boxes.PlayPause.Height >= 22);
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(4)]
        public void EveryChannel_GetsTheLaneHeight_TheComponentGrowsDownWithThem(int channels)
        {
            var wave = new RectangleF(10, Controls.Bottom, 430, PlayerLayout.WaveHeight(channels, 44));
            var lanes = Waveform.Lanes(PlayerLayout.Arrange(Controls, wave).Lanes, channels);
            Assert.All(lanes, lane => Assert.Equal(44f, lane.Height, 3));
            Assert.Equal(44 + Waveform.LaneGap, PlayerLayout.WaveHeight(2, 44) - PlayerLayout.WaveHeight(1, 44), 3);
        }

        [Fact]
        public void SeekTime_MapsTheLaneWidthToTheDuration()
        {
            var lanes = new RectangleF(10, 0, 200, 50);
            Assert.Equal(0.0, PlayerLayout.SeekTime(lanes, 10, 4.0));
            Assert.Equal(2.0, PlayerLayout.SeekTime(lanes, 110, 4.0), 9);
            Assert.Equal(4.0, PlayerLayout.SeekTime(lanes, 500, 4.0));
        }

        [Fact]
        public void PlayheadX_IsThePositionAcrossTheLanes()
        {
            Assert.Equal(60f, PlayerLayout.PlayheadX(new RectangleF(10, 0, 200, 50), 1.0, 4.0), 3);
            Assert.Equal(10f, PlayerLayout.PlayheadX(new RectangleF(10, 0, 200, 50), 1.0, 0.0), 3);
        }
    }

    public class TransportTests
    {
        private double _now;
        private Transport Make(double duration = 4.0) => new(() => _now) { Duration = duration };

        [Fact]
        public void Play_FromStopped_StartsAtThePlayhead()
        {
            var t = Make();
            Assert.Equal(new PlayCommand(0.0), t.Play());
            Assert.Equal(TransportState.Playing, t.State);
            _now = 1.5;
            Assert.Equal(1.5, t.Position, 9);
            Assert.Null(t.Play());
        }

        [Fact]
        public void Transport_PlayPauseResume_KeepsThePosition()
        {
            var t = Make();
            t.Play();
            _now = 2;
            Assert.Equal(new PauseCommand(), t.Pause());
            _now = 5;
            Assert.Equal(2.0, t.Position, 9);
            Assert.Equal(TransportState.Paused, t.State);
            Assert.Equal(new ResumeCommand(), t.Play());
            _now = 6;
            Assert.Equal(3.0, t.Position, 9);
        }

        [Fact]
        public void Transport_Stop_ReturnsToZero()
        {
            var t = Make();
            t.Play();
            _now = 1;
            Assert.Equal(new StopCommand(), t.Stop());
            Assert.Equal(0.0, t.Position);
            Assert.Equal(TransportState.Stopped, t.State);
            Assert.Null(t.Stop());
        }

        [Fact]
        public void Transport_Seek_WhilePlaying_IsAPlayCommandFromThere()
        {
            var t = Make();
            t.Play();
            _now = 1;
            Assert.Equal(new PlayCommand(3.0), t.Seek(3.0));
            _now = 1.5;
            Assert.Equal(3.5, t.Position, 9);
        }

        [Fact]
        public void Transport_Seek_WhileStoppedOrPaused_MovesThePlayheadAndPlayStartsThere()
        {
            var t = Make();
            Assert.Null(t.Seek(2.5));
            Assert.Equal(2.5, t.Position);
            Assert.Equal(new PlayCommand(2.5), t.Play());
            _now = 1;
            t.Pause();
            Assert.Null(t.Seek(1.0));
            Assert.Equal(TransportState.Stopped, t.State);
            Assert.Equal(new PlayCommand(1.0), t.Play());
        }

        [Fact]
        public void Transport_StopsWhenTheWorkerReportsTheEnd()
        {
            var t = Make();
            t.Play();
            _now = 4.2;
            t.Sync(false, 4.0);
            Assert.Equal(TransportState.Stopped, t.State);
            Assert.Equal(4.0, t.Position);
            // a play at the end starts over
            Assert.Equal(new PlayCommand(0.0), t.Play());
        }

        [Fact]
        public void Transport_SyncCorrectsDrift()
        {
            var t = Make();
            t.Play();
            _now = 2.0;
            t.Sync(true, 2.5);
            Assert.Equal(2.5, t.Position, 9);
            _now = 2.5;
            Assert.Equal(3.0, t.Position, 9);
        }

        [Fact]
        public void Transport_ClampsToDuration()
        {
            var t = Make(1.0);
            t.Play();
            _now = 10;
            Assert.Equal(1.0, t.Position);
            Assert.Equal(new PlayCommand(0.0), t.Seek(-3));
        }

        [Fact]
        public void Transport_IgnoresSyncWhenNotPlaying()
        {
            var t = Make();
            t.Seek(1.0);
            t.Sync(false, 0.0);
            Assert.Equal(1.0, t.Position);
        }
    }

    public class ImpulseResponseAudioTests
    {
        private static ImpulseResponseData Ir(int channels, string output, int? order = null)
        {
            var ir = new ImpulseResponseData { SamplingRate = 48000 };
            ir.Samples.Add(Enumerable.Range(0, 4 * channels).Select(i => (double)i));
            ir.Shape.Add(new long[] { 4, channels });
            ir.Generation = WorkerClient.Serializer.PackAsAnyData(new Dictionary<string, object?> { ["output"] = output });
            if (order is int o)
                ir.AmbisonicOrder = o;
            return ir;
        }

        [Fact]
        public void AnImpulseResponse_PlaysAsAudioOfItsFormat()
        {
            var binaural = AudioPlayerComponent.Audio(new ImpulseResponseGoo(Ir(2, "binaural")));
            Assert.Equal("binaural", binaural.Format);
            Assert.Equal(48000, binaural.SamplingRate);
            Assert.Equal(new long[] { 4, 2 }, binaural.Shape);
            Assert.Equal(Enumerable.Range(0, 8).Select(i => (double)i), binaural.Samples);
            Assert.Equal("mono", AudioPlayerComponent.Audio(new ImpulseResponseGoo(Ir(1, "mono"))).Format);
            Assert.Equal("ambisonic", AudioPlayerComponent.Audio(new ImpulseResponseGoo(Ir(4, "ambisonic", 1))).Format);
        }

        [Fact]
        public void TheSameImpulseResponse_GivesTheSameSignal_SoARecomputeDoesNotStopIt()
        {
            var player = new AudioPlayerComponent();
            var ir = Ir(2, "binaural");
            var first = player.SignalOf(new ImpulseResponseGoo(ir));
            Assert.Same(first, player.SignalOf(new ImpulseResponseGoo(ir)));
            Assert.NotSame(first, player.SignalOf(new ImpulseResponseGoo(Ir(2, "binaural"))));
        }
    }
}
