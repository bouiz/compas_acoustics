using System;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using Grasshopper.GUI;
using Grasshopper.GUI.Canvas;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The Audio Player's body: the transport and volume right of the icon, and below the inputs a header, a lane per
    /// channel, ruler and playhead across the whole component; resizable by its corner.</summary>
    public sealed class AudioPlayerUi : ComponentUi
    {
        private const float MinWidth = 180;
        private const float MinLaneHeight = 20;
        private const float Grip = 8;
        private static readonly SliderControl Volume = new("volume", -40, 6, "dB");
        private (object? signal, int columns, (float min, float max)[][] envelope)? _cache;
        private bool _resizing;
        private bool _dragging;
        private double? _draggedGain;
        private string? _down;

        public AudioPlayerUi(AudioPlayerComponent owner) : base(owner) { }

        private AudioPlayerComponent Player => (AudioPlayerComponent)Owner;

        private RectangleF _controls;
        private RectangleF _wave;

        protected override void Layout()
        {
            base.Layout();
            // the inputs and the icon, the transport to their right; the channels below, across the whole width
            var b = Bounds;
            var top = Math.Max(b.Height, PlayerLayout.ControlsHeight);
            var width = b.Width + Player.BodyWidth;
            _controls = new RectangleF(b.Right, b.Y + (top - PlayerLayout.ControlsHeight) / 2, Player.BodyWidth, PlayerLayout.ControlsHeight);
            var wave = Player.WaveHeight;
            _wave = wave > 0 ? new RectangleF(b.X, b.Y + top, width, wave) : RectangleF.Empty;
            Bounds = new RectangleF(b.X, b.Y, width, top + wave);
        }

        private PlayerBoxes Boxes => PlayerLayout.Arrange(_controls, _wave);

        protected override void RenderExtra(GH_Canvas canvas, Graphics g)
        {
            var boxes = Boxes;
            var signal = Player.Signal;
            using var small = new Font(FontFamily.GenericSansSerif, 6f);
            var channels = Player.Channels;
            g.DrawString("Volume", global::Grasshopper.Kernel.GH_FontServer.StandardAdjusted, Brushes.Black, boxes.VolumeLabel.Left, boxes.VolumeLabel.Top);
            if (signal != null)
            {
                var header = Player.Problem ?? Waveform.Header(signal.Format, channels, signal.SamplingRate, Player.Transport.Duration, Peak(signal));
                g.DrawString(header, small, Player.Problem != null ? Brushes.DarkRed : Brushes.Black,
                    new RectangleF(boxes.Header.Location, new SizeF(boxes.Header.Width, boxes.Header.Height + 2)));
                var lanes = Waveform.Lanes(boxes.Lanes, channels);
                using (var back = new SolidBrush(Color.FromArgb(245, 245, 245)))
                    foreach (var lane in lanes)
                        g.FillRectangle(back, lane);
                var columns = Math.Max((int)boxes.Lanes.Width, 1);
                if (_cache == null || !ReferenceEquals(_cache.Value.signal, signal) || _cache.Value.columns != columns)
                    _cache = (signal, columns, Waveform.Envelope(signal.Samples, channels, columns));
                var peak = _cache.Value.envelope.SelectMany(l => l).Select(c => Math.Max(Math.Abs(c.min), Math.Abs(c.max))).DefaultIfEmpty(0).Max();
                var scale = peak > 0 ? 1 / peak : 1;
                using var pen = new Pen(Color.FromArgb(40, 90, 160), 1);
                for (var c = 0; c < lanes.Length; c++)
                {
                    var lane = lanes[c];
                    var mid = lane.Top + lane.Height / 2;
                    for (var col = 0; col < columns; col++)
                    {
                        var (min, max) = _cache.Value.envelope[c][col];
                        g.DrawLine(pen, lane.Left + col, mid - max * scale * lane.Height / 2, lane.Left + col, mid - min * scale * lane.Height / 2 + 0.5f);
                    }
                }
                var x = PlayerLayout.PlayheadX(boxes.Lanes, Player.Transport.Position, Player.Transport.Duration);
                using var head = new Pen(Color.FromArgb(200, 40, 40), 1.5f);
                g.DrawLine(head, x, boxes.Lanes.Top, x, boxes.Lanes.Bottom);
                foreach (var (tx, tick) in Waveform.Ticks(Player.Transport.Duration, boxes.Ruler.Width))
                {
                    g.DrawLine(Pens.Gray, boxes.Ruler.Left + tx, boxes.Ruler.Top, boxes.Ruler.Left + tx, boxes.Ruler.Top + 3);
                    g.DrawString(tick, small, Brushes.Gray, boxes.Ruler.Left + tx - 3, boxes.Ruler.Top + 2);
                }
            }
            // one button plays and pauses: it shows what a press does
            var playing = Player.Transport.State == TransportState.Playing;
            DrawIconButton(g, boxes.PlayPause, playing ? Glyph.Pause : Glyph.Play, Player.CanPlay, _down == "playpause");
            DrawIconButton(g, boxes.Stop, Glyph.Stop, Player.CanPlay && Player.Transport.State != TransportState.Stopped, _down == "stop");
            // a toggle: dark when the signal is scaled to full level
            using (var norm = GH_Capsule.CreateTextCapsule(Rectangle.Round(boxes.Normalize), Rectangle.Round(boxes.Normalize),
                       Player.Normalize ? GH_Palette.Black : GH_Palette.White, "Normalize", 2, 0))
                norm.Render(g, Selected, Owner.Locked, false);
            var gain = _draggedGain ?? Player.GainDb;
            Painter.Draw(g, new ControlBox(Volume, boxes.Volume, Array.Empty<RectangleF>()),
                (gain.ToString(System.Globalization.CultureInfo.InvariantCulture), false), Selected, Owner.Locked, false, _draggedGain);
            // the resize grip, at the component's corner
            var corner = Bounds;
            g.DrawLine(Pens.Gray, corner.Right - Grip, corner.Bottom - 2, corner.Right - 2, corner.Bottom - Grip);
            g.DrawLine(Pens.Gray, corner.Right - Grip / 2, corner.Bottom - 2, corner.Right - 2, corner.Bottom - Grip / 2);
        }

        private static double Peak(CompasAcoustics.Pb.AudioData signal)
        {
            var peak = signal.Samples.Count == 0 ? 0 : signal.Samples.Max(Math.Abs);
            return peak > 0 ? 20 * Math.Log10(peak) : double.NegativeInfinity;
        }

        /// <summary>A capsule with the usual transport glyph, as the Visualise components draw theirs.</summary>
        private void DrawIconButton(Graphics g, RectangleF box, Glyph glyph, bool enabled, bool down)
        {
            DrawButton(g, box, "", enabled, down);
            Painter.DrawGlyph(g, box, glyph, enabled);
        }

        private void DrawButton(Graphics g, RectangleF box, string label, bool enabled, bool down)
        {
            var palette = !enabled ? GH_Palette.Hidden : down ? GH_Palette.Grey : GH_Palette.Black;
            using var capsule = GH_Capsule.CreateTextCapsule(Rectangle.Round(box), Rectangle.Round(box), palette, label, 2, 0);
            capsule.Render(g, Selected, Owner.Locked, false);
        }

        private static string? ButtonAt(PlayerBoxes boxes, PointF p) => PlayerLayout.Hit(boxes, p) switch
        {
            PlayerHit.PlayPause => "playpause",
            PlayerHit.Stop => "stop",
            PlayerHit.Normalize => "normalize",
            _ => null,
        };

        protected override GH_ObjectResponse? RespondToExtraMouseDown(GH_Canvas sender, GH_CanvasMouseEvent e)
        {
            var p = e.CanvasLocation;
            if (Bounds.Contains(p) && p.X > Bounds.Right - Grip * 1.5f && p.Y > Bounds.Bottom - Grip * 1.5f)
            {
                _resizing = true;
                return GH_ObjectResponse.Capture;
            }
            var boxes = Boxes;
            switch (PlayerLayout.Hit(boxes, p))
            {
                case PlayerHit.PlayPause or PlayerHit.Stop or PlayerHit.Normalize:
                    _down = ButtonAt(boxes, p);
                    sender.Invalidate();
                    return GH_ObjectResponse.Capture;
                case PlayerHit.Volume:
                    _dragging = true;
                    _draggedGain = SliderValue(Volume, boxes.Volume, p.X);
                    sender.Invalidate();
                    return GH_ObjectResponse.Capture;
                case PlayerHit.Lanes:
                    Player.OnSeek(PlayerLayout.SeekTime(boxes.Lanes, p.X, Player.Transport.Duration));
                    sender.Invalidate();
                    return GH_ObjectResponse.Handled;
                default:
                    // the header, the gaps, the icon and the input names: Grasshopper's own press, which drags the component
                    return null;
            }
        }

        public override bool IsTooltipRegion(PointF canvasPoint) => Boxes.Normalize.Contains(canvasPoint) || base.IsTooltipRegion(canvasPoint);

        public override void SetupTooltip(PointF canvasPoint, GH_TooltipDisplayEventArgs e)
        {
            if (Boxes.Normalize.Contains(canvasPoint))
            {
                (e.Title, e.Text) = PlayerLayout.NormalizeTooltip;
                e.Description = "";
                return;
            }
            base.SetupTooltip(canvasPoint, e);
        }

        public override GH_ObjectResponse RespondToMouseDoubleClick(GH_Canvas sender, GH_CanvasMouseEvent e)
        {
            // the volume slider, dot or track: back to 0 dB
            if (Boxes.Volume.Contains(e.CanvasLocation))
            {
                _dragging = false;
                _draggedGain = null;
                Player.OnResetVolume();
                sender.Invalidate();
                return GH_ObjectResponse.Handled;
            }
            return base.RespondToMouseDoubleClick(sender, e);
        }

        public override GH_ObjectResponse RespondToMouseMove(GH_Canvas sender, GH_CanvasMouseEvent e)
        {
            if (_resizing)
            {
                // the width, and the lanes' height: the channels' height shared by them
                Player.BodyWidth = Math.Max(e.CanvasLocation.X - _controls.Left, MinWidth);
                // without a signal there are no lanes: only the width changes
                if (!_wave.IsEmpty)
                {
                    var fixedHeight = PlayerLayout.WaveHeight(Player.Channels, 0);
                    Player.LaneHeight = Math.Max((e.CanvasLocation.Y - _wave.Top - fixedHeight) / Player.Channels, MinLaneHeight);
                }
                ExpireLayout();
                sender.Invalidate();
                return GH_ObjectResponse.Handled;
            }
            if (_dragging)
            {
                _draggedGain = SliderValue(Volume, Boxes.Volume, e.CanvasLocation.X);
                sender.Invalidate();
                return GH_ObjectResponse.Handled;
            }
            return base.RespondToMouseMove(sender, e);
        }

        public override GH_ObjectResponse RespondToMouseUp(GH_Canvas sender, GH_CanvasMouseEvent e)
        {
            if (_resizing)
            {
                _resizing = false;
                return GH_ObjectResponse.Release;
            }
            if (_dragging)
            {
                _dragging = false;
                if (_draggedGain is double gain)
                    Player.OnGain(gain);
                _draggedGain = null;
                sender.Invalidate();
                return GH_ObjectResponse.Release;
            }
            if (_down != null)
            {
                var button = _down;
                _down = null;
                if (ButtonAt(Boxes, e.CanvasLocation) == button)
                {
                    if (button == "playpause") Player.OnPlayPause();
                    else if (button == "normalize") Player.OnNormalize(!Player.Normalize);
                    else Player.OnStop();
                }
                sender.Invalidate();
                return GH_ObjectResponse.Release;
            }
            return base.RespondToMouseUp(sender, e);
        }
    }
}
