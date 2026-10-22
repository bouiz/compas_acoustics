using System;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using Grasshopper.GUI;
using Grasshopper.GUI.Canvas;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The Audio Player's body: header, a lane per channel, ruler, playhead, transport and volume; resizable by its corner.</summary>
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

        private RectangleF _body;

        protected override void Layout()
        {
            base.Layout();
            // the inputs and the icon on the left, the player to the right of the icon, as tall as the two need
            var b = Bounds;
            var size = Player.BodySize;
            Bounds = new RectangleF(b.X, b.Y, b.Width + size.Width, Math.Max(b.Height, size.Height));
            _body = new RectangleF(b.Right, b.Y + (Bounds.Height - size.Height) / 2, size.Width, size.Height);
        }

        private RectangleF Body => _body;

        private PlayerBoxes Boxes => PlayerLayout.Arrange(Body, Player.Signal != null);

        protected override void RenderExtra(GH_Canvas canvas, Graphics g)
        {
            var boxes = Boxes;
            var signal = Player.Signal;
            using var small = new Font(FontFamily.GenericSansSerif, 6f);
            using var label = new Font(FontFamily.GenericSansSerif, 7f);
            var channels = Player.Channels;
            g.DrawString("Volume", label, Brushes.Black, boxes.VolumeLabel.Left, boxes.VolumeLabel.Top - 1);
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
            DrawIconButton(g, boxes.PlayPause, playing ? Icon.Pause : Icon.Play, Player.CanPlay, _down == "playpause");
            DrawIconButton(g, boxes.Stop, Icon.Stop, Player.CanPlay && Player.Transport.State != TransportState.Stopped, _down == "stop");
            // a toggle: dark when the signal is scaled to full level
            using (var norm = GH_Capsule.CreateTextCapsule(Rectangle.Round(boxes.Normalize), Rectangle.Round(boxes.Normalize),
                       Player.Normalize ? GH_Palette.Black : GH_Palette.White, "Norm", 2, 0))
                norm.Render(g, Selected, Owner.Locked, false);
            var gain = _draggedGain ?? Player.GainDb;
            Painter.Draw(g, new ControlBox(Volume, boxes.Volume, Array.Empty<RectangleF>()),
                (gain.ToString(System.Globalization.CultureInfo.InvariantCulture), false), Selected, Owner.Locked, false, _draggedGain);
            // the resize grip
            var corner = Body;
            g.DrawLine(Pens.Gray, corner.Right - Grip, corner.Bottom - 2, corner.Right - 2, corner.Bottom - Grip);
            g.DrawLine(Pens.Gray, corner.Right - Grip / 2, corner.Bottom - 2, corner.Right - 2, corner.Bottom - Grip / 2);
        }

        private static double Peak(CompasAcoustics.Pb.AudioData signal)
        {
            var peak = signal.Samples.Count == 0 ? 0 : signal.Samples.Max(Math.Abs);
            return peak > 0 ? 20 * Math.Log10(peak) : double.NegativeInfinity;
        }

        private enum Icon { Play, Pause, Stop }

        /// <summary>A capsule with the usual transport glyph: a triangle, two bars, a square.</summary>
        private void DrawIconButton(Graphics g, RectangleF box, Icon icon, bool enabled, bool down)
        {
            DrawButton(g, box, "", enabled, down);
            var size = Math.Min(box.Height, box.Width) * 0.45f;
            var (cx, cy) = (box.Left + box.Width / 2, box.Top + box.Height / 2);
            using var brush = new SolidBrush(enabled ? Color.White : Color.FromArgb(150, 150, 150));
            var state = g.SmoothingMode;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            switch (icon)
            {
                case Icon.Play:
                    g.FillPolygon(brush, new[] { new PointF(cx - size * 0.45f, cy - size / 2), new PointF(cx - size * 0.45f, cy + size / 2), new PointF(cx + size * 0.55f, cy) });
                    break;
                case Icon.Pause:
                    g.FillRectangle(brush, cx - size / 2, cy - size / 2, size * 0.35f, size);
                    g.FillRectangle(brush, cx + size * 0.15f, cy - size / 2, size * 0.35f, size);
                    break;
                case Icon.Stop:
                    g.FillRectangle(brush, cx - size * 0.4f, cy - size * 0.4f, size * 0.8f, size * 0.8f);
                    break;
            }
            g.SmoothingMode = state;
        }

        private void DrawButton(Graphics g, RectangleF box, string label, bool enabled, bool down)
        {
            var palette = !enabled ? GH_Palette.Hidden : down ? GH_Palette.Grey : GH_Palette.Black;
            using var capsule = GH_Capsule.CreateTextCapsule(Rectangle.Round(box), Rectangle.Round(box), palette, label, 2, 0);
            capsule.Render(g, Selected, Owner.Locked, false);
        }

        private string? ButtonAt(PlayerBoxes boxes, PointF p) =>
            boxes.PlayPause.Contains(p) ? "playpause" : boxes.Stop.Contains(p) ? "stop" : boxes.Normalize.Contains(p) ? "normalize" : null;

        protected override GH_ObjectResponse? RespondToExtraMouseDown(GH_Canvas sender, GH_CanvasMouseEvent e)
        {
            var body = Body;
            var p = e.CanvasLocation;
            if (!body.Contains(p))
                return null;
            var boxes = Boxes;
            if (p.X > body.Right - Grip * 1.5f && p.Y > body.Bottom - Grip * 1.5f)
            {
                _resizing = true;
                return GH_ObjectResponse.Capture;
            }
            if (ButtonAt(boxes, p) is string button)
            {
                _down = button;
                sender.Invalidate();
                return GH_ObjectResponse.Capture;
            }
            if (boxes.Volume.Contains(p))
            {
                _dragging = true;
                _draggedGain = SliderValue(Volume, boxes.Volume, p.X);
                sender.Invalidate();
                return GH_ObjectResponse.Capture;
            }
            if (!boxes.Lanes.IsEmpty && boxes.Lanes.Contains(p))
            {
                Player.OnSeek(PlayerLayout.SeekTime(boxes.Lanes, p.X, Player.Transport.Duration));
                sender.Invalidate();
                return GH_ObjectResponse.Handled;
            }
            return GH_ObjectResponse.Handled;
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
                // the width, and the lanes' height: the body's height shared by its channels
                Player.BodyWidth = Math.Max(e.CanvasLocation.X - Body.Left, MinWidth);
                // without a signal there are no lanes: only the width changes
                if (Player.Signal != null)
                {
                    var fixedHeight = PlayerLayout.BodyHeight(Player.Channels, 0);
                    Player.LaneHeight = Math.Max((e.CanvasLocation.Y - Body.Top - fixedHeight) / Player.Channels, MinLaneHeight);
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
