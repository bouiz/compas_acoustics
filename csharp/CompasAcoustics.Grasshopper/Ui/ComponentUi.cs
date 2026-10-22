using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using Grasshopper.GUI;
using Grasshopper.GUI.Canvas;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Attributes;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The attributes of a component with controls: its usual box, and the controls in rows below its ports.</summary>
    public class ComponentUi : GH_ComponentAttributes
    {
        private List<ControlBox> _boxes = new();
        private ControlBox? _down;
        private double? _dragged;

        public ComponentUi(ComponentBase owner) : base(owner) { }

        private ComponentBase Component => (ComponentBase)Owner;

        /// <summary>The controls' area, below the ports.</summary>
        protected RectangleF ControlArea { get; private set; }

        /// <summary>The height of a body drawn under the controls (the Audio Player's waveform); 0 for none.</summary>
        protected virtual float ExtraHeight => 0;

        /// <summary>The area of that body.</summary>
        protected RectangleF ExtraArea { get; private set; }

        protected override void Layout()
        {
            base.Layout();
            var controls = Component.BodyControls;
            var box = Bounds;
            var height = ControlLayout.Height(controls);
            ControlArea = new RectangleF(box.X, box.Bottom, box.Width, height);
            ExtraArea = new RectangleF(box.X, box.Bottom + height, box.Width, ExtraHeight);
            _boxes = ControlLayout.Arrange(controls, ControlArea);
            Bounds = new RectangleF(box.X, box.Y, box.Width, box.Height + height + ExtraHeight);
        }

        protected override void Render(GH_Canvas canvas, Graphics graphics, GH_CanvasChannel channel)
        {
            base.Render(canvas, graphics, channel);
            if (channel != GH_CanvasChannel.Objects)
                return;
            foreach (var box in _boxes)
            {
                if (!box.Title.IsEmpty)
                    Painter.DrawTitle(graphics, box);
                Painter.Draw(graphics, box, Value(box), Selected, Owner.Locked, ReferenceEquals(box, _down), _dragged);
            }
            RenderExtra(canvas, graphics);
        }

        /// <summary>Draw the body below the controls.</summary>
        protected virtual void RenderExtra(GH_Canvas canvas, Graphics graphics) { }

        private (string value, bool wired) Value(ControlBox box)
        {
            var port = Component.ControlPort(box.Control.Port);
            return box.Control switch
            {
                SwitchControl s when port != null => ControlState.Of(port, s.Options),
                DropdownControl d when port != null => ControlState.Of(port, d.Options),
                SwitchControl or DropdownControl when Component.IsChoice(box.Control.Port) => (Component.Choice(box.Control.Port), false),
                SliderControl when port != null => (Convert.ToString(ControlState.Persistent(port) ?? 0.0, System.Globalization.CultureInfo.InvariantCulture)!, port.SourceCount > 0),
                ButtonControl b => (Component.ButtonLabel(b.Port) ?? b.Label, port != null && port.SourceCount > 0),
                ToggleControl t => (Component.Choice(t.Port), false),
                _ => ("", false),
            };
        }

        public override GH_ObjectResponse RespondToMouseDown(GH_Canvas sender, GH_CanvasMouseEvent e)
        {
            if (e.Button != MouseButtons.Left)
                return base.RespondToMouseDown(sender, e);
            var hit = ControlLayout.HitTest(_boxes, e.CanvasLocation);
            if (hit == null)
                return RespondToExtraMouseDown(sender, e) ?? base.RespondToMouseDown(sender, e);
            var (box, segment) = hit.Value;
            var port = Component.ControlPort(box.Control.Port);
            if (port != null && port.SourceCount > 0 && box.Control is not ButtonControl)
                return GH_ObjectResponse.Handled;
            switch (box.Control)
            {
                case SwitchControl s when segment >= 0 && port != null:
                    ControlState.Choose(Component, port, s.Options[segment]);
                    return GH_ObjectResponse.Handled;
                case SwitchControl s when segment >= 0 && Component.IsChoice(s.Port):
                    Component.SetChoice(s.Port, s.Options[segment]);
                    return GH_ObjectResponse.Handled;
                case DropdownControl d when port != null || Component.IsChoice(d.Port):
                    OpenMenu(sender, e, d, port);
                    return GH_ObjectResponse.Handled;
                case ToggleControl t:
                    Component.SetChoice(t.Port, Component.Choice(t.Port) == "on" ? "off" : "on");
                    sender.Invalidate();
                    return GH_ObjectResponse.Handled;
                case ButtonControl:
                case SliderControl:
                    _down = box;
                    _dragged = box.Control is SliderControl slider ? SliderValue(slider, box.Bounds, e.CanvasLocation.X) : null;
                    sender.Invalidate();
                    return GH_ObjectResponse.Capture;
            }
            return GH_ObjectResponse.Handled;
        }

        public override bool IsTooltipRegion(PointF canvasPoint) => ControlLayout.HitTest(_boxes, canvasPoint) != null || base.IsTooltipRegion(canvasPoint);

        public override void SetupTooltip(PointF canvasPoint, GH_TooltipDisplayEventArgs e)
        {
            if (ControlLayout.HitTest(_boxes, canvasPoint) is { } hit && hit.box.Control is not ButtonControl { Port: ComponentBase.CancelPort })
            {
                // a control: what its input means
                (e.Title, e.Text) = Component.ControlTooltip(hit.box.Control.Port);
                e.Description = "";
                return;
            }
            base.SetupTooltip(canvasPoint, e);
            if (Component.FullMessage is string full)
                e.Description = string.IsNullOrEmpty(e.Description) ? full : $"{full}\n\n{e.Description}";
        }

        /// <summary>A press in the body below the controls; null when it is not handled there.</summary>
        protected virtual GH_ObjectResponse? RespondToExtraMouseDown(GH_Canvas sender, GH_CanvasMouseEvent e) => null;

        public override GH_ObjectResponse RespondToMouseMove(GH_Canvas sender, GH_CanvasMouseEvent e)
        {
            if (_down?.Control is SliderControl slider)
            {
                _dragged = SliderValue(slider, _down.Bounds, e.CanvasLocation.X);
                sender.Invalidate();
                return GH_ObjectResponse.Handled;
            }
            return base.RespondToMouseMove(sender, e);
        }

        public override GH_ObjectResponse RespondToMouseUp(GH_Canvas sender, GH_CanvasMouseEvent e)
        {
            if (_down == null)
                return base.RespondToMouseUp(sender, e);
            var box = _down;
            _down = null;
            if (box.Control is ButtonControl button && box.Bounds.Contains(e.CanvasLocation))
            {
                if (Component.HandlesButton(button.Port))
                    Component.OnButton(button.Port);
                else
                    ControlState.Press(Component, button.Port);
            }
            else if (box.Control is SliderControl && _dragged is double value)
            {
                var port = Component.ControlPort(box.Control.Port);
                if (port != null)
                    ControlState.ChooseNumber(Component, port, value);
                else
                    Component.SliderChanged(box.Control.Port, value);
            }
            _dragged = null;
            sender.Invalidate();
            return GH_ObjectResponse.Release;
        }

        public static double SliderValue(SliderControl slider, RectangleF bounds, float x)
        {
            var t = Math.Clamp((x - bounds.Left) / Math.Max(bounds.Width, 1), 0, 1);
            return Math.Round(slider.Min + t * (slider.Max - slider.Min), 1);
        }

        private void OpenMenu(GH_Canvas sender, GH_CanvasMouseEvent e, DropdownControl dropdown, IGH_Param? port)
        {
            var current = port != null ? ControlState.Of(port, dropdown.Options).value : Component.Choice(dropdown.Port);
            var menu = new ContextMenuStrip();
            // the input's name heads its options
            menu.Items.Add(new ToolStripMenuItem(dropdown.Port) { Enabled = false });
            menu.Items.Add(new ToolStripSeparator());
            foreach (var option in dropdown.Options)
            {
                var item = new ToolStripMenuItem(option) { Checked = option == current };
                item.Click += (s, a) =>
                {
                    if (port != null)
                        ControlState.Choose(Component, port, option);
                    else
                        Component.SetChoice(dropdown.Port, option);
                };
                menu.Items.Add(item);
            }
            menu.Show(sender, e.ControlLocation);
        }
    }

    /// <summary>How the controls look: Grasshopper's capsules, so they match the canvas.</summary>
    internal static class Painter
    {
        private static Rectangle R(RectangleF r) => Rectangle.Round(r);

        /// <summary>The input's name above its switch or dropdown.</summary>
        public static void DrawTitle(Graphics g, ControlBox box)
        {
            using var format = new StringFormat { Trimming = StringTrimming.EllipsisCharacter, FormatFlags = StringFormatFlags.NoWrap, LineAlignment = StringAlignment.Far };
            // Grasshopper's font for the input and output names, so the titles read as the ports do
            g.DrawString(ControlLayout.Title(box.Control), GH_FontServer.StandardAdjusted, Brushes.Black, box.Title, format);
        }

        public static void Draw(Graphics g, ControlBox box, (string value, bool wired) state, bool selected, bool locked, bool down, double? dragged)
        {
            switch (box.Control)
            {
                case SwitchControl s:
                    for (var i = 0; i < s.Options.Length; i++)
                    {
                        var chosen = s.Options[i] == state.value;
                        var palette = chosen ? (state.wired ? GH_Palette.Grey : GH_Palette.Black) : GH_Palette.White;
                        using var capsule = GH_Capsule.CreateTextCapsule(R(box.Segments[i]), R(box.Segments[i]), palette, s.Options[i], 2, 0);
                        capsule.Render(g, selected, locked, false);
                    }
                    break;
                case DropdownControl:
                {
                    using var capsule = GH_Capsule.CreateTextCapsule(R(box.Bounds), R(box.Bounds), state.wired ? GH_Palette.Grey : GH_Palette.White,
                        ControlLayout.Text(box.Control, state.value), 2, 0);
                    capsule.Render(g, selected, locked, false);
                    var x = box.Bounds.Right - 9;
                    var y = box.Bounds.Top + box.Bounds.Height / 2 - 2;
                    g.FillPolygon(Brushes.Black, new[] { new PointF(x, y), new PointF(x + 6, y), new PointF(x + 3, y + 4) });
                    break;
                }
                case ToggleControl toggle:
                {
                    // dark when on
                    using var capsule = GH_Capsule.CreateTextCapsule(R(box.Bounds), R(box.Bounds), state.value == "on" ? GH_Palette.Black : GH_Palette.White,
                        toggle.Label, 2, 0);
                    capsule.Render(g, selected, locked, false);
                    break;
                }
                case ButtonControl:
                {
                    using var capsule = GH_Capsule.CreateTextCapsule(R(box.Bounds), R(box.Bounds), down ? GH_Palette.Grey : GH_Palette.Black, state.value, 2, 0);
                    capsule.Render(g, selected, locked, false);
                    break;
                }
                case SliderControl slider:
                {
                    var value = dragged ?? (double.TryParse(state.value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : slider.Min);
                    var t = (float)((value - slider.Min) / Math.Max(slider.Max - slider.Min, 1e-9));
                    var track = box.Bounds;
                    var mid = track.Top + track.Height / 2;
                    using var pen = new Pen(Color.FromArgb(120, 120, 120), 2);
                    g.DrawLine(pen, track.Left + 4, mid, track.Right - 4, mid);
                    var knobX = track.Left + 4 + t * (track.Width - 8);
                    g.FillEllipse(state.wired ? Brushes.Gray : Brushes.Black, knobX - 5, mid - 5, 10, 10);
                    using var font = new Font(FontFamily.GenericSansSerif, 6f);
                    g.DrawString(string.Format(System.Globalization.CultureInfo.InvariantCulture, "{0:0.0} {1}", value, slider.Unit), font, Brushes.Black, track.Left + 2, track.Top);
                    break;
                }
            }
        }
    }
}
