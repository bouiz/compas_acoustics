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
                var range = box.Control is SliderControl slider ? Component.SliderRange(slider) : ((double, double)?)null;
                Painter.Draw(graphics, box, Value(box), Selected, Owner.Locked, ReferenceEquals(box, _down), ReferenceEquals(box, _down) ? _dragged : null, range);
            }
            RenderExtra(canvas, graphics);
        }

        /// <summary>Draw the body below the controls.</summary>
        protected virtual void RenderExtra(GH_Canvas canvas, Graphics graphics) { }

        private (string value, bool wired) Value(ControlBox box)
        {
            var port = Component.ControlPort(box.Control.Port);
            var c = System.Globalization.CultureInfo.InvariantCulture;
            switch (box.Control)
            {
                case SwitchControl s when port != null:
                    return ControlState.Of(port, s.Options, s.Default);
                case DropdownControl d when port != null:
                {
                    var (value, wired) = ControlState.Of(port, Component.OptionsOf(d).values, d.Default);
                    return (DropdownText(d, value), wired);
                }
                case SwitchControl or DropdownControl when Component.IsChoice(box.Control.Port):
                    var choice = Component.Choice(box.Control.Port);
                    return (box.Control is DropdownControl dd ? DropdownText(dd, choice) : choice, false);
                case MultiSwitchControl m when port != null:
                    return (string.Join("\n", ControlState.Members(port, m.Default)), port.SourceCount > 0);
                case SliderControl slider when port != null:
                {
                    var (min, max) = Component.SliderRange(slider);
                    var shown = Component.SliderShown(slider) ?? (ControlState.Persistent(port) is { } v ? Convert.ToDouble(v, c) : max);
                    return (Convert.ToString(Math.Clamp(shown, min, Math.Max(min, max)), c)!, port.SourceCount > 0);
                }
                case ButtonControl b:
                    return (Component.ButtonLabel(b.Port) ?? b.Label, port != null && port.SourceCount > 0);
                case ToggleControl t when port != null:
                {
                    var (on, wired) = ControlState.Toggle(port, t.Default);
                    return (on ? "on" : "off", wired);
                }
                case ToggleControl t:
                    return (Component.Choice(t.Port), false);
                default:
                    return ("", false);
            }
        }

        /// <summary>What a dropdown reads for ``value``: its label among the options offered now, else the first's ("All").</summary>
        private string DropdownText(DropdownControl dropdown, string value)
        {
            var (values, labels) = Component.OptionsOf(dropdown);
            var i = Array.IndexOf(values, value);
            return i >= 0 ? labels[i] : value.Length == 0 && labels.Length > 0 ? labels[0] : ControlLayout.Label(dropdown, value);
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
                case MultiSwitchControl m when segment >= 0 && port != null:
                    ControlState.ToggleMember(Component, port, m.Options[segment], m.Default);
                    return GH_ObjectResponse.Handled;
                case ToggleControl t when port != null:
                    ControlState.Choose(Component, port, ControlState.Toggle(port, t.Default).on ? "False" : "True");
                    sender.Invalidate();
                    return GH_ObjectResponse.Handled;
                case ToggleControl t:
                    Component.SetChoice(t.Port, Component.Choice(t.Port) == "on" ? "off" : "on");
                    sender.Invalidate();
                    return GH_ObjectResponse.Handled;
                case ButtonControl:
                case SliderControl:
                    _down = box;
                    _dragged = box.Control is SliderControl slider ? Drag(slider, box.Bounds, e.CanvasLocation.X) : null;
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
                _dragged = Drag(slider, _down.Bounds, e.CanvasLocation.X);
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

        private double Drag(SliderControl slider, RectangleF bounds, float x)
        {
            var (min, max) = Component.SliderRange(slider);
            return SliderValue(slider, min, max, bounds, x);
        }

        public static double SliderValue(SliderControl slider, RectangleF bounds, float x) => SliderValue(slider, slider.Min, slider.Max, bounds, x);

        /// <summary>The value under ``x`` on a slider from ``min`` to ``max``, in its steps.</summary>
        public static double SliderValue(SliderControl slider, double min, double max, RectangleF bounds, float x)
        {
            var t = Math.Clamp((x - bounds.Left) / Math.Max(bounds.Width, 1), 0, 1);
            var value = min + t * (max - min);
            var step = slider.Step > 0 ? slider.Step : 0.1;
            return Math.Clamp(Math.Round(Math.Round((value - min) / step) * step + min, 6), min, Math.Max(min, max));
        }

        private void OpenMenu(GH_Canvas sender, GH_CanvasMouseEvent e, DropdownControl dropdown, IGH_Param? port)
        {
            var (values, labels) = Component.OptionsOf(dropdown);
            var current = port != null ? ControlState.Of(port, values, dropdown.Default).value : Component.Choice(dropdown.Port);
            var menu = new ContextMenuStrip();
            // the input's name heads its options
            menu.Items.Add(new ToolStripMenuItem(dropdown.Port) { Enabled = false });
            menu.Items.Add(new ToolStripSeparator());
            for (var i = 0; i < values.Length; i++)
            {
                var option = values[i];
                var item = new ToolStripMenuItem(labels[i]) { Checked = option == current };
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

        public static void Draw(Graphics g, ControlBox box, (string value, bool wired) state, bool selected, bool locked, bool down, double? dragged,
            (double min, double max)? range = null)
        {
            switch (box.Control)
            {
                case SwitchControl s:
                    for (var i = 0; i < s.Options.Length; i++)
                    {
                        var chosen = s.Options[i] == state.value;
                        var palette = chosen ? (state.wired ? GH_Palette.Grey : GH_Palette.Black) : GH_Palette.White;
                        using var capsule = GH_Capsule.CreateTextCapsule(R(box.Segments[i]), R(box.Segments[i]), palette, ControlLayout.Label(s, s.Options[i]), 2, 0);
                        capsule.Render(g, selected, locked, false);
                    }
                    break;
                case MultiSwitchControl m:
                {
                    var lit = state.value.Split('\n');
                    for (var i = 0; i < m.Options.Length; i++)
                    {
                        var palette = lit.Contains(m.Options[i]) ? (state.wired ? GH_Palette.Grey : GH_Palette.Black) : GH_Palette.White;
                        using var capsule = GH_Capsule.CreateTextCapsule(R(box.Segments[i]), R(box.Segments[i]), palette, m.Options[i], 2, 0);
                        capsule.Render(g, selected, locked, false);
                    }
                    break;
                }
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
                    // dark when on, grey when a wire sets it
                    var palette = state.value == "on" ? (state.wired ? GH_Palette.Grey : GH_Palette.Black) : GH_Palette.White;
                    using var capsule = GH_Capsule.CreateTextCapsule(R(box.Bounds), R(box.Bounds), palette, toggle.Label, 2, 0);
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
                    var (min, max) = range ?? (slider.Min, slider.Max);
                    var value = dragged ?? (double.TryParse(state.value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : min);
                    var t = Math.Clamp((float)((value - min) / Math.Max(max - min, 1e-9)), 0f, 1f);
                    var track = box.Bounds;
                    var mid = track.Top + track.Height / 2;
                    using var pen = new Pen(Color.FromArgb(120, 120, 120), 2);
                    g.DrawLine(pen, track.Left + 4, mid, track.Right - 4, mid);
                    var knobX = track.Left + 4 + t * (track.Width - 8);
                    g.FillEllipse(state.wired ? Brushes.Gray : Brushes.Black, knobX - 5, mid - 5, 10, 10);
                    using var font = new Font(FontFamily.GenericSansSerif, 6f);
                    g.DrawString(string.Format(System.Globalization.CultureInfo.InvariantCulture, slider.Step >= 1 ? "{0:0} {1}" : "{0:0.0} {1}", value, slider.Unit).Trim(),
                        font, Brushes.Black, track.Left + 2, track.Top);
                    break;
                }
            }
        }
    }
}
