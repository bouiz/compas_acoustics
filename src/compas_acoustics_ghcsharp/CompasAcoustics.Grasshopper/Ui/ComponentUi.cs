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
        private double? _azimuth; // the last azimuth chosen on a ring of several, kept when the colatitude changes
        private string? _stepped; // the stop a step slider's knob is dragged to, set on the input when it is let go

        public ComponentUi(ComponentBase owner) : base(owner) { }

        private ComponentBase Component => (ComponentBase)Owner;

        /// <summary>The controls' area, below the ports, or right of them in a column.</summary>
        protected RectangleF ControlArea { get; private set; }

        /// <summary>The height of a body drawn under the controls; 0 for none.</summary>
        protected virtual float ExtraHeight => 0;

        /// <summary>The area of that body.</summary>
        protected RectangleF ExtraArea { get; private set; }

        /// <summary>The width of a column right of Grasshopper's box <paramref name="box"/> that holds the controls; 0 for rows below it.</summary>
        protected virtual float ControlColumn(RectangleF box) => 0;

        protected override void Layout()
        {
            base.Layout();
            var controls = Component.BodyControls;
            var height = ControlLayout.Height(controls);
            var wider = Math.Max(MinBodyWidthOf(Component) - Bounds.Width, 0);
            var column = ControlColumn(new RectangleF(Bounds.X, Bounds.Y, Bounds.Width + wider, Bounds.Height));
            var taller = column > 0 ? Math.Max(height - Bounds.Height, 0) : 0;
            if (wider > 0 || taller > 0)
            {
                // a minimum width, or controls that need room (the incidence dials, a column taller than the ports): the component's box grown by the difference, its ports on its edges
                var inner = LayoutComponentBox(Owner);
                inner.Inflate(wider / 2, taller / 2);
                LayoutInputParams(Owner, inner);
                LayoutOutputParams(Owner, inner);
                m_innerBounds = inner;
                Bounds = LayoutBounds(Owner, inner);
            }
            var (area, box) = ControlLayout.Place(Bounds, height, column);
            ControlArea = area;
            ExtraArea = new RectangleF(box.X, box.Bottom, box.Width, ExtraHeight);
            _boxes = ControlLayout.Arrange(controls, ControlArea);
            Bounds = new RectangleF(box.X, box.Y, box.Width, box.Height + ExtraHeight);
        }

        /// <summary>The narrowest the component's box is drawn: its own minimum, or the width its controls need (the incidence dials).</summary>
        internal static float MinBodyWidthOf(ComponentBase component) => Math.Max(component.MinBodyWidth, ControlLayout.MinWidth(component.BodyControls));

        /// <summary>Whether a press on <paramref name="control"/> acts: not when a wire sets its input (buttons aside), nor while the component rests it.</summary>
        internal static bool Pressable(ComponentBase component, Control control, IGH_Param? port) =>
            (port == null || port.SourceCount == 0 || control is ButtonControl) && component.ControlEnabled(control.Port);

        protected override void Render(GH_Canvas canvas, Graphics graphics, GH_CanvasChannel channel)
        {
            base.Render(canvas, graphics, channel);
            if (channel != GH_CanvasChannel.Objects)
                return;
            foreach (var box in _boxes)
            {
                if (!box.Title.IsEmpty)
                    Painter.DrawTitle(graphics, box);
                if (box.Control is StepSliderControl step)
                {
                    var (value, wired, offered) = StepState(step, box);
                    StepSlider.Draw(graphics, box.Bounds, step, ReferenceEquals(box, _down) ? _stepped ?? value : value, offered, wired);
                    continue;
                }
                if (box.Control is IncidenceControl incidence)
                {
                    var (directions, current, wired) = IncidenceState(incidence);
                    IncidenceDials.Draw(graphics, box.Bounds, directions, current, wired);
                    continue;
                }
                var range = box.Control is SliderControl slider ? Component.SliderRange(slider) : ((double, double)?)null;
                Painter.Draw(graphics, box, Value(box), Selected, Owner.Locked, ReferenceEquals(box, _down), ReferenceEquals(box, _down) ? _dragged : null, range,
                    Component.ControlEnabled(box.Control.Port));
            }
            RenderExtra(canvas, graphics);
        }

        /// <summary>A step slider's stop (its input's value, or the nearest offered stop), whether a wire sets it, and the stops offered.</summary>
        private (string? value, bool wired, IReadOnlyCollection<string>? offered) StepState(StepSliderControl step, ControlBox box)
        {
            var offered = Component.OfferedOptionsOf(step.Port);
            var port = Component.ControlPort(step.Port);
            var (value, wired) = port == null ? (step.Default ?? step.Options[0], false) : ControlState.Of(port, step.Options, step.Default);
            return (wired ? value : StepSlider.Shown(box.Bounds, value, step.Options, offered), wired, offered);
        }

        /// <summary>An incidence control's directions, the one its input holds (the first until it holds one), and whether a wire sets it.</summary>
        private (IReadOnlyList<Incidence> directions, Incidence? current, bool wired) IncidenceState(IncidenceControl control)
        {
            var directions = Component.IncidencesOf(control.Port);
            var port = Component.ControlPort(control.Port);
            var (value, wired) = port == null ? ("0", false) : ControlState.Of(port, Array.Empty<string>(), "0");
            var index = int.TryParse(value, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var i) ? i : 0;
            return (directions, directions.FirstOrDefault(d => d.Index == index) ?? directions.FirstOrDefault(), wired);
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
                    return ControlState.Of(port, s.Options, ComponentBase.DefaultOf(s));
                case DropdownControl d when port != null:
                {
                    var (value, wired) = ControlState.Of(port, Component.OptionsOf(d).values, ComponentBase.DefaultOf(d));
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

        /// <summary>What a dropdown reads for <paramref name="value"/>: its label among the options offered now, else the first's ("All").</summary>
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
            if (!Pressable(Component, box.Control, port))
                return GH_ObjectResponse.Handled;
            switch (box.Control)
            {
                case StepSliderControl step when port != null:
                    _down = box;
                    _stepped = StepSlider.Nearest(box.Bounds, e.CanvasLocation.X, step.Options, Component.OfferedOptionsOf(step.Port));
                    sender.Invalidate();
                    return GH_ObjectResponse.Capture;
                case IncidenceControl incidence when port != null:
                {
                    var (directions, current, _) = IncidenceState(incidence);
                    var (theta, phi) = IncidenceDials.Hit(box.Bounds, e.CanvasLocation, directions, current);
                    if (theta == null && phi == null)
                        return GH_ObjectResponse.Handled;
                    var chosen = IncidenceDials.Choose(directions, current, theta, phi, _azimuth);
                    if (chosen == null)
                        return GH_ObjectResponse.Handled;
                    if (IncidenceDials.Ring(directions, chosen.Theta).Length > 1)
                        _azimuth = chosen.Phi;
                    if (chosen != current)
                        ControlState.Choose(Component, port, chosen.Index.ToString(System.Globalization.CultureInfo.InvariantCulture));
                    return GH_ObjectResponse.Handled;
                }
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
                    if (box.Control is SliderControl { Live: true } && port != null && _dragged is double start)
                    {
                        // one undo step for the whole drag, then the input follows the knob
                        if (Component.OnPingDocument() != null)
                            Component.RecordUndoEvent($"{port.Name}: drag");
                        ControlState.Scrub(Component, port, Invariant(start));
                    }
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
            if (_down?.Control is StepSliderControl step)
            {
                // the knob snaps from offered stop to offered stop; the input is set when it is let go
                _stepped = StepSlider.Nearest(_down.Bounds, e.CanvasLocation.X, step.Options, Component.OfferedOptionsOf(step.Port)) ?? _stepped;
                sender.Invalidate();
                return GH_ObjectResponse.Handled;
            }
            if (_down?.Control is SliderControl slider)
            {
                var before = _dragged;
                _dragged = Drag(slider, _down.Bounds, e.CanvasLocation.X);
                if (slider.Live && _dragged != before && _dragged is double value && Component.ControlPort(slider.Port) is { } live)
                    ControlState.Scrub(Component, live, Invariant(value));
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
            else if (box.Control is StepSliderControl && _stepped is string stop && Component.ControlPort(box.Control.Port) is { } stepPort)
            {
                var (shown, _, _) = StepState((StepSliderControl)box.Control, box);
                if (stop != shown)
                    ControlState.Choose(Component, stepPort, stop);
            }
            else if (box.Control is SliderControl { Live: true })
            {
                // set at every step of the drag already
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
            _stepped = null;
            sender.Invalidate();
            return GH_ObjectResponse.Release;
        }

        private static string Invariant(double value) => value.ToString(System.Globalization.CultureInfo.InvariantCulture);

        private double Drag(SliderControl slider, RectangleF bounds, float x)
        {
            var (min, max) = Component.SliderRange(slider);
            return SliderValue(slider, min, max, bounds, x);
        }

        public static double SliderValue(SliderControl slider, RectangleF bounds, float x) => SliderValue(slider, slider.Min, slider.Max, bounds, x);

        /// <summary>The value under <paramref name="x"/> on a slider from <paramref name="min"/> to <paramref name="max"/>, in its steps.</summary>
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
            var current = port != null ? ControlState.Of(port, values, ComponentBase.DefaultOf(dropdown)).value : Component.Choice(dropdown.Port);
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
        /// <summary>The size of a resize grip.</summary>
        public const float Grip = 8;

        /// <summary>A Grasshopper text capsule filling <paramref name="box"/>.</summary>
        public static void Capsule(Graphics g, RectangleF box, GH_Palette palette, string text, bool selected, bool locked)
        {
            var r = Rectangle.Round(box);
            using var capsule = GH_Capsule.CreateTextCapsule(r, r, palette, text, 2, 0);
            capsule.Render(g, selected, locked, false);
        }

        /// <summary>The resize grip in the bottom right corner of <paramref name="box"/>.</summary>
        public static void DrawGrip(Graphics g, RectangleF box)
        {
            g.DrawLine(Pens.Gray, box.Right - Grip, box.Bottom - 2, box.Right - 2, box.Bottom - Grip);
            g.DrawLine(Pens.Gray, box.Right - Grip / 2, box.Bottom - 2, box.Right - 2, box.Bottom - Grip / 2);
        }

        /// <summary>Whether <paramref name="point"/> is on the resize grip of <paramref name="box"/>.</summary>
        public static bool OnGrip(RectangleF box, PointF point) =>
            box.Contains(point) && point.X >= box.Right - Grip * 1.5f && point.Y >= box.Bottom - Grip * 1.5f;

        /// <summary>A transport glyph in the middle of <paramref name="box"/>: white, grey when the button cannot be pressed.</summary>
        public static void DrawGlyph(Graphics g, RectangleF box, Glyph glyph, bool enabled)
        {
            if (glyph == Glyph.None)
                return;
            var size = Math.Min(box.Height, box.Width) * 0.45f;
            var (cx, cy) = (box.Left + box.Width / 2, box.Top + box.Height / 2);
            using var brush = new SolidBrush(enabled ? Color.White : Color.FromArgb(150, 150, 150));
            var state = g.SmoothingMode;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            switch (glyph)
            {
                case Glyph.Play:
                    g.FillPolygon(brush, new[] { new PointF(cx - size * 0.45f, cy - size / 2), new PointF(cx - size * 0.45f, cy + size / 2), new PointF(cx + size * 0.55f, cy) });
                    break;
                case Glyph.Pause:
                    g.FillRectangle(brush, cx - size / 2, cy - size / 2, size * 0.35f, size);
                    g.FillRectangle(brush, cx + size * 0.15f, cy - size / 2, size * 0.35f, size);
                    break;
                case Glyph.Stop:
                    g.FillRectangle(brush, cx - size * 0.4f, cy - size * 0.4f, size * 0.8f, size * 0.8f);
                    break;
                case Glyph.Rewind:
                    // a bar, then two triangles pointing back to it
                    var h = size / 2;
                    g.FillRectangle(brush, cx - size * 0.75f, cy - h, size * 0.15f, size);
                    g.FillPolygon(brush, new[] { new PointF(cx - size * 0.55f, cy), new PointF(cx, cy - h), new PointF(cx, cy + h) });
                    g.FillPolygon(brush, new[] { new PointF(cx, cy), new PointF(cx + size * 0.55f, cy - h), new PointF(cx + size * 0.55f, cy + h) });
                    break;
            }
            g.SmoothingMode = state;
        }

        /// <summary>The input's name above its switch, dropdown or slider.</summary>
        public static void DrawTitle(Graphics g, ControlBox box)
        {
            using var format = new StringFormat { Trimming = StringTrimming.EllipsisCharacter, FormatFlags = StringFormatFlags.NoWrap, LineAlignment = StringAlignment.Far };
            // Grasshopper's font for the input and output names, so the titles read as the ports do
            g.DrawString(ControlLayout.Title(box.Control), GH_FontServer.StandardAdjusted, Brushes.Black, box.Title, format);
        }

        public static void Draw(Graphics g, ControlBox box, (string value, bool wired) state, bool selected, bool locked, bool down, double? dragged,
            (double min, double max)? range = null, bool enabled = true)
        {
            switch (box.Control)
            {
                case SwitchControl s:
                    for (var i = 0; i < s.Options.Length; i++)
                    {
                        var chosen = s.Options[i] == state.value;
                        var palette = chosen ? (state.wired ? GH_Palette.Grey : GH_Palette.Black) : GH_Palette.White;
                        Capsule(g, box.Segments[i], palette, ControlLayout.Label(s, s.Options[i]), selected, locked);
                    }
                    break;
                case MultiSwitchControl m:
                {
                    var lit = state.value.Split('\n');
                    for (var i = 0; i < m.Options.Length; i++)
                    {
                        var palette = lit.Contains(m.Options[i]) ? (state.wired ? GH_Palette.Grey : GH_Palette.Black) : GH_Palette.White;
                        Capsule(g, box.Segments[i], palette, m.Options[i], selected, locked);
                    }
                    break;
                }
                case DropdownControl:
                {
                    Capsule(g, box.Bounds, state.wired ? GH_Palette.Grey : GH_Palette.White, state.value, selected, locked || !enabled);
                    var x = box.Bounds.Right - 9;
                    var y = box.Bounds.Top + box.Bounds.Height / 2 - 2;
                    g.FillPolygon(enabled ? Brushes.Black : Brushes.Gray, new[] { new PointF(x, y), new PointF(x + 6, y), new PointF(x + 3, y + 4) });
                    break;
                }
                case ToggleControl toggle:
                {
                    // dark when on, grey when a wire sets it
                    var palette = state.value == "on" ? (state.wired ? GH_Palette.Grey : GH_Palette.Black) : GH_Palette.White;
                    Capsule(g, box.Bounds, palette, toggle.Label, selected, locked);
                    break;
                }
                case ButtonControl button:
                {
                    var glyph = button.Icon ? ControlLayout.GlyphOf(state.value) : Glyph.None;
                    Capsule(g, box.Bounds, down ? GH_Palette.Grey : GH_Palette.Black, glyph == Glyph.None ? state.value : "", selected, locked);
                    DrawGlyph(g, box.Bounds, glyph, true);
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
                    g.DrawString(string.Format(System.Globalization.CultureInfo.InvariantCulture, slider.Step >= 1 ? "{0:0} {1}" : "{0:0.0} {1}", value, slider.Unit).Trim(),
                        ChartPainter.Small, Brushes.Black, track.Left + 2, track.Top);
                    break;
                }
            }
        }
    }
}
