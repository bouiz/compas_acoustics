using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>A control drawn on a component's body. ``Port`` is the input it sets; a wire into that input wins.</summary>
    public abstract record Control(string Port);

    /// <summary>A row of two to four options, one chosen; ``Default`` (else the first) when none is; ``Labels`` what the segments read.</summary>
    public sealed record SwitchControl(string Port, string[] Options, string? Default = null, string[]? Labels = null) : Control(Port);

    /// <summary>
    /// The chosen option, with a menu of the others. ``Options`` are the input's values (empty clears it), ``Labels`` what
    /// they read; ``Title`` replaces the input's name above it. A component may give the options itself (<see cref="ComponentBase.OptionsOf"/>).
    /// </summary>
    public sealed record DropdownControl(string Port, string[] Options, string? Default = null, string[]? Labels = null, string? Title = null) : Control(Port);

    /// <summary>Several options lit at once, for a list input: none lit is ``Default``.</summary>
    public sealed record MultiSwitchControl(string Port, string[] Options, string[] Default) : Control(Port);

    /// <summary>
    /// A trigger: a press runs the component once with ``Pressed(Port)`` true. ``Port`` need not be an input (``cancel``).
    /// ``Inline`` buttons that follow each other share one row; an ``Icon`` button draws the glyph of what it reads.
    /// </summary>
    public sealed record ButtonControl(string Port, string Label, bool Inline = false, bool Icon = false) : Control(Port);

    /// <summary>The transport glyphs: a triangle, two bars, a square, a bar and two triangles back.</summary>
    public enum Glyph { None, Play, Pause, Stop, Rewind }

    /// <summary>
    /// One button that switches on and off, dark when on: a bool input (``Default`` until chosen), or a choice the component
    /// holds (``on``/``off``). Toggles marked ``Inline`` that follow each other share one row.
    /// </summary>
    public sealed record ToggleControl(string Port, string Label, bool Inline = false, bool Default = true) : Control(Port);

    /// <summary>
    /// A number from ``Min`` to ``Max`` in steps of ``Step``, dragged; a component may give the range itself
    /// (<see cref="ComponentBase.SliderRange"/>). ``Live``: the input follows the knob while it is dragged (Billiard's time),
    /// else it is set when the knob is let go. ``Title`` replaces the input's name above it.
    /// </summary>
    public sealed record SliderControl(string Port, double Min, double Max, string Unit, double Step = 0.1, bool Live = false, string? Title = null) : Control(Port);

    /// <summary>Where a control is drawn: its row, and for a switch the box of each option.</summary>
    /// <summary>``Title``: the row above a switch or dropdown that names its input; empty for a button.</summary>
    public sealed record ControlBox(Control Control, RectangleF Bounds, RectangleF[] Segments, RectangleF Title = default);

    /// <summary>The rows of a component's controls, below its ports.</summary>
    public static class ControlLayout
    {
        public const float RowHeight = 20;
        public const float Margin = 3;
        /// <summary>The title row: Grasshopper's font for input and output names fits in it.</summary>
        public const float TitleHeight = 16;
        /// <summary>Between a title and its control.</summary>
        public const float TitleGap = 2;
        /// <summary>
        /// Above a row without a title (toggles, buttons) that follows another control: as far as a title's words stand from
        /// the control above it, its margin and the empty top of its row.
        /// </summary>
        public const float ButtonGap = 10;

        /// <summary>Whether a control has its title above it: switches, dropdowns and sliders; a button or toggle carries its label.</summary>
        public static bool HasTitle(Control control) => control is SwitchControl or DropdownControl or MultiSwitchControl or SliderControl;

        /// <summary>The title of a control: a dropdown's own, else its input's name, capitalised, ``time_scale`` as ``Time scale``.</summary>
        public static string Title(Control control)
        {
            switch (control)
            {
                case DropdownControl { Title: string title }:
                    return title;
                case SliderControl { Title: string title }:
                    return title;
            }
            var name = control.Port.Replace('_', ' ');
            return name.Length == 0 ? name : char.ToUpperInvariant(name[0]) + name.Substring(1);
        }

        /// <summary>What an option of a switch or dropdown reads: its label, else the value itself.</summary>
        public static string Label(Control control, string value)
        {
            var (options, labels) = control switch
            {
                SwitchControl s => (s.Options, s.Labels),
                DropdownControl d => (d.Options, d.Labels),
                _ => (Array.Empty<string>(), null),
            };
            var i = Array.IndexOf(options, value);
            return labels != null && i >= 0 && i < labels.Length ? labels[i] : value;
        }

        private static bool IsInline(Control? control) => control is ToggleControl { Inline: true } or ButtonControl { Inline: true };

        /// <summary>Whether ``control`` goes on the row of ``previous``: inline toggles with toggles, inline buttons with buttons.</summary>
        private static bool InRow(Control control, Control? previous) =>
            IsInline(control) && IsInline(previous) && control.GetType() == previous!.GetType();

        /// <summary>The glyph an icon button draws for what it reads (Play reads Stop while playing); None for a word.</summary>
        public static Glyph GlyphOf(string label) => label switch
        {
            "Play" => Glyph.Play,
            "Pause" => Glyph.Pause,
            "Stop" => Glyph.Stop,
            "Rewind" => Glyph.Rewind,
            _ => Glyph.None,
        };

        public static float Height(IReadOnlyList<Control> controls) =>
            controls.Count == 0 ? 0 : Arrange(controls, new RectangleF(0, 0, 1, 0)).Max(b => b.Bounds.Bottom) + Margin;

        public static List<ControlBox> Arrange(IReadOnlyList<Control> controls, RectangleF body)
        {
            var boxes = new List<ControlBox>();
            var y = body.Top + Margin;
            Control? previous = null;
            for (var c = 0; c < controls.Count; c++)
            {
                var control = controls[c];
                if (InRow(control, previous))
                    continue;
                if (previous != null && control is (ButtonControl or ToggleControl))
                    y += ButtonGap - Margin;
                previous = control;
                if (IsInline(control))
                {
                    // the toggles, or the buttons, that follow share the row, side by side
                    var count = 1;
                    while (c + count < controls.Count && InRow(controls[c + count], controls[c + count - 1]))
                        count++;
                    var width = (body.Width - 2 * Margin) / count;
                    for (var k = 0; k < count; k++)
                        boxes.Add(new ControlBox(controls[c + k], new RectangleF(body.Left + Margin + k * width, y, width, RowHeight), Array.Empty<RectangleF>()));
                    previous = controls[c + count - 1];
                    y += RowHeight + Margin;
                    continue;
                }
                var title = RectangleF.Empty;
                if (HasTitle(control))
                {
                    title = new RectangleF(body.Left + Margin, y, body.Width - 2 * Margin, TitleHeight);
                    y += TitleHeight + TitleGap;
                }
                var row = new RectangleF(body.Left + Margin, y, body.Width - 2 * Margin, RowHeight);
                var segments = Array.Empty<RectangleF>();
                var options = control switch { SwitchControl s => s.Options, MultiSwitchControl m => m.Options, _ => Array.Empty<string>() };
                if (options.Length > 0)
                {
                    var width = row.Width / options.Length;
                    segments = Enumerable.Range(0, options.Length)
                        .Select(i => new RectangleF(row.Left + i * width, row.Top, width, row.Height)).ToArray();
                }
                boxes.Add(new ControlBox(control, row, segments, title));
                y += RowHeight + Margin;
            }
            return boxes;
        }

        /// <summary>What a control reads: its value, or a button's label (the input's name is the title above).</summary>
        public static string Text(Control control, string value) => value;

        /// <summary>The control under ``point``, and the option of a switch (-1 for the others); null outside every control.</summary>
        public static (ControlBox box, int segment)? HitTest(IEnumerable<ControlBox> boxes, PointF point)
        {
            foreach (var box in boxes)
            {
                if (!box.Bounds.Contains(point))
                    continue;
                for (var i = 0; i < box.Segments.Length; i++)
                    if (box.Segments[i].Contains(point))
                        return (box, i);
                return (box, -1);
            }
            return null;
        }
    }

    /// <summary>What a control shows, and how it sets its port.</summary>
    public static class ControlState
    {
        private static string Invariant(object value) => Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) ?? "";

        /// <summary>The value a choice shows: a wired value wins (and the control greys), else the port's own value, else ``fallback`` or the first option.</summary>
        public static (string value, bool wired) Resolve(int sourceCount, string? wiredValue, string? persistent, string[] options, string? fallback = null)
        {
            if (sourceCount > 0)
                return (wiredValue ?? "", true);
            return (string.IsNullOrEmpty(persistent) ? fallback ?? options.FirstOrDefault() ?? "" : persistent!, false);
        }

        public static (string value, bool wired) Of(IGH_Param port, string[] options, string? fallback = null)
        {
            var wired = port.SourceCount > 0 ? port.VolatileData.AllData(true).FirstOrDefault()?.ScriptVariable() is { } v ? Invariant(v) : null : null;
            var persistent = Persistent(port);
            return Resolve(port.SourceCount, wired, persistent == null ? null : Invariant(persistent), options, fallback);
        }

        /// <summary>A toggle on a bool input: on or off, and whether a wire sets it; ``fallback`` until the input has a value.</summary>
        public static (bool on, bool wired) Toggle(IGH_Param port, bool fallback)
        {
            var (value, wired) = Of(port, new[] { "True", "False" }, fallback ? "True" : "False");
            return (string.Equals(value, "True", StringComparison.OrdinalIgnoreCase), wired);
        }

        private static IEnumerable<IGH_Goo> Data(IGH_Param port) => port switch
        {
            GH_PersistentParam<GH_String> s => s.PersistentData.AllData(true),
            GH_PersistentParam<GH_Integer> i => i.PersistentData.AllData(true),
            GH_PersistentParam<GH_Number> n => n.PersistentData.AllData(true),
            GH_PersistentParam<GH_Boolean> b => b.PersistentData.AllData(true),
            _ => Array.Empty<IGH_Goo>(),
        };

        /// <summary>The port's own (persistent) value, null when it has none.</summary>
        public static object? Persistent(IGH_Param port) => Data(port).FirstOrDefault()?.ScriptVariable();

        /// <summary>Every value of the port's own data, as text (a list input).</summary>
        public static string[] PersistentAll(IGH_Param port) => Data(port).Select(g => Invariant(g.ScriptVariable())).ToArray();

        /// <summary>The options lit on a list input: its own values, or ``fallback`` when it has none.</summary>
        public static string[] Members(IGH_Param port, string[] fallback)
        {
            var values = PersistentAll(port);
            return values.Length > 0 ? values : fallback;
        }

        private static IGH_Goo? Goo(IGH_Param port, string value)
        {
            var c = System.Globalization.CultureInfo.InvariantCulture;
            return port switch
            {
                GH_PersistentParam<GH_String> => new GH_String(value),
                GH_PersistentParam<GH_Integer> when int.TryParse(value, System.Globalization.NumberStyles.Integer, c, out var i) => new GH_Integer(i),
                GH_PersistentParam<GH_Number> when double.TryParse(value, System.Globalization.NumberStyles.Float, c, out var d) => new GH_Number(d),
                GH_PersistentParam<GH_Boolean> when bool.TryParse(value, out var b) => new GH_Boolean(b),
                _ => null,
            };
        }

        /// <summary>Give the port ``values`` as its own data (none: empty) without solving: a file read, or a solve in progress.</summary>
        public static void Assign(IGH_Param port, IEnumerable<string> values)
        {
            var goo = values.Select(v => Goo(port, v)).Where(g => g != null).ToList();
            switch (port)
            {
                case GH_PersistentParam<GH_String> s:
                    s.PersistentData.Clear();
                    foreach (var g in goo) s.PersistentData.Append((GH_String)g!);
                    break;
                case GH_PersistentParam<GH_Integer> i:
                    i.PersistentData.Clear();
                    foreach (var g in goo) i.PersistentData.Append((GH_Integer)g!);
                    break;
                case GH_PersistentParam<GH_Number> n:
                    n.PersistentData.Clear();
                    foreach (var g in goo) n.PersistentData.Append((GH_Number)g!);
                    break;
                case GH_PersistentParam<GH_Boolean> b:
                    b.PersistentData.Clear();
                    foreach (var g in goo) b.PersistentData.Append((GH_Boolean)g!);
                    break;
            }
        }

        /// <summary>Make ``values`` the port's own data (an empty value clears it), as an undoable change, and recompute the component.</summary>
        public static void ChooseAll(GH_Component component, IGH_Param port, string[] values)
        {
            values = values.Where(v => v.Length > 0).ToArray();
            if (values.Any(v => Goo(port, v) == null))
                return;
            if (component.OnPingDocument() != null)
                component.RecordUndoEvent($"{port.Name}: {string.Join(", ", values)}");
            Assign(port, values);
            Expire(component, port);
        }

        /// <summary>
        /// A step of a live slider's drag: ``value`` becomes the port's own and the component solves again, without an undo
        /// step (the drag records one when it starts).
        /// </summary>
        public static void Scrub(GH_Component component, IGH_Param port, string value)
        {
            if (Goo(port, value) == null)
                return;
            Assign(port, new[] { value });
            Expire(component, port);
        }

        private static void Expire(GH_Component component, IGH_Param port)
        {
            // a shown input expires itself, and with it the component: expiring only the component clears only its
            // outputs, and the input would keep what it collected. A hidden input is not wired to the component.
            if (component is ComponentBase owner && owner.IsHidden(port))
                component.ExpireSolution(true);
            else
            {
                port.ExpireSolution(false);
                component.ExpireSolution(true);
            }
        }

        /// <summary>Make ``value`` the port's own value (empty clears it), as an undoable change, and recompute the component.</summary>
        public static void Choose(GH_Component component, IGH_Param port, string value) => ChooseAll(component, port, new[] { value });

        public static void ChooseNumber(GH_Component component, IGH_Param port, double value) => Choose(component, port, Invariant(value));

        /// <summary>Light or unlight ``option`` on a list input, from what it shows (``fallback`` when it has no values).</summary>
        public static void ToggleMember(GH_Component component, IGH_Param port, string option, string[] fallback)
        {
            var lit = Members(port, fallback).ToList();
            if (!lit.Remove(option))
                lit.Add(option);
            ChooseAll(component, port, lit.OrderBy(v => double.TryParse(v, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var d) ? d : 0).ToArray());
        }

        /// <summary>A button press: the component's next solution sees ``Pressed(port)``.</summary>
        public static void Press(ComponentBase component, string port)
        {
            component.MarkPressed(port);
            component.ExpireSolution(true);
        }
    }
}
