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

    /// <summary>A row of two to four options, one chosen; ``Default`` (else the first) when none is.</summary>
    public sealed record SwitchControl(string Port, string[] Options, string? Default = null) : Control(Port);

    /// <summary>The chosen option, with a menu of the others.</summary>
    public sealed record DropdownControl(string Port, string[] Options) : Control(Port);

    /// <summary>A trigger: a press runs the component once with ``Pressed(Port)`` true. ``Port`` need not be an input (``cancel``).</summary>
    public sealed record ButtonControl(string Port, string Label) : Control(Port);

    /// <summary>One button that switches a choice on and off (``on``/``off``): dark when on.</summary>
    public sealed record ToggleControl(string Port, string Label) : Control(Port);

    /// <summary>A number from ``Min`` to ``Max``, dragged.</summary>
    public sealed record SliderControl(string Port, double Min, double Max, string Unit) : Control(Port);

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

        /// <summary>Whether a control has its title above it: switches and dropdowns; a button carries its label.</summary>
        public static bool HasTitle(Control control) => control is SwitchControl or DropdownControl;

        /// <summary>The title of a control: its input's name, capitalised, ``time_scale`` as ``Time scale``.</summary>
        public static string Title(Control control)
        {
            var name = control.Port.Replace('_', ' ');
            return name.Length == 0 ? name : char.ToUpperInvariant(name[0]) + name.Substring(1);
        }

        public static float Height(IReadOnlyCollection<Control> controls) =>
            controls.Count == 0 ? 0 : controls.Sum(c => RowHeight + (HasTitle(c) ? TitleHeight : 0)) + (controls.Count + 1) * Margin;

        public static List<ControlBox> Arrange(IReadOnlyList<Control> controls, RectangleF body)
        {
            var boxes = new List<ControlBox>();
            var y = body.Top + Margin;
            foreach (var control in controls)
            {
                var title = RectangleF.Empty;
                if (HasTitle(control))
                {
                    title = new RectangleF(body.Left + Margin, y, body.Width - 2 * Margin, TitleHeight);
                    y += TitleHeight;
                }
                var row = new RectangleF(body.Left + Margin, y, body.Width - 2 * Margin, RowHeight);
                var segments = Array.Empty<RectangleF>();
                if (control is SwitchControl choice && choice.Options.Length > 0)
                {
                    var width = row.Width / choice.Options.Length;
                    segments = Enumerable.Range(0, choice.Options.Length)
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
        /// <summary>The value a choice shows: a wired value wins (and the control greys), else the port's own value, else the first option.</summary>
        public static (string value, bool wired) Resolve(int sourceCount, string? wiredValue, string? persistent, string[] options)
        {
            if (sourceCount > 0)
                return (wiredValue ?? "", true);
            return (string.IsNullOrEmpty(persistent) ? options.FirstOrDefault() ?? "" : persistent!, false);
        }

        public static (string value, bool wired) Of(IGH_Param port, string[] options)
        {
            var wired = port.SourceCount > 0 ? port.VolatileData.AllData(true).FirstOrDefault()?.ToString() : null;
            return Resolve(port.SourceCount, wired, Persistent(port)?.ToString(), options);
        }

        /// <summary>The port's own (persistent) value, null when it has none.</summary>
        public static object? Persistent(IGH_Param port) => port switch
        {
            GH_PersistentParam<GH_String> s => s.PersistentData.AllData(true).OfType<GH_String>().FirstOrDefault()?.Value,
            GH_PersistentParam<GH_Number> n => n.PersistentData.AllData(true).OfType<GH_Number>().FirstOrDefault()?.Value,
            GH_PersistentParam<GH_Boolean> b => b.PersistentData.AllData(true).OfType<GH_Boolean>().FirstOrDefault()?.Value,
            _ => null,
        };

        /// <summary>Make ``value`` the port's own value, as an undoable change, and recompute the component.</summary>
        public static void Choose(GH_Component component, IGH_Param port, string value)
        {
            if (port is not GH_PersistentParam<GH_String> text)
                return;
            if (component.OnPingDocument() != null)
                component.RecordUndoEvent($"{port.Name}: {value}");
            text.PersistentData.Clear();
            if (component is ComponentBase owner && owner.IsHidden(port))
                // a hidden port has no owner to expire: set its data, the component solves again
                text.PersistentData.Append(new GH_String(value));
            else
                text.SetPersistentData(new GH_String(value));
            component.ExpireSolution(true);
        }

        public static void ChooseNumber(GH_Component component, IGH_Param port, double value)
        {
            if (port is not GH_PersistentParam<GH_Number> number)
                return;
            if (component.OnPingDocument() != null)
                component.RecordUndoEvent($"{port.Name}: {value}");
            number.PersistentData.Clear();
            number.SetPersistentData(new GH_Number(value));
            component.ExpireSolution(true);
        }

        /// <summary>A button press: the component's next solution sees ``Pressed(port)``.</summary>
        public static void Press(ComponentBase component, string port)
        {
            component.MarkPressed(port);
            component.ExpireSolution(true);
        }
    }
}
