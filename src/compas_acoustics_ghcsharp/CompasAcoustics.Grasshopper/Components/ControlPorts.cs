using System;
using System.Collections.Generic;
using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// "Show control inputs": the inputs of the body's controls are hidden on a new component, so the body is
    /// the place to choose; shown, they take wires for parameter studies. A wired port never hides. A hidden port keeps
    /// its object, so the body still sets its value, and the value is saved with the document.
    /// </summary>
    public abstract partial class ComponentBase
    {
        private const string ShownKey = "control_inputs_shown";
        private const string HiddenKey = "control_inputs_hidden";
        private const string ValueKey = "control_value_";

        private readonly Dictionary<string, IGH_Param> _hidden = new();
        private bool _layoutRead;
        private IReadOnlyList<string>? _controlPorts;

        /// <summary>The inputs that a control on the body sets: every control on an input, buttons included.</summary>
        public IReadOnlyList<string> ControlPorts =>
            _controlPorts ??= Controls.Select(c => c.Port).Where(n => Spec.Inputs.Any(p => p.Name == n)).Distinct().ToList();

        public bool ControlPortsShown { get; private set; } = true;

        /// <summary>The input <paramref name="name"/>, shown or hidden.</summary>
        public IGH_Param? ControlPort(string name) => Params.Input.Find(p => p.Name == name) ?? _hidden.GetValueOrDefault(name);

        public bool IsHidden(IGH_Param port) => _hidden.ContainsValue(port);

        /// <summary>The value the body chose for the hidden input <paramref name="name"/> (the first, for a list), as text; null when it is not hidden or has none.</summary>
        public string? HiddenValue(string name) =>
            _hidden.TryGetValue(name, out var port) && ControlState.PersistentAll(port).FirstOrDefault() is string value && value.Length > 0 ? value : null;

        /// <summary>
        /// The hidden input <paramref name="name"/>'s value for a solve: null when it is not hidden, has none, or holds its
        /// declared default (the method's own, as for an unconnected input).
        /// </summary>
        internal T? HiddenAs<T>(string name) where T : struct
        {
            if (!_hidden.TryGetValue(name, out var port) || ControlState.Persistent(port) is not { } stored)
                return null;
            var value = (T)Convert.ChangeType(stored, typeof(T), System.Globalization.CultureInfo.InvariantCulture);
            var declared = Spec.Inputs.FirstOrDefault(p => p.Name == name)?.Default;
            return PortDefaults.Untouched(0, declared, value) ? null : value;
        }

        /// <summary>Every value of the hidden list input <paramref name="name"/>.</summary>
        internal List<T> HiddenList<T>(string name) =>
            _hidden.TryGetValue(name, out var port)
                ? ControlState.PersistentAll(port).Select(v => (T)Convert.ChangeType(v, typeof(T), System.Globalization.CultureInfo.InvariantCulture)).ToList()
                : new List<T>();

        /// <summary>Show or hide the control inputs (a wired one stays), and solve again.</summary>
        public void ShowControlPorts(bool show)
        {
            SetControlPorts(show, show ? Array.Empty<string>() : ControlPorts.Where(n => Params.Input.Find(p => p.Name == n)?.SourceCount == 0));
            if (OnPingDocument() != null)
                ExpireSolution(true);
        }

        private int SpecIndex(string name) => Spec.Inputs.ToList().FindIndex(p => p.Name == name);

        private void SetControlPorts(bool shown, IEnumerable<string> hide)
        {
            foreach (var (name, port) in _hidden.OrderBy(h => SpecIndex(h.Key)).ToList())
            {
                var index = Params.Input.Count(p => SpecIndex(p.Name) < SpecIndex(name));
                Params.RegisterInputParam(port, index);
                _hidden.Remove(name);
            }
            foreach (var name in hide)
                if (Params.Input.Find(p => p.Name == name) is { } port)
                {
                    Params.UnregisterInputParameter(port, false);
                    _hidden[name] = port;
                }
            ControlPortsShown = shown;
            Params.OnParametersChanged();
            Attributes?.ExpireLayout();
        }

        public override void AddedToDocument(GH_Document document)
        {
            // a new component, placed from the toolbar: one read from a file or pasted has its own layout and preview
            if (!_layoutRead)
            {
                if (ControlPorts.Count > 0)
                    SetControlPorts(false, ControlPorts);
                if (PreviewOffByDefault)
                    Hidden = true;
            }
            _layoutRead = true;
            base.AddedToDocument(document);
        }

        public override bool Write(GH_IWriter writer)
        {
            WriteChoices(writer);
            if (ControlPorts.Count > 0)
            {
                writer.SetBoolean(ShownKey, ControlPortsShown);
                writer.SetString(HiddenKey, string.Join("\n", _hidden.Keys));
                foreach (var (name, port) in _hidden)
                    if (ControlState.PersistentAll(port) is { Length: > 0 } values)
                        writer.SetString(ValueKey + name, string.Join("\n", values));
            }
            return base.Write(writer);
        }

        public override bool Read(GH_IReader reader)
        {
            _layoutRead = true;
            ReadChoices(reader);
            if (ControlPorts.Count == 0 || !reader.ItemExists(ShownKey))
                return base.Read(reader);
            var hidden = reader.GetString(HiddenKey).Split('\n', StringSplitOptions.RemoveEmptyEntries);
            // the ports must be those that were saved before Grasshopper reads them by index
            SetControlPorts(reader.GetBoolean(ShownKey), hidden);
            var read = base.Read(reader);
            foreach (var name in hidden)
                if (reader.ItemExists(ValueKey + name) && _hidden.TryGetValue(name, out var port))
                    ControlState.Assign(port, reader.GetString(ValueKey + name).Split('\n'));
            return read;
        }
    }
}
