using System;
using System.Collections.Generic;
using System.Linq;
using GH_IO.Serialization;
using GH_IO.Types;
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
                // ports outside the spec (a removed input read from an old file) take no place in the count
                var index = Params.Input.Count(p => SpecIndex(p.Name) >= 0 && SpecIndex(p.Name) < SpecIndex(name));
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

        /// <summary>
        /// After a read of an old layout that set inputs aside and put them back shown: hide those the body sets, unwired, when
        /// the definition saved its control inputs hidden.
        /// </summary>
        protected void HideControlPortsAsRead()
        {
            if (!ControlPortsShown)
                SetControlPorts(false, ControlPorts.Where(n => Params.Input.Find(p => p.Name == n)?.SourceCount == 0).ToList());
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
            var read = ReadPorts(reader);
            TakeTheSpecsOutputNames();
            TakeTheSpecsDescriptions();
            return read;
        }

        /// <summary>
        /// The outputs' names of the installed plugin, by position: Grasshopper restores a port's names from the file, and an
        /// output a file saved under its former name would never be found by name. A name chosen on the canvas stays.
        /// </summary>
        private void TakeTheSpecsOutputNames()
        {
            if (Params.Output.Count != Spec.Outputs.Count)
                return;
            for (var i = 0; i < Params.Output.Count; i++)
            {
                var (port, name) = (Params.Output[i], Spec.Outputs[i].Name);
                if (port.NickName == port.Name)
                    port.NickName = name;
                port.Name = name;
            }
        }

        private bool ReadPorts(GH_IReader reader)
        {
            if (ControlPorts.Count == 0 || !reader.ItemExists(ShownKey))
                return base.Read(reader);
            var hidden = reader.GetString(HiddenKey).Split('\n', StringSplitOptions.RemoveEmptyEntries);
            // the ports must be those that were saved before Grasshopper reads them by index
            SetControlPorts(reader.GetBoolean(ShownKey), hidden);
            var read = base.Read(reader);
            foreach (var name in hidden)
                if (reader.ItemExists(ValueKey + name) && _hidden.TryGetValue(name, out var port))
                    ControlState.Assign(port, reader.GetString(ValueKey + name).Split('\n'));
            // an input saved hidden behind a control the body no longer has shows again, with its value
            if (_hidden.Keys.Any(n => !ControlPorts.Contains(n)))
                SetControlPorts(ControlPortsShown, _hidden.Keys.Where(ControlPorts.Contains).ToList());
            return read;
        }

        /// <summary>
        /// The descriptions of the installed plugin, not those a file saved with an earlier release: Grasshopper restores a
        /// component's and its ports' descriptions from the file.
        /// </summary>
        private void TakeTheSpecsDescriptions()
        {
            Description = Spec.Description;
            foreach (var port in Params.Input.Concat(_hidden.Values))
                if (Spec.Inputs.FirstOrDefault(p => p.Name == port.Name) is { } spec)
                    port.Description = spec.Description;
            foreach (var port in Params.Output)
                if (Spec.Outputs.FirstOrDefault(p => p.Name == port.Name) is { } spec)
                    port.Description = spec.Description;
        }

        /// <summary>The boolean in the saved persistent data of the input <paramref name="name"/>, if it held one.</summary>
        protected static bool? SavedBoolean(GH_IReader reader, string name)
        {
            for (var i = 0; reader.ChunkExists("param_input", i); i++)
            {
                var chunk = reader.FindChunk("param_input", i);
                if (chunk.ItemExists("Name") && chunk.GetString("Name") == name)
                    return chunk.FindChunk("PersistentData") is { } data ? FirstBoolean(data) : null;
            }
            return null;
        }

        private static bool? FirstBoolean(GH_IReader chunk)
        {
            foreach (var item in chunk.Items)
                if (item.Type == GH_Types.gh_bool)
                    return item._bool;
            foreach (var child in chunk.Chunks)
                if (child is GH_IReader inner && FirstBoolean(inner) is bool value)
                    return value;
            return null;
        }

        /// <summary>Whether the definition saved the input <paramref name="name"/> as a port.</summary>
        protected static bool SavedInput(GH_IReader reader, string name)
        {
            for (var i = 0; reader.ChunkExists("param_input", i); i++)
            {
                var chunk = reader.FindChunk("param_input", i);
                if (chunk.ItemExists("Name") && chunk.GetString("Name") == name)
                    return true;
            }
            return false;
        }
    }
}
