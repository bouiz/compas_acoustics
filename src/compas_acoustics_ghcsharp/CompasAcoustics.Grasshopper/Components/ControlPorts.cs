using System;
using System.Collections.Generic;
using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;

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

        /// <summary>
        /// Written with every component: a document without it was saved before the preview defaults took effect (v2: a
        /// build in between kept Grasshopper's "on" for components saved before them).
        /// </summary>
        public const string PreviewKey = "preview_default_v2";

        private readonly Dictionary<string, IGH_Param> _hidden = new();
        private bool _layoutRead;

        /// <summary>The inputs that a control on the body sets: every control on an input, buttons included (Run, Play).</summary>
        public IReadOnlyList<string> ControlPorts =>
            Controls.Select(c => c.Port).Where(n => CurrentSpec.Inputs.Any(p => p.Name == n)).Distinct().ToList();

        public bool ControlPortsShown { get; private set; } = true;

        /// <summary>The input ``name``, shown or hidden.</summary>
        public IGH_Param? ControlPort(string name) => Params.Input.Find(p => p.Name == name) ?? _hidden.GetValueOrDefault(name);

        public bool IsHidden(IGH_Param port) => _hidden.ContainsValue(port);

        /// <summary>The value the body chose for the hidden input ``name`` (the first, for a list), as text; null when it is not hidden or has none.</summary>
        public string? HiddenValue(string name) =>
            _hidden.TryGetValue(name, out var port) && ControlState.PersistentAll(port).FirstOrDefault() is string value && value.Length > 0 ? value : null;

        /// <summary>
        /// The hidden input ``name``'s value for a solve: null when it is not hidden, has none, or holds its declared default
        /// (the method's own, as for an unconnected input).
        /// </summary>
        internal T? HiddenAs<T>(string name) where T : struct
        {
            if (!_hidden.TryGetValue(name, out var port) || ControlState.Persistent(port) is not { } stored)
                return null;
            var value = (T)Convert.ChangeType(stored, typeof(T), System.Globalization.CultureInfo.InvariantCulture);
            var declared = CurrentSpec.Inputs.FirstOrDefault(p => p.Name == name)?.Default;
            return PortDefaults.Untouched(0, declared, value) ? null : value;
        }

        /// <summary>Every value of the hidden list input ``name``.</summary>
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

        private int SpecIndex(string name) => CurrentSpec.Inputs.ToList().FindIndex(p => p.Name == name);

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
            writer.SetBoolean(PreviewKey, true);
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
            var read = ReadLayout(reader);
            // saved before the preview defaults: Grasshopper's "on" was never chosen, the surfaces would draw over Rhino's
            if (PreviewOffByDefault && !reader.ItemExists(PreviewKey))
                Hidden = true;
            return read;
        }

        /// <summary>
        /// Grasshopper's own read, which restores the inputs by position, made to match them by name: a definition saved
        /// before an input was added (Material Library's absorption and scattering) or removed (Source's directivity,
        /// Auralise's level) would otherwise read an input's data into its neighbour, or report a corrupt archive. A saved
        /// input that is gone is read into a stand-in and dropped; an input the file does not know keeps its defaults.
        /// </summary>
        private bool ReadParameters(GH_IReader reader)
        {
            var saved = new List<string>();
            while (reader.ChunkExists("param_input", saved.Count))
            {
                var chunk = reader.FindChunk("param_input", saved.Count);
                saved.Add(chunk.ItemExists("Name") ? chunk.GetString("Name") : "");
            }
            var current = Params.Input.ToList();
            if (saved.SequenceEqual(current.Select(p => p.Name)) || !reader.ChunkExists("param_output", 0))
                return base.Read(reader);
            foreach (var param in current)
                Params.UnregisterInputParameter(param, false);
            foreach (var name in saved)
                Params.RegisterInputParam(current.FirstOrDefault(p => p.Name == name) ?? new Param_GenericObject { Name = name, NickName = name });
            var read = base.Read(reader);
            foreach (var param in Params.Input.ToList())
                Params.UnregisterInputParameter(param, false);
            foreach (var param in current)
                Params.RegisterInputParam(param);
            Params.OnParametersChanged();
            return read;
        }

        private bool ReadLayout(GH_IReader reader)
        {
            _layoutRead = true;
            ReadChoices(reader);
            if (ControlPorts.Count == 0 || !reader.ItemExists(ShownKey))
            {
                // saved before the toggle existed: every port was there
                if (_hidden.Count > 0)
                    SetControlPorts(true, Array.Empty<string>());
                return ReadParameters(reader);
            }
            var hidden = reader.GetString(HiddenKey).Split('\n', StringSplitOptions.RemoveEmptyEntries);
            // the ports must be those that were saved before Grasshopper reads them by index
            SetControlPorts(reader.GetBoolean(ShownKey), hidden);
            var read = ReadParameters(reader);
            foreach (var name in hidden)
                if (reader.ItemExists(ValueKey + name) && _hidden.TryGetValue(name, out var port))
                    ControlState.Assign(port, reader.GetString(ValueKey + name).Split('\n'));
            return read;
        }
    }
}
