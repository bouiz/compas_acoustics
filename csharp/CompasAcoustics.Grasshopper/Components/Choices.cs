using System;
using System.Collections.Generic;
using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// Choices the component holds itself, without an input: a switch or dropdown on the body whose port is not an input
    /// (Auralise's level), or a menu entry (Analyse IR's time scale on its time output). Saved with the document.
    /// </summary>
    public abstract partial class ComponentBase
    {
        private const string ChoiceKey = "choice_";
        private readonly Dictionary<string, string> _choices = new();

        /// <summary>Choices offered elsewhere than on the body, by name, the first option the default.</summary>
        protected virtual IReadOnlyDictionary<string, string[]> MenuChoices => new Dictionary<string, string[]>();

        /// <summary>Every choice without an input, with its options.</summary>
        public IReadOnlyDictionary<string, string[]> Choices
        {
            get
            {
                var choices = new Dictionary<string, string[]>(MenuChoices);
                foreach (var control in Controls)
                    if (CurrentSpec.Inputs.All(p => p.Name != control.Port))
                        switch (control)
                        {
                            case SwitchControl s:
                                choices[s.Port] = s.Options;
                                break;
                            case DropdownControl d:
                                choices[d.Port] = d.Options;
                                break;
                        }
                return choices;
            }
        }

        public bool IsChoice(string name) => Choices.ContainsKey(name);

        /// <summary>The tooltip of the control of ``name``: its input's description, or the options of a choice without an input.</summary>
        public (string title, string text) ControlTooltip(string name)
        {
            if (CurrentSpec.Inputs.FirstOrDefault(p => p.Name == name) is { } port)
                return (name, port.Description);
            if (Choices.TryGetValue(name, out var options) && options.Length > 0)
                return (name, options.Length == 1 ? options[0] : $"{string.Join(", ", options.Take(options.Length - 1))} or {options[^1]}");
            return (name, "");
        }

        /// <summary>The chosen option of ``name``, or its default: the switch's ``Default``, else its first option.</summary>
        public virtual string Choice(string name)
        {
            if (_choices.TryGetValue(name, out var value))
                return value;
            if (Controls.OfType<SwitchControl>().FirstOrDefault(c => c.Port == name)?.Default is string preset)
                return preset;
            return Choices.TryGetValue(name, out var options) && options.Length > 0 ? options[0] : "";
        }

        /// <summary>Choose ``value`` for ``name`` (an option of it), as an undoable change, and solve again.</summary>
        public virtual void SetChoice(string name, string value)
        {
            if (!Choices.TryGetValue(name, out var options) || !options.Contains(value) || Choice(name) == value)
                return;
            var document = OnPingDocument();
            if (document != null)
                RecordUndoEvent($"{name}: {value}");
            _choices[name] = value;
            if (document != null)
                ExpireSolution(true);
        }

        private void WriteChoices(GH_IWriter writer)
        {
            foreach (var (name, value) in _choices)
                writer.SetString(ChoiceKey + name, value);
        }

        private void ReadChoices(GH_IReader reader)
        {
            foreach (var (name, options) in Choices)
                if (reader.ItemExists(ChoiceKey + name) && options.Contains(reader.GetString(ChoiceKey + name)))
                    _choices[name] = reader.GetString(ChoiceKey + name);
        }
    }
}
