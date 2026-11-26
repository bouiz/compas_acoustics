using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using GH_IO.Serialization;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// Choices the component holds itself, without an input: a switch, dropdown or toggle on the body whose port is not an
    /// input, or a menu entry. Saved with the document.
    /// </summary>
    public abstract partial class ComponentBase
    {
        private const string ChoiceKey = "choice_";
        private readonly Dictionary<string, string> _choices = new();
        private IReadOnlyDictionary<string, string[]>? _choiceOptions;

        /// <summary>Choices offered elsewhere than on the body, by name, the first option the default.</summary>
        protected virtual IReadOnlyDictionary<string, string[]> MenuChoices => new Dictionary<string, string[]>();

        /// <summary>Every choice without an input, with its options.</summary>
        public IReadOnlyDictionary<string, string[]> Choices => _choiceOptions ??= ChoiceOptions();

        private Dictionary<string, string[]> ChoiceOptions()
        {
            var choices = new Dictionary<string, string[]>(MenuChoices);
            foreach (var control in Controls)
                if (Spec.Inputs.All(p => p.Name != control.Port))
                    switch (control)
                    {
                        case SwitchControl s:
                            choices[s.Port] = s.Options;
                            break;
                        case DropdownControl d:
                            choices[d.Port] = d.Options;
                            break;
                        case ToggleControl t:
                            choices[t.Port] = new[] { "on", "off" };
                            break;
                    }
            return choices;
        }

        /// <summary>The option a switch or dropdown declares as its default; null for another control or none.</summary>
        internal static string? DefaultOf(Control control) => control switch
        {
            SwitchControl s => s.Default,
            DropdownControl d => d.Default,
            _ => null,
        };

        public bool IsChoice(string name) => Choices.ContainsKey(name);

        /// <summary>
        /// The tooltip of the control of <paramref name="name"/>: its input's description, or the options of a choice without
        /// an input; and its default, unless the description names it already.
        /// </summary>
        public (string title, string text) ControlTooltip(string name)
        {
            var text = "";
            // a button without an input: what it does now
            if (Controls.FirstOrDefault(c => c.Port == name) is ButtonControl button && Spec.Inputs.All(p => p.Name != name))
                return (ButtonLabel(name) ?? button.Label, "");
            if (Spec.Inputs.FirstOrDefault(p => p.Name == name) is { } port)
                text = port.Description;
            else if (Choices.TryGetValue(name, out var options) && options.Length > 0)
                text = options.Length == 1 ? options[0] : $"{string.Join(", ", options.Take(options.Length - 1))} or {options[^1]}";
            if (text.Contains("default", StringComparison.OrdinalIgnoreCase) || DefaultLabel(name) is not string shown)
                return (name, text);
            return (name, text.Length == 0 ? $"(default {shown})" : $"{text} (default {shown})");
        }

        /// <summary>What the control of <paramref name="name"/> shows untouched: its declared default, as it reads; null when there is none.</summary>
        private string? DefaultLabel(string name)
        {
            var control = Controls.FirstOrDefault(c => c.Port == name);
            var declared = Spec.Inputs.FirstOrDefault(p => p.Name == name)?.Default is { } d ? Convert.ToString(d, CultureInfo.InvariantCulture) : null;
            string? value = control switch
            {
                ToggleControl t when Spec.Inputs.Any(p => p.Name == name) => t.Default ? "on" : "off",
                ToggleControl => Choice(name),
                SwitchControl s => DefaultOf(s) ?? declared ?? s.Options.FirstOrDefault(),
                DropdownControl or SliderControl => DefaultOf(control) ?? declared,
                _ => null,
            };
            return value == null || control == null ? null : ControlLayout.Label(control, value);
        }

        private readonly Dictionary<string, (string[] values, string[] labels)> _indexOptions = new();

        /// <summary>The values and labels a dropdown offers: those of the wired scene or graph for an index, else its own.</summary>
        /// <summary>The options of a step slider it offers now (Polar Preview's: the octaves its result simulated); null for all of them.</summary>
        public virtual IReadOnlyCollection<string>? OfferedOptionsOf(string port) => null;

        /// <summary>The directions an incidence control offers (Polar Preview's: those of its result); none by default.</summary>
        public virtual IReadOnlyList<Incidence> IncidencesOf(string port) => Array.Empty<Incidence>();

        public virtual (string[] values, string[] labels) OptionsOf(DropdownControl dropdown)
        {
            if (_indexOptions.TryGetValue(dropdown.Port, out var given))
                return given;
            return (dropdown.Options, dropdown.Options.Select(o => ControlLayout.Label(dropdown, o)).ToArray());
        }

        /// <summary>
        /// Offer the indices of <paramref name="names"/> on the index input <paramref name="port"/>: its dropdown reads "All"
        /// and each index with its name, and the port, when shown, its named values.
        /// </summary>
        public void SetIndexOptions(string port, IEnumerable<string> names)
        {
            var named = SceneIndices.Of(names);
            _indexOptions[port] = (new[] { "" }.Concat(named.Select(n => n.value.ToString(CultureInfo.InvariantCulture))).ToArray(),
                new[] { "All" }.Concat(named.Select(n => n.name)).ToArray());
            if (ControlPort(port) is global::Grasshopper.Kernel.Parameters.Param_Integer integer)
            {
                integer.ClearNamedValues();
                foreach (var (label, value) in named)
                    integer.AddNamedValue(label, value);
            }
        }

        /// <summary>
        /// Offer the integers ``values`` with their ``labels`` on the integer input ``port`` (a result's octaves, its preview
        /// sources): its dropdown reads the labels, no "All", and the port, when shown, its named values.
        /// </summary>
        public void SetValueOptions(string port, IReadOnlyList<int> values, IReadOnlyList<string> labels)
        {
            _indexOptions[port] = (values.Select(v => v.ToString(System.Globalization.CultureInfo.InvariantCulture)).ToArray(), labels.ToArray());
            if (ControlPort(port) is global::Grasshopper.Kernel.Parameters.Param_Integer integer)
            {
                integer.ClearNamedValues();
                for (var i = 0; i < values.Count; i++)
                    integer.AddNamedValue(labels[i], values[i]);
            }
        }

        /// <summary>The range a slider runs over now: its own, unless the component sets it from its data.</summary>
        public virtual (double min, double max) SliderRange(SliderControl slider) => (slider.Min, slider.Max);

        /// <summary>The value a slider shows when the component moves it; null for its input's.</summary>
        public virtual double? SliderShown(SliderControl slider) => null;

        /// <summary>The chosen option of <paramref name="name"/>, or its default: the control's <c>Default</c>, else its first option.</summary>
        public virtual string Choice(string name)
        {
            if (_choices.TryGetValue(name, out var value))
                return value;
            var preset = Controls.FirstOrDefault(c => c.Port == name) is { } control ? DefaultOf(control) : null;
            if (preset != null)
                return preset;
            return Choices.TryGetValue(name, out var options) && options.Length > 0 ? options[0] : "";
        }

        /// <summary>Choose <paramref name="value"/> for <paramref name="name"/> (an option of it), as an undoable change, and solve again.</summary>
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
