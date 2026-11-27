using System;
using System.IO;
using System.Linq;
using GH_IO.Serialization;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// The Anechoic sound dropdown of Auralise and Source on their <c>sound</c> input: the bundled recordings, the file Load...
    /// chose (a wired path instead, while one is wired) by its name, and Load..., which opens a file dialog. The loaded file is
    /// saved with the definition.
    /// </summary>
    public sealed class AnechoicSound
    {
        /// <summary>The entry of the dropdown that opens a file dialog.</summary>
        public const string LoadOption = "Load...";

        private const string LoadedKey = "anechoic_sound_loaded";

        /// <summary>
        /// The bundled recordings, by the names the worker's <c>auralise</c> and <c>make_source</c> take. None is chosen on a new
        /// component: its <c>Default</c> is only what the dropdown reads then, "...", and the input stays empty.
        /// </summary>
        public static readonly DropdownControl Dropdown = new("sound", new[] { "female_speech", "male_speech", "viola", "guitar", "castanets" },
            "...", new[] { "Female speech", "Male speech", "Viola", "Guitar", "Castanets" }, "Anechoic sound");

        private readonly ComponentBase _component;

        // the file Load... chose last, offered after the bundled recordings once another is chosen
        private string? _loaded;

        public AnechoicSound(ComponentBase component) => _component = component;

        /// <summary>The file the dropdown offers after the bundled recordings: the one chosen, else the one Load... chose last; null for none.</summary>
        public string? Loaded => _component.ControlPort(Dropdown.Port) is { } port && ControlState.Persistent(port) is string own && IsFile(own) ? own : _loaded;

        private static bool IsFile(string? value) => value is { Length: > 0 } && value != LoadOption && value != Dropdown.Default && !Dropdown.Options.Contains(value);

        /// <summary>The open-file dialog of Load..., given the folder it starts in; the file chosen, null when cancelled.</summary>
        internal Func<string?, string?> Open { get; set; } = folder =>
            FileButtons.Open("Anechoic sound to auralise", folder, "Audio", ".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3");

        /// <summary>The dropdown's entries: the bundled recordings, the loaded file (a wired path instead, while one is wired) by its name, and Load...</summary>
        public (string[] values, string[] labels) Options()
        {
            var port = _component.ControlPort(Dropdown.Port);
            var wired = port is { SourceCount: > 0 } ? ControlState.Of(port, Dropdown.Options).value : null;
            var file = IsFile(wired) ? wired : port is { SourceCount: > 0 } ? null : Loaded;
            var files = file == null ? Array.Empty<string>() : new[] { file };
            return (Dropdown.Options.Concat(files).Append(LoadOption).ToArray(),
                Dropdown.Labels!.Concat(files.Select(f => Path.GetFileName(f))).Append(LoadOption).ToArray());
        }

        /// <summary>Load...: choose a file and set it on the input. False for any other option, which the dropdown sets itself.</summary>
        public bool OnOption(string option)
        {
            if (option != LoadOption)
                return false;
            if (_component.ControlPort(Dropdown.Port) is not { } input || Open(Loaded is string last ? Path.GetDirectoryName(last) : null) is not string file)
                return true;
            // the undo step Choose records holds the file loaded before
            ControlState.Choose(_component, input, file);
            _loaded = file;
            return true;
        }

        public void Write(GH_IWriter writer)
        {
            if (Loaded is string loaded)
                writer.SetString(LoadedKey, loaded);
        }

        public void Read(GH_IReader reader) => _loaded = reader.ItemExists(LoadedKey) ? reader.GetString(LoadedKey) : null;
    }
}
