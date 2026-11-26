using System;
using System.IO;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>A button that chooses a file for a text input: the path is the input's value.</summary>
    internal static class FileButtons
    {
        /// <summary>The file chosen on <paramref name="port"/>, null when none is (or a wire sets the input).</summary>
        public static string? Chosen(ComponentBase component, string port) =>
            component.ControlPort(port) is { SourceCount: 0 } input && ControlState.Persistent(input) is string path && path.Length > 0 ? path : null;

        /// <summary>The button's label: <paramref name="empty"/> without a file, else <paramref name="prefix"/> and the file's name.</summary>
        public static string Label(ComponentBase component, string port, string prefix, string empty) =>
            Chosen(component, port) is string path ? prefix + Path.GetFileName(path) : empty;

        /// <summary>Open a file dialog and make the file chosen <paramref name="port"/>'s value; a wired input is left as it is.</summary>
        public static void Choose(ComponentBase component, string port, string title, string filterName, params string[] extensions) =>
            Choose(component, port, title, () => null, filterName, extensions);

        /// <param name="startFolder">The folder the dialog opens in when no file is chosen; asked only then, and null leaves it to the system.</param>
        public static void Choose(ComponentBase component, string port, string title, Func<string?> startFolder, string filterName, params string[] extensions)
        {
            if (component.ControlPort(port) is not { } input || input.SourceCount > 0)
                return;
            var dialog = new Eto.Forms.OpenFileDialog { Title = title };
            dialog.Filters.Add(new Eto.Forms.FileFilter(filterName, extensions));
            var folder = Chosen(component, port) is string current ? Path.GetDirectoryName(current) : startFolder();
            if (Directory.Exists(folder))
                dialog.Directory = new Uri(folder);
            if (dialog.ShowDialog(Rhino.UI.RhinoEtoApp.MainWindow) == Eto.Forms.DialogResult.Ok && !string.IsNullOrEmpty(dialog.FileName))
                ControlState.Choose(component, input, dialog.FileName);
        }

        /// <summary>The right-click entry that clears the file of <paramref name="port"/>, when one is chosen.</summary>
        public static void AppendClear(ComponentBase component, System.Windows.Forms.ToolStripDropDown menu, string port, string text)
        {
            if (Chosen(component, port) != null)
                GH_DocumentObject.Menu_AppendItem(menu, text, (sender, args) => ControlState.Choose(component, component.ControlPort(port)!, ""));
        }
    }
}
