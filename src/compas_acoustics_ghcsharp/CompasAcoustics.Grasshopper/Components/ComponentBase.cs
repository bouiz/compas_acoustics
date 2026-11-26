using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// The base of every COMPAS Acoustics component: its names, description and ports come from its spec (the
    /// component's metadata.json), its icon from the spec's icon.png; worker errors become one Error message.
    /// </summary>
    public abstract partial class ComponentBase : GH_Component
    {
        public const string CategoryName = "COMPAS Acoustics";

        /// <summary>The ribbon's panels, left to right: the subcategories the specs name.</summary>
        public static readonly IReadOnlyList<string> Subcategories = new[]
        {
            "Elements", "Params", "Model", "Simulation", "Analysis", "Auralisation", "Display", "Util", "Optimization", "Lab",
        };

        /// <summary>
        /// The panel name Grasshopper gets for a subcategory. Grasshopper sorts a tab's panels by name, so each one gets a leading
        /// space more than the panel after it: a space sorts before any letter, with ordinal, invariant and culture comparison alike.
        /// </summary>
        public static string RibbonSubcategory(string subcategory)
        {
            var index = Subcategories.ToList().IndexOf(subcategory);
            if (index < 0)
                throw new ArgumentException($"No ribbon panel {subcategory}.", nameof(subcategory));
            return new string(' ', Subcategories.Count - index) + subcategory;
        }

        protected ComponentBase(string key) : this(ComponentSpec.Load(key))
        {
        }

        private ComponentBase(ComponentSpec spec) : base(spec.Name, spec.Nickname, spec.Description, CategoryName, RibbonSubcategory(spec.Subcategory))
        {
        }

        private ComponentSpec Spec => ComponentSpec.Load(SpecKey);

        /// <summary>The spec's folder name, e.g. <c>CA_Model</c>.</summary>
        protected abstract string SpecKey { get; }

        public override GH_Exposure Exposure => Spec.Exposure;

        protected override Bitmap Icon
        {
            get
            {
                try
                {
                    return new Bitmap(new MemoryStream(ComponentSpec.IconBytes(SpecKey)));
                }
                catch (Exception)
                {
                    return null!;
                }
            }
        }

        /// <summary>The parameter of a <c>ghdoc</c> input or of an output; the default follows the spec's type hint.</summary>
        protected virtual IGH_Param? CreateParam(PortSpec port, bool input) => null;

        /// <summary>The controls drawn on the body, below the ports; each sets its input unless a wire does.</summary>
        public virtual IReadOnlyList<Control> Controls => Array.Empty<Control>();

        private readonly HashSet<string> _pressed = new();

        /// <summary>Whether the button of <paramref name="port"/> was pressed for this solution (or a wire into it is true).</summary>
        public bool Pressed(string port) => _pressed.Contains(port);

        internal void MarkPressed(string port) => _pressed.Add(port);

        public void ClearPresses() => _pressed.Clear();

        /// <summary>Whether a button on the body acts at once (<see cref="OnButton"/>) instead of solving the component once.</summary>
        public virtual bool HandlesButton(string port) => false;

        /// <summary>A button the component handles itself was pressed.</summary>
        public virtual void OnButton(string port) { }

        /// <summary>A slider without an input port was set; the component keeps the value itself.</summary>
        public virtual void SliderChanged(string name, double value) { }

        /// <summary>The whole text of a message cut to <see cref="MessageText.Max"/> characters; the body's tooltip shows it.</summary>
        public string? FullMessage { get; private set; }

        /// <summary>Set the message strip: at most <see cref="MessageText.Max"/> characters a line, the full text in the tooltip.</summary>
        protected void Status(string text, string? full = null) => (Message, FullMessage) = MessageText.Status(text, full);

        protected override void BeforeSolveInstance()
        {
            FullMessage = null;
            base.BeforeSolveInstance();
        }

        protected sealed override void AppendAdditionalComponentMenuItems(System.Windows.Forms.ToolStripDropDown menu)
        {
            AppendMenu(menu);
            var log = Worker.Instance.LogPath;
            if (ControlPorts.Count > 0)
                Menu_AppendItem(menu, "Show control inputs", (sender, args) => ShowControlPorts(!ControlPortsShown), true, ControlPortsShown);
            Menu_AppendItem(menu, "Open worker log", (sender, args) => OpenWorkerLog(), log != null && File.Exists(log));
        }

        /// <summary>The component's own right-click entries, above "Open worker log".</summary>
        protected virtual void AppendMenu(System.Windows.Forms.ToolStripDropDown menu) { }

        /// <summary>Open the worker's log with the system's viewer.</summary>
        public static void OpenWorkerLog()
        {
            var log = Worker.Instance.LogPath;
            if (log == null || !File.Exists(log))
                return;
            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(log) { UseShellExecute = true });
        }

        /// <summary>Whether a new component starts with its preview off (the surfaces it previews are in Rhino already).</summary>
        public virtual bool PreviewOffByDefault => false;

        /// <summary>Whether <paramref name="output"/> is drawn in the viewport. Inputs never are: they would repeat the geometry wired into them.</summary>
        public virtual bool PreviewsOutput(IGH_Param output) => true;

        private IEnumerable<IGH_PreviewObject> PreviewedOutputs => Params.Output.Where(PreviewsOutput).OfType<IGH_PreviewObject>();

        /// <summary>The label a button shows now, when it differs from its own; null for its own.</summary>
        public virtual string? ButtonLabel(string port) => null;

        // Grasshopper draws a component that cannot preview in the dark "preview off" grey; every component here is drawn
        // light, and one that has nothing to draw draws nothing (its outputs are typed, none previewed).
        public override bool IsPreviewCapable => true;

        public override Rhino.Geometry.BoundingBox ClippingBox
        {
            get
            {
                var box = Rhino.Geometry.BoundingBox.Empty;
                foreach (var output in PreviewedOutputs)
                    box.Union(output.ClippingBox);
                return box;
            }
        }

        public override void DrawViewportWires(IGH_PreviewArgs args)
        {
            foreach (var output in PreviewedOutputs)
                output.DrawViewportWires(args);
        }

        public override void DrawViewportMeshes(IGH_PreviewArgs args)
        {
            foreach (var output in PreviewedOutputs)
                output.DrawViewportMeshes(args);
        }

        public override void CreateAttributes()
        {
            if (Controls.Count > 0 || RunsInBackground)
                m_attributes = new ComponentUi(this);
            else
                base.CreateAttributes();
        }

        protected override void AfterSolveInstance()
        {
            base.AfterSolveInstance();
            // a press triggers one solution
            ClearPresses();
        }

        private static IGH_Param DefaultParam(string typeHint) => typeHint switch
        {
            "str" => new Param_String(),
            "int" => new Param_Integer(),
            "float" => new Param_Number(),
            "bool" => new Param_Boolean(),
            "point" => new Param_Point(),
            "vector" => new Param_Vector(),
            "plane" => new Param_Plane(),
            "geometrybase" => new Param_Geometry(),
            _ => new Param_GenericObject(),
        };

        protected override void RegisterInputParams(GH_InputParamManager manager)
        {
            foreach (var port in Spec.Inputs)
            {
                var param = CreateParam(port, true) ?? DefaultParam(port.TypeHint);
                var index = manager.AddParameter(param, port.Name, port.Name, port.Description, port.Access);
                manager[index].Optional = port.Optional;
                PortDefaults.Apply(manager[index], port);
                if (port.Flatten)
                    manager[index].DataMapping = GH_DataMapping.Flatten;
            }
        }

        protected override void RegisterOutputParams(GH_OutputParamManager manager)
        {
            foreach (var port in Spec.Outputs)
                manager.AddParameter(CreateParam(port, false) ?? DefaultParam(port.TypeHint), port.Name, port.Name, port.Description, GH_ParamAccess.item);
        }

        protected sealed override void SolveInstance(IGH_DataAccess access)
        {
            try
            {
                if (Pressed(CancelPort))
                    _runs?.CancelAll();
                Solve(access);
            }
            // an HttpRequestException: the worker stopped answering twice in a row (the call restarts it once)
            catch (Exception e) when (e is WorkerException or System.Net.Http.HttpRequestException)
            {
                Error(ErrorText(e));
            }
        }

        protected abstract void Solve(IGH_DataAccess access);

        /// <summary>The text an error shows: a worker error's message (the last line of a traceback), or that the worker does not answer.</summary>
        public static string ErrorText(Exception error) => error switch
        {
            WorkerException w => WorkerException.Summary(w.Message),
            System.Net.Http.HttpRequestException h => $"The worker does not answer: {h.Message}. Recompute to start it again.",
            _ => error.Message,
        };

        protected void Warn(string text) => AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, text);

        protected void Error(string text) => AddRuntimeMessage(GH_RuntimeMessageLevel.Error, text);

        protected void Remark(string text) => AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, text);

        /// <summary>Warn that the input <paramref name="name"/> has no data.</summary>
        protected void WarnMissing(string name) => Warn($"Input parameter '{name}' failed to collect data.");

        /// <summary>The warnings a worker result lists, each as a Warning.</summary>
        protected void ShowWarnings(IDictionary<string, object?> result)
        {
            if (result.TryGetValue("warnings", out var warnings) && warnings is IEnumerable<object?> texts)
                foreach (var text in texts.OfType<string>().Distinct())
                    Warn(text);
        }

        /// <summary>The notes a worker result lists, each as a Remark (a grey note, not a warning).</summary>
        protected void ShowNotes(IDictionary<string, object?> result)
        {
            foreach (var text in LabResults.Notes(result))
                Remark(text);
        }

        protected static Worker Worker => Worker.Instance;

        /// <summary>An optional item input: its value, or null when nothing is connected or the input is empty.</summary>
        protected static T? Item<T>(IGH_DataAccess access, string name) where T : class
        {
            T? value = null;
            return access.GetData(name, ref value) ? value : null;
        }

        /// <summary>A required item input: its valid value, or null after a warning.</summary>
        protected T? Required<T>(IGH_DataAccess access, string name) where T : class, IGH_Goo
        {
            if (Item<T>(access, name) is { IsValid: true } item)
                return item;
            WarnMissing(name);
            return null;
        }

        /// <summary>An optional value input: null when nothing is connected and the port still holds its declared default, so that the worker applies its own.</summary>
        protected T? Value<T>(IGH_DataAccess access, string name) where T : struct
        {
            // a hidden input has no wire: the body's value (a button's press is read with Pressed)
            if (_hidden.ContainsKey(name))
                return HiddenAs<T>(name);
            var value = default(T);
            if (!access.GetData(name, ref value))
                return null;
            var port = Params.Input.Find(p => p.Name == name);
            var declared = Spec.Inputs.FirstOrDefault(p => p.Name == name)?.Default;
            return port != null && PortDefaults.Untouched(port.SourceCount, declared, value) ? null : value;
        }

        /// <summary>An optional text input: null when nothing is connected or the text is empty.</summary>
        protected string? Text(IGH_DataAccess access, string name)
        {
            if (_hidden.ContainsKey(name))
                return HiddenValue(name);
            if (IsChoice(name))
                return Choice(name);
            string? value = null;
            return access.GetData(name, ref value) && !string.IsNullOrEmpty(value) ? value : null;
        }

        /// <summary>A list input: its values, empty when nothing is connected.</summary>
        protected List<T> Items<T>(IGH_DataAccess access, string name)
        {
            if (_hidden.ContainsKey(name))
                return HiddenList<T>(name);
            var values = new List<T>();
            access.GetDataList(name, values);
            return values;
        }

        /// <summary>The responses numbered as pairs; null after a warning when there are none, or an error when two are of one pair.</summary>
        public List<Pair>? Numbered(IEnumerable<ImpulseResponseGoo?> irs)
        {
            List<Pair> pairs;
            try
            {
                pairs = PairChoice.Number(irs.Where(ir => ir != null).Select(ir => ir!));
            }
            catch (ArgumentException e)
            {
                Error(e.Message);
                return null;
            }
            if (pairs.Count == 0)
            {
                WarnMissing("irs");
                return null;
            }
            return pairs;
        }

        /// <summary>
        /// The responses on <c>irs</c>, flattened and numbered as pairs, and the pairs <c>source_idx</c> and <c>receiver_idx</c>
        /// choose, once their dropdowns offer the pairs' names; null when none is chosen, after the message that says why.
        /// </summary>
        protected (List<Pair> All, List<Pair> Chosen)? ChosenPairs(IGH_DataAccess access)
        {
            access.GetDataTree("irs", out GH_Structure<ImpulseResponseGoo> irs);
            if (Numbered(irs?.FlattenData() ?? new List<ImpulseResponseGoo>()) is not { } pairs)
                return null;
            var (sources, receivers) = PairChoice.Names(pairs);
            SetIndexOptions("source_idx", sources);
            SetIndexOptions("receiver_idx", receivers);
            var chosen = PairChoice.Choose(pairs, Items<int>(access, "source_idx"), Items<int>(access, "receiver_idx"), Warn);
            return chosen.Count == 0 ? null : (pairs, chosen);
        }
    }
}
