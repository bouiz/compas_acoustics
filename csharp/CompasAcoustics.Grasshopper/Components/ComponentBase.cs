using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// The base of every COMPAS Acoustics component: its names, description and ports come from its spec (the Python
    /// component's metadata.json), its icon from the spec's icon.png; worker errors become one Error message.
    /// </summary>
    public abstract partial class ComponentBase : GH_Component
    {
        public const string CategoryName = "COMPAS Acoustics";

        protected ComponentBase(string key) : this(ComponentSpec.Load(key))
        {
        }

        private ComponentBase(ComponentSpec spec) : base(spec.Name, spec.Nickname, spec.Description, CategoryName, spec.Subcategory)
        {
            Spec = spec;
        }

        // set by the constructor; read by RegisterInputParams, which GH_Component's constructor calls before it
        protected ComponentSpec Spec { get; private set; } = null!;

        private ComponentSpec CurrentSpec => Spec ?? ComponentSpec.Load(SpecKey);

        /// <summary>The spec's folder name, e.g. ``CA_Model``.</summary>
        protected abstract string SpecKey { get; }

        public override GH_Exposure Exposure => CurrentSpec.Exposure;

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

        /// <summary>The parameter of a ``ghdoc`` input or of an output; the default follows the spec's type hint.</summary>
        protected virtual IGH_Param? CreateParam(PortSpec port, bool input) => null;

        /// <summary>The controls drawn on the body, below the ports; each sets its input unless a wire does.</summary>
        public virtual IReadOnlyList<Control> Controls => Array.Empty<Control>();

        private readonly HashSet<string> _pressed = new();

        /// <summary>Whether the button of ``port`` was pressed for this solution (or a wire into ``port`` is true).</summary>
        public bool Pressed(string port) => _pressed.Contains(port);

        internal void MarkPressed(string port) => _pressed.Add(port);

        public void ClearPresses() => _pressed.Clear();

        /// <summary>Whether a button on the body acts at once (<see cref="OnButton"/>) instead of solving the component once.</summary>
        public virtual bool HandlesButton(string port) => false;

        /// <summary>A button the component handles itself was pressed (Auralise's Save opens a dialog).</summary>
        public virtual void OnButton(string port) { }

        /// <summary>A slider without an input port was set; the component keeps the value itself.</summary>
        public virtual void SliderChanged(string name, double value) { }

        /// <summary>The whole text of a message cut to <see cref="MessageText.Max"/> characters; the body's tooltip shows it.</summary>
        public string? FullMessage { get; private set; }

        /// <summary>Set the message strip: at most 40 characters, the full text in the tooltip.</summary>
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

        // Grasshopper previews every previewable parameter, inputs included: a geometry, point or plane input would draw
        // the geometry wired into it again, on top of the Rhino objects it comes from. Only the outputs preview.
        /// <summary>Whether ``output`` previews; the Visualise components draw themselves instead, and data drawn by the
        /// component that makes it (a source, a receiver) is not drawn again where it passes through.</summary>
        public virtual bool PreviewsOutput(IGH_Param output) => true;

        private IEnumerable<IGH_PreviewObject> PreviewedOutputs => Params.Output.Where(PreviewsOutput).OfType<IGH_PreviewObject>();

        /// <summary>The label a button shows now, when it changes (Play reads Stop while playing); null for its own.</summary>
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
            foreach (var port in CurrentSpec.Inputs)
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
            foreach (var port in CurrentSpec.Outputs)
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
            catch (WorkerException e)
            {
                Error(ErrorText(e));
            }
            catch (System.Net.Http.HttpRequestException e)
            {
                // the worker stopped answering twice in a row (the call restarts it once)
                Error($"The worker does not answer: {e.Message}. Recompute to start it again.");
            }
        }

        protected abstract void Solve(IGH_DataAccess access);

        /// <summary>The text a worker error shows: its message, the last line of a traceback.</summary>
        public static string ErrorText(WorkerException error) => WorkerException.Summary(error.Message);

        protected void Warn(string text) => AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, text);

        protected void Error(string text) => AddRuntimeMessage(GH_RuntimeMessageLevel.Error, text);

        protected void Remark(string text) => AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, text);

        /// <summary>The warnings a worker result lists, each as a Warning.</summary>
        protected void ShowWarnings(IDictionary<string, object?> result)
        {
            if (result.TryGetValue("warnings", out var warnings) && warnings is IEnumerable<object?> texts)
                foreach (var text in texts.OfType<string>().Distinct())
                    Warn(text);
        }

        protected static Worker Worker => Worker.Instance;

        /// <summary>An optional item input: its value, or null when nothing is connected or the input is empty.</summary>
        protected static T? Item<T>(IGH_DataAccess access, string name) where T : class
        {
            T? value = null;
            return access.GetData(name, ref value) ? value : null;
        }

        /// <summary>
        /// An optional value input: its value, or null when nothing is connected and the port shows its untouched default
        /// (the method's own default, so leaving it out changes nothing and keeps the messages that name given inputs).
        /// </summary>
        protected T? Value<T>(IGH_DataAccess access, string name) where T : struct
        {
            // a hidden input has no wire and no data: a button's press is read with Pressed
            if (_hidden.ContainsKey(name))
                return null;
            var value = default(T);
            if (!access.GetData(name, ref value))
                return null;
            var port = Params.Input.Find(p => p.Name == name);
            var declared = CurrentSpec.Inputs.FirstOrDefault(p => p.Name == name)?.Default;
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

        protected static List<T> List<T>(IGH_DataAccess access, string name)
        {
            var values = new List<T>();
            access.GetDataList(name, values);
            return values;
        }
    }
}
