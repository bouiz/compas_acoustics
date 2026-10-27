using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using Grasshopper.GUI.Canvas;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The worker's state, Python, version and schema, with Restart and Open log.</summary>
    public sealed class WorkerComponent : ComponentBase
    {
        public WorkerComponent() : base("CA_Worker") { }
        protected override string SpecKey => "CA_Worker";
        public override Guid ComponentGuid => new Guid("5f0b1c2e-7a43-5d8e-9c61-2b7e4a9d3f10");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => port.Name switch
        {
            "running" => new Param_Boolean(),
            _ => new Param_String(),
        };

        public override IReadOnlyList<Control> Controls { get; } = new Control[]
        {
            new ButtonControl("restart", "Restart"),
            new ButtonControl("open_log", "Open log"),
        };

        public override void CreateAttributes() => m_attributes = new WorkerUi(this);

        public override void AddedToDocument(GH_Document document)
        {
            base.AddedToDocument(document);
            Worker.StateChanged += OnStateChanged;
        }

        public override void RemovedFromDocument(GH_Document document)
        {
            Worker.StateChanged -= OnStateChanged;
            base.RemovedFromDocument(document);
        }

        private void OnStateChanged() =>
            Rhino.RhinoApp.InvokeOnUiThread(new Action(() => OnPingDocument()?.ScheduleSolution(1, _ => ExpireSolution(false))));

        /// <summary>The lines the body shows: state, Python, version and schema.</summary>
        public static string[] Lines(Worker worker)
        {
            var state = MessageText.WorkerLine(worker.State, worker.StartedAt is DateTime t ? (DateTime.Now - t).TotalSeconds : null);
            var version = worker.State == WorkerState.Running
                ? $"compas_acoustics {Worker.Version}, schema ok"
                : worker.LastError is string error && error.Contains("schema")
                    ? "schema mismatch: run the installer again"
                    : $"plugin {Worker.Version}";
            return new[] { state, worker.Python ?? "no Python: run the installer", version };
        }

        protected override void Solve(IGH_DataAccess access)
        {
            var worker = Worker;
            if (Pressed("restart") && worker.State != WorkerState.Starting)
                Task.Run(() =>
                {
                    try
                    {
                        worker.Restart();
                    }
                    catch (Exception)
                    {
                        // LastError says why; the state change solves this component again
                    }
                });
            if (Pressed("open_log"))
                OpenWorkerLog();
            Message = MessageText.WorkerLine(worker.State, worker.StartedAt is DateTime t ? (DateTime.Now - t).TotalSeconds : null);
            if (worker.LastError is string error && worker.State != WorkerState.Running)
                Warn(WorkerException.Summary(error));
            access.SetData("python", worker.Python);
            access.SetData("version", worker.State == WorkerState.Running ? Worker.Version : null);
            access.SetData("running", worker.State == WorkerState.Running);
        }
    }

    /// <summary>The Worker component's body: its buttons, then three lines of state.</summary>
    public sealed class WorkerUi : ComponentUi
    {
        private const float LineHeight = 10;

        public WorkerUi(WorkerComponent owner) : base(owner) { }

        protected override float ExtraHeight => 3 * LineHeight + 4;

        protected override void Layout()
        {
            base.Layout();
            var b = Bounds;
            Bounds = new RectangleF(b.X, b.Y, Math.Max(b.Width, 150), b.Height);
        }

        protected override void RenderExtra(GH_Canvas canvas, Graphics g)
        {
            var area = ExtraArea;
            using var font = new Font(FontFamily.GenericSansSerif, 6f);
            using var format = new StringFormat { Trimming = StringTrimming.EllipsisPath, FormatFlags = StringFormatFlags.NoWrap };
            var lines = WorkerComponent.Lines(Worker.Instance);
            for (var i = 0; i < lines.Length; i++)
                g.DrawString(lines[i], font, Brushes.Black, new RectangleF(area.Left + 3, area.Top + 2 + i * LineHeight, Bounds.Width - 6, LineHeight), format);
        }
    }
}
