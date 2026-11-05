using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>Creates the worker in the background with its Install button, and shows the log.</summary>
    public sealed class InstallWorkerComponent : ComponentBase
    {
        private readonly object _lock = new();
        private readonly List<string> _log = new();
        private bool _running;
        private string? _error;

        public InstallWorkerComponent() : base("CA_Install_Worker") { }
        protected override string SpecKey => "CA_Install_Worker";
        public override Guid ComponentGuid => new Guid("8c2e4f6a-1b3d-4e5f-a7c9-6d0f2b4e8a13");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => port.Name switch
        {
            "ok" => new Param_Boolean(),
            _ => new Param_String(),
        };

        public override IReadOnlyList<Control> Controls { get; } = new Control[] { new ButtonControl("install", "Install") };

        protected override void Solve(IGH_DataAccess access)
        {
            string text;
            bool running;
            string? error;
            lock (_lock)
            {
                if (Pressed("install") && !_running)
                    Start();
                (text, running, error) = (string.Join("\n", _log), _running, _error);
            }
            var installed = !running && error == null && Worker.Python != null;
            Status(running ? "installing" : error != null ? "failed" : installed ? "installed" : "not installed", text.Split('\n').Last());
            if (error != null)
                Error(error);
            access.SetData("log", text);
            access.SetData("ok", installed);
        }

        private void Start()
        {
            _running = true;
            _error = null;
            _log.Clear();
            var worker = Worker;
            var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            Task.Run(() =>
            {
                string? error = null;
                try
                {
                    new WorkerInstaller(home, Worker.Version, worker.StopAll).Install(Log);
                }
                catch (Exception e)
                {
                    // not only WorkerException: a locked file or a missing program must end the run too
                    error = e.Message;
                    Log(e.Message);
                }
                lock (_lock)
                    (_running, _error) = (false, error);
                Refresh();
            });
        }

        private void Log(string line)
        {
            lock (_lock)
                _log.Add(line);
            Refresh();
        }

        private void Refresh() =>
            Rhino.RhinoApp.InvokeOnUiThread(new Action(() => OnPingDocument()?.ScheduleSolution(100, _ => ExpireSolution(false))));
    }
}
