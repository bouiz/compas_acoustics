using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class WorkerInstallerTests : IDisposable
    {
        private readonly string _home = Directory.CreateTempSubdirectory("compas_acoustics_install_").FullName;
        private readonly List<string> _calls = new();
        private readonly List<string> _rhinoPythonCalls = new();
        private readonly List<string> _stops = new();
        private string? _failOn;

        public WorkerInstallerTests()
        {
            var python = WorkerInstaller.RhinoPython(_home);
            Directory.CreateDirectory(Path.GetDirectoryName(python)!);
            File.WriteAllText(python, "");
        }

        public void Dispose() => Directory.Delete(_home, true);

        private string Config => Path.Combine(_home, ".compas_acoustics", "worker.json");

        private string Environment => Path.Combine(_home, ".compas_acoustics", "worker");

        private int Run(string program, IReadOnlyList<string> args, Action<string> log)
        {
            var call = Path.GetFileName(program) + " " + string.Join(" ", args);
            _calls.Add(call);
            // by path: on Windows the worker's Python has the same file name as Rhino's
            if (program == WorkerInstaller.RhinoPython(_home))
                _rhinoPythonCalls.Add(call);
            if (args.Contains("uv"))
            {
                var bin = Path.Combine(args[args.Count - 2], "bin");
                Directory.CreateDirectory(bin);
                File.WriteAllText(Path.Combine(bin, "uv"), "");
            }
            return _failOn != null && call.Contains(_failOn) ? 3 : 0;
        }

        private WorkerInstaller Installer(string version = "1.2.3") => new(_home, version, () => _stops.Add("stopped"), Run);

        [Fact]
        public void Install_PinsThePluginsVersionAndRecordsThePython()
        {
            var python = Installer().Install(_ => { });

            Assert.Contains(_calls, c => c.EndsWith("--python " + python + " compas_acoustics==1.2.3"));
            using var json = JsonDocument.Parse(File.ReadAllText(Config));
            Assert.Equal(python, json.RootElement.GetProperty("python").GetString());
            Assert.True(json.RootElement.GetProperty("hybrid").GetBoolean());
        }

        [Fact]
        public void Install_NeverInstallsCompasAcousticsIntoRhinosPython()
        {
            Installer().Install(_ => { });

            Assert.EndsWith(" uv", Assert.Single(_rhinoPythonCalls));
        }

        [Fact]
        public void Install_StopsTheWorkersBeforeReplacingTheEnvironment()
        {
            Directory.CreateDirectory(Environment);
            var old = Path.Combine(Environment, "old");
            File.WriteAllText(old, "");
            var existedAtStop = false;
            var installer = new WorkerInstaller(_home, "1.2.3", () => existedAtStop = File.Exists(old), Run);

            installer.Install(_ => { });

            Assert.True(existedAtStop);
            Assert.False(File.Exists(old));
        }

        [Theory]
        [InlineData("compas_acoustics==")]
        [InlineData("import compas_acoustics")]
        public void Install_ThatFails_LeavesNoRecord(string step)
        {
            _failOn = step;

            var error = Assert.Throws<WorkerException>(() => Installer().Install(_ => { }));

            Assert.Contains("exit code 3", error.Message);
            Assert.False(File.Exists(Config));
        }

        [Fact]
        public void Install_WithoutRhinosPython_SaysWhatToDo()
        {
            File.Delete(WorkerInstaller.RhinoPython(_home));

            var error = Assert.Throws<WorkerException>(() => Installer().Install(_ => { }));

            Assert.Contains("ScriptEditor", error.Message);
            Assert.Empty(_calls);
            Assert.Empty(_stops);
        }
    }
}
