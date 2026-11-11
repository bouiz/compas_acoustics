using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// Creates the worker: a uv-managed Python with the compas_acoustics of this plugin's version from PyPI, recorded in
    /// ~/.compas_acoustics/worker.json. Rhino's Python only fetches uv; compas_acoustics is never installed into it.
    /// </summary>
    public sealed class WorkerInstaller
    {
        private const string PythonVersion = "3.12";
        // the first import of the scientific stack is slow: done here, not on the first Grasshopper call
        private const string Warmup = "import compas_acoustics.simulation_methods as m; m.Binaural";

        /// <summary>Runs a program, logs what it prints and returns its exit code; the tests replace it.</summary>
        public delegate int Runner(string program, IReadOnlyList<string> args, Action<string> log);

        private readonly string _home;
        private readonly string _version;
        private readonly Action _stopWorkers;
        private readonly Runner _run;

        public WorkerInstaller(string home, string version, Action stopWorkers, Runner? run = null)
        {
            _home = home;
            _version = version;
            _stopWorkers = stopWorkers;
            _run = run ?? RunProcess;
        }

        public static string RhinoPython(string home) =>
            Path.Combine(home, ".rhinocode", "py39-rh8", OperatingSystem.IsWindows() ? "python.exe" : "python3.9");

        /// <summary>Replace the worker's environment; returns its interpreter. The record comes last: a failed install leaves no worker.</summary>
        public string Install(Action<string> log)
        {
            var rhinoPython = RhinoPython(_home);
            if (!File.Exists(rhinoPython))
                throw new WorkerException($"Rhino's Python is not set up ({rhinoPython} is missing): open Rhino's ScriptEditor once, then install again.");
            var root = Path.Combine(_home, ".compas_acoustics");
            var environment = Path.Combine(root, "worker");
            var python = Path.Combine(environment, OperatingSystem.IsWindows() ? "Scripts/python.exe" : "bin/python");

            log("Fetching uv with Rhino's Python...");
            var uvFolder = Path.Combine(root, "uv");
            Delete(uvFolder);
            Run(rhinoPython, log, "-I", "-m", "pip", "--isolated", "install", "--disable-pip-version-check", "--no-warn-script-location", "--target", uvFolder, "uv");
            var uv = Uv(uvFolder);

            log("Stopping running workers...");
            // a running worker holds the files of the environment open, which Windows does not let go of
            _stopWorkers();
            Delete(environment);

            log($"Creating a Python {PythonVersion} environment...");
            Run(uv, log, "venv", "--python", PythonVersion, environment);
            log($"Installing compas_acoustics {_version}; the first time this can download several GB...");
            Run(uv, log, "pip", "install", "--python", python, $"compas_acoustics[hybrid]=={_version}");
            log("Importing compas_acoustics, the first time is slow...");
            Run(python, log, "-c", Warmup);

            File.WriteAllText(Path.Combine(root, "worker.json"), JsonSerializer.Serialize(new { python, hybrid = true }));
            log($"Installed compas_acoustics {_version} at {python}");
            return python;
        }

        private static void Delete(string folder)
        {
            if (Directory.Exists(folder))
                Directory.Delete(folder, true);
        }

        private static string Uv(string folder)
        {
            foreach (var candidate in new[] { "bin/uv", "bin/uv.exe", "Scripts/uv.exe" })
                if (File.Exists(Path.Combine(folder, candidate)))
                    return Path.Combine(folder, candidate);
            throw new WorkerException($"uv was installed into {folder}, but its executable was not found there.");
        }

        private void Run(string program, Action<string> log, params string[] args)
        {
            var code = _run(program, args, log);
            if (code != 0)
                throw new WorkerException($"{Path.GetFileName(program)} {args[0]} failed with exit code {code}.");
        }

        private static int RunProcess(string program, IReadOnlyList<string> args, Action<string> log)
        {
            var info = new ProcessStartInfo(program) { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true };
            foreach (var arg in args)
                info.ArgumentList.Add(arg);
            // Rhino's Python paths would load its 3.9 packages into the worker
            info.Environment.Remove("PYTHONHOME");
            info.Environment.Remove("PYTHONPATH");
            using var process = new Process { StartInfo = info };
            DataReceivedEventHandler print = (_, e) =>
            {
                if (!string.IsNullOrWhiteSpace(e.Data))
                    log("  " + e.Data);
            };
            process.OutputDataReceived += print;
            process.ErrorDataReceived += print;
            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            process.WaitForExit();
            return process.ExitCode;
        }
    }
}
