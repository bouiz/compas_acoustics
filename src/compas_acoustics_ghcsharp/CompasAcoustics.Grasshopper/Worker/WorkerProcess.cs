using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>A started worker process.</summary>
    public interface IWorkerProcess
    {
        bool HasExited { get; }

        /// <summary>The end of what the process printed, for an error message.</summary>
        string Output { get; }

        /// <summary>The file with everything the process printed; null when there is none.</summary>
        string? LogPath { get; }

        void Kill();

        bool WaitForExit(TimeSpan timeout);
    }

    /// <summary>Starts worker processes; the tests replace it.</summary>
    public interface IProcessStarter
    {
        /// <param name="env">Variables to set; a null value removes the variable.</param>
        IWorkerProcess Start(string python, IReadOnlyList<string> args, IDictionary<string, string?> env);
    }

    internal sealed class ProcessStarter : IProcessStarter
    {
        public IWorkerProcess Start(string python, IReadOnlyList<string> args, IDictionary<string, string?> env)
        {
            var info = new ProcessStartInfo(python)
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            foreach (var arg in args)
                info.ArgumentList.Add(arg);
            foreach (var (name, value) in env)
            {
                if (value == null)
                    info.Environment.Remove(name);
                else
                    info.Environment[name] = value;
            }
            return new WorkerProcess(Process.Start(info) ?? throw new WorkerException($"The worker at {python} could not be started."));
        }
    }

    internal sealed class WorkerProcess : IWorkerProcess
    {
        private const int Keep = 4000;
        private readonly Process _process;
        private readonly StringBuilder _output = new();

        private readonly StreamWriter? _log;

        public WorkerProcess(Process process)
        {
            _process = process;
            try
            {
                LogPath = Path.Combine(Path.GetTempPath(), $"compas_acoustics_worker_{process.Id}.log");
                _log = new StreamWriter(LogPath, false) { AutoFlush = true };
            }
            catch (Exception e) when (e is IOException || e is UnauthorizedAccessException)
            {
                LogPath = null;
            }
            _process.OutputDataReceived += (s, e) => Append(e.Data);
            _process.ErrorDataReceived += (s, e) => Append(e.Data);
            _process.BeginOutputReadLine();
            _process.BeginErrorReadLine();
        }

        private void Append(string? line)
        {
            if (line == null)
                return;
            lock (_output)
            {
                _output.AppendLine(line);
                _log?.WriteLine(line);
                if (_output.Length > 2 * Keep)
                    _output.Remove(0, _output.Length - Keep);
            }
        }

        public string? LogPath { get; }

        public bool HasExited => _process.HasExited;

        public string Output
        {
            get
            {
                lock (_output)
                    return _output.Length > Keep ? _output.ToString(_output.Length - Keep, Keep) : _output.ToString();
            }
        }

        public void Kill()
        {
            try
            {
                if (!_process.HasExited)
                    _process.Kill(true);
            }
            catch (InvalidOperationException)
            {
            }
        }

        public bool WaitForExit(TimeSpan timeout) => _process.WaitForExit((int)timeout.TotalMilliseconds);
    }
}
