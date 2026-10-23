using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using CompasAcoustics.Pb;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The impulse responses and sounds of Auralise as the worker's ``auralise`` takes them, one group per branch.</summary>
    public static class AuraliseGroups
    {
        /// <summary>The folder and the name of the signals a chosen file gives: ``take.wav`` saves ``take_mic_0_0.wav`` and so on.</summary>
        public static (string folder, string name) SaveTarget(string path)
        {
            var file = Path.GetFileName(path);
            var name = Path.GetExtension(file).Equals(".wav", StringComparison.OrdinalIgnoreCase) ? Path.GetFileNameWithoutExtension(file) : file;
            return (Path.GetDirectoryName(path) ?? "", name);
        }

        private static string? SoundPath(IGH_Goo? goo) => goo switch
        {
            null => null,
            GH_String s => string.IsNullOrEmpty(s.Value) ? null : s.Value,
            SourceGoo source => source.SoundFile,
            _ => goo.ToString(),
        };

        /// <param name="paths">The branch paths by their text, to put each signal back in its branch.</param>
        public static List<Dictionary<string, object?>> FromTrees(GH_Structure<ImpulseResponseGoo> irs, GH_Structure<IGH_Goo>? sounds,
            out Dictionary<string, GH_Path> paths)
        {
            var hasSounds = sounds != null && sounds.DataCount > 0;
            if (hasSounds)
            {
                if (irs.PathCount != sounds!.PathCount)
                    throw new ArgumentException("DataTree branch count mismatch.");
                for (var i = 0; i < irs.PathCount; i++)
                    if (irs.Branches[i].Count != sounds.Branches[i].Count)
                        throw new ArgumentException("DataTree branch length mismatch.");
            }
            paths = new Dictionary<string, GH_Path>();
            var groups = new List<Dictionary<string, object?>>();
            for (var i = 0; i < irs.PathCount; i++)
            {
                var path = irs.Paths[i];
                var branch = irs.Branches[i].Where(ir => ir?.Value != null).ToList();
                if (branch.Count == 0)
                    continue;
                paths[path.ToString()] = path;
                groups.Add(new Dictionary<string, object?>
                {
                    ["branch"] = path.ToString(),
                    ["irs"] = branch.Select(ir => (object?)ir.Value).ToList(),
                    ["sounds"] = Enumerable.Range(0, branch.Count).Select(j => (object?)(hasSounds ? SoundPath(sounds!.Branches[i][j]) : null)).ToList(),
                });
            }
            return groups;
        }

        public static string Message(IEnumerable<string> formats, string level, double peakDbfs) =>
            string.Format(CultureInfo.InvariantCulture, "{0}, {1}, peak {2:0.0} dBFS", string.Join(" + ", formats.Distinct().OrderBy(f => f, StringComparer.Ordinal)),
                level, peakDbfs);

        public static string ClipWarning(double peakDbfs) =>
            string.Format(CultureInfo.InvariantCulture, "Output peaks at {0}{1:0.0} dBFS and will clip on playback. Use level 'relative' or 'individual'.",
                peakDbfs >= 0 ? "+" : "", peakDbfs);
    }

    public sealed class AuraliseComponent : ComponentBase
    {
        private static readonly string[] Levels = { "none", "relative", "individual" };

        public AuraliseComponent() : base("CA_Auralise") { }
        protected override string SpecKey => "CA_Auralise";
        public override Guid ComponentGuid => new Guid("4d9abc0d-a247-5525-8835-b5907a4ae2a8");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => (input, port.Name) switch
        {
            (true, "irs") => new Param_ImpulseResponse(),
            (false, "signals") => new Param_Audio(),
            _ => null,
        };

        public override IReadOnlyList<Control> Controls { get; } = new Control[]
        {
            new SwitchControl("level", Levels, Default: "relative"), new ButtonControl("save", "Save"),
        };

        // the last solution's request, which Save renders again into the chosen files
        private List<object?>? _groups;
        private Dictionary<string, object?>? _settings;
        private string? _saved;

        public override bool HandlesButton(string port) => port == "save";

        public override void OnButton(string port)
        {
            if (port != "save")
                return;
            if (_groups == null || _settings == null)
            {
                Rhino.UI.Dialogs.ShowMessage("Connect impulse responses to Auralise first.", "COMPAS Acoustics");
                return;
            }
            var dialog = new Eto.Forms.SaveFileDialog { Title = "Save signals", FileName = "convolved_sound.wav" };
            dialog.Filters.Add(new Eto.Forms.FileFilter("WAV audio", ".wav"));
            if (dialog.ShowDialog(Rhino.UI.RhinoEtoApp.MainWindow) != Eto.Forms.DialogResult.Ok || string.IsNullOrEmpty(dialog.FileName))
                return;
            var (folder, name) = AuraliseGroups.SaveTarget(dialog.FileName);
            var kwargs = new Dictionary<string, object?>(_settings) { ["save_folder"] = folder, ["save_name"] = name };
            var groups = _groups;
            Message = "saving...";
            global::Grasshopper.Instances.RedrawCanvas();
            System.Threading.Tasks.Task.Run(() =>
            {
                string note;
                try
                {
                    var result = (IDictionary<string, object?>)Worker.Call("auralise", new object?[] { groups }, kwargs)!;
                    var count = ((IEnumerable<object?>)result["signals"]!).Count();
                    note = $"Saved {count} signal(s) to {folder} as {name}_mic_*.wav.";
                }
                catch (Exception e) when (e is WorkerException or System.Net.Http.HttpRequestException)
                {
                    note = "Saving failed: " + (e is WorkerException w ? ErrorText(w) : e.Message);
                }
                _saved = note;
                Rhino.RhinoApp.InvokeOnUiThread(new Action(() => ExpireSolution(true)));
            });
        }

        protected override void Solve(IGH_DataAccess access)
        {
            access.GetDataTree("irs", out GH_Structure<ImpulseResponseGoo> irs);
            if (irs == null || irs.DataCount == 0)
            {
                Warn("Input parameter 'irs' failed to collect data.");
                return;
            }
            var level = Choice("level");
            access.GetDataTree("anechoic_sounds", out GH_Structure<IGH_Goo> sounds);
            List<Dictionary<string, object?>> groups;
            Dictionary<string, GH_Path> paths;
            try
            {
                groups = AuraliseGroups.FromTrees(irs, sounds, out paths);
            }
            catch (ArgumentException e)
            {
                Error(e.Message);
                return;
            }
            var kwargs = new Dictionary<string, object?>
            {
                ["level"] = level,
                ["mix_sources"] = Value<bool>(access, "mix_sources") ?? false,
            };
            (_groups, _settings) = (groups.Cast<object?>().ToList(), kwargs);
            // what the last Save did, shown once
            if (_saved != null)
            {
                Remark(_saved);
                _saved = null;
            }
            RunInBackground(access, new Background("auralise", new object?[] { groups.Cast<object?>().ToList() }, kwargs, (paths, level)), (outcome, _) =>
            {
                var result = (IDictionary<string, object?>)outcome.Result!;
                var (branches, rendered) = ((Dictionary<string, GH_Path>, string))outcome.Context!;
                ShowWarnings(result);
                var signals = new DataTree<AudioGoo>();
                var rates = new DataTree<int>();
                var formats = new List<string>();
                foreach (var signal in ((IEnumerable<object?>)result["signals"]!).Cast<IDictionary<string, object?>>())
                {
                    var audio = new AudioGoo((AudioData)signal["audio"]!);
                    var path = branches[(string)signal["branch"]!];
                    signals.Add(audio, path);
                    rates.Add(audio.SamplingRate, path);
                    formats.Add(audio.Format);
                }
                var peak = Convert.ToDouble(result["peak_dbfs"], CultureInfo.InvariantCulture);
                Status(AuraliseGroups.Message(formats, rendered, peak));
                if (peak > 0)
                    Warn(AuraliseGroups.ClipWarning(peak));
                if (formats.Contains("ambisonic"))
                    Remark("Ambisonic signals (ACN/SN3D) cannot be played by the Audio Player. Save them as a multichannel file for an ambisonic decoder, " +
                           "or simulate with Simulate Binaural for headphone playback.");
                access.SetDataTree(0, signals);
                access.SetDataTree(1, rates);
            });
        }

        protected override bool RunsInBackground => true;
    }

    /// <summary>
    /// The Audio Player: it shows the signal it plays (a lane per channel), and plays, pauses, stops and seeks it in the worker.
    /// The transport keeps its own clock while playing, repaints only its body, and polls the worker once a second.
    /// </summary>
    public sealed class AudioPlayerComponent : ComponentBase
    {
        private static readonly System.Diagnostics.Stopwatch Clock = System.Diagnostics.Stopwatch.StartNew();
        private System.Threading.Timer? _timer;
        private int _ticks;
        private readonly Dictionary<string, AudioData> _files = new();

        public AudioPlayerComponent() : base("CA_Audio_Player") { }
        protected override string SpecKey => "CA_Audio_Player";
        public override Guid ComponentGuid => new Guid("ebf0dfa8-2bf3-5fd2-b59e-5ac8fb327480");

        public Transport Transport { get; } = new(() => Clock.Elapsed.TotalSeconds);

        /// <summary>The signal shown and played; a file's samples as read by the worker.</summary>
        public AudioData? Signal { get; private set; }

        /// <summary>The file played when the input is a path or a source, which plays whole: no pause or seek.</summary>
        public string? File { get; private set; }

        public double GainDb { get; private set; }

        /// <summary>Whether a signal is scaled to full level before playing: the Norm toggle, off by default, saved with the component.</summary>
        public bool Normalize { get; private set; }

        /// <summary>Why the signal cannot play (more than two channels), or the last playback error; shown in the header.</summary>
        public string? Problem { get; private set; }

        public float BodyWidth { get; set; } = 240;

        /// <summary>The height of one channel's lane; the component grows down with the channels.</summary>
        public float LaneHeight { get; set; } = 50;

        public int Channels => Signal == null ? 1 : Signal.Shape.Count > 1 ? Math.Max((int)Signal.Shape[1], 1) : 1;

        /// <summary>The channels below the inputs, once a signal is there; 0 without one.</summary>
        public float WaveHeight => Signal == null ? 0 : PlayerLayout.WaveHeight(Channels, LaneHeight);

        public bool CanPlay => Signal != null && Problem == null;

        public bool CanSeek => CanPlay && File == null;

        public override void CreateAttributes() => m_attributes = new AudioPlayerUi(this);

        protected override void Solve(IGH_DataAccess access)
        {
            var audio = Item<IGH_Goo>(access, "audio");
            AudioData? signal = null;
            string? file = null;
            switch (audio)
            {
                case null:
                    break;
                case AudioGoo goo when goo.Value != null:
                    signal = goo.Value;
                    break;
                case ImpulseResponseGoo ir when ir.Value != null:
                    signal = SignalOf(ir);
                    break;
                case SourceGoo source when !string.IsNullOrEmpty(source.SoundFile):
                    file = source.SoundFile;
                    break;
                case GH_String path when !string.IsNullOrEmpty(path.Value):
                    file = path.Value;
                    break;
                default:
                    Error("Input parameter 'audio' is not valid.");
                    break;
            }
            if (file != null)
            {
                if (!_files.TryGetValue(file, out var samples))
                    _files[file] = samples = (AudioData)Worker.Call("sound_samples", file)!;
                signal = samples;
            }
            Load(signal, file);
            if (Problem != null)
                Warn(Problem);
        }

        private (ImpulseResponseData of, AudioData audio)? _response;

        /// <summary>An impulse response as a signal to listen to; the same response gives the same signal, so a recompute does not stop it.</summary>
        internal AudioData SignalOf(ImpulseResponseGoo ir)
        {
            if (_response is not { } known || !ReferenceEquals(known.of, ir.Value))
                _response = known = (ir.Value, Audio(ir));
            return known.audio;
        }

        /// <summary>The response's samples at its raw level, in its format (an ambisonic response cannot play here).</summary>
        internal static AudioData Audio(ImpulseResponseGoo ir)
        {
            var audio = new AudioData { SamplingRate = ir.SamplingRate, Format = ir.Format };
            audio.Samples.Add(ir.Value.Samples);
            audio.Shape.Add(ir.Value.Shape);
            return audio;
        }

        /// <summary>Show and play ``signal`` (a file's samples when ``file`` is given); another signal stops what plays.</summary>
        internal void Load(AudioData? signal, string? file)
        {
            if (!ReferenceEquals(signal, Signal) || file != File)
            {
                // another signal: what plays stops
                if (Transport.State != TransportState.Stopped)
                    Send(Transport.Stop());
                var before = WaveHeight;
                Signal = signal;
                // more or fewer channels, or a signal at all: the component grows or shrinks
                if (WaveHeight != before)
                    Attributes?.ExpireLayout();
                File = file;
                Transport.Duration = signal == null || signal.SamplingRate == 0 || signal.Shape.Count == 0 ? 0 : signal.Shape[0] / (double)signal.SamplingRate;
            }
            var channels = signal == null ? 0 : signal.Shape.Count > 1 ? (int)signal.Shape[1] : 1;
            Problem = channels > 2
                ? $"A {channels}-channel ambisonic signal cannot be played here. Save it with Auralise as a multichannel file for an ambisonic decoder, " +
                  "or simulate with Simulate Binaural for headphone playback."
                : null;
        }

        public void OnPlay()
        {
            if (CanPlay)
                Send(Transport.Play());
        }

        /// <summary>The Play/Pause button: pause what plays, else play (or resume).</summary>
        public void OnPlayPause()
        {
            if (Transport.State == TransportState.Playing)
                OnPause();
            else
                OnPlay();
        }

        public void OnPause()
        {
            if (!CanPlay)
                return;
            // a file plays whole: pausing it stops it
            Send(File == null ? Transport.Pause() : Transport.Stop());
        }

        public void OnStop() => Send(Transport.Stop());

        public void OnSeek(double seconds)
        {
            if (CanSeek)
                Send(Transport.Seek(seconds));
        }

        public void OnGain(double gainDb)
        {
            GainDb = Math.Clamp(gainDb, -40, 6);
            // the worker plays at a fixed level: a running signal restarts from where it is at the new one
            if (Transport.State == TransportState.Playing && File == null)
                Send(Transport.Seek(Transport.Position));
        }

        /// <summary>A double-click on the volume slider: back to 0 dB.</summary>
        public void OnResetVolume() => OnGain(0.0);

        public void OnNormalize(bool normalize)
        {
            Normalize = normalize;
            // as for the volume: a running signal restarts from where it is at the new level
            if (Transport.State == TransportState.Playing && File == null)
                Send(Transport.Seek(Transport.Position));
        }

        public override void SliderChanged(string name, double value) => OnGain(value);

        private void Send(TransportCommand? command)
        {
            if (command == null)
                return;
            try
            {
                switch (command)
                {
                    case PlayCommand play when File != null:
                        Worker.Call("play", File);
                        break;
                    case PlayCommand play:
                        Worker.Call("play", new object?[] { Signal }, new Dictionary<string, object?>
                        {
                            ["normalize"] = Normalize, ["gain_db"] = GainDb, ["start"] = play.Start,
                        });
                        break;
                    case PauseCommand:
                        Worker.Call("pause");
                        break;
                    case ResumeCommand:
                        Worker.Call("resume");
                        break;
                    case StopCommand:
                        Worker.Call("stop");
                        break;
                }
                if (Problem != null && Problem.StartsWith("The sound could not be played"))
                    Problem = null;
            }
            catch (WorkerException e)
            {
                Transport.Stop();
                Problem = $"The sound could not be played: {ErrorText(e)}";
            }
            UpdateTimer();
            Repaint();
        }

        private void UpdateTimer()
        {
            if (Transport.State == TransportState.Playing)
            {
                _ticks = 0;
                _timer ??= new System.Threading.Timer(_ => Tick(), null, 40, 40);
            }
            else
            {
                _timer?.Dispose();
                _timer = null;
            }
        }

        private void Tick()
        {
            // once a second the worker says where it is, and whether the sound ended
            if (++_ticks % 25 == 0)
            {
                try
                {
                    var status = (IDictionary<string, object?>)Worker.Call("playback_status")!;
                    Transport.Sync(status["playing"] is true, Convert.ToDouble(status["position"], System.Globalization.CultureInfo.InvariantCulture));
                }
                catch (Exception)
                {
                    // the next tick tries again
                }
                if (Transport.State != TransportState.Playing)
                {
                    Rhino.RhinoApp.InvokeOnUiThread((Action)UpdateTimer);
                }
            }
            Repaint();
        }

        private static void Repaint()
        {
            try
            {
                Rhino.RhinoApp.InvokeOnUiThread((Action)(() => global::Grasshopper.Instances.ActiveCanvas?.Invalidate()));
            }
            catch (Exception)
            {
                // outside Rhino (tests) there is no canvas
            }
        }

        public override void RemovedFromDocument(GH_Document document)
        {
            if (Transport.State != TransportState.Stopped)
                OnStop();
            _timer?.Dispose();
            _timer = null;
            base.RemovedFromDocument(document);
        }

        public override bool Write(GH_IO.Serialization.GH_IWriter writer)
        {
            writer.SetDouble("gain_db", GainDb);
            writer.SetBoolean("normalize", Normalize);
            writer.SetDouble("width", BodyWidth);
            writer.SetDouble("lane_height", LaneHeight);
            return base.Write(writer);
        }

        public override bool Read(GH_IO.Serialization.GH_IReader reader)
        {
            if (reader.ItemExists("gain_db"))
                GainDb = reader.GetDouble("gain_db");
            if (reader.ItemExists("normalize"))
                Normalize = reader.GetBoolean("normalize");
            if (reader.ItemExists("width"))
                BodyWidth = (float)reader.GetDouble("width");
            if (reader.ItemExists("lane_height"))
                LaneHeight = (float)reader.GetDouble("lane_height");
            return base.Read(reader);
        }
    }

    public sealed class RegularizedFitnessComponent : ComponentBase
    {
        public RegularizedFitnessComponent() : base("CA_Regularized_Fitness") { }
        protected override string SpecKey => "CA_Regularized_Fitness";
        public override Guid ComponentGuid => new Guid("d703105e-a921-5762-a9fe-2fde25c091d2");

        protected override void Solve(IGH_DataAccess access)
        {
            var values = List<double>(access, "values");
            if (values.Count == 0)
            {
                Warn("Input parameter 'values' failed to collect data.");
                return;
            }
            var target = Value<double>(access, "target");
            if (target == null)
            {
                Warn("Input parameter 'target' failed to collect data.");
                return;
            }
            var fitness = Worker.Call("regularized_fitness", values.Cast<object?>().ToList(), target.Value, Value<double>(access, "k") ?? 1.0);
            access.SetData(0, Convert.ToDouble(fitness, CultureInfo.InvariantCulture));
        }
    }
}
