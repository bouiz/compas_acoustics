using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using CompasAcoustics.Pb;
using GH_IO.Serialization;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The chosen pairs of Auralise as the worker's <c>auralise</c> and <c>save_impulse_responses</c> take them, and its signals back in their branches.</summary>
    public static class AuraliseGroups
    {
        /// <summary>Where an ambisonic signal goes instead of the Audio Player.</summary>
        internal const string AmbisonicAdvice = "as a multichannel file for an ambisonic decoder, or simulate with Simulate Binaural for headphone playback.";

        /// <summary>The folder and the name of the signals a chosen file gives: <c>take.wav</c> saves <c>take_mic_0_0.wav</c> and so on.</summary>
        public static (string folder, string name) SaveTarget(string path)
        {
            var file = Path.GetFileName(path);
            var name = Path.GetExtension(file).Equals(".wav", StringComparison.OrdinalIgnoreCase) ? Path.GetFileNameWithoutExtension(file) : file;
            return (Path.GetDirectoryName(path) ?? "", name);
        }

        /// <summary>
        /// Every connected response, the index of each chosen pair among them and its label: the relative level takes its gain
        /// from all of them, and renders only the chosen.
        /// </summary>
        public static (List<object?> Irs, List<long> Chosen, List<string> Labels) Request(IReadOnlyList<Pair> all, IReadOnlyList<Pair> chosen)
        {
            var index = all.Select((p, i) => (p, i)).ToDictionary(t => (t.p.Receiver, t.p.Source), t => (long)t.i);
            return (all.Select(p => (object?)p.Response.Value).ToList(), chosen.Select(p => index[(p.Receiver, p.Source)]).ToList(), chosen.Select(PairChoice.LabelOf).ToList());
        }

        /// <summary>The worker's signals in the branches of their pairs, {receiver;source}, found by label; their sampling rates alike, and their formats.</summary>
        public static (DataTree<AudioGoo> Signals, DataTree<int> Rates, List<string> Formats) Trees(IEnumerable<IDictionary<string, object?>> signals, IReadOnlyList<Pair> chosen)
        {
            var paths = chosen.ToDictionary(PairChoice.LabelOf, PairChoice.PathOf);
            var audio = new DataTree<AudioGoo>();
            var rates = new DataTree<int>();
            var formats = new List<string>();
            foreach (var signal in signals)
            {
                var goo = new AudioGoo((AudioData)signal["audio"]!);
                var path = paths[(string)signal["label"]!];
                audio.Add(goo, path);
                rates.Add(goo.SamplingRate, path);
                formats.Add(goo.Format);
            }
            return (audio, rates, formats);
        }

        /// <summary>The folder of the anechoic sounds the worker's compas_acoustics bundles, where the Sound... dialog opens; null when the worker cannot tell.</summary>
        public static string? LibraryFolder()
        {
            try
            {
                return Worker.Instance.Call("anechoic_library_folder") as string;
            }
            catch (Exception)
            {
                // whatever keeps the worker from answering, the dialog opens where the system puts it
                return null;
            }
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
        private static readonly string[] Levels = { "relative", "individual", "none" };

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
            new DropdownControl("level", Levels, "relative"), SceneIndices.SourceDropdown, SceneIndices.ReceiverDropdown,
            new ButtonControl("sound", "Sound..."), new ButtonControl("save_rir", "Save RIR"), new ButtonControl("save_auralisation", "Save auralisation"),
        };

        /// <summary>
        /// A definition saved before Auralise chose pairs, with anechoic_sounds and mix_sources: Grasshopper reads the saved
        /// inputs by position, so the removed ones hold their places while the new ones are set aside, then the removed ones go
        /// with their wires and the new ones come back empty, hidden when the definition hid its control inputs.
        /// </summary>
        public override bool Read(GH_IReader reader)
        {
            if (!SavedInput(reader, "anechoic_sounds"))
                return base.Read(reader);
            var added = Params.Input.Skip(1).ToList();
            foreach (var port in added)
                Params.UnregisterInputParameter(port, false);
            var removed = new List<IGH_Param> { new Param_GenericObject { Name = "anechoic_sounds", NickName = "anechoic_sounds", Optional = true, Access = GH_ParamAccess.tree } };
            if (SavedInput(reader, "mix_sources"))
                removed.Add(new Param_Boolean { Name = "mix_sources", NickName = "mix_sources", Optional = true });
            foreach (var port in removed)
                Params.RegisterInputParam(port);
            var read = base.Read(reader);
            foreach (var port in removed)
                Params.UnregisterInputParameter(port, true);
            foreach (var port in added)
                Params.RegisterInputParam(port);
            Params.OnParametersChanged();
            HideControlPortsAsRead();
            return read;
        }

        private sealed record LastRequest(List<object?> Irs, List<object?> Chosen, List<object?> Labels, string? Sound, string Level);

        // the last solution's request, which the Save buttons write into the chosen files
        private LastRequest? _last;
        private string? _saved;

        public override bool HandlesButton(string port) => port is "sound" or "save_rir" or "save_auralisation";

        public override string? ButtonLabel(string port) => port == "sound" ? FileButtons.Label(this, port, "Sound: ", "Sound...") : null;

        protected override void AppendMenu(System.Windows.Forms.ToolStripDropDown menu) => FileButtons.AppendClear(this, menu, "sound", "Clear sound");

        public override void OnButton(string port)
        {
            if (port == "sound")
            {
                FileButtons.Choose(this, port, "Anechoic sound to auralise", AuraliseGroups.LibraryFolder, "Audio", ".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3");
                return;
            }
            if (!HandlesButton(port))
                return;
            var last = _last;
            if (last == null)
            {
                Rhino.UI.Dialogs.ShowMessage("Connect impulse responses to Auralise first.", CategoryName);
                return;
            }
            var rir = port == "save_rir";
            if (!rir && last.Sound == null)
            {
                Rhino.UI.Dialogs.ShowMessage("Choose a sound with the Sound... button, or connect a File Path to 'sound', first.", CategoryName);
                return;
            }
            var dialog = new Eto.Forms.SaveFileDialog
            {
                Title = rir ? "Save impulse responses" : "Save signals",
                FileName = rir ? "impulse_response.wav" : "convolved_sound.wav",
            };
            dialog.Filters.Add(new Eto.Forms.FileFilter("WAV audio", ".wav"));
            if (dialog.ShowDialog(Rhino.UI.RhinoEtoApp.MainWindow) != Eto.Forms.DialogResult.Ok || string.IsNullOrEmpty(dialog.FileName))
                return;
            var (folder, name) = AuraliseGroups.SaveTarget(dialog.FileName);
            var (function, args, kwargs) = rir
                ? ("save_impulse_responses", new object?[] { last.Irs, last.Chosen, last.Labels, folder },
                    new Dictionary<string, object?> { ["level"] = last.Level, ["save_name"] = name })
                : ("auralise", new object?[] { last.Irs, last.Sound, last.Chosen, last.Labels },
                    new Dictionary<string, object?> { ["level"] = last.Level, ["save_folder"] = folder, ["save_name"] = name });
            Status("saving...");
            global::Grasshopper.Instances.RedrawCanvas();
            System.Threading.Tasks.Task.Run(() =>
            {
                string note;
                try
                {
                    var result = (IDictionary<string, object?>)Worker.Call(function, args, kwargs)!;
                    var count = ((IEnumerable<object?>)result[rir ? "files" : "signals"]!).Count();
                    note = rir
                        ? $"Saved {count} impulse response(s) to {folder} as {name}_mic_<receiver>_<source>.wav, each with its JSON metadata."
                        : $"Saved {count} signal(s) to {folder} as {name}_mic_<receiver>_<source>.wav.";
                }
                catch (Exception e) when (e is WorkerException or System.Net.Http.HttpRequestException)
                {
                    note = "Saving failed: " + (e is WorkerException ? ErrorText(e) : e.Message);
                }
                _saved = note;
                Rhino.RhinoApp.InvokeOnUiThread(new Action(() => ExpireSolution(true)));
            });
        }

        // here, not in Solve: a solution without irs solves no instance, and the Save buttons must not write the request before it
        protected override void BeforeSolveInstance()
        {
            _last = null;
            base.BeforeSolveInstance();
        }

        protected override void Solve(IGH_DataAccess access)
        {
            if (ChosenPairs(access) is not { } pairs)
                return;
            var level = Choice("level");
            var sound = Text(access, "sound");
            var (irs, indices, names) = AuraliseGroups.Request(pairs.All, pairs.Chosen);
            var (chosen, labels) = (indices.Cast<object?>().ToList(), names.Cast<object?>().ToList());
            _last = new LastRequest(irs, chosen, labels, sound, level);
            // what the last Save did, shown once
            if (_saved != null)
            {
                Remark(_saved);
                _saved = null;
            }
            if (sound == null)
            {
                Warn("No sound to auralise: choose an anechoic sound file with the Sound... button, or connect a File Path to 'sound'.");
                return;
            }
            var args = new object?[] { irs, sound, chosen, labels };
            RunInBackground(access, new Background("auralise", args, new Dictionary<string, object?> { ["level"] = level }, (pairs.Chosen, level)), (outcome, _) =>
            {
                var result = outcome.Result;
                var (paired, rendered) = ((List<Pair>, string))outcome.Context!;
                var (signals, rates, formats) = AuraliseGroups.Trees(((IEnumerable<object?>)result["signals"]!).Cast<IDictionary<string, object?>>(), paired);
                var peak = Convert.ToDouble(result["peak_dbfs"], CultureInfo.InvariantCulture);
                Status(AuraliseGroups.Message(formats, rendered, peak));
                if (peak > 0)
                    Warn(AuraliseGroups.ClipWarning(peak));
                if (formats.Contains("ambisonic"))
                    Remark("Ambisonic signals (ACN/SN3D) cannot be played by the Audio Player. Save them " + AuraliseGroups.AmbisonicAdvice);
                access.SetDataTree(0, signals);
                access.SetDataTree(1, rates);
            });
        }

        protected override bool RunsInBackground => true;
    }

    /// <summary>
    /// The Audio Player: it shows the signal it plays (a lane per channel), and plays, pauses, stops and seeks it in the worker.
    /// The transport keeps its own clock while playing, repaints the canvas without a new solution, and polls the worker once
    /// a second.
    /// </summary>
    public sealed class AudioPlayerComponent : ComponentBase
    {
        private static readonly System.Diagnostics.Stopwatch Clock = System.Diagnostics.Stopwatch.StartNew();
        private System.Threading.Timer? _timer;
        private int _ticks;
        private readonly Dictionary<string, AudioData> _files = new();
        private string? _channelProblem;
        private string? _playbackError;

        public AudioPlayerComponent() : base("CA_Audio_Player") { }
        protected override string SpecKey => "CA_Audio_Player";
        public override Guid ComponentGuid => new Guid("ebf0dfa8-2bf3-5fd2-b59e-5ac8fb327480");

        public Transport Transport { get; } = new(() => Clock.Elapsed.TotalSeconds);

        /// <summary>The signal shown and played; a file's samples as read by the worker.</summary>
        public AudioData? Signal { get; private set; }

        /// <summary>The file played when the input is a path, which plays whole: no pause or seek.</summary>
        public string? File { get; private set; }

        public double GainDb { get; private set; }

        /// <summary>Whether a signal is scaled to full level before playing: the Normalize toggle, off by default, saved with the component.</summary>
        public bool Normalize { get; private set; }

        /// <summary>The last playback error, or why the signal cannot play (more than two channels); shown in the header.</summary>
        public string? Problem => _playbackError ?? _channelProblem;

        public float BodyWidth { get; set; } = 240;

        /// <summary>The height of one channel's lane; the component grows down with the channels.</summary>
        public float LaneHeight { get; set; } = 50;

        public int Channels => Signal == null ? 1 : Signal.Shape.Count > 1 ? Math.Max((int)Signal.Shape[1], 1) : 1;

        /// <summary>The channels below the inputs, once a signal is there; 0 without one.</summary>
        public float WaveHeight => Signal == null ? 0 : PlayerLayout.WaveHeight(Channels, LaneHeight);

        public bool CanPlay => Signal != null && Problem == null;

        public bool CanSeek => CanPlay && File == null;

        public override void CreateAttributes() => m_attributes = new AudioPlayerUi(this);

        // a list on the item input solves the player once per item, and the last one stays
        private int _solves;

        protected override void BeforeSolveInstance()
        {
            _solves = 0;
            base.BeforeSolveInstance();
        }

        protected override void AfterSolveInstance()
        {
            if (_solves > 1)
                Remark($"{_solves} items connected to 'audio'; playing the last: pick one with List Item.");
            base.AfterSolveInstance();
        }

        protected override void Solve(IGH_DataAccess access)
        {
            _solves++;
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
                case GH_String path when !string.IsNullOrEmpty(path.Value):
                    file = path.Value;
                    break;
                case SourceGoo:
                    Error("The Audio Player no longer plays a Source's sound file: wire a File Path, or Auralise's signals.");
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

        /// <summary>Show and play <paramref name="signal"/> (a file's samples when <paramref name="file"/> is given); another signal stops what plays.</summary>
        internal void Load(AudioData? signal, string? file)
        {
            if (!ReferenceEquals(signal, Signal) || file != File)
            {
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
            _playbackError = null;
            _channelProblem = channels > 2 ? $"A {channels}-channel ambisonic signal cannot be played here. Save it with Auralise " + AuraliseGroups.AmbisonicAdvice : null;
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
                    case PlayCommand when File != null:
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
                _playbackError = null;
            }
            catch (WorkerException e)
            {
                Transport.Stop();
                _playbackError = $"The sound could not be played: {ErrorText(e)}";
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
                    Transport.Sync(status["playing"] is true, Convert.ToDouble(status["position"], CultureInfo.InvariantCulture));
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
            var values = Items<double>(access, "values");
            if (values.Count == 0)
            {
                WarnMissing("values");
                return;
            }
            var target = Value<double>(access, "target");
            if (target == null)
            {
                WarnMissing("target");
                return;
            }
            var fitness = Worker.Call("regularized_fitness", values.Cast<object?>().ToList(), target.Value, Value<double>(access, "k") ?? 1.0);
            access.SetData(0, Convert.ToDouble(fitness, CultureInfo.InvariantCulture));
        }
    }
}
