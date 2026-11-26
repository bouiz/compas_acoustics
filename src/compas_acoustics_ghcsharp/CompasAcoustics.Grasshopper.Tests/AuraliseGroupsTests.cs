using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Reflection;
using Grasshopper;
using Grasshopper.Kernel.Types;
using CompasAcoustics.Pb;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Parameters;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class AuraliseGroupsTests
    {
        internal static ImpulseResponseGoo Ir(string receiver, string source) => new(new ImpulseResponseData
        {
            Name = $"{receiver}_{source}",
            Shape = { 1 },
            Samples = { 1 },
            Source = new SourceData { Guid = source, Name = source },
            Receiver = new ReceiverData { Guid = receiver, Name = receiver },
        });

        private static List<Pair> FourPairs() => PairChoice.Number(new[] { Ir("r0", "a"), Ir("r0", "b"), Ir("r1", "a"), Ir("r1", "b") });

        [Fact]
        public void Request_IndexesTheChosenInEveryConnected()
        {
            var all = FourPairs();
            var chosen = all.Where(p => p.Source == 1).ToList();
            var (irs, indices, labels) = AuraliseGroups.Request(all, chosen);
            Assert.Equal(all.Select(p => (object?)p.Response.Value), irs);
            Assert.Equal(new long[] { 1, 3 }, indices);
            Assert.Equal(new[] { "0_1", "1_1" }, labels);
        }

        [Fact]
        public void Trees_PutEachSignalInTheBranchOfItsPair()
        {
            var chosen = FourPairs().Where(p => p.Source == 1).ToList();
            IDictionary<string, object?> Signal(string label, int rate) =>
                new Dictionary<string, object?> { ["label"] = label, ["audio"] = new AudioData { SamplingRate = rate, Format = "mono", Shape = { 1 }, Samples = { 0.5 } } };
            var (signals, formats) = AuraliseGroups.Trees(new[] { Signal("1_1", 44100), Signal("0_1", 48000) }, chosen);
            Assert.Equal(new[] { "{0;1}", "{1;1}" }, signals.Paths.Select(p => p.ToString()).OrderBy(p => p));
            Assert.Equal(48000, signals.Branch(new GH_Path(0, 1)).Single().SamplingRate);
            Assert.Equal(44100, signals.Branch(new GH_Path(1, 1)).Single().SamplingRate);
            Assert.Equal(new[] { "mono", "mono" }, formats);
        }

        [Fact]
        public void Message_NamesFormatsAndPeak()
        {
            Assert.Equal("binaural + mono, peak -1.0 dBFS", AuraliseGroups.Message(new[] { "mono", "binaural", "mono" }, -1.0));
            Assert.Equal("Output peaks at +2.5 dBFS and will clip on playback. Use level 'relative' or 'individual'.", AuraliseGroups.ClipWarning(2.5));
        }

        [Theory]
        [InlineData("/renders/take.wav", "/renders", "take")]
        [InlineData("/renders/take", "/renders", "take")]
        [InlineData("/renders/take.WAV", "/renders", "take")]
        public void ASavedFile_GivesTheFolderAndTheNameOfTheSignals(string path, string folder, string name) =>
            Assert.Equal((folder.Replace('/', Path.DirectorySeparatorChar), name), AuraliseGroups.SaveTarget(path));

        [Fact]
        public void Auralise_InputsAndBody()
        {
            var spec = ComponentSpec.Load("CA_Auralise");
            Assert.Equal(new[] { "irs", "source_idx", "receiver_idx", "sound" }, spec.Inputs.Select(p => p.Name));
            Assert.Equal(new[] { GH_ParamAccess.tree, GH_ParamAccess.list, GH_ParamAccess.list, GH_ParamAccess.item }, spec.Inputs.Select(p => p.Access));
            Assert.Equal(new[] { "ghdoc", "int", "int", "str" }, spec.Inputs.Select(p => p.TypeHint));
            Assert.All(spec.Inputs.Skip(1), p => Assert.True(p.Optional));
            var auralise = new AuraliseComponent();
            Assert.Equal(new Control[]
            {
                new DropdownControl("level", new[] { "none", "relative", "individual" }, "none"),
                SceneIndices.SourceDropdown, SceneIndices.ReceiverDropdown, AuraliseComponent.SoundDropdown,
                new ButtonControl("save_rir", "Save RIR"), new ButtonControl("save_auralisation", "Save auralisation"),
            }.Select(c => c.ToString()), auralise.Controls.Select(c => c.ToString()));
            Assert.DoesNotContain(auralise.Controls, c => c.Port == "mix_sources");
            Assert.Equal(new[] { "source_idx", "receiver_idx", "sound" }, auralise.ControlPorts);
            Assert.False(auralise.HandlesButton("sound"));
            Assert.True(auralise.HandlesButton("save_rir"));
            Assert.True(auralise.HandlesButton("save_auralisation"));
            Assert.False(auralise.HandlesButton("run"));
            Assert.Null(auralise.ButtonLabel("save_rir"));
        }

        [Fact]
        public void TheAnechoicSound_IsADropdownOfTheBundledRecordings_NoneChosenOnANewComponent()
        {
            var auralise = new AuraliseComponent();
            var dropdown = AuraliseComponent.SoundDropdown;
            Assert.Equal("Anechoic sound", ControlLayout.Title(dropdown));
            Assert.Equal(new[] { "female_speech", "male_speech", "viola", "guitar", "castanets", "Load..." }, auralise.OptionsOf(dropdown).values);
            Assert.Equal(new[] { "Female speech", "Male speech", "Viola", "Guitar", "Castanets", "Load..." }, auralise.OptionsOf(dropdown).labels);
            var port = auralise.ControlPort("sound")!;
            // the placeholder shows, and is no entry of the menu
            var shown = ControlState.Of(port, auralise.OptionsOf(dropdown).values, ComponentBase.DefaultOf(dropdown)).value;
            Assert.Equal("...", ControlLayout.Label(dropdown, shown));
            Assert.DoesNotContain("...", auralise.OptionsOf(dropdown).labels);
            Assert.Null(ControlState.Persistent(port));
            Assert.Null(auralise.Loaded);
            Assert.Equal(("sound", port.Description), auralise.ControlTooltip("sound"));
            Assert.Contains("nothing is chosen by default", port.Description);
            // a recording chosen: the input's value, read by its label
            ControlState.Choose(auralise, port, "viola");
            Assert.Equal("viola", ControlState.Persistent(port));
            Assert.Equal("Viola", ControlLayout.Label(dropdown, ControlState.Of(port, auralise.OptionsOf(dropdown).values, ComponentBase.DefaultOf(dropdown)).value));
        }

        [Fact]
        public void TheAnechoicSound_ReadsThePlaceholder_WhenAWireHoldsNothing()
        {
            var auralise = new AuraliseComponent();
            var dropdown = AuraliseComponent.SoundDropdown;
            Assert.Equal("...", ComponentUi.DropdownText(auralise, dropdown, ""));
            Assert.Equal("...", ComponentUi.DropdownText(auralise, dropdown, "..."));
            Assert.Equal("Viola", ComponentUi.DropdownText(auralise, dropdown, "viola"));
            var all = SceneIndices.SourceDropdown;
            Assert.Equal("All", ComponentUi.DropdownText(auralise, all, ""));
        }

        [Fact]
        public void Load_ChoosesTheFile_ThenOffersItBeforeLoad()
        {
            var auralise = new AuraliseComponent();
            var asked = new List<string?>();
            auralise.OpenSound = folder =>
            {
                asked.Add(folder);
                return Path.Combine("sounds", "bsn_tele_44.wav");
            };
            var dropdown = AuraliseComponent.SoundDropdown;
            Assert.True(auralise.OnDropdownOption("sound", "Load..."));
            var loaded = Path.Combine("sounds", "bsn_tele_44.wav");
            Assert.Equal(loaded, ControlState.Persistent(auralise.ControlPort("sound")!));
            Assert.Equal(loaded, auralise.Loaded);
            Assert.Equal(new[] { "female_speech", "male_speech", "viola", "guitar", "castanets", loaded, "Load..." }, auralise.OptionsOf(dropdown).values);
            Assert.Equal(new[] { "Female speech", "Male speech", "Viola", "Guitar", "Castanets", "bsn_tele_44.wav", "Load..." }, auralise.OptionsOf(dropdown).labels);
            // a bundled recording chosen after: the loaded file stays on offer, and the next dialog opens in its folder
            ControlState.Choose(auralise, auralise.ControlPort("sound")!, "viola");
            Assert.Equal(new[] { "female_speech", "male_speech", "viola", "guitar", "castanets", loaded, "Load..." }, auralise.OptionsOf(dropdown).values);
            auralise.OnDropdownOption("sound", "Load...");
            Assert.Equal(new[] { null, "sounds" }, asked);
        }

        [Fact]
        public void Load_RecordsItsUndoStepBeforeItRemembersTheNewFile()
        {
            var auralise = new AuraliseComponent();
            var (first, second) = (Path.Combine("a", "first.wav"), Path.Combine("b", "second.wav"));
            auralise.OpenSound = _ => first;
            auralise.OnDropdownOption("sound", "Load...");
            ControlState.Choose(auralise, auralise.ControlPort("sound")!, "viola");
            // ControlState.Choose records the undo step (a snapshot written by Write) before it expires the component
            var loaded = typeof(AuraliseComponent).GetField("_loaded", BindingFlags.Instance | BindingFlags.NonPublic)!;
            var during = new List<object?>();
            auralise.SolutionExpired += (_, _) => during.Add(loaded.GetValue(auralise));
            auralise.OpenSound = _ => second;
            auralise.OnDropdownOption("sound", "Load...");
            Assert.NotEmpty(during);
            Assert.All(during, value => Assert.Equal(first, value));
            Assert.Equal(second, auralise.Loaded);
        }

        [Fact]
        public void ALiteralLoadValue_IsNoLoadedFile()
        {
            var auralise = new AuraliseComponent();
            ControlState.Choose(auralise, auralise.ControlPort("sound")!, "Load...");
            Assert.Null(auralise.Loaded);
            Assert.Single(auralise.OptionsOf(AuraliseComponent.SoundDropdown).values, v => v == "Load...");
            var port = auralise.ControlPort("sound")!;
            port.AddSource(new Param_String());
            port.AddVolatileData(new GH_Path(0), 0, new GH_String("Load..."));
            Assert.Single(auralise.OptionsOf(AuraliseComponent.SoundDropdown).values, v => v == "Load...");
        }

        [Fact]
        public void Load_Cancelled_KeepsTheSelection()
        {
            var auralise = new AuraliseComponent();
            ControlState.Choose(auralise, auralise.ControlPort("sound")!, "viola");
            auralise.OpenSound = _ => null;
            Assert.True(auralise.OnDropdownOption("sound", "Load..."));
            Assert.Equal("viola", ControlState.Persistent(auralise.ControlPort("sound")!));
            Assert.Null(auralise.Loaded);
            Assert.Equal(new[] { "female_speech", "male_speech", "viola", "guitar", "castanets", "Load..." }, auralise.OptionsOf(AuraliseComponent.SoundDropdown).values);
        }

        [Fact]
        public void AnyOtherOption_IsSetByTheDropdown()
        {
            var auralise = new AuraliseComponent();
            auralise.OpenSound = _ => throw new InvalidOperationException("no dialog");
            Assert.False(auralise.OnDropdownOption("sound", "viola"));
            Assert.False(auralise.OnDropdownOption("level", "Load..."));
            Assert.False(new AnalyseIrComponent().OnDropdownOption("source_idx", "Load..."));
        }

        [Fact]
        public void AWiredFilePath_WinsAndShowsItsFileName()
        {
            var auralise = new AuraliseComponent();
            var port = auralise.ControlPort("sound")!;
            ControlState.Choose(auralise, port, "viola");
            port.AddSource(new Param_String());
            var wired = Path.Combine("takes", "bsn_tele_44.wav");
            port.AddVolatileData(new GH_Path(0), 0, new GH_String(wired));
            var options = auralise.OptionsOf(AuraliseComponent.SoundDropdown);
            var (value, isWired) = ControlState.Of(port, options.values, ComponentBase.DefaultOf(AuraliseComponent.SoundDropdown));
            Assert.True(isWired);
            Assert.Equal("bsn_tele_44.wav", options.labels[Array.IndexOf(options.values, value)]);
        }
    }

    /// <summary>A solution's inputs, as Grasshopper's data access gives them: the trees by input name, nothing else connected.</summary>
    internal sealed class FakeAccess : IGH_DataAccess
    {
        public Dictionary<string, object> Trees { get; } = new();
        public Dictionary<string, object> Items { get; } = new();
        public Dictionary<string, System.Collections.IEnumerable> Lists { get; } = new();
        public Dictionary<string, List<object?>> OutputLists { get; } = new();

        public bool GetDataTree<T>(string name, out GH_Structure<T> tree) where T : IGH_Goo
        {
            tree = Trees.TryGetValue(name, out var given) ? (GH_Structure<T>)given : null!;
            return tree != null;
        }

        public bool GetDataTree<T>(int index, out GH_Structure<T> tree) where T : IGH_Goo
        {
            tree = null!;
            return false;
        }

        public bool GetData<T>(int index, ref T destination) => false;
        public bool GetData<T>(string name, ref T destination)
        {
            if (!Items.TryGetValue(name, out var given) || given is not T value)
                return false;
            destination = value;
            return true;
        }
        public bool GetDataList<T>(int index, List<T> list) => false;
        public bool GetDataList<T>(string name, List<T> list)
        {
            if (!Lists.TryGetValue(name, out var given))
                return false;
            list.AddRange(given.Cast<T>());
            return true;
        }
        public int Iteration => 0;
        public void AbortComponentSolution() { }
        public bool BlitData<Q>(int paramIndex, GH_Structure<Q> tree, bool overwrite) where Q : IGH_Goo => false;
        public void DisableGapLogic() { }
        public void DisableGapLogic(int paramIndex) { }
        public void IncrementIteration() { }
        public int ParameterTargetIndex(int paramIndex) => 0;
        public GH_Path ParameterTargetPath(int paramIndex) => new(0);
        public bool SetData(int paramIndex, object data) => false;
        public bool SetData(int paramIndex, object data, int itemIndexOverride) => false;
        public bool SetData(string paramName, object data) => false;
        public bool SetDataList(int paramIndex, System.Collections.IEnumerable data) => false;
        public bool SetDataList(int paramIndex, System.Collections.IEnumerable data, int listIndexOverride) => false;
        public bool SetDataList(string paramName, System.Collections.IEnumerable data)
        {
            OutputLists[paramName] = data.Cast<object?>().ToList();
            return true;
        }
        public bool SetDataTree(int paramIndex, IGH_DataTree tree) => false;
        public bool SetDataTree(int paramIndex, IGH_Structure tree) => false;
        public int Util_CountNonNullRefs<T>(List<T> list) => 0;
        public int Util_CountNullRefs<T>(List<T> list) => 0;
        public bool Util_EnsureNonNullCount<T>(List<T> list, int n) => false;
        public int Util_FirstNonNullItem<T>(List<T> list) => -1;
        public List<T> Util_RemoveNullRefs<T>(List<T> list) => list;
    }

    [Collection("worker override")]
    public class AuraliseComponentTests : IDisposable
    {
        private readonly List<string> _calls = new();
        private readonly List<IDictionary<string, object?>?> _payloads = new();
        private readonly string _home = Directory.CreateTempSubdirectory("compas_acoustics_home_").FullName;

        public AuraliseComponentTests()
        {
            var python = Path.Combine(_home, "python");
            File.WriteAllText(python, "");
            Worker.Override = new Worker(new FakeStarter(), (uri, token) => new WorkerClient(uri, token, new FakeHandler(Answer)),
                name => name == Worker.PythonVariable ? python : null, _home, TimeSpan.Zero);
        }

        public void Dispose()
        {
            Worker.Override = null;
            Directory.Delete(_home, true);
        }

        private HttpResponseMessage Answer(HttpRequestMessage request, byte[] body)
        {
            var path = request.RequestUri!.AbsolutePath;
            if (path == "/ping")
                return FakeHandler.Pong();
            if (path == "/api/info")
                return FakeHandler.Result(new Dictionary<string, object?> { ["compas_acoustics"] = Versions.Library, ["schema"] = Schema.Hash, ["hybrid"] = true });
            _calls.Add(path);
            _payloads.Add(body.Length > 0 ? FakeHandler.Serializer.Unpack(body) as IDictionary<string, object?> : null);
            return FakeHandler.Result(null);
        }

        private static void Solve(ComponentBase component, IGH_DataAccess access) =>
            typeof(GH_Component).GetMethod("SolveInstance", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(component, new object?[] { access });

        [Theory]
        [InlineData("nothing chosen")]
        [InlineData("nothing chosen, hidden")]
        [InlineData("an empty wire")]
        public void Auralise_NoSound_WarnsAndCallsNoWorker(string sound)
        {
            var irs = new GH_Structure<ImpulseResponseGoo>();
            irs.Append(AuraliseGroupsTests.Ir("rec", "a"), new GH_Path(0));
            irs.Append(AuraliseGroupsTests.Ir("rec", "b"), new GH_Path(0));
            var access = new FakeAccess();
            access.Trees["irs"] = irs;
            var auralise = new AuraliseComponent();
            if (sound == "an empty wire")
                auralise.ControlPort("sound")!.AddSource(new Param_String());
            if (sound == "nothing chosen, hidden")
                auralise.ShowControlPorts(false);
            Solve(auralise, access);
            var warning = Assert.Single(auralise.RuntimeMessages(GH_RuntimeMessageLevel.Warning));
            Assert.Contains("'sound'", warning);
            Assert.Empty(auralise.RuntimeMessages(GH_RuntimeMessageLevel.Error));
            Assert.Empty(_calls);
            // the pairs are numbered all the same: the dropdowns offer them before a sound is chosen
            Assert.Equal(new[] { "All", "0: a", "1: b" }, auralise.OptionsOf(SceneIndices.SourceDropdown).labels);
        }

        [Fact]
        public void Auralise_AWiredSoundPath_WinsOverTheChoiceAndReachesTheWorker()
        {
            var irs = new GH_Structure<ImpulseResponseGoo>();
            irs.Append(AuraliseGroupsTests.Ir("rec", "a"), new GH_Path(0));
            var access = new FakeAccess();
            access.Trees["irs"] = irs;
            var wired = Path.Combine("sounds", "wired.wav");
            access.Items["sound"] = wired;
            var auralise = new AuraliseComponent();
            var port = auralise.ControlPort("sound")!;
            ControlState.Choose(auralise, port, "viola");
            port.AddSource(new Param_String());
            Solve(auralise, access);
            var deadline = DateTime.UtcNow.AddSeconds(5);
            while (_calls.Count == 0 && DateTime.UtcNow < deadline)
                System.Threading.Thread.Sleep(10);
            Assert.Equal("/api/auralise", _calls.Single());
            Assert.Equal(wired, ((IList<object?>)_payloads.Single()!["args"]!)[1]);
            Assert.Empty(auralise.RuntimeMessages(GH_RuntimeMessageLevel.Warning));
        }

        [Fact]
        public void Auralise_ASolutionWithoutIrs_ForgetsTheLastRequest()
        {
            var irs = new GH_Structure<ImpulseResponseGoo>();
            irs.Append(AuraliseGroupsTests.Ir("rec", "a"), new GH_Path(0));
            var access = new FakeAccess();
            access.Trees["irs"] = irs;
            var auralise = new AuraliseComponent();
            var last = typeof(AuraliseComponent).GetField("_last", BindingFlags.Instance | BindingFlags.NonPublic)!;
            Solve(auralise, access);
            Assert.NotNull(last.GetValue(auralise));
            // irs disconnected: the required input has no data, so Grasshopper solves no instance
            typeof(GH_Component).GetMethod("BeforeSolveInstance", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(auralise, null);
            Assert.Null(last.GetValue(auralise));
        }
    }
}
