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
        public void Message_NamesFormatsLevelAndPeak()
        {
            Assert.Equal("binaural + mono, relative, peak -1.0 dBFS", AuraliseGroups.Message(new[] { "mono", "binaural", "mono" }, "relative", -1.0));
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
                new DropdownControl("level", new[] { "relative", "individual", "none" }, "relative"),
                SceneIndices.SourceDropdown, SceneIndices.ReceiverDropdown,
                new ButtonControl("sound", "Anechoic sound"), new ButtonControl("save_rir", "Save RIR"), new ButtonControl("save_auralisation", "Save auralisation"),
            }.Select(c => c.ToString()), auralise.Controls.Select(c => c.ToString()));
            Assert.DoesNotContain(auralise.Controls, c => c.Port == "mix_sources");
            Assert.Equal(new[] { "source_idx", "receiver_idx", "sound" }, auralise.ControlPorts);
            Assert.True(auralise.HandlesButton("sound"));
            Assert.True(auralise.HandlesButton("save_rir"));
            Assert.True(auralise.HandlesButton("save_auralisation"));
            Assert.False(auralise.HandlesButton("run"));
            Assert.Equal("Anechoic sound", auralise.ButtonLabel("sound"));
            ControlState.Choose(auralise, auralise.ControlPort("sound")!, "/sounds/viola.wav");
            Assert.Equal("Anechoic sound: viola.wav", auralise.ButtonLabel("sound"));
            Assert.Null(auralise.ButtonLabel("save_rir"));
        }
    }

    /// <summary>A solution's inputs, as Grasshopper's data access gives them: the trees by input name, nothing else connected.</summary>
    internal sealed class FakeAccess : IGH_DataAccess
    {
        public Dictionary<string, object> Trees { get; } = new();
        public Dictionary<string, object> Items { get; } = new();

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
        public bool GetDataList<T>(string name, List<T> list) => false;
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
        public bool SetDataList(string paramName, System.Collections.IEnumerable data) => false;
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
        private readonly string _home = Directory.CreateTempSubdirectory("compas_acoustics_home_").FullName;
        private bool _fail;

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
            if (_fail)
                return FakeHandler.Answer(System.Net.HttpStatusCode.BadRequest, new Dictionary<string, object?> { ["error"] = "ImportError: broken" });
            return FakeHandler.Result(path == "/api/anechoic_library_folder" ? "/site-packages/compas_acoustics/anechoic_soundfiles_library" : null);
        }

        private static void Solve(ComponentBase component, IGH_DataAccess access) =>
            typeof(GH_Component).GetMethod("SolveInstance", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(component, new object?[] { access });

        [Fact]
        public void Auralise_NoSound_Warns()
        {
            var irs = new GH_Structure<ImpulseResponseGoo>();
            irs.Append(AuraliseGroupsTests.Ir("rec", "a"), new GH_Path(0));
            irs.Append(AuraliseGroupsTests.Ir("rec", "b"), new GH_Path(0));
            var access = new FakeAccess();
            access.Trees["irs"] = irs;
            var auralise = new AuraliseComponent();
            Solve(auralise, access);
            var warning = Assert.Single(auralise.RuntimeMessages(GH_RuntimeMessageLevel.Warning));
            Assert.Contains("'sound'", warning);
            Assert.Empty(auralise.RuntimeMessages(GH_RuntimeMessageLevel.Error));
            Assert.Empty(_calls);
            // the pairs are numbered all the same: the dropdowns offer them before a sound is chosen
            Assert.Equal(new[] { "All", "0: a", "1: b" }, auralise.OptionsOf(SceneIndices.SourceDropdown).labels);
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

        [Fact]
        public void LibraryFolder_IsTheWorkersAnechoicLibrary_OrNullWhenTheWorkerFails()
        {
            Assert.Equal("/site-packages/compas_acoustics/anechoic_soundfiles_library", AuraliseGroups.LibraryFolder());
            Assert.Equal(new[] { "/api/anechoic_library_folder" }, _calls);
            _fail = true;
            Assert.Null(AuraliseGroups.LibraryFolder());
        }
    }
}
