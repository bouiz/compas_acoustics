using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CompasAcoustics.Pb;
using CompasPb.Data;
using Grasshopper.Kernel.Data;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>A test that needs a real worker: COMPAS_ACOUSTICS_WORKER_PYTHON, a Python with this checkout's compas_acoustics.</summary>
    public sealed class WorkerFactAttribute : FactAttribute
    {
        public WorkerFactAttribute()
        {
            var python = Environment.GetEnvironmentVariable(Worker.PythonVariable);
            if (string.IsNullOrEmpty(python) || !File.Exists(python))
                Skip = "set COMPAS_ACOUSTICS_WORKER_PYTHON to a Python with this checkout's compas_acoustics";
        }
    }

    public class WorkerIntegrationTests
    {
        internal static Worker RealWorker() => new Worker(new ProcessStarter(), (uri, token) => new WorkerClient(uri, token),
            Environment.GetEnvironmentVariable, Directory.CreateTempSubdirectory("compas_acoustics_home_").FullName, TimeSpan.FromMilliseconds(100));

        internal static MeshData Quad(params double[][] corners)
        {
            var mesh = new MeshData();
            foreach (var corner in corners)
                mesh.Vertices.Add(corner);
            mesh.FaceVertices.Add(new uint[] { 0, 1, 2, 3 });
            mesh.FaceSizes.Add(4);
            return mesh;
        }

        internal static IEnumerable<MeshData> ShoeboxWalls(double x, double y, double z)
        {
            double[] P(double a, double b, double c) => new[] { a, b, c };
            yield return Quad(P(0, 0, 0), P(0, y, 0), P(x, y, 0), P(x, 0, 0));
            yield return Quad(P(0, 0, z), P(x, 0, z), P(x, y, z), P(0, y, z));
            yield return Quad(P(0, 0, 0), P(x, 0, 0), P(x, 0, z), P(0, 0, z));
            yield return Quad(P(x, 0, 0), P(x, y, 0), P(x, y, z), P(x, 0, z));
            yield return Quad(P(x, y, 0), P(0, y, 0), P(0, y, z), P(x, y, z));
            yield return Quad(P(0, y, 0), P(0, 0, 0), P(0, 0, z), P(0, y, z));
        }

        [WorkerFact]
        public void RealWorker_BuildsASceneFromCSharpMessagesAndSimulates()
        {
            var worker = RealWorker();
            try
            {
                var material = (AcousticMaterialData)worker.Call("make_material", "wall", new List<object?> { 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 })!;
                var objects = ShoeboxWalls(7, 5, 3).Select(wall => worker.Call("make_acoustic_object", wall, material)).ToList<object?>();
                Assert.All(objects, o => Assert.IsType<AcousticObjectData>(o));
                var source = (SourceData)worker.Call("make_source", "src", new List<object?> { 1.5, 1.5, 1.6 })!;
                var receiver = (IDictionary<string, object?>)worker.Call("make_receiver", "rec", new List<object?> { 5.0, 3.0, 1.5 })!;
                Assert.Equal(new[] { "frame_overrides", "receiver" }, receiver.Keys.OrderBy(k => k).ToArray());
                var scene = (SceneData)worker.Call("make_scene", objects, new List<object?> { source }, new List<object?> { receiver["receiver"] }, "room", 44100L)!;
                Assert.Equal(6, scene.AcousticObjects.Count);
                var settings = new Dictionary<string, object?> { ["image_source_order"] = 1L, ["ray_tracing"] = false };
                var result = (IDictionary<string, object?>)worker.Call("simulate", scene, "mono", settings)!;
                var responses = (IDictionary<string, object?>)((IDictionary<string, object?>)result["responses"]!)["mono"]!;
                var ir = (ImpulseResponseData)responses.Values.Single()!;
                Assert.Equal(44100, ir.SamplingRate);
                Assert.True(ir.Samples.Count > 1000);
                Assert.Equal(source.Guid, ir.Source.Guid);
                Assert.Equal($"{ir.Receiver.Guid}_{ir.Source.Guid}", responses.Keys.Single());
                Assert.True(worker.SpeedOfSound > 300);
            }
            finally
            {
                worker.Stop();
            }
        }

        [WorkerFact]
        public void RealWorker_AnswersAnErrorAsAWorkerException()
        {
            var worker = RealWorker();
            try
            {
                var error = Assert.Throws<WorkerException>(() => worker.Call("make_receiver", "r", new List<object?> { 0.0, 0.0, 1.0 },
                    new List<object?> { 0.0, 0.0, 1.0 }));
                Assert.Contains("parallel", error.Message);
                Assert.Equal(400, error.Status);
            }
            finally
            {
                worker.Stop();
            }
        }
    
        [WorkerFact]
        public void RealWorker_TakesAFrameAndAPolylineFromRhinoGeometry()
        {
            var worker = RealWorker();
            try
            {
                var plane = Rhino.Geometry.Plane.WorldXY;
                plane.Origin = new Rhino.Geometry.Point3d(1, 2, 3);
                plane.XAxis = new Rhino.Geometry.Vector3d(0, 1, 0);
                plane.YAxis = new Rhino.Geometry.Vector3d(-1, 0, 0);
                plane.ZAxis = new Rhino.Geometry.Vector3d(0, 0, 1);
                var result = (IDictionary<string, object?>)worker.Call("make_receiver", new object?[] { "r" },
                    new Dictionary<string, object?> { ["frame"] = MeshConversion.ToFrameData(plane), ["position"] = null })!;
                var receiver = (ReceiverData)result["receiver"]!;
                Assert.Equal(1.0, receiver.Direction.Y, 9);
                Assert.Equal(1.0, receiver.Up.Z, 9);
                Assert.Equal(new[] { 1.0, 2.0, 3.0 }, new[] { receiver.Point.X, receiver.Point.Y, receiver.Point.Z });
                Assert.Equal(false, result["frame_overrides"]);
                var polyline = new Rhino.Geometry.Polyline(new[] { new Rhino.Geometry.Point3d(0, 0, 0), new Rhino.Geometry.Point3d(1, 0, 0), new Rhino.Geometry.Point3d(1, 1, 0) });
                var material = worker.Call("make_material", "m", new List<object?> { 0.1 });
                var obj = (AcousticObjectData)worker.Call("make_acoustic_object", MeshConversion.ToPolylineData(polyline), material, null, "obstacle")!;
                Assert.Equal(AcousticObjectData.GeometryOneofCase.Polyline, obj.GeometryCase);
                Assert.Equal("obstacle", obj.Role);
            }
            finally
            {
                worker.Stop();
            }
        }
    
        internal static SceneData Shoebox(Worker worker)
        {
            var material = worker.Call("make_material", "wall", new List<object?> { 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 });
            var objects = ShoeboxWalls(7, 5, 3).Select(wall => worker.Call("make_acoustic_object", wall, material)).ToList<object?>();
            var source = worker.Call("make_source", "src", new List<object?> { 1.5, 1.5, 1.6 });
            var receiver = ((IDictionary<string, object?>)worker.Call("make_receiver", "rec", new List<object?> { 5.0, 3.0, 1.5 })!)["receiver"];
            return (SceneData)worker.Call("make_scene", objects, new List<object?> { source }, new List<object?> { receiver }, "room", 32000L)!;
        }

        [WorkerFact]
        public void RealWorker_ACancelledSimulationStopsAtItsNextStage()
        {
            var worker = RealWorker();
            try
            {
                var material = (AcousticMaterialData)worker.Call("make_material", "wall", new List<object?> { 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 })!;
                var objects = ShoeboxWalls(7, 5, 3).Select(wall => worker.Call("make_acoustic_object", wall, material)).ToList<object?>();
                var source = worker.Call("make_source", "src", new List<object?> { 1.5, 1.5, 1.6 });
                var receivers = Enumerable.Range(0, 40).Select(i =>
                    ((IDictionary<string, object?>)worker.Call("make_receiver", $"rec{i}", new List<object?> { 2.0 + 0.1 * i, 3.0, 1.5 })!)["receiver"]).ToList();
                var scene = worker.Call("make_scene", objects, new List<object?> { source }, receivers, "room", 44100L);
                var settings = new Dictionary<string, object?> { ["image_source_order"] = 8L, ["ray_tracing"] = false };
                using var cancel = new System.Threading.CancellationTokenSource();
                var call = worker.CallAsync("simulate", "cancel-test", new object?[] { scene, "mono", settings }, new Dictionary<string, object?>(), cancel.Token);
                var watch = System.Diagnostics.Stopwatch.StartNew();
                string? stage = null;
                while (!call.IsCompleted && stage == null && watch.Elapsed < TimeSpan.FromSeconds(60))
                {
                    stage = WorkerJob.Describe(worker.Progress("cancel-test"));
                    System.Threading.Thread.Sleep(20);
                }
                Assert.Equal("image sources", stage);
                cancel.Cancel();
                // the worker answered 499: a finished simulation would complete the task with its result
                var ended = Assert.Throws<AggregateException>(() => call.Wait(TimeSpan.FromSeconds(30)));
                Assert.True(call.IsCanceled, ended.ToString());
                Assert.Null(worker.Progress("cancel-test"));
                Assert.IsType<SourceData>(worker.Call("make_source", "after", new List<object?> { 1.0, 1.0, 1.0 }));
            }
            finally
            {
                worker.Stop();
            }
        }

        [WorkerFact]
        public void RealWorker_RunsBinauralAndAmbisonicsWithAMonoReference()
        {
            var worker = RealWorker();
            try
            {
                worker.EnsureStarted();
                if (!worker.Hybrid)
                    return;
                var scene = new SceneGoo(Shoebox(worker));
                var mono = (IDictionary<string, object?>)worker.Call("simulate", scene.Value, "mono",
                    new Dictionary<string, object?> { ["image_source_order"] = 1L, ["ray_tracing"] = false })!;
                var reference = IrGraphGoo.FromSimulation(scene, "mono",
                    (IDictionary<string, object?>)((IDictionary<string, object?>)mono["responses"]!)["mono"]!);
                var settings = HybridRun.Settings("reference", 1, null, null, 3, null, reference, scene.SamplingRate, out var references);
                IrGraphGoo Run(string method, Dictionary<string, object?> given)
                {
                    var result = (IDictionary<string, object?>)worker.Call("simulate", scene.Value, method, given,
                        references!.ToDictionary(kv => kv.Key, kv => (object?)kv.Value))!;
                    var responses = (IDictionary<string, object?>)result["responses"]!;
                    Assert.Equal(new[] { method }, responses.Keys);
                    return IrGraphGoo.FromSimulation(scene, method, (IDictionary<string, object?>)responses[method]!);
                }
                var binaural = Run("binaural", settings);
                var ambisonic = Run("ambisonics", new Dictionary<string, object?>(settings) { ["sh_order"] = 1L });
                Assert.Equal(2, binaural.Value.Responses.Single().Channels);
                Assert.Equal("binaural", binaural.Value.Responses.Single().Format);
                Assert.Equal(4, ambisonic.Value.Responses.Single().Channels);
                Assert.Equal("reference", ambisonic.Value.Responses.Single().Generation["tail"]);
            }
            finally
            {
                worker.Stop();
            }
        }
    
        [WorkerFact]
        public void RealWorker_AnalysesAResponseAndReadsASoundFile()
        {
            var worker = RealWorker();
            try
            {
                var scene = new SceneGoo(Shoebox(worker));
                var mono = (IDictionary<string, object?>)worker.Call("simulate", scene.Value, "mono",
                    new Dictionary<string, object?> { ["image_source_order"] = 2L, ["ray_tracing"] = false })!;
                var ir = (ImpulseResponseData)((IDictionary<string, object?>)((IDictionary<string, object?>)mono["responses"]!)["mono"]!).Values.Single()!;
                var result = (IDictionary<string, object?>)worker.Call("analysis", ir, "db")!;
                var parameters = ((IEnumerable<object?>)result["parameters"]!).Cast<IDictionary<string, object?>>().Select(ParameterGoo.FromDictionary).ToList();
                Assert.Contains(parameters, p => p.Value.Name == "T30" && p.Value.Values[0].Length == p.Value.Frequencies.Length);
                Assert.Contains(parameters, p => p.Value.Name == "STI");
                var time = ParameterGoo.FromDictionary((IDictionary<string, object?>)result["time"]!);
                Assert.Equal("linear", time.Value.XScale);
                Assert.StartsWith("T30, Time (s): ", parameters.Single(p => p.Value.Name == "T30").ToString());
                var viola = Path.Combine(RepositoryRootForTests(), "src", "compas_acoustics", "anechoic_soundfiles_library", "viola.wav");
                Assert.True(SoundChannels.Of(viola, path => Convert.ToInt32(((IDictionary<string, object?>)worker.Call("sound_info", path)!)["channels"])) >= 1);
            }
            finally
            {
                worker.Stop();
            }
        }

        private static string RepositoryRootForTests() => InteropTests.RepositoryRoot();
    
        [WorkerFact]
        public void RealWorker_AuralisesAResponseWithItsSourcesSound()
        {
            var worker = RealWorker();
            try
            {
                var viola = Path.Combine(InteropTests.RepositoryRoot(), "src", "compas_acoustics", "anechoic_soundfiles_library", "viola.wav");
                var material = worker.Call("make_material", "wall", new List<object?> { 0.3 });
                var objects = ShoeboxWalls(7, 5, 3).Select(wall => worker.Call("make_acoustic_object", wall, material)).ToList<object?>();
                var source = worker.Call("make_source", "src", new List<object?> { 1.5, 1.5, 1.6 }, null, null, viola);
                var receiver = ((IDictionary<string, object?>)worker.Call("make_receiver", "rec", new List<object?> { 5.0, 3.0, 1.5 })!)["receiver"];
                var scene = worker.Call("make_scene", objects, new List<object?> { source }, new List<object?> { receiver }, "room", 16000L);
                var mono = (IDictionary<string, object?>)worker.Call("simulate", scene, "mono",
                    new Dictionary<string, object?> { ["image_source_order"] = 1L, ["ray_tracing"] = false })!;
                var ir = (ImpulseResponseData)((IDictionary<string, object?>)((IDictionary<string, object?>)mono["responses"]!)["mono"]!).Values.Single()!;
                var irs = new GH_Structure<ImpulseResponseGoo>();
                irs.Append(new ImpulseResponseGoo(ir), new GH_Path(0, 2));
                var groups = AuraliseGroups.FromTrees(irs, null, out var paths);
                var result = (IDictionary<string, object?>)worker.Call("auralise", new object?[] { groups.Cast<object?>().ToList() },
                    new Dictionary<string, object?> { ["level"] = "relative" })!;
                var signal = ((IEnumerable<object?>)result["signals"]!).Cast<IDictionary<string, object?>>().Single();
                var audio = new AudioGoo((AudioData)signal["audio"]!);
                Assert.Equal("mono", audio.Format);
                Assert.True(audio.Value.Samples.Count > 10000);
                Assert.Equal(new GH_Path(0, 2), paths[(string)signal["branch"]!]);
                Assert.Equal(-1.0, Convert.ToDouble(result["peak_dbfs"]), 2);
            }
            finally
            {
                worker.Stop();
            }
        }
    
        [WorkerFact]
        public void RealWorker_TracesRaysAndFindsReflections()
        {
            var worker = RealWorker();
            try
            {
                var scene = Shoebox(worker);
                var result = (IDictionary<string, object?>)worker.Call("ray_field", new object?[] { scene },
                    new Dictionary<string, object?> { ["n_rays"] = 20L, ["max_order"] = 3L, ["scattering"] = true, ["seed"] = 1L, ["order"] = 2L })!;
                var field = (RayFieldData)result["field"]!;
                var shown = (RayFieldData)result["shown"]!;
                Assert.Equal(20, field.Rays.Count);
                Assert.All(RayGeometry.Specular(shown), r => Assert.True(r.Points.Length <= 3));
                Assert.Contains(RayGeometry.FromField(field), r => r.Scattered);
                Assert.Equal(1L, result["seed"]);
                var reflections = (IDictionary<string, object?>)worker.Call("specular_reflections", scene, new List<object?> { 1L })!;
                var paths = RayGeometry.FromReflections((SpecularReflectionsData)reflections["reflections"]!);
                Assert.Equal(6, paths.Count);
                Assert.All(paths, p => Assert.Equal(1, p.Order));
                Assert.All(paths, p => Assert.True(p.Ray.TravelTimeMs(worker.SpeedOfSound) > 0));
                RayFieldGoo? asField = null;
                Assert.True(new ReflectionsGoo((SpecularReflectionsData)reflections["reflections"]!).CastTo(ref asField));
                Assert.Equal(6, asField!.Value.Rays.Count);
            }
            finally
            {
                worker.Stop();
            }
        }
    
        [WorkerFact]
        public void RealWorker_PlaybackStatusFollowsPlayPauseStop()
        {
            var worker = RealWorker();
            try
            {
                // one second of silence, so the test makes no sound
                var audio = new AudioData { SamplingRate = 8000, Format = "binaural", Shape = { 8000, 2 } };
                audio.Samples.Add(new double[16000]);
                try
                {
                    worker.Call("play", new object?[] { audio }, new Dictionary<string, object?> { ["start"] = 0.25 });
                }
                catch (WorkerException e) when (e.Message.Contains("PortAudio") || e.Message.Contains("device"))
                {
                    return; // no audio device on this machine
                }
                var status = (IDictionary<string, object?>)worker.Call("playback_status")!;
                Assert.Equal(1.0, (double)status["duration"]!, 3);
                Assert.True((double)status["position"]! >= 0.25);
                worker.Call("pause");
                status = (IDictionary<string, object?>)worker.Call("playback_status")!;
                Assert.Equal(true, status["paused"]);
                Assert.Equal(false, status["playing"]);
                worker.Call("stop");
                status = (IDictionary<string, object?>)worker.Call("playback_status")!;
                Assert.Equal(0.0, (double)status["position"]!);
            }
            finally
            {
                worker.Stop();
            }
        }
    }
}
