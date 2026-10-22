using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>Definitions saved with an earlier build: a component that has gained inputs since reads without errors.</summary>
    public class ArchiveCompatibilityTests
    {
        // the test runner reads these signatures before WinFormsResolver is registered: no plugin type in them
        private static (GH_Archive archive, bool read) SaveThenRead(GH_Component saved, GH_Component fresh)
        {
            // as Grasshopper does on the canvas: a component and its parameters write their chunks with their attributes
            foreach (var c in new[] { saved, fresh })
            {
                c.CreateAttributes();
                foreach (var p in c.Params)
                    if (p.Attributes == null)
                        p.CreateAttributes();
            }
            var archive = new GH_Archive();
            Assert.True(saved.Write(archive.CreateTopLevelNode("component")));
            var back = new GH_Archive();
            back.Deserialize_Binary(archive.Serialize_Binary());
            var read = fresh.Read(back.GetRootNode.FindChunk("component"));
            return (back, read);
        }

        private static string[] Errors(GH_Archive archive) =>
            (archive.Messages ?? new System.Collections.Generic.List<GH_Message>()).Where(m => m.Type != GH_Message_Type.info).Select(m => m.Message).ToArray();

        [Fact]
        public void AMaterialLibrarySavedBeforeItsInputs_ReadsWithoutErrors()
        {
            var old = new MaterialLibraryComponent();
            foreach (var input in old.Params.Input.ToList())
                old.Params.UnregisterInputParameter(input);
            var fresh = new MaterialLibraryComponent();
            var (archive, read) = SaveThenRead(old, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.Equal(new[] { "absorption", "scattering" }, fresh.Params.Input.Select(p => p.Name));
        }

        [Fact]
        public void ASourceSavedWithItsFormerDirectivityInput_ReadsEachInputByName()
        {
            // the Source had a directivity input before sound_file; its saved choice must not land in sound_file
            var old = new SourceComponent();
            var directivity = new Param_String { Name = "directivity", NickName = "directivity" };
            directivity.PersistentData.Append(new GH_String("cardioid"));
            old.Params.RegisterInputParam(directivity, old.Params.Input.FindIndex(p => p.Name == "sound_file"));
            var fresh = new SourceComponent();
            var (archive, read) = SaveThenRead(old, fresh);
            Assert.True(read);
            Assert.Empty(Errors(archive));
            Assert.Equal(new[] { "position", "name", "direction", "sound_file" }, fresh.Params.Input.Select(p => p.Name));
            Assert.True(fresh.Params.Input.Single(p => p.Name == "sound_file").VolatileDataCount == 0
                        && ((Param_String)fresh.Params.Input.Single(p => p.Name == "sound_file")).PersistentData.IsEmpty);
            Assert.Equal(old.Params.Input.Single(p => p.Name == "sound_file").InstanceGuid, fresh.Params.Input.Single(p => p.Name == "sound_file").InstanceGuid);
        }

        [Fact]
        public void AComponentSavedNow_ReadsWithoutErrors()
        {
            var (archive, read) = SaveThenRead(new SimulateHybridComponent(), new SimulateHybridComponent());
            Assert.True(read);
            Assert.Empty(Errors(archive));
        }
    }
}
