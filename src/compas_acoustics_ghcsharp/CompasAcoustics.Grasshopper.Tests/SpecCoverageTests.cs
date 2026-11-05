using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>Every spec has its component, read from the assembly's metadata without loading the types.</summary>
    public class SpecCoverageTests
    {
        // spec folder: component class
        private static readonly Dictionary<string, string> Components = new()
        {
            ["CA_Acoustic_Object"] = "AcousticObjectComponent",
            ["CA_Analyse_IR"] = "AnalyseIrComponent",
            ["CA_Audio_Player"] = "AudioPlayerComponent",
            ["CA_Auralise"] = "AuraliseComponent",
            ["CA_Channel"] = "CheckChannelComponent",
            ["CA_Construct_Material"] = "ConstructMaterialComponent",
            ["CA_Deconstruct_IR"] = "DeconstructParameterComponent",
            ["CA_Deconstruct_IR_Graph"] = "DeconstructIrGraphComponent",
            ["CA_Deconstruct_Material"] = "DeconstructMaterialComponent",
            ["CA_Install_Worker"] = "InstallWorkerComponent",
            ["CA_Material_Library"] = "MaterialLibraryComponent",
            ["CA_Model"] = "RoomSceneComponent",
            ["CA_Plot_2D"] = "Plot2DComponent",
            ["CA_Receiver"] = "ReceiverComponent",
            ["CA_Regularized_Fitness"] = "RegularizedFitnessComponent",
            ["CA_Simulate_Ambisonics"] = "SimulateAmbisonicsComponent",
            ["CA_Simulate_Binaural"] = "SimulateBinauralComponent",
            ["CA_Simulate_Mono"] = "SimulateMonoComponent",
            ["CA_Source"] = "SourceComponent",
            ["CA_Visualise_Billiard"] = "VisualiseBilliardComponent",
            ["CA_Visualise_Energy_Map"] = "VisualiseEnergyMapComponent",
            ["CA_Visualise_Rays"] = "VisualiseRaysComponent",
            ["CA_Visualise_Reflections"] = "VisualiseReflectionsComponent",
            ["CA_Worker"] = "WorkerComponent",
        };

        private static HashSet<string> TypeNames()
        {
            using var stream = File.OpenRead(typeof(ComponentBase).Assembly.Location);
            using var pe = new PEReader(stream);
            var reader = pe.GetMetadataReader();
            return reader.TypeDefinitions.Select(h => reader.GetString(reader.GetTypeDefinition(h).Name)).ToHashSet();
        }

        [Fact]
        public void EverySpec_HasItsComponent()
        {
            Assert.Equal(ComponentSpec.Names().OrderBy(n => n, StringComparer.Ordinal), Components.Keys.OrderBy(n => n, StringComparer.Ordinal));
            var types = TypeNames();
            Assert.All(Components.Values, name => Assert.Contains(name, types));
        }

        [Fact]
        public void EveryComponent_LoadsInTheTestHost()
        {
            var loadable = AllComponentsTests.Components().Select(c => (string)c[0]).ToHashSet();
            var notLoaded = Components.Values.Where(name => !loadable.Contains(name)).OrderBy(n => n).ToArray();
            Assert.Empty(notLoaded);
        }
    }
}
