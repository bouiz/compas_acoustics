using System.Linq;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class PortDefaultsTests
    {
        private static PortSpec Port(string component, string port) => ComponentSpec.Load(component).Inputs.Single(p => p.Name == port);

        [Theory]
        [InlineData("CA_Simulate_Mono", "ism_order", 3)]
        [InlineData("CA_Simulate_Mono", "n_rays", 20000)]
        [InlineData("CA_Simulate_Ambisonics", "sh_order", 1)]
        [InlineData("CA_Simulate_Ambisonics", "ism_order", 3)]
        [InlineData("CA_Simulate_Binaural", "ism_order", 3)]
        [InlineData("CA_Visualise_Rays", "n_rays", 500)]
        [InlineData("CA_Visualise_Rays", "max_order", 10)]
        [InlineData("CA_Model", "sampling_rate", 48000)]
        public void Specs_DeclareTheIntegerDefaultsOfThePlan(string component, string port, int value) => Assert.Equal(value, Port(component, port).Default);

        [Fact]
        public void Specs_DeclareTheNumberDefaultsOfThePlan()
        {
            Assert.Equal(1e-7, Port("CA_Simulate_Mono", "energy_thres").Default);
            Assert.Equal(1.0, Port("CA_Regularized_Fitness", "k").Default);
            Assert.Null(Port("CA_Simulate_Mono", "seed").Default);
            Assert.Null(Port("CA_Simulate_Mono", "time_thres").Default);
            Assert.Null(Port("CA_Visualise_Billiard", "speed").Default);
        }

        [Fact]
        public void APortWithADefault_NoLongerSaysItInItsDescription()
        {
            foreach (var key in ComponentSpec.Names())
                foreach (var port in ComponentSpec.Load(key).Inputs.Where(p => p.Default != null))
                {
                    Assert.DoesNotContain("(default", port.Description);
                    Assert.DoesNotContain("Default is", port.Description);
                }
        }

        [Fact]
        public void AComponent_ShowsItsDefaultOnThePort()
        {
            var component = new SimulateMonoComponent();
            var ism = Assert.IsType<Param_Integer>(component.Params.Input.Single(p => p.Name == "ism_order"));
            Assert.Equal(3, ism.PersistentData.AllData(true).OfType<GH_Integer>().Single().Value);
            var energy = Assert.IsType<Param_Number>(component.Params.Input.Single(p => p.Name == "energy_thres"));
            Assert.Equal(1e-7, energy.PersistentData.AllData(true).OfType<GH_Number>().Single().Value);
            var seed = Assert.IsType<Param_Integer>(component.Params.Input.Single(p => p.Name == "seed"));
            Assert.True(seed.PersistentData.IsEmpty);
        }

        [Fact]
        public void NamedValues_AreDeclaredAndOnThePort()
        {
            Assert.Equal(new[] { ("1", 1), ("2", 2), ("3", 3) }, Port("CA_Simulate_Ambisonics", "sh_order").NamedValues);
            Assert.Equal(Enumerable.Range(0, 11).Select(i => (i.ToString(), i)), Port("CA_Simulate_Mono", "ism_order").NamedValues);
            Assert.Equal(new[] { 44100, 48000, 96000 }, Port("CA_Model", "sampling_rate").NamedValues.Select(v => v.value));
            var hybrid = new SimulateAmbisonicsComponent();
            Assert.True(((Param_Integer)hybrid.Params.Input.Single(p => p.Name == "sh_order")).HasNamedValues);
            Assert.False(((Param_Integer)hybrid.Params.Input.Single(p => p.Name == "seed")).HasNamedValues);
        }

        [Theory]
        [InlineData(0, 3, 3, true)]
        [InlineData(1, 3, 3, false)]
        [InlineData(0, 3, 4, false)]
        [InlineData(0, null, 3, false)]
        public void AnUntouchedDefault_ReadsAsUnset(int sources, object? declared, object value, bool unset) =>
            Assert.Equal(unset, PortDefaults.Untouched(sources, declared, value));

        [Fact]
        public void AnUntouchedNumberDefault_ReadsAsUnset() => Assert.True(PortDefaults.Untouched(0, 1e-7, 1e-7));

        [Fact]
        public void SceneIndices_NameEachIndex() =>
            Assert.Equal(new[] { ("0: src", 0), ("1: b", 1) }, SceneIndices.Of(new[] { "src", "b" }));
    }
}
