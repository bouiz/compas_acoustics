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
        [InlineData("CA_Simulate_Binaural", "seed", 42)]
        [InlineData("CA_Simulate_Ambisonics", "seed", 42)]
        [InlineData("CA_Visualise_Rays", "n_rays", 500)]
        [InlineData("CA_Visualise_Rays", "max_order", 10)]
        public void Specs_DeclareTheIntegerDefaults(string component, string port, int value) => Assert.Equal(value, Port(component, port).Default);

        [Fact]
        public void Specs_DeclareTheNumberDefaults()
        {
            Assert.Equal(1.0, Port("CA_Regularized_Fitness", "k").Default);
            Assert.Null(Port("CA_Simulate_Mono", "duration").Default);
            Assert.Null(Port("CA_Simulate_Mono", "n_rays").Default);
            Assert.Null(Port("CA_Visualise_Rays", "speed").Default);
        }

        [Fact]
        public void APortDescription_LeavesTheDefaultToThePort()
        {
            foreach (var key in EmbeddedSpecs.Names())
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
            var seed = Assert.IsType<Param_Integer>(component.Params.Input.Single(p => p.Name == "seed"));
            Assert.Equal(42, seed.PersistentData.AllData(true).OfType<GH_Integer>().Single().Value);
        }

        [Fact]
        public void NamedValues_AreDeclaredAndOnThePort()
        {
            Assert.Equal(Enumerable.Range(0, 11).Select(i => (i.ToString(), i)), Port("CA_Simulate_Mono", "ism_order").NamedValues);
            var hybrid = new SimulateAmbisonicsComponent();
            Assert.True(((Param_Integer)hybrid.Params.Input.Single(p => p.Name == "ism_order")).HasNamedValues);
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
    }
}
