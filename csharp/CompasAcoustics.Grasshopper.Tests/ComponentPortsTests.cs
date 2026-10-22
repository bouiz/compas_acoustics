using System;
using System.Linq;
using Grasshopper.Kernel;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class ComponentPortsTests
    {
        private sealed class Probe : ComponentBase
        {
            public Probe() : base("CA_Model") { }
            protected override string SpecKey => "CA_Model";
            public override Guid ComponentGuid => new Guid("00000000-0000-0000-0000-0000000000aa");
            protected override void Solve(IGH_DataAccess access) { }
        }

        [Fact]
        public void AComponent_RegistersThePortsOfItsSpec()
        {
            var probe = new Probe();
            Assert.Equal("Acoustic Scene", probe.Name);
            Assert.Equal(new[] { "objects", "sources", "receivers", "name", "sampling_rate" }, probe.Params.Input.Select(p => p.Name).ToArray());
            Assert.Equal(GH_ParamAccess.list, probe.Params.Input[0].Access);
            Assert.Equal(GH_DataMapping.Flatten, probe.Params.Input[0].DataMapping);
            Assert.True(probe.Params.Input[3].Optional);
            Assert.Equal(new[] { "scene" }, probe.Params.Output.Select(p => p.Name).ToArray());
        }
    
        [Fact]
        public void TheReceiver_HasNoUpAndOutputsOnlyTheReceiver()
        {
            var receiver = new ReceiverComponent();
            Assert.Equal(new[] { "position", "name", "direction" }, receiver.Params.Input.Select(p => p.Name).ToArray());
            Assert.Equal(new[] { "receiver" }, receiver.Params.Output.Select(p => p.Name).ToArray());
            Assert.IsType<Param_Receiver>(receiver.Params.Output[0]);
        }
    }
}
