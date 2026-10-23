using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Grasshopper.Kernel;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class ComponentBaseTests
    {
        [Fact]
        public void Specs_AreEmbeddedForEveryComponent()
        {
            Assert.Equal(23, ComponentSpec.Names().Count());
            var model = ComponentSpec.Load("CA_Model");
            Assert.Equal("Acoustic Scene", model.Name);
            Assert.Equal("Scene", model.Nickname);
            Assert.Equal("01 Model", model.Subcategory);
            Assert.Equal(GH_Exposure.primary, model.Exposure);
            Assert.Equal(new[] { "objects", "sources", "receivers", "name", "sampling_rate" }, model.Inputs.Select(i => i.Name).ToArray());
            var objects = model.Inputs[0];
            Assert.Equal(GH_ParamAccess.list, objects.Access);
            Assert.True(objects.Flatten);
            Assert.False(objects.Optional);
            Assert.Equal("ghdoc", objects.TypeHint);
            Assert.True(model.Inputs[3].Optional);
            Assert.Equal(new[] { "scene" }, model.Outputs.Select(o => o.Name).ToArray());
        }

        [Fact]
        public void Specs_ReadTreeAccessAndOutputDescriptions()
        {
            var auralise = ComponentSpec.Load("CA_Auralise");
            Assert.Equal(GH_ParamAccess.tree, auralise.Inputs[0].Access);
            Assert.False(string.IsNullOrEmpty(auralise.Outputs[0].Description));
        }

        [Fact]
        public void Specs_HaveAnIcon()
        {
            Assert.All(ComponentSpec.Names(), name => Assert.True(ComponentSpec.IconBytes(name).Length > 0));
        }

        [Fact]
        public void ErrorText_TakesTheLastLineOfATraceback()
        {
            var error = new WorkerException(WorkerException.Summary("Traceback (most recent call last):\n  File \"x\"\nValueError: bad\n"));
            Assert.Equal("ValueError: bad", ComponentBase.ErrorText(error));
        }

        [Fact]
        public void ErrorText_KeepsAPlainMessage()
        {
            Assert.Equal("No worker: run the compas_acoustics installer.", ComponentBase.ErrorText(new WorkerException("No worker: run the compas_acoustics installer.")));
        }

        private sealed class Failing : ComponentBase
        {
            private readonly System.Exception _error;
            public Failing(System.Exception error) : base("CA_Channel") { _error = error; }
            protected override string SpecKey => "CA_Channel";
            public override System.Guid ComponentGuid => new System.Guid("00000000-0000-0000-0000-0000000000ab");
            protected override void Solve(IGH_DataAccess access) => throw _error;
            public void Run() => typeof(GH_Component).GetMethod("SolveInstance", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
                .Invoke(this, new object?[] { null });
        }

        [Fact]
        public void AWorkerThatStopsAnswering_IsOneErrorMessage()
        {
            var component = new Failing(new System.Net.Http.HttpRequestException("Connection refused"));
            component.Run();
            Assert.Equal(new[] { "The worker does not answer: Connection refused. Recompute to start it again." },
                component.RuntimeMessages(GH_RuntimeMessageLevel.Error).ToArray());
        }

        [Fact]
        public void AWorkerError_IsOneErrorMessage()
        {
            var component = new Failing(new WorkerException("ValueError: bad"));
            component.Run();
            Assert.Equal(new[] { "ValueError: bad" }, component.RuntimeMessages(GH_RuntimeMessageLevel.Error).ToArray());
        }

        [Fact]
        public void OrderColors_Order0IsBlack_Order1IsTab10Red_Order11RepeatsOrder1()
        {
            Assert.Equal(Color.FromArgb(0, 0, 0).ToArgb(), OrderColors.Of(0).ToArgb());
            Assert.Equal(Color.FromArgb(214, 39, 40).ToArgb(), OrderColors.Of(1).ToArgb());
            Assert.Equal(Color.FromArgb(31, 119, 180).ToArgb(), OrderColors.Of(2).ToArgb());
            Assert.Equal(OrderColors.Of(1).ToArgb(), OrderColors.Of(11).ToArgb());
            Assert.Equal(OrderColors.Of(0).ToArgb(), OrderColors.Of(-3).ToArgb());
        }
    }
}
