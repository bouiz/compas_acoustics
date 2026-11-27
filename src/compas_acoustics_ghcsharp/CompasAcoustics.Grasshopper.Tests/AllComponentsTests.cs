using System;
using System.Collections.Generic;
using System.Linq;
using Grasshopper.Kernel;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>Every component of the plugin, built outside Rhino, against its spec (the component's metadata.json).</summary>
    public class AllComponentsTests
    {
        /// <summary>The plugin's types that this test host can load.</summary>
        private static Type[] Loadable()
        {
            try
            {
                return typeof(ComponentBase).Assembly.GetTypes();
            }
            catch (System.Reflection.ReflectionTypeLoadException e)
            {
                return e.Types.Where(t => t != null).ToArray()!;
            }
        }

        public static IEnumerable<object[]> Components() =>
            Loadable().Where(t => t.IsSubclassOf(typeof(ComponentBase)) && !t.IsAbstract).Select(t => new object[] { t.Name });

        private static ComponentBase Make(string typeName) =>
            (ComponentBase)Activator.CreateInstance(Loadable().Single(t => t.Name == typeName))!;

        [Theory]
        [MemberData(nameof(Components))]
        public void AComponent_HasThePortsNamesAndPlaceOfItsSpec(string typeName)
        {
            var component = Make(typeName);
            var key = EmbeddedSpecs.Names().Single(name => ComponentSpec.Load(name).Name == component.Name);
            var spec = ComponentSpec.Load(key);
            Assert.Equal(spec.Nickname, component.NickName);
            Assert.Equal(spec.Description, component.Description);
            Assert.Equal(ComponentBase.CategoryName, component.Category);
            Assert.Equal(ComponentBase.RibbonSubcategory(spec.Subcategory), component.SubCategory);
            Assert.Equal(spec.Inputs.Select(p => p.Name), component.Params.Input.Select(p => p.Name));
            Assert.Equal(spec.Inputs.Select(p => p.Access), component.Params.Input.Select(p => p.Access));
            Assert.Equal(spec.Inputs.Select(p => p.Optional), component.Params.Input.Select(p => p.Optional));
            Assert.Equal(spec.Outputs.Select(p => p.Name), component.Params.Output.Select(p => p.Name));
            Assert.NotEqual(Guid.Empty, component.ComponentGuid);
        }

        [Theory]
        [MemberData(nameof(Components))]
        public void TheSceneSourceReceiverSimulateAnalyseAuraliseAndVisualiseComponents_AreDrawnOneWidth_TheVisualiseOnesWiderAndTheOthersAtTheirOwn(string typeName)
        {
            var component = Make(typeName);
            var shared = typeName is nameof(RoomSceneComponent) or nameof(SourceComponent) or nameof(ReceiverComponent)
                or nameof(SimulateMonoComponent) or nameof(SimulateBinauralComponent) or nameof(SimulateAmbisonicsComponent)
                or nameof(AnalyseIrComponent) or nameof(AuraliseComponent);
            var visualise = typeName is nameof(VisualiseEnergyMapComponent) or nameof(VisualiseRaysComponent) or nameof(VisualiseBilliardComponent) or nameof(VisualiseReflectionsComponent);
            Assert.Equal(visualise ? ComponentBase.VisualiseBodyWidth : shared ? ComponentBase.SharedBodyWidth : 0, component.MinBodyWidth);
            Assert.Equal(Math.Max(component.MinBodyWidth, ControlLayout.MinWidth(component.BodyControls)), ComponentUi.MinBodyWidthOf(component));
            if (shared)
                Assert.Equal(135f, ComponentUi.MinBodyWidthOf(component));
            if (visualise)
                Assert.Equal(140f, ComponentUi.MinBodyWidthOf(component));
        }

        [Theory]
        [MemberData(nameof(Components))]
        public void AComponent_HasTypedOutputs_SoNothingDrawsThroughAGenericOne(string typeName)
        {
            // a generic output draws whatever previewable data passes through it
            var component = Make(typeName);
            Assert.Empty(component.Params.Output.Where(p => p is global::Grasshopper.Kernel.Parameters.Param_GenericObject).Select(p => p.Name));
        }

        /// <summary>The outputs each component draws in the viewport; every other component draws nothing (the Visualise ones draw themselves).</summary>
        private static readonly Dictionary<string, string[]> Previewed = new()
        {
            ["SourceComponent"] = new[] { "source" },
            ["ReceiverComponent"] = new[] { "receiver" },
            ["AcousticObjectComponent"] = new[] { "acoustic_object" },
            ["RoomSceneComponent"] = new[] { "scene" },
            ["ScatteringSampleComponent"] = new[] { "sample" },
        };

        [Theory]
        [MemberData(nameof(Components))]
        public void AComponent_IsDrawnLight_AndPreviewsOnlyWhatItShould(string typeName)
        {
            var component = Make(typeName);
            // light, as every other: Grasshopper draws a component that cannot preview in the dark "preview off" grey
            Assert.True(component.IsPreviewCapable);
            var previewed = component.Params.Output.Where(component.PreviewsOutput).Where(p => p is IGH_PreviewObject).Select(p => p.Name);
            Assert.Equal(Previewed.TryGetValue(typeName, out var expected) ? expected : Array.Empty<string>(), previewed);
        }

        [Fact]
        public void DeconstructParameter_IsGone()
        {
            Assert.DoesNotContain("CA_Deconstruct_IR", EmbeddedSpecs.Names());
            Assert.DoesNotContain("DeconstructParameterComponent", Components().Select(c => (string)c[0]));
        }

        [Fact]
        public void CheckChannel_IsGone()
        {
            Assert.DoesNotContain("CA_Channel", EmbeddedSpecs.Names());
            Assert.DoesNotContain("CheckChannelComponent", Components().Select(c => (string)c[0]));
        }

        [Fact]
        public void Components_HaveDistinctGuids()
        {
            var guids = Components().Select(c => Make((string)c[0]).ComponentGuid).ToList();
            Assert.Equal(guids.Count, guids.Distinct().Count());
        }

        [Fact]
        public void TheSimulateComponents_HaveOneOutputEach_NamedAfterTheirFormat()
        {
            Assert.Equal(new[] { "irs" }, new SimulateMonoComponent().Params.Output.Select(p => p.Name));
            Assert.Equal(new[] { "irs" }, new SimulateBinauralComponent().Params.Output.Select(p => p.Name));
            Assert.Equal(new[] { "irs" }, new SimulateAmbisonicsComponent().Params.Output.Select(p => p.Name));
            Assert.All(new ComponentBase[] { new SimulateMonoComponent(), new SimulateBinauralComponent(), new SimulateAmbisonicsComponent() },
                c => Assert.IsType<Param_ImpulseResponse>(c.Params.Output.Single()));
            foreach (var hybrid in new SimulateHybridComponent[] { new SimulateBinauralComponent(), new SimulateAmbisonicsComponent() })
            {
                var reference = hybrid.Params.Input.Single(p => p.Name == "reference");
                Assert.IsType<Param_ImpulseResponse>(reference);
                Assert.Equal(GH_ParamAccess.list, reference.Access);
            }
            Assert.Contains("hrtf", new SimulateBinauralComponent().Params.Input.Select(p => p.Name));
            Assert.DoesNotContain("sh_order", new SimulateBinauralComponent().Params.Input.Select(p => p.Name));
            // the ambisonic order is a slider of Simulate Ambisonics, not an input
            Assert.DoesNotContain("sh_order", new SimulateAmbisonicsComponent().Params.Input.Select(p => p.Name));
            Assert.True(new SimulateAmbisonicsComponent().IsChoice("sh_order"));
            Assert.DoesNotContain("hrtf", new SimulateAmbisonicsComponent().Params.Input.Select(p => p.Name));
        }
    }
}
