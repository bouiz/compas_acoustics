using System;
using System.Collections.Generic;
using System.Linq;
using Grasshopper.Kernel;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>Every component of the plugin, built outside Rhino, against its spec (the Python component's metadata.json).</summary>
    public class AllComponentsTests
    {
        /// <summary>
        /// The component types this test host can load: the ones with a right-click menu need System.Windows.Forms, which
        /// Rhino provides and the test host does not have (<see cref="MenuComponents"/>).
        /// </summary>
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

        public static readonly string[] MenuComponents = { };

        public static IEnumerable<object[]> Components() =>
            Loadable().Where(t => t.IsSubclassOf(typeof(ComponentBase)) && !t.IsAbstract).Select(t => new object[] { t.Name });

        private static ComponentBase Make(string typeName) =>
            (ComponentBase)Activator.CreateInstance(Loadable().Single(t => t.Name == typeName))!;

        [Theory]
        [MemberData(nameof(Components))]
        public void AComponent_HasThePortsNamesAndPlaceOfItsSpec(string typeName)
        {
            var component = Make(typeName);
            var key = ComponentSpec.Names().Single(name => ComponentSpec.Load(name).Name == component.Name);
            var spec = ComponentSpec.Load(key);
            Assert.Equal(spec.Nickname, component.NickName);
            Assert.Equal(spec.Description, component.Description);
            Assert.Equal(ComponentBase.CategoryName, component.Category);
            Assert.Equal(spec.Subcategory, component.SubCategory);
            Assert.Equal(spec.Inputs.Select(p => p.Name), component.Params.Input.Select(p => p.Name));
            Assert.Equal(spec.Inputs.Select(p => p.Access), component.Params.Input.Select(p => p.Access));
            Assert.Equal(spec.Inputs.Select(p => p.Optional), component.Params.Input.Select(p => p.Optional));
            Assert.Equal(spec.Outputs.Select(p => p.Name), component.Params.Output.Select(p => p.Name));
            Assert.NotEqual(Guid.Empty, component.ComponentGuid);
        }

        [Fact]
        public void Components_HaveDistinctGuids()
        {
            var guids = Components().Select(c => Make((string)c[0]).ComponentGuid).ToList();
            Assert.Equal(guids.Count, guids.Distinct().Count());
        }
    }
}
