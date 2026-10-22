using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>
    /// Where no Windows Forms is installed (the Mac, Linux), the test host loads WinFormsStub, which has the menu type
    /// Grasshopper's virtual methods take, so that every component loads (none of the tests opens a menu).
    /// </summary>
    internal static class WinFormsResolver
    {
        [ModuleInitializer]
        internal static void Register() =>
            AssemblyLoadContext.Default.Resolving += (context, name) =>
            {
                if (name.Name != "System.Windows.Forms")
                    return null;
                var stub = Path.Combine(Path.GetDirectoryName(typeof(WinFormsResolver).Assembly.Location)!, "winforms-stub", "System.Windows.Forms.dll");
                return File.Exists(stub) ? context.LoadFromAssemblyPath(stub) : null;
            };
    }
}
