using System.Threading;
using CompasAcoustics.Pb;
using CompasPb.Data;

[assembly: CompasPbRegistrations(typeof(CompasAcoustics.Grasshopper.CompasPbRegistration))]

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// Makes compas_acoustics' messages known to CompasPb, so that payloads from the worker unpack as typed messages.
    /// CompasPb runs <see cref="Register"/> when it finds the assembly attribute; <see cref="Ensure"/> does it explicitly,
    /// for hosts (Rhino's plugin load context) whose assembly loads CompasPb may not see.
    /// </summary>
    public static class CompasPbRegistration
    {
        private static int _registered;

        public static void Register() => Registry.RegisterAssembly(typeof(SceneData).Assembly);

        public static void Ensure()
        {
            if (Interlocked.Exchange(ref _registered, 1) == 0)
                Register();
        }
    }
}
