using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The defaults and named values a spec declares, on the ports.</summary>
    public static class PortDefaults
    {
        public static void Apply(IGH_Param param, PortSpec port)
        {
            switch (param)
            {
                case Param_Integer integer:
                    if (port.Default is int i)
                        integer.SetPersistentData(new GH_Integer(i));
                    foreach (var (name, value) in port.NamedValues)
                        integer.AddNamedValue(name, value);
                    break;
                case Param_Number number when port.Default is double or int:
                    number.SetPersistentData(new GH_Number(Convert.ToDouble(port.Default, CultureInfo.InvariantCulture)));
                    break;
                case Param_Boolean boolean when port.Default is bool b:
                    boolean.SetPersistentData(new GH_Boolean(b));
                    break;
                case Param_String text when port.Default is string s:
                    text.SetPersistentData(new GH_String(s));
                    break;
            }
        }

        /// <summary>Whether <paramref name="value"/> is the declared default of a port nothing is connected to.</summary>
        public static bool Untouched(int sourceCount, object? declared, object value) =>
            sourceCount == 0 && declared != null && Equals(Convert.ChangeType(declared, value.GetType(), CultureInfo.InvariantCulture), value);
    }

    /// <summary>
    /// The <c>source_idx</c> and <c>receiver_idx</c> ports: their dropdowns (All, or one of the connected scene's), and their
    /// named values, the indices with their names (<see cref="ComponentBase.SetIndexOptions"/>).
    /// </summary>
    public static class SceneIndices
    {
        public static readonly DropdownControl SourceDropdown = new("source_idx", new[] { "" }, "", new[] { "All" }, "Source");

        public static readonly DropdownControl ReceiverDropdown = new("receiver_idx", new[] { "" }, "", new[] { "All" }, "Receiver");

        public static List<(string name, int value)> Of(IEnumerable<string> names) => names.Select((name, i) => ($"{i}: {name}", i)).ToList();
    }
}
