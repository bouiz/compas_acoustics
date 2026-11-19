using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using GH_IO.Serialization;
using Grasshopper.Kernel.Types;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// A room acoustic parameter per band, or a curve (time or frequency response), as Analyse IR gives it and Plot 2D draws
    /// it. ``XScale`` is ``bands``, ``linear`` (time) or ``log`` (frequency); ``Values`` has one row per channel.
    /// </summary>
    public sealed class ParameterGoo : GH_Goo<ParameterGoo.Content>
    {
        public sealed record Content(string Name, string Title, string XUnit, string YUnit, string XScale, double[] Frequencies,
            double[][] Values, double? Overall, string[] Channels);

        public ParameterGoo() { }

        public ParameterGoo(Content value) { Value = value; }

        /// <summary>A parameter of the worker's ``analysis``.</summary>
        public static ParameterGoo FromDictionary(IDictionary<string, object?> p) =>
            new ParameterGoo(new Content(
                (string)p["name"]!, (string)p["title"]!, (string)p["x_unit"]!, (string)p["y_unit"]!, (string)p["x_scale"]!,
                Numbers(p["frequencies"]),
                ((IEnumerable<object?>)p["values"]!).Select(Numbers).ToArray(),
                p.TryGetValue("overall", out var overall) && overall != null ? Convert.ToDouble(overall, CultureInfo.InvariantCulture) : null,
                p.TryGetValue("channels", out var channels) && channels is IEnumerable<object?> names ? names.Select(n => (string)n!).ToArray() : Array.Empty<string>()));

        internal static double[] Numbers(object? list) =>
            ((IEnumerable<object?>)list!).Select(v => v == null ? double.NaN : Convert.ToDouble(v, CultureInfo.InvariantCulture)).ToArray();

        public override bool IsValid => Value != null;
        public override string TypeName => "AcousticParameter";
        public override string TypeDescription => "A room acoustic parameter per band, or a time or frequency response.";

        /// <summary>What a panel shows: a curve's title and x range, or a parameter's first channel per band and its overall value.</summary>
        public override string ToString()
        {
            if (Value == null)
                return "Null AcousticParameter";
            var f = Value.Frequencies;
            if (f.Length == 0)
                return Value.Title;
            if (Value.XScale != "bands")
                return $"{Value.Title}: {Value.XUnit} {PythonFormat.G(f[0])} to {PythonFormat.G(f[f.Length - 1])}, {f.Length} points";
            var values = Value.Values.Length > 0 ? Value.Values[0] : Array.Empty<double>();
            var bands = string.Join(", ", f.Zip(values, (band, v) => $"{PythonFormat.G(band)} Hz {PythonFormat.G(v, 3)}"));
            var text = $"{Value.Name}, {Value.YUnit}: {bands}";
            if (Value.Overall is double overall)
                text += $"; overall {PythonFormat.G(overall, 3)}";
            return text;
        }

        public override IGH_Goo Duplicate() => new ParameterGoo(Value);

        public override bool Write(GH_IWriter writer)
        {
            if (Value == null)
                return true;
            writer.SetString("name", Value.Name);
            writer.SetString("title", Value.Title);
            writer.SetString("x_unit", Value.XUnit);
            writer.SetString("y_unit", Value.YUnit);
            writer.SetString("x_scale", Value.XScale);
            writer.SetDoubleArray("frequencies", Value.Frequencies);
            writer.SetInt32("rows", Value.Values.Length);
            for (var i = 0; i < Value.Values.Length; i++)
                writer.SetDoubleArray("values", i, Value.Values[i]);
            if (Value.Overall is double overall)
                writer.SetDouble("overall", overall);
            writer.SetString("channels", string.Join("\n", Value.Channels));
            return true;
        }

        public override bool Read(GH_IReader reader)
        {
            if (!reader.ItemExists("name"))
                return true;
            var rows = reader.GetInt32("rows");
            var channels = reader.GetString("channels");
            Value = new Content(reader.GetString("name"), reader.GetString("title"), reader.GetString("x_unit"), reader.GetString("y_unit"),
                reader.GetString("x_scale"), reader.GetDoubleArray("frequencies"),
                Enumerable.Range(0, rows).Select(i => reader.GetDoubleArray("values", i)).ToArray(),
                reader.ItemExists("overall") ? reader.GetDouble("overall") : null,
                channels.Length == 0 ? Array.Empty<string>() : channels.Split('\n'));
            return true;
        }
    }
}
