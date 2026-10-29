using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// One port of a component, as its metadata.json describes it. ``Default`` (an int, double, bool or string, after the
    /// type hint) is the value an unconnected port shows; ``NamedValues`` are the integers its menu offers.
    /// </summary>
    public sealed record PortSpec(string Name, string Description, bool Optional, GH_ParamAccess Access, string TypeHint, bool Flatten,
        object? Default = null, IReadOnlyList<(string name, int value)>? NamedValues = null)
    {
        public IReadOnlyList<(string name, int value)> NamedValues { get; init; } = NamedValues ?? Array.Empty<(string, int)>();
    }

    /// <summary>
    /// A component's names, description, ports and icon: the metadata.json and icon.png the Python component had,
    /// embedded in the assembly (Components/Specs/&lt;name&gt;), so that the texts stay what users know.
    /// </summary>
    public sealed record ComponentSpec(string Key, string Name, string Nickname, string Description, string Subcategory, GH_Exposure Exposure,
        IReadOnlyList<PortSpec> Inputs, IReadOnlyList<PortSpec> Outputs)
    {
        private const string Prefix = "Specs/";

        private static Stream Resource(string path) =>
            typeof(ComponentSpec).Assembly.GetManifestResourceStream(Prefix + path)
            ?? typeof(ComponentSpec).Assembly.GetManifestResourceStream(Prefix + path.Replace('/', '\\'))
            ?? throw new InvalidOperationException($"No embedded component spec {path}.");

        public static IEnumerable<string> Names() =>
            typeof(ComponentSpec).Assembly.GetManifestResourceNames()
                .Where(n => n.StartsWith(Prefix) && n.EndsWith("metadata.json"))
                .Select(n => n.Substring(Prefix.Length).Split('/', '\\')[0])
                .OrderBy(n => n, StringComparer.Ordinal);

        public static byte[] IconBytes(string key)
        {
            using var stream = Resource(key + "/icon.png");
            using var buffer = new MemoryStream();
            stream.CopyTo(buffer);
            return buffer.ToArray();
        }

        public static ComponentSpec Load(string key)
        {
            using var stream = Resource(key + "/metadata.json");
            using var json = JsonDocument.Parse(stream);
            var root = json.RootElement;
            var ports = root.GetProperty("ports");
            return new ComponentSpec(
                key,
                root.GetProperty("name").GetString()!,
                root.GetProperty("nickname").GetString()!,
                root.GetProperty("description").GetString()!,
                root.GetProperty("subcategory").GetString()!,
                root.TryGetProperty("exposure", out var exposure) ? (GH_Exposure)exposure.GetInt32() : GH_Exposure.primary,
                Ports(ports, "inputParameters"),
                Ports(ports, "outputParameters"));
        }

        private static IReadOnlyList<PortSpec> Ports(JsonElement spec, string key)
        {
            if (!spec.TryGetProperty(key, out var ports))
                return Array.Empty<PortSpec>();
            return ports.EnumerateArray().Select(p => new PortSpec(
                p.GetProperty("name").GetString()!,
                p.TryGetProperty("description", out var d) ? d.GetString() ?? "" : "",
                p.TryGetProperty("optional", out var o) && o.GetBoolean(),
                p.TryGetProperty("scriptParamAccess", out var a) ? (GH_ParamAccess)a.GetInt32() : GH_ParamAccess.item,
                p.TryGetProperty("typeHintID", out var t) ? t.GetString() ?? "ghdoc" : "ghdoc",
                p.TryGetProperty("flatten", out var f) && f.GetBoolean(),
                p.TryGetProperty("default", out var v) ? Value(v, p.TryGetProperty("typeHintID", out var h) ? h.GetString() : null) : null,
                p.TryGetProperty("namedValues", out var n) ? n.EnumerateArray().Select(e => (e[0].GetString()!, e[1].GetInt32())).ToList() : null)).ToList();
        }

        private static object? Value(JsonElement value, string? typeHint) => value.ValueKind switch
        {
            JsonValueKind.Number when typeHint == "int" => value.GetInt32(),
            JsonValueKind.Number => value.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.String => value.GetString(),
            _ => null,
        };
    }
}
