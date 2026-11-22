using System;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The plugin's version, and the compas_acoustics versions it works with.</summary>
    public static class Versions
    {
        /// <summary>The plugin's version.</summary>
        public static string Plugin => typeof(Versions).Assembly.GetName().Version!.ToString(3);

        /// <summary>The oldest compas_acoustics the plugin works with (CompasAcousticsVersion in the .csproj).</summary>
        public static string Library =>
            typeof(Versions).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>().Single(a => a.Key == "CompasAcousticsVersion").Value!;

        /// <summary>The compas_acoustics versions the plugin works with, as a pip specifier: e.g. <c>&gt;=0.1.0,&lt;0.2.0</c>.</summary>
        public static string Requirement => $">={Library},<{Below(Parse(Library)!)}";

        /// <summary>Whether the plugin works with this compas_acoustics: from <see cref="Library"/> up to its next breaking release.</summary>
        public static bool Compatible(string? version) => Compatible(version, Library);

        internal static bool Compatible(string? version, string minimum)
        {
            var v = Parse(version);
            var min = Parse(minimum)!;
            return v != null && v >= min && v < Below(min);
        }

        // semver: in 0.x a minor release may break, after that a major one
        private static Version Below(Version min) => min.Major == 0 ? new Version(0, min.Minor + 1, 0) : new Version(min.Major + 1, 0, 0);

        // a pre-release (0.2.0rc1) counts as its release; pip installs none unless asked to
        private static Version? Parse(string? version) =>
            Regex.Match(version ?? "", @"^(\d+)\.(\d+)\.(\d+)") is { Success: true } m
                ? new Version(int.Parse(m.Groups[1].Value), int.Parse(m.Groups[2].Value), int.Parse(m.Groups[3].Value))
                : null;
    }
}
