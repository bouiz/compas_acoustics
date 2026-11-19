using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The material picker's search: names that contain the query, ignoring case, under their categories.</summary>
    public static class MaterialSearch
    {
        public static List<(string category, IReadOnlyList<string> names)> Filter(IEnumerable<(string category, IReadOnlyList<string> names)> library, string? query)
        {
            var q = query?.Trim() ?? "";
            return library
                .Select(g => (g.category, names: (IReadOnlyList<string>)(q.Length == 0 ? g.names : g.names.Where(n => n.Contains(q, StringComparison.OrdinalIgnoreCase)).ToList())))
                .Where(g => g.names.Count > 0)
                .ToList();
        }
    }
}
