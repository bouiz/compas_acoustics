using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>Plot 2D's body: the parameter's dropdown right of the data input, at the start of a column as wide as the charts; below, its charts stacked, then a row with the Save button.</summary>
    public static class Plot2DLayout
    {
        public const float SaveRow = 22;
        public const float SaveWidth = 44;
        public const float SelectWidth = 120;
        public const float MinWidth = 160;

        /// <summary>The width of the dropdown's column right of Grasshopper's box <paramref name="box"/>: to the charts' right edge, at least <see cref="SelectWidth"/>.</summary>
        public static float SelectColumn(RectangleF box, SizeF chart) => Math.Max(SelectWidth, chart.Width - box.Width);

        /// <summary>Where the dropdown and its title go in the column's area <paramref name="area"/>: <see cref="SelectWidth"/> wide at its start, the rest empty.</summary>
        public static RectangleF SelectArea(RectangleF area) => new(area.X, area.Y, Math.Min(area.Width, SelectWidth), area.Height);

        /// <summary>Where the description of the chosen parameter goes in the column's area <paramref name="area"/>: right of <see cref="SelectArea"/>, the rest of the column; empty when the column has no room beyond the dropdown.</summary>
        public static RectangleF DescriptionArea(RectangleF area)
        {
            var left = area.X + Math.Min(area.Width, SelectWidth);
            return new RectangleF(left, area.Y + 3, Math.Max(area.Right - left - 3, 0), Math.Max(area.Height - 6, 0));
        }

        /// <summary>The narrowest the charts are drawn: wide enough for Grasshopper's box <paramref name="box"/> and the dropdown's column right of it, and not below <see cref="MinWidth"/>.</summary>
        public static float MinChartWidth(RectangleF box) => Math.Max(MinWidth, box.Width + SelectWidth);

        public static float BodyHeight(int charts, SizeF chart) => Math.Max(charts, 1) * chart.Height + SaveRow;

        /// <summary>A chart resized to <paramref name="width"/>, at least <paramref name="minWidth"/>, keeping its aspect ratio.</summary>
        public static SizeF Resize(SizeF current, float width, float minWidth)
        {
            var w = Math.Max(width, minWidth);
            return new SizeF(w, w * current.Height / Math.Max(current.Width, 1));
        }

        public static RectangleF Slot(RectangleF body, int i, SizeF chart) =>
            new(body.Left + 3, body.Top + i * chart.Height + 2, body.Width - 6, chart.Height - 4);

        public static RectangleF SaveButton(RectangleF body) => new(body.Left + 4, body.Bottom - SaveRow + 3, SaveWidth, SaveRow - 6);

        /// <summary>The file a saved plot gets: its title without the characters a file name cannot hold.</summary>
        public static string FileName(string title)
        {
            var invalid = Path.GetInvalidFileNameChars().Concat(new[] { ':', '/', '\\', '*', '?', '"', '<', '>', '|' }).ToHashSet();
            var name = new string(title.Select(c => invalid.Contains(c) ? ' ' : c).ToArray());
            name = Regex.Replace(name, @"\s+", " ").Trim();
            return (name.Length == 0 ? "plot" : name) + ".png";
        }
    }
}
