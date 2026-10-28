using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>Plot 2D's body: its charts stacked, then a row with the Save button.</summary>
    public static class Plot2DLayout
    {
        public const float SaveRow = 22;
        public const float SaveWidth = 44;

        public static float BodyHeight(int charts, SizeF chart) => Math.Max(charts, 1) * chart.Height + SaveRow;

        /// <summary>A chart resized to ``width``, at least ``minWidth``, keeping its aspect ratio.</summary>
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
