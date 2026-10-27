using System.Drawing;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The colour of a reflection order, the same in every display component.</summary>
    public static class OrderColors
    {
        // order 0 (the direct sound) black, then matplotlib's tab10 from order 1, repeating after order 10
        private static readonly int[][] Rgb =
        {
            new[] { 0, 0, 0 }, new[] { 214, 39, 40 }, new[] { 31, 119, 180 }, new[] { 44, 160, 44 }, new[] { 255, 127, 14 },
            new[] { 148, 103, 189 }, new[] { 140, 86, 75 }, new[] { 227, 119, 194 }, new[] { 127, 127, 127 }, new[] { 188, 189, 34 },
            new[] { 23, 190, 207 },
        };

        public static Color Of(int order)
        {
            var rgb = order <= 0 ? Rgb[0] : Rgb[1 + (order - 1) % (Rgb.Length - 1)];
            return Color.FromArgb(rgb[0], rgb[1], rgb[2]);
        }
    }
}
