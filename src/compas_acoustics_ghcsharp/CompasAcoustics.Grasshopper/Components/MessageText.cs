using System.Globalization;
using System.Linq;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The texts of the message strips under the components.</summary>
    public static class MessageText
    {
        public const int Max = 40;

        /// <summary>The strip's text, each line at most <see cref="Max"/> characters, and the full text for the tooltip when it is cut or given.</summary>
        public static (string shown, string? full) Status(string text, string? full = null)
        {
            var lines = text.Split('\n');
            if (lines.All(l => l.Length <= Max))
                return (text, full);
            return (string.Join("\n", lines.Select(l => l.Length <= Max ? l : l.Substring(0, Max - 3) + "...")), full ?? text);
        }

        private static string Count(int n, string noun) => $"{n} {noun}{(n == 1 ? "" : "s")}";

        /// <summary>The Acoustic Object's strip: its surface.</summary>
        public static string Object(double area) => string.Format(CultureInfo.InvariantCulture, "Total area: {0:0.#} m2", area);

        /// <summary>The Scene's strip: the room's volume and total surface; the tooltip adds what it holds.</summary>
        public static (string shown, string full) Scene(double volume, double area, int objects, int sources, int receivers, int rate)
        {
            // a line each: Grasshopper's strip draws a line break
            var shown = string.Format(CultureInfo.InvariantCulture, "Volume: {0:0} m3\nTotal area: {1:0} m2", volume, area);
            var full = string.Format(CultureInfo.InvariantCulture, "{0}\n{1}, {2}, {3}, {4:0.#} kHz",
                shown, Count(objects, "object"), Count(sources, "source"), Count(receivers, "receiver"), rate / 1000.0);
            return (shown, full);
        }

        public static string WorkerLine(WorkerState state, double? runningSeconds) => state switch
        {
            WorkerState.NotConfigured => "not configured",
            WorkerState.Starting => "starting",
            WorkerState.Running => $"running {(int)((runningSeconds ?? 0) / 60)} min",
            _ => "stopped",
        };
    }
}
