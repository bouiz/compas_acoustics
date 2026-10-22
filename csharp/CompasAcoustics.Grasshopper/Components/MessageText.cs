using System.Globalization;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The texts of the message strips under the components.</summary>
    public static class MessageText
    {
        public const int Max = 40;

        /// <summary>The strip's text, at most <see cref="Max"/> characters, and the full text for the tooltip when it is cut or given.</summary>
        public static (string shown, string? full) Status(string text, string? full = null) =>
            text.Length <= Max ? (text, full) : (text.Substring(0, Max - 3) + "...", full ?? text);

        private static string Count(int n, string noun) => $"{n} {noun}{(n == 1 ? "" : "s")}";

        public static string Scene(int objects, int sources, int receivers, int rate, double volume) =>
            string.Format(CultureInfo.InvariantCulture, "{0}, {1}, {2}, {3:0.#} kHz, V {4:0} m3",
                Count(objects, "object"), Count(sources, "source"), Count(receivers, "receiver"), rate / 1000.0, volume);

        public static string WorkerLine(WorkerState state, double? runningSeconds) => state switch
        {
            WorkerState.NotConfigured => "not configured",
            WorkerState.Starting => "starting",
            WorkerState.Running => $"running {(int)((runningSeconds ?? 0) / 60)} min",
            _ => "stopped",
        };
    }
}
