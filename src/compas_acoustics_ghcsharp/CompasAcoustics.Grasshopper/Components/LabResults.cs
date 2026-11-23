using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using CompasAcoustics.Pb;
using CompasPb.Data;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>What the lab components show of the worker's answers, from plain data so that tests can drive it.</summary>
    public static class LabResults
    {
        /// <summary>
        /// The answer of <c>run_scattering_lab</c>: the result, the coefficients of the 7 octaves, the simulated octaves, and the
        /// diffusion coefficient of each of them (the result's own, empty when it has none).
        /// </summary>
        public static (ScatteringResultData result, double[] scattering, int[] octaves, double[] diffusion) Read(IDictionary<string, object?> result)
        {
            var data = (ScatteringResultData)result["result"]!;
            var scattering = Numbers(result["scattering_coefficients"]).ToArray();
            var octaves = Numbers(result["octave_bands"]).Select(v => (int)Math.Round(v)).ToArray();
            var diffusion = data.DOctaves == null ? Array.Empty<double>() : data.DOctaves.Values.ToArray();
            return (data, scattering, octaves, diffusion);
        }

        /// <summary>
        /// The notes of a worker answer (<c>"notes"</c>: what the method is, not a fault): its distinct strings in order; empty when the
        /// key is missing, null or not a list (an older worker has none).
        /// </summary>
        public static string[] Notes(IDictionary<string, object?> result) =>
            result.TryGetValue("notes", out var notes) && notes is IEnumerable<object?> texts ? texts.OfType<string>().Distinct().ToArray() : Array.Empty<string>();

        private static IEnumerable<double> Numbers(object? list) =>
            list is IEnumerable<object?> items ? items.Select(v => Convert.ToDouble(v, CultureInfo.InvariantCulture)) : Enumerable.Empty<double>();

        /// <summary>
        /// The Run button's label: what pressing it does while a wave method waits for it (<paramref name="waiting"/>) — start, or update
        /// outputs that are the last run's (<paramref name="stale"/>) — and plain Run otherwise.
        /// </summary>
        public static string RunLabel(bool waiting, bool stale) => !waiting ? "Run" : stale ? "Press Run to update" : "Press Run to start";

        /// <summary>
        /// The starts of the sample's warnings that state a fact and flag no fault (the ISO 17497-1 depth limit, a proxy reference):
        /// Scattering Sample shows them as Remarks. They are the starts of the texts of <c>compas_acoustics.lab.sample._warnings</c> and
        /// <c>_PROXY_WARNING</c>, which tests/lab/test_sample.py pins.
        /// </summary>
        public static readonly string[] SampleNoteStarts = { "deeper than ISO 17497-1 allows", "with a 'reference' proxy" };

        /// <summary>The sample's warnings as <c>(warnings, notes)</c>: those that start as one of <see cref="SampleNoteStarts"/> are the notes.</summary>
        public static (string[] warnings, string[] notes) SampleMessages(IEnumerable<string> messages)
        {
            var all = messages.Distinct().ToArray();
            bool IsNote(string m) => SampleNoteStarts.Any(s => m.StartsWith(s, StringComparison.Ordinal));
            return (all.Where(m => !IsNote(m)).ToArray(), all.Where(IsNote).ToArray());
        }

        /// <summary>
        /// What the Lab warns when its outputs are not those of its inputs: the outputs are the last run's (<paramref name="delivered"/>),
        /// or a run was stopped because the inputs changed (<paramref name="interrupted"/>). Null when nothing is stale.
        /// </summary>
        public static string? StaleNotice(bool delivered, bool interrupted) =>
            interrupted ? "Inputs changed during the run, so it stopped. Press Run to restart."
            : delivered ? "Inputs changed: the outputs are the last run's. Press Run to update."
            : null;

        /// <summary>The strip of a result: its method and the range of its scattering coefficients, as <c>geometric, s 0.00–0.44</c>.</summary>
        public static string Summary(ScatteringResultData result) =>
            result.SOctaves.Count == 0
                ? result.Method
                : string.Create(CultureInfo.InvariantCulture, $"{result.Method}, s {result.SOctaves.Min():0.00}–{result.SOctaves.Max():0.00}");

        /// <summary>The level of each face of a balloon: the mean of the levels of its vertices.</summary>
        public static double[] FaceLevels(double[] levels, int[][] faces) => faces.Select(face => face.Average(v => levels[v])).ToArray();

        /// <summary>The answer of <c>polar_balloon</c>: the balloon, the level of each vertex, the vector towards the source and the label.</summary>
        public static (MeshData mesh, double[] levels, double[] source, string label) ReadBalloon(IDictionary<string, object?> answer) =>
            ((MeshData)answer["mesh"]!, Numbers(answer["levels"]).ToArray(), Numbers(answer["source"]).ToArray(), Convert.ToString(answer["label"], CultureInfo.InvariantCulture) ?? "");
    }
}
