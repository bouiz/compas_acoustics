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

        /// <summary>The strip of a wave-method Lab whose inputs changed since its last run, whose outputs are still the last run's.</summary>
        public const string StaleStatus = "inputs changed, press Run";

        /// <summary>
        /// What the Lab warns when its outputs are not those of its inputs: the outputs are the last run's (<paramref name="delivered"/>),
        /// or a run was stopped because the inputs changed (<paramref name="interrupted"/>). Null when nothing is stale.
        /// </summary>
        public static string? StaleNotice(bool delivered, bool interrupted) =>
            interrupted ? "The inputs changed while a run was going: it was stopped; press Run to start it again."
            : delivered ? "The inputs differ from the last run: the outputs are the last run's; press Run to update."
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
