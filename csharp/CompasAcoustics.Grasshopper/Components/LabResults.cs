using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using CompasAcoustics.Pb;

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

        private static IEnumerable<double> Numbers(object? list) =>
            list is IEnumerable<object?> items ? items.Select(v => Convert.ToDouble(v, CultureInfo.InvariantCulture)) : Enumerable.Empty<double>();

        /// <summary>The strip of a result: its method and the range of its scattering coefficients, as <c>geometric, s 0.00–0.44</c>.</summary>
        public static string Summary(ScatteringResultData result) =>
            result.SOctaves.Count == 0
                ? result.Method
                : string.Create(CultureInfo.InvariantCulture, $"{result.Method}, s {result.SOctaves.Min():0.00}–{result.SOctaves.Max():0.00}");
    }
}
