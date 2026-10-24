using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using CompasAcoustics.Pb;
using CompasPb.Data;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>What the lab components send to the worker: the arguments of each call, built from plain data so that tests can drive them.</summary>
    public static class LabRequests
    {
        /// <summary>
        /// The mesh of a Rhino geometry: a mesh as is; a Brep, surface or SubD meshed with
        /// <see cref="MeshingParameters.QualityRenderMesh"/> and joined; null for anything else (and for a geometry that meshes to nothing).
        /// </summary>
        public static MeshData? MeshOf(object? geometry)
        {
            var mesh = geometry switch
            {
                Mesh m => m,
                Brep brep => Meshed(brep),
                Surface surface => Meshed(surface.ToBrep()),
                SubD subd => Meshed(subd.ToBrep()),
                _ => null,
            };
            return mesh == null || mesh.Faces.Count == 0 ? null : MeshConversion.ToMeshData(mesh);
        }

        private static Mesh? Meshed(Brep? brep)
        {
            var parts = brep == null ? null : Mesh.CreateFromBrep(brep, MeshingParameters.QualityRenderMesh);
            if (parts == null || parts.Length == 0)
                return null;
            var joined = new Mesh();
            joined.Append(parts);
            return joined;
        }

        /// <summary>
        /// The call of <c>make_scattering_sample</c>: the mesh and the mode as arguments, every input that is set as a keyword;
        /// <c>problem</c> says why the inputs cannot make a sample, and then there is no call to make.
        /// </summary>
        public static (object?[] args, Dictionary<string, object?> kwargs, string? problem) Sample(MeshData mesh, string mode, double? diameter,
            MeshData? reference, double? baseplate, string? symmetry, string? name)
        {
            var kwargs = new Dictionary<string, object?>();
            if (diameter is double d)
                kwargs["diameter"] = d;
            if (reference != null)
                kwargs["reference"] = reference;
            if (baseplate is double b)
                kwargs["baseplate"] = b;
            if (!string.IsNullOrEmpty(symmetry))
                kwargs["symmetry"] = symmetry;
            if (!string.IsNullOrEmpty(name))
                kwargs["name"] = name;
            var problem = mode == "disc" && diameter == null ? "A disc sample needs 'diameter'."
                : mode == "disc" && reference != null ? "'reference' is used in as_drawn mode only."
                : null;
            return (new object?[] { mesh, mode }, kwargs, problem);
        }

        /// <summary>The strip of a sample: its mode and diameter, as the lab measures it.</summary>
        public static string SampleStatus(string mode, double diameter) => string.Create(CultureInfo.InvariantCulture, $"{mode}, d {diameter:0.00} m");

        /// <summary>Whether a method runs when its inputs change: only the geometric method (seconds); kirchhoff and bem (minutes to hours) wait for Run.</summary>
        public static bool StartsWithoutRun(string method) => method == "geometric";

        /// <summary>
        /// The call of <c>run_scattering_lab</c>: the sample and the method as arguments; the preset (if set), the band range and
        /// the method's settings (<c>n_rays</c> of the geometric method, <c>cache_dir</c> of bem) as keywords. <c>problem</c> says
        /// why the inputs cannot make a call: a setting of another method.
        /// </summary>
        public static (object?[] args, Dictionary<string, object?> kwargs, string? problem) Lab(ScatteringSampleData sample, string method, string? preset,
            string bandRange, int? nRays, string? cacheDir)
        {
            var kwargs = new Dictionary<string, object?>();
            if (!string.IsNullOrEmpty(preset))
                kwargs["preset"] = preset;
            kwargs["band_range"] = bandRange;
            var settings = new Dictionary<string, object?>();
            if (nRays is int rays)
                settings["n_rays"] = (long)rays;
            if (!string.IsNullOrEmpty(cacheDir))
                settings["cache_dir"] = cacheDir;
            if (settings.Count > 0)
                kwargs["settings"] = settings;
            var problem = nRays != null && method != "geometric" ? "'n_rays' is used by the geometric method only."
                : !string.IsNullOrEmpty(cacheDir) && method != "bem" ? "'cache_dir' is used by bem only."
                : null;
            return (new object?[] { sample, method }, kwargs, problem);
        }

        /// <summary>The preview sources of a result as its Polar Preview dropdown lists them: <c>θ 30°, φ 0°</c>, the colatitude and azimuth in degrees.</summary>
        public static string[] PreviewLabels(ScatteringResultData result) =>
            result.PolarSourceIndices.Select(i => string.Create(CultureInfo.InvariantCulture, $"θ {result.SourceColatitude[i]:0.##}°, φ {result.SourceAzimuth[i]:0.##}°")).ToArray();

        /// <summary>
        /// The call of <c>polar_balloon</c>: the result as the argument, the preview, octave, body, radius and floor as keywords.
        /// <c>problem</c> says why the inputs cannot make a call: an octave the result did not simulate or a preview it does not have.
        /// </summary>
        public static (object?[] args, Dictionary<string, object?> kwargs, string? problem) Balloon(ScatteringResultData result, int preview, int octave,
            string body, double radius, double floorDb)
        {
            var kwargs = new Dictionary<string, object?>
            {
                ["preview"] = (long)preview,
                ["octave"] = (long)octave,
                ["body"] = body,
                ["radius"] = radius,
                ["floor_db"] = floorDb,
            };
            var previews = result.PolarSourceIndices.Count;
            var problem = previews == 0 ? "The result has no preview sources."
                : preview < 0 || preview >= previews ? $"'preview' must be 0 to {previews - 1}"
                : result.OctaveBands.Count == 0 ? "The result has no simulated octaves."
                : !result.OctaveBands.Contains(octave) ? $"'octave' must be one of {string.Join(", ", result.OctaveBands)} Hz"
                : null;
            return (new object?[] { result }, kwargs, problem);
        }
    }
}
