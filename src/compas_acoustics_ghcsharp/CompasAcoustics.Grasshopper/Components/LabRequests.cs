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
        /// The mesh of a Rhino geometry: a mesh as is; a Brep, surface, SubD, box or extrusion meshed with
        /// <see cref="MeshingParameters.QualityRenderMesh"/> and joined; null for anything else (and for a geometry that meshes to nothing).
        /// </summary>
        public static MeshData? MeshOf(object? geometry)
        {
            var mesh = geometry switch
            {
                Mesh m => m,
                Brep brep => Meshed(brep),
                Extrusion extrusion => Meshed(extrusion.ToBrep()),
                Surface surface => Meshed(surface.ToBrep()),
                SubD subd => Meshed(subd.ToBrep()),
                Box box => Meshed(box.ToBrep()),
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
        /// The call of <c>make_scattering_sample</c>: the mesh and the mode as arguments, every input that is set as a keyword
        /// (<paramref name="frame"/>, the mounting plane, as <c>frame</c>);
        /// <c>problem</c> says why the inputs cannot make a sample, and then there is no call to make; <c>unused</c> names a given input the
        /// mode does not use (a disc's <c>reference</c>), which is left out of the call.
        /// </summary>
        public static (object?[] args, Dictionary<string, object?> kwargs, string? problem, string? unused) Sample(MeshData mesh, string mode, double? diameter,
            MeshData? reference, double? baseplate, string? symmetry, FrameData? frame)
        {
            var kwargs = new Dictionary<string, object?>();
            if (diameter is double d)
                kwargs["diameter"] = d;
            var disc = mode == "disc";
            if (reference != null && !disc)
                kwargs["reference"] = reference;
            if (baseplate is double b)
                kwargs["baseplate"] = b;
            if (!string.IsNullOrEmpty(symmetry))
                kwargs["symmetry"] = symmetry;
            if (frame != null)
                kwargs["frame"] = frame;
            var problem = disc && diameter == null ? "Disc mode needs a 'diameter'." : null;
            var unused = disc && reference != null ? "Disc mode does not use 'reference'." : null;
            return (new object?[] { mesh, mode }, kwargs, problem, unused);
        }

        /// <summary>The note that <paramref name="method"/> does not use the inputs <paramref name="names"/>: <c>kirchhoff does not use 'n_rays' or 'max_bounces'.</c></summary>
        public static string? Unused(string method, IReadOnlyList<string> names) => names.Count switch
        {
            0 => null,
            1 => $"{method} does not use '{names[0]}'.",
            _ => $"{method} does not use {string.Join(", ", names.Take(names.Count - 1).Select(n => $"'{n}'"))} or '{names[^1]}'.",
        };

        /// <summary>The strip of a sample: its mode and diameter, as the lab measures it.</summary>
        public static string SampleStatus(string mode, double diameter) => string.Create(CultureInfo.InvariantCulture, $"{mode}, d {diameter:0.00} m");

        /// <summary>Whether a method runs when its inputs change: only the geometric method (seconds); kirchhoff and bem (minutes to hours) wait for Run.</summary>
        public static bool StartsWithoutRun(string method) => method == "geometric";

        /// <summary>
        /// The call of <c>run_scattering_lab</c>: the sample and the method as arguments; the preset (if set), the band range and
        /// the method's settings (<c>n_rays</c> and <c>max_bounces</c> of the geometric method, <c>cache_dir</c> of bem) as keywords.
        /// A setting of another method is left out of the call, and <c>unused</c> names it.
        /// </summary>
        public static (object?[] args, Dictionary<string, object?> kwargs, string? unused) Lab(ScatteringSampleData sample, string method, string? preset,
            string bandRange, int? nRays, int? maxBounces, string? cacheDir)
        {
            var kwargs = new Dictionary<string, object?>();
            if (!string.IsNullOrEmpty(preset))
                kwargs["preset"] = preset;
            kwargs["band_range"] = bandRange;
            var settings = new Dictionary<string, object?>();
            var unused = new List<string>();
            void Set(string name, object? value, string owner)
            {
                if (value == null)
                    return;
                if (method == owner)
                    settings[name] = value;
                else
                    unused.Add(name);
            }
            Set("n_rays", nRays is int rays ? (long)rays : null, "geometric");
            Set("max_bounces", maxBounces is int bounces ? (long)bounces : null, "geometric");
            Set("cache_dir", string.IsNullOrEmpty(cacheDir) ? null : cacheDir, "bem");
            if (settings.Count > 0)
                kwargs["settings"] = settings;
            return (new object?[] { sample, method }, kwargs, Unused(method, unused));
        }

        /// <summary>
        /// The key of a <c>run_scattering_lab</c> call: a hash of the call with the sample's guid cleared, because every solution of
        /// Scattering Sample makes a sample with a new guid and the same geometry must keep its run (or its result).
        /// </summary>
        public static string RunKey(object?[] args, IDictionary<string, object?> kwargs)
        {
            var keyed = (object?[])args.Clone();
            if (keyed.Length > 0 && keyed[0] is ScatteringSampleData sample)
            {
                var clone = sample.Clone();
                clone.Guid = "";
                keyed[0] = clone;
            }
            return AsyncRuns.Key("run_scattering_lab", keyed, kwargs);
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
