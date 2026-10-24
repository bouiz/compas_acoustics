using System.Collections.Generic;
using System.Globalization;
using System.Linq;
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
    }
}
