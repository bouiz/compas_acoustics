using System;
using System.Drawing;
using System.IO;
using Grasshopper;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    public sealed class CompasAcousticsInfo : GH_AssemblyInfo
    {
        public override string Name => "COMPAS Acoustics";
        public override string Version => Worker.Version;
        public override string Description => "Room acoustic simulation, analysis and auralisation with compas_acoustics, run in its Python worker.";
        public override Guid Id => new Guid("7f0b6a3e-2f4d-4a51-9a1c-3c8e2d5b9e41");
        public override string AuthorName => "Gramazio Kohler Research, ETH Zurich";
        public override string AuthorContact => "https://github.com/gramaziokohler/compas_acoustics";
    }

    /// <summary>The category's icon and short name; the worker stops with Rhino.</summary>
    public sealed class CompasAcousticsPriority : GH_AssemblyPriority
    {
        public override GH_LoadingInstruction PriorityLoad()
        {
            try
            {
                Instances.ComponentServer.AddCategoryShortName(ComponentBase.CategoryName, "CA");
                Instances.ComponentServer.AddCategorySymbolName(ComponentBase.CategoryName, 'A');
                Instances.ComponentServer.AddCategoryIcon(ComponentBase.CategoryName, new Bitmap(new MemoryStream(ComponentSpec.IconBytes("CA_Model"))));
            }
            catch (Exception)
            {
                // an icon is not worth failing the load for
            }
            Rhino.RhinoApp.Closing += (sender, args) =>
            {
                if (Worker.Created)
                    Worker.Instance.Stop();
            };
            return GH_LoadingInstruction.Proceed;
        }
    }
}
