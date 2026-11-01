using System.Collections.Generic;
using System.Linq;
using System;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>The parameters of the compas_acoustics data types: hidden, the components create them.</summary>
    public abstract class CompasAcousticsParam<T> : GH_Param<T> where T : class, IGH_Goo
    {
        protected CompasAcousticsParam(string name, string nickname, string description)
            : base(name, nickname, description, ComponentBase.CategoryName, ComponentBase.RibbonSubcategory("Params"), GH_ParamAccess.item)
        {
        }

        public override GH_Exposure Exposure => GH_Exposure.hidden;
    }

    /// <summary>
    /// A parameter of model data (sources, receivers, acoustic objects, scenes): it draws them itself, with the component's
    /// preview toggle; Grasshopper's generic parameters, which only draw IGH_PreviewData, pass them through without drawing.
    /// </summary>
    public abstract class OwnPreviewParam<T> : CompasAcousticsParam<T>, IGH_PreviewObject where T : class, IGH_Goo, IOwnPreview
    {
        protected OwnPreviewParam(string name, string nickname, string description) : base(name, nickname, description) { }

        public bool Hidden { get; set; }

        public bool IsPreviewCapable => true;

        private IEnumerable<T> Items => VolatileData.AllData(true).OfType<T>();

        private bool Selected => Attributes?.GetTopLevel?.Selected == true;

        public Rhino.Geometry.BoundingBox ClippingBox
        {
            get
            {
                var box = Rhino.Geometry.BoundingBox.Empty;
                foreach (var item in Items)
                    box.Union(item.ClippingBox);
                return box;
            }
        }

        public void DrawViewportWires(IGH_PreviewArgs args)
        {
            foreach (var item in Items)
                item.DrawWires(args.Display, Selected ? args.WireColour_Selected : args.WireColour, args.DefaultCurveThickness);
        }

        public void DrawViewportMeshes(IGH_PreviewArgs args) { }
    }

    public sealed class Param_Material : CompasAcousticsParam<MaterialGoo>
    {
        public Param_Material() : base("Material", "Mat", "An acoustic material.") { }

        public override Guid ComponentGuid => new Guid("fcff26b1-3776-5d3f-813e-0e62f111eecc");
    }

    public sealed class Param_Source : OwnPreviewParam<SourceGoo>
    {
        public Param_Source() : base("Source", "Src", "A sound source.") { }

        public override Guid ComponentGuid => new Guid("b24c8d47-d588-5fa4-9df0-8eff723964f9");
    }

    public sealed class Param_Receiver : OwnPreviewParam<ReceiverGoo>
    {
        public Param_Receiver() : base("Receiver", "Rcv", "A receiver.") { }

        public override Guid ComponentGuid => new Guid("ebf79d87-17ed-521c-a252-07ea00963afd");
    }

    public sealed class Param_AcousticObject : OwnPreviewParam<AcousticObjectGoo>
    {
        public Param_AcousticObject() : base("Acoustic Object", "AObj", "A surface of the room or an obstacle, with its material.") { }

        public override Guid ComponentGuid => new Guid("b25b9536-0380-5708-bf77-580108d287b2");
    }

    public sealed class Param_Scene : OwnPreviewParam<SceneGoo>
    {
        public Param_Scene() : base("Acoustic Scene", "Scene", "Acoustic objects, sources and receivers.") { }

        public override Guid ComponentGuid => new Guid("bbcdb8dd-b04a-5f03-ad3c-8d849944d784");
    }

    public sealed class Param_ImpulseResponse : CompasAcousticsParam<ImpulseResponseGoo>
    {
        public Param_ImpulseResponse() : base("Impulse Response", "IR", "An impulse response.") { }

        public override Guid ComponentGuid => new Guid("01777b71-1eef-5db4-ab66-df91dac8014b");
    }

    public sealed class Param_IrGraph : CompasAcousticsParam<IrGraphGoo>
    {
        public Param_IrGraph() : base("IR Graph", "IRG", "The impulse responses of a simulation output, per source and receiver.") { }

        public override Guid ComponentGuid => new Guid("c2b639d2-98e8-5278-a7c9-c6c385443532");
    }

    public sealed class Param_Audio : CompasAcousticsParam<AudioGoo>
    {
        public Param_Audio() : base("Audio", "Audio", "A rendered signal.") { }

        public override Guid ComponentGuid => new Guid("4ce3744d-a381-5d83-ac3e-aeb3ecc19a7f");
    }

    /// <summary>Analyse IR's time output: a right-click on it chooses the time scale, dB or linear, which the component holds.</summary>
    public sealed class Param_TimeParameter : CompasAcousticsParam<ParameterGoo>
    {
        public const string Choice = "time_scale";

        public Param_TimeParameter() : base("Time Response", "Time", "The impulse response against time; right-click to choose dB or linear.") { }

        public override Guid ComponentGuid => new Guid("8e2f6b0d-4c1a-5e93-a7d2-3b6f9c10e5a4");

        /// <summary>The component whose choice this output shows.</summary>
        internal ComponentBase? Owner { get; set; }

        public string[] Options => new[] { "db", "linear" };

        private static string Label(string option) => option == "db" ? "Time scale: dB" : "Time scale: linear";

        public override void AppendAdditionalMenuItems(System.Windows.Forms.ToolStripDropDown menu)
        {
            base.AppendAdditionalMenuItems(menu);
            if (Owner is not { } owner)
                return;
            Menu_AppendSeparator(menu);
            foreach (var option in Options)
                Menu_AppendItem(menu, Label(option), (sender, args) => owner.SetChoice(Choice, option), true, owner.Choice(Choice) == option);
        }
    }

    public sealed class Param_Parameter : CompasAcousticsParam<ParameterGoo>
    {
        public Param_Parameter() : base("Acoustic Parameter", "Param", "A room acoustic parameter or a response curve.") { }

        public override Guid ComponentGuid => new Guid("1b48ad63-cbcf-5cdf-96b8-60e53b36b1d6");
    }

    public sealed class Param_RayField : CompasAcousticsParam<RayFieldGoo>
    {
        public Param_RayField() : base("Ray Field", "Rays", "Traced acoustic rays.") { }

        public override Guid ComponentGuid => new Guid("29b7e131-32dd-5e9b-b07e-d0ebc720d710");
    }

    public sealed class Param_Reflections : CompasAcousticsParam<ReflectionsGoo>
    {
        public Param_Reflections() : base("Specular Reflections", "Refl", "Specular reflection paths.") { }

        public override Guid ComponentGuid => new Guid("37788fab-7cfa-5e9a-9d1e-62bc1efeae50");
    }
}
