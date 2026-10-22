using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using CompasAcoustics.Pb;
using GH_IO.Serialization;
using Grasshopper.Kernel;

namespace CompasAcoustics.Grasshopper
{
    public sealed class MaterialLibraryComponent : ComponentBase, IMaterialBody
    {
        private const string ChoiceKey = "material_choice";
        private static List<(string category, IReadOnlyList<string> names)>? _library;
        private static readonly Dictionary<string, AcousticMaterialData> Materials = new();
        private MaterialChoice? _choice;
        private AcousticMaterialData? _shown;

        public MaterialLibraryComponent() : base("CA_Material_Library") { }
        protected override string SpecKey => "CA_Material_Library";
        public override Guid ComponentGuid => new Guid("cc1cdaf6-43f5-5ac1-bf23-f5a93a6c6cfc");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => input ? null : new Param_Material();

        public override void CreateAttributes() => m_attributes = new MaterialUi(this);

        public AcousticMaterialData? Shown => _choice == null ? null : _shown;

        public string Caption => _choice?.Name ?? "click to choose a material";

        public Action? Pick => OpenPicker;

        /// <summary>The library's names by category, read from the worker once per session.</summary>
        private static List<(string category, IReadOnlyList<string> names)> Library()
        {
            if (_library == null)
            {
                var listed = (IEnumerable<object?>)Worker.Call("materials_library")!;
                _library = listed.Cast<IList<object?>>()
                    .Select(group => ((string)group[0]!, (IReadOnlyList<string>)((IEnumerable<object?>)group[1]!).Cast<string>().ToList()))
                    .ToList();
            }
            return _library;
        }

        /// <summary>A material of the library, read from the worker once per session.</summary>
        private static AcousticMaterialData Material(string name)
        {
            lock (Materials)
                if (Materials.TryGetValue(name, out var known))
                    return known;
            var material = (AcousticMaterialData)Worker.Call("material_from_name", name)!;
            lock (Materials)
                Materials[name] = material;
            return material;
        }

        protected override void Solve(IGH_DataAccess access)
        {
            _shown = null;
            if (_choice is not { } choice)
            {
                Remark("Click the component's strip, or double-click the component, to choose a material.");
                Message = null;
                return;
            }
            var absorption = List<double>(access, "absorption");
            var scattering = List<double>(access, "scattering");
            AcousticMaterialData material;
            if (choice.AsInTheLibrary && absorption.Count == 0 && scattering.Count == 0)
                material = Material(choice.Library!);
            else
            {
                // a library material known by its name only gets its coefficients first
                var known = choice.AsInTheLibrary && choice.Absorption.Length == 0 ? MaterialChoice.Of(Material(choice.Library!)) : choice;
                var scatter = scattering.Count > 0 ? scattering : known.Scattering?.ToList();
                material = (AcousticMaterialData)Worker.Call("make_material", known.Name,
                    (absorption.Count > 0 ? absorption : known.Absorption.ToList()).Cast<object?>().ToList(),
                    scatter?.Cast<object?>().ToList(), string.IsNullOrEmpty(known.Category) ? null : known.Category)!;
            }
            _shown = material;
            var (message, warning) = MaterialChoice.Notes(choice, absorption.Count > 0, material);
            if (message != null)
                Status(message, "This material has no scattering coefficients: simulations use 0.05 in every band. Set them in the picker or connect 'scattering'.");
            else
                Message = null;
            if (warning != null)
                Warn(warning);
            access.SetData(0, new MaterialGoo(material));
        }

        protected override void AppendMenu(ToolStripDropDown menu)
        {
            Menu_AppendItem(menu, "Choose material...", (sender, args) => OpenPicker());
            Menu_AppendSeparator(menu);
        }

        private void OpenPicker()
        {
            List<(string category, IReadOnlyList<string> names)> library;
            try
            {
                library = Library();
            }
            catch (WorkerException e)
            {
                Rhino.UI.Dialogs.ShowMessage(ErrorText(e), "COMPAS Acoustics");
                return;
            }
            AcousticMaterialData? Lookup(string name)
            {
                try
                {
                    return Material(name);
                }
                catch (WorkerException)
                {
                    return null;
                }
            }
            var chosen = new MaterialPickerDialog(library, _choice, Lookup).ShowModal(Rhino.UI.RhinoEtoApp.MainWindow);
            if (chosen == null || chosen == _choice)
                return;
            RecordUndoEvent("Material");
            _choice = chosen;
            NickName = chosen.Name;
            ExpireSolution(true);
        }

        public override bool Write(GH_IWriter writer)
        {
            if (_choice != null)
                writer.SetString(ChoiceKey, _choice.Write());
            return base.Write(writer);
        }

        public override bool Read(GH_IReader reader)
        {
            _choice = reader.ItemExists(ChoiceKey) ? MaterialChoice.Read(reader.GetString(ChoiceKey))
                // saved before the picker edited coefficients: the library material's name
                : reader.ItemExists("material") ? MaterialChoice.Named(reader.GetString("material")) : null;
            return base.Read(reader);
        }
    }

    public sealed class ConstructMaterialComponent : ComponentBase, IMaterialBody
    {
        private AcousticMaterialData? _shown;

        public ConstructMaterialComponent() : base("CA_Construct_Material") { }

        public override void CreateAttributes() => m_attributes = new MaterialUi(this);

        public AcousticMaterialData? Shown => _shown;

        public string Caption => _shown?.Name ?? "no material";

        public Action? Pick => null;
        protected override string SpecKey => "CA_Construct_Material";
        public override Guid ComponentGuid => new Guid("85a391f4-dd65-55e0-be79-d14c979e945b");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => input ? null : new Param_Material();

        protected override void Solve(IGH_DataAccess access)
        {
            _shown = null;
            var absorption = List<double>(access, "absorption");
            if (absorption.Count == 0)
            {
                Warn("Input parameter 'absorption' failed to collect data.");
                return;
            }
            var name = Text(access, "name");
            if (name == null)
            {
                Warn("Input parameter 'name' failed to collect data.");
                return;
            }
            var scattering = List<double>(access, "scattering");
            var material = Worker.Call("make_material", name, absorption.Cast<object?>().ToList(),
                scattering.Count == 0 ? null : scattering.Cast<object?>().ToList(), Text(access, "category"));
            _shown = (AcousticMaterialData)material!;
            access.SetData(0, new MaterialGoo(_shown));
        }
    }

    public sealed class DeconstructMaterialComponent : ComponentBase
    {
        public DeconstructMaterialComponent() : base("CA_Deconstruct_Material") { }
        protected override string SpecKey => "CA_Deconstruct_Material";
        public override Guid ComponentGuid => new Guid("138da7b6-85e2-5d80-8372-6f9ab1d370f5");

        protected override IGH_Param? CreateParam(PortSpec port, bool input) => input ? new Param_Material() : null;

        protected override void Solve(IGH_DataAccess access)
        {
            var goo = Item<MaterialGoo>(access, "material");
            if (goo?.Value == null)
            {
                Warn("Input parameter 'material' failed to collect data.");
                return;
            }
            var material = goo.Value;
            access.SetDataList("absorption", material.AbsorptionCoefficients);
            access.SetDataList("scattering", material.ScatteringCoefficients);
            access.SetData("name", material.Name);
            access.SetData("category", material.Category);
            access.SetData("data_origin", material.DataOrigin);
        }
    }
}
