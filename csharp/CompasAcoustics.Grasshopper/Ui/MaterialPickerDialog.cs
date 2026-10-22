using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using CompasAcoustics.Pb;
using Eto.Drawing;
using Eto.Forms;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>
    /// The Material Library's picker: a search box, Custom and the library's categories as a tree, and for the highlighted
    /// material its name, category and origin and a chart of absorption and scattering per band, on which a click or a drag
    /// sets a band. A custom material's name and category are editable. Answers the choice, edited or not.
    /// </summary>
    internal sealed class MaterialPickerDialog : Dialog<MaterialChoice?>
    {
        private const string CustomTag = "\u0001custom";
        private static readonly Color Absorption = Color.FromArgb(60, 60, 60);
        private static readonly Color Scattering = Color.FromArgb(110, 165, 220);

        private readonly IReadOnlyList<(string category, IReadOnlyList<string> names)> _library;
        private readonly Func<string, AcousticMaterialData?> _lookup;
        private readonly Dictionary<string, MaterialChoice> _choices = new();
        private readonly SearchBox _search = new() { PlaceholderText = "Search materials" };
        private readonly TreeGridView _tree = new() { ShowHeader = false };
        private readonly TextBox _name = new();
        private readonly DropDown _category = new();
        private readonly Label _origin = new();
        private readonly RadioButtonList _series = new() { Orientation = Orientation.Horizontal };
        private readonly Drawable _chart = new() { Size = new Size(320, 240) };
        // a box per band for absorption and for scattering: typed values set the chart, the chart's edits show here
        private readonly TextBox[] _absorptionBoxes = Enumerable.Range(0, MaterialChoice.Bands).Select(_ => new TextBox { Width = 40 }).ToArray();
        private readonly TextBox[] _scatteringBoxes = Enumerable.Range(0, MaterialChoice.Bands).Select(_ => new TextBox { Width = 40 }).ToArray();
        private string? _selected;
        private bool _filling;

        public MaterialPickerDialog(IReadOnlyList<(string category, IReadOnlyList<string> names)> library, MaterialChoice? current,
            Func<string, AcousticMaterialData?> lookup)
        {
            _library = library;
            _lookup = lookup;
            _choices[CustomTag] = current is { IsCustom: true } ? current : MaterialChoice.NewCustom();
            if (current is { Library: { } name, Edited: true })
                _choices[name] = current;
            _selected = current == null ? null : current.IsCustom ? CustomTag : current.Library;
            Title = "Choose material";
            Resizable = true;
            ClientSize = new Size(680, 480);

            _tree.Columns.Add(new GridColumn { DataCell = new TextBoxCell(0), Expand = true });
            _tree.SelectedItemChanged += (s, e) =>
            {
                if (!_filling && (_tree.SelectedItem as TreeGridItem)?.Tag is string tag)
                    Highlight(tag);
            };
            _tree.CellDoubleClick += (s, e) =>
            {
                if (Highlighted != null)
                    Close(Highlighted);
            };
            _search.TextChanged += (s, e) => Fill();
            foreach (var category in _library.Select(g => g.category).Append(MaterialChoice.CustomCategory).Distinct())
                _category.Items.Add(category);
            _name.TextChanged += (s, e) =>
            {
                if (!_filling && _selected == CustomTag)
                    _choices[CustomTag] = _choices[CustomTag] with { Name = _name.Text };
            };
            _category.SelectedValueChanged += (s, e) =>
            {
                if (!_filling && _selected == CustomTag && _category.SelectedValue is ListItem item)
                    _choices[CustomTag] = _choices[CustomTag] with { Category = item.Text };
            };
            _series.Items.Add("absorption");
            _series.Items.Add("scattering");
            _series.SelectedIndex = 0;
            _chart.Paint += PaintChart;
            _chart.MouseDown += (s, e) => Edit(e.Location);
            _chart.MouseMove += (s, e) =>
            {
                if (e.Buttons == MouseButtons.Primary)
                    Edit(e.Location);
            };

            for (var i = 0; i < MaterialChoice.Bands; i++)
            {
                var band = i;
                foreach (var (boxes, scattering) in new[] { (_absorptionBoxes, false), (_scatteringBoxes, true) })
                {
                    var box = boxes[band];
                    box.TextChanged += (s, e) => Typed(scattering, band, box.Text);
                    // a value that does not read stays as it was
                    box.LostFocus += (s, e) => ShowValues();
                }
            }

            var ok = new Button { Text = "OK" };
            ok.Click += (s, e) => Close(Highlighted);
            var cancel = new Button { Text = "Cancel" };
            cancel.Click += (s, e) => Close(null);
            DefaultButton = ok;
            AbortButton = cancel;
            var fields = new TableLayout
            {
                Spacing = new Size(6, 4),
                Rows =
                {
                    new TableRow(new Label { Text = "Name" }, new TableCell(_name, true)),
                    new TableRow(new Label { Text = "Category" }, new TableCell(_category, true)),
                    new TableRow(new Label { Text = "Origin" }, new TableCell(_origin, true)),
                },
            };
            var right = new TableLayout
            {
                Spacing = new Size(6, 6),
                Rows =
                {
                    fields,
                    new TableRow(new TableLayout(new TableRow(new Label { Text = "Edit" }, _series, new TableCell(null, true))) { Spacing = new Size(6, 0) }),
                    new TableRow(_chart) { ScaleHeight = true },
                    Coefficients(),
                    new Label { Text = "Click or drag on the chart, or type a value, to set a band.", TextColor = Colors.Gray },
                },
            };
            Content = new TableLayout
            {
                Padding = 10,
                Spacing = new Size(8, 8),
                Rows =
                {
                    _search,
                    new TableRow(new TableCell(_tree, true), right) { ScaleHeight = true },
                    new TableLayout(new TableRow(new TableCell(null, true), ok, cancel)) { Spacing = new Size(8, 0) },
                },
            };
            Fill();
            if (_selected != null)
                Highlight(_selected);
        }

        /// <summary>The band labels, then a row of boxes for absorption and one for scattering.</summary>
        private TableLayout Coefficients()
        {
            TableRow Row(Eto.Forms.Control first, IEnumerable<Eto.Forms.Control> cells)
            {
                var row = new TableRow(new TableCell(first));
                foreach (var cell in cells)
                    row.Cells.Add(new TableCell(cell));
                return row;
            }
            return new TableLayout
            {
                Spacing = new Size(4, 3),
                Rows =
                {
                    Row(new Label(), Enumerable.Range(0, MaterialChoice.Bands).Select(i => new Label { Text = MaterialChart.Band(i), TextAlignment = TextAlignment.Center })),
                    Row(new Label { Text = "absorption", TextColor = Absorption }, _absorptionBoxes),
                    Row(new Label { Text = "scattering", TextColor = Scattering }, _scatteringBoxes),
                },
            };
        }

        /// <summary>A value typed for a band sets it, as a click on the chart does; the box is left as typed.</summary>
        private void Typed(bool scattering, int band, string text)
        {
            if (_filling || Highlighted is not { } choice || MaterialChart.Parse(text) is not double value)
                return;
            var current = scattering ? choice.Scattering : choice.Absorption;
            if (current != null && band < current.Length && current[band] == value)
                return;
            _choices[_selected!] = choice.WithBand(scattering, band, value);
            _origin.Text = _choices[_selected!].Origin.Replace("&", "&&");
            _chart.Invalidate();
            // a first scattering value gives the other bands the default: they show it
            if (scattering && choice.Scattering == null)
                ShowValues(except: _scatteringBoxes[band]);
        }

        /// <summary>The boxes show the highlighted material's coefficients; empty scattering boxes say the default.</summary>
        private void ShowValues(TextBox? except = null)
        {
            var choice = Highlighted;
            _filling = true;
            for (var i = 0; i < MaterialChoice.Bands; i++)
            {
                if (!ReferenceEquals(_absorptionBoxes[i], except))
                    _absorptionBoxes[i].Text = choice != null && i < choice.Absorption.Length ? MaterialChart.Format(choice.Absorption[i]) : "";
                if (!ReferenceEquals(_scatteringBoxes[i], except))
                    _scatteringBoxes[i].Text = choice?.Scattering != null && i < choice.Scattering.Length ? MaterialChart.Format(choice.Scattering[i]) : "";
                _scatteringBoxes[i].PlaceholderText = MaterialChart.Format(MaterialChoice.DefaultScattering);
                _absorptionBoxes[i].Enabled = _scatteringBoxes[i].Enabled = choice != null;
            }
            _filling = false;
        }

        /// <summary>The highlighted material, with what was edited on it.</summary>
        private MaterialChoice? Highlighted => _selected != null && _choices.TryGetValue(_selected, out var choice) ? choice : null;

        private void Fill()
        {
            var query = _search.Text?.Trim() ?? "";
            var items = new TreeGridItemCollection();
            TreeGridItem? current = null;
            if (query.Length == 0 || "custom".Contains(query, StringComparison.OrdinalIgnoreCase))
            {
                var custom = new TreeGridItem { Values = new object[] { "Custom" }, Tag = CustomTag };
                items.Add(custom);
                if (_selected == CustomTag)
                    current = custom;
            }
            foreach (var (category, names) in MaterialSearch.Filter(_library, query))
            {
                var group = new TreeGridItem { Values = new object[] { category }, Expanded = query.Length > 0 || names.Contains(_selected ?? "") };
                foreach (var name in names)
                {
                    var item = new TreeGridItem { Values = new object[] { name }, Tag = name };
                    group.Children.Add(item);
                    if (name == _selected)
                        current = item;
                }
                items.Add(group);
            }
            _filling = true;
            _tree.DataStore = items;
            if (current != null)
                _tree.SelectedItem = current;
            _filling = false;
        }

        private void Highlight(string tag)
        {
            if (!_choices.ContainsKey(tag))
            {
                if (_lookup(tag) is not { } material)
                    return;
                _choices[tag] = MaterialChoice.Of(material);
            }
            _selected = tag;
            var choice = _choices[tag];
            _filling = true;
            _name.Text = choice.Name;
            _name.ReadOnly = !choice.IsCustom;
            if (!_category.Items.Any(i => i.Text == choice.Category))
                _category.Items.Add(choice.Category);
            _category.SelectedIndex = _category.Items.ToList().FindIndex(i => i.Text == choice.Category);
            _category.Enabled = choice.IsCustom;
            // a label would read "&" as a mnemonic and drop it
            _origin.Text = choice.Origin.Replace("&", "&&");
            _filling = false;
            ShowValues();
            _chart.Invalidate();
        }

        private RectangleF Plot => new(32, 22, _chart.Width - 42, _chart.Height - 46);

        private static System.Drawing.RectangleF Gdi(RectangleF r) => new(r.X, r.Y, r.Width, r.Height);

        /// <summary>Set the band under ``location`` to the value there, in the series chosen above the chart.</summary>
        private void Edit(PointF location)
        {
            if (Highlighted is not { } choice)
                return;
            var bands = Math.Max(choice.Absorption.Length, MaterialChoice.Bands);
            var (band, value) = MaterialChart.Hit(Gdi(Plot), bands, new System.Drawing.PointF(location.X, location.Y));
            _choices[_selected!] = choice.WithBand(_series.SelectedIndex == 1, band, value);
            _origin.Text = _choices[_selected!].Origin.Replace("&", "&&");
            ShowValues();
            _chart.Invalidate();
        }

        private void PaintChart(object? sender, PaintEventArgs e)
        {
            var g = e.Graphics;
            var font = SystemFonts.Default(8);
            if (Highlighted is not { } choice)
            {
                g.DrawText(font, Colors.Gray, 4, 4, "Choose a material, or Custom.");
                return;
            }
            var plot = Plot;
            g.FillRectangle(Absorption, plot.Left, 6, 8, 8);
            g.DrawText(font, Absorption, plot.Left + 11, 3, "absorption");
            g.FillRectangle(Scattering, plot.Left + 90, 6, 8, 8);
            g.DrawText(font, Scattering, plot.Left + 101, 3, choice.Scattering == null ? "scattering (0.05 by default)" : "scattering");
            foreach (var tick in MaterialChart.Ticks)
            {
                var y = plot.Bottom - (float)tick * plot.Height;
                g.DrawLine(Colors.LightGrey, plot.Left, y, plot.Right, y);
                g.DrawText(font, Colors.Gray, 2, y - 6, tick.ToString("0.0", CultureInfo.InvariantCulture));
            }
            var (absorption, scattering) = MaterialChart.PairedBars(choice.Absorption, choice.Scattering, Gdi(plot));
            foreach (var r in absorption)
                g.FillRectangle(Absorption, r.X, r.Y, r.Width, r.Height);
            foreach (var r in scattering)
                g.FillRectangle(Scattering, r.X, r.Y, r.Width, r.Height);
            var bands = Math.Max(choice.Absorption.Length, MaterialChoice.Bands);
            var step = plot.Width / bands;
            for (var i = 0; i < bands; i++)
                g.DrawText(font, Colors.Gray, plot.Left + i * step + step / 2 - 8, plot.Bottom + 4, MaterialChart.Band(i));
        }
    }
}
