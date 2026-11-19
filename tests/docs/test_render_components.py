from ghcomponents.render import component_page
from ghcomponents.render import escape
from ghcomponents.render import overview_page


def page(c, **kw):
    args = dict(image="../../_images/components/{}.png".format(c.key), icon="_icons/{}.png".format(c.key), notes=None, used_in=[], python=None)
    args.update(kw)
    return component_page(c, **args)


def test_page_has_header_description_image_and_tables(component):
    text = page(component, used_in=["simple_simulation_shoebox"], python="compas_acoustics.Analyser")
    assert text.startswith("# Fixture A\n") and "Elements · FixA" in text
    assert "Make a fixture. It has two inputs." in text
    assert "](../../_images/components/CA_Fixture_A.png)" in text and "](_icons/CA_Fixture_A.png)" in text
    assert text.count("\n|---|") == 2
    assert "(../examples/simple_simulation_shoebox.md)" in text
    assert "[`compas_acoustics.Analyser`][compas_acoustics.Analyser]" in text
    assert text.index("Inputs") < text.index("Outputs") < text.index("Used in") < text.index("Python equivalent")


def test_escape_makes_markup_literal():
    assert escape("a|b *c* `d` ir_ [e] <f> \\") == r"a\|b \*c\* \`d\` ir\_ \[e\] \<f\> \\"


def test_markup_in_descriptions_renders_literally(component):
    component.description = "a|b *c* `d` ir_"
    component.inputs[0].description = "x|y"
    text = page(component)
    assert r"a\|b \*c\* \`d\` ir\_" in text and r"x\|y" in text


def test_no_ports_and_no_python_equivalent_say_so(components):
    (b,) = [c for c in components if c.key == "CA_Fixture_B"]
    text = page(b)
    assert "|---|" not in text and text.count("None.") == 2
    assert "No Python equivalent (Grasshopper only)." in text
    assert "Used in" not in text


def test_choices_optional_and_default_are_shown(component):
    text = page(component)
    assert "Choices: 0, 1, 2" in text and "*Optional.*" in text and "0.25" in text
    assert "| item |" in text and "| list |" in text and "| int |" in text


def test_notes_are_included_when_given(component, tmp_path):
    notes = tmp_path / "CA_Fixture_A.md"
    notes.write_text("Hello")
    text = page(component, notes=notes)
    assert "\n## Notes\n\nHello\n" in text


def test_overview_follows_the_order_and_skips_empty_subcategories(components):
    text = overview_page(components, ["Model", "Params", "Elements"], lambda c: "_icons/{}.png".format(c.key))
    assert text.index("\n## Model\n") < text.index("\n## Elements\n") and "Params" not in text
    assert '<div class="grid cards" markdown>' in text and "(CA_Fixture_A.md)" in text and "(CA_Fixture_B.md)" in text
    assert "Make a fixture." in text and "It has two inputs." not in text  # first sentence only


def cell(text, line):
    return [ln for ln in text.splitlines() if line in ln]


def test_a_multiline_port_description_stays_in_its_cell(component):
    component.inputs[0].description = "First line.\nSecond line."
    text = page(component)
    assert cell(text, "First line.") and "First line. Second line." in cell(text, "First line.")[0]


def test_an_empty_default_shows_as_an_empty_string(component):
    component.inputs[1].default = ""
    text = page(component)
    assert "``" not in text and '`""`' in text


def test_a_description_that_looks_like_a_list_or_comment_stays_text(component):
    component.inputs[0].description = "- not a list"
    component.outputs[0].description = ".. not a comment"
    text = page(component)
    assert r"\- not a list" in text and ".. not a comment" in text


def test_overview_cards_show_the_icon_at_a_fixed_size(components):
    text = overview_page(components, ["Elements", "Model"], lambda c: "_icons/{}.png".format(c.key))
    assert text.count("{ .ca-component-icon }") == text.count("](_icons/") > 0
