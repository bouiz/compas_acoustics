from ghcomponents.examples import components_in
from ghcomponents.examples import discover
from ghcomponents.examples import docstring
from ghcomponents.examples import walkthrough
from ghcomponents.render import examples_index
from ghcomponents.render import ghx_example_page
from ghcomponents.render import python_example_page


def by_name(fixtures):
    return {e.name: e for e in discover(fixtures / "examples")}


def test_examples_pair_by_name_and_either_side_may_be_missing(fixtures):
    assert [(e.name, bool(e.py), bool(e.ghx)) for e in discover(fixtures / "examples")] == [("both", True, True), ("ghx_only", False, True), ("py_only", True, False)]


def test_docstring_gives_title_and_body(fixtures):
    assert docstring(fixtures / "examples/both.py") == ("A room on both sides", "The Python twin of both.ghx.\nIt prints one line.")


def test_docstring_is_none_without_one(tmp_path):
    (tmp_path / "x.py").write_text("print(1)\n")
    assert docstring(tmp_path / "x.py") is None


def test_walkthrough_gives_title_and_body(fixtures):
    assert walkthrough(fixtures / "walkthroughs/ghx_only.rst") == ("Grasshopper only", "A definition.")


def test_components_in_finds_guids_case_insensitively(fixtures):
    guids = {"CA_Fixture_A": "410e7b44-f239-5e1e-ba74-5884e208b73e", "CA_Other": "ddca21b5-bcc8-5806-85f8-fc0a45ced2e5"}
    assert components_in(fixtures / "examples/both.ghx", guids) == {"CA_Fixture_A"}


def test_python_page_links_its_grasshopper_twin_only_when_there_is_one(fixtures):
    ex = by_name(fixtures)
    assert ":doc:`/grasshopper/examples/both`" in python_example_page(ex["both"], "T", "B", [], None)
    assert "grasshopper/examples" not in python_example_page(ex["py_only"], "T", "B", [], None)


def test_python_page_includes_script_figures_and_stdout(fixtures):
    text = python_example_page(by_name(fixtures)["both"], "A room", "Body *x*.", ["/_images/examples/both/figure_1.png"], "hi\n")
    assert text.startswith(".. _example-py-both:\n\nA room\n======\n") and "Body *x*." in text
    assert ".. literalinclude:: /../examples/both.py" in text
    assert ".. image:: /_images/examples/both/figure_1.png" in text
    assert "Output\n------" in text and "::\n\n    hi\n" in text


def test_python_page_without_output_has_no_output_section(fixtures):
    assert "Output" not in python_example_page(by_name(fixtures)["py_only"], "T", "B", [], "")


def test_ghx_page_has_download_images_walkthrough_and_python_link(fixtures):
    text = ghx_example_page(
        by_name(fixtures)["both"], fixtures / "walkthroughs/both.rst", "_downloads/both.ghx", "/_images/examples/both/canvas.png", "/_images/examples/both/viewport.png"
    )
    assert text.startswith(".. _example-gh-both:\n\nA room on both sides\n====================\n")
    assert ":download:`both.ghx <_downloads/both.ghx>`" in text
    assert text.index("canvas.png") < text.index("viewport.png") < text.index("#. Build the room")
    assert ":doc:`/python/examples/both`" in text


def test_ghx_page_without_twin_has_no_python_link(fixtures):
    text = ghx_example_page(by_name(fixtures)["ghx_only"], fixtures / "walkthroughs/ghx_only.rst", "d", "c", "v")
    assert "python/examples" not in text


def test_examples_index_lists_pages_in_a_toctree():
    text = examples_index("Python examples", [("both", "A room"), ("py_only", "Python only")])
    assert text.startswith("Python examples\n===============\n")
    assert ":doc:`both`" in text and "   both\n   py_only\n" in text
