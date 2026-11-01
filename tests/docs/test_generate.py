import json
import os
import shutil

import pytest

from ghcomponents import DocsCheckError
from ghcomponents import Paths
from ghcomponents import generate
from ghcomponents.fingerprints import load
from ghcomponents.fingerprints import record_component
from ghcomponents.fingerprints import record_file
from ghcomponents.fingerprints import save
from ghcomponents.specs import load_components


@pytest.fixture
def tree(fixtures, tmp_path):
    for name in ("Specs", "cs", "examples", "walkthroughs"):
        shutil.copytree(fixtures / name, tmp_path / name)
    shutil.copy(fixtures / "ComponentBase.cs", tmp_path / "ComponentBase.cs")
    paths = Paths(
        specs=tmp_path / "Specs",
        component_base_cs=tmp_path / "ComponentBase.cs",
        csharp=tmp_path / "cs",
        examples=tmp_path / "examples",
        docs=tmp_path / "docs",
        images=tmp_path / "docs/_images",
        equivalents=tmp_path / "docs/grasshopper/python_equivalents.json",
        notes=tmp_path / "docs/grasshopper/notes",
        walkthroughs=tmp_path / "walkthroughs",
    )
    paths.equivalents.parent.mkdir(parents=True)
    paths.equivalents.write_text(json.dumps({"CA_Fixture_A": "compas_acoustics.Scene", "CA_Fixture_B": None}))
    data = {"components": {}, "py": {}, "ghx": {}}
    for c in load_components(paths.specs):
        record_component(data, c)
        touch(paths.images / "components" / (c.key + ".png"))
    for name in ("both", "py_only"):
        record_file(data, "py", paths.examples / (name + ".py"))
        touch(paths.images / "examples" / name / "stdout.txt", "hi\n")
    for name in ("both", "ghx_only"):
        record_file(data, "ghx", paths.examples / (name + ".ghx"))
        touch(paths.images / "examples" / name / "canvas.png")
        touch(paths.images / "examples" / name / "viewport.png")
    touch(paths.images / "examples/both/figure_1.png")
    save(data, paths.images / "fingerprints.json")
    return paths


def touch(path, text="png"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_a_complete_fixture_tree_generates_without_errors(tree):
    written = generate(tree)
    docs = tree.docs
    for rel in (
        "grasshopper/components/index.rst",
        "grasshopper/components/CA_Fixture_A.rst",
        "grasshopper/components/_icons/CA_Fixture_A.png",
        "grasshopper/examples/index.rst",
        "grasshopper/examples/both.rst",
        "grasshopper/examples/_downloads/both.ghx",
        "python/examples/index.rst",
        "python/examples/both.rst",
        "python/examples/py_only.rst",
    ):
        assert docs / rel in written and (docs / rel).exists(), rel
    a = (docs / "grasshopper/components/CA_Fixture_A.rst").read_text()
    assert ":doc:`../examples/both`" in a and ":obj:`compas_acoustics.Scene`" in a
    assert ".. image:: /_images/components/CA_Fixture_A.png" in a
    both = (docs / "python/examples/both.rst").read_text()
    assert ".. image:: /_images/examples/both/figure_1.png" in both and "    hi\n" in both


def test_notes_are_included_when_the_file_exists(tree):
    touch(tree.notes / "CA_Fixture_A.rst", "A note.")
    generate(tree)
    assert ".. include:: /grasshopper/notes/CA_Fixture_A.rst" in (tree.docs / "grasshopper/components/CA_Fixture_A.rst").read_text()


def test_every_problem_is_reported_in_one_error(tree):
    (tree.images / "components/CA_Fixture_A.png").unlink()
    meta = tree.specs / "CA_Fixture_B/metadata.json"
    meta.write_text(meta.read_text().replace('"Model"', '"Mystery"'))
    tree.equivalents.write_text(json.dumps({"CA_Fixture_A": "compas_acoustics.Scene"}))
    with pytest.raises(DocsCheckError) as e:
        generate(tree)
    text = str(e.value)
    assert "CA_Fixture_A   no screenshot" in text and "#screenshots" in text
    assert "capture_grasshopper.py" in text
    assert "CA_Fixture_B   Mystery" in text and "#subcategories" in text and "ComponentBase.cs" in text
    assert "CA_Fixture_B" in text.split("#subcategories")[1] and "#python-equivalents" in text


def test_a_changed_script_asks_for_invoke_docs_images(tree):
    with open(tree.examples / "both.py", "a") as f:
        f.write("print(2)\n")
    with pytest.raises(DocsCheckError) as e:
        generate(tree)
    assert "both   file changed" in str(e.value) and "invoke docs-images" in str(e.value)
    assert "#example-outputs" in str(e.value)


def test_a_changed_ghx_asks_for_a_capture(tree):
    with open(tree.examples / "ghx_only.ghx", "a") as f:
        f.write("<!-- -->\n")
    with pytest.raises(DocsCheckError) as e:
        generate(tree)
    assert "ghx_only   file changed" in str(e.value) and "#example-images" in str(e.value)


def test_a_script_without_docstring_and_a_ghx_without_walkthrough_fail_with_a_fix(tree):
    (tree.examples / "py_only.py").write_text("print(1)\n")
    record = load(tree.images / "fingerprints.json")
    record_file(record, "py", tree.examples / "py_only.py")
    save(record, tree.images / "fingerprints.json")
    (tree.walkthroughs / "ghx_only.rst").unlink()
    with pytest.raises(DocsCheckError) as e:
        generate(tree)
    text = str(e.value)
    assert "py_only" in text and "#example-docstrings" in text and "Start the script with a docstring" in text
    assert "ghx_only" in text and "#walkthroughs" in text and "walkthroughs/ghx_only.rst" in text


def test_pages_of_removed_components_are_deleted(tree):
    touch(tree.docs / "grasshopper/components/CA_Gone.rst", "old")
    generate(tree)
    assert not (tree.docs / "grasshopper/components/CA_Gone.rst").exists()


def test_unchanged_pages_are_not_rewritten(tree):
    generate(tree)
    page = tree.docs / "grasshopper/components/CA_Fixture_A.rst"
    os.utime(page, (1, 1))
    generate(tree)
    assert page.stat().st_mtime == 1


def test_check_false_writes_pages_and_raises_nothing(tree):
    (tree.images / "components/CA_Fixture_A.png").unlink()
    generate(tree, check=False)
    assert (tree.docs / "grasshopper/components/CA_Fixture_A.rst").exists()


def test_viewer_captures_are_shown_after_the_figures(tree):
    touch(tree.images / "examples/both/viewer_1.png")
    touch(tree.images / "examples/both/figure_2.png")
    generate(tree)
    page = (tree.docs / "python/examples/both.rst").read_text()
    order = [page.index(name) for name in ("figure_1.png", "figure_2.png", "viewer_1.png")]
    assert order == sorted(order)
