from ghcomponents.fingerprints import file_fingerprint
from ghcomponents.fingerprints import load
from ghcomponents.fingerprints import port_fingerprint
from ghcomponents.fingerprints import record_component
from ghcomponents.fingerprints import record_file
from ghcomponents.fingerprints import save
from ghcomponents.fingerprints import stale_components
from ghcomponents.fingerprints import stale_files


def empty():
    return {"components": {}, "py": {}, "ghx": {}}


def image(images, rel):
    path = images / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")


def test_file_fingerprint_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_bytes(b"x = 1\ny = 2\n")
    b.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert file_fingerprint(a) == file_fingerprint(b)
    assert len(file_fingerprint(a)) == 12


def test_port_fingerprint_changes_with_a_port_name(component):
    before = port_fingerprint(component)
    component.outputs[0].name = "other"
    assert port_fingerprint(component) != before


def test_a_component_with_a_matching_fingerprint_and_image_is_fresh(component, tmp_path):
    data = empty()
    record_component(data, component)
    image(tmp_path, "components/CA_Fixture_A.png")
    assert stale_components([component], data, tmp_path) == []


def test_a_missing_image_is_stale_with_reason_no_image(component, tmp_path):
    data = empty()
    record_component(data, component)
    (stale,) = stale_components([component], data, tmp_path)
    assert (stale.kind, stale.name, stale.reason) == ("components", "CA_Fixture_A", "no image")


def test_a_component_never_captured_has_no_image(component, tmp_path):
    image(tmp_path, "components/CA_Fixture_A.png")
    assert stale_components([component], empty(), tmp_path)[0].reason == "no image"


def test_changed_ports_name_what_was_added_and_removed(component, tmp_path):
    data = empty()
    record_component(data, component)
    image(tmp_path, "components/CA_Fixture_A.png")
    component.outputs.append(type(component.outputs[0])(name="frequency", description=""))
    component.inputs.pop()
    (stale,) = stale_components([component], data, tmp_path)
    assert stale.reason == "ports changed (removed input 'weights', added output 'frequency')"


def test_a_reordered_port_is_a_change(component, tmp_path):
    data = empty()
    record_component(data, component)
    image(tmp_path, "components/CA_Fixture_A.png")
    component.inputs.reverse()
    assert stale_components([component], data, tmp_path)[0].reason == "ports changed (order)"


def test_a_changed_example_file_is_stale(tmp_path):
    script = tmp_path / "room.py"
    script.write_text("x = 1\n")
    data = empty()
    record_file(data, "py", script)
    image(tmp_path, "examples/room/stdout.txt")
    assert stale_files("py", [script], data, tmp_path) == []
    script.write_text("x = 2\n")
    (stale,) = stale_files("py", [script], data, tmp_path)
    assert (stale.kind, stale.name, stale.reason) == ("py", "room", "file changed")


def test_a_ghx_needs_both_canvas_and_viewport(tmp_path):
    ghx = tmp_path / "room.ghx"
    ghx.write_text("<Archive/>")
    data = empty()
    record_file(data, "ghx", ghx)
    image(tmp_path, "examples/room/canvas.png")
    assert stale_files("ghx", [ghx], data, tmp_path)[0].reason == "no image"
    image(tmp_path, "examples/room/viewport.png")
    assert stale_files("ghx", [ghx], data, tmp_path) == []


def test_load_of_a_missing_file_gives_empty_sections(tmp_path):
    assert load(tmp_path / "none.json") == empty()


def test_save_and_load_round_trip(component, tmp_path):
    data = empty()
    record_component(data, component)
    save(data, tmp_path / "f.json")
    assert load(tmp_path / "f.json") == data
    assert (tmp_path / "f.json").read_text().endswith("}\n")
