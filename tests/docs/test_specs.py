from ghcomponents.specs import COMPONENT_BASE_CS
from ghcomponents.specs import CSHARP_DIR
from ghcomponents.specs import SPECS_DIR
from ghcomponents.specs import load_components
from ghcomponents.specs import read_component_guids
from ghcomponents.specs import read_subcategory_order


def test_ports_carry_access_defaults_and_choices(fixtures):
    (c,) = [c for c in load_components(fixtures / "Specs") if c.key == "CA_Fixture_A"]
    assert (c.name, c.nickname, c.subcategory) == ("Fixture A", "FixA", "Elements")
    assert [p.access for p in c.inputs] == ["item", "list"]
    assert c.inputs[0].choices == ["0", "1", "2"] and c.inputs[1].default == 0.25
    assert c.inputs[1].optional is True and c.inputs[0].optional is False
    assert c.outputs[0].type_hint is None and c.icon.name == "icon.png"


def test_a_component_without_ports_has_empty_lists(fixtures):
    (c,) = [c for c in load_components(fixtures / "Specs") if c.key == "CA_Fixture_B"]
    assert c.inputs == [] and c.outputs == []


def test_subcategory_order_is_read_from_the_csharp_list(fixtures):
    assert read_subcategory_order(fixtures / "ComponentBase.cs") == ["Elements", "Params", "Model"]


def test_guids_pair_spec_keys_with_component_guids_including_digits(fixtures):
    # a Params class with a GUID and no SpecKey is ignored; GUIDs are lowercased
    assert read_component_guids(fixtures / "cs") == {
        "CA_Fixture_A": "410e7b44-f239-5e1e-ba74-5884e208b73e",
        "CA_Plot_2D": "ddca21b5-bcc8-5806-85f8-fc0a45ced2e5",
    }


def test_the_real_specs_all_have_a_guid_and_a_known_subcategory():
    comps = load_components(SPECS_DIR)
    assert len(comps) == 23
    assert {c.key for c in comps} <= set(read_component_guids(CSHARP_DIR))
    assert {c.subcategory for c in comps} <= set(read_subcategory_order(COMPONENT_BASE_CS))
