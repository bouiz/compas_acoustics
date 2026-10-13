import json

import pytest
from compas.data import json_loads

from compas_acoustics import AcousticMaterial
from compas_acoustics import MaterialCategory
from compas_acoustics import MaterialsLibrary


def make_material(**kwargs):
    kwargs.setdefault("name", "rough_concrete (from pyroomacoustics)")
    kwargs.setdefault("absorption_coefficients", 0.1)
    return AcousticMaterial(**kwargs)


def test_category_defaults_to_other():
    assert make_material().category is MaterialCategory.OTHER


@pytest.mark.parametrize("value", [MaterialCategory.STONE_AND_TILES, "Stone & Tiles", "STONE_AND_TILES"])
def test_category_accepts_member_value_or_name(value):
    assert make_material(category=value).category is MaterialCategory.STONE_AND_TILES


def test_unknown_category_raises():
    with pytest.raises(ValueError, match="Unknown material category"):
        make_material(category="Cheese")


def test_names_can_contain_spaces_and_the_slug_is_file_friendly():
    material = make_material(name="  wood_1.6cm (from pyroomacoustics) ")
    assert material.name == "wood_1.6cm (from pyroomacoustics)"
    assert material.slug == "wood_1.6cm_from_pyroomacoustics"


def test_empty_name_raises():
    with pytest.raises(ValueError):
        make_material(name=" ")


def test_category_survives_serialization():
    material = make_material(category=MaterialCategory.CONCRETE)
    data = json.loads(material.to_jsonstring())
    assert data["data"]["category"] == "Concrete"
    assert json_loads(material.to_jsonstring()).category is MaterialCategory.CONCRETE


def test_material_without_category_loads_as_other():
    data = make_material().__data__
    del data["category"]
    assert AcousticMaterial.__from_data__(data).category is MaterialCategory.OTHER


@pytest.fixture
def library(tmp_path):
    for material in (
        make_material(name="concrete", category="Concrete"),
        make_material(name="rough_concrete (from pyroomacoustics)", category="Concrete"),
        make_material(name="carpet", category="Floor Coverings"),
    ):
        material.to_json(str(tmp_path / "{}.json".format(material.slug)))
    return MaterialsLibrary(str(tmp_path))


def test_library_looks_up_materials_by_name(library):
    assert sorted(library.material_names) == ["carpet", "concrete", "rough_concrete (from pyroomacoustics)"]
    assert library.material_from_name("rough_concrete (from pyroomacoustics)").category is MaterialCategory.CONCRETE


def test_library_groups_materials_by_category_in_enum_order(library):
    groups = library.materials_by_category
    assert list(groups) == [MaterialCategory.CONCRETE, MaterialCategory.FLOOR_COVERINGS]
    assert groups[MaterialCategory.CONCRETE] == ["concrete", "rough_concrete (from pyroomacoustics)"]


def test_bundled_library_materials_all_have_a_category():
    library = MaterialsLibrary()
    assert "concrete" in library.material_names
    assert sum(len(names) for names in library.materials_by_category.values()) == len(library.material_names)
