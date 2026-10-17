import pathlib


def test_soundfile_is_the_maintained_package():
    import soundfile

    assert tuple(int(part) for part in soundfile.__version__.split(".")[:2]) >= (0, 12)


def test_the_upstream_revision_constant_matches_the_pin():
    from compas_acoustics.hybrid import UPSTREAM_REVISION

    assert UPSTREAM_REVISION == pathlib.Path(__file__).with_name("UPSTREAM_REVISION").read_text().strip()
