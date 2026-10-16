import pathlib

ROOT = pathlib.Path(__file__).parents[1]


def test_pyroomacoustics_is_pinned_below_0_11():
    """The parallel ray tracer writes private Room state (rt_histograms, simulator_state)."""
    requirements = (ROOT / "requirements.txt").read_text()
    assert "pyroomacoustics >=0.10.1, <0.11" in requirements
