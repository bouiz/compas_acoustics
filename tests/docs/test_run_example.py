import sys

import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "docs" / "_tools"))

from run_example import run_example  # noqa: E402

SCRIPT = """
import matplotlib.pyplot as plt

print("hi")
plt.figure()
plt.plot([0, 1])
plt.figure()
plt.plot([1, 0])
plt.show()
plt.figure()
plt.show()
"""


def test_figures_and_stdout_are_saved(tmp_path):
    script = tmp_path / "room.py"
    script.write_text(SCRIPT)
    out = tmp_path / "out"
    out.mkdir()
    (out / "figure_9.png").write_bytes(b"old")
    run_example(script, out)
    assert sorted(p.name for p in out.iterdir()) == ["figure_1.png", "figure_2.png", "figure_3.png", "stdout.txt"]
    assert (out / "stdout.txt").read_text() == "hi\n"
    assert (out / "figure_1.png").read_bytes().startswith(b"\x89PNG")


def test_show_is_restored_afterwards(tmp_path):
    import matplotlib.pyplot as plt

    before = plt.show
    script = tmp_path / "room.py"
    script.write_text("print(1)\n")
    run_example(script, tmp_path / "out")
    assert plt.show is before
