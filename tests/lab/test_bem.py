import subprocess
import sys
import time

import numpy as np
import pytest
from lab_shapes import flat_tile
from lab_shapes import sine_tile

from compas_acoustics.lab.bands import EXTENDED_BAND_RANGE
from compas_acoustics.lab.bands import simulation_frequencies
from compas_acoustics.lab.coefficients import scattering_freefield
from compas_acoustics.lab.grids import ReceiverGrid
from compas_acoustics.lab.grids import SourceGrid
from compas_acoustics.lab.sample import ScatteringSample
from compas_acoustics.lab.solvers.bem import BemSolver
from compas_acoustics.lab.solvers.bem import cache_key
from compas_acoustics.lab.solvers.bem import estimate_elements
from compas_acoustics.lab.solvers.bem import plan_splits


def test_splits_group_simulation_frequencies_by_octave():
    f = simulation_frequencies()
    splits = plan_splits(f)
    assert len(splits) == 6 and all(len(s) == 9 for s in splits)  # default: complete octaves 125-4000
    ext = plan_splits(simulation_frequencies(EXTENDED_BAND_RANGE))
    assert len(ext) == 7 and len(ext[-1]) == 6  # extended: an 8 kHz split from 6300 and 8000
    assert np.all(np.diff(np.concatenate(splits)) > 0)


def test_element_estimate_scales_with_area():
    plate = ScatteringSample.disc(flat_tile(), diameter=0.8).reference
    # top + bottom discs + rim wall of the 10 mm baseplate
    assert estimate_elements(plate, 0.01) == pytest.approx((2 * np.pi * 0.16 + np.pi * 0.8 * 0.01) / (np.sqrt(3) / 4 * 1e-4), rel=0.02)


def test_cache_key_changes_when_sample_moves_1_mm():
    a = ScatteringSample.disc(sine_tile(0.0708, 0.0204), diameter=0.4)
    b = ScatteringSample.disc(sine_tile(0.0708, 0.0204), diameter=0.4, offset=(0.001, 0.0))
    args = (SourceGrid.preset("survey"), ReceiverGrid.hemisphere(5.0, 10.0), simulation_frequencies(), 343.0, 0.007)
    assert cache_key(a, *args) == cache_key(ScatteringSample.disc(sine_tile(0.0708, 0.0204), diameter=0.4), *args)
    assert cache_key(a, *args) != cache_key(b, *args)


def test_element_estimate_far_above_limit_raises(tmp_path):
    big = ScatteringSample.disc(flat_tile(100.0), diameter=800.0)  # a model left in millimetres
    with pytest.raises(ValueError, match="elements"):
        BemSolver(cache_dir=str(tmp_path)).solve(big, SourceGrid.preset("survey", "rotational"), ReceiverGrid.hemisphere(5.0, 10.0), np.array([8000.0]), 343.0)
    assert not any(tmp_path.iterdir())  # nothing written


def test_remeshed_body_far_above_limit_raises_before_writing(tmp_path, monkeypatch):
    pytest.importorskip("mesh2scattering")
    import compas_acoustics.lab.solvers.bem as bem

    # the estimate (27 elements per body) is below 10 x 5 = 50, the mesh handed to the project (about 590 faces) is not
    monkeypatch.setattr(bem, "_remesh", lambda mesh, edge: bem._arrays(mesh))
    sample = ScatteringSample.disc(flat_tile(), diameter=0.3)
    assert estimate_elements(sample.sample, 343.0 / (500.0 * 6)) <= 50
    with pytest.warns(UserWarning, match="max_elements"):
        with pytest.raises(ValueError, match="elements"):
            BemSolver(cache_dir=str(tmp_path), numcalc_path="NumCalc", max_elements=5).solve(
                sample, SourceGrid.preset("survey", "rotational"), ReceiverGrid.hemisphere(5.0, 10.0), np.array([500.0]), 343.0
            )
    assert not any(tmp_path.iterdir())  # nothing written


def test_missing_mesh2scattering_names_the_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mesh2scattering", None)
    with pytest.raises(ImportError, match=r"compas_acoustics\[lab\]"):
        BemSolver(cache_dir=str(tmp_path)).solve(
            ScatteringSample.disc(flat_tile(), diameter=0.3),
            SourceGrid.preset("survey", "rotational"),
            ReceiverGrid.hemisphere(5.0, 10.0),
            np.array([500.0]),
            343.0,
        )


def test_the_bem_module_imports_without_its_dependencies():
    code = "import sys\nfor name in ('mesh2scattering', 'trimesh'):\n    sys.modules[name] = None\nimport compas_acoustics.lab.solvers.bem\n"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_remesh_keeps_the_body_closed_and_near_the_area_estimate():
    pytest.importorskip("trimesh")
    from compas_acoustics.lab.solvers.bem import _remesh

    s = ScatteringSample.disc(sine_tile(0.0708, 0.0204), diameter=0.4)
    for body in (s.sample, s.reference):
        edge = 343.0 / (4000.0 * 6)
        V, F = _remesh(body, edge)
        import trimesh

        mesh = trimesh.Trimesh(V, F, process=False)
        assert mesh.is_watertight and mesh.is_winding_consistent
        V0, F0 = body.to_vertices_and_faces()
        assert mesh.volume == pytest.approx(trimesh.Trimesh(np.asarray(V0), np.asarray(F0), process=False).volume, rel=1e-3)
        lengths = np.linalg.norm(V[mesh.edges_unique[:, 0]] - V[mesh.edges_unique[:, 1]], axis=1)
        assert lengths.max() <= edge * (1 + 1e-9)
        # the caps and walls are fans of slivers; plain midpoint subdivision makes 30-45 times the estimate of them
        assert len(F) < 3 * estimate_elements(body, edge)


@pytest.mark.slow
def test_bem_sine_surface_scatters_above_the_first_grating_lobe(tmp_path):
    pytest.importorskip("mesh2scattering")
    s = ScatteringSample.disc(sine_tile(period=0.0708, depth=0.0204), diameter=0.4)
    src = SourceGrid(np.array([31.5]), np.array([0.0]))
    rec = ReceiverGrid.hemisphere(5.0, 10.0)
    f = np.array([0.4, 1.0]) * 343.0 / 0.0708
    p = BemSolver(cache_dir=str(tmp_path)).solve(s, src, rec, f, 343.0)
    s_low, s_high = scattering_freefield(p.sample, p.reference, rec.weights)[0]
    print("s_low = {:.4f}, s_high = {:.4f}, generation = {}".format(s_low, s_high, p.generation))
    assert s_low < 0.2 and s_high > 0.4
    # a rerun reads the cache instead of solving again
    t0 = time.perf_counter()
    BemSolver(cache_dir=str(tmp_path)).solve(s, src, rec, f, 343.0)
    assert time.perf_counter() - t0 < 30
