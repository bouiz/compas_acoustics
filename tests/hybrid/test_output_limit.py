"""The hybrid methods' limit_output: a per-pair output length from the pair's own reference."""

import warnings

import pytest

import compas_acoustics as ca

pytest.importorskip("pyFDN")

FS = 48000


def _scene():
    scene = ca.Scene("limit")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
    scene.add_source(ca.Source("s", 1.5, 1.5, 1.6))
    scene.add_receiver(ca.Receiver("near", 3.0, 2.0, 1.2))
    scene.add_receiver(ca.Receiver("far", 5.0, 4.0, 1.2))
    return scene


def _simulate(**kwargs):
    reference = ca.Mono(image_source_order=1, n_rays=2000, duration=0.6, seed=3)
    method = ca.Ambisonics(reference=reference, sh_order=1, image_source_order=1, seed=3, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {ir.receiver.name: ir for ir in _scene().simulate(method).values()}, method


def test_each_pair_takes_its_own_length_from_its_reference():
    seen = []

    def limit(references, fs):
        ((key, reference),) = references.items()
        seen.append(key)
        return 0.3 if reference.receiver.name == "near" else 0.4

    rirs, _ = _simulate(limit_output=limit)
    assert len(seen) == 2
    assert rirs["near"].ir.shape[0] == int(round(0.3 * FS)) and rirs["far"].ir.shape[0] == int(round(0.4 * FS))


def test_a_given_duration_wins_and_none_keeps_the_default():
    default, _ = _simulate()
    fixed, _ = _simulate(duration=0.25, limit_output=lambda references, fs: 0.1)
    assert all(ir.ir.shape[0] == int(round(0.25 * FS)) for ir in fixed.values())
    unlimited, _ = _simulate(limit_output=lambda references, fs: None)
    for name, ir in default.items():
        assert unlimited[name].ir.shape == ir.ir.shape


def test_limit_output_must_be_callable_and_positive():
    with pytest.raises(TypeError):
        ca.Ambisonics(limit_output=0.5)
    with pytest.raises(ValueError, match="positive"):
        _simulate(limit_output=lambda references, fs: 0.0)
