"""Shared test helpers."""

import numpy as np
import pytest


def _assert_meets_contract(rirs, context, method, direct_sound=True):
    """Check a method's responses against the output contract in :mod:`compas_acoustics.simulation_methods`.

    ``direct_sound=False`` skips the free-field amplitude check, for scenes where
    an obstacle blocks the direct path.
    """
    expected = {context.key(s, r) for s in context.sources for r in context.receivers}
    assert set(rirs) == expected, "one response per source-receiver pair inside the room"
    for key, ir in rirs.items():
        # 1. the package scale
        assert ir.generation.get("scale") == "package", f"{key}: not on the package scale"
        if direct_sound:
            raw = ir.denormalized()
            omni = raw if raw.ndim == 1 else raw[:, 0]
            distance = float(np.linalg.norm(np.subtract(ir.receiver.position, ir.source.position)))
            n = ir.direct_sound_index()
            peak = float(np.max(np.abs(omni[max(n - 1, 0):n + 2])))
            free_field = 1.0 / (4.0 * np.pi * distance)
            assert 0.5 * free_field < peak < 1.2 * free_field, f"{key}: direct sound {peak:.4g}, free field {free_field:.4g}"
        # 2. the context's sample rate
        assert ir.sampling_rate == context.sampling_rate, f"{key}: sample rate {ir.sampling_rate}"
        # 3. how it was made
        assert ir.generation.get("method") == method.name, f"{key}: generation['method'] is not {method.name!r}"
        assert "valid_range_hz" in ir.generation, f"{key}: no valid_range_hz"
        # 4. the channel convention
        if method.output == "mono":
            assert ir.n_channels == 1 and ir.ambisonic_order is None, f"{key}: not mono"
        else:
            assert ir.ambisonic_order is not None, f"{key}: no ambisonic order"
            assert ir.n_channels == (ir.ambisonic_order + 1) ** 2, f"{key}: channel count"
            assert ir.normalization in ("sn3d", "n3d") and ir.channel_order == "acn", f"{key}: convention"
        # 5. normalized view by default, raw level kept
        if getattr(method, "normalize", True):
            assert ir.normalized and np.max(np.abs(ir.ir)) == pytest.approx(1.0), f"{key}: not normalized"
        assert np.max(np.abs(ir.denormalized())) == pytest.approx(ir.normalization_factor)


@pytest.fixture
def assert_meets_contract():
    """The contract check, as a fixture: ``assert_meets_contract(rirs, context, method)``."""
    return _assert_meets_contract
