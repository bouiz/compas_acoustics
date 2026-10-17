"""Higher ambisonic orders must work without the caller hunting for a grid.

``num_delays`` and ``sh_order`` are coupled twice over: the direction grid has
to hold exactly one point per delay line, and it has to support the order well
enough that the delay lines stay decorrelated across ambisonic channels. With
a fixed default of 16 the second order failed the leakage check at 7.8 per
cent and the third could not even be built -- ``equal_area`` returns 17 points
when asked for 16 at ``n_max=3``.

The set of counts that satisfies both is irregular (20, 23, 24, 27 ... for
second order), so the default is searched for rather than computed.

Ported from hybrid-rir (gpapadimitri/hybrid-rir @ a272da7, tests/test_late_hoa_defaults.py).
"""

import numpy as np
import pyFDN
import pytest
import spharpy

from compas_acoustics.hybrid.bands import BANDS
from compas_acoustics.hybrid.late import ambisonic_output_matrix, default_num_delays, design_fdn
from compas_acoustics.hybrid.target import RoomTarget


def flat_target(rt=0.6):
    return RoomTarget(
        rt=np.full(BANDS.size, rt),
        level=np.ones(BANDS.size),
        bands=BANDS.copy(),
        fs=48000.0,
    )


def delays_of(n):
    return pyFDN.sample_delay_lengths(n, (500, 1500), coprime=True, rng=0)


@pytest.mark.parametrize("sh_order", [0, 1, 2, 3])
def test_the_default_grid_supports_the_order_it_was_chosen_for(sh_order):
    num_delays = default_num_delays(sh_order)
    directions = spharpy.samplings.equal_area(n_max=sh_order, n_points=num_delays)
    assert directions.cshape[0] == num_delays
    ambisonic_output_matrix(sh_order, directions)  # raises if leakage is too high


def test_first_order_still_defaults_to_sixteen_delay_lines():
    assert default_num_delays(1) == 16


def test_the_default_grows_with_the_ambisonic_order():
    counts = [default_num_delays(n) for n in range(1, 4)]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


@pytest.mark.parametrize("sh_order", [1, 2, 3])
def test_design_fdn_works_at_every_order_with_the_default_count(sh_order):
    builds = design_fdn(
        flat_target(), delays_of(default_num_delays(sh_order)), sh_order=sh_order
    )
    assert builds[0].C.shape[0] == (sh_order + 1) ** 2


def test_a_delay_count_that_cannot_support_the_order_raises():
    """The guard must not be softened into a silent substitution."""
    with pytest.raises(ValueError, match="leaks|equal_area returned"):
        design_fdn(flat_target(), delays_of(16), sh_order=2)
