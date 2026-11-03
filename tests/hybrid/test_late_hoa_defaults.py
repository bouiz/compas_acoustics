"""Higher ambisonic orders must work without the caller hunting for a grid.

``num_delays`` and ``sh_order`` are coupled twice over: the direction grid has
to hold exactly one point per delay line, and it has to support the order well
enough that the delay lines stay decorrelated across ambisonic channels.
``equal_area`` does not always return the count asked for (17 points for 16
at ``n_max=3``), and the set of counts that satisfies both is irregular, so
the default is searched for rather than computed.
"""

import numpy as np
import pytest
import spharpy

from compas_acoustics.hybrid.late import MIN_DELAYS, fdn_directions
from compas_acoustics.hybrid.ambisonics import ambisonic_output_matrix, default_num_delays


@pytest.mark.parametrize("sh_order", [0, 1, 2, 3])
def test_the_default_grid_supports_the_order_it_was_chosen_for(sh_order):
    num_delays = default_num_delays(sh_order)
    directions = spharpy.samplings.equal_area(n_max=sh_order, n_points=num_delays)
    assert directions.cshape[0] == num_delays
    ambisonic_output_matrix(sh_order, directions)  # raises if leakage is too high


def test_orders_zero_to_three_default_to_kirschs_96_delay_lines():
    assert MIN_DELAYS == 96 and [default_num_delays(n) for n in range(4)] == [96] * 4
    assert default_num_delays(1, minimum=16) == 16 and default_num_delays(2, minimum=16) == 36


def test_the_default_grows_with_the_ambisonic_order_above_the_floor():
    counts = [default_num_delays(n) for n in range(1, 5)]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]  # fourth order needs more than the 96 lines of the floor


@pytest.mark.parametrize("sh_order", [1, 2, 3, 4])
def test_the_output_matrix_works_at_every_order_with_the_default_count(sh_order):
    count = default_num_delays(sh_order)
    assert ambisonic_output_matrix(sh_order, fdn_directions(count)).shape == ((sh_order + 1) ** 2, count)


def test_the_grid_depends_on_the_count_alone():
    for sh_order in range(4):
        grid = spharpy.samplings.equal_area(n_max=sh_order, n_points=96)
        np.testing.assert_array_equal(grid.cartesian, fdn_directions(96).cartesian)


def test_a_delay_count_that_cannot_support_the_order_raises():
    """An order the grid cannot carry raises rather than being replaced."""
    with pytest.raises(ValueError, match="leaks"):
        ambisonic_output_matrix(2, fdn_directions(16))
