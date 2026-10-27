"""Run NumCalc on the projects in one folder: ``python -m compas_acoustics.lab.solvers._numcalc <folder> <numcalc_path>``.

:class:`compas_acoustics.lab.solvers.bem.BemSolver` starts this module in a child process, so that a cancelled job can kill it together with
its NumCalc instances. It calls ``mesh2scattering.numcalc.manage_numcalc`` (Kreuzer et al. 2024), which skips the frequency steps that
finished before and restarts the half-finished ones.

``manage_numcalc`` waits ``wait_time`` seconds after each launch so that it can measure the RAM the new instance takes; its default of 15 s
would add hours to a project with thousands of steps. Here the wait is short and the number of parallel instances is bounded instead by
NumCalc's own RAM estimate (``NumCalc -estimate_ram``, within about 4 % of the measured peak on the spike machine).

The wait also caps the parallelism: instances are launched one per wait, so no more than (step time / wait) overlap. In the low
splits a step takes about 3 s (2026-10-26, a 1.76 m sample, 1854 elements, 97 sources), and a wait of 2 s kept 1-2 of 14 cores busy,
about 1 h per split whatever its size; 0.25 s gave about 7.5. The wait is not zero: ``manage_numcalc`` counts running instances by process
name, and one just launched through its shell may not be counted yet. Its CPU gate (``max_cpu_load``, 90 % by default, the whole
system over 0.1 s) is raised to 95 %, so that it stops launches only when the machine is nearly full.
"""

import os
import sys

_WAIT = 0.1  # s between launches
_MAX_CPU = 95  # %, manage_numcalc's max_cpu_load: no launch above this system CPU load
_RAM_SAFETY = 1.05  # as manage_numcalc's ram_safety_factor
_RAM_SHARE = 0.9  # of the RAM available at the start


def max_instances(ram_per_instance, available, cpus):
    """The parallel NumCalc instances: as many as the RAM estimate allows, at least 1, at most the logical CPUs.

    Parameters
    ----------
    ram_per_instance : float
        NumCalc's largest RAM estimate of one frequency step in GB.
    available : float
        The available RAM in GB.
    cpus : int
        The logical CPUs.
    """
    if ram_per_instance <= 0:
        return max(1, int(cpus))
    return int(max(1, min(int(cpus), (_RAM_SHARE * available) // (_RAM_SAFETY * ram_per_instance))))


def main(argv=None):
    """Run the unfinished frequency steps of the NumCalc projects in ``argv[0]`` with the NumCalc binary ``argv[1]``."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit("usage: python -m compas_acoustics.lab.solvers._numcalc <folder> <numcalc_path>")
    folder = os.path.abspath(argv[0])  # manage_numcalc fails on a relative path
    numcalc_path = argv[1]

    import psutil
    from mesh2scattering import numcalc

    projects = [os.path.join(folder, name) for name in sorted(os.listdir(folder)) if os.path.isfile(os.path.join(folder, name, "parameters.json"))]
    if not projects:
        raise SystemExit("no NumCalc projects in {}".format(folder))
    # every source folder of a project has the same mesh and frequencies, so source_1's estimate stands for all of them
    ram = max(float(numcalc.calc_and_read_ram(project, numcalc_path)[:, 2].max()) for project in projects)
    instances = max_instances(ram, psutil.virtual_memory().available / 2**30, psutil.cpu_count())
    numcalc.manage_numcalc(project_path=folder, numcalc_path=numcalc_path, max_instances=instances, max_cpu_load=_MAX_CPU, wait_time=_WAIT, confirm_errors=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
