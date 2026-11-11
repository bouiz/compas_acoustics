"""Image sources and ray tracing over several processes.

Two levels, as :func:`split_processes` decides:

- **emitters**: the sources are independent rooms (one image source search, one ray tracing each), so several run
  at once, each in its own process;
- **rays**: the rays of one source are independent and the histograms they fill add up, so they are split into
  slices traced in their own copy of the C++ room (:func:`trace_parallel`), and the histograms summed. Per-ray
  energies are scaled by the slice size, so the sum equals what one process collects, up to Monte-Carlo noise.

No :class:`pyroomacoustics.Room` state is written from outside: each worker rebuilds the C++ engine from a
picklable description of the walls (:func:`engine_spec`), and a room's response is assembled from pyroomacoustics'
own image source and ray-tracing synthesis functions (:func:`room_responses`).

Workers are spawned (never forked) and watch their parent: if it dies (a cancelled job, a killed process) they
exit instead of finishing work nobody waits for. They may be started from a daemonic process (a job runner's).

Ported from the pyroomacoustics fork of compas_soundscape (``simulation/rt_parallel.py``) and from compas_soundscape
(``backend/utils/child_pool.py``).
"""

from __future__ import annotations

import contextlib
import multiprocessing
import multiprocessing.util
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

__all__ = ["MIN_RAYS_PER_PROCESS", "SLICES_PER_PROCESS", "split_processes", "process_pool", "engine_spec", "trace_parallel", "room_responses"]

#: Below this many rays per process the spawn and room rebuild cost more than they save.
MIN_RAYS_PER_PROCESS = 500
#: Slices per process: a few more than one, so a slow slice does not leave the other processes idle.
SLICES_PER_PROCESS = 2


def split_processes(n_jobs: int, n_emitters: int) -> Tuple[int, int]:
    """``(emitters simulated at once, processes splitting the rays of each)`` for ``n_jobs`` processes."""
    at_once = max(1, min(int(n_jobs), int(n_emitters)))
    return at_once, max(1, int(n_jobs) // at_once)


# ─── Processes ──────────────────────────────────────────────────────────────


def _exit_when_parent_dies(parent_pid: int) -> None:
    if sys.platform == "win32":
        import ctypes

        synchronize = 0x00100000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(synchronize, False, parent_pid)
        if handle:
            kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        os._exit(1)
    while os.getppid() == parent_pid:
        time.sleep(1.0)
    os._exit(1)


def _init_worker(parent_pid: int) -> None:
    threading.Thread(target=_exit_when_parent_dies, args=(parent_pid,), daemon=True).start()


@contextlib.contextmanager
def _children_allowed() -> Iterator[None]:
    """Workers may start here even from a daemonic process, and do not re-import the caller's ``__main__``.

    ``multiprocessing`` refuses children of a daemonic process (a job runner's), to avoid orphans; these workers
    watch their parent themselves. Spawned workers re-import the parent's ``__main__`` (a whole application) unless
    it is hidden while they start; they only need this package.
    """
    config = getattr(multiprocessing.current_process(), "_config", None)
    was_daemon = bool(config and config.get("daemon"))
    if was_daemon:
        config["daemon"] = False
    main = sys.modules.get("__main__")
    hidden = {name: getattr(main, name) for name in ("__spec__", "__file__") if hasattr(main, name)}
    for name in hidden:
        if name == "__spec__":
            main.__spec__ = None
        else:
            del main.__file__
    try:
        yield
    finally:
        for name, value in hidden.items():
            setattr(main, name, value)
        if was_daemon:
            config["daemon"] = True


class _Pool:
    def __init__(self, n_workers: int):
        self._pool = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(os.getpid(),),
        )

    def submit(self, fn, *args):
        # workers start on demand, so every submit may start one
        with _children_allowed():
            return self._pool.submit(fn, *args)

    def close(self, failed: bool) -> None:
        if failed:
            # running tasks can take minutes and nobody needs their result any more
            for process in list(getattr(self._pool, "_processes", {}).values()):
                with contextlib.suppress(Exception):
                    process.kill()
        self._pool.shutdown(wait=not failed, cancel_futures=True)


@contextlib.contextmanager
def process_pool(n_workers: int) -> Iterator[_Pool]:
    """A pool of ``n_workers`` spawned processes; on an exception (a cancellation) its workers are killed."""
    pool = _Pool(n_workers)
    failed = True
    try:
        yield pool
        failed = False
    finally:
        pool.close(failed)


# The ray pools are kept per size and reused: starting a worker imports pyroomacoustics, which costs seconds.
_RAY_POOLS = {}
_RAY_POOLS_LOCK = threading.Lock()


def _ray_pool(n_workers: int) -> _Pool:
    with _RAY_POOLS_LOCK:
        pool = _RAY_POOLS.get(n_workers)
        if pool is None:
            pool = _RAY_POOLS[n_workers] = _Pool(n_workers)
        return pool


def shutdown_pools() -> None:
    """Stop the kept ray-tracing workers (they also stop on their own when this process ends)."""
    with _RAY_POOLS_LOCK:
        pools = list(_RAY_POOLS.values())
        _RAY_POOLS.clear()
    for pool in pools:
        pool.close(failed=False)


# The workers are non-daemonic children, which a multiprocessing child joins in its exit hook before ``atexit``
# handlers run, so a plain ``atexit`` shutdown would wait forever on idle workers. A finalizer with a non-negative
# exit priority runs before that join.
multiprocessing.util.Finalize(None, shutdown_pools, exitpriority=100)


# ─── Rays ───────────────────────────────────────────────────────────────────


def engine_spec(room) -> dict:
    """A picklable description of a polygonal 3-D room's C++ engine, as :meth:`pyroomacoustics.Room._init_room_engine` builds it."""
    return {
        "walls": [
            (
                np.array(w.corners, dtype=np.float32),
                np.array(w.absorption, dtype=np.float32),
                np.array(w.scatter, dtype=np.float32),
                str(w.name),
            )
            for w in room.walls
        ],
        "obstructing": [int(i) for i in room.room_engine.obstructing_walls],
        "params": (
            float(room.c),
            int(room.max_order),
            float(room.rt_args["energy_thres"]),
            float(room.rt_args["time_thres"]),
            float(room.rt_args["receiver_radius"]),
            float(room.rt_args["hist_bin_size"]),
            bool(room.simulator_state["ism_needed"] and room.simulator_state["rt_needed"]),
        ),
    }


def _trace_slice(job):
    """Worker: rebuild the room, trace one slice of rays, return every microphone's histograms (sparse)."""
    from pyroomacoustics import libroom
    from pyroomacoustics.room import wall_factory

    spec, mics, dirs, energies, position, seed = job
    libroom.set_rng_seed(int(seed))
    walls = [wall_factory(corners, absorption, scatter, name) for corners, absorption, scatter, name in spec["walls"]]
    engine = libroom.Room(walls, spec["obstructing"], [], *spec["params"])
    # all microphones are added before any is configured: configuring while the C++ vector grows crashes the tracer
    for location, _ in mics:
        engine.add_mic(np.asarray(location, dtype=np.float32).reshape(3, 1))
    for k, (_, directions) in enumerate(mics):
        if directions is None:
            engine.microphones[k].make_omnidirectional()
        else:
            engine.microphones[k].set_directions(np.asarray(directions, dtype=np.float32))
    engine.ray_tracing(dirs, energies, np.asarray(position, dtype=np.float32))
    return [[_to_sparse(h.get_hist()) for h in mic.histograms] for mic in engine.microphones]


def _to_sparse(hist):
    """(shape, flat indices, values) of the non-zero cells: a directional receiver's histograms are mostly empty."""
    hist = np.asarray(hist, dtype=np.float32)
    flat = hist.reshape(-1)
    idx = np.flatnonzero(flat)
    return hist.shape, idx.astype(np.int64), flat[idx]


def _accumulate(acc, part):
    for m, mic_hists in enumerate(part):
        if acc[m] is None:
            acc[m] = [None] * len(mic_hists)
        for d, (shape, idx, values) in enumerate(mic_hists):
            dense = acc[m][d]
            if dense is None:
                dense = np.zeros(shape, dtype=np.float32)
            elif dense.shape[-1] < shape[-1]:  # a longer histogram: rays that travelled farther
                dense = np.pad(dense, [(0, 0)] * (dense.ndim - 1) + [(0, shape[-1] - dense.shape[-1])])
            if idx.size:
                dense[np.unravel_index(idx, shape)] += values  # indices are unique within a slice
            acc[m][d] = dense


def trace_parallel(spec, mics, dirs, energies, position, n_processes: int, seeds, progress=None) -> List[List[np.ndarray]]:
    """Trace ``dirs`` (n_rays, 3) from ``position`` over ``n_processes`` processes.

    ``mics`` is one ``(location, directions or None)`` per engine microphone; ``energies`` (n_rays, n_bands) the
    rays' relative energies; ``seeds`` one RNG seed per slice (``SLICES_PER_PROCESS * n_processes``).
    ``progress(done_rays, total_rays)`` is called as slices complete. Returns, per microphone, the per-direction
    ``(bands, bins)`` histograms.
    """
    n_total = dirs.shape[0]
    n_slices = max(1, min(n_processes * SLICES_PER_PROCESS, n_total // MIN_RAYS_PER_PROCESS))
    slices = [np.arange(i, n_total, n_slices) for i in range(n_slices)]
    jobs = []
    for i, idx in enumerate(slices):
        # the C++ engine divides the initial energy by the number of rays it is given; this keeps ENERGY_0 / n_total
        jobs.append(
            (
                spec,
                mics,
                np.ascontiguousarray(dirs[idx], dtype=np.float32),
                np.ascontiguousarray(np.asarray(energies)[idx] * (len(idx) / n_total), dtype=np.float32),
                np.asarray(position, dtype=np.float32).reshape(-1),
                int(seeds[i]),
            )
        )
    acc = [None] * len(mics)
    pool = _ray_pool(n_processes)
    futures = {pool.submit(_trace_slice, job): len(slices[i]) for i, job in enumerate(jobs)}
    try:
        done = 0
        for future in as_completed(futures):
            _accumulate(acc, future.result())
            done += futures[future]
            if progress is not None:
                progress(done, n_total)
    except BaseException:
        # a cancelled run: drop the slices not started (running ones are short)
        for future in futures:
            future.cancel()
        raise
    return acc


def source_rays(source, n_rays: int, n_bands: int):
    """The directions and per-band energies pyroomacoustics shoots from ``source``, as its own ``Room.ray_tracing`` does."""
    import pyroomacoustics as pra

    if source.directivity is None:
        return pra.fibonacci_spherical_sampling(n_rays).T, np.ones((n_rays, n_bands))
    directions, energies = source.directivity.sample_rays(n_rays)
    return directions, np.broadcast_to(energies, (n_rays, n_bands))


def room_responses(room, histograms: Optional[List[List[List[np.ndarray]]]]) -> List[List[np.ndarray]]:
    """``[mic][source]`` responses of ``room`` from its image sources and given ray histograms, as ``Room.compute_rir``.

    ``room`` has run ``image_source_model()`` if its image sources are needed, and has omnidirectional microphones.
    ``histograms[m][s]`` is the per-direction list of microphone ``m`` for source ``s``, or None without ray tracing.
    """
    from pyroomacoustics.parameters import constants
    from pyroomacoustics.simulation import compute_ism_rir
    from pyroomacoustics.simulation import compute_rt_rir
    from pyroomacoustics.utilities import design_highpass_filter_sos
    from scipy.signal import sosfiltfilt

    sos = None
    if constants.get("rir_hpf_enable"):
        sos = design_highpass_filter_sos(room.fs, constants.get("rir_hpf_fc"), **constants.get("rir_hpf_kwargs"))
    fdl = constants.get("frac_delay_length")
    volume = room.get_volume()
    omni = {"response": np.ones(1), "is_impulse_response": False}
    out = []
    for m, mic in enumerate(room.mic_array.R.T):
        out.append([])
        for s, src in enumerate(room.sources):
            parts = []
            if room.simulator_state["ism_needed"]:
                parts.append(
                    compute_ism_rir(
                        src,
                        mic,
                        room.mic_array.directivity[m],
                        src.directions[m, :, :],
                        room.visibility[s][m, :],
                        fdl,
                        room.c,
                        room.fs,
                        room.octave_bands,
                        min_phase=room.min_phase,
                        air_abs_coeffs=room.air_absorption,
                    )
                )
            if histograms is not None:
                t0 = np.linalg.norm(src.position - mic) / room.c
                parts.append(
                    compute_rt_rir(
                        t0.item(),
                        histograms[m][s],
                        omni,
                        room.rt_args["hist_bin_size"],
                        room.rt_args["hist_bin_size_samples"],
                        volume,
                        fdl,
                        room.c,
                        room.fs,
                        room.octave_bands,
                        air_abs_coeffs=room.air_absorption,
                    )
                )
            rir = np.zeros(max(p.shape[0] for p in parts))
            for p in parts:
                rir[: p.shape[0]] += p
            if sos is not None:
                rir = np.ascontiguousarray(sosfiltfilt(sos, rir))
            out[m].append(rir)
    return out
