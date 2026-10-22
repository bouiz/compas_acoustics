"""Progress and cancellation of the worker's calls.

A call that comes with a call id runs as a job of its request thread. Long computations report their stages with
:func:`progress`, which also stops them, with :class:`Cancelled`, once a client cancelled the call. Outside a job
(a script, a test, the Python client's own process) both are no-ops, so the methods report unconditionally.
A stage that is running when the call is cancelled finishes; the next report raises.
"""

import threading

__all__ = ["Cancelled", "start", "finish", "cancel", "progress", "check", "status"]


class Cancelled(Exception):
    """The client cancelled the call."""


class _Job:
    def __init__(self, call_id):
        self.id = call_id
        self.stage = None
        self.done = None
        self.of = None
        self.cancelled = False


_LOCK = threading.Lock()
_JOBS = {}
_LOCAL = threading.local()


def start(call_id):
    """Run the rest of this thread's work as the job ``call_id``."""
    job = _Job(call_id)
    with _LOCK:
        _JOBS[call_id] = job
    _LOCAL.job = job
    return job


def finish(call_id):
    """Forget the job: its answer is sent."""
    with _LOCK:
        _JOBS.pop(call_id, None)
    if getattr(_LOCAL, "job", None) is not None and _LOCAL.job.id == call_id:
        _LOCAL.job = None


def cancel(call_id):
    """Ask the job ``call_id`` to stop at its next report; False if there is no such job."""
    with _LOCK:
        job = _JOBS.get(call_id)
        if job is None:
            return False
        job.cancelled = True
        return True


def check():
    """Raise :class:`Cancelled` if this thread's job was cancelled."""
    job = getattr(_LOCAL, "job", None)
    if job is not None and job.cancelled:
        raise Cancelled()


def progress(stage, done=None, of=None):
    """Report that this thread's job is at ``stage`` (``done`` of ``of``), and stop it if it was cancelled."""
    job = getattr(_LOCAL, "job", None)
    if job is None:
        return
    job.stage, job.done, job.of = stage, done, of
    check()


def status(call_id):
    """The last report of the job ``call_id``, ``{"stage", "done", "of", "cancelled"}``; None if there is no such job."""
    with _LOCK:
        job = _JOBS.get(call_id)
        if job is None:
            return None
        return {"stage": job.stage, "done": job.done, "of": job.of, "cancelled": job.cancelled}
