"""The worker's compas.rpc service: localhost only, only :mod:`compas_acoustics.worker.api`, only for callers that hold the token.

compas.rpc's default service listens on every network interface and imports any
module it is sent, which on a shared network lets anyone run code on this
machine. Started by :func:`compas_acoustics.worker.proxy` as
``python -m compas_acoustics.worker.service --port N --no-autoreload``, with the token in
``COMPAS_ACOUSTICS_WORKER_TOKEN`` and the client's process id in ``COMPAS_ACOUSTICS_WORKER_PARENT``.
The worker exits when that process is gone.
"""

import argparse
import atexit
import hmac
import json
import os
import sys
import threading
import time

from compas.data import DataEncoder
from compas.rpc import Dispatcher
from compas.rpc import Server

from compas_acoustics.worker import PARENT_VARIABLE
from compas_acoustics.worker import TOKEN_VARIABLE
from compas_acoustics.worker import WORKERS

__all__ = ["WorkerService", "process_is_alive", "start_service"]

API = "compas_acoustics.worker.api."
PARENT_CHECK_SECONDS = 2.0


def _refusal(error):
    return json.dumps({"data": None, "error": error, "profile": None}, cls=DataEncoder)


class WorkerService(Dispatcher):
    def __init__(self, token):
        super().__init__()
        self._token = token

    def _dispatch(self, name, args):
        if not name.startswith(API) or "." in name[len(API):]:
            return _refusal(f"{name} is not part of the compas_acoustics worker API.")
        # The client sends the token as the first argument of every call
        try:
            payload = json.loads(args[0])
            token = payload["args"].pop(0)
        except (IndexError, KeyError, TypeError, ValueError):
            return _refusal("Unauthorised.")
        if not hmac.compare_digest(str(token), self._token):
            return _refusal("Unauthorised.")
        # args[1] would be put on sys.path by the dispatcher; the worker needs nothing from the client's paths
        return super()._dispatch(name, [json.dumps(payload)])


def process_is_alive(pid):
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _record(port):
    WORKERS.mkdir(parents=True, exist_ok=True)
    record = WORKERS / f"{os.getpid()}.json"
    record.write_text(json.dumps({"pid": os.getpid(), "port": port}))
    atexit.register(lambda: record.unlink(missing_ok=True))
    return record


def _exit_with_the_client(pid, record):
    while process_is_alive(pid):
        time.sleep(PARENT_CHECK_SECONDS)
    record.unlink(missing_ok=True)
    os._exit(0)


def start_service(port, token, parent=None, host="127.0.0.1"):
    server = Server((host, port))
    server.register_instance(WorkerService(token))
    record = _record(port)
    if parent:
        threading.Thread(target=_exit_with_the_client, args=(parent, record), daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    # compas.rpc.Proxy always passes one of these; the worker never reloads modules
    parser.add_argument("--autoreload", action="store_true")
    parser.add_argument("--no-autoreload", dest="autoreload", action="store_false")
    token = os.environ.pop(TOKEN_VARIABLE, None)
    if not token:
        sys.exit(f"{TOKEN_VARIABLE} is not set: the worker is started by compas_acoustics.worker.proxy().")
    parent = os.environ.pop(PARENT_VARIABLE, None)
    start_service(parser.parse_args().port, token, int(parent) if parent else None)
