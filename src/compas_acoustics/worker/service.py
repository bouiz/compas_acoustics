"""The worker's HTTP service on 127.0.0.1, for callers that hold the token.

Started by :func:`compas_acoustics.worker.client` or the Grasshopper plugin as
``python -m compas_acoustics.worker.service --port N``, with the token in ``COMPAS_ACOUSTICS_WORKER_TOKEN`` and the
client's process id in ``COMPAS_ACOUSTICS_WORKER_PARENT``; it exits when that process is gone. Bodies are compas_pb
payloads.

* ``GET /ping``: ``pong <id>``, without the token; ``<id>`` is the first 16 hex digits of the token's SHA-256.
* ``POST /api/<function>``: ``{"args": [...], "kwargs": {...}}`` for a function of
  :data:`compas_acoustics.worker.api.__all__`; answers ``{"result": ...}``.
* ``POST /cancel/<call id>``, ``GET /progress/<call id>``: stop, or report on, the call sent with ``X-Call-Id``
  (:mod:`compas_acoustics.worker.jobs`).
* ``POST /shutdown``.

All but ``/ping`` need ``Authorization: Bearer <token>``. Errors answer ``{"error": text}``: 401 without the token,
404 for an unknown route, function or call, 400 for a body that is not a compas_pb payload and for a function's
ValueError or TypeError, 499 for a cancelled call, 500 for any other exception (its traceback).
"""

import argparse
import atexit
import hashlib
import hmac
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer

from compas_acoustics.worker import CONTENT_TYPE
from compas_acoustics.worker import PARENT_VARIABLE
from compas_acoustics.worker import TOKEN_VARIABLE
from compas_acoustics.worker import WORKERS
from compas_acoustics.worker import jobs

__all__ = ["WorkerService", "dispatch", "process_is_alive", "start_service"]


def _encode(value):
    from compas_pb import pb_dump_bts

    from compas_acoustics.pb import plain

    return pb_dump_bts(plain(value))


def _error(status, text):
    return status, _encode({"error": text})


def authorized(token, authorization):
    """Whether the ``Authorization`` header carries ``token``."""
    scheme, _, given = (authorization or "").partition(" ")
    return scheme == "Bearer" and bool(given) and hmac.compare_digest(given.encode(), token.encode())


def dispatch(token, authorization, function, body, call_id=None):
    """Answer a call of ``function`` with the compas_pb ``body``: ``(status, compas_pb payload)``.

    With a ``call_id`` the call runs as a job that :func:`compas_acoustics.worker.jobs.cancel` can stop.
    """
    if not authorized(token, authorization):
        return _error(401, "Unauthorised.")
    if not call_id:
        return _dispatch(function, body)
    jobs.start(call_id)
    try:
        return _dispatch(function, body)
    finally:
        jobs.finish(call_id)


def _dispatch(function, body):
    from compas_pb import pb_load_bts

    import compas_acoustics.pb  # noqa: F401  (registers the converters)
    from compas_acoustics.worker import api

    if function not in api.__all__:
        return _error(404, f"{function} is not part of the compas_acoustics worker API.")
    try:
        request = pb_load_bts(body)
        args, kwargs = list(request.get("args", [])), dict(request.get("kwargs", {}))
    except Exception as e:
        return _error(400, f"The request is not a compas_pb payload of args and kwargs: {e}")
    try:
        result = getattr(api, function)(*args, **kwargs)
    except jobs.Cancelled:
        return _error(499, "cancelled")
    except (ValueError, TypeError) as e:
        return _error(400, str(e))
    except Exception:
        return _error(500, traceback.format_exc())
    return 200, _encode({"result": result})


class _Handler(BaseHTTPRequestHandler):
    server_version = "compas_acoustics_worker"

    def log_message(self, format, *args):
        pass

    def _answer(self, status, body, content_type=CONTENT_TYPE):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # HTTP/1.0 closes after each answer; without the header .NET's HttpClient at times reuses the closed connection
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/ping":
            self._answer(200, b"pong " + hashlib.sha256(self.server.token.encode()).hexdigest()[:16].encode(), "text/plain")
        elif self.path.startswith("/progress/"):
            if not authorized(self.server.token, self.headers.get("Authorization")):
                self._answer(*_error(401, "Unauthorised."))
                return
            status = jobs.status(self.path[len("/progress/") :])
            self._answer(*((200, _encode({"result": status})) if status is not None else _error(404, "No such call.")))
        else:
            self._answer(*_error(404, "Not found."))

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        authorization = self.headers.get("Authorization")
        if self.path.startswith("/api/"):
            call_id = (self.headers.get("X-Call-Id") or "")[:64] or None
            self._answer(*dispatch(self.server.token, authorization, self.path[len("/api/") :], body, call_id))
        elif not authorized(self.server.token, authorization):
            self._answer(*_error(401, "Unauthorised."))
        elif self.path == "/shutdown":
            self._answer(200, b"", "text/plain")
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        elif self.path.startswith("/cancel/"):
            self._answer(200, _encode({"result": jobs.cancel(self.path[len("/cancel/") :])}))
        else:
            self._answer(*_error(404, "Not found."))


class WorkerService(ThreadingHTTPServer):
    """The HTTP server of the worker, answering only callers that send ``token``."""

    daemon_threads = True

    def __init__(self, address, token):
        super().__init__(address, _Handler)
        self.token = token


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


def _record(port, token):
    # The token lets Install Worker stop this worker before it replaces the environment; the folder is the user's own
    WORKERS.mkdir(parents=True, exist_ok=True)
    record = WORKERS / f"{os.getpid()}.json"
    record.write_text(json.dumps({"pid": os.getpid(), "port": port, "token": token}))
    try:
        os.chmod(record, 0o600)
    except OSError:
        pass
    atexit.register(lambda: record.unlink(missing_ok=True))
    return record


def _exit_with_the_client(pid, record):
    while process_is_alive(pid):
        time.sleep(2.0)
    record.unlink(missing_ok=True)
    os._exit(0)


def start_service(port, token, parent=None):
    import compas_acoustics.pb  # noqa: F401  (the first call should not pay for the imports)

    server = WorkerService(("127.0.0.1", port), token)
    record = _record(port, token)
    if parent:
        threading.Thread(target=_exit_with_the_client, args=(parent, record), daemon=True).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
        record.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    token = os.environ.pop(TOKEN_VARIABLE, None)
    if not token:
        sys.exit(f"{TOKEN_VARIABLE} is not set: the worker is started by compas_acoustics.worker.client() or the Grasshopper plugin.")
    parent = os.environ.pop(PARENT_VARIABLE, None)
    start_service(parser.parse_args().port, token, int(parent) if parent else None)
