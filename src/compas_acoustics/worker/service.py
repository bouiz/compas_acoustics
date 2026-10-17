"""The worker's compas.rpc service: localhost only, and only :mod:`compas_acoustics.worker.api`.

compas.rpc's default service listens on every network interface and imports any
module it is sent, which on a shared network lets anyone run code on this
machine. Started by :func:`compas_acoustics.worker.proxy` as
``python -m compas_acoustics.worker.service --port N --no-autoreload``.
"""

import argparse
import json

from compas.data import DataEncoder
from compas.rpc import Dispatcher
from compas.rpc import Server

API = "compas_acoustics.worker.api."


class WorkerService(Dispatcher):
    def _dispatch(self, name, args):
        if not name.startswith(API) or "." in name[len(API):]:
            error = f"{name} is not part of the compas_acoustics worker API."
            return json.dumps({"data": None, "error": error, "profile": None}, cls=DataEncoder)
        # args[1] would be put on sys.path by the dispatcher; the worker needs nothing from the client's paths
        return super()._dispatch(name, args[:1])


def start_service(port, host="127.0.0.1"):
    server = Server((host, port))
    server.register_instance(WorkerService())
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    # compas.rpc.Proxy always passes one of these; the worker never reloads modules
    parser.add_argument("--autoreload", action="store_true")
    parser.add_argument("--no-autoreload", dest="autoreload", action="store_false")
    start_service(parser.parse_args().port)
