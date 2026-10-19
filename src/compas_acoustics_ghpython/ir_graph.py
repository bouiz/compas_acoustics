"""The impulse response graph a Simulate component returns, tagged with the method that produced the wire."""

__all__ = ["MethodGraph"]


class MethodGraph:
    """A scene's IR graph, tagged with the simulation method whose output it is.

    Every simulation of a scene writes to the scene's one graph, so a wire from Simulate Hybrid and one from
    Simulate Pyroomacoustics carry the same data. The tag tells Deconstruct IR Graph which method's responses to
    read. Everything else is the graph's own: attribute access passes through to it.

    Parameters
    ----------
    graph : :class:`compas.datastructures.Graph`
        The scene's ``ir_graph``.
    method : str
        Name of the simulation method, as the graph's edge attribute that holds its responses, e.g. ``'hybrid'``.
    """

    def __init__(self, graph, method):
        self.graph = graph
        self.method = method

    def __getattr__(self, name):
        if name == "graph":
            raise AttributeError(name)
        return getattr(self.graph, name)
