import logging
from typing import Iterable
from typing import Union

from compas.data import Data
from compas.datastructures import Graph

from compas_acoustics.defaults import Defaults
from compas_acoustics.ir import ImpulseResponse
from compas_acoustics.receiver import Receiver
from compas_acoustics.source import Source

__all__ = ['Scene']

_log = logging.getLogger(__name__)


class Scene(Data):
    """The acoustic model of a room: its surfaces with materials, sources, receivers, and the impulse responses simulated for them.

    Parameters
    ----------
    name : str
        The name of the scene.
    description : str, optional
        A description of the scene.
    sampling_rate : int, optional
        The sample rate in Hz at which every simulation method runs. Default is 48000.

    Examples
    --------
    >>> import compas_acoustics as ca
    >>> scene = ca.Scene("office")
    >>> scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
    >>> scene.add_source(ca.Source("talker", 1.5, 1.5, 1.6))
    >>> scene.add_receiver(ca.Receiver("listener", 4.0, 3.0, 1.2))
    >>> len(scene.acoustic_objects), len(scene.sources), len(scene.receivers)
    (6, 1, 1)
    >>> scene.sampling_rate
    48000
    """

    
    def __init__(self, name:str, description:str="", sampling_rate:int=None):
        super().__init__()
        self.name = name
        self.description = description
        self._acoustic_objects = []
        self._sources = []
        self._receivers = []
        self.ir_graph = Graph()
        self.sampling_rate = Defaults().SAMPLING_RATE if sampling_rate is None else sampling_rate


    @property
    def sampling_rate(self) -> int:
        """Sample rate in hertz at which :meth:`simulate` runs every method; 48000 unless given."""
        return self._sampling_rate


    @sampling_rate.setter
    def sampling_rate(self, sampling_rate:int):
        if sampling_rate <= 0 or int(sampling_rate) != sampling_rate:
            raise ValueError(f"sampling_rate must be a positive whole number of hertz, got {sampling_rate!r}")
        self._sampling_rate = int(sampling_rate)


    @property
    def __data__(self):
        # The IR graph is not serialised: responses travel as WAV files (ImpulseResponse.to_wav)
        return {
            "name": self.name,
            "description": self.description,
            "sampling_rate": self.sampling_rate,
            "acoustic_objects": self.acoustic_objects,
            "sources": self._sources,
            "receivers": self._receivers,
        }


    @classmethod
    def __from_data__(cls, data):
        scene = cls(data["name"], data["description"], data.get("sampling_rate"))
        scene.add_acoustic_objects(data["acoustic_objects"])
        for source in data["sources"]:
            scene.add_source(source)
        for receiver in data["receivers"]:
            scene.add_receiver(receiver)
        return scene


    @property
    def id(self):
        return self.guid
    
    
    @property
    def acoustic_objects(self):
        return self._acoustic_objects
    
    
    @property
    def sources(self):
        all_sources = []
        
        # add standalone sources
        all_sources.extend(self._sources)
        
        # add sources from acoustic objects
        for acoustic_object in self.acoustic_objects:
            all_sources.extend(acoustic_object.sources)
        
        return all_sources
    
    
    @property
    def receivers(self):
        return self._receivers
    
    
    @property
    def number_of_acoustic_objects(self):
        return len(self._acoustic_objects)
    
    
    @property
    def number_of_sources(self):
        return len(self.sources)
    
    
    @property
    def number_of_receivers(self):
        return len(self._receivers)
    
    
    @property
    def description(self):
        return self._description
    
    
    @description.setter
    def description(self, description:str):
        self._description = description
        return self._description
    
    
    @property
    def name(self):
        return self._name
    
    
    @name.setter
    def name(self, name:str):
        self._name = name
        return self._name
    
    
          
    
    
    @property
    def acoustic_materials(self):
        """The unique acoustic materials of the scene, keyed by name (as :attr:`scene_materials`)."""
        return self.scene_materials


    @acoustic_materials.setter
    def acoustic_materials(self, materials):
        # Assigning materials by surface name was removed; an attribute set here would do nothing
        raise AttributeError(
            "Scene materials are no longer assigned by surface name. Give each AcousticObject its material, e.g. "
            "AcousticObjectsLibrary.shoebox_room(..., materials={'floor': carpet, ...}) or "
            "AcousticObject.from_obj(path, materials={'<group>': material})."
        )


    @property
    def scene_materials(self):
        """The unique acoustic materials used in the scene, keyed by material name."""
        materials = {}
        for acoustic_object in self.acoustic_objects:
            material = acoustic_object.material
            if material is not None:
                materials[material.name] = material
        return materials


    def print_scene_materials(self):
        print(f"Scene '{self.name}' materials:")
        for acoustic_object in self.acoustic_objects:
            material_name = acoustic_object.material.name if acoustic_object.material else "none"
            print(f"  {acoustic_object.name}: {material_name}")
                
    
    def display_materials_in_viewer(self):
        import compas
        if compas.is_rhino():
            raise RuntimeError("Displaying materials is not supported in Rhino.")
        from random import random

        from compas.colors import Color
        from compas_viewer import Viewer  # Does not work in Rhino
        from compas_viewer.scene import Tag  # Does not work in Rhino

        material_colors = {name: Color(random(), random(), random(), 0.5) for name in self.scene_materials}
        default_color = Color(0.5, 0.5, 0.5, 0.5)

        viewer = Viewer()

        for acoustic_object in self.acoustic_objects:
            material = acoustic_object.material
            material_name = material.name if material else "none"
            color = material_colors.get(material_name, default_color)

            mesh = acoustic_object.acoustic_geometry
            tag = Tag(material_name, mesh.centroid(), height=30, color=color)
            viewer.scene.add(tag, name=f"tag_{acoustic_object.name}")

            facecolor = {fkey: color for fkey in mesh.faces()}
            viewer.scene.add(mesh, name=acoustic_object.name, show_points=True, facecolor=facecolor)
        viewer.show()
           

    def add_acoustic_object(self, acoustic_object):
        self._acoustic_objects.append(acoustic_object)
        for source in acoustic_object.sources:
            self._register_ir_graph_node(source)
        
    
    def add_acoustic_objects(self, acoustic_objects:list):
        for acoustic_object in acoustic_objects:
            self.add_acoustic_object(acoustic_object)


    def add_source(self, source):
        self._sources.append(source)
        self._register_ir_graph_node(source)


    def add_receiver(self, receiver):
        self._receivers.append(receiver)
        self._register_ir_graph_node(receiver)

    
    def _ir_node_key(self, node: Union[Source, Receiver]) -> str:
        return str(node.guid)

    
    def _register_ir_graph_node(self, node_data: Union[Source, Receiver]) -> None:
        if isinstance(node_data, Receiver):
            node_type = 'receiver'
        elif isinstance(node_data, Source):
            node_type = 'source'
        else:
            raise ValueError("node_data must be an instance of Receiver or Source")

        node_key = self._ir_node_key(node_data)
        if self.ir_graph.has_node(node_key):
            return

        node_attributes = {
            'type': node_type,
            'data': node_data
        }
        self.ir_graph.add_node(node_key, attr_dict=node_attributes)

    
    def add_node_to_ir_graph(self, node_data: Union[Source, Receiver]) -> None:
        self._register_ir_graph_node(node_data)

    
    def clear_ir_edges(self) -> None:
        for edge in list(self.ir_graph.edges()):
            self.ir_graph.delete_edge(edge)

    
    def add_impulse_response(self, source: Source, receiver: Receiver, ir: ImpulseResponse, attribute: str = 'data') -> None:
        """Store ``ir`` on the source-receiver edge under ``attribute``.

        Each simulation method stores its responses under its own name
        (``'pyroomacoustics'``, ``'hybrid'``, ``'hybrid_binaural'``); ``'data'``
        holds the most recent mono response. An edge carries any number of them.
        """
        source_key = self._ir_node_key(source)
        receiver_key = self._ir_node_key(receiver)

        if not self.ir_graph.has_node(source_key):
            raise ValueError(f"Source '{source.name}' is not registered in the IR graph.")
        if not self.ir_graph.has_node(receiver_key):
            raise ValueError(f"Receiver '{receiver.name}' is not registered in the IR graph.")

        source_attrs = self.ir_graph.node_attributes(source_key)
        receiver_attrs = self.ir_graph.node_attributes(receiver_key)
        if source_attrs.get('type') != 'source':
            raise ValueError(f"Node '{source.name}' is not a source.")
        if receiver_attrs.get('type') != 'receiver':
            raise ValueError(f"Node '{receiver.name}' is not a receiver.")

        edge = (source_key, receiver_key)
        if self.ir_graph.has_edge(edge):
            self.ir_graph.edge_attribute(edge, attribute, ir)
        else:
            self.ir_graph.add_edge(source_key, receiver_key, attr_dict={attribute: ir})

    
    def populate_ir_graph(self, impulse_responses: Union[dict, Iterable[ImpulseResponse]], attribute: str = 'data') -> None:
        """Replace every response stored under ``attribute``; other attributes are kept."""
        for edge in list(self.ir_graph.edges()):
            self.ir_graph.unset_edge_attribute(edge, attribute)
            if not self.ir_graph.edge_attributes(edge):
                self.ir_graph.delete_edge(edge)

        if isinstance(impulse_responses, dict):
            responses = impulse_responses.values()
        else:
            responses = impulse_responses

        for ir in responses:
            self.add_impulse_response(ir.source, ir.receiver, ir, attribute=attribute)

    
    def get_impulse_response(self, source: Source, receiver: Receiver, attribute: str = 'data') -> ImpulseResponse:
        source_key = self._ir_node_key(source)
        receiver_key = self._ir_node_key(receiver)

        if not self.ir_graph.has_node(source_key):
            raise KeyError(f"Source '{source.name}' is not registered in the IR graph.")
        if not self.ir_graph.has_node(receiver_key):
            raise KeyError(f"Receiver '{receiver.name}' is not registered in the IR graph.")
        if not self.ir_graph.has_edge((source_key, receiver_key)):
            raise KeyError(f"No impulse response found for source '{source.name}' and receiver '{receiver.name}'.")
        ir = self.ir_graph.edge_attribute((source_key, receiver_key), attribute)
        if ir is None:
            raise KeyError(f"No '{attribute}' impulse response for source '{source.name}' and receiver '{receiver.name}'.")
        return ir


    def impulse_responses(self, attribute: str = 'data') -> dict:
        """Every response stored under ``attribute``, keyed ``'<receiver guid>_<source guid>'``."""
        out = {}
        for edge in self.ir_graph.edges():
            ir = self.ir_graph.edge_attribute(edge, attribute)
            if ir is not None:
                out[f"{ir.receiver.guid}_{ir.source.guid}"] = ir
        return out


    def simulation_methods(self) -> list:
        """The names of the simulation methods whose responses are on the IR graph, ``'data'`` aside."""
        names = {}
        for edge in self.ir_graph.edges():
            for name in self.ir_graph.edge_attributes(edge):
                if name != 'data':
                    names.setdefault(name, None)
        return list(names)


    def simulate(self, method) -> dict:
        """Run a simulation method on the scene and store its responses on the IR graph.

        The responses go on the graph under ``method.attribute``, the method's
        name unless it renders several formats; a method's responses in further
        formats from the same run (``method.extra_responses``) under theirs. A
        mono method's responses also go under ``'data'``, which the analysis and
        auralisation components read by default. Coplanar faces of the room are
        merged. The method runs at the scene's :attr:`sampling_rate`.

        Parameters
        ----------
        method : :class:`~compas_acoustics.simulation_methods.SimulationMethod`
            For example ``Pyroomacoustics(ray_tracing=True)`` or ``Hybrid()``.

        Returns
        -------
        Dict[str, ImpulseResponse]
            One response per source-receiver pair inside the room, keyed
            ``'<receiver guid>_<source guid>'``: those stored under ``method.attribute``.
        """
        from compas_acoustics.simulation_methods import SimulationContext

        context = SimulationContext.from_scene(self)
        _log.info("Running %s at %s Hz on %d sources and %d receivers",
                  method.name, context.sampling_rate, len(context.sources), len(context.receivers))
        rirs = method.compute(context)
        self.populate_ir_graph(rirs, attribute=method.attribute)
        for attribute, responses in method.extra_responses.items():
            self.populate_ir_graph(responses, attribute=attribute)
        if method.output == "mono":
            self.populate_ir_graph(rirs, attribute="data")
        return rirs
