from typing import Iterable
from typing import Union

from compas.data import Data
from compas.datastructures import Graph

from compas_acoustics.ir import ImpulseResponse
from compas_acoustics.receiver import Receiver
from compas_acoustics.source import Source

__all__ = ['Scene']


class Scene(Data):
    
    def __init__(self, name:str, description:str="", simulation_platform:str='pyroomacoustics'):
        super().__init__()
        self.name = name
        self.description = description
        self._acoustic_objects = []
        self._sources = []
        self._receivers = []
        self.acoustic_materials = {}
        self.ir_graph = Graph()
        
        
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
        return len(self._sources)
    
    
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
        return self._acoustic_materials
    
    
    @acoustic_materials.setter
    def acoustic_materials(self, acoustic_materials:dict):
        self._acoustic_materials = acoustic_materials          
    
    
    def iter_acoustic_objects(self):
        """Yield every leaf :class:`AcousticObject` in the scene, recursing into groups."""
        for acoustic_object in self.acoustic_objects:
            yield from acoustic_object.iter_acoustic_objects()
    
    
    @property
    def scene_materials(self):
        """The unique acoustic materials used in the scene, keyed by material name."""
        materials = {}
        for acoustic_object in self.iter_acoustic_objects():
            material = acoustic_object.material
            if material is not None:
                materials[material.name] = material
        return materials
    
    
    def print_scene_materials(self):
        print(f"Scene '{self.name}' materials:")
        for acoustic_object in self.iter_acoustic_objects():
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

        for acoustic_object in self.iter_acoustic_objects():
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

    
    def add_impulse_response(self, source: Source, receiver: Receiver, ir: ImpulseResponse) -> None:
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

        if self.ir_graph.has_edge((source_key, receiver_key)):
            self.ir_graph.delete_edge((source_key, receiver_key))

        self.ir_graph.add_edge(source_key, receiver_key, attr_dict={'data': ir})

    
    def populate_ir_graph(self, impulse_responses: Union[dict, Iterable[ImpulseResponse]]) -> None:
        self.clear_ir_edges()

        if isinstance(impulse_responses, dict):
            responses = impulse_responses.values()
        else:
            responses = impulse_responses

        for ir in responses:
            self.add_impulse_response(ir.source, ir.receiver, ir)

    
    def get_impulse_response(self, source: Source, receiver: Receiver) -> ImpulseResponse:
        source_key = self._ir_node_key(source)
        receiver_key = self._ir_node_key(receiver)

        if not self.ir_graph.has_node(source_key):
            raise KeyError(f"Source '{source.name}' is not registered in the IR graph.")
        if not self.ir_graph.has_node(receiver_key):
            raise KeyError(f"Receiver '{receiver.name}' is not registered in the IR graph.")
        if not self.ir_graph.has_edge((source_key, receiver_key)):
            raise KeyError(f"No impulse response found for source '{source.name}' and receiver '{receiver.name}'.")

        return self.ir_graph.edge_attributes((source_key, receiver_key))['data']
