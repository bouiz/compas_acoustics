# from compas_viewer import Viewer # Does not work in Rhino
# from compas_viewer.scene import Tag # Does not work in Rhino
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
    
    
    @property
    def scene_surface_groups(self):
        surface_groups = []
        for acoustic_geometry in self.acoustic_objects:
            groups = acoustic_geometry.surface_groups.keys()
            surface_groups.extend(groups)
        return surface_groups
    
    
    def print_scene_surface_groups(self):
        print(f"Scene '{self.name}' surface groups:")
        for acoustic_geometry in self.acoustic_objects:
            print(f"Acoustic Object '{acoustic_geometry.name}':")
            for group_name in acoustic_geometry.surface_groups.keys():
                print(f"  {group_name}")
                
    
    def display_surface_groups(self):
        viewer = Viewer()
        
        for acoustic_geometry in self.acoustic_objects:
            group_colors = acoustic_geometry.surface_group_colors
            
            for fkey in acoustic_geometry.acoustic_geometry.faces():
                f_centroid = acoustic_geometry.acoustic_geometry.face_centroid(fkey)
                surface_group = acoustic_geometry.acoustic_geometry.face_attribute(fkey, 'surface_group')
                tag = Tag(surface_group, f_centroid, height=30, color=group_colors[surface_group])
                viewer.scene.add(tag, name=f"tag_{fkey}")
            
            viewer.scene.add(acoustic_geometry.acoustic_geometry, name=acoustic_geometry.name, show_points=True, facecolor=acoustic_geometry.face_colors)
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
