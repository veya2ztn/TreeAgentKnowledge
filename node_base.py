from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Literal
import json

@dataclass
class Node:
    """
    This class define the converter between json format and node structure
    """
    key: str = None
    value: Dict = field(default_factory=dict)
    parent = None
    children: List = field(default_factory=list)
    _logs: List[str] = field(default_factory=list)

    @property
    def children_map(self):
        return {child.key: child for child in self.children}

    @property
    def children_keys(self):
        return [child.key for child in self.children]
    
    
    @property
    def concept_abstract(self):
        return self.value.get("concept_abstract", "") or self.value.get("scheme", "") or self.value.get("idea_paradigm", "") 


class NodeStructure(Node):
    
    @property
    def child_snapshot_information(self):
        """
        This property is used to get the snapshot information of the children nodes. Please use follow format:
        - [Child_0_key]:[Child_0_description]
        - [Child_1_key]:[Child_1_description]
        ....
        """
        #return "\n".join([f"Concept Key: {child.key}, Concept Description: {child.concept_abstract}" for child in self.children])
        return "\n".join([f"    - [{child.key}]:[{child.concept_abstract}]" for child in self.children])
    
    
    
    def text_snapshot(self):
        """
        Please use follow format to generate the text snapshot of the node and its children
        ```
        Concept Key: {self.key}, Concept Description: {self.concept_abstract}.
        Sub Concepts:
        - [Child_0_key]:[Child_0_description]
        - [Child_1_key]:[Child_1_description]
        ...
        ```
        """
        output = f"""
Concept Key: {self.key}
Concept Description: {self.concept_abstract}
Sub Concepts:
{self.child_snapshot_information}
        """.strip()
        return output

    def add_child(self, child:Node) -> None:
        """Add a child node"""
        child.parent = self
        self.children.append(child)
        #self.children_map[child.key] = child
        #logger.info(f"Added child {child.key} to node {self.key}")
        self._logs.append(f"Added child: {child.key}")

    def get_child_via_key(self, key: str)->Node:
        """Get a child node via key"""
        for child in self.children:
            if child.key == key:
                return child
        raise ValueError(f"Child with key {key} not found, available keys: {self.children_keys}")

    @classmethod
    def load_from_dict(cls, data: Dict) -> Node:
        """Load tree from dictionary data"""
        parent = cls()
        cls.append_node_data_to_parent(data,parent)
        return parent.children[0]
    
    @classmethod
    def load_from_file(cls, file_path: str) -> Node:
        """Load tree from JSON file"""
        with open(file_path, 'r') as file:
            data = json.load(file)
        return cls.load_from_dict(data)
    
    def __repr__(self):
        return self.text_snapshot()
    
    @classmethod
    def append_node_data_to_parent(cls, node_data: Dict, parent: Node):
        keys = node_data.keys()
        metadata_allowed_keys = {"concept_key_word", "concept_abstract", "concept_represents",
                                    "title", "area", "idea_paradigm"}
        metadata_keys = [k for k in keys if k in metadata_allowed_keys]
        concept_keys  = [k for k in keys if k not in metadata_allowed_keys]
        
        assert len(concept_keys)==1, f"concept_keys: {concept_keys}"
        metadata = {k:node_data.pop(k) for k in metadata_keys}
        node_title=metadata.get("concept_key_word", None) or metadata.get("title", None)

        node = cls(key=node_title,value=metadata)
        
        parent.add_child(node)

        for concept_data in node_data.get(concept_keys[0], []):
            cls.append_node_data_to_parent(concept_data, node)

        
if __name__ == "__main__":
    print(NodeStructure())
    print(NodeStructure.load_from_file("TreeKnowledge.example.json").text_snapshot())