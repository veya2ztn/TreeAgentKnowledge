from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Literal, Any, Tuple
import json
import datetime
from rich import print as rprint
from rich.tree import Tree as RichTree
from rich.text import Text
from colorama import init, Fore, Style
import colorsys


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
    represented_pool: List[Dict] = field(default_factory=list)
    is_content_updated: bool = False
    evolved_count: int = 0
    is_merged_after_add: bool = True
    
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
    
    @property
    def child_snapshot_information_before_self_evolution(self):
        """
        This property is used to get the snapshot information of the children nodes. Please use follow format:
        - [Child_0_key]:[Child_0_description]
        - [Child_1_key]:[Child_1_description]
        ....
        """
        #return "\n".join([f"Concept Key: {child.key}, Concept Description: {child.concept_abstract}" for child in self.children])
        return "\n".join([f"    - [{child.key}]:[{child.concept_abstract}]" for child in self.children if child.evolved_count == -1])
    
    def text_snapshot_before_self_evolution(self):
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
{self.child_snapshot_information_before_self_evolution}
        """.strip()
        return output
    
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
        #self._logs.append(f"Added child: {child.key}")

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
        metadata_allowed_keys = {"concept_key_word", "concept_abstract", 
                                    "title", "area", "idea_paradigm"}
        represented_pool_keys = {"concept_represents"}
        metadata_keys = [k for k in keys if k in metadata_allowed_keys]
        concept_keys  = [k for k in keys if k not in metadata_allowed_keys|represented_pool_keys]
        
        represented_paper_id_address = node_data.get("concept_represents", [])
        represented_pool = []
        for paper_id_address in represented_paper_id_address:
            represented_pool.append({
                "content_metadata": {
                    # "added_time": datetime.now().isoformat(),
                    "content_id_address": paper_id_address, ## currently we save content here, but it is better to save the content id address in the future
                },
            "is_used_for_update": True,
            })
        assert len(concept_keys)==1, f"concept_keys: {concept_keys}"
        metadata = {k:node_data.pop(k) for k in metadata_keys}
        node_title=metadata.get("concept_key_word", None) or metadata.get("title", None)

        node = cls(key=node_title,value=metadata)
        node.represented_pool = represented_pool
        parent.add_child(node)

        for concept_data in node_data.get(concept_keys[0], []):
            cls.append_node_data_to_parent(concept_data, node)

    def tree_snapshot_colored(self, output_type='both',show_represent=True):
        """
        Returns a colored tree snapshot where node colors are based on evolved_count.
        Adds boxes around nodes where is_content_updated=True.

        Args:
            output_type (str): 'jupyter', 'terminal', or 'both'
        """
        init()  # Initialize colorama for terminal colors

        def get_normalized_color(count: int, min_count: int, max_count: int) -> Tuple[str, tuple]:
            """Returns both terminal and RGB colors"""
            if count == 0:  # New nodes
                return Fore.LIGHTRED_EX, (255, 100, 100)

            if max_count == min_count:
                return Fore.BLACK, (0, 0, 0)

            # Normalize count to 0-1 range (inverted so larger count = darker)
            normalized = 1.0 - (count - min_count) / (max_count - min_count)

            # Generate RGB color - from black (0,0,0) to light blue (173, 216, 230)
            light_blue = (173, 216, 230)
            rgb = tuple(int(c * normalized) for c in light_blue)

            # Map to terminal colors
            if normalized > 0.8:
                term_color = Fore.LIGHTBLUE_EX
            elif normalized > 0.6:
                term_color = Fore.BLUE
            elif normalized > 0.4:
                term_color = Fore.LIGHTBLACK_EX
            elif normalized > 0.2:
                term_color = Fore.BLACK
            else:
                term_color = Fore.BLACK

            return term_color, rgb

        def get_min_max_counts(node: NodeStructure) -> Tuple[int, int]:
            counts = [node.evolved_count]
            for child in node.children:
                child_min, child_max = get_min_max_counts(child)
                counts.extend([child_min, child_max])
            return min(c for c in counts if c >= 0), max(counts)

        def get_plain_text_from_node(node: NodeStructure, show_represent=True)->str:
            text = node.key
            if show_represent:
                represents = [str(t['content_metadata']['content_id_address']) for t in node.represented_pool]
                represents = ",".join(represents)
                if represents:
                    text = f"{text}[{represents}]"
            if node.is_content_updated:
                text = f"{text}*"  # Add box brackets
            return text
        
        def build_rich_tree(node: NodeStructure, min_count: int, max_count: int, show_represent=True) -> RichTree:
            """Builds a rich tree for Jupyter display"""
            _, rgb = get_normalized_color(node.evolved_count+0.5*int(node.is_content_updated), 
                                        min_count, max_count)

            # Add box decoration for merged nodes
            text = get_plain_text_from_node(node,show_represent)

            tree = RichTree(Text(text, style=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"))

            for child in node.children:
                tree.add(build_rich_tree(child, min_count, max_count,show_represent))
            return tree

        def build_terminal_tree(node: NodeStructure, depth: int, min_count: int, max_count: int,show_represent=True) -> str:
            """Builds a colored string tree for terminal display"""
            term_color, _ = get_normalized_color(node.evolved_count+0.5*int(node.is_content_updated), 
                                                min_count, max_count)

            # Add box decoration for merged nodes
            text = get_plain_text_from_node(node,show_represent)
            
            if depth == 0:
                snapshot = f"{term_color}{text}{Style.RESET_ALL}\n"
            else:
                indent = "    " * (depth-1)
                snapshot = f"{indent}|--- {term_color}{text}{Style.RESET_ALL}\n"

            for child in node.children:
                snapshot += build_terminal_tree(child, depth + 1, min_count, max_count,show_represent)
            return snapshot

        min_count, max_count = get_min_max_counts(self)

        if output_type == 'jupyter':
            return build_rich_tree(self, min_count, max_count,show_represent=show_represent)
        elif output_type == 'terminal':
            return build_terminal_tree(self, 0, min_count, max_count,show_represent=show_represent)
        else:  # both
            rich_tree = build_rich_tree(self, min_count, max_count,show_represent)
            term_tree = build_terminal_tree(self, 0, min_count, max_count,show_represent)
            return rich_tree, term_tree 
    
    def tree_snapshot(self) -> str:
        """
        This method returns the tree snapshot in the form of a string. 
        It will traverse the tree in a depth-first manner and view the tree as a tree structure.
        For example:
        Literature Research Report
        |--- Self-Supervised Learning (SSL) Techniques
            |--- Contrastive Learning
            |--- Masked Image Modeling (MIM)
        |--- Object Detection Innovations
        |--- Multi-Modal Representations and Learning
        """
        def _traverse_tree(node: NodeStructure, depth: int = 0) -> str:
            """Helper method to traverse the tree and create a snapshot."""
            if depth == 0:
                snapshot = f"{node.key}\n"
            else:
                snapshot = "    " * (depth-1) + "|--- " + node.key + "\n"
            for child in node.children:
                snapshot += _traverse_tree(child, depth + 1)
            return snapshot

        return _traverse_tree(self)

    def export_to_json(self, indent: int = 4) -> str:
        """
        Export the tree structure to a JSON string.

        Args:
            indent (int): Number of spaces for JSON indentation

        Returns:
            str: JSON string representation of the tree
        """
        def _build_dict(node: NodeStructure,depth=0) -> Dict:
            """Helper method to build dictionary representation of the tree"""
            # Base node attributes
            represent_list = node.represented_pool
            represents = [t['content_metadata']['content_id_address'] for t in represent_list]
            node_dict = {
                "concept_key_word": node.key,
                "concept_abstract": node.concept_abstract,
                "concept_represents":represents ,
            }

            # Handle children based on level
            if node.children:
                # Determine the sub-concept level
                sub_concept_key = f"sub_concept_{depth + 1}"
                node_dict[sub_concept_key] = [_build_dict(child,depth+1) for child in node.children]
            else:
                # Leaf nodes should have an empty sub_concept array
                sub_concept_key = f"sub_concept_{depth + 1}"
                node_dict[sub_concept_key] = []

            return node_dict

        # Build the full JSON structure
        json_dict = {
            "title": self.value.get("title", "Untitled"),
            "area": self.value.get("area", ""),
            "idea_paradigm": self.value.get("idea_paradigm", ""),
            "root_level_concept": [_build_dict(child,0) for child in self.children]
        }

        # Convert to JSON string with specified indentation
        return json.dumps(json_dict, indent=indent)

    def save_to_json(self, filepath: str, indent: int = 4) -> None:
        """
        Save the tree structure to a JSON file.

        Args:
            filepath (str): Path to the output JSON file
            indent (int): Number of spaces for JSON indentation
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.export_to_json(indent=indent))

if __name__ == "__main__":
    print(NodeStructure())
    print(NodeStructure.load_from_file("TreeKnowledge.example.json").text_snapshot())