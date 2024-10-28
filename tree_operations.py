import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import RoleType
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    DISPATCH = "dispatch"
    UPDATE = "update"
    MERGE = "merge"
    VALIDATE = "validate"
    QUERY = "query"

class NodeRelation(Enum):
    RELEVANT = "relevant"
    LOWER = "lower"
    HIGHER = "higher"
    UNRELATED = "unrelated"

@dataclass
class AgentMessage:
    msg_type: MessageType
    content: Dict
    sender: str
    receiver: str
    metadata: Dict = None

class NodeAgent(ChatAgent):
    def __init__(self, key: str, value: Any):
        system_message = BaseMessage(
            role_name="KnowledgeNode",
            role_type=RoleType.ASSISTANT,
            content=f"""You are a knowledge node agent responsible for managing content about: {key}
            Your responsibilities include:
            1. Evaluating content relevance to your topic
            2. Managing relationships with parent and child nodes
            3. Processing and responding to messages from other nodes
            4. Making decisions about content organization""",
        )
        super().__init__(system_message=system_message)
        
        self.key = key
        self.value = value
        self.parent = None
        self.children = []
        self.content_pool: List[Dict] = []  # Store relevant content
        self._logs: List[str] = []

    def generate_relevant_decision_via_LLM(self, content: Dict) -> NodeRelation:
        """Generate a decision about content relevance using LLM"""
        user_msg = BaseMessage.make_user_message(
            role_name="Dispatcher",
            content=f"""Evaluate if this content belongs to your node about {self.key}:
            Content: {content}
            
            Respond with one of:
            - RELEVANT: If it directly relates to your topic
            - LOWER: If it should go to a child node
            - HIGHER: If it belongs to your parent
            - UNRELATED: If it doesn't belong in your branch
            
            Explain your reasoning."""
        )
        response = self.step(user_msg)
        return self._parse_relevance_decision(response.msg.content)

    def process_message(self, message: AgentMessage) -> Tuple[NodeRelation, Optional[AgentMessage]]:
        """Process incoming messages and return decision and optional response"""
        logger.info(f"Node {self.key} processing message of type {message.msg_type}")
        
        if message.msg_type == MessageType.DISPATCH:
            decision = self.generate_relevant_decision_via_LLM(message.content)
            return decision, AgentMessage(
                msg_type=MessageType.DISPATCH,
                content={"decision": decision, "original_content": message.content},
                sender=self.key,
                receiver=message.sender
            )
        
        elif message.msg_type == MessageType.UPDATE:
            self.content_pool.append(message.content)
            logger.info(f"Node {self.key} updated with new content")
            return NodeRelation.RELEVANT, None

        return NodeRelation.UNRELATED, None

    def _parse_relevance_decision(self, response: str) -> NodeRelation:
        """Parse LLM response to determine content relevance"""
        response = response.upper()
        if "RELEVANT" in response:
            return NodeRelation.RELEVANT
        elif "LOWER" in response:
            return NodeRelation.LOWER
        elif "HIGHER" in response:
            return NodeRelation.HIGHER
        return NodeRelation.UNRELATED

    def add_child(self, child: 'NodeAgent') -> None:
        """Add a child node"""
        child.parent = self
        self.children.append(child)
        logger.info(f"Added child {child.key} to node {self.key}")
        self._logs.append(f"Added child: {child.key}")

class KnowledgeTree:
    """Manages the tree structure and content routing"""
    
    def __init__(self):
        self.root = NodeAgent("root", {})
        self._logs: List[str] = []

    def dispatch_content(self, content: Dict) -> List[str]:
        """Route content to appropriate node in the tree"""
        current_node = self.root
        path = []
        
        while True:
            message = AgentMessage(
                msg_type=MessageType.DISPATCH,
                content=content,
                sender="dispatcher",
                receiver=current_node.key
            )
            
            decision, _ = current_node.process_message(message)
            path.append(current_node.key)
            
            if decision == NodeRelation.RELEVANT:
                self._update_node_content(current_node, content)
                break
            elif decision == NodeRelation.LOWER and current_node.children:
                # Select most appropriate child node
                current_node = self._select_best_child(current_node, content)
            else:
                # Create new node if content doesn't fit existing structure
                if decision == NodeRelation.LOWER:
                    new_node = self._create_new_node(content)
                    current_node.add_child(new_node)
                    current_node = new_node
                    path.append(new_node.key)
                break
        
        return path

    def _update_node_content(self, node: NodeAgent, content: Dict) -> None:
        """Update node with new content"""
        update_msg = AgentMessage(
            msg_type=MessageType.UPDATE,
            content=content,
            sender="updater",
            receiver=node.key
        )
        node.process_message(update_msg)

    def _select_best_child(self, node: NodeAgent, content: Dict) -> NodeAgent:
        """Select the most appropriate child node for content"""
        best_child = node.children[0]
        best_score = float('-inf')
        
        for child in node.children:
            # Use LLM to score relevance to each child
            message = AgentMessage(
                msg_type=MessageType.QUERY,
                content=content,
                sender="selector",
                receiver=child.key
            )
            decision, _ = child.process_message(message)
            if decision == NodeRelation.RELEVANT:
                return child
            
        return best_child

    def _create_new_node(self, content: Dict) -> NodeAgent:
        """Create a new node based on content"""
        # Extract key concepts from content using LLM
        key = content.get("title", "New Node")  # Simplified for example
        return NodeAgent(key, content)

    @classmethod
    def load_from_file(cls, file_path: str) -> 'KnowledgeTree':
        """Load tree from JSON file"""
        with open(file_path, 'r') as file:
            data = json.load(file)
        tree = cls()
        tree._build_tree_from_dict(data)
        return tree

    def _build_tree_from_dict(self, data: Dict) -> None:
        """Build the tree structure from dictionary data"""
        def append_node_data_to_parent(node_data: Dict, parent: NodeAgent):
            keys = node_data.keys()
            metadata_allowed_keys = {"concept_key_word", "concept_abstract", "concept_represents",
                                     "title", "area", "idea_paradigm"}
            metadata_keys = [k for k in keys if k in metadata_allowed_keys]
            concept_keys  = [k for k in keys if k not in metadata_allowed_keys]
            
            assert len(concept_keys)==1, f"concept_keys: {concept_keys}"
            metadata = {k:node_data.pop(k) for k in metadata_keys}
            node_title=metadata.get("concept_key_word", None) or metadata.get("title", None)

            node = NodeAgent(key=node_title,value=metadata)
            
            parent.add_child(node)

            for concept_data in node_data.get(concept_keys[0], []):
                append_node_data_to_parent(concept_data, node)
                
        append_node_data_to_parent(data, self.root)

# Example usage
if __name__ == "__main__":
    # Load the tree
    tree = KnowledgeTree.load_from_file("TreeKnowledge.example.json")
    
    # # Example content to process
    # content = {
    #     "title": "New Research in Self-Supervised Learning",
    #     "abstract": "Recent advances in SSL techniques...",
    #     "keywords": ["machine learning", "computer vision"]
    # }
    
    # # Process the content
    # path = tree.dispatch_content(content)
    # print(f"Content placed in path: {' -> '.join(path)}")
