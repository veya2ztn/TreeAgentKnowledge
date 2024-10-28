import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import RoleType
from langgraph.graph import StateGraph, END

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    DISPATCH = "dispatch"      # For content placement decisions
    UPDATE = "update"         # For node content updates
    MERGE = "merge"          # For merging related nodes
    VALIDATE = "validate"    # For validating node relationships
    QUERY = "query"          # For information retrieval

@dataclass
class AgentMessage:
    msg_type: MessageType
    content: Dict
    sender: str
    receiver: str
    metadata: Dict = None

class NodeAgent(ChatAgent):
    """Agent representing a node in the knowledge tree"""
    
    def __init__(self, key: str, value: Any):
        # Initialize the base agent with appropriate system message
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
        self.message_queue: List[AgentMessage] = []
        self._logs: List[str] = []

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages and generate appropriate responses"""
        
        if message.msg_type == MessageType.DISPATCH:
            # Evaluate content relevance and decide where it belongs
            user_msg = BaseMessage.make_user_message(
                role_name="Dispatcher",
                content=f"""Evaluate if this content belongs to your node about {self.key}:
                Content: {message.content}
                
                Respond with one of:
                - RELEVANT: If it directly relates to your topic
                - LOWER: If it should go to a child node
                - HIGHER: If it belongs to your parent
                - UNRELATED: If it doesn't belong in your branch
                
                Explain your reasoning."""
            )
            response = self.step(user_msg)
            
            # Process the agent's decision
            decision = self._parse_relevance_decision(response.msg.content)
            return AgentMessage(
                msg_type=MessageType.DISPATCH,
                content={"decision": decision, "original_content": message.content},
                sender=self.key,
                receiver=message.sender
            )

    def _parse_relevance_decision(self, response: str) -> str:
        """Parse the agent's decision about content relevance"""
        # Implementation would extract the decision from the agent's response
        # This is a placeholder
        return "RELEVANT"

    def add_child(self, child: 'NodeAgent') -> None:
        """Add a child agent node"""
        child.parent = self
        self.children.append(child)
        
        # Notify both nodes of the relationship
        self._handle_relationship_update(child, "add")

    def _handle_relationship_update(self, other_node: 'NodeAgent', action: str) -> None:
        """Handle updates to node relationships"""
        if action == "add":
            message = f"New child node added: {other_node.key}"
        elif action == "remove":
            message = f"Child node removed: {other_node.key}"
        self._logs.append(message)

class TreeGraph(StateGraph):
    """Manages the flow of messages between node agents"""
    
    def __init__(self):
        super().__init__()
        self.root = NodeAgent("root", {})
        self.node_path_map = {}

    def build_graph(self):
        """Configure the message passing graph"""
        
        # Define the dispatch node for content placement
        self.add_node("dispatch", self._dispatch_content)
        
        # Define the update node for modifying tree structure
        self.add_node("update", self._update_tree)
        
        # Add edges for message flow
        self.add_edge("dispatch", "update")
        self.add_edge("update", END)

    def _dispatch_content(self, state: Dict) -> Dict:
        """Handle content dispatch through the tree"""
        content = state["content"]
        current_node = self.root
        path = []
        
        while True:
            # Create dispatch message
            message = AgentMessage(
                msg_type=MessageType.DISPATCH,
                content=content,
                sender="dispatcher",
                receiver=current_node.key
            )
            
            # Get node's decision
            response = current_node.process_message(message)
            decision = response.content["decision"]
            
            if decision == "RELEVANT":
                path.append(current_node.key)
                break
            elif decision == "LOWER" and current_node.children:
                # Continue search in children
                current_node = current_node.children[0]  # Need better child selection
                path.append(current_node.key)
            else:
                break
        
        return {"path": path, "content": content}

    def _update_tree(self, state: Dict) -> Dict:
        """Handle tree updates based on dispatch results"""
        path = state["path"]
        content = state["content"]
        
        if path:
            target_node = self.get_node(path)
            if target_node:
                # Update node with new content
                update_msg = AgentMessage(
                    msg_type=MessageType.UPDATE,
                    content=content,
                    sender="updater",
                    receiver=target_node.key
                )
                target_node.process_message(update_msg)
        
        return {"status": "complete"}

    def get_node(self, path: List[str]) -> Optional[NodeAgent]:
        """Get a node from the tree given a path"""
        current = self.root
        for key in path:
            found = False
            for child in current.children:
                if child.key == key:
                    current = child
                    found = True
                    break
            if not found:
                return None
        return current

class Tree:
    def __init__(self):
        self.graph = TreeGraph()
        self.graph.build_graph()

    def process_content(self, content: Dict):
        """Process new content through the agent graph"""
        app = self.graph.compile()
        result = app.invoke({"content": content})
        return result

    @classmethod
    def load_from_file(cls, file_path: str) -> 'Tree':
        """Load tree from JSON file"""
        with open(file_path, 'r') as file:
            data = json.load(file)
        tree = cls()
        tree._build_tree_from_dict(data)
        return tree

    def _build_tree_from_dict(self, data: Dict) -> None:
        """Build the agent tree from dictionary data"""
        def build_node(node_data: Dict, parent: NodeAgent):
            for key, value in node_data.items():
                if isinstance(value, dict):
                    node = NodeAgent(key, value)
                    parent.add_child(node)
                    build_node(value, node)

        build_node(data, self.graph.root)

# Example usage
if __name__ == "__main__":
    # Load the tree
    tree = Tree.load_from_file("TreeKnowledge.example.json")
    
    # Example content to process
    content = {
        "title": "New Research in Self-Supervised Learning",
        "abstract": "Recent advances in SSL techniques...",
        "keywords": ["machine learning", "computer vision"]
    }
    
    # Process the content
    result = tree.process_content(content)
    print(f"Content processed: {result}")
