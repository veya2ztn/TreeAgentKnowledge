import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import RoleType,ModelType
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.configs import ChatGPTConfig
import logging
from numpy import isin
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Set, Literal
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from prompts.node_action.node_action import *
from prompts.node_action.concept_relevance import RelevanceCategory,DispatchDecision,PROMPT_for_concept_relevance
from prompts.node_action.action_decision_make_prompt import NodeActionDecision, PROMT_for_action_decision_make
from prompts.node_action.update_prompt_after_add_action import PROMT_after_add_action
from prompts.node_action.update_prompt_after_update_action import PROMPT_after_update_action

from node_base import NodeStructure

class MessageType(Enum):
    DISPATCH = "dispatch"
    UPDATE = "update"
    MERGE = "merge"
    VALIDATE = "validate"
    QUERY = "query"



@dataclass
class AgentMessage:
    msg_type: MessageType
    content: Dict
    sender:    str
    receiver: str
    metadata: Dict = None

with open("prompts/node_action/node_responsibility.txt", "r") as f:
    node_responsibility_prompt = f.read()



class NodeAgent(ChatAgent,NodeStructure):
    

    def __init__(self, key: str=None, value: Any=None, metadata: Dict=None):
        system_message = BaseMessage(
            role_name=f"Concept Builder in Tree Agent Knowledge System",
            role_type=RoleType.CRITIC,
            meta_dict=metadata,
            content=node_responsibility_prompt,
        )
        model_backend = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict=ChatGPTConfig().as_dict()|{"response_format":{ "type": "json_object" }},
        )

        super().__init__(system_message=system_message, model=model_backend)
        self.key: str = key
        self.value: Dict = value
        self.parent = None
        self.children = []
        self.content_pool: List[Any] = []  # Store relevant content
        self._logs: List[str] = []

    def build_relevant_decision_prompt(self, content: Dict)->BaseMessage:
        
        return BaseMessage.make_user_message(
            role_name="Dispatcher",
            content=PROMPT_for_concept_relevance.format(
                current_node_snapshot=self.text_snapshot(),
                new_incoming_content=content,
                children_keys="\n"+"\n".join(["- "+k for k in self.children_keys])+"\n",  
                )
        )

    def generate_relevant_decision_via_simple_LLM(self, content: Dict) -> DispatchDecision:
        """
        Generate a decision about content relevance using LLM
        The simple means we have not hard structured the output in json format. Usually, structure json can be required by 
        - `json_decoder` in sglang/outlines
        - set `json_format` in openai api
        """
        user_msg = self.build_relevant_decision_prompt(content)
        response = self.step(user_msg, output_schema=DispatchDecision)
        response = eval(response.msg.content) ## <-- make sure the response is a valid json
        return DispatchDecision(**response)
    
    def generate_relevant_decision_via_json_output_LLM(self, content: Dict) -> DispatchDecision:
        """
        Generate a decision about content relevance using LLM
        The simple means we have not hard structured the output in json format. Usually, structure json can be required by 
        - `json_decoder` in sglang/outlines
        - set `json_format` in openai api
        """
        user_msg = self.build_relevant_decision_prompt(content)
        response = self.step(user_msg)
        response = eval(response.msg.content) ## <-- make sure the response is a valid json
        def basement_polish(string):
            return string.strip().strip('[]')
        response = {k:basement_polish(v) for k,v in response.items()}
        return DispatchDecision(**response)
    
    def make_dispatch_decision(self, content: Dict) -> DispatchDecision:
        """Make dispatch decision for content"""
        # message = AgentMessage(
        #         msg_type=MessageType.DISPATCH,
        #         content=content,
        #         sender="dispatcher",
        #         receiver=self.key
        #     )
        #decision = self.generate_relevant_decision_via_simple_LLM(content)
        decision = self.generate_relevant_decision_via_json_output_LLM(content)
        # next_message=AgentMessage(
        #         msg_type=MessageType.DISPATCH,
        #         content={"decision": decision, "original_content": message.content},
        #         sender=self.key,
        #         receiver=message.sender
        #     )
        return decision
    


    def modify_the_node_information_via_content(self) -> None:
        """Modify the node information via content"""
        if len(self.content_pool) == 0:
            return

    def generate_node_action_decision_via_simple_LLM(self, content: List[Any]) -> NodeActionDecision:
        """Generate a decision about node action using LLM"""
        prompt   = self.build_node_action_decision_prompt(content)
        class NodeActionSchema(BaseModel):
            # for single node action, since we will do dispatch first, 
            # thus the appended data must belong to current node which implies the action must be ADD/MERGE/UPDATE
            action: Literal["ADD", "MERGE", 
                            #"REARRANGE", "UPDATE", "NO_ACTION", "DELETE"
                            ]
            details: Dict = Field(description="Action-specific details")
            reasoning: str = Field(description="Detailed explanation of why this action was chosen")

            @property
            def action_details(self) -> BaseModel:
                """Returns the appropriate details model based on the action type"""
                details_models = {
                    "ADD": AddActionDetails,
                    "MERGE": MergeActionDetails,
                    # "DELETE": DeleteActionDetails,
                    # "REARRANGE": RearrangeActionDetails,
                    # "UPDATE": UpdateActionDetails,
                    # "NO_ACTION": NoActionDetails,
                }
                return details_models[self.action](**self.details)
        response = self.step(prompt, output_schema=NodeActionSchema)
        return eval(response.msg.content)

    def generate_node_action_decision_via_json_output_LLM(self, content: List[Any]) -> NodeActionDecision:
        """Generate a decision about node action using LLM"""
        prompt   = self.build_node_action_decision_prompt(content)
        response = self.step(prompt)
        response = eval(response.msg.content)
        return NodeActionDecision.load_from_dict(response)

    def build_node_action_decision_prompt(self, extra_content) -> BaseMessage:
        
        return BaseMessage.make_user_message(
            role_name="NodeAnalyzer",
            content=PROMT_for_action_decision_make.format(
                current_node_snapshot=self.text_snapshot(),
                extra_content_snapshot=extra_content,
            )
        )

    def execute_action(self, action_decision: NodeActionDecision) -> None:
        """Execute the determined action"""
        logger.info(f"Executing {action_decision.action} on node {self.key}")
        
        if isinstance(action_decision, AddAction):
            self._execute_add_action(action_decision)
        elif isinstance(action_decision, MergeAction):
            self._execute_merge_action(action_decision)
        elif isinstance(action_decision, UpdateAction):
            self._execute_update_action(action_decision)

        # Clear processed content
        self.content_pool = []

    def make_node_action_decision_via_content(self, content: List[Any]) -> NodeActionDecision:
        """Make node action decision"""
        return self.generate_node_action_decision_via_json_output_LLM(content)

    def _execute_add_action(self, action: AddAction) -> None:
        """Add new child nodes"""
        details = action.details
        new_node = NodeAgent(
            key=details.new_concept_key_word,
            value={"concept_key_word":details.new_concept_key_word,
                   "concept_abstract":details.new_concept_abstract}
        )
        
        prompt = BaseMessage.make_user_message(
            role_name="NodeEvolution",
            content=PROMT_after_add_action.format(
                current_node_snapshot=self.text_snapshot(),
                new_node_snapshot    =new_node.text_snapshot(),
            )
        )
        self.add_child(new_node)
        response = self.step(prompt)
        self_update_pool = eval(response.msg.content)
        self.key = self_update_pool['new_concept_key_word']
        self.value['concept_abstract'] = self_update_pool['new_concept_abstract']
        self.value['concept_key_word'] = self_update_pool['new_concept_key_word']

   
    def _execute_delete_action(self, details: Dict) -> None:
        """Delete specified child nodes"""
        raise NotImplementedError("This method should be implemented by the subclass")

    def _execute_merge_action(self, details: MergeActionDetails) -> None:
        """Merge specified child nodes"""
        raise NotImplementedError("This method should be implemented by the subclass")
        for nodes_to_merge in details.get("nodes_to_merge", []):
            if len(nodes_to_merge) < 2:
                continue
                
            # Get nodes to merge
            nodes = [self.children_map[key] for key in nodes_to_merge if key in self.children_map]
            if len(nodes) < 2:
                continue

            # Create merged node
            merged_key = f"Merged_{nodes_to_merge[0]}"
            merged_abstract = self._generate_merged_abstract(nodes)
            merged_node = NodeAgent(
                key=merged_key,
                value={"concept_abstract": merged_abstract}
            )

            # Transfer children from merged nodes to new node
            for node in nodes:
                for child in node.children:
                    merged_node.add_child(child)

            # Remove old nodes and add merged node
            for key in nodes_to_merge:
                if key in self.children_map:
                    del self.children_map[key]
            self.add_child(merged_node)

    def _execute_rearrange_action(self, details: Dict) -> None:
        """Rearrange child nodes in specified order"""
        raise NotImplementedError("This method should be implemented by the subclass")
        new_order = details.get("new_order", [])
        if not new_order:
            return

        # Create new children map in specified order
        new_children_map = {}
        for key in new_order:
            if key in self.children_map:
                new_children_map[key] = self.children_map[key]
        
        self.children_map = new_children_map

    def _execute_update_action(self, action: UpdateAction) -> None:
        """Update node's own information"""
        assert len(self.content_pool) == 1, "Current please use only one content to update"
        new_content = self.content_pool[0]
        prompt = BaseMessage.make_user_message(
            role_name="NodeEvolution",
            content=PROMPT_after_update_action.format(
                current_node_snapshot=self.text_snapshot(),
                new_content    =new_content,
            )
        )
        response = self.step(prompt)
        self_update_pool = eval(response.msg.content)
        self.key = self_update_pool['new_concept_key_word']
        self.value['concept_abstract'] = self_update_pool['new_concept_abstract']
        self.value['concept_key_word'] = self_update_pool['new_concept_key_word']

    def _generate_merged_abstract(self, nodes: List['NodeAgent']) -> str:
        """Generate abstract for merged node using LLM"""
        prompt = f"""
        Generate a comprehensive abstract that combines the concepts from these nodes:
        {[f"- {node.key}: {node.concept_abstract}" for node in nodes]}
        """
        response = self.step(
            BaseMessage.make_user_message(
                role_name="Merger",
                content=prompt
            )
        )
        return response.msg.content

    

class NodeTree(NodeAgent):   
    def dispatch_content(self, content: Dict, start_node: Optional[NodeAgent] =None) -> List[str]:
        """Route content to appropriate node in the tree"""
        current_node = start_node or self
        path = []
        
        while True:
            decision:DispatchDecision = current_node.make_dispatch_decision(content)
            print(decision)
            path.append(current_node.key)
            if decision.decision == RelevanceCategory.RELEVANT:
                self._update_node_candidate_content(current_node, content)
                break
            elif decision.decision == RelevanceCategory.LOWER and current_node.children:
                # Select most appropriate child node
                # current_node = self._select_best_child(current_node, content)
                current_node = current_node.get_child_via_key(decision.next_position)
            else:
                # Create new node if content doesn't fit existing structure
                if decision.decision == RelevanceCategory.LOWER:
                    self._update_node_candidate_content(current_node, content)
                    # print("=== create new node, then we need genereate the concept_key_word and concept_abstract ===")
                    # #new_node = self._create_new_leaf_node_based_on_content(content)
                    # new_node_key = decision.next_position
                    # new_node_abstract = decision.reasoning
                    # new_node = NodeAgent(key=new_node_key, 
                    #                      value={"concept_key_word":new_node_key, "concept_abstract":new_node_abstract})
                    # current_node.add_child(new_node)
                    # current_node = new_node
                    path.append("New Child")
                break
        
        return path

    def _update_node_candidate_content(self, node: NodeAgent, content: Dict) -> None:
        """Update node with new content"""
        # update_msg = AgentMessage(
        #     msg_type=MessageType.UPDATE,
        #     content=content,
        #     sender="updater",
        #     receiver=node.key
        # )
        # node.process_message(update_msg)
        node.content_pool.append(content)


    def _create_new_leaf_node_based_on_content(self, content: Dict) -> NodeAgent:
        """Create a new node based on content"""
        raise NotImplementedError("This method should be implemented by the subclass")
        return NodeAgent(key, content)
     
    def back_propagate_content(self, content: Dict, path: List[str]) -> None:
        """Back propagate content to the root node"""
        current_node = self.root
        for node_key in path:
            current_node = current_node.get_child_via_key(node_key)
            self._update_node_candidate_content(current_node, content)

    def backpropagate(self, start_node: Optional[NodeAgent] = None) -> None:
        start_node = start_node or self

        # Process leaves first (post-order traversal)
        for child in start_node.children:
            self.backpropagate(child)

        # Process current node
        for content in start_node.content_pool:
            action_result = start_node.make_node_action_decision_via_content(content)
            start_node.execute_action(action_result)
        
        #self._logs.append(f"Node {start_node.key}: {action_result.action}")

    def __getitem__(self, key: str) -> NodeAgent:
        return self.get_child_via_key(key)
    
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
        def _traverse_tree(node: NodeAgent, depth: int = 0) -> str:
            """Helper method to traverse the tree and create a snapshot."""
            if depth == 0:
                snapshot = f"{node.key}\n"
            else:
                snapshot = "    " * (depth-1) + "|--- " + node.key + "\n"
            for child in node.children:
                snapshot += _traverse_tree(child, depth + 1)
            return snapshot

        return _traverse_tree(self)
        
# Example usage
if __name__ == "__main__":
    # Load the tree
    tree = KnowledgeTree.load_from_file("TreeKnowledge.example.json")
    
    # Example content to process
    # content = {
    #     "title": "New Research in Self-Supervised Learning",
    #     "abstract": "Recent advances in SSL techniques...",
    #     "keywords": ["machine learning", "computer vision"]
    # }
    
    # Process the content
    #path = tree.dispatch_content(content)
    #print(f"Content placed in path: {' -> '.join(path)}")
