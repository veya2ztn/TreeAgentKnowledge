import json
from typing import Dict, List, Any, Optional, Tuple, Set, Coroutine
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
import asyncio
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

from prompts.node_action.node_action import *
from prompts.node_action.concept_relevance import RelevanceCategory,DispatchDecision,PROMPT_for_concept_relevance
from prompts.node_action.action_decision_make_prompt import NodeActionDecision, PROMT_for_action_decision_make
from prompts.node_action.update_prompt_after_add_action import PROMT_after_add_action
from prompts.node_action.update_prompt_after_update_action import PROMPT_after_update_action

from node_base import NodeStructure

from sglang.srt.constrained import build_regex_from_object
import sglang as sgl
import os

class MessageType(Enum):
    DISPATCH = "dispatch"
    UPDATE = "update"
    MERGE = "merge"
    VALIDATE = "validate"
    QUERY = "query"

@dataclass
class Content:
    id_address: str
    content: Dict

    def __str__(self):
        return f"Paper {self.id_address}: \n{self.content}"
    
@dataclass
class AgentMessage:
    msg_type: MessageType
    content: Dict
    sender:    str
    receiver: str
    metadata: Dict = None

with open("prompts/node_action/node_responsibility.txt", "r") as f:
    node_responsibility_prompt = f.read()



class NodeAgent(NodeStructure):
    

    def __init__(self, key: str=None, value: Any=None, metadata: Dict=None, is_merged_after_add: bool=True):
        super().__init__()
        
        self.key: str = key
        self.value: Dict = value
        self.parent = None
        self.children = []
        self.content_pool: List[Any] = []  # Store relevant content
        self._logs: List[str] = []
        self.executed_action_history: List = []
        self.is_merged_after_add: int = is_merged_after_add
        self.represented_pool: List[Any] = []
        self._step_semaphore = asyncio.Semaphore(10)
        self.evolved_count = 0
        self.is_content_updated = False
        self.build_backend_agent('auto')
        
    def build_backend_agent(self, provider='auto'):
        system_message = BaseMessage(
                role_name=f"Concept Builder in Tree Agent Knowledge System",
                role_type=RoleType.CRITIC,
                meta_dict={},
                content=node_responsibility_prompt,
            )
        base_url = os.environ.get("OPENAI_API_BASE_URL")
        api_key  = os.environ.get("OPENAI_API_KEY")
        if provider == 'auto':
            if 'api' in base_url:
                provider = self.provider = 'openai'
            else:
                provider = self.provider = 'sglang'
        else:
            self.provider=provider
        if provider == 'openai':
            if 'deepseek' in base_url:
                model_platform = ModelPlatformType.OPENAI_COMPATIBILITY_MODEL
                model_type = "deepseek-chat"
                max_tokens = 8192
            else:
                model_platform = ModelPlatformType.OPENAI
                model_type = ModelType.GPT_4O_MINI
                max_tokens = 16384
            self.relevant_decision_model = self.action_decision_model = self.evolved_model  = ChatAgent(
                    system_message=system_message, 
                    token_limit=100000,
                    message_window_size=1, ### <--- this is important, since we dont need the memory of the agent
                    model=ModelFactory.create(
                        model_platform=model_platform,
                        model_type=model_type,
                        url=base_url,
                        api_key=api_key,
                        model_config_dict=ChatGPTConfig().as_dict()|{'temperature': 0.8, "max_tokens":max_tokens,"response_format":{ "type": "json_object" }},
                )
            )
            
        
        elif provider == 'sglang':
            # relevance_regex = (
            #     r"""\{\n"""
            #     + r"""    "decision": "(RELEVANT|LOWER|HIGHER|UNRELATED)",\n"""
            #     + r"""    "next_position": "[\w\d\s]{1,128}",\n"""
            #     + r"""    "reasoning": "[\w\d\s]{128,512}"\n"""
            #     + r"""\}"""
            # )
    
            sgl.set_default_backend(sgl.RuntimeEndpoint(base_url.replace('/v1','')))
            max_tokens = 12800
            relevance_regex = build_regex_from_object(DispatchDecision)
            self.relevant_decision_model = ChatAgent(
                system_message=system_message, 
                token_limit=100000,
                message_window_size=1, ### <--- this is important, since we dont need the memory of the agent ### notice, this will erase the system prompt
                model=ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI_COMPATIBILITY_MODEL,
                    model_type="nvidia/llama-3.1-nemotron-70b-instruct",
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    url=base_url,
                    model_config_dict=ChatGPTConfig().as_dict()|{"max_tokens":max_tokens,"extra_body":{'regex':relevance_regex}},
                )
            )
            class Actions(str, Enum):
                Add = "Add"
                Update = "UPDATE"

            class TemplateActionDetail(BaseModel):
                new_concept_abstract: Optional[str]    = Field(default="this field is NOT Used", description="Updated abstract for the current node")
                new_concept_key_word: Optional[str]    = Field(default="this field is NOT Used", description="Updated key word for the current node")
            
            class NodeActionTemplate(BaseModel):
                action: Actions
                details: Dict[str,str] #TemplateActionDetail
                reason:str

            action_regex = build_regex_from_object(NodeActionTemplate)
            self.action_decision_model = ChatAgent(
                system_message=system_message, 
                model=ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI_COMPATIBILITY_MODEL,
                    model_type="nvidia/llama-3.1-nemotron-70b-instruct",
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    url=base_url,
                    model_config_dict=ChatGPTConfig().as_dict()|{"max_tokens":max_tokens,"extra_body":{'regex':action_regex}},
                )
            )


            evolved_regex = (
                r"""\{\n"""
                + r"""    "new_concept_key_word": "[\w\d\s]{6,32}",\n"""
                + r"""    "new_concept_abstract": "[\w\d\s]{128,512}"\n"""
                + r"""    "reason": "[\w\d\s]{128,512}"\n"""
                + r"""\}"""
            )
            self.evolved_model = ChatAgent(
                system_message=system_message, 
                model=ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI_COMPATIBILITY_MODEL,
                    model_type="nvidia/llama-3.1-nemotron-70b-instruct",
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    url=base_url,
                    model_config_dict=ChatGPTConfig().as_dict()|{"max_tokens":max_tokens,"extra_body":{'regex':evolved_regex}},
                )
            )

    def step(self, stage, user_msg:BaseMessage, output_schema=None):
        if stage == 'evolved':
            backend = self.evolved_model
        elif stage == 'action':
            backend = self.action_decision_model
        elif stage == 'relevant':
            backend = self.relevant_decision_model
        else:
            raise ValueError(f"Invalid stage: {stage}")
        

        if self.provider == 'sglang':
            # if stage == 'relevant':
            #     regex = build_regex_from_object(DispatchDecision.build_with_children_name(self.children_keys)) ### please use fixed regex as the backend always compile a new regex
            # else:
            regex = backend.model_config_dict['extra_body']['regex']
            message= user_msg.content  + "\n" + "The JSON output is:\n"
            @sgl.function
            def sglang_step(s):
                s += message
                s += sgl.gen("json_output", max_tokens=2048, regex=regex)
            max_retries = 3
            for _ in range(max_retries):
                try:
                    state = sglang_step.run()
                    output=state.text()
                    output=output[len(message):]
                    output_dict = json.loads(output)
                    return output_dict
                except Exception as e:
                    print(f"Error in parsing JSON: {output}")
            raise ValueError("Max retries reached. Unable to parse JSON.")
        else:
            response= backend.step(user_msg, output_schema)
            return eval(response.msg.content.replace("null","None"))
        
    async def step_with_semaphore(self, stage, user_msg, output_schema=None):
        """Wrapper to make step() async and use semaphore for rate limiting"""
        async with self._step_semaphore:
            # Convert the synchronous step() to async using asyncio.to_thread
            response = await asyncio.to_thread(self.step, stage, user_msg, output_schema)
            return response #eval(response.msg.content)
    

    def self_update(self, self_update_pool):
        old_concept_key_word = self.value.get('concept_key_word',"")
        old_concept_abstract = self.value.get('concept_abstract',"")
        self.key = self_update_pool['new_concept_key_word']
        self.value['concept_abstract'] = self_update_pool['new_concept_abstract']
        self.value['concept_key_word'] = self_update_pool['new_concept_key_word']
        self._logs.append({
            "parent": self.key,
            "action": "update",
            "old_concept_key_word": old_concept_key_word,
            "old_concept_abstract": old_concept_abstract,
            "new_concept_key_word": self_update_pool['new_concept_abstract'],
            "new_concept_abstract": self_update_pool['new_concept_key_word'],
            "reason": self_update_pool.get('reason',""),
            "timestamp": datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
        })
    async def self_evolve(self) -> None:
        """Async version of self evolution"""
        if len(self.executed_action_history) == 0:
            return
        
        # Handle add actions
        old_node_snapshot = self.text_snapshot_before_self_evolution
        # add_action_list = [action for action in self.executed_action_history if isinstance(action, AddAction)]
        # add_node_list   = [self[action.details.new_concept_key_word] for action in add_action_list]
        add_node_list = [child for child in self.children if not child.is_merged_after_add]
        new_node_snapshot = "\n".join([node.text_snapshot() for node in add_node_list])
        
        prompt = BaseMessage.make_user_message(
            role_name="NodeEvolution",
            content=PROMT_after_add_action.format(
                current_node_snapshot=old_node_snapshot,
                new_node_snapshot=new_node_snapshot,
            )
        )
        
        self_update_pool = await self.step_with_semaphore("evolved", prompt)
        self.self_update(self_update_pool)
        
        # Update evolved counts
        for node in add_node_list:
            node.is_merged_after_add = True

        # Handle update actions
        new_content = "\n".join([
            str(content_object["content_metadata"]["content_id_address"] )
            for content_object in self.represented_pool 
            if not content_object["is_used_for_update"]
        ])
        
        if new_content:  # Only make LLM call if there's content to update
            prompt = BaseMessage.make_user_message(
                role_name="NodeEvolution",
                content=PROMPT_after_update_action.format(
                    current_node_snapshot=self.text_snapshot(),
                    new_content=new_content,
                )
            )
            self_update_pool = await self.step_with_semaphore("evolved", prompt)
        
            self.self_update(self_update_pool)
            self.is_content_updated = True
            for content_object in self.represented_pool:
                if not content_object["is_used_for_update"]:
                    content_object["is_used_for_update"] = True

    def build_relevant_decision_prompt(self, content: Content) -> BaseMessage:
        return BaseMessage.make_user_message(
            role_name="Dispatcher",
            content=PROMPT_for_concept_relevance.format(
                current_node_snapshot=self.text_snapshot(),
                new_incoming_content=content,
                children_keys="\n"+"\n".join(["- "+k for k in self.children_keys])+"\n",  
            )
        )
   
    async def generate_relevant_decision_via_json_output_LLM(self, content: Content) -> DispatchDecision:
        """Async version of relevance decision generation"""
        user_msg = self.build_relevant_decision_prompt(content)
        response = await self.step_with_semaphore("relevant", user_msg)  # Assuming step method is made async
 
        def basement_polish(string):
            return string.strip().strip('[]') if string else None
        response = {k:basement_polish(v) for k,v in response.items()}
        return DispatchDecision(**response)
    
    async def make_dispatch_decision(self, content: Content) -> DispatchDecision:
        """Async version of dispatch decision making"""
        decision = await self.generate_relevant_decision_via_json_output_LLM(content)
        return decision
    
    async def generate_node_action_decision_via_simple_LLM(self, content:Content) -> NodeActionDecision:
        """Generate a decision about node action using LLM"""
        raise NotImplementedError("This method should be implemented by the subclass")
        user_msg   = self.build_node_action_decision_prompt(content)
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
        response =await self.step_with_semaphore(user_msg, output_schema=NodeActionSchema)
        return eval(response.msg.content)

    def build_node_action_decision_prompt(self, extra_content) -> BaseMessage:
        
        return BaseMessage.make_user_message(
            role_name="NodeAnalyzer",
            content=PROMT_for_action_decision_make.format(
                current_node_snapshot=self.text_snapshot(),
                extra_content_snapshot=extra_content,
            )
        )

    async def generate_node_action_decision_via_json_output_LLM(self, content: Content) -> NodeActionDecision:
        """Async version of node action decision generation"""
        prompt   = self.build_node_action_decision_prompt(content)
        response = await self.step_with_semaphore("action", prompt)
        action = NodeActionDecision.load_from_dict(response)
        action.details.content = content
        return action

    async def make_node_action_decision_via_content(self, content: Content) -> NodeActionDecision:
        """Async version of node action decision making"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await self.generate_node_action_decision_via_json_output_LLM(content)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
        logger.error("Max retries reached. Unable to generate node action decision.")
        return None

    async def execute_action(self, action_decision: NodeActionDecision) -> None:
        """
        Execute the determined action
        We will record the action history in the executed_action_history
        """
        logger.info(f"Executing {action_decision.action} on node {self.key}")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Execute the action based on the decision
                if isinstance(action_decision, AddAction):
                    await self._execute_add_action(action_decision)
                elif isinstance(action_decision, MergeAction):
                    await self._execute_merge_action(action_decision)
                elif isinstance(action_decision, UpdateAction):
                    await self._execute_update_action(action_decision)
                print(f"<- Action: {action_decision.action} at {self.key}")
                self.content_pool = []
                self.executed_action_history.append(action_decision)

                self._logs.append({
                    "parent": self.key,
                    "old_concept_key_word": self.key,
                    "old_concept_abstract": "",
                    "timestamp": datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),

                }|action_decision.flatten_dict())

                return 
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("Max retries reached. Unable to execute action.")
                    return
        
    async def _execute_add_action(self, action: AddAction) -> None:
        """Async version of add action execution"""
        details = action.details
        new_node = NodeAgent(
            key=details.new_concept_key_word,
            value={"concept_key_word": details.new_concept_key_word,
                   "concept_abstract": details.new_concept_abstract},
            is_merged_after_add=False,
        )
        content:Content = action.details.content
        new_node.represented_pool.append({
            "content_metadata": {
                "content_id_address": content.id_address,
            },
            "is_used_for_update": False,
        })
        self.add_child(new_node)
        ### we will record each action and its reason, and its timestamp via json format
        
    async def _execute_update_action(self, action: UpdateAction) -> None:
        """
        Async version of update action execution
        Records the updated content in the represented_pool
        """
        content:Content = action.details.content
        self.represented_pool.append({
            "content_metadata": {
                "content_id_address": content.id_address,
            },
            "is_used_for_update": False,
        })

    ### other methods
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

    def get_whole_represented_pool(self) -> List[Dict]:
        """
        This method traverse the tree and collect all the id address of content in the represented_pool
        """
        content_id_address_pool = [content["content_metadata"]["content_id_address"] for content in self.represented_pool]
        for child in self.children:
            content_id_address_pool.extend(child.get_whole_represented_pool())
        return content_id_address_pool
    
    def __getitem__(self, key: str):
        return self.get_child_via_key(key)

    ## current we record each node modification at node side, let traverse the tree and collect all the logs
    
    def retrieve_all_logs(self):
        logs = self._logs.copy()  # Make a copy to avoid modifying the original list during iteration
        for child in self.children:
            child_logs = child.retrieve_all_logs()  # Collect logs from the child
            logs.extend(child_logs)  # Extend the main logs list with the child's logs
        return logs
    
    ## we will reset the logs after collection. notice we should clean thole the child
    def clean_logs(self):
        self._logs = []
        for child in self.children:
            child.clean_logs()

    def clean_after_retrieve_all_logs(self):
        logs = self.retrieve_all_logs()
        self.clean_logs()
        return logs
        
    
class NodeTree(NodeAgent):   
    
    
    
    async def dispatch_content(self, content: Content, start_node: Optional[NodeAgent] = None) -> List[str]:
        """Async version of content dispatch"""
        assert isinstance(content, Content), "content must be a Content object"
        current_node = start_node or self
        path = []
        max_retries = 3
        while True:
            decision = None
            for attempt in range(max_retries):
                try:    
                    decision: DispatchDecision = await current_node.make_dispatch_decision(content)
                    #print(decision)
                    break
                except Exception as e:
                    logger.warning(f"Error in dispatch_content: {e}, retry")
            if decision is None:
                return None
            path.append(current_node.key)
            
            if decision.decision == RelevanceCategory.RELEVANT:
                await self._update_node_candidate_content(current_node, content)
                break
            elif decision.decision == RelevanceCategory.LOWER and current_node.children and decision.next_position in set(current_node.children_keys):
                current_node = current_node.get_child_via_key(decision.next_position)
            else:
                if decision.decision == RelevanceCategory.LOWER:
                    await self._update_node_candidate_content(current_node, content)
                    path.append(f"[{decision.next_position}]")
                else:
                    raise ValueError(f"Invalid decision: {decision}")
                break
                
        print("-> "+ " -> ".join(path))
        return path

    async def _update_node_candidate_content(self, node: NodeAgent, content: Content) -> None:
        """Async version of node content update"""
        node.content_pool.append(content)
        #print(f"Content added to node {node.key}: {content}")
    async def dispatch_multiple_contents(self, contents: List[Dict]) -> List[List[str]]:
        """Dispatch multiple contents in parallel"""
        tasks = [self.dispatch_content(content) for content in contents]
        paths = await asyncio.gather(*tasks)
        return paths

    async def backpropagate(self, start_node: Optional[NodeAgent] = None, do_evolve: bool=True) -> None:
        """Async version of backpropagation"""
        start_node = start_node or self
        start_node.evolved_count += 1
        start_node.is_content_updated = False
        # Process leaves first (post-order traversal)
        # Process children in parallel
        child_tasks = [self.backpropagate(child) for child in start_node.children]
        if child_tasks:
            await asyncio.gather(*child_tasks)

        # Process current node's content in parallel
        if start_node.content_pool:
            # Make decisions in parallel
            action_tasks = [
                start_node.make_node_action_decision_via_content(content) 
                for content in start_node.content_pool
            ]
            action_decisions = await asyncio.gather(*action_tasks)
            
            # Execute actions and collect their tasks
            action_execution_tasks = []
            for action_decision in action_decisions:
                if action_decision is None:continue
                task = start_node.execute_action(action_decision)
                action_execution_tasks.append(task)
            
            if do_evolve:
                # Wait for ALL actions to complete before evolving
                await asyncio.gather(*action_execution_tasks)
                
                # Now we can safely evolve the node
                await start_node.self_evolve()
            
    async def process_multiple_contents(self, contents: List[Dict]) -> None:
        """Process multiple contents through dispatch and backpropagation"""
        # First dispatch all contents in parallel
        paths = await self.dispatch_multiple_contents(contents)
        
        # Then perform backpropagation
        await self.backpropagate()

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
