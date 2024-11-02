from pydantic import BaseModel, Field,ConfigDict
from typing import Dict, List, Optional, Set, Literal
import json
class NodeActionDecision(BaseModel):
    model_config = ConfigDict(extra='allow')

    def json_snapshot(self):
        the_dict = self.model_dump()
        return json.dumps(the_dict, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    
    

    @staticmethod
    def load_from_dict(pool: Dict):
        details_models = {"ADD": (AddAction, AddActionDetails),
                          "UPDATE": (UpdateAction, UpdateActionDetails),
                          "MERGE": (MergeAction, MergeActionDetails),
                          "DELETE": (DeleteAction, DeleteActionDetails),
                          "REARRANGE": (RearrangeAction, RearrangeActionDetails),
                          "NO_ACTION": (NoAction, NoActionDetails)}
        action_type, details_type = details_models[pool['action']]
        return action_type(
            action=pool['action'],
            details=details_type(**pool['details']),
            reason=pool['reason']
        )
#### AddAction
class AddActionDetails(NodeActionDecision):
    """Details for ADD action"""
    new_concept_key_word: str = Field(default="new_concept_key_word", description="Key word for the new concept")
    new_concept_abstract: str = Field(default="new_concept_abstract", description="Abstract description of the new concept")
    
    def load_from_dict(dict: Dict):
        return AddActionDetails(**dict)
    
    
class AddAction(NodeActionDecision):
    action:  Literal["ADD"] = "ADD"
    details: AddActionDetails = Field(default=AddActionDetails(), description="Details for ADD action")
    reason:  str = Field(default="explanation", description="Reasoning for the action")

    def flatten_dict(self):
        return {
            "action": "add",
            "new_concept_key_word": self.details.new_concept_key_word,
            "new_concept_abstract": self.details.new_concept_abstract,
            "reason": self.reason
        }


#### UpdateAction
class UpdateActionDetails(NodeActionDecision):
    """
    Details for UPDATE action
    Update means we will add current incoming content to the current node represent list.
    Then we need update the abstract/description of the current node.
    Lets always do information update in post processing
    """
    pass
    # content:str = Field(default="please do not add anything here", description="Content place holder")
    # new_concept_abstract: str = Field(default="new abstract", description="Updated abstract for the current node")
    # new_concept_key_word: str = Field(default="new key word", description="Updated key word for the current node")

class UpdateAction(NodeActionDecision):
    action: Literal["UPDATE"] = "UPDATE"
    details: UpdateActionDetails = Field(default=UpdateActionDetails(), description="Details for UPDATE action")
    reason: str = Field(default="explanation", description="Reasoning for the action")

    def flatten_dict(self):
        return {
            "action": "append",
            "new_concept_key_word": f"Paper {self.details.content.id_address}",
            "new_concept_abstract": "",
            "reason": self.reason
        }


#### DeleteAction
class DeleteActionDetails(NodeActionDecision):
    """Details for DELETE action"""
    target_concept_key_word: str = Field(default="the_key_word_need_to_be_deleted", description="Key of the node to be deleted")
    reassign_children_to: Optional[str] = Field(
        default=None, 
        description="Optional key of the node to reassign children to"
    )

class DeleteAction(NodeActionDecision):
    action: Literal["DELETE"] = "DELETE"
    details: DeleteActionDetails = Field(default=DeleteActionDetails(), description="Details for DELETE action")
    reason: str = Field(default="explanation", description="Reasoning for the action")

#### MergeAction
class MergeActionDetails(NodeActionDecision):
    """Details for MERGE action"""
    source_nodes: List[str]   = Field(default=["node1", "node2"], description="List of node keys to be merged")
    new_concept_key_word: str = Field(default="merged_concept", description="Key word for the merged concept")
    new_concept_abstract: str = Field(default="merged description", description="Abstract for the merged concept")

class MergeAction(NodeActionDecision):
    action: Literal["MERGE"] = "MERGE"
    details: MergeActionDetails = Field(default=MergeActionDetails(), description="Details for MERGE action")
    reason: str = Field(default="explanation", description="Reasoning for the action")

#### RearrangeAction
class RearrangeActionDetails(NodeActionDecision):
    """Details for REARRANGE action"""
    new_order: List[str] = Field(default=["node1", "node2", "node3"], description="New order of child nodes")

class RearrangeAction(NodeActionDecision):
    action: Literal["REARRANGE"] = "REARRANGE"
    details: RearrangeActionDetails = Field(default=RearrangeActionDetails(), description="Details for REARRANGE action")
    reason: str = Field(default="explanation", description="Reasoning for the action")



#### NoAction
class NoActionDetails(NodeActionDecision):
    """Details for NO_ACTION"""
    pass

class NoAction(NodeActionDecision):
    action: Literal["NO_ACTION"] = "NO_ACTION"
    details: NoActionDetails = Field(default=NoActionDetails(), description="Details for NO_ACTION")
    reason: str = Field(default="explanation", description="Reasoning for the action")