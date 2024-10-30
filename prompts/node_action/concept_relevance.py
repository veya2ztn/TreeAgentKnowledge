import sys, os
from enum import Enum
from typing import List
from pydantic import BaseModel, Field
### lets add the root directory to the sys path
current_file_path = os.path.dirname(os.path.abspath(__file__))
root_file_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(root_file_path)

from dataclasses import dataclass
from typing import Dict, Any, Literal
import json


class RelevanceCategory:
    RELEVANT = "RELEVANT"
    LOWER = "LOWER"
    HIGHER = "HIGHER"
    UNRELATED = "UNRELATED"
Available_Category_List = [value for name, value in RelevanceCategory.__dict__.items() if not name.startswith('__') and not callable(value)]
Example= "["+"/".join(Available_Category_List)+"]"
Available_Category = Literal[tuple(Available_Category_List+[Example])] # type: ignore

class DispatchDecision(BaseModel):
    """Template for the concept relevance response"""
    decision: Available_Category = Field( # type: ignore
                description=f"The category assigned to the content based on its relevance. Possible values are '{RelevanceCategory.RELEVANT}', '{RelevanceCategory.LOWER}', '{RelevanceCategory.HIGHER}', or '{RelevanceCategory.UNRELATED}'."
            )
    next_position: str= Field(
                description="Indicates the next position for dispatching the content. Possible values are 'child' to dispatch to a sub-node or 'nowhere' to handle within the current node."
            )
    reasoning: str =  Field( description="A detailed explanation of why the content was categorized under the specified Category."
            )
    @staticmethod
    def load_from_dict(dict: Dict[str, Any]):
        return DispatchDecision(**dict)
    
    def json_snapshot(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


example_1_dispatch = DispatchDecision(decision=f"{RelevanceCategory.LOWER}", 
                                      next_position="[ChildKey]", 
                                      reasoning="The content introduces advanced techniques that are extensions of the current concept, making it suitable for a child node.")

example_2_dispatch = DispatchDecision(decision=f"{RelevanceCategory.RELEVANT}", 
                                      next_position="nowhere", 
                                      reasoning="The content introduces advanced techniques that are extensions of the current concept, making it suitable for a child node.")


template_for_relevance_decision = DispatchDecision(decision=Example, 
                                      next_position="[ChildKey] or nowhere", 
                                      reasoning="[Provide your detailed reasoning here.]")


class PROMPT_for_concept_relevance:
    """
    This class is used to generate the prompt for analyzing the relevance of new content
    to an existing concept node in the knowledge tree.
    """
    @staticmethod
    def format(current_node_snapshot: str, new_incoming_content: str, children_keys: List[str]) -> str:
        return f"""You are a content relevance analyzer in the Tree Agent Knowledge System (TAKS). Your role is to determine the most appropriate placement for new content within the knowledge tree hierarchy.

### Current Node Information
{current_node_snapshot}

### New Content for Analysis
{new_incoming_content}

### Core Decision Guidelines
1. ALWAYS prefer to dispatch content to more specific ({RelevanceCategory.LOWER}) levels when possible
2. Only mark as RELEVANT to current node if:
   - Content is too general for any children
   - Content directly addresses the current concept level
   - Content cannot be logically placed in any existing child nodes
3. Mark as HIGHER if the content is clearly more general than the current node
4. Mark as UNRELATED only if there's no logical connection to this branch

### Decision Priority (in order)
1. First, check if content can fit into any potential child nodes ({RelevanceCategory.LOWER})
2. If not, evaluate if it belongs at current level ({RelevanceCategory.RELEVANT})
3. If content seems more general, suggest moving up ({RelevanceCategory.HIGHER})
4. If no connection exists, mark as UNRELATED ({RelevanceCategory.UNRELATED})

### Response Format
```
{template_for_relevance_decision.json_snapshot()}
```

### Example Responses

For content that should go to a child node:
```
{example_1_dispatch.json_snapshot()}
```

For content at current level:
```
{example_2_dispatch.json_snapshot()}
```


### **Guidelines for Reasoning:**
- **Relevance Justification:** Explain how the content aligns or diverges from the current concept.
- **Hierarchy Consideration:** If categorizing as `{RelevanceCategory.LOWER}` or `{RelevanceCategory.HIGHER}`, specify the relationship to child or parent nodes.
- **Clarity and Precision:** Ensure that the reasoning is clear, logical, and free from ambiguity.

### **Best Practices:**
- **Consistency:** Maintain consistent criteria for categorization to uphold the integrity of the knowledge tree.
- **Comprehensiveness:** Consider all aspects of the content to make an informed decision.
- **Documentation:** Keep a record of your decisions and the accompanying reasoning for future reference and analysis.
- **Recursive Dispatching:** When the `Category` is set to `{RelevanceCategory.LOWER}`, ensure that the content is dispatched to the specified child node for further evaluation. Continue this process recursively until the content is categorized as `{RelevanceCategory.RELEVANT}`, `{RelevanceCategory.HIGHER}`, or `{RelevanceCategory.UNRELATED}` at the final node.
- **Child Node Verification:** Before dispatching to a child node, verify that the child key is one of below: 
{children_keys}

### **Additional Requirement for LOWER Category:**
- When **Category** is set to `{RelevanceCategory.LOWER}`, the response **must** include the `ChildKey` field, specifying the exact child node under which the new content should be added. This ensures precise dispatching and maintains the hierarchical structure of the knowledge tree.
- The system should automatically dispatch the content to the specified `ChildKey` and repeat the evaluation process at the child node level, enabling deep hierarchical integration of the content.

"""

if __name__ == "__main__":
    # Example usage
    example_prompt = PROMPT_for_concept_relevance.format(
        concept_key="Machine Learning",
        concept_abstract="Machine Learning involves algorithms that enable computers to learn from and make decisions based on data.",
        content="Neural Networks are a type of machine learning model inspired by biological neural networks."
    )
    print(example_prompt)