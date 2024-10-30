from enum import Enum
from typing import List,Literal
from dataclasses import dataclass
import json


@dataclass
class DispatchDecision:
    decision: Literal["relevant", "lower", "higher", "unrelated"]
    next_position: str
    reasoning: str = None
    def json_snapshot(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

example_1_dispatch = DispatchDecision(decision="lower", 
                                      next_position="[ChildKey]", 
                                      reasoning="The content introduces advanced techniques that are extensions of the current concept, making it suitable for a child node.")
template_for_relevance_decision = DispatchDecision(decision="[RELEVANT/LOWER/HIGHER/UNRELATED]", 
                                      next_position="[ChildKey] or nowhere", 
                                      reasoning="[Provide your detailed reasoning here.]")
class Prompt_for_relevance_decision:
    def format(current_node_snapshot: str, new_incoming_content: str, children_keys: List[str]) -> str:
        return f"""Here is the concept detail as your Responsibilities
{current_node_snapshot}
Here is the new content that you need to analyze
{new_incoming_content}

### **Response Format:**
```
{template_for_relevance_decision.json_snapshot()}
```

### **Example Response:**
```
{example_1_dispatch.json_snapshot()}
```

### **Guidelines for Reasoning:**
- **Relevance Justification:** Explain how the content aligns or diverges from the current concept.
- **Hierarchy Consideration:** If categorizing as `LOWER` or `HIGHER`, specify the relationship to child or parent nodes.
- **Clarity and Precision:** Ensure that the reasoning is clear, logical, and free from ambiguity.

### **Best Practices:**
- **Consistency:** Maintain consistent criteria for categorization to uphold the integrity of the knowledge tree.
- **Comprehensiveness:** Consider all aspects of the content to make an informed decision.
- **Documentation:** Keep a record of your decisions and the accompanying reasoning for future reference and analysis.
- **Recursive Dispatching:** When the `Category` is set to `LOWER`, ensure that the content is dispatched to the specified child node for further evaluation. Continue this process recursively until the content is categorized as `RELEVANT`, `HIGHER`, or `UNRELATED` at the final node.
- **Child Node Verification:** Before dispatching to a child node, verify that the child key is one of [{children_keys}].

### **Additional Requirement for LOWER Category:**
- When **Category** is set to `LOWER`, the response **must** include the `ChildKey` field, specifying the exact child node under which the new content should be added. This ensures precise dispatching and maintains the hierarchical structure of the knowledge tree.
- The system should automatically dispatch the content to the specified `ChildKey` and repeat the evaluation process at the child node level, enabling deep hierarchical integration of the content.

```
"""
  