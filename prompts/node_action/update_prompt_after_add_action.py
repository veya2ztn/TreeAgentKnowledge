import sys, os
### lets add the root directory to the sys path
current_file_path = os.path.dirname(os.path.abspath(__file__))
root_file_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(root_file_path)

from node_base import NodeStructure
from dataclasses import dataclass
from typing import Dict, Any
import json

@dataclass
class Template_update_after_add_action:
    new_concept_key_word: str = "New Key Word"
    new_concept_abstract: str = "New updated abstract."

    @staticmethod
    def load_from_dict(dict: Dict[str, Any]):
        return Template_update_after_add_action(**dict)
    
    def json_snapshot(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

example_1_before = NodeStructure.load_from_dict({
    "concept_key_word": "Machine Learning",
    "concept_abstract": "Machine Learning involves algorithms that enable computers to learn from and make decisions based on data.",
    "concept_represents": [],
    "sub_concept_0": [
        {
            "concept_key_word": "Supervised Learning",
            "concept_abstract": "Techniques where models are trained on labeled data.",
            "concept_represents": [],
            "sub_concept_1": []
        },
        {
            "concept_key_word": "Unsupervised Learning",
            "concept_abstract": "Methods that find patterns in unlabeled data.",
            "concept_represents": [],
            "sub_concept_3": []
        }
    ]
})

example_1_add    = NodeStructure.load_from_dict({
    "concept_key_word": "Reinforcement Learning",
    "concept_abstract": "A type of machine learning where agents learn to make decisions by taking actions in an environment to achieve maximum cumulative reward.",
    "concept_represents": [],
    "sub_concept_0": []
})

example_1_after  = NodeStructure.load_from_dict({
    "concept_key_word": "Machine Learning",
    "concept_abstract": "Machine Learning involves algorithms that enable computers to learn from and make decisions based on data, including supervised learning, unsupervised learning, and reinforcement learning.",
    "concept_represents": [],
    "sub_concept_0": [
        {
            "concept_key_word": "Supervised Learning",
            "concept_abstract": "Techniques where models are trained on labeled data.",
            "concept_represents": [],
            "sub_concept_1": []
        },
        {
            "concept_key_word": "Unsupervised Learning",
            "concept_abstract": "Methods that find patterns in unlabeled data.",
            "concept_represents": [],
            "sub_concept_3": []
        },
        {
            "concept_key_word": "Reinforcement Learning",
            "concept_abstract": "A type of machine learning where agents learn to make decisions by taking actions in an environment to achieve maximum cumulative reward.",
            "concept_represents": [],
            "sub_concept_1": []
        }
    ]
})

example_1_patch  = Template_update_after_add_action.load_from_dict({
    "new_concept_key_word": "Machine Learning",
    "new_concept_abstract": "Machine Learning involves algorithms that enable computers to learn from and make decisions based on data, including supervised learning, unsupervised learning, and reinforcement learning."
})

# Example 2
example_2_before = NodeStructure.load_from_dict({
    "concept_key_word": "Data Analysis",
    "concept_abstract": "Data Analysis involves inspecting, cleansing, transforming, and modeling data to discover useful information and support decision-making.",
    "concept_represents": [],
    "sub_concept_0": [
        {
            "concept_key_word": "Descriptive Analytics",
            "concept_abstract": "Techniques for summarizing historical data to identify trends.",
            "concept_represents": [],
            "sub_concept_1": []
        }
    ]
})

example_2_add    = NodeStructure.load_from_dict({
    "concept_key_word": "Predictive Analytics",
    "concept_abstract": "Techniques that use statistical models and forecasts to understand future outcomes based on historical data.",
    "concept_represents": [],
    "sub_concept_0": []
})

example_2_after  = NodeStructure.load_from_dict({
    "concept_key_word": "Data and Predictive Analytics",
    "concept_abstract": "Data Analysis involves inspecting, cleansing, transforming, and modeling data to discover useful information and support decision-making. This includes descriptive analytics, predictive analytics, and other advanced techniques to enhance the decision-making process.",
    "concept_represents": [],
    "sub_concept_0": [
        {
            "concept_key_word": "Descriptive Analytics",
            "concept_abstract": "Techniques for summarizing historical data to identify trends.",
            "concept_represents": [],
            "sub_concept_1": []
        },
        {
            "concept_key_word": "Predictive Analytics",
            "concept_abstract": "Techniques that use statistical models and forecasts to understand future outcomes based on historical data.",
            "concept_represents": [],
            "sub_concept_1": []
        }
    ]
})

example_2_patch  = Template_update_after_add_action.load_from_dict({
    "new_concept_key_word": "Data and Predictive Analytics",
    "new_concept_abstract": "Data Analysis involves inspecting, cleansing, transforming, and modeling data to discover useful information and support decision-making. This includes descriptive analytics, predictive analytics, and other advanced techniques to enhance the decision-making process."
})

# Example 3
example_3_before = NodeStructure.load_from_dict({
    "concept_key_word": "Project Management",
    "concept_abstract": "Project Management involves planning, executing, and finalizing projects, ensuring they are completed on time and within budget.",
    "concept_represents": [],
    "sub_concept_0": [
        {
            "concept_key_word": "Agile Methodology",
            "concept_abstract": "An iterative approach to project management and software development that helps teams deliver value to their customers faster.",
            "concept_represents": [],
            "sub_concept_1": []
        },
        {
            "concept_key_word": "Waterfall Model",
            "concept_abstract": "A sequential design process used in software development processes, where progress flows downward through distinct phases.",
            "concept_represents": [],
            "sub_concept_1": []
        }
    ]
})

example_3_add    = NodeStructure.load_from_dict({
    "concept_key_word": "Lean Management",
    "concept_abstract": "A methodology that focuses on minimizing waste within manufacturing systems while simultaneously maximizing productivity.",
    "concept_represents": [],
    "sub_concept_0": []
})

example_3_after  = NodeStructure.load_from_dict({
    "concept_key_word": "Management Practices",
    "concept_abstract": "Project Management involves planning, executing, and finalizing projects, ensuring they are completed on time and within budget. This includes various management practices such as Agile Methodology, Waterfall Model, and Lean Management to optimize efficiency and effectiveness.",
    "concept_represents": [],
    "sub_concept_0": [
        {
            "concept_key_word": "Agile Methodology",
            "concept_abstract": "An iterative approach to project management and software development that helps teams deliver value to their customers faster.",
            "concept_represents": [],
            "sub_concept_1": []
        },
        {
            "concept_key_word": "Waterfall Model",
            "concept_abstract": "A sequential design process used in software development processes, where progress flows downward through distinct phases.",
            "concept_represents": [],
            "sub_concept_1": []
        },
        {
            "concept_key_word": "Lean Management",
            "concept_abstract": "A methodology that focuses on minimizing waste within manufacturing systems while simultaneously maximizing productivity.",
            "concept_represents": [],
            "sub_concept_1": []
        }
    ]
})

example_3_patch  = Template_update_after_add_action.load_from_dict({
    "new_concept_key_word": "Management Practices",
    "new_concept_abstract": "Project Management involves planning, executing, and finalizing projects, ensuring they are completed on time and within budget. This includes various management practices such as Agile Methodology, Waterfall Model, and Lean Management to optimize efficiency and effectiveness."
})

class PROMT_after_add_action:
    """
    This class is used to generate the prompt for updating the concept information for a knowledge tree node in the Tree Agent Knowledge System (TAKS).
    """
    @staticmethod
    def format(current_node_snapshot: str, new_node_snapshot: str) -> str:
        return f"""You are tasked with updating the concept information for a knowledge tree node in the Tree Agent Knowledge System (TAKS).

### Context

**Project Description: Tree Agent Knowledge System**
- **Objective:** Create a tree-based knowledge management system that organizes concepts hierarchically.

**Current Workflow:**
- **Dispatch:** Identify the appropriate location for new content without performing updates.
- **Backpropagation:** Update the tree by adding, deleting, merging, or rearranging nodes based on the content.

**Node Attributes:**
- Concept Keyword
- Concept Description
- Children

### Current Node Information
{current_node_snapshot}

### New Child Node Information
{new_node_snapshot}

### Instructions
Based on the addition of the new child node, update the current node's concept keyword or concept abstract to better reflect the expanded knowledge structure. Ensure that the updated information maintains coherence and logical flow with the new child node.

**Guidelines:**
- **Consistency:** The updated abstract should incorporate both the existing content and the new information from the added child node.
- **Clarity:** Use clear and concise language.
- **Relevance:** Ensure the updated keyword or abstract remains relevant to the node's overarching concept and its relationship with child nodes.
- **Format:** Provide the updated information in the following JSON format:

```
{Template_update_after_add_action().json_snapshot()}
```

### Positive Examples
#### Positive Example 1 (It is ok to keep the current key word same but only update the abstract)
**Before Adding Child Node:**
{example_1_before.text_snapshot()}

**Added Child Node:**
{example_1_add.text_snapshot()}

**Expected Output:**
```
{example_1_patch.json_snapshot()}
```

#### Positive Example 2 (It is preferred to update the key word and abstract both. Notice we tend to reflect a higher concept name for key work update)
**Before Adding Child Node:**
{example_3_before.text_snapshot()}

**Added Child Node:**
{example_3_add.text_snapshot()}

**Expected Output:**
```
{example_3_patch.json_snapshot()}
```

### Negative Examples
#### Negative Example 1 (It is not preferred that updating the key word to a combination name such as "xxx and xxxx". Please summarize the mergered concept into a higher level key word)
**Before Adding Child Node:**
{example_2_before.text_snapshot()}

**Added Child Node:**
{example_2_add.text_snapshot()}

**Expected Output:**
```
{example_2_patch.json_snapshot()}
```


### Final Output Format
Provide only the JSON object with the updated concept key word and abstract as shown above. Do not include any additional text, explanations, or formatting.

Ensure that the response strictly adheres to the specified format to facilitate automated parsing and integration into the knowledge tree system.
"""

if __name__ == "__main__":
    print(PROMT_after_add_action.format(example_1_before.text_snapshot(), example_1_add.text_snapshot()))