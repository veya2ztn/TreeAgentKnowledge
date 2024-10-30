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
class Template_update_after_update_action:
    new_concept_key_word: str = "Updated Key Word"
    new_concept_abstract: str = "Updated abstract incorporating new information."

    @staticmethod
    def load_from_dict(dict: Dict[str, Any]):
        return Template_update_after_update_action(**dict)
    
    def json_snapshot(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

# Example 1: Expanding knowledge with new information
example_1_before = NodeStructure.load_from_dict({
    "concept_key_word": "Neural Networks",
    "concept_abstract": "Neural networks are computing systems inspired by biological neural networks, used for machine learning tasks.",
    "concept_represents": [],
    "sub_concept_0": []
})

example_1_new_content = """
Neural networks are computational models inspired by the human brain's structure and function. They consist of interconnected nodes (neurons) organized in layers, capable of learning complex patterns through training. Modern neural networks form the backbone of deep learning, enabling breakthrough achievements in computer vision, natural language processing, and other AI applications.
"""

example_1_after = Template_update_after_update_action.load_from_dict({
    "new_concept_key_word": "Neural Networks",
    "new_concept_abstract": "Neural networks are computational models inspired by the human brain's structure and function, consisting of interconnected nodes organized in layers. These systems form the backbone of deep learning, enabling breakthroughs in computer vision, natural language processing, and other AI applications through pattern learning."
})

# Example 2: Refining and clarifying existing knowledge
example_2_before = NodeStructure.load_from_dict({
    "concept_key_word": "Cloud Computing",
    "concept_abstract": "Cloud computing is the delivery of computing services over the internet.",
    "concept_represents": [],
    "sub_concept_0": []
})

example_2_new_content = """
Cloud computing represents a paradigm shift in IT infrastructure, offering on-demand access to computing resources including servers, storage, databases, networking, software, and analytics over the internet. This model eliminates the need for direct active management by users and enables pay-as-you-go pricing, making it highly scalable and cost-effective for businesses of all sizes.
"""

example_2_after = Template_update_after_update_action.load_from_dict({
    "new_concept_key_word": "Cloud Computing",
    "new_concept_abstract": "Cloud computing is a technology paradigm that delivers on-demand computing resources (servers, storage, databases, networking, software, and analytics) over the internet. It features pay-as-you-go pricing and eliminates direct infrastructure management, providing scalable and cost-effective solutions for businesses."
})

# Example 3: Correcting and updating outdated information
example_3_before = NodeStructure.load_from_dict({
    "concept_key_word": "Quantum Computing",
    "concept_abstract": "Quantum computing uses quantum bits to perform calculations faster than classical computers.",
    "concept_represents": [],
    "sub_concept_0": []
})

example_3_new_content = """
Quantum computing harnesses quantum mechanical phenomena such as superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits that can exist in multiple states simultaneously. This enables them to solve certain complex problems exponentially faster than classical computers, particularly in areas like cryptography, drug discovery, and optimization problems.
"""

example_3_after = Template_update_after_update_action.load_from_dict({
    "new_concept_key_word": "Quantum Computing",
    "new_concept_abstract": "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information using qubits, which can exist in multiple states simultaneously. This technology enables exponentially faster computation for specific problems in cryptography, drug discovery, and optimization compared to classical computers."
})

class PROMPT_after_update_action:
    """
    This class generates prompts for updating node information based on new content in the Tree Agent Knowledge System (TAKS).
    """
    @staticmethod
    def format(current_node_snapshot: str, new_content: str) -> str:
        return f"""You are tasked with updating the concept information for a knowledge tree node in the Tree Agent Knowledge System (TAKS) based on new content.

### Context

**Project Description: Tree Agent Knowledge System**
- **Objective:** Maintain and update a tree-based knowledge management system with accurate and comprehensive information.

**Current Task:**
- Update the node's concept information based on new, more detailed or updated content while maintaining consistency and accuracy.

### Current Node Information
{current_node_snapshot}

### New Content to Incorporate
{new_content}

### Instructions
Update the node's concept keyword and abstract based on the new content provided. The update should:
1. Maintain or improve the clarity of the concept
2. Incorporate new relevant information
3. Ensure accuracy and completeness
4. Be concise yet comprehensive

**Guidelines:**
- **Integration:** Blend existing knowledge with new information seamlessly
- **Clarity:** Use clear, precise language
- **Conciseness:** Keep the abstract focused and well-structured
- **Format:** Provide the updated information in the following JSON format:

```
{Template_update_after_update_action().json_snapshot()}
```

### Positive Examples

#### Example 1: Expanding Knowledge
**Current Node:**
{example_1_before.text_snapshot()}

**New Content:**
{example_1_new_content}

**Expected Output:**
```
{example_1_after.json_snapshot()}
```

#### Example 2: Refining Knowledge
**Current Node:**
{example_2_before.text_snapshot()}

**New Content:**
{example_2_new_content}

**Expected Output:**
```
{example_2_after.json_snapshot()}
```

#### Example 3: Updating Information
**Current Node:**
{example_3_before.text_snapshot()}

**New Content:**
{example_3_new_content}

**Expected Output:**
```
{example_3_after.json_snapshot()}
```

### Final Output Format
Provide only the JSON object with the updated concept key word and abstract. Do not include any additional text or explanations.

The response must strictly follow the specified format for automated processing within the knowledge tree system.
"""

if __name__ == "__main__":
    print(PROMPT_after_update_action.format(example_1_before.text_snapshot(), example_1_new_content)) 