import sys, os
current_file_path = os.path.dirname(os.path.abspath(__file__))
root_file_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(root_file_path)
from node_base import NodeStructure
from prompts.node_action.node_action import *

example_1_before = NodeStructure.load_from_dict({
    "concept_key_word": "Neural Networks",
    "concept_abstract": "Computational models inspired by biological neural networks, capable of learning patterns from data",
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

example_1_new_content="""
Recent studies show neural networks excel in both supervised and unsupervised learning tasks, with applications ranging from image recognition to natural language processing.
"""

example_1_update_action = UpdateAction(
    action="UPDATE",
    details=UpdateActionDetails(),
    reason="The new content enriches the existing concept by highlighting its versatility and application domains without introducing a new sub-concept"
)

template_update_action = UpdateAction(
    action="UPDATE",
    details=UpdateActionDetails(),
    reason="explanation of how the new content enhances the existing concept"
)

example_2_before= NodeStructure.load_from_dict({
    "concept_key_word": "Deep Learning",
    "concept_abstract": "Advanced neural network architectures with multiple layers for complex pattern recognition",
    "concept_represents": [],
    "sub_concept_0": []
})

example_2_new_content="""
Graph Neural Networks (GNNs) apply deep learning techniques to graph-structured data, enabling learning on non-euclidean information.
"""
example_2_add_action = AddAction(
    action="ADD",
    details=AddActionDetails(
        new_concept_abstract="Graph Neural Networks (GNNs) are specialized deep learning architectures designed for processing and learning from graph-structured data, enabling analysis of non-euclidean information structures",
        new_concept_key_word="Graph Neural Networks"),
    reason="GNNs represent a distinct specialized architecture that warrants its own sub-concept under deep learning, operating on a specific data type (graphs) with unique methodologies"
)

example_3_before = NodeStructure.load_from_dict({
    "concept_key_word": "Model Optimization",
    "concept_abstract": "Techniques and methods for improving neural network performance and efficiency",
    "concept_represents": [],
    "sub_concept_0": []
})

example_3_new_content="""
Knowledge distillation enables the transfer of knowledge from large complex models to smaller, more efficient ones while maintaining performance.
"""
example_3_add_action = AddAction(
    action="ADD",
    details=AddActionDetails(
        new_concept_abstract="Knowledge distillation is a technique that transfers learned information from large, complex models to smaller, more efficient ones while preserving performance capabilities",
        new_concept_key_word="Knowledge Distillation"),
    reason="Knowledge distillation represents a specific optimization technique with its own methodology and use cases, warranting a dedicated sub-concept under model optimization"
)

template_add_action = AddAction(
    action="ADD",
    details=AddActionDetails(
        new_concept_key_word="new_concept_key_word",
        new_concept_abstract="new_concept_abstract"
    ),
    reason="explanation of why the new content warrants a new sub-concept"
)

class PROMT_for_action_decision_make:
    """
    This class is used to generate the prompt for making the decision of which action to take for a knowledge tree node.
    """
    @staticmethod
    def format(current_node_snapshot: str, extra_content_snapshot: str) -> str:
        return f"""# Task

You are analyzing a knowledge tree node to determine necessary actions to maintain and enhance the hierarchical structure of higher-level concepts.

**Objective:**
Determine the most appropriate action to integrate the new content into the knowledge tree, ensuring that higher-level concepts are accurately extracted, enhanced, and organized.

### **Detailed Action Guidelines:**

**[UPDATE] Action**
   When to use:

   - The new content directly enhances or refines the current concept level
   - The information provides additional perspectives or details about the existing concept
   - The content helps clarify or strengthen the current concept's description
   - The content represents the same abstraction level as the current node

   Key considerations:

   - Does not change the fundamental concept structure
   - Maintains the same concept keyword
   - Enhances the concept's abstract by incorporating new insights
   - Preserves the hierarchical relationship with parent and child nodes

2. **[ADD] Action**
   When to use:

   - The content represents a distinct sub-concept
   - The information introduces a new specialized area
   - The content requires its own branch in the hierarchy
   - The content represents a lower abstraction level than the current node

   Key considerations:

   - Creates a new node in the hierarchy
   - Establishes clear parent-child relationships
   - Maintains proper abstraction levels
   - Avoids redundancy with existing concepts

### **Action Templates:**

#### For [UPDATE] action:
```
{template_update_action.json_snapshot()}
```

#### For [ADD] action:
```
{template_add_action.json_snapshot()}
```

### **Detailed Examples:**

#### 1. **[UPDATE] Examples:**

##### Example A - Enhancing Core Concept:

- Current Node:
{example_1_before.text_snapshot()}

- New Content: 
{example_1_new_content}

- Decision:
```
{example_1_update_action.json_snapshot()}
```

#### 2. **ADD Examples:**

##### Example A - New Sub-concept Introduction:

- Current Node:
{example_2_before.text_snapshot()}

- New Content: 
{example_2_new_content}

- Decision:
```
{example_2_add_action.json_snapshot()}
```

##### Example B - Specialized Technique:

- Current Node:
{example_3_before.text_snapshot()}

- New Content:
{example_3_new_content}

- Decision:
```
{example_3_add_action.json_snapshot()}
```

**IMPORTANT GUIDELINES:**

1. **Abstraction Level Analysis:**
   - Always evaluate the abstraction level of new content relative to the current node
   - Maintain consistent abstraction levels within each layer of the hierarchy

2. **Content Integration:**
   - For UPDATE: Focus on seamlessly incorporating new information
   - For ADD: Ensure clear differentiation from existing concepts

3. **Hierarchy Maintenance:**
   - Preserve logical relationships between concepts
   - Maintain clear parent-child relationships
   - Avoid redundancy across different branches

4. **Decision Quality:**
   - Provide detailed reasoning for your decisions
   - Consider the impact on the overall knowledge structure
   - Ensure consistency with existing concept organization

---
**Current Node Information:**
{current_node_snapshot}

**New Content Received:**
{extra_content_snapshot}

Please analyze the current node information and new content carefully to determine the most appropriate action based on these guidelines.

### Final Output Format
- Provide only the JSON object with the updated concept key word and abstract as shown above. Do not include any additional text, explanations, or formatting.
- Ensure that the response strictly adheres to the specified format to facilitate automated parsing and integration into the knowledge tree system.

"""

example_batch_add_action_1= AddAction(
    action="ADD",
    details=AddActionDetails(
        new_concept_key_word="Neural Architectures",
        new_concept_abstract="Fundamental patterns and structures in neural network design, including layering strategies and connectivity patterns"
    ),
    reason="Represents a distinct structural aspect of neural networks that warrants its own sub-concept"
)

example_batch_add_action_2 = AddAction(
    action="ADD",
    details=AddActionDetails(
        new_concept_key_word="Training Optimization",
        new_concept_abstract="Methods and techniques for improving neural network training efficiency and effectiveness"
    ),
    reason="Describes a specific methodological category with unique approaches and considerations"
)

example_batch_update_action_1 = UpdateAction(
    action="UPDATE",
    details=UpdateActionDetails(),
    reason="Provides practical application examples that enhance understanding of the existing concept without introducing new sub-concepts"
)

class PROMT_for_batch_action_decision_make__________________________:
    """
    This class generates prompts for making decisions on multiple content entries simultaneously,
    returning individual action decisions for each entry while only querying the LLM once.
    """
    @staticmethod
    def format(current_node_snapshot: str, extra_contents: list[str]) -> str:
        # Validate input
        if not 3 <= len(extra_contents) <= 5:
            raise ValueError("Batch processing requires 3-5 content entries")

        # Generate numbered content sections
        content_sections = "\n\n".join([
            f"Content Entry #{i+1}:\n{content}"
            for i, content in enumerate(extra_contents)
        ])

        return f"""# Batch Analysis Task

You are analyzing a knowledge tree node to determine necessary actions for multiple pieces of new content, ensuring proper maintenance and enhancement of the hierarchical knowledge structure.

**Objective:**
Analyze multiple content entries simultaneously and determine the most appropriate action for each, maintaining consistency across decisions while preserving the knowledge tree's hierarchical integrity.

### Available Actions:
1. **[UPDATE]** - Enhance existing concept
2. **[ADD]** - Create new sub-concept

### Action Templates:

For [UPDATE]:
```
{template_update_action.json_snapshot()}
```

For [ADD]:
```
{template_add_action.json_snapshot()}
```

### Batch Processing Guidelines:

1. **Consistency Analysis:**
   - Look for patterns across content entries
   - Identify related or overlapping information
   - Ensure consistent abstraction levels in decisions

2. **Relationship Consideration:**
   - Consider how multiple additions might relate to each other
   - Evaluate potential connections between new sub-concepts
   - Maintain logical grouping of related concepts

### Response Format:

Provide your analysis as a list of individual action decisions, one for each content entry. Each decision should follow the exact same format as the single-entry version:

```json
[
    {template_update_action.json_snapshot()},
    {template_add_action.json_snapshot()},
    {template_update_action.json_snapshot()}
]
```

### Example Batch Analysis:

Consider these entries about Machine Learning:

1. Content about neural network architectures
2. Content about training optimization techniques
3. Content about specific applications

Example Response:
```json
[
    {example_batch_add_action_1.json_snapshot()},
    {example_batch_add_action_2.json_snapshot()},
    {example_batch_update_action_1.json_snapshot()}
]
```

### Current Node Information:
{current_node_snapshot}

### New Content Entries to Analyze:
{content_sections}

Please analyze all content entries and provide a list of individual action decisions following the format above. Each decision should be independent and follow the same format as the single-entry version, while considering the context of other entries for consistency.

"""

if __name__ == "__main__":
    # Example usage of both prompts
    print(PROMT_for_action_decision_make.format(current_node_snapshot="", extra_content_snapshot=""))
    print("\nBatch version example:")
    print(PROMT_for_batch_action_decision_make.format(
        current_node_snapshot="",
        extra_contents=[
            "Content 1",
            "Content 2",
            "Content 3"
        ]
    ))