
# Multi-Agent AI System - Capstone Project

## Overview 

An **Autonomous Multi-Agent AI System** built using LangGraph, LangChain, and Groq. This system demonstrates advanced agentic AI concepts including task decomposition, tool-use, planning, and multi-agent orchestration.

## System Architecture

The system consists of four specialized agents working in a coordinated workflow:

```
User Query → Planner → Executor → Analyzer → Summarizer → Final Answer
```

### Agents

| Agent | Role | Tools |
|-------|------|-------|
| **Planner** | Decomposes complex tasks into subtasks | None |
| **Executor** | Performs specific tasks using tools | calculate, search_web, data_analyzer, code_generator |
| **Analyzer** | Evaluates results and extracts insights | None |
| **Summarizer** | Compiles final comprehensive output | None |

### Tools

1. **calculate** - Performs mathematical calculations
2. **search_web** - Simulates web search functionality
3. **data_analyzer** - Analyzes data and provides insights
4. **code_generator** - Generates code snippets based on task descriptions

## Technology Stack

- **LangGraph**: Multi-agent orchestration and state management
- **LangChain**: Agent framework and tool integration
- **Groq**: LLM inference (openai/gpt-oss-120b)
- **Gradio**: Interactive user interface
- **Matplotlib/Seaborn**: Performance visualization
- **Python 3.8+**: Core programming language

## Features

- ✅ **Multi-Agent Collaboration** - Agents work together to solve complex tasks
- ✅ **Tool Integration** - Agents can use specialized tools for specific tasks
- ✅ **Task Planning** - Automatic decomposition of complex queries
- ✅ **Real-Time Processing** - Fast response times with Groq inference
- ✅ **Interactive UI** - Professional Gradio interface for testing
- ✅ **Performance Analytics** - Comprehensive evaluation and visualization
- ✅ **100% Success Rate** - Verified on 8 diverse test cases

## Installation

### Prerequisites

- Python 3.8 or higher
- Groq API key (free tier available)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-agent-ai-system.git
cd multi-agent-ai-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Groq API key:
```python
# In the notebook, Cell 3 will prompt you for your API key
# Or set environment variable:
export GROQ_API_KEY="your-api-key-here"
```

## Usage

### Running the System

1. Open the Jupyter notebook:
```bash
jupyter notebook multi_agent_system.ipynb
```

2. Run cells in order from 1 to 15
3. The Gradio interface will launch automatically

### Interactive Testing

Once the Gradio interface is launched, you can test queries like:

```
Calculate 2 * 4
what is 10 + 20
Calculate compound interest for $10,000 at 5% for 3 years
Analyze recent sales trends
Create a Python function to sort a list
Explain the benefits of AI in healthcare
```

### Example Outputs

**Input:** "what is 2*4"
```
Final Answer: The answer is: 8
Task Plan: Calculate: 2*4
Completed Tasks: ✓ Calculate: 2*4
Execution Time: 1.80s
```

**Input:** "Calculate compound interest for $10,000 at 5% for 3 years"
```
Final Answer: The compound interest is $1,576.25
Task Plan: 
  1. Calculate compound interest
  2. Provide explanation
Completed Tasks: ✓ Calculate compound interest
Execution Time: 2.10s
```

## Evaluation Results

The system was evaluated on 8 diverse test cases with the following results:

| Metric | Value |
|--------|-------|
| **Success Rate** | 100.0% |
| **Average Execution Time** | 0.12 seconds |
| **Average Tasks Completed** | 2.0 per query |
| **Total Test Cases** | 8 |
| **Successful Queries** | 8 |
| **Failed Queries** | 0 |

### Test Cases

1. Calculate compound interest for $10,000 at 5% for 3 years
2. Analyze recent sales trends and predict future growth
3. Create a Python function to sort a list of numbers
4. Explain benefits of AI in healthcare with data analysis
5. Plan a marketing strategy for a new tech product launch
6. Analyze customer feedback and suggest improvements
7. Generate code for a weather API integration
8. Evaluate market conditions for renewable energy investments

## Project Structure

```
multi-agent-ai-system/
├── multi_agent_system.ipynb    # Main notebook with all code
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── system_performance.png       # Performance visualizations
├── performance_summary.png      # Summary metrics chart
├── evaluation_results.csv       # Test case results
├── system_metrics.json          # Performance metrics
├── detailed_results.json        # Detailed evaluation data
└── multi_agent_system_capstone.zip # Complete project package
```

## Code Walkthrough

### Cell 1: Setup and Installations
Installs required packages for the project.

### Cell 2: Import Libraries
Imports all necessary libraries including LangGraph, LangChain, and Gradio.

### Cell 3: API Key Configuration
Securely prompts for and configures the Groq API key.

### Cell 4: Define Tools
Creates specialized tools for the agents to use.

### Cell 5: Define Agent State
Sets up the state schema for the multi-agent system.

### Cell 6: Initialize Language Models
Configures LLM models using Groq's openai/gpt-oss-120b.

### Cell 7: Create Specialized Agents
Defines each agent with its specific role and tools.

### Cell 8: Build Orchestration Graph
Creates the LangGraph workflow for agent coordination.

### Cell 9: Create Test Cases
Prepares 8 diverse test queries for evaluation.

### Cell 10: Evaluation Framework
Implements system evaluation and performance metrics.

### Cell 11: Visualization Functions
Creates performance visualizations using matplotlib/seaborn.

### Cell 12: Gradio UI Interface
Builds the professional user interface.

### Cell 13: Main Execution Pipeline
Runs the complete evaluation pipeline.

### Cell 14: Launch Gradio Interface
Starts the interactive UI.

### Cell 15: Export Package
Creates the final zip package of all deliverables.

## Customization

### Adding New Tools

1. Define a new tool function with the `@tool` decorator in Cell 4:
```python
@tool
def new_tool(param: str) -> str:
    """Tool description"""
    # Implementation
    return result
```

2. Add the tool to the available tools list:
```python
available_tools = [calculate, search_web, data_analyzer, code_generator, new_tool]
```

3. Update the Executor agent's system prompt to mention the new tool.

### Adding New Agents

1. Create a new agent function in Cell 7:
```python
def create_new_agent(llm):
    system_prompt = """Your system prompt here"""
    return Agent("NewAgent", llm, [], system_prompt)
```

2. Add the agent to the MultiAgentSystem initialization in Cell 8.

```
