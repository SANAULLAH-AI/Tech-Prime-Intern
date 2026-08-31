# Week 8: Agentic AI (Capstone) - Complete Study Notes

---

## Table of Contents

1. Introduction to Agentic AI
2. Agent Architecture Fundamentals
3. Agent Frameworks: LangGraph & LangChain
4. Agent Frameworks: CrewAI
5. Tool Use and Function Calling
6. Agentic Workflows and Patterns
7. Planning and Reasoning
8. Multi-Agent Orchestration
9. Memory and State Management
10. Human-in-the-Loop
11. Evaluation and Monitoring
12. Complete Project: Autonomous Multi-Agent AI System
13. 2026 Modern Agentic AI Trends

---

## 1. Introduction to Agentic AI

### 1.1 What is Agentic AI?

Agentic AI refers to systems where LLMs act as autonomous agents that can plan, use tools, collaborate with other agents, and complete complex tasks with minimal human intervention.

```text
AGENTIC AI DEFINITION:
═══════════════════════════════════════════════════════════════

An AI agent is a system that:
1. Uses an LLM as its reasoning engine
2. Can call external tools and APIs
3. Makes autonomous decisions
4. Plans and executes multi-step workflows
5. Maintains state and memory
6. Collaborates with other agents
7. Learns from feedback

KEY CHARACTERISTICS:
────────────────────────────────────────────────────────────
- Autonomy: Operates without constant human input
- Tool Use: Invokes external functions and APIs
- Planning: Breaks down complex tasks into steps
- Memory: Maintains context across interactions
- Collaboration: Works with other agents
- Adaptability: Adjusts based on feedback
```

### 1.2 Why Agentic AI?

```text
AGENTIC AI BENEFITS:
═══════════════════════════════════════════════════════════════

1. COMPLEX TASK EXECUTION
   - Break down complex problems into manageable steps
   - Execute multi-stage workflows
   - Handle dynamic and changing requirements

2. TOOL ORCHESTRATION
   - Combine multiple tools and APIs
   - Use the right tool at the right time
   - Handle tool dependencies

3. AUTONOMOUS OPERATION
   - Reduce human intervention
   - Scale operations
   - Handle repetitive tasks

4. COLLABORATION
   - Specialized agents for specific tasks
   - Parallel execution of subtasks
   - Knowledge sharing between agents

5. CONTINUOUS IMPROVEMENT
   - Learn from feedback
   - Adapt to new situations
   - Improve over time
```

### 1.3 The Hallmarks of Agentic AI Solutions

A comprehensive agentic solution should demonstrate :

```text
HALLMARKS OF AGENTIC AI:
═══════════════════════════════════════════════════════════════

1. PROBLEM DECOMPOSITION
   Breaking a larger problem into smaller steps carried out by individual processes or models

2. TOOL USE
   Using function calling to interact with external systems

3. COLLABORATIVE ENVIRONMENT
   An environment in which agents can collaborate

4. COORDINATION
   A planning agent that coordinates activities

5. AUTONOMY AND MEMORY
   Existing beyond a single chat session with a human
```

### 1.4 Agent Paradigms

```text
THREE PROMINENT PARADIGMS :
═══════════════════════════════════════════════════════════════

1. TOOL USE
   - Agents invoke external tools to extend capabilities
   - Examples: search, calculation, API calls
   - Focus: Accessing and utilizing external resources

2. PLANNING (including RAG)
   - Agents plan sequences of actions
   - Retrieve relevant information before acting
   - Focus: Structured task execution

3. FEEDBACK LEARNING
   - Agents learn from outcomes
   - Adapt based on success/failure
   - Focus: Continuous improvement
```

---

## 2. Agent Architecture Fundamentals

### 2.1 Core Agent Components

```text
AGENT ARCHITECTURE:
═══════════════════════════════════════════════════════════════

CORE COMPONENTS:
────────────────────────────────────────────────────────────
1. LLM (Reasoning Engine)
   - The brain of the agent
   - Handles reasoning and decision-making
   - Generates plans and actions

2. Tools
   - External functions the agent can call
   - APIs, databases, search engines
   - Any external capability

3. State/Memory
   - Context across interactions
   - Short-term and long-term memory
   - Conversation history

4. Planner
   - Breaks down tasks into steps
   - Determines sequence of actions
   - Adapts to changing conditions

5. Executor
   - Carries out planned actions
   - Calls tools and APIs
   - Handles execution results

6. Evaluator
   - Assesses action outcomes
   - Determines if goals are met
   - Provides feedback for improvement

7. Router
   - Directs tasks to appropriate agents/tools
   - Routes based on task type and context
   - Orchestrates agent collaboration
```

### 2.2 The Plan-Act-Observe-Reflect Loop

The fundamental agentic workflow follows a continuous loop :

```text
PLAN-ACT-OBSERVE-REFLECT LOOP:
═══════════════════════════════════════════════════════════════

      ┌──────────────────────────────────────────────┐
      │                                              │
      │   ┌──────────┐    ┌──────────┐    ┌───────┐ │
      │   │   PLAN   │───▶│   ACT    │───▶│OBSERVE│ │
      │   └──────────┘    └──────────┘    └───────┘ │
      │        │                           │        │
      │        │         ┌──────────┐     │        │
      │        └────────▶│  REFLECT │◀────┘        │
      │                  └──────────┘               │
      │                        │                    │
      └────────────────────────┴────────────────────┘
                              │
                              ▼
                        GOAL ACHIEVED?

PLAN:
- Understand the task
- Break into subtasks
- Determine required tools

ACT:
- Call tools
- Execute actions
- Generate responses

OBSERVE:
- Collect results
- Gather feedback
- Monitor outcomes

REFLECT:
- Evaluate progress
- Adjust strategy
- Learn from outcomes
```

### 2.3 Agent Frameworks Overview

```text
AGENT FRAMEWORKS COMPARISON:
═══════════════════════════════════════════════════════════════

┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Framework       │ Architecture     │ Best For         │ Key Features     │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ LangGraph       │ Graph-based      │ Complex flows    │ Cycles, HITL,    │
│                 │                  │                  │ state management │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ CrewAI          │ Role-based       │ Multi-agent      │ Roles, JSON      │
│                 │                  │ collaboration    │ config, CLI      │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ LangChain       │ Chain-based      │ Single agents    │ Tool calling,    │
│                 │                  │                  │ memory           │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Custom Workflow │ Bespoke graphs   │ Specialized      │ Full control,    │
│ (LangGraph)     │                  │ use cases        │ deterministic+  │
│                 │                  │                  │ agentic mixing   │
└─────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

---

## 3. Agent Frameworks: LangGraph & LangChain

### 3.1 LangGraph Overview

LangGraph is a library for building stateful, multi-actor applications with LLMs, using a graph-based architecture .

```text
LANGGRAPH CORE CONCEPTS:
═══════════════════════════════════════════════════════════════

1. STATE
   - TypedDict defining the state schema
   - Passed between nodes
   - Updated by nodes

2. NODES
   - Functions that process state
   - Can be LLM calls, tools, or deterministic functions
   - Each node receives and returns state

3. EDGES
   - Define the flow between nodes
   - Can be conditional or unconditional
   - Determine execution paths

4. GRAPH
   - The complete workflow definition
   - Composed of nodes and edges
   - Compiled into a runnable agent

BASIC PATTERNS :
────────────────────────────────────────────────────────────
1. Supervisor: Central coordinator routes tasks
2. Swarm: Agents hand off peer-to-peer
3. Human-in-the-Loop: Approval gates
4. Structured Output: JSON-schema constrained
5. RAG: Retrieval-augmented generation
6. Customer Support: Multi-step support flow
```

### 3.2 LangGraph Implementation

```python
# LANGGRAPH BASIC IMPLEMENTATION:
# ============================================================

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool

# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    iteration_count: int                      # Track steps
    classification: str                       # Task classification
    response: str                             # Final response

# 2. Define Tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except:
        return "Error evaluating expression"

search_tool = TavilySearchResults(max_results=3)

# 3. Define Agent Nodes
def classify_node(state: AgentState) -> AgentState:
    """Classify the user's request."""
    # In practice, use LLM to classify
    last_message = state["messages"][-1].content
    if "calculate" in last_message.lower():
        state["classification"] = "calculation"
    elif "search" in last_message.lower():
        state["classification"] = "search"
    else:
        state["classification"] = "general"
    return state

def tool_node(state: AgentState) -> AgentState:
    """Execute the appropriate tool."""
    classification = state.get("classification", "general")
    
    if classification == "calculation":
        # Extract expression and calculate
        result = calculator.invoke(state["messages"][-1].content)
        state["response"] = result
    elif classification == "search":
        # Search and format results
        results = search_tool.invoke(state["messages"][-1].content)
        state["response"] = str(results)
    else:
        state["response"] = "I can help with calculations or searches."
    
    return state

def respond_node(state: AgentState) -> AgentState:
    """Generate final response."""
    # Add response to messages
    state["messages"].append(state["response"])
    return state

# 4. Build Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("classify", classify_node)
workflow.add_node("tool_execution", tool_node)
workflow.add_node("respond", respond_node)

# Define edges
workflow.add_edge(START, "classify")
workflow.add_edge("classify", "tool_execution")
workflow.add_edge("tool_execution", "respond")
workflow.add_edge("respond", END)

# 5. Compile
agent = workflow.compile()

# 6. Use the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "Calculate 15 * 37"}],
    "iteration_count": 0,
    "classification": "",
    "response": ""
})

print(result["response"])
```

### 3.3 Conditional Routing in LangGraph

Conditional edges allow dynamic routing based on state :

```python
# CONDITIONAL ROUTING EXAMPLE:
# ============================================================

from typing import Literal

def route_by_classification(state: AgentState) -> Literal["calculation", "search", "general"]:
    """Route based on classification."""
    classification = state.get("classification", "general")
    
    if classification == "calculation":
        return "calculation"
    elif classification == "search":
        return "search"
    return "general"

# Build graph with conditional edges
workflow = StateGraph(AgentState)

# Add nodes for each route
workflow.add_node("classify", classify_node)
workflow.add_node("calculation_node", tool_node)
workflow.add_node("search_node", tool_node)
workflow.add_node("general_node", respond_node)

# Conditional routing
workflow.add_edge(START, "classify")
workflow.add_conditional_edges(
    "classify",
    route_by_classification,
    {
        "calculation": "calculation_node",
        "search": "search_node",
        "general": "general_node"
    }
)

# All routes end
workflow.add_edge("calculation_node", END)
workflow.add_edge("search_node", END)
workflow.add_edge("general_node", END)

agent = workflow.compile()
```

### 3.4 LangGraph Agent Starter Kit

The LangGraph Starter Kit provides multiple agent patterns with modern APIs :

```text
LANGGRAPH STARTER KIT FEATURES:
═══════════════════════════════════════════════════════════════

QUICK START:
────────────────────────────────────────────────────────────
npx create-langgraph-app

INTERACTIVE CLI:
1. Choose project name
2. Select LLM provider:
   - OpenAI (gpt-4o-mini)
   - Anthropic (Claude Sonnet)
   - Google (Gemini 2.0 Flash)
   - Groq (Llama 3.3)
   - DeepSeek
   - Ollama (local)

3. Choose agent patterns:
   - Supervisor
   - Swarm
   - Human-in-the-Loop
   - Structured Output
   - RAG

4. Get a fully scaffolded project

PROVIDERS :
────────────────────────────────────────────────────────────
┌──────────────┬──────────────────────┬──────────────────┐
│ Provider     │ Default Model        │ API Key          │
├──────────────┼──────────────────────┼──────────────────┤
│ OpenAI       │ gpt-4o-mini          │ OPENAI_API_KEY   │
│ Anthropic    │ claude-sonnet-4      │ ANTHROPIC_API_KEY│
│ Google       │ gemini-2.0-flash     │ GOOGLE_API_KEY   │
│ Groq         │ llama-3.3-70b        │ GROQ_API_KEY     │
│ DeepSeek     │ deepseek-v4-flash    │ DEEPSEEK_API_KEY │
│ Ollama       │ llama3.2             │ None (local)     │
└──────────────┴──────────────────────┴──────────────────┘
```

### 3.5 Custom Workflow in LangGraph

LangGraph allows you to define custom execution flows with complete control :

```python
# CUSTOM WORKFLOW WITH LANGGRAPH:
# ============================================================

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

class CustomState(TypedDict):
    question: str
    rewritten_query: str
    documents: list[str]
    answer: str

# Agent node that invokes LangChain agent directly
agent = create_agent(model="openai:gpt-4o-mini", tools=[search_tool])

def agent_node(state: CustomState) -> dict:
    """LangGraph node that invokes a LangChain agent."""
    result = agent.invoke({
        "messages": [{"role": "user", "content": state["question"]}]
    })
    return {"answer": result["messages"][-1].content}

def rewrite_node(state: CustomState) -> dict:
    """Rewrite query for better retrieval."""
    # LLM call to rewrite query
    # In practice, use a model call
    state["rewritten_query"] = state["question"] + " (rewritten)"
    return state

def retrieve_node(state: CustomState) -> dict:
    """Vector similarity search - no LLM involved."""
    # Deterministic retrieval
    # In practice, query vector database
    state["documents"] = ["Document 1", "Document 2"]
    return state

# Build workflow
workflow = (
    StateGraph(CustomState)
    .add_node("rewrite", rewrite_node)
    .add_node("retrieve", retrieve_node)
    .add_node("agent", agent_node)
    .add_edge(START, "rewrite")
    .add_edge("rewrite", "retrieve")
    .add_edge("retrieve", "agent")
    .add_edge("agent", END)
    .compile()
)
```

---

## 4. Agent Frameworks: CrewAI

### 4.1 CrewAI Overview

CrewAI is a framework for orchestrating role-playing, autonomous AI agents that collaborate to complete tasks .

```text
CREWAI CORE CONCEPTS:
═══════════════════════════════════════════════════════════════

1. AGENTS
   - Role: The agent's function (e.g., "Research Analyst")
   - Goal: What the agent aims to achieve
   - Backstory: Personality and context
   - Tools: External capabilities
   - LLM: The underlying model

2. TASKS
   - Description: What needs to be done
   - Expected Output: Format and content
   - Agent: Who executes the task
   - Context: Dependencies on other tasks
   - Output File: Where to save results

3. CREW
   - Collection of agents
   - Set of tasks
   - Process: Sequential or hierarchical
   - Memory: Shared memory between agents

4. PROCESS
   - Sequential: Tasks run one after another
   - Hierarchical: Manager oversees execution

5. JSON CONFIGURATION
   - Agents defined in agents/*.jsonc
   - Tasks defined in crew.jsonc
   - CLI-driven workflow
```

### 4.2 CrewAI Implementation

```python
# CREWAI BASIC IMPLEMENTATION:
# ============================================================

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# 1. Define Tools
search_tool = SerperDevTool()

# 2. Define Agents 
researcher = Agent(
    role="Research Analyst",
    goal="Find comprehensive, accurate information on the given topic",
    backstory=(
        "You are an expert research analyst with deep knowledge across many domains. "
        "You excel at finding and synthesizing information from multiple sources."
    ),
    tools=[search_tool],
    verbose=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear, concise reports from research findings",
    backstory=(
        "You are a skilled technical writer who can transform complex research "
        "into clear, readable reports for a technical audience."
    ),
    verbose=True,
)

# 3. Define Tasks
research_task = Task(
    description="Research the topic: {topic}. Find key facts, recent developments, and expert opinions.",
    expected_output="A structured summary of findings with bullet points and sources.",
    agent=researcher,
)

write_task = Task(
    description="Using the research findings, write a comprehensive report on: {topic}",
    expected_output="A well-structured report with introduction, key findings, and conclusion.",
    agent=writer,
    context=[research_task],  # writer receives research_task output
)

# 4. Create Crew 
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,
)

# 5. Run the crew
result = crew.kickoff(inputs={"topic": "AI agent frameworks in 2025"})
print(result)
```

### 4.3 JSON-Based Crew Configuration

CrewAI supports JSON-first crew configuration :

```json
// agents/researcher.jsonc
{
  "role": "Senior Research Specialist for {topic}",
  "goal": "Find comprehensive and accurate information about {topic}",
  "backstory": "You are an experienced research specialist who organizes complex information into clear, useful notes.",
  "llm": "openai/gpt-4o-mini",
  "tools": ["SerperDevTool"],
  "settings": {
    "verbose": true,
    "allow_delegation": false
  }
}

// crew.jsonc
{
  "name": "Research Crew",
  "agents": ["researcher", "analyst"],
  "tasks": [
    {
      "id": "research_task",
      "description": "Conduct thorough research on {topic}.",
      "expected_output": "A comprehensive research document.",
      "agent": "researcher"
    },
    {
      "id": "analysis_task",
      "description": "Analyze findings and create a report.",
      "expected_output": "A professional markdown report.",
      "agent": "analyst",
      "context": ["research_task"],
      "output_file": "output/report.md"
    }
  ],
  "process": "sequential",
  "memory": true
}
```

### 4.4 CrewAI vs LangGraph

```text
FRAMEWORK COMPARISON:
═══════════════════════════════════════════════════════════════

┌──────────────┬────────────────────┬─────────────────────┐
│ Aspect       │ CrewAI             │ LangGraph           │
├──────────────┼────────────────────┼─────────────────────┤
│ Architecture │ Role-based         │ Graph-based         │
│ Focus        │ Multi-agent teams  │ Complex flows       │
│ Setup        │ JSON config, CLI   │ Python code         │
│ Memory       │ Built-in shared    │ State-based         │
│ HITL         │ Limited            │ Native support      │
│ Complexity   │ Simpler            │ More flexible       │
│ Use Case     │ Research teams     │ Production agents   │
└──────────────┴────────────────────┴─────────────────────┘

WHEN TO USE CREWAI:
────────────────────────────────────────────────────────────
- Well-defined, structured workflows
- Specialized agent roles
- Research and report generation
- Sequential task execution
- Quick prototyping

WHEN TO USE LANGGRAPH:
────────────────────────────────────────────────────────────
- Complex conditional flows
- Production-grade applications
- Human-in-the-loop needed
- Custom state management
- Integration with existing systems
```

---

## 5. Tool Use and Function Calling

### 5.1 Tool Calling Architecture

Tools extend agent capabilities by allowing interaction with external systems :

```text
TOOL CALLING ARCHITECTURE:
═══════════════════════════════════════════════════════════════

USER
  ↓
AGENT (LLM)
  ↓
DECISION: Tool needed?
  ↓ Yes
TOOL CALL FORMAT
  {
    "name": "tool_name",
    "arguments": {"param1": "value1"}
  }
  ↓
EXECUTION ENVIRONMENT
  ↓
TOOL EXECUTION
  - API call
  - Database query
  - Computation
  - External service
  ↓
TOOL RESULT
  ↓
AGENT (LLM)
  ↓
DECISION: More tools or final answer?
  ↓
FINAL RESPONSE
```

### 5.2 Defining Tools

```python
# TOOL DEFINITION IN LANGCHAIN:
# ============================================================

from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for information about a query."""
    # Implementation using Tavily, Serper, etc.
    return search_results

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    # Safe evaluation
    try:
        return str(eval(expression))
    except:
        return "Error evaluating expression"

@tool
def add_memory(memory: str) -> str:
    """Add a memory to the agent's memory database."""
    # Store memory
    return "Memory added successfully"

@tool
def retrieve_memory(query: str) -> str:
    """Retrieve memory from the agent's memory database."""
    # Search memory
    return "Retrieved memory"

@tool
def database_query(sql: str) -> list:
    """Execute a SQL query on the database."""
    # Safe SQL execution
    return results
```

### 5.3 Tool Execution Pattern

```python
# TOOL EXECUTION WITH STATE:
# ============================================================

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: list
    iteration_count: int

# Tool node in LangGraph
tool_node = ToolNode(tools=[web_search, calculator, add_memory])

# Tool execution loop
def tool_execution_loop(state: AgentState) -> AgentState:
    """Execute tools and process results."""
    # Get the current tool call
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            # Execute each tool
            result = execute_tool(tool_call)
            # Add result to messages
            state["messages"].append(result)
    
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    return state
```

### 5.4 Safe Tool Execution

Production agents require safety measures :

```python
# SAFE TOOL EXECUTION:
# ============================================================

import ast
import operator as op

class SafeCalculator:
    """Safe mathematical expression evaluator using AST."""
    
    _SAFE_OPS = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
        ast.UAdd: op.pos, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    }
    
    _MAX_EXPONENT = 1000  # Guard against DoS
    
    def _safe_eval_expr(self, node):
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Expression):
            return self._safe_eval_expr(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in self._SAFE_OPS:
            left = self._safe_eval_expr(node.left)
            right = self._safe_eval_expr(node.right)
            if isinstance(node.op, ast.Pow):
                if not isinstance(right, (int, float)) or abs(right) > self._MAX_EXPONENT:
                    raise ValueError(f"Exponent {right} exceeds maximum")
            return self._SAFE_OPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in self._SAFE_OPS:
            return self._SAFE_OPS[type(node.op)](self._safe_eval_expr(node.operand))
        else:
            raise ValueError(f"Unsupported expression node: {ast.dump(node)}")
    
    def evaluate(self, expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        try:
            tree = ast.parse(expression, mode='eval')
            result = self._safe_eval_expr(tree)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
```

---

## 6. Agentic Workflows and Patterns

### 6.1 Common Agent Patterns

```text
AGENT PATTERNS :
═══════════════════════════════════════════════════════════════

1. SUPERVISOR
   ┌─────────────────────────────────────┐
   │            SUPERVISOR AGENT         │
   │         (Central Coordinator)       │
   └──────────┬──────────┬───────────────┘
              │          │
         ┌────▼───┐ ┌────▼────┐ ┌───────┐
         │Worker 1│ │Worker 2 │ │Worker 3│
         └────────┘ └─────────┘ └───────┘

2. SWARM
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Agent A │──▶│ Agent B │──▶│ Agent C │
   └─────────┘   └─────────┘   └─────────┘
      Peer-to-peer handoffs

3. HUMAN-IN-THE-LOOP
   ┌───────┐   ┌───────┐   ┌───────┐
   │Agent  │──▶│Human  │──▶│Agent  │
   └───────┘   └───────┘   └───────┘
      Approval gate

4. RAG
   ┌───────┐   ┌───────┐   ┌───────┐
   │Query  │──▶│Retrie │──▶│Agent  │
   └───────┘   └───────┘   └───────┘
      + documents

5. CUSTOM WORKFLOW
   ┌─────────────────────────────────┐
   │    Deterministic + Agentic      │
   │    Conditional Routing          │
   │    Parallel Execution           │
   └─────────────────────────────────┘
```

### 6.2 Customer Support Agent Pattern

A complete customer support agent workflow :

```python
# CUSTOMER SUPPORT AGENT:
# ============================================================

from typing import Literal, Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    classification: str
    response: str

def classify_node(state: SupportState) -> SupportState:
    """Classify the customer message."""
    last_message = state["messages"][-1].content
    
    # Determine classification using LLM
    classification = classify_message(last_message)
    state["classification"] = classification
    
    return state

def rag_search_node(state: SupportState) -> SupportState:
    """Search knowledge base for relevant information."""
    last_message = state["messages"][-1].content
    # RAG search on support documents
    context = rag_search(last_message)
    state["response"] = context
    return state

def respond_node(state: SupportState) -> SupportState:
    """Generate response based on RAG context."""
    response = generate_response(state["response"], state["messages"][-1].content)
    state["messages"].append(response)
    return state

def create_ticket_node(state: SupportState) -> SupportState:
    """Create a support ticket."""
    last_message = state["messages"][-1].content
    ticket_id = create_ticket(last_message)
    state["response"] = f"Ticket created: {ticket_id}"
    return state

def escalate_node(state: SupportState) -> SupportState:
    """Escalate to human agent."""
    last_message = state["messages"][-1].content
    ticket_id = escalate_to_human(last_message)
    escalation_message = (
        "I've escalated your case to a senior support specialist "
        f"(Reference: {ticket_id}). A human agent will reach out "
        "within the next 2 hours."
    )
    state["response"] = escalation_message
    state["messages"].append(escalation_message)
    return state

def route_by_classification(state: SupportState) -> Literal["rag_search", "escalate"]:
    """Route based on classification."""
    classification = state.get("classification", "general")
    if classification in ("sensitive", "escalate"):
        return "escalate"
    return "rag_search"

def route_after_rag(state: SupportState) -> Literal["create_ticket", "respond"]:
    """After RAG, decide if we need a ticket."""
    classification = state.get("classification", "general")
    if classification in ("technical", "complaint"):
        return "create_ticket"
    return "respond"

# Build graph
workflow = StateGraph(SupportState)

workflow.add_node("classify", classify_node)
workflow.add_node("rag_search", rag_search_node)
workflow.add_node("respond", respond_node)
workflow.add_node("create_ticket", create_ticket_node)
workflow.add_node("escalate", escalate_node)

workflow.add_edge(START, "classify")

workflow.add_conditional_edges(
    "classify",
    route_by_classification,
    {
        "rag_search": "rag_search",
        "escalate": "escalate"
    }
)

workflow.add_conditional_edges(
    "rag_search",
    route_after_rag,
    {
        "create_ticket": "create_ticket",
        "respond": "respond"
    }
)

workflow.add_edge("create_ticket", "respond")
workflow.add_edge("respond", END)
workflow.add_edge("escalate", END)

support_agent = workflow.compile()
```

### 6.3 ReAct Pattern

The ReAct (Reason + Act) pattern is fundamental to agentic AI:

```text
REACT PATTERN:
═══════════════════════════════════════════════════════════════

ITERATION 1:
────────────────────────────────────────────────────────────
Thought: I need to find information about this topic
Action: search_tool("query")
Observation: Search results...

ITERATION 2:
────────────────────────────────────────────────────────────
Thought: I have the information I need
Action: None (final answer)
Observation: Final response

IMPLEMENTATION :
────────────────────────────────────────────────────────────
# The agent interleaves thought, tool_name, and tool_args

[[ ## thought_0 ## ]]
I need to gather information and check the user's preferences.

[[ ## tool_name_0 ## ]]
retrieve_memory

[[ ## tool_args_0 ## ]]
{"query": "user preferences"}

[[ ## thought_1 ## ]]
Now I have the user's preferred style. I will gather specific information.

[[ ## tool_name_1 ## ]]
search_tool

[[ ## tool_args_1 ## ]]
{"query": "topic information"}

[[ ## thought_2 ## ]]
I have all the information needed. I can now provide the final answer.

[[ ## tool_name_2 ## ]]
finish

[[ ## tool_args_2 ## ]]
{"answer": "Final response here"}
```

---

## 7. Planning and Reasoning

### 7.1 Planning in Agentic Systems

Advanced agent planning involves decomposing tasks and determining optimal execution paths .

```text
PLANNING ARCHITECTURE:
═══════════════════════════════════════════════════════════════

1. TASK DECOMPOSITION
   Complex Task
        ↓
   ┌───┼───┐
   ▼   ▼   ▼
  T1  T2  T3
   |   |   |
   ▼   ▼   ▼
  S1  S2  S3

2. DEPENDENCY ANALYSIS
   - Identify dependencies between subtasks
   - Determine parallel vs. sequential execution
   - Build execution graph

3. RESOURCE ALLOCATION
   - Assign tools to tasks
   - Allocate agents to tasks
   - Manage resource contention

4. EXECUTION PLANNING
   - Determine optimal order
   - Handle dynamic changes
   - Plan for contingencies
```

### 7.2 Graph-Based Planning

Modern agentic systems use graph-based planning for complex tasks :

```text
GRAPH-BASED PLANNING :
═══════════════════════════════════════════════════════════════

NAVIAGENT ARCHITECTURE:
────────────────────────────────────────────────────────────
                    ┌─────────────────┐
                    │   Planning      │
                    │   Level         │
                    │                 │
                    │  LLM-based      │
                    │  Agent decides  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Tool World     │
                    │  Navigation     │
                    │  Model (TWNM)   │
                    │                 │
                    │  Graph-based    │
                    │  tool relations │
                    └─────────────────┘

KEY FEATURES:
────────────────────────────────────────────────────────────
- Bilevel architecture: Planning vs. Execution
- Graph-based modeling of tool relations
- Structural and behavioral relations
- Closed-loop feedback
- Adaptive navigation

BENEFITS:
────────────────────────────────────────────────────────────
- 13.1% average improvement in Task Success Rate
- 4.3-12.0% gains across 7 domains
- Fewer steps and lower latency
- Better scalability with large tool sets
```

### 7.3 Graph-Based Planning with Parallel Execution 

```text
GAP (GRAPH-BASED AGENT PLANNING) :
═══════════════════════════════════════════════════════════════

PARALLEL TOOL EXECUTION:
────────────────────────────────────────────────────────────
Traditional ReAct (Sequential):
  Step 1 → Tool A → Wait → Step 2 → Tool B → Wait → Step 3

GAP (Parallel):
  ┌──────────────┐
  │  Task Graph  │
  │  Dependency  │
  │  Analysis    │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Parallel    │  ┌──────────────┐
  │  Tool A      │  │  Tool B      │
  └──────────────┘  └──────────────┘
         │                   │
         └────────┬──────────┘
                  ▼
         ┌──────────────┐
         │  Sequential  │
         │  Tool C      │
         └──────────────┘

TRAINING APPROACH :
────────────────────────────────────────────────────────────
1. Supervised Fine-Tuning (SFT)
   - Curated graph-based planning traces
   - MHQA benchmark derived

2. Reinforcement Learning (RL)
   - Correctness-based reward function
   - Strategic query sampling
   - Value-maximizing tool-based reasoning
```

### 7.4 Planning Implementation

```python
# PLANNING IMPLEMENTATION:
# ============================================================

class TaskPlanner:
    """Plan and schedule tasks for agent execution."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def plan(self, goal: str, tools: list) -> dict:
        """Create a plan for achieving the goal."""
        
        # 1. Decompose goal into subtasks
        subtasks = self._decompose(goal)
        
        # 2. Determine dependencies
        dependencies = self._analyze_dependencies(subtasks)
        
        # 3. Assign tools to subtasks
        tool_assignments = self._assign_tools(subtasks, tools)
        
        # 4. Determine execution order
        execution_plan = self._schedule(subtasks, dependencies)
        
        return {
            'goal': goal,
            'subtasks': subtasks,
            'dependencies': dependencies,
            'tool_assignments': tool_assignments,
            'execution_plan': execution_plan
        }
    
    def _decompose(self, goal: str) -> list:
        """Break down goal into subtasks."""
        prompt = f"""Break down the following goal into 3-5 specific subtasks:
        Goal: {goal}
        List the subtasks:"""
        response = self.llm.invoke(prompt)
        return self._parse_subtasks(response)
    
    def _analyze_dependencies(self, subtasks: list) -> dict:
        """Analyze dependencies between subtasks."""
        # Build dependency graph
        dependencies = {}
        for i, task in enumerate(subtasks):
            dependencies[task] = []
            # Determine which tasks must precede this one
        return dependencies
    
    def _assign_tools(self, subtasks: list, tools: list) -> dict:
        """Assign tools to subtasks."""
        assignments = {}
        for task in subtasks:
            # Determine which tools can help with this task
            assignments[task] = self._find_relevant_tools(task, tools)
        return assignments
    
    def _schedule(self, subtasks: list, dependencies: dict) -> list:
        """Create execution schedule."""
        # Topological sort of dependency graph
        scheduled = []
        remaining = set(subtasks)
        
        while remaining:
            ready = []
            for task in remaining:
                deps = [d for d in dependencies[task] if d in remaining]
                if not deps:
                    ready.append(task)
            
            # Add ready tasks to schedule
            scheduled.extend(ready)
            remaining -= set(ready)
        
        return scheduled
```

---

## 8. Multi-Agent Orchestration

### 8.1 Multi-Agent Systems

Multi-agent systems involve multiple agents collaborating to achieve goals :

```text
MULTI-AGENT ORCHESTRATION:
═══════════════════════════════════════════════════════════════

ARCHITECTURES:
────────────────────────────────────────────────────────────
1. SUPERVISOR PATTERN
   ┌─────────────────────────────────────────────────┐
   │              SUPERVISOR AGENT                   │
   │         (Central Coordinator)                   │
   └──────────┬──────────┬──────────┬───────────────┘
              │          │          │
         ┌────▼───┐ ┌────▼────┐ ┌──▼──────┐
         │ Agent 1│ │ Agent 2 │ │ Agent 3 │
         └────────┘ └─────────┘ └─────────┘

2. SWARM PATTERN 
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Agent A │──▶│ Agent B │──▶│ Agent C │
   └─────────┘   └─────────┘   └─────────┘
      (Peer-to-peer handoff)

3. HIERARCHICAL
   ┌─────────────────────────────────────────────────┐
   │              MANAGER AGENT                      │
   └──────────┬──────────┬──────────┬───────────────┘
              │          │          │
         ┌────▼───┐ ┌────▼────┐ ┌──▼──────┐
         │ Team 1 │ │ Team 2  │ │ Team 3  │
         └────┬───┘ └────┬────┘ └──┬──────┘
              │          │          │
         ┌────▼───┐ ┌────▼────┐ ┌──▼──────┐
         │ Worker │ │ Worker │ │ Worker  │
         └────────┘ └─────────┘ └─────────┘
```

### 8.2 Supervisor Pattern Implementation

```python
# SUPERVISOR PATTERN WITH LANGGRAPH:
# ============================================================

from langgraph.graph import StateGraph, END
from typing import Literal

class SupervisorState(TypedDict):
    messages: list
    next_agent: str
    task: str
    results: dict

# Define specialized agents
def researcher_node(state: SupervisorState) -> SupervisorState:
    """Research agent - finds information."""
    # Research logic
    state["results"]["research"] = "Research findings"
    return state

def analyst_node(state: SupervisorState) -> SupervisorState:
    """Analyst agent - analyzes data."""
    # Analysis logic
    state["results"]["analysis"] = "Analysis results"
    return state

def writer_node(state: SupervisorState) -> SupervisorState:
    """Writer agent - creates content."""
    # Writing logic
    state["results"]["writing"] = "Final content"
    return state

def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Supervisor - routes to appropriate agent."""
    # Determine which agent to use next
    state["next_agent"] = self._select_agent(state)
    return state

def _select_agent(state: SupervisorState) -> Literal["researcher", "analyst", "writer", "finish"]:
    """Select the next agent to invoke."""
    task = state.get("task", "")
    
    if "research" in task.lower():
        return "researcher"
    elif "analyze" in task.lower():
        return "analyst"
    elif "write" in task.lower():
        return "writer"
    
    # Check if all work is done
    if len(state.get("results", {})) >= 3:
        return "finish"
    
    return "researcher"

# Build the supervisor graph
workflow = StateGraph(SupervisorState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)

# Define edges
workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    _select_agent,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "finish": END
    }
)

# Agents return to supervisor
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("analyst", "supervisor")
workflow.add_edge("writer", "supervisor")

supervisor_agent = workflow.compile()
```

### 8.3 Swarm Pattern Implementation 

```python
# SWARM PATTERN IMPLEMENTATION:
# ============================================================

from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

class SwarmState(TypedDict):
    messages: list
    current_agent: str
    handoff: bool

# Transfer tools for handoffs
@tool
def transfer_to_agent_b(message: str) -> str:
    """Transfer the conversation to Agent B."""
    return f"Transferring to Agent B with: {message}"

@tool
def transfer_to_agent_c(message: str) -> str:
    """Transfer the conversation to Agent C."""
    return f"Transferring to Agent C with: {message}"

def agent_a_node(state: SwarmState) -> SwarmState:
    """Agent A - handles initial conversation."""
    messages = state["messages"]
    # Process with Agent A
    # If needs to hand off, use transfer tool
    return state

def agent_b_node(state: SwarmState) -> SwarmState:
    """Agent B - handles specific tasks."""
    # Process with Agent B
    return state

def agent_c_node(state: SwarmState) -> SwarmState:
    """Agent C - handles final tasks."""
    # Process with Agent C
    return state

# Build swarm graph
workflow = StateGraph(SwarmState)

workflow.add_node("agent_a", agent_a_node)
workflow.add_node("agent_b", agent_b_node)
workflow.add_node("agent_c", agent_c_node)

# Start with agent A
workflow.add_edge(START, "agent_a")

# Agents can hand off to each other
workflow.add_conditional_edges(
    "agent_a",
    lambda s: s["current_agent"],
    {
        "agent_b": "agent_b",
        "agent_c": "agent_c",
        "finish": END
    }
)

workflow.add_conditional_edges(
    "agent_b",
    lambda s: s["current_agent"],
    {
        "agent_a": "agent_a",
        "agent_c": "agent_c",
        "finish": END
    }
)

swarm_agent = workflow.compile()
```

---

## 9. Memory and State Management

### 9.1 Types of Memory

```text
AGENT MEMORY TYPES:
═══════════════════════════════════════════════════════════════

1. SHORT-TERM MEMORY
   - Current conversation context
   - In-scope information
   - Limited by context window

2. LONG-TERM MEMORY
   - Persistent across sessions
   - User preferences
   - Historical interactions

3. WORKING MEMORY
   - Current task state
   - Intermediate results
   - Active plan

4. EPISODIC MEMORY
   - Past experiences
   - Successful/unsuccessful strategies
   - Learning history

5. SEMANTIC MEMORY
   - Knowledge base
   - Facts and information
   - Domain knowledge
```

### 9.2 Memory Implementation

```python
# MEMORY IMPLEMENTATION:
# ============================================================

class AgentMemory:
    """Memory management for agents."""
    
    def __init__(self):
        self.short_term = []        # Current session
        self.long_term = {}         # Persistent storage
        self.episodic = []          # Past experiences
        self.semantic = {}          # Knowledge base
    
    def add_short_term(self, message: str):
        """Add to short-term memory."""
        self.short_term.append(message)
        if len(self.short_term) > 20:  # Limit
            self.short_term = self.short_term[-20:]
    
    def add_long_term(self, key: str, value: str):
        """Add to long-term memory."""
        self.long_term[key] = value
    
    def add_episodic(self, experience: dict):
        """Record an experience."""
        self.episodic.append(experience)
        if len(self.episodic) > 100:
            self.episodic = self.episodic[-100:]
    
    def retrieve_long_term(self, key: str) -> str:
        """Retrieve from long-term memory."""
        return self.long_term.get(key, "")
    
    def search_episodic(self, query: str) -> list:
        """Search past experiences."""
        # Simple matching - in practice use embeddings
        results = []
        for exp in self.episodic:
            if query.lower() in str(exp).lower():
                results.append(exp)
        return results[:5]
    
    def get_context(self) -> str:
        """Get full context for the agent."""
        context = []
        
        # Short-term context
        if self.short_term:
            context.append("Recent conversation:")
            context.extend(self.short_term[-5:])
        
        # Relevant long-term memories
        # In practice, retrieve based on relevance
        
        return "\n".join(context)

# Session-based memory 
session_store: dict[str, list] = {}

def get_session_memory(session_id: str) -> list:
    """Get or create session memory."""
    if session_id not in session_store:
        session_store[session_id] = []
    return session_store[session_id]

def update_session_memory(session_id: str, message: str, is_user: bool = True):
    """Update session memory."""
    memory = get_session_memory(session_id)
    role = "user" if is_user else "assistant"
    memory.append({"role": role, "content": message})
    
    # Trim to prevent unbounded growth
    max_messages = 20
    if len(memory) > max_messages:
        session_store[session_id] = memory[-max_messages:]
```

### 9.3 State Management in LangGraph

```python
# STATE MANAGEMENT IN LANGGRAPH:
# ============================================================

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Messages with automatic adding
    messages: Annotated[list, add_messages]
    
    # User context
    user_id: str
    session_id: str
    
    # Task tracking
    current_task: str
    task_history: list
    
    # Tool execution
    tool_calls: list
    tool_results: list
    
    # Agent state
    current_agent: str
    iteration_count: int
    max_iterations: int
    
    # Metadata
    timestamp: str
    confidence: float
    status: str

# State update functions
def update_state(state: AgentState, update: dict) -> AgentState:
    """Update agent state with new values."""
    for key, value in update.items():
        if key == "messages":
            state["messages"] = add_messages(state.get("messages", []), value)
        elif key == "task_history":
            state.setdefault("task_history", []).append(value)
        else:
            state[key] = value
    return state

# State persistence (simplified)
class StateStore:
    def __init__(self):
        self.states: dict[str, AgentState] = {}
    
    def save(self, session_id: str, state: AgentState):
        self.states[session_id] = state
    
    def load(self, session_id: str) -> AgentState:
        return self.states.get(session_id, {})
```

---

## 10. Human-in-the-Loop

### 10.1 HITL Architecture

Human-in-the-Loop (HITL) ensures human oversight for critical operations :

```text
HUMAN-IN-THE-LOOP ARCHITECTURE:
═══════════════════════════════════════════════════════════════

AGENT
  ↓
REVIEW NEEDED?
  ├── Yes ────▶ HUMAN APPROVAL
  │              ├── Approved ────▶ Continue
  │              └── Rejected ───▶ Revise / Abort
  └── No ──────▶ Continue

WHEN TO USE HITL:
────────────────────────────────────────────────────────────
1. Financial transactions
2. Deleting data
3. Sending external messages
4. Production deployments
5. Privilege changes
6. Irreversible operations
7. High-stakes decisions
8. Sensitive information

GUARDRAILS :
────────────────────────────────────────────────────────────
- Approval gates before high-risk actions
- Hard budget limits on token spend
- Iteration count limits
- Scope constraints for tools
- Safety checks before execution
```

### 10.2 HITL Implementation

```python
# HUMAN-IN-THE-LOOP IMPLEMENTATION:
# ============================================================

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

class HITLState(TypedDict):
    messages: list
    pending_action: str
    requires_approval: bool
    approved: bool
    response: str

# Human review node
def human_review_node(state: HITLState) -> HITLState:
    """Wait for human review and approval."""
    # In production, this would be an API endpoint
    # For demo, we simulate with a decision
    
    pending_action = state.get("pending_action", "")
    action_description = state.get("action_description", "")
    
    print(f"\n=== HUMAN REVIEW REQUIRED ===")
    print(f"Action: {pending_action}")
    print(f"Description: {action_description}")
    print("1. Approve")
    print("2. Reject")
    print("3. Modify")
    
    # In production, this would wait for external input
    # For demo, auto-approve
    
    state["approved"] = True
    state["response"] = "Action approved by human review"
    return state

# Action execution node
def execute_action_node(state: HITLState) -> HITLState:
    """Execute the approved action."""
    if state.get("approved", False):
        # Execute the pending action
        action = state["pending_action"]
        result = execute_action(action)
        state["response"] = f"Action executed: {result}"
    else:
        state["response"] = "Action rejected by human review"
    return state

# Check if approval is needed
def needs_approval(state: HITLState) -> Literal["human_review", "execute_action"]:
    """Check if the action requires human approval."""
    pending_action = state.get("pending_action", "")
    requires_approval = state.get("requires_approval", False)
    
    if requires_approval or pending_action in ["delete_data", "transfer_funds"]:
        return "human_review"
    return "execute_action"

# Build HITL graph
workflow = StateGraph(HITLState)

workflow.add_node("human_review", human_review_node)
workflow.add_node("execute_action", execute_action_node)

workflow.add_conditional_edges(
    "human_review",
    lambda s: "execute_action" if s["approved"] else END,
    {
        "execute_action": "execute_action",
        "reject": END
    }
)

workflow.add_edge("execute_action", END)

# Use checkpointer for persistence 
checkpointer = MemorySaver()
hitl_agent = workflow.compile(checkpointer=checkpointer)

# Usage
result = hitl_agent.invoke(
    {
        "pending_action": "delete_user_data",
        "requires_approval": True,
        "action_description": "Delete all data for user ID: 12345"
    },
    config={"configurable": {"thread_id": "session_1"}}
)
```

### 10.3 HITL with Customer Support 

```python
# HITL IN CUSTOMER SUPPORT:
# ============================================================

def escalate_to_human(customer_message: str, reason: str) -> str:
    """Escalate a customer issue to a human agent."""
    ticket_id = f"TICKET-{int(time.time())}"
    print(f"[Escalation] Escalated with ticket: {ticket_id}")
    print(f"[Escalation] Reason: {reason}")
    print(f"[Escalation] Message: {customer_message}")
    return ticket_id

def escalation_node(state: SupportState) -> SupportState:
    """Escalate to human agent."""
    last_message = state["messages"][-1].content
    classification = state.get("classification", "")
    
    ticket_id = escalate_to_human(
        customer_message=last_message,
        reason=f"Classified as: {classification}"
    )
    
    escalation_message = (
        "I understand this is important to you. I've escalated your case to a "
        f"senior support specialist (Reference: {ticket_id}). A human agent will "
        "reach out to you within the next 2 hours. Is there anything else I can "
        "help with in the meantime?"
    )
    
    state["response"] = escalation_message
    state["messages"].append(escalation_message)
    state["ticket_id"] = ticket_id
    
    return state
```

---

## 11. Evaluation and Monitoring

### 11.1 Agent Evaluation Metrics

```text
AGENT EVALUATION METRICS:
═══════════════════════════════════════════════════════════════

1. TASK SUCCESS RATE (TSR)
   - Percentage of tasks completed successfully
   - Primary metric for agent performance
   - Measure: Completed / Total tasks

2. TOOL CALL ACCURACY
   - Correct tool selection
   - Correct argument formatting
   - Successful tool execution

3. EFFICIENCY METRICS
   - Number of steps per task
   - Token usage per task
   - Time per task

4. QUALITY METRICS
   - Answer correctness
   - Completeness
   - Usefulness

5. SAFETY METRICS 
   - Safety violation rate
   - Unauthorized tool access
   - Data exposure incidents

6. COST METRICS
   - Cost per task
   - Token efficiency
   - API costs
```

### 11.2 Evaluation Implementation

```python
# AGENT EVALUATION:
# ============================================================

class AgentEvaluator:
    """Evaluate agent performance across multiple dimensions."""
    
    def __init__(self):
        self.metrics = {
            'task_success_rate': [],
            'avg_steps': [],
            'avg_tokens': [],
            'tool_accuracy': [],
            'safety_violations': 0
        }
    
    def evaluate_task(self, task: dict, result: dict) -> dict:
        """Evaluate a single task execution."""
        
        metrics = {}
        
        # 1. Task Success
        metrics['success'] = self._check_success(task, result)
        
        # 2. Steps
        metrics['steps'] = result.get('steps', 0)
        
        # 3. Token usage
        metrics['tokens'] = result.get('tokens', 0)
        
        # 4. Tool selection accuracy
        metrics['tool_accuracy'] = self._check_tool_accuracy(task, result)
        
        # 5. Safety check
        metrics['safe'] = self._check_safety(result)
        
        # Update aggregated metrics
        self._update_metrics(metrics)
        
        return metrics
    
    def _check_success(self, task: dict, result: dict) -> bool:
        """Check if task was completed successfully."""
        expected = task.get('expected_output', '')
        actual = result.get('answer', '')
        # In practice, use more sophisticated comparison
        return len(actual) > 10  # Placeholder
    
    def _check_tool_accuracy(self, task: dict, result: dict) -> float:
        """Check if correct tools were used."""
        expected_tools = set(task.get('required_tools', []))
        used_tools = set(result.get('tools_used', []))
        
        if not expected_tools:
            return 1.0
        
        correct = len(expected_tools & used_tools)
        return correct / len(expected_tools)
    
    def _check_safety(self, result: dict) -> bool:
        """Check for safety violations."""
        unsafe_patterns = [
            'delete', 'drop', 'truncate',  # Dangerous operations
            'password', 'secret', 'key',    # Sensitive data
            'sudo', 'admin'                 # Privileged commands
        ]
        
        response = result.get('answer', '').lower()
        for pattern in unsafe_patterns:
            if pattern in response:
                self.metrics['safety_violations'] += 1
                return False
        return True
    
    def _update_metrics(self, metrics: dict):
        """Update aggregated metrics."""
        self.metrics['task_success_rate'].append(1 if metrics['success'] else 0)
        self.metrics['avg_steps'].append(metrics['steps'])
        self.metrics['avg_tokens'].append(metrics['tokens'])
        self.metrics['tool_accuracy'].append(metrics['tool_accuracy'])
    
    def get_report(self) -> dict:
        """Generate evaluation report."""
        return {
            'task_success_rate': np.mean(self.metrics['task_success_rate']),
            'avg_steps': np.mean(self.metrics['avg_steps']),
            'avg_tokens': np.mean(self.metrics['avg_tokens']),
            'avg_tool_accuracy': np.mean(self.metrics['tool_accuracy']),
            'safety_violations': self.metrics['safety_violations'],
            'total_tasks': len(self.metrics['task_success_rate'])
        }
```

### 11.3 Observability and Monitoring 

```text
AGENT OBSERVABILITY:
═══════════════════════════════════════════════════════════════

WHAT TO TRACK:
────────────────────────────────────────────────────────────
1. Request ID
2. Agent ID
3. Model used
4. Input tokens
5. Output tokens
6. Retrieved documents (if RAG)
7. Tool calls made
8. Tool results
9. Latency per step
10. Errors and exceptions
11. Safety events
12. Final outcome

IMPLEMENTATION:
────────────────────────────────────────────────────────────
# Production trace
User → Router → Retriever → Tool → LLM → Validator → Response

# Each major step should be inspectable
- Log every tool call
- Log every token count
- Log every decision point
- Log safety checks
- Log iteration counts
```

---

## 12. Complete Project: Autonomous Multi-Agent AI System

### 12.1 Project Overview

**Goal:** Build an autonomous multi-agent system for research, analysis, and reporting.

```text
PROJECT REQUIREMENTS:
═══════════════════════════════════════════════════════════════

AGENTS:
────────────────────────────────────────────────────────────
1. Research Agent: Gathers information from multiple sources
2. Analysis Agent: Analyzes and synthesizes research
3. Writing Agent: Creates structured reports
4. Supervisor Agent: Coordinates and routes tasks

TOOLS:
────────────────────────────────────────────────────────────
1. Web Search (Tavily/Serper)
2. RAG (document retrieval)
3. Calculator
4. File I/O
5. API Calls

WORKFLOW:
────────────────────────────────────────────────────────────
1. User submits topic
2. Supervisor decomposes task
3. Research agent gathers information
4. Analysis agent synthesizes findings
5. Writing agent produces report
6. Supervisor reviews and finalizes
```

### 12.2 Complete Implementation

```python
# day5_final_multi_agent_project.ipynb

# ============================================================
# DAY 5: AUTONOMOUS MULTI-AGENT AI SYSTEM
# Complete production-ready multi-agent system
# ============================================================

# ---------------------- SETUP ----------------------
!pip install -q langgraph langchain-openai tavily-python
!pip install -q crewai crewai-tools
!pip install -q gradio

import os
import json
import time
from typing import List, Dict, Any, Literal, TypedDict, Annotated
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint import MemorySaver

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool

# ---------------------- CONFIGURATION ----------------------

class Config:
    """Configuration for the multi-agent system."""
    
    # LLM Configuration
    LLM_PROVIDER = "openai"  # openai, anthropic, google, groq, ollama
    MODEL_NAME = "gpt-4o-mini"
    
    # API Keys
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
    
    # Agent Settings
    MAX_ITERATIONS = 10
    MAX_TOKENS = 4096
    TEMPERATURE = 0.7
    
    # Search Settings
    SEARCH_MAX_RESULTS = 5
    RETRIEVAL_TOP_K = 5

# ---------------------- TOOLS ----------------------

@tool
def web_search(query: str) -> str:
    """
    Search the web for information about a query.
    Returns relevant information with sources.
    """
    if not Config.TAVILY_API_KEY:
        return "Search not available. Please set TAVILY_API_KEY."
    
    search_tool = TavilySearchResults(
        max_results=Config.SEARCH_MAX_RESULTS
    )
    
    try:
        results = search_tool.invoke(query)
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Search error: {e}"

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use for calculations, not for general text.
    """
    # Safe evaluation using AST
    import ast
    import operator as op
    
    _SAFE_OPS = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
        ast.UAdd: op.pos, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    }
    
    def _safe_eval(node):
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            return _SAFE_OPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
        else:
            raise ValueError(f"Unsupported expression")
    
    try:
        tree = ast.parse(expression, mode='eval')
        result = _safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def save_file(content: str, filename: str) -> str:
    """Save content to a file."""
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"File saved: {filename}"
    except Exception as e:
        return f"Error saving file: {e}"

@tool
def read_file(filename: str) -> str:
    """Read content from a file."""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

# Tool registry
TOOLS = [web_search, calculator, save_file, read_file]
TOOL_MAP = {tool.name: tool for tool in TOOLS}

# ---------------------- AGENT DEFINITIONS ----------------------

class AgentState(TypedDict):
    """State schema for the multi-agent system."""
    messages: Annotated[list, add_messages]
    task: str
    current_agent: str
    completed_steps: list
    research_results: str
    analysis_results: str
    final_report: str
    iteration_count: int
    max_iterations: int
    status: str
    errors: list

# Initialize LLM
def get_llm():
    """Get the appropriate LLM based on configuration."""
    if Config.LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            api_key=Config.OPENAI_API_KEY
        )
    else:
        raise ValueError(f"Unsupported provider: {Config.LLM_PROVIDER}")

llm = get_llm()

# ---------------------- AGENT NODES ----------------------

def supervisor_node(state: AgentState) -> AgentState:
    """
    Supervisor agent - coordinates the entire workflow.
    Determines which agent should act next.
    """
    state["iteration_count"] += 1
    
    # Check if we've exceeded max iterations
    if state["iteration_count"] > state.get("max_iterations", Config.MAX_ITERATIONS):
        state["status"] = "max_iterations_reached"
        state["messages"].append("Maximum iterations reached. Finalizing...")
        return state
    
    # Determine next agent based on current progress
    completed = state.get("completed_steps", [])
    
    if not completed or "research" not in completed:
        state["current_agent"] = "researcher"
    elif "analysis" not in completed:
        state["current_agent"] = "analyst"
    elif "writing" not in completed:
        state["current_agent"] = "writer"
    else:
        state["current_agent"] = "finish"
        state["status"] = "complete"
    
    return state

def researcher_node(state: AgentState) -> AgentState:
    """
    Research Agent - Gathers information from multiple sources.
    """
    print("\n[Researcher] Gathering information...")
    
    task = state.get("task", "")
    
    # Use search tool
    search_results = web_search.invoke({
        "query": f"{task} latest developments research"
    })
    
    # Store results
    state["research_results"] = search_results
    
    # Mark step as completed
    if "research" not in state.get("completed_steps", []):
        state["completed_steps"] = state.get("completed_steps", []) + ["research"]
    
    state["messages"].append(f"[Researcher] Completed research on: {task}")
    
    return state

def analyst_node(state: AgentState) -> AgentState:
    """
    Analysis Agent - Analyzes and synthesizes research findings.
    """
    print("\n[Analyst] Analyzing research...")
    
    research = state.get("research_results", "")
    task = state.get("task", "")
    
    # Use LLM for analysis
    analysis_prompt = f"""
    Analyze the following research findings about: {task}
    
    Research:
    {research[:4000]}
    
    Provide:
    1. Key findings
    2. Main themes
    3. Important insights
    4. Recommendations
    
    Analysis:
    """
    
    try:
        response = llm.invoke(analysis_prompt)
        state["analysis_results"] = response.content
    except Exception as e:
        state["analysis_results"] = f"Analysis error: {e}"
    
    # Mark step as completed
    if "analysis" not in state.get("completed_steps", []):
        state["completed_steps"] = state.get("completed_steps", []) + ["analysis"]
    
    state["messages"].append("[Analyst] Completed analysis")
    
    return state

def writer_node(state: AgentState) -> AgentState:
    """
    Writer Agent - Creates structured reports.
    """
    print("\n[Writer] Creating report...")
    
    research = state.get("research_results", "")
    analysis = state.get("analysis_results", "")
    task = state.get("task", "")
    
    # Use LLM for report writing
    report_prompt = f"""
    Write a comprehensive report on: {task}
    
    Research:
    {research[:2000]}
    
    Analysis:
    {analysis[:2000]}
    
    Report Structure:
    1. Executive Summary
    2. Introduction
    3. Key Findings
    4. Detailed Analysis
    5. Recommendations
    6. Conclusion
    7. References
    
    Report:
    """
    
    try:
        response = llm.invoke(report_prompt)
        state["final_report"] = response.content
    except Exception as e:
        state["final_report"] = f"Report generation error: {e}"
    
    # Mark step as completed
    if "writing" not in state.get("completed_steps", []):
        state["completed_steps"] = state.get("completed_steps", []) + ["writing"]
    
    state["messages"].append("[Writer] Report completed")
    state["status"] = "report_ready"
    
    return state

def finalize_node(state: AgentState) -> AgentState:
    """
    Finalize node - prepares final output.
    """
    print("\n[Supervisor] Finalizing...")
    
    # Ensure we have a report
    if not state.get("final_report"):
        state["final_report"] = "No report was generated. Please try again."
    
    state["status"] = "complete"
    
    return state

# ---------------------- ROUTING LOGIC ----------------------

def route_by_agent(state: AgentState) -> Literal["researcher", "analyst", "writer", "finalize", "finish"]:
    """Route to the appropriate agent based on current state."""
    current = state.get("current_agent", "researcher")
    
    if current == "researcher":
        return "researcher"
    elif current == "analyst":
        return "analyst"
    elif current == "writer":
        return "writer"
    elif current == "finalize":
        return "finalize"
    else:
        return "finish"

def should_continue(state: AgentState) -> Literal["continue", "finish"]:
    """Check if we should continue or finish."""
    if state.get("status") == "complete":
        return "finish"
    if state.get("status") == "max_iterations_reached":
        return "finish"
    return "continue"

# ---------------------- BUILD THE MULTI-AGENT SYSTEM ----------------------

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)
workflow.add_node("finalize", finalize_node)

# Start with supervisor
workflow.add_edge(START, "supervisor")

# Conditional routing from supervisor
workflow.add_conditional_edges(
    "supervisor",
    route_by_agent,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "finalize": "finalize",
        "finish": END
    }
)

# All agents return to supervisor
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("analyst", "supervisor")
workflow.add_edge("writer", "supervisor")
workflow.add_edge("finalize", END)

# Add checkpointer for state persistence
checkpointer = MemorySaver()
multi_agent = workflow.compile(checkpointer=checkpointer)

print("\n=== MULTI-AGENT SYSTEM INITIALIZED ===\n")
print("Agents:")
print("  - Supervisor: Coordinates workflow")
print("  - Researcher: Gathers information")
print("  - Analyst: Analyzes findings")
print("  - Writer: Creates reports")
print("  - Finalize: Prepares output")
print("\nTools available:")
for tool in TOOLS:
    print(f"  - {tool.name}: {tool.description}")

# ---------------------- RUN THE SYSTEM ----------------------

def run_agent_system(task: str) -> Dict[str, Any]:
    """
    Run the multi-agent system on a task.
    """
    print(f"\n{'='*60}")
    print(f"Processing task: {task}")
    print(f"{'='*60}\n")
    
    # Initialize state
    initial_state = {
        "messages": [],
        "task": task,
        "current_agent": "supervisor",
        "completed_steps": [],
        "research_results": "",
        "analysis_results": "",
        "final_report": "",
        "iteration_count": 0,
        "max_iterations": Config.MAX_ITERATIONS,
        "status": "running",
        "errors": []
    }
    
    # Run the system
    try:
        result = multi_agent.invoke(
            initial_state,
            config={"configurable": {"thread_id": f"task_{int(time.time())}"}}
        )
        
        return {
            "success": True,
            "task": task,
            "steps": result.get("completed_steps", []),
            "report": result.get("final_report", "No report generated"),
            "research": result.get("research_results", ""),
            "analysis": result.get("analysis_results", ""),
            "status": result.get("status", "unknown"),
            "iterations": result.get("iteration_count", 0)
        }
        
    except Exception as e:
        return {
            "success": False,
            "task": task,
            "error": str(e),
            "status": "failed"
        }

# ---------------------- TEST THE SYSTEM ----------------------

print("\n=== TESTING MULTI-AGENT SYSTEM ===\n")

test_tasks = [
    "The impact of artificial intelligence on healthcare",
    "Renewable energy trends and future outlook",
    "Machine learning applications in finance"
]

test_results = []

for task in test_tasks:
    print(f"\n--- Processing task: {task} ---\n")
    result = run_agent_system(task)
    test_results.append(result)
    
    print(f"\nResults for '{task}':")
    print(f"  Status: {result['status']}")
    print(f"  Steps: {result.get('steps', [])}")
    print(f"  Iterations: {result.get('iterations', 0)}")
    print(f"  Report preview: {result.get('report', '')[:200]}...")
    print("-" * 40)

# ---------------------- CREATE UI ----------------------

import gradio as gr

def create_ui():
    """Create Gradio UI for the multi-agent system."""
    
    def process_task(task: str, history: list):
        """Process a task and update UI."""
        if not task:
            return "", history, ""
        
        # Run the system
        result = run_agent_system(task)
        
        # Format output
        output = f"""
## Task: {task}

### Status: {result['status']}

### Steps Completed:
{chr(10).join(f'- {step}' for step in result.get('steps', []))}

### Report:
{result.get('report', 'No report generated')}

---
**Metadata:**
- Iterations: {result.get('iterations', 0)}
- Success: {result['success']}
"""
        
        # Update history
        history.append((f"Task: {task}", result.get('report', 'No report')))
        
        return "", history, output
    
    with gr.Blocks(title="Multi-Agent AI System") as demo:
        gr.Markdown("""
        # 🤖 Autonomous Multi-Agent AI System
        
        This system uses multiple specialized agents to research, analyze, and report on any topic.
        
        **Agents:**
        - 🎯 Supervisor: Coordinates the workflow
        - 🔍 Researcher: Gathers information
        - 📊 Analyst: Analyzes findings
        - ✍️ Writer: Creates reports
        
        **Tools:** Web Search, Calculator, File I/O
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=400)
                msg = gr.Textbox(
                    label="Enter a topic to research",
                    placeholder="e.g., The impact of AI on healthcare",
                    lines=2
                )
                send_btn = gr.Button("Research", variant="primary")
            
            with gr.Column(scale=1):
                output_display = gr.Markdown(label="Detailed Results")
        
        # Event handlers
        send_btn.click(
            process_task,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, output_display]
        )
        
        msg.submit(
            process_task,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, output_display]
        )
    
    return demo

# ---------------------- LAUNCH ----------------------

print("\n=== LAUNCHING MULTI-AGENT SYSTEM UI ===\n")
demo = create_ui()
demo.launch(share=True, debug=False)

# ---------------------- FINAL SUMMARY ----------------------

print("\n" + "="*60)
print("=== DAY 5: MULTI-AGENT SYSTEM COMPLETE ===")
print("="*60)

print("\n📊 SYSTEM ARCHITECTURE:")
print("  - Supervisor Agent: Coordinates workflow")
print("  - Researcher Agent: Gathers information")
print("  - Analyst Agent: Analyzes findings")
print("  - Writer Agent: Creates reports")
print("  - Finalize Node: Prepares output")

print("\n🔧 TOOLS:")
for tool in TOOLS:
    print(f"  - {tool.name}: {tool.description}")

print("\n📈 TEST RESULTS:")
successful = sum(1 for r in test_results if r['success'])
print(f"  - Total tasks: {len(test_results)}")
print(f"  - Successful: {successful}")
print(f"  - Success rate: {successful/len(test_results)*100:.1f}%")

print("\n📁 OUTPUTS GENERATED:")
print("  - Research findings")
print("  - Analysis results")
print("  - Final reports")
print("  - Conversation history")

print("\n🚀 HOW TO USE:")
print("  1. Run this notebook in Kaggle")
print("  2. Wait for the Gradio interface")
print("  3. Enter a research topic")
print("  4. Watch the agents collaborate")
print("  5. Review the generated report")

print("\n✨ THANK YOU FOR COMPLETING THE AGENTIC AI PROJECT!")
print("="*60)
```

---

## 13. 2026 Modern Agentic AI Trends

### 13.1 Key Trends

```text
MODERN AGENTIC AI TRENDS (2026):
═══════════════════════════════════════════════════════════════

1. GRAPH-BASED PLANNING 
   - Bilevel architecture: Planning vs. Execution
   - Tool relationship modeling
   - Parallel tool execution
   - 13.1% improvement in TSR

2. PARALLEL TOOL EXECUTION 
   - Moving beyond sequential ReAct
   - Dependency-aware orchestration
   - Intelligent parallelization
   - Dramatic efficiency improvements

3. STANDARDIZED PROTOCOLS 
   - MCP (Model Context Protocol)
   - Tool interoperability
   - Cross-framework compatibility

4. PRODUCTION GUARDRAILS 
   - Token spend limits
   - Iteration count limits
   - Scope constraints
   - Approval gates

5. OBSERVABILITY
   - Complete tracing
   - Safety monitoring
   - Cost tracking
   - Performance metrics

6. UNIFIED TAXONOMY 
   - Tool use, planning, feedback learning
   - Consistent terminology
   - Framework comparison
```

### 13.2 The Future of Agentic AI

```text
THE FUTURE OF AGENTIC AI:
═══════════════════════════════════════════════════════════════

1. INCREASED AUTONOMY
   - Less human intervention
   - Longer-running tasks
   - Self-correction and learning

2. BETTER PLANNING
   - Graph-based planning
   - Parallel execution
   - Dynamic adaptation

3. STANDARDIZATION
   - MCP adoption
   - Common tool interfaces
   - Interoperable agents

4. SAFETY AND SECURITY
   - Production guardrails
   - Scope limitations
   - Budget controls

5. EVALUATION
   - Task success rates
   - Safety metrics
   - Cost efficiency
   - Quality metrics

6. ENTERPRISE ADOPTION
   - Production-ready frameworks
   - Integration with existing systems
   - Monitoring and observability
```

---

## Summary

### Week 8 - Key Takeaways

```text
WEEK 8 - AGENTIC AI KEY TAKEAWAYS:
═══════════════════════════════════════════════════════════════

1. AGENTIC AI
   - LLMs act as autonomous agents
   - Plan, use tools, collaborate
   - Three paradigms: Tool Use, Planning, Feedback Learning 

2. AGENT FRAMEWORKS
   - LangGraph: Graph-based, complex flows 
   - CrewAI: Role-based, multi-agent teams 
   - Custom workflows: Full control 

3. AGENT PATTERNS 
   - Supervisor: Central coordinator
   - Swarm: Peer-to-peer handoffs
   - Human-in-the-Loop: Approval gates
   - RAG: Retrieval-augmented generation
   - Custom: Bespoke workflows

4. TOOL USE
   - Function calling for external APIs
   - Safe execution with guardrails 
   - Parallel execution for efficiency 

5. PLANNING 
   - Bilevel architecture
   - Graph-based planning
   - Task decomposition
   - Dependency management

6. MULTI-AGENT SYSTEMS
   - Collaboration between agents
   - Specialized roles
   - Hierarchical organization
   - Peer-to-peer handoffs

7. MEMORY AND STATE
   - Short-term, long-term, episodic
   - Session-based memory 
   - State management with LangGraph 

8. HUMAN-IN-THE-LOOP 
   - Approval gates
   - Safety checks
   - High-stakes operations
   - Production guardrails

9. EVALUATION 
   - Task Success Rate (TSR)
   - Tool call accuracy
   - Efficiency metrics
   - Safety metrics

10. MODERN TRENDS
    - Graph-based planning 
    - Parallel tool execution 
    - MCP standardization 
    - Production guardrails 
    - Comprehensive observability
```
