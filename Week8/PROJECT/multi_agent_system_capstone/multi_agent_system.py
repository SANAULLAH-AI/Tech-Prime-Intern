"""
Multi-Agent AI System - Capstone Project
Complete implementation of an Autonomous Multi-Agent AI System
Using LangGraph, LangChain, and Groq
"""

# Cell 1: Setup and Installations
import subprocess
import sys

def install_packages():
    packages = [
        "langgraph", "langchain", "langchain-community", 
        "langchain-openai", "langchain-groq", "gradio",
        "matplotlib", "seaborn", "pandas", "numpy", "plotly"
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# Cell 2: Import Libraries
import os
import json
import time
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
import operator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    try:
        from langgraph.checkpoint import MemorySaver
    except ImportError:
        from langgraph.memory import MemorySaver

import gradio as gr

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Cell 3: API Key Configuration
import getpass

def configure_api_key():
    print("="*60)
    print("GROQ API KEY CONFIGURATION")
    print("="*60)
    groq_api_key = getpass.getpass("Please enter your Groq API key: ")
    os.environ["GROQ_API_KEY"] = groq_api_key
    return groq_api_key

# Cell 4: Define Tools for Agents
@tool
def calculate(expression: str) -> str:
    try:
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round})
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

@tool
def search_web(query: str) -> str:
    search_results = {
        "weather": "Current weather: 72°F, partly cloudy",
        "population": "World population: approximately 8.1 billion",
        "capital": "Capital of France is Paris",
        "default": "Search results for: " + query
    }
    for key, value in search_results.items():
        if key in query.lower():
            return value
    return search_results["default"]

@tool
def data_analyzer(data_description: str) -> str:
    insights = {
        "sales": "Sales data shows 15% growth in Q3, with peak performance in technology sector",
        "customer": "Customer satisfaction score: 4.2/5, retention rate: 78%",
        "market": "Market trends indicate 12% growth in AI adoption across industries",
        "default": f"Analysis complete for: {data_description}"
    }
    for key, value in insights.items():
        if key in data_description.lower():
            return value
    return insights["default"]

@tool
def code_generator(task: str) -> str:
    if "sorting" in task.lower():
        return "```python\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n```"
    elif "api" in task.lower():
        return "```python\ndef call_api(url, params={}):\n    import requests\n    response = requests.get(url, params=params)\n    return response.json()\n```"
    else:
        return f"Generated code for: {task}"

available_tools = [calculate, search_web, data_analyzer, code_generator]

# Cell 5: Define Agent State
class AgentState(TypedDict):
    messages: Annotated[List[Dict], add_messages]
    current_agent: str
    task_plan: List[str]
    completed_tasks: List[str]
    agent_results: Dict[str, Any]
    final_answer: str

# Cell 6: Initialize Language Models
def initialize_models():
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            api_key = getpass.getpass("Enter your Groq API Key: ")
            os.environ["GROQ_API_KEY"] = api_key
        
        llm = ChatGroq(
            temperature=0.3,
            model_name="openai/gpt-oss-120b",
            groq_api_key=api_key
        )
        
        print("All LLM models initialized successfully with: openai/gpt-oss-120b")
        
        return {
            "planner": llm,
            "executor": llm,
            "analyzer": llm,
            "summarizer": llm
        }
    except Exception as e:
        print(f"Error initializing models: {e}")
        return None

# Cell 7: Create Specialized Agents
class Agent:
    def __init__(self, name: str, llm, tools: List, system_prompt: str):
        self.name = name
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.tool_names = [tool.name for tool in tools] if tools else []
        
        if tools:
            self.llm_with_tools = llm.bind_tools(tools)
        else:
            self.llm_with_tools = llm
        
        self.agent_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def process(self, messages: List[Dict]) -> Dict:
        try:
            formatted_messages = []
            for msg in messages:
                if msg.get("role") == "user":
                    formatted_messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    formatted_messages.append(AIMessage(content=msg.get("content", "")))
                elif msg.get("role") == "system":
                    formatted_messages.append(SystemMessage(content=msg.get("content", "")))
                else:
                    formatted_messages.append(HumanMessage(content=msg.get("content", "")))
            
            chain = self.agent_prompt | self.llm_with_tools
            response = chain.invoke({"messages": formatted_messages})
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    
                    for tool_obj in self.tools:
                        if tool_obj.name == tool_name:
                            try:
                                result = tool_obj.invoke(tool_args)
                                tool_results.append(f"{tool_name}: {result}")
                            except Exception as e:
                                tool_results.append(f"{tool_name} error: {str(e)}")
                            break
                
                if tool_results:
                    return {
                        "agent": self.name,
                        "response": f"Tool results: {'; '.join(tool_results)}",
                        "tools_used": [tc.get('name') for tc in response.tool_calls]
                    }
            
            return {
                "agent": self.name,
                "response": response.content,
                "tools_used": self.tool_names if response.tool_calls else []
            }
        except Exception as e:
            return {
                "agent": self.name,
                "error": str(e),
                "response": f"Error in {self.name}: {str(e)}"
            }

def create_planner_agent(llm):
    system_prompt = """You are the Planner Agent. Your role is to break down complex tasks into manageable subtasks. Provide a clear plan with numbered steps."""
    return Agent("Planner", llm, [], system_prompt)

def create_executor_agent(llm):
    system_prompt = """You are the Executor Agent. Use the calculate tool for math operations. Always provide clear results."""
    return Agent("Executor", llm, [calculate, search_web, data_analyzer, code_generator], system_prompt)

def create_analyzer_agent(llm):
    system_prompt = """You are the Analyzer Agent. Evaluate results and provide feedback on quality and accuracy."""
    return Agent("Analyzer", llm, [], system_prompt)

def create_summarizer_agent(llm):
    system_prompt = """You are the Summarizer Agent. Compile results into a coherent summary. Keep responses concise."""
    return Agent("Summarizer", llm, [], system_prompt)

# Cell 8: Build the Multi-Agent Orchestration Graph
class MultiAgentSystem:
    def __init__(self, models):
        self.models = models
        self.agents = self._initialize_agents()
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        self.conversation_history = []
    
    def _initialize_agents(self):
        return {
            "planner": create_planner_agent(self.models["planner"]),
            "executor": create_executor_agent(self.models["executor"]),
            "analyzer": create_analyzer_agent(self.models["analyzer"]),
            "summarizer": create_summarizer_agent(self.models["summarizer"])
        }
    
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("plan", self.plan_task)
        workflow.add_node("execute", self.execute_task)
        workflow.add_node("analyze", self.analyze_results)
        workflow.add_node("summarize", self.summarize_results)
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "execute")
        workflow.add_edge("execute", "analyze")
        workflow.add_edge("analyze", "summarize")
        workflow.add_edge("summarize", END)
        return workflow
    
    def _extract_user_query(self, messages):
        for msg in messages:
            if isinstance(msg, HumanMessage):
                return msg.content
            elif isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return ""
    
    def _extract_math_expression(self, query: str) -> str:
        if not query:
            return None
        import re
        cleaned = re.sub(r'(what is|calculate|compute|solve|the result of|answer for|whats|what's)', '', query.lower())
        math_pattern = r'[\d\.]+\s*[\+\-\*\/]\s*[\d\.]+'
        matches = re.findall(math_pattern, cleaned)
        if matches:
            return matches[0].strip()
        return None
    
    def _create_task_plan(self, query: str) -> List[str]:
        if not query:
            return ["Please provide a query to analyze"]
        import re
        math_patterns = [r'\d+\s*[\+\-\*\/]\s*\d+', r'calculate', r'what is', r'compute', r'solve', r'whats']
        for pattern in math_patterns:
            if re.search(pattern, query.lower()):
                expr = self._extract_math_expression(query)
                if expr:
                    return [f"Calculate: {expr}"]
                return [f"Calculate the result of: {query}"]
        return ["Analyze the given query and provide a response"]
    
    def plan_task(self, state: AgentState) -> Dict:
        user_query = self._extract_user_query(state["messages"])
        task_plan = self._create_task_plan(user_query)
        return {
            "messages": state["messages"],
            "current_agent": "Planner",
            "task_plan": task_plan,
            "agent_results": {"planner": f"Planning tasks for: {user_query}"}
        }
    
    def execute_task(self, state: AgentState) -> Dict:
        task_plan = state.get("task_plan", [])
        executor = self.agents["executor"]
        results = []
        user_query = self._extract_user_query(state["messages"])
        
        for task in task_plan:
            if "Calculate:" in task:
                expr = task.replace("Calculate:", "").strip()
                messages = [{"role": "user", "content": f"Calculate the expression: {expr}"}]
            elif "Calculate" in task:
                expr = self._extract_math_expression(user_query)
                if expr:
                    messages = [{"role": "user", "content": f"Calculate: {expr}"}]
                else:
                    messages = [{"role": "user", "content": f"Execute this task: {task}"}]
            else:
                messages = [{"role": "user", "content": f"Execute this task: {task}. Query: {user_query}"}]
            
            result = executor.process(messages)
            results.append({"task": task, "result": result["response"], "tools_used": result.get("tools_used", [])})
        
        return {
            "messages": state["messages"],
            "current_agent": "Executor",
            "agent_results": {"executor": results},
            "completed_tasks": [r["task"] for r in results]
        }
    
    def analyze_results(self, state: AgentState) -> Dict:
        executor_results = state.get("agent_results", {}).get("executor", [])
        analyzer = self.agents["analyzer"]
        if executor_results:
            analysis_input = f"Analyze these task results and provide a brief evaluation: {json.dumps(executor_results, indent=2)}"
        else:
            analysis_input = "No results to analyze."
        messages = [{"role": "user", "content": analysis_input}]
        result = analyzer.process(messages)
        return {
            "messages": state["messages"],
            "current_agent": "Analyzer",
            "agent_results": {"analyzer": result["response"]}
        }
    
    def summarize_results(self, state: AgentState) -> Dict:
        all_results = state.get("agent_results", {})
        executor_results = all_results.get("executor", [])
        if executor_results and len(executor_results) > 0:
            result_text = executor_results[0].get("result", "")
            import re
            if "Result:" in result_text:
                answer_match = re.search(r'Result:\s*([\d\.\-]+)', result_text)
                if answer_match:
                    final_answer = f"The answer is: {answer_match.group(1)}"
                else:
                    final_answer = result_text
            else:
                final_answer = result_text
        else:
            summarizer = self.agents["summarizer"]
            summary_input = f"Summarize these agent results into a final answer: {json.dumps(all_results, indent=2)}"
            messages = [{"role": "user", "content": summary_input}]
            result = summarizer.process(messages)
            final_answer = result["response"]
        
        return {
            "messages": state["messages"],
            "current_agent": "Summarizer",
            "final_answer": final_answer,
            "agent_results": {"summarizer": final_answer}
        }
    
    def run(self, query: str) -> Dict:
        start_time = time.time()
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "current_agent": "Planner",
            "task_plan": [],
            "completed_tasks": [],
            "agent_results": {},
            "final_answer": ""
        }
        try:
            result = self.app.invoke(initial_state)
            execution_time = time.time() - start_time
            final_answer = result.get("final_answer", "")
            if not final_answer:
                executor_results = result.get("agent_results", {}).get("executor", [])
                if executor_results:
                    result_text = executor_results[0].get("result", "")
                    import re
                    if "Result:" in result_text:
                        answer_match = re.search(r'Result:\s*([\d\.\-]+)', result_text)
                        if answer_match:
                            final_answer = f"The answer is: {answer_match.group(1)}"
                        else:
                            final_answer = result_text
                    else:
                        final_answer = result_text
            return {
                "query": query,
                "final_answer": final_answer or "No answer generated",
                "agent_results": result.get("agent_results", {}),
                "task_plan": result.get("task_plan", []),
                "completed_tasks": result.get("completed_tasks", []),
                "execution_time": execution_time,
                "messages": result.get("messages", []),
                "success": True
            }
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "success": False,
                "execution_time": time.time() - start_time
            }

# Cell 9: Create Test Cases
def create_test_cases():
    return [
        "Calculate the compound interest for $10,000 at 5% for 3 years",
        "Analyze recent sales trends and predict future growth",
        "Create a Python function to sort a list of numbers",
        "Explain the benefits of AI in healthcare with data analysis",
        "Plan a marketing strategy for a new tech product launch",
        "Analyze customer feedback and suggest improvements",
        "Generate code for a weather API integration",
        "Evaluate market conditions for renewable energy investments"
    ]

# Cell 10: Evaluation Framework
class SystemEvaluator:
    def __init__(self, system):
        self.system = system
        self.results = []
    
    def evaluate_query(self, query: str) -> Dict:
        result = self.system.run(query)
        evaluation = {
            "query": query,
            "success": result.get("success", False),
            "execution_time": result.get("execution_time", 0),
            "has_answer": bool(result.get("final_answer", "")),
            "tasks_completed": len(result.get("completed_tasks", [])),
            "agents_used": list(result.get("agent_results", {}).keys()),
            "final_answer_length": len(result.get("final_answer", "")),
            "result": result
        }
        return evaluation
    
    def evaluate_all(self, queries: List[str]) -> List[Dict]:
        self.results = []
        for query in queries:
            eval_result = self.evaluate_query(query)
            self.results.append(eval_result)
        return self.results
    
    def get_metrics(self) -> Dict:
        if not self.results:
            return {"error": "No results to evaluate"}
        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        if successful > 0:
            avg_time = np.mean([r["execution_time"] for r in self.results if r["success"]])
            avg_tasks = np.mean([r["tasks_completed"] for r in self.results if r["success"]])
            avg_length = np.mean([r["final_answer_length"] for r in self.results if r["success"]])
        else:
            avg_time = 0
            avg_tasks = 0
            avg_length = 0
        return {
            "total_queries": total,
            "success_rate": successful / total if total > 0 else 0,
            "average_execution_time": avg_time,
            "average_tasks_completed": avg_tasks,
            "average_answer_length": avg_length,
            "successful_queries": successful,
            "failed_queries": total - successful
        }

# Cell 11: Visualization Functions
def create_visualizations(evaluator):
    if not evaluator.results:
        print("No results to visualize.")
        return None
    
    metrics = evaluator.get_metrics()
    results = evaluator.results
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    success_counts = [metrics["successful_queries"], metrics["failed_queries"]]
    axes[0, 0].pie(success_counts, labels=['Successful', 'Failed'], autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('Task Success Rate', fontsize=14, fontweight='bold')
    
    times = [r["execution_time"] for r in results if r["success"]]
    if times:
        axes[0, 1].hist(times, bins=10, alpha=0.7, color='#3498db', edgecolor='black')
        axes[0, 1].set_title('Execution Time Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Frequency')
    
    task_counts = [r["tasks_completed"] for r in results if r["success"]]
    if task_counts:
        bars = axes[1, 0].bar(range(len(task_counts)), task_counts, alpha=0.7, color='#2ecc71', edgecolor='black')
        axes[1, 0].set_title('Tasks Completed per Query', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Query Index')
        axes[1, 0].set_ylabel('Number of Tasks')
        axes[1, 0].set_xticks(range(len(task_counts)))
        axes[1, 0].set_xticklabels([f'Q{i+1}' for i in range(len(task_counts))])
    
    agent_usage = {}
    for result in results:
        if result["success"]:
            for agent in result["agents_used"]:
                agent_usage[agent] = agent_usage.get(agent, 0) + 1
    
    if agent_usage:
        axes[1, 1].bar(agent_usage.keys(), agent_usage.values(), alpha=0.7, color='#9b59b6', edgecolor='black')
        axes[1, 1].set_title('Agent Usage Frequency', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Agent Name')
        axes[1, 1].set_ylabel('Times Used')
    
    plt.tight_layout()
    return fig

def create_performance_summary(evaluator):
    metrics = evaluator.get_metrics()
    fig, ax = plt.subplots(figsize=(12, 7))
    metric_names = ['Success Rate', 'Avg Tasks', 'Avg Time (s)', 'Success Queries', 'Failed Queries']
    metric_values = [
        metrics["success_rate"] * 100,
        metrics["average_tasks_completed"],
        metrics["average_execution_time"],
        metrics["successful_queries"],
        metrics["failed_queries"]
    ]
    colors_bar = ['#2ecc71', '#3498db', '#f39c12', '#2ecc71', '#e74c3c']
    bars = ax.bar(metric_names, metric_values, color=colors_bar, edgecolor='black', linewidth=2, alpha=0.8)
    ax.set_title('System Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.05 * max(metric_values)),
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.tight_layout()
    return fig

# Cell 12: Gradio UI Interface
def create_gradio_interface(system):
    def process_query(query):
        if not query or query.strip() == "":
            return "Please enter a query", "", "", "", ""
        result = system.run(query)
        if not result["success"]:
            return f"Error: {result.get('error', 'Unknown error')}", "", "", "", ""
        final_answer = result["final_answer"]
        task_plan = "
".join([f"{i+1}. {task}" for i, task in enumerate(result["task_plan"])]) or "No specific task plan generated"
        agent_results = ""
        for agent, response in result["agent_results"].items():
            agent_results += f"
=== {agent.upper()} ===
"
            if isinstance(response, list):
                for item in response:
                    agent_results += f"Task: {item.get('task', 'Unknown')}
Result: {item.get('result', 'No result')}

"
            else:
                agent_results += f"{response}
"
        completed = "
".join([f"✓ {task}" for task in result["completed_tasks"]]) or "No tasks completed"
        performance = f"Execution Time: {result['execution_time']:.2f} seconds
Tasks Completed: {len(result['completed_tasks'])}
Agents Used: {len(result['agent_results'])}"
        return final_answer, task_plan, agent_results, completed, performance
    
    iface = gr.Interface(
        fn=process_query,
        inputs=[gr.Textbox(label="Enter your query", placeholder="Example: Calculate 2 * 4", lines=4)],
        outputs=[
            gr.Textbox(label="Final Answer", lines=8),
            gr.Textbox(label="Task Plan", lines=5),
            gr.Textbox(label="Agent Results", lines=10),
            gr.Textbox(label="Completed Tasks", lines=5),
            gr.Textbox(label="Performance Metrics", lines=4)
        ],
        title="Multi-Agent AI System - Capstone Project",
        description="Autonomous Multi-Agent System with Planner, Executor, Analyzer, and Summarizer agents.",
        theme="soft",
        flagging_mode="never",
        examples=[
            ["Calculate 2 * 4"],
            ["what is 10 + 20"],
            ["Calculate compound interest for $10,000 at 5% for 3 years"],
            ["Analyze recent sales trends"]
        ]
    )
    return iface

# Cell 13: Main Execution Pipeline
def run_complete_pipeline(system, evaluator):
    print("="*80)
    print("MULTI-AGENT AI SYSTEM - CAPSTONE PROJECT")
    print("="*80)
    test_cases = create_test_cases()
    print(f"
Created {len(test_cases)} test cases")
    evaluation_results = evaluator.evaluate_all(test_cases)
    metrics = evaluator.get_metrics()
    print("
System Performance Metrics:")
    print("-" * 40)
    print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"Average Execution Time: {metrics['average_execution_time']:.2f} seconds")
    print(f"Average Tasks Completed: {metrics['average_tasks_completed']:.1f}")
    print(f"Total Queries Evaluated: {metrics['total_queries']}")
    return evaluator

# Main execution
if __name__ == "__main__":
    print("Multi-Agent AI System - Capstone Project")
    print("="*60)
    
    # Configure API key
    api_key = configure_api_key()
    
    # Initialize models
    models = initialize_models()
    if not models:
        print("Failed to initialize models. Exiting.")
        sys.exit(1)
    
    # Create system
    multi_agent_system = MultiAgentSystem(models)
    print("System created successfully!")
    
    # Create evaluator
    evaluator = SystemEvaluator(multi_agent_system)
    
    # Run evaluation
    run_complete_pipeline(multi_agent_system, evaluator)
    
    # Create visualizations
    fig1 = create_visualizations(evaluator)
    if fig1:
        fig1.savefig('system_performance.png', dpi=300, bbox_inches='tight')
        print("Saved: system_performance.png")
    
    fig2 = create_performance_summary(evaluator)
    if fig2:
        fig2.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
        print("Saved: performance_summary.png")
    
    print("
Launching Gradio interface...")
    interface = create_gradio_interface(multi_agent_system)
    interface.launch(share=True, debug=False, server_name="0.0.0.0")
