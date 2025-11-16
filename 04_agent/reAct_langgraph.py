
import os
os.environ["OPENAI_API_KEY"] = ''
os.environ["SERPAPI_API_KEY"] = ''

from typing import Annotated, Sequence, TypedDict, Dict, Any
from langchain_openai import OpenAI
from langchain_community.utilities import SerpAPIWrapper 
from langchain_community.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from langgraph.graph import StateGraph, END
import json

# 定义状态类型
class AgentState(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], operator.add]
    agent_outcome: Dict[str, Any]
    intermediate_steps: Annotated[Sequence[Dict[str, Any]], operator.add]

# 初始化大模型
llm = OpenAI(temperature=0)

# 定义工具
tools = [
    Tool(
        name="Search",
        func=SerpAPIWrapper().run,
        description="Useful for searching the web for current information."
    ),
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Useful for performing mathematical calculations."
    )
]


tool_map = {tool.name: tool for tool in tools}

# 定义 ReAct 提示模板
react_template = """Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_template)

def format_intermediate_steps(intermediate_steps):
    """格式化中间步骤为scratchpad文本"""
    if not intermediate_steps:
        return ""
    
    scratchpad = ""
    for step in intermediate_steps:
        scratchpad += f"Thought: {step['thought']}\n"
        if 'action' in step:
            scratchpad += f"Action: {step['action']}\n"
            scratchpad += f"Action Input: {step['action_input']}\n"
            scratchpad += f"Observation: {step['observation']}\n"
    return scratchpad

def should_continue(state: AgentState) -> str:
    """决定是否继续执行"""
    last_message = state["chat_history"][-1].content if state["chat_history"] else ""
    if "Final Answer:" in last_message:
        return "end"
    elif "Action:" in last_message and "Action Input:" in last_message:
        return "continue"
    else:
        return "end"

def agent_node(state: AgentState) -> Dict[str, Any]:
    """Agent节点 - 负责推理和决策"""
    # 准备工具信息
    tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    
    # 格式化中间步骤
    intermediate_steps = state.get("intermediate_steps", [])
    agent_scratchpad = format_intermediate_steps(intermediate_steps)
    
    # 构造提示
    formatted_prompt = prompt.format(
        tool_descriptions=tool_descriptions,
        tool_names=tool_names,
        input=state["input"],
        agent_scratchpad=agent_scratchpad
    )
    
    # 调用LLM
    response = llm.invoke(formatted_prompt)
    
    # 打印思维过程
    print("=" * 50)
    print("AGENT THINKING:")
    print("-" * 30)
    print(response)
    print("=" * 50)

    # 添加到聊天历史
    new_history = [AIMessage(content=response)]
    
    return {
        "chat_history": new_history,
        "agent_outcome": {"response": response}
    }

def tool_node(state: AgentState) -> Dict[str, Any]:
    """工具节点 - 负责执行工具"""
    last_message = state["chat_history"][-1].content
    
    # 解析行动
    action = None
    action_input = None
    
    lines = last_message.split('\n')
    for line in lines:
        if line.startswith("Action:"):
            action = line.split("Action:")[1].strip()
        elif line.startswith("Action Input:"):
            action_input = line.split("Action Input:")[1].strip()
    
    # 打印即将执行的行动
    print("EXECUTING TOOL:")
    print(f"  Action: {action}")
    print(f"  Action Input: {action_input}")

    # 执行工具
    observation = "Invalid action or tool not found"
    if action and action_input and action in tool_map:
        try:
            observation = tool_map[action].func(action_input)
            print(f"  Observation: {observation}")
        except Exception as e:
            observation = f"Error executing tool: {str(e)}"
            print(f"  Error: {observation}")
    
    # 记录中间步骤
    intermediate_step = {
        "thought": "Previous thought processed",
        "action": action,
        "action_input": action_input,
        "observation": observation
    }
    
    return {
        "intermediate_steps": [intermediate_step]
    }

# 创建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# 设置入口点
workflow.set_entry_point("agent")

# 添加边
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")

# 编译图
app = workflow.compile()

def run_react_agent_with_langgraph(question: str, max_iterations: int = 5) -> str:
    """使用LangGraph运行ReAct Agent"""
    initial_state = {
        "input": question,
        "chat_history": [],
        "agent_outcome": {},
        "intermediate_steps": []
    }
    
    # 运行图
    final_state = app.invoke(initial_state)
    
    # 提取最终答案
    last_message = final_state["chat_history"][-1].content if final_state["chat_history"] else ""
    if "Final Answer:" in last_message:
        return last_message.split("Final Answer:")[-1].strip()
    else:
        return last_message


question = "目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？"
result = run_react_agent_with_langgraph(question)
print(f"最终答案: {result}")