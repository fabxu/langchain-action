import os
os.environ["OPENAI_API_KEY"] = ''
os.environ["SERPAPI_API_KEY"] = ''

from typing import Dict, Any
from langchain_openai import OpenAI
from langchain_community.utilities import SerpAPIWrapper 
from langchain_community.tools import Tool
from langchain_core.prompts import PromptTemplate

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

def simple_react_agent(input_question: str, max_iterations: int = 5):
    """简单的 ReAct agent 实现"""
    tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    
    # agent 执行过程记录
    agent_scratchpad = ""
    
    for i in range(max_iterations):
        # 构造提示
        formatted_prompt = prompt.format(
            tool_descriptions=tool_descriptions,
            tool_names=tool_names,
            input=input_question,
            agent_scratchpad=agent_scratchpad
        )
        
        # 调用 LLM
        llm_output = llm.invoke(formatted_prompt)
        
        # 如果返回的是 AIMessage 对象，获取内容
        if hasattr(llm_output, 'content'):
            llm_output = llm_output.content
        
        print(f"Iteration {i+1}:")
        print(f"LLM Output:\n{llm_output}\n")

        # 检查是否得到最终答案
        if "Final Answer:" in llm_output:
            return llm_output.split("Final Answer:")[-1].strip()  # strip() 去除字符串两端的空白字符
        
        #解析行动
        if "Action:" in llm_output and "Action Input:" in llm_output:
            lines = llm_output.split('\n')
            action = None
            action_input = None
            
            for line in lines:
                if line.startswith("Action:"):
                    action = line.split("Action:")[1].strip()
                elif line.startswith("Action Input:"):
                    action_input = line.split("Action Input:")[1].strip()
            
            if action and action_input and action in tool_map:
                try:
                    observation = tool_map[action].func(action_input)
                    agent_scratchpad += llm_output + f"\nObservation: {observation}\nThought:"
                except Exception as e:
                    agent_scratchpad += llm_output + f"\nObservation: Error - {str(e)}\nThought:"
            else:
                agent_scratchpad += llm_output + "\nObservation: Invalid action\nThought:"
        else:
            agent_scratchpad += llm_output + "\nThought:"
    
    return "无法在限定迭代次数内找到答案"


question = "目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？"
result = simple_react_agent(question)
print(f"最终答案: {result}")