
import os
os.environ["OPENAI_API_KEY"] = ''
# 构建两个场景的模板
flower_care_template = """
你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
下面是需要你来回答的问题:
{input}
"""

flower_deco_template = """
你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
下面是需要你来回答的问题:
{input}
"""

# 构建提示信息
prompt_infos = [
    {
        "key": "flower_care",
        "description": "适合回答关于鲜花护理的问题",
        "template": flower_care_template,
    },
    {
        "key": "flower_decoration",
        "description": "适合回答关于鲜花装饰的问题",
        "template": flower_deco_template,
    }
]

# 初始化语言模型
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = OpenAI()

# 构建目标链

destination_chains = {}
for info in prompt_infos:
    prompt = PromptTemplate.from_template(info['template'])
    chain = prompt | llm | StrOutputParser()
    destination_chains[info["key"]] = chain
    print(f"创建链 {info['key']}: {info['description']}")


# 创建路由提示模板
destinations = [f"{p['key']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

router_template = f"""
Given a raw text input to a language model, select the model prompt best suited for \
the input. You will be given the names of the available prompts and a description of \
what the prompt is best suited for. 

The prompts are:

{{destinations}}

Given the user question below, classify it as one of the prompt names. \
If the question doesn't fit any of the prompt types, respond with "DEFAULT".

Question: {{input}}

Format instructions: Output only the prompt name, nothing else.

Prompt name:"""


# 构建路由链
router_prompt = PromptTemplate.from_template(router_template)
router_chain = router_prompt | llm | StrOutputParser()
print("路由提示模板:")
print(router_template.format(destinations=destinations_str, input="{input}"))


default_template = PromptTemplate.from_template("请回答以下问题:\n{input}")
default_chain = default_template | llm | StrOutputParser()
    

# 构建完整的路由链
def full_chain(input_dict):
    query = input_dict["input"]
    # 获取路由决策
    route = router_chain.invoke({"destinations": destinations_str, "input": query})
    route = route.strip()
    
    print(f"路由决策: {route}")
    
    if route in destination_chains:
        # 使用对应的专门链
        return destination_chains[route].invoke({"input": query})
    else:
        # 使用默认链
        return default_chain.invoke({"input": query})


# 测试函数
def test_router():
    test_queries = [
        "如何为玫瑰浇水？",
        "如何为婚礼场地装饰花朵？",
        "如何区分阿豆和罗豆？"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"问题: {query}")
        print(f"{'='*50}")
        try:
            result = full_chain({"input": query})
            print(result)
        except Exception as e:
            print(f"错误: {e}")


# 运行测试
# 其实是构建路由提示模版来选择chain,重构后路由chain 也就是普通的链
test_router()

