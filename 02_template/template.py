# 可以定义输出格式，具备解析结果并保存的能力
import os
os.environ["OPENAI_API_KEY"] = ''
from langchain_openai import OpenAI
# 创建模型实例
model = OpenAI(model_name='gpt-3.5-turbo-instruct')

# 导入LangChain中的提示模板
from langchain.prompts import PromptTemplate

prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
并按如下json格式进行回复：
{{"description":xx,"reason":xx}}"""
# # 根据原始模板创建提示，同时在提示中加入输出解析器的说明
prompt = PromptTemplate.from_template(prompt_template,) 

# 数据准备
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 创建一个空的DataFrame用于存储结果
import pandas as pd
import json
df = pd.DataFrame(columns=["flower", "price", "description", "reason"]) # 先声明列名

for flower, price in zip(flowers, prices):
    # 根据提示准备模型的输入
    input = prompt.format_prompt(flower_name=flower, price=price)
    # print(model.invoke(input))


    parsed_output=json.loads(model.invoke(input))
    print(parsed_output)
    

    # 在解析后的输出中添加“flower”和“price”
    parsed_output['flower'] = flower
    parsed_output['price'] = price

    # 将解析后的输出添加到DataFrame中
    df.loc[len(df)] = parsed_output  

print(df)

# 这种写法更加简单