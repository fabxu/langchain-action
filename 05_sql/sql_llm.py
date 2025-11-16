import os
os.environ["OPENAI_API_KEY"] = ''

# 导入langchain的实用工具和相关的模块
from langchain_community.utilities import SQLDatabase
from langchain_openai  import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# 连接到FlowerShop数据库（之前我们使用的是Chinook.db）
db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")

# 创建OpenAI的语言模型实例
llm = OpenAI(temperature=0)

# 创建提示模板
template = """
Based on the table schema below, write a SQL query that would answer the user's question.
{schema}

Question: {question}
SQL Query:
"""

prompt = ChatPromptTemplate.from_template(template)

def get_schema(_):
    return db.get_table_info()

def run_query(query):
    try:
        # 清理查询语句
        clean_query = query.strip().split(';')[0] + ';'
        return db.run(clean_query)
    except Exception as e:
        return f"Error executing query: {e}"

# 创建SQL链  这个是标准的链式调用，并不是系统认为这是sql chain。只是说到达run_query 的时候是个sql 语句
sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
    | run_query
)

def testCommon():
    questions = [
        "有多少种不同的鲜花？",
        "哪种鲜花的存货数量最少？",
        "平均销售价格是多少？",
        "从法国进口的鲜花有多少种？",
        "哪种鲜花的销售量最高？"
    ]

    for i, question in enumerate(questions, 1):
        print(f"问题{i}: {question}")
        try:
            response = sql_chain.invoke({"question": question})
            print(f"回答: {response}")
        except Exception as e:
            print(f"执行出错: {e}")
        print("-" * 50)

#testCommon()

# 自定义处理函数来捕获和显示中间结果
def generate_and_show_sql(inputs):
    # 构造提示
    schema = inputs['schema']
    question = inputs['question']
    
    # 格式化提示
    formatted_prompt = prompt.format(schema=schema, question=question)
    
    # 生成SQL
    generated_sql = llm.invoke(formatted_prompt)
    sql_text = generated_sql.strip() if isinstance(generated_sql, str) else getattr(generated_sql, 'content', str(generated_sql)).strip()
    
    # 打印生成的SQL
    print(f"生成的SQL语句: {sql_text}")
    
    return sql_text

def run_and_show_result(query):
    try:
        # 清理查询语句
        clean_query = query.strip().split(';')[0] + ';'
        print(f"实际执行的SQL: {clean_query}")
        
        # 执行查询
        result = db.run(clean_query)
        print(f"查询结果: {result}")
        
        # 让LLM解释结果
        explain_prompt = f"""
        Question: {current_question}
        SQL Query: {clean_query}
        Query Result: {result}
        
        Please provide a clear answer to the original question based on the query result:
        """
        
        explanation = llm.invoke(explain_prompt)
        final_answer = explanation.strip() if isinstance(explanation, str) else getattr(explanation, 'content', str(explanation)).strip()
        
        return final_answer
    except Exception as e:
        return f"Error executing query: {e}"

# 全局变量来保存当前问题
current_question = ""

# 修改主流程以显示完整过程
def process_question(question):
    global current_question
    current_question = question
    
    print(f"问题: {question}")
    print("-" * 40)
    
    # 获取schema
    schema = get_schema(None)
    
    # 生成SQL
    inputs = {'schema': schema, 'question': question}
    sql_query = generate_and_show_sql(inputs)
    print("-" * 40)
    
    # 执行并获取结果
    final_result = run_and_show_result(sql_query)
    print(f"最终回答: {final_result}")
    print("=" * 60)
    
    return final_result


def testDetail():
    # 测试问题
    questions = [
        "有多少种不同的鲜花？",
        "哪种鲜花的存货数量最少？",
        "平均销售价格是多少？"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}] ", end="")
        process_question(question)

testDetail()

#有关回调 不用BaseCallbackHandler 和 AsyncCallbackHandler 
# from langchain_core.callbacks import BaseCallbackHandler

# class MyCallbackHandler(BaseCallbackHandler):
#     def on_chain_start(self, serialized, inputs, **kwargs):
#         print(f"Chain started with inputs: {inputs}")
        
#     def on_chain_end(self, outputs, **kwargs):
#         print(f"Chain ended with outputs: {outputs}")

# # 使用回调
# response = sql_chain.invoke(
#     {"question": "有多少种不同的鲜花？"}, 
#     config={"callbacks": [MyCallbackHandler()]}
# )