import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser

os.environ["OPENAI_API_KEY"] = ''
from langchain_core.prompts import PromptTemplate

def sequential_chain():
    from langchain_openai import OpenAI
    from langchain_community.chains import LLMChain, SequentialChain
    from langchain_core.prompts import PromptTemplate

    # 第一个LLMChain：生成鲜花的介绍
    llm = OpenAI(temperature=.7)
    template = """
    你是一个植物学家。给定花的名称和类型，你需要为这种花写一个30字左右的介绍。
    花名: {name}
    颜色: {color}
    植物学家: 这是关于上述花的介绍:"""
    prompt_template = PromptTemplate(
        input_variables=["name", "color"],
        template=template
    )
    introduction_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_key="introduction"
    )

    # 第二个LLMChain：根据鲜花的介绍写出鲜花的评论
    template = """
    你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇30字左右的评论。
    鲜花介绍:
    {introduction}
    花评人对上述花的评论:"""
    prompt_template = PromptTemplate(
        input_variables=["introduction"],
        template=template
    )
    review_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_key="review"
    )

    # 第三个LLMChain：根据鲜花的介绍和评论写出一篇自媒体的文案
    template = """
    你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300字左右。
    鲜花介绍:
    {introduction}
    花评人对上述花的评论:
    {review}
    社交媒体帖子:
    """
    prompt_template = PromptTemplate(
        input_variables=["introduction", "review"],
        template=template
    )
    social_post_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_key="social_post_text"
    )

    # 总的链：按顺序运行三个链
    overall_chain = SequentialChain(
        chains=[introduction_chain, review_chain, social_post_chain],
        input_variables=["name", "color"],
        output_variables=["introduction", "review", "social_post_text"],
        verbose=True
    )

    # 运行链并打印结果
    result = overall_chain.invoke({
        "name": "百合",
        "color": "白色"
    })
    print(result)


# 使用最新runnable 范式代替之前的 llmChain 和 sequentialChain
def sequential_chain_new():
    from langchain_openai import OpenAI
    
    # 定义各个步骤的提示模板
    introduction_template = PromptTemplate.from_template("""
    你是一个植物学家。给定花的名称和类型，你需要为这种花写一个100字左右的介绍。
    花名: {name}
    颜色: {color}
    植物学家: 这是关于上述花的介绍:""")
    
    review_template = PromptTemplate.from_template("""
    你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇100字左右的评论。
    鲜花介绍:
    {introduction}
    花评人对上述花的评论:""")
    
    social_post_template = PromptTemplate.from_template("""
    你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300字左右。
    鲜花介绍:
    {introduction}
    花评人对上述花的评论:
    {review}
    社交媒体帖子:""")
    
    # 创建LLM实例
    llm = OpenAI(temperature=.7)
    output_parser = StrOutputParser()
    
    # 创建各个步骤的链
    introduction_chain = introduction_template | llm | output_parser
    review_chain = {"introduction": introduction_chain} | review_template | llm | output_parser
    social_post_chain = (
        {
            "introduction": introduction_chain,
            "review": review_chain
        } 
        | social_post_template 
        | llm 
        | output_parser
    )
    
    # 组合最终的链，输出键名字而已
    full_chain = RunnableParallel(
        introduction1=introduction_chain,
        review1=review_chain,
        social_post_text=social_post_chain
    )
    
    # 运行链并打印结果
    inputs = {
        "name": "喇叭花",
        "color": "白色"
    }

    result = full_chain.invoke(inputs)
    print(result)

sequential_chain_new()