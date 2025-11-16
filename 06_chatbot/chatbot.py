import os
os.environ["OPENAI_API_KEY"] = ''
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI # ChatOpenAI模型
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory 


# 使用RunnableWithMessageHistory的现代化版本
class ModernChatbotWithRetrieval:
    def __init__(self, dir):
        # 加载Documents
        base_dir = dir
        documents = []
        for file in os.listdir(base_dir): 
            file_path = os.path.join(base_dir, file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        
        # 文本的分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents)
        
        # 向量数据库
        self.vectorstore = Qdrant.from_documents(
            documents=all_splits,
            embedding=OpenAIEmbeddings(),
            location=":memory:",
            collection_name="my_documents",
        )
        
        # 初始化LLM
        self.llm = ChatOpenAI()
        
        
        # Contextualize question
        contextualize_q_system_prompt = """根据聊天历史和最新用户问题，\
        可能引用了聊天历史中的上下文，形成一个独立的问题。\
        不要回答问题，只需要在必要时重新表述它，否则原样返回。"""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.vectorstore.as_retriever(), contextualize_q_prompt
        )
        
        # Answer question
        qa_system_prompt = """你是一个问答助手，根据检索到的上下文回答问题。\
        如果不知道答案，就说你不知道。最多使用三句话，保持答案简洁。\
        \n\n{context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取或创建会话历史"""
        if not hasattr(self, "_histories"):
            self._histories = {}
        if session_id not in self._histories:
            self._histories[session_id] = InMemoryChatMessageHistory()
        return self._histories[session_id]
    def chat_loop(self):
        print("Modern Chatbot 已启动! 输入'exit'来退出程序。")
        session_id = "default_session"  # 简单的会话ID
        while True:
            user_input = input("你: ")
            if user_input.lower() == 'exit':
                print("再见!")
                break
                
             # 获取当前会话历史
            history = self.get_session_history(session_id)
            # 调用现代RAG链
            response = self.rag_chain.invoke({
                "input": user_input,
                "chat_history": history.messages
            })
            
            # 更新聊天历史
            history.add_message(HumanMessage(content=user_input))
            history.add_message(AIMessage(content=response["answer"]))
            
            print(f"Chatbot: {response['answer']}")
if __name__ == "__main__":
    # 启动Chatbot
    folder = "."
    bot = ModernChatbotWithRetrieval(folder)
    bot.chat_loop()

# 做了历史信息管理和rag 检索，效果一般