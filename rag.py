"""
RAG 接口文档数据处理与 LLM 连接模块

功能：
    1. 从 Chroma 向量数据库中加载已索引的文档向量
    2. 连接 DeepSeek 大语言模型（通过 OpenAI 兼容 API）
    3. 构建 RAG（检索增强生成）链，实现基于文档的智能问答

工作流程：
    用户问题 -> 向量检索（BGE 模型）-> 获取相关文档块 -> LLM 生成回答

环境变量（需在 .env 文件中配置）：
    MODEL_DIR          : BGE 本地嵌入模型的路径
    OPENAI_API_KEY     : DeepSeek API 密钥（实际用于访问 DeepSeek 服务）
"""
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#加入memory模块
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from dotenv import load_dotenv

#代码读取.env文件
load_dotenv()

#创建检索，模型用本地路径下的模型，数据库用绝对路径
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH=os.path.join(BASE_DIR, "vector_db")

embeddings = HuggingFaceBgeEmbeddings(
    model_name=os.getenv("MODEL_DIR"),  # 本地路径
    model_kwargs={'device': 'cpu'}
)

vectorstore=Chroma(persist_directory=VECTOR_DB_PATH,embedding_function=embeddings)

#使用MMR检索增加多样性
retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":20,
        "fetch_k":20,
        "lambda_mult":0.5
    }
) #从向量数据库中找出与用户问题最相似的 20 个文本块，并返回



#连接大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,
    api_key = os.getenv("OPENAI_API_KEY")

)

#创建历史记录存储,保存不同用户的聊天记录。
store={}
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id]=InMemoryChatMessageHistory()
    return store[session_id]

#文本提取
def get_question(x):

    print("进入get_question:", x)
    if isinstance(x, dict):
        return x.get("question","")
    elif isinstance(x, str):
        return x
    else:
        return str(x)

#去重函数：
def dedup_context(question):
    docs=retriever.invoke(get_question(question))
    seen=set()
    unique=[]
    for doc in docs:
        key=doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique

#rag chain
def create_rag_chain(question):
    prompt=ChatPromptTemplate.from_messages([

        ("system","""你现在是一名金融领域的专家，根据资料回答问题.请注意：
        1.如果资料中涉及到表格，请确保读到完整的表格，有些表格是跨页的
        2.如果多个文档块包含相关信息，请综合所有信息给出完整答案
        3.如果资料中没有答案，明确说不知道"""),
        MessagesPlaceholder(variable_name="history"),#自动处理历史消息
        ("human","资料:{context},当前问题:{question}")
       ] )


    rag_chain=(
        {
        "context":lambda x:dedup_context(x),
        "question":lambda x:get_question(x),
        "history":lambda x:x["history"]
    }|prompt|llm
    )

    chain=RunnableWithMessageHistory(rag_chain,get_session_history,input_messages_key="question",history_messages_key="history")
    return chain

def rag_qa(question):
    """RAG问答函数，直接返回答案字符串"""
    chain=create_rag_chain(question)
    result=chain.invoke(
        {"question":question},
        config={"configurable":{"session_id":"default_user"}}
    )
    return result.content

# answer=chain.invoke( {"question": "目前支持互联网交易平台的哪些业务？"},
#     config={"configurable": {"session_id": "default_user"}} )
# print(answer.content)

