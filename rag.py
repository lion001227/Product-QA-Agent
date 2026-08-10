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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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

retriever=vectorstore.as_retriever(search_kwargs={"k":4}) #从向量数据库中找出与用户问题最相似的 4 个文本块，并返回

#连接大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,
    api_key = os.getenv("OPENAI_API_KEY")

)

#rag chain
prompt=ChatPromptTemplate.from_template(
   """
   你现在是一名金融领域的专家，根据以下资料回答：{context}，问题：{question}
   """ )

chain=(
    {
    "context":retriever,
    "question":RunnablePassthrough()
}|prompt|llm
)

# answer=chain.invoke("目前支持互联网交易平台的哪些业务？")
# print(answer.content)

