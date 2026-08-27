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
from operator import itemgetter
import os
from dotenv import load_dotenv
from pathlib import Path

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
        "k":5,
        "fetch_k":20,
        "lambda_mult":0.5
    }
) #从向量数据库中找出与用户问题最相似的 5 个文本块，初筛数量是20，相关性是0.5，并返回



#连接大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,
    api_key = os.getenv("OPENAI_API_KEY")

)


def get_question(x):
    """文本提取"""
    print("进入get_question:", x)
    if isinstance(x, dict):
        return x.get("question","")
    elif isinstance(x, str):
        return x
    else:
        return str(x)

def format_docs(docs):
    """格式化文档"""
    return "\n\n".join(doc.page_content for doc in docs)


def dedup_context(question):
    """去重函数"""
    docs=retriever.invoke(get_question(question))
    seen=set()
    unique=[]
    for doc in docs:
        key=doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique

def search_document_sources(question:str,k:int=5)->str:
    """检索与问题相关的文档来源，页码和原文片段"""
    docs=vectorstore.similarity_search(question,k=k)

    results=[]
    seen=set()

    for doc in docs:
        source=doc.metadata.get("source","未知文件")
        file_name=Path(source).name

        #PyPDFLoader的page从0开始
        page=doc.metadata.get("page")
        page_text=f"第{page+1}页"if page is not None else "页码未知"

        #同一文件，同一页，相同片段只保留一次
        key=(source,page,doc.page_content[:100])
        if key in seen:
            continue
        seen.add(key)

        excerpt=doc.page_content.strip().replace("\n","")
        if len(excerpt)>300:
            excerpt=excerpt[:300]+"..."

        results.append(
            f"【来源{len(results)+1}】\n"
            f"文件：{file_name}\n"
            f"位置：{page_text}\n"
            f"相关原文{excerpt}"
        )

    if not results:
        return "未在知识库中检索到与该问题相关的来源"

    return "\n\n".join(results)


def create_rag_chain():
    """RAG CHAIN"""
    prompt=ChatPromptTemplate.from_messages([

        ("system","""你现在是一名金融领域的专家，根据资料回答问题.请注意：
        1.如果资料中涉及到表格，请确保读到完整的表格，有些表格是跨页的
        2.如果多个文档块包含相关信息，请综合所有信息给出完整答案
        3.如果资料中没有答案，明确说不知道"""),
        ("human","资料:{context},当前问题:{question}")
       ] )

    return (
        {
        "context":lambda x:format_docs(dedup_context(x["question"])),
        "question":itemgetter("question"),
    }|prompt|llm
    )

rag_chain=create_rag_chain()

def rag_qa(question):
    """RAG问答函数，直接返回答案字符串"""
    return rag_chain.invoke({"question":question}).content

# answer=chain.invoke( {"question": "目前支持互联网交易平台的哪些业务？"},
# print(answer.content)

