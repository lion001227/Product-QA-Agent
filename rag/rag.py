"""
RAG 检索问答模块 —— LangGraph 版本

功能：
    1. 从 Chroma 向量数据库中加载已索引的文档向量
    2. 连接 DeepSeek 大语言模型（通过 OpenAI 兼容 API）
    3. 用 LangGraph 的 StateGraph 显式表达 RAG 流程（而不是隐式的 LCEL 链），
       方便未来插入更多节点（例如相关性判断、query 改写、检索不足时重试等）

工作流程（图结构）：
    START -> retrieve -> grade
                   ├─ 足够 -> generate -> END
                   └─ 不足 -> rewrite -> retrieve

    retrieve : 用 MMR 检索 + 去重，从向量库中拿到相关文档块
    generate : 结合检索到的上下文，调用 DeepSeek 生成回答

对外接口保持不变：
    rag_qa(question) -> str
    search_document_sources(question, k=5) -> str
均与旧版一致，tools.py 无需改动即可继续调用。

环境变量（需在 .env 文件中配置）：
    MODEL_DIR          : BGE 本地嵌入模型的路径
    OPENAI_API_KEY     : DeepSeek API 密钥（实际用于访问 DeepSeek 服务）
"""
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List,TypedDict
from langchain_core.documents import Document
from langgraph.graph import END,START,StateGraph

#代码读取.env文件
load_dotenv()

#创建检索，模型用本地路径下的模型，数据库用绝对路径
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH=os.path.join(BASE_DIR, "../vector_db")

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

RAG_SYSTEM_PROMPT="""你现在是一名金融领域的专家，根据资料回答问题.请注意：
        1.如果资料中涉及到表格，请确保读到完整的表格，有些表格是跨页的
        2.如果多个文档块包含相关信息，请综合所有信息给出完整答案
        3.如果资料中没有答案，明确说不知道"""

rag_prompt=ChatPromptTemplate.from_messages([
    ("system",RAG_SYSTEM_PROMPT),
    ("human","资料：{context},当前问题:{question}"),
])

class RAGState(TypedDict):
    """RAG 图的状态：START -> retrieve -> grade
                   ├─ 足够 -> generate -> END
                   └─ 不足 -> rewrite -> retrieve
    """
    question: str
    query:str
    docs:List[Document]
    context:str
    answer:str
    retrieval_count:int
    is_sufficient:bool



def format_docs(docs):
    """格式化文档"""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve_node(state:RAGState)->dict:
    """检索节点：MMR检索+按内容前80字去重"""
    docs = retriever.invoke(state["query"])
    seen=set()
    unique=[]
    for doc in docs:
        key=doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return {
        "docs":unique,
        "context":format_docs(unique),
        "retrieval_count":state.get("retrieval_count",0)+1,
            }

def retrieve_for_evaluation(question):
    """返回检索文档供评估文档精确度"""
    docs = retriever.invoke(question)
    seen = set()
    unique = []
    for doc in docs:
        key = doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique

def grade_node(state:RAGState)->dict:
    prompt=f"""判断以下资料是否足以回答用户的问题。
            用户问题：{state["question"]}
            资料：{state["context"][:5000]}
            只能输出enough或者 not_enough
"""
    result=llm.invoke(prompt).content.strip().lower()
    return{"is_sufficient":result=="enough"}

def route_after_grade(state:RAGState)->str:
    """路由逻辑，最多检索三次"""
    if state["is_sufficient"]:
        return "generate"
    if state["retrieval_count"]>=3:
        return "generate"
    return "rewrite"

def rewrite_node(state:RAGState)->dict:
    prompt=f"""
    用户原始问题是{state["question"]}
    当前检索词是{state["query"]}
    检索结果不足.请生成一个更适合向量检索的中文查询词。只输出改写后的查询词，不要解释
    """
    new_query=llm.invoke(prompt).content.strip()
    return {"query":new_query}

def generate_node(state:RAGState)->dict:
    """生成节点：综合上下文调用LLM生成回答"""
    messages=rag_prompt.invoke({"context":state["context"],"question":state["question"]})
    response=llm.invoke(messages)
    return {"answer":response.content}



def build_rag_graph():
    graph=StateGraph(RAGState)
    graph.add_node("retrieve",retrieve_node)
    graph.add_node("generate",generate_node)
    graph.add_node("rewrite",rewrite_node)
    graph.add_node("grade",grade_node)
    graph.add_conditional_edges("grade",route_after_grade,{"rewrite":"rewrite","generate":"generate"})

    graph.add_edge(START,"retrieve")
    graph.add_edge("retrieve","grade")
    graph.add_edge("rewrite","retrieve")
    graph.add_edge("generate",END)

    return graph.compile()

#编译好的RAG图，供rag_qa()复用
rag_app=build_rag_graph()

def rag_qa(question):
    """RAG问答函数，直接返回答案字符串"""
    result=rag_app.invoke({"question": question,"query":question,"retrieval_count":0})
    # print(result)
    return result["answer"]

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

# answer=chain.invoke( {"question": "目前支持互联网交易平台的哪些业务？"},
# print(answer.content)

