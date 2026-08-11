"""
STEP 接口文档数据处理与向量化模块

功能：
1. 加载指定 PDF 文档
2. 将文档切分为语义块
3. 使用 BGE 模型生成向量并持久化到 Chroma DB

环境变量（需在 .env 文件中配置）：
    MODEL_DIR          : BGE 本地嵌入模型的路径
    OPENAI_API_KEY     : DeepSeek API 密钥（实际用于访问 DeepSeek 服务）
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from glob import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


#读取文件
# loader=PyPDFLoader("data/STEP.pdf")
# documents=loader.load()

pdf_dir="data"
pdf_files=glob(os.path.join(pdf_dir,"*.pdf"))
all_documents=[]

for pdf in pdf_files:
    print("正在加载文档：",pdf)
    loader=PyPDFLoader(pdf)
    documents=loader.load()
    all_documents.extend(documents) #合并所有文档


# print(len(documents))

#print(documents[0].page_content[:500])

#切割文本知识块
slipper=RecursiveCharacterTextSplitter(chunk_size=6000,chunk_overlap=500) #每个文本块最大长度=6000，相邻两块之间重叠的字符数=500

chunks=slipper.split_documents(all_documents)

#print(len(chunks))

#创建向量数据库，模型下载到了本地，配置本地路径，数据库用绝对路径
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH=os.path.join(BASE_DIR, "vector_db")

embeddings = HuggingFaceBgeEmbeddings(
    model_name=os.getenv("MODEL_DIR"),  # 本地路径
    model_kwargs={'device': 'cpu'}
)

vectorstore=Chroma.from_documents(documents=chunks,
                                  embedding=embeddings,
                                  persist_directory=VECTOR_DB_PATH)#指定向量数据库的持久化存储绝对路径vector_db


