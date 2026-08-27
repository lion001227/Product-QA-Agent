# import os
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
#
# load_dotenv()
#
# model_dir = os.getenv("MODEL_DIR")
#
# print("模型路径：", model_dir)
# print("路径是否存在：", os.path.exists(model_dir))
# print("是否为目录：", os.path.isdir(model_dir))
#
# print("\n目录内容：")
# for item in os.listdir(model_dir):
#     print("  ", item)
#
# model = SentenceTransformer(
#     model_dir,
#     local_files_only=True
# )
#
# print("\n模型加载成功！")
#
# embedding = model.encode(
#     "这是一个测试文本",
#     normalize_embeddings=True
# )
#
# print("向量维度：", len(embedding))


import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings

model_dir = os.getenv("MODEL_DIR")

print("模型路径：", model_dir)

embeddings = HuggingFaceEmbeddings(
    model_name=model_dir,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

print("Embedding 模型加载成功！")

vector = embeddings.embed_query("这是一个测试文本")

print("向量维度：", len(vector))
print("前10个向量值：", vector[:10])