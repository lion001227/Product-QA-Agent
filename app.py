"""
交易所文档读取 QA 助手 - Web 界面模块

功能：
    1. 提供基于 Streamlit 的图形用户界面
    2. 接收用户输入的问题
    3. 调用 RAG 链处理问题并展示回答

工作流程：
    用户输入问题 -> 调用 rag.chain -> 显示回答内容

依赖：
    - 需先运行 ingest.py 完成文档向量化
    - rag.py 模块必须可导入且正常运行

"""
import streamlit as st

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from rag import chain

st.title(
    "交易所文档读取QA助手"
)


question = st.text_input(
    "请输入问题"
)


if question:

    result = chain.invoke(question)

    st.write(
        result.content
    )