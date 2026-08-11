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

st.set_page_config(
    page_title="交易所文档QA助手",
    page_icon="💬",
    layout="wide"
)

st.title("💬 交易所文档读取 QA 助手")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 接收用户输入（自带清空功能）
if question := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    # 调用RAG链获取回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                result = chain.invoke({
                    "question": question,
                    "history": []
                },
                config={
                    "configurable": {
                        "session_id": "default_user"
                    }
                }
                )
                answer = result.content
            except Exception as e:
                answer = f"❌ 处理请求时出错：{str(e)}"

        st.write(answer)

    # 保存助手回复
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# 侧边栏信息
with st.sidebar:
    st.markdown("## 📚 关于助手")
    st.markdown("""
    这是一个基于 **RAG (检索增强生成)** 技术的文档问答助手。
    
    ### 功能特点：
    - 📄 读取交易所相关文档
    - 🔍 智能检索相关内容
    - 💡 基于上下文生成回答
    
    ### 使用说明：
    直接在输入框输入问题，按回车发送
    """)

    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("💡 提示：问题越具体，回答越精准")