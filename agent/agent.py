"""
交易所文档 QA 智能体 —— Supervisor 多 Agent LangGraph 实现

图由一个 supervisor 和三个专职 agent 组成。supervisor 根据最新的用户问题
选择 rag、source 或 announcement；每个专职 agent 都只绑定自己可用的工具，
避免将文档检索、来源追溯和交易所公告查询混在同一组工具中。

图结构：
    START -> supervisor -> rag ----------(需要工具)--> rag_tools ----------┐
                          source -------(需要工具)--> source_tools -------┼-> 对应 agent
                          announcement -(需要工具)--> announcement_tools -┘
                          任一专职 agent --(无需工具)--> END

    supervisor：只负责路由，不直接面向用户回答。
    专职 agent：结合历史消息决定回答或发起结构化 tool call。
    ToolNode：执行工具，并将 ToolMessage 追加回 messages，供对应 agent 生成最终答复。

多轮对话：
    SqliteSaver 以 thread_id 为键保存图状态。同一 thread_id 会自动读取既有
    messages，并在本轮完成后持久化新增消息；应用重启后仍可继续该会话。
"""

import os

from langchain_openai import ChatOpenAI
from .tools import profile_search,return_document_sources,sse_latest_announcements,szse_latest_announcements
from dotenv import load_dotenv
from typing import Annotated,Sequence,TypedDict
from langgraph.graph import END,START,StateGraph
from langchain_core.messages import  BaseMessage,SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AIMessageChunk

load_dotenv()

llm=ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY")
)

class AgentState(TypedDict):
    """agent 图的状态：只需要维护消息列表，add_messages 会自动做增量合并
        （新消息追加到历史后面，而不是整体覆盖）"""
    messages: Annotated[Sequence[BaseMessage],add_messages]
    next:str

supervisor_prompt="""
你是一个任务路由器，需要判断用户的问题应该交给哪个agent。
可选agent：
1. rag
处理交易所文档知识库相关问题，例如：
- 业务规则
- 交易规则
- 业务定义
- 制度说明
- 文档内容

2. source
处理文档来源、依据、出处相关问题，例如：
- 这个答案来自哪个文件？
- 依据是什么？
- 原文在哪里？
- 第几页？
- 给我相关原文

3. announcement
处理交易所最新公告，例如：
- 上交所最新公告
- 深交所最新公告
- 某股票最新公告

只返回一个单词：

rag
source
announcement
"""

def supervisor_agent(state:AgentState):
    """supervisor节点：判断交给哪个agent处理"""
    response=llm.invoke([SystemMessage(content=supervisor_prompt),state["messages"][-1]])
    route=response.content.strip().lower()
    return {"next":route}

rag_prompt="""
你是交易所文档知识库 Agent。

你的任务是回答用户关于交易所业务、规则、制度和文档内容的问题。

如果需要查询知识库，必须调用 profile_search。

如果用户的问题依赖之前的对话，请结合历史上下文理解问题。

不要编造知识库中不存在的信息。
"""

rag_llm=llm.bind_tools([profile_search])

def rag_agent(state:AgentState):
    """RAG Agent"""
    response=rag_llm.invoke([SystemMessage(content=rag_prompt),*state["messages"]])
    # ToolNode 和 add_messages 都约定使用 messages。若写成 message，
    # ToolNode 会把最后一条用户消息误当作工具调用请求并报错。
    return {"messages":[response]}
rag_tools=ToolNode([profile_search])

source_prompt="""
你是文档来源查询 Agent。

你的任务是查询交易所知识库中的：
- 文件来源
- 页码
- 相关原文
- 文档依据

必须调用 return_document_sources。

Tool 返回的内容应该直接返回给用户，不要修改。
"""
source_llm=llm.bind_tools([return_document_sources])
def source_agent(state:AgentState):
    """Source Agent"""
    response=source_llm.invoke([SystemMessage(content=source_prompt),*state["messages"]])
    return {"messages":[response]}
source_tools=ToolNode([return_document_sources])

announcement_prompt="""
你是交易所公告查询 Agent。

你的任务是查询交易所最新公告。

如果用户明确说：
- 上交所 → 调用 sse_latest_announcements
- 深交所 → 调用 szse_latest_announcements

不要混用两个交易所。

如果用户没有明确指定交易所，根据上下文判断。
"""
announcement_llm=llm.bind_tools([sse_latest_announcements,szse_latest_announcements])
def announcement_agent(state:AgentState):
    """Announcement Agent"""
    response=announcement_llm.invoke([SystemMessage(content=announcement_prompt),*state["messages"]])
    return {"messages":[response]}
announcement_tools=ToolNode([sse_latest_announcements,szse_latest_announcements])

def supervisor_router(state:AgentState):
    return state["next"]

def build_agent_graph()->StateGraph:
    graph=StateGraph(AgentState)
    graph.add_node("supervisor",supervisor_agent)
    graph.add_node("rag",rag_agent)
    graph.add_node("source",source_agent)
    graph.add_node("announcement",announcement_agent)
    graph.add_node("rag_tools", rag_tools)
    graph.add_node("source_tools", source_tools)
    graph.add_node("announcement_tools", announcement_tools)

    graph.add_edge(START,"supervisor")
    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {"rag":"rag",
                  "source":"source",
                  "announcement":"announcement"
                  })
    # 只有模型实际发出了 tool_call 才进入 ToolNode；工具结果再交回
    # 对应 agent，由模型组织最终答复。没有 tool_call 时直接结束。
    graph.add_conditional_edges("rag", tools_condition, {"tools": "rag_tools", END: END})
    graph.add_edge("rag_tools", "rag")
    graph.add_conditional_edges("source", tools_condition, {"tools": "source_tools", END: END})
    graph.add_edge("source_tools", "source")
    graph.add_conditional_edges(
        "announcement", tools_condition, {"tools": "announcement_tools", END: END}
    )
    graph.add_edge("announcement_tools", "announcement")


    return graph

#设置 SQLite 检查点存储器（Checkpointer），让对话状态在程序重启后依然能持久保存。
connection=sqlite3.connect(
    "langgraph_checkpoints.sqlite",
    check_same_thread=False
)

checkpointer=SqliteSaver(connection)
checkpointer.setup()

agent=build_agent_graph().compile(checkpointer=checkpointer)

def stream_agent(question:str,thread_id:str):
    """流式输出"""
    config={"configurable":{"thread_id":thread_id }}
    for token,metadata in agent.stream(
            {"messages":[{"role":"user","content":question}]},
        config=config,
        stream_mode="messages"
    ):
        is_agent_response = metadata.get("langgraph_node") in {
            "rag", "source", "announcement"
        }
        is_ai_chunk = isinstance(token, AIMessageChunk)
        if is_agent_response and is_ai_chunk and token.content:
            yield token.content
