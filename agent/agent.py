"""
交易所文档 QA 智能体 —— 手写 LangGraph StateGraph 版本

原来用 langchain.agents.create_agent 隐式搭好了一个 ReAct 循环。
这里改成显式的图结构，方便以后自己控制路由逻辑
（比如加限流、加日志节点、给某些问题强制走某个工具、加人工确认节点等）。

图结构：
    START -> agent --(有工具调用)--> tools -> agent -> ...
                  --(没有工具调用)--> END

    agent 节点：LLM 结合历史消息 + 系统提示词，决定直接回答还是调用某个工具
    tools 节点：执行 agent 节点请求的工具调用，把结果作为 ToolMessage 加回消息列表

多轮对话：
    用 SqliteSaver 做 checkpointer，相同的 thread_id 代表同一段对话，
    LangGraph 会自动读取此前的消息历史，本轮结束后再把新消息追加保存回去，重启程序后之前的对话仍能保留
    （用法和旧版 create_agent 完全一致，thread_id 语义不变）。
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

SYSTEM_PROMPT=system_prompt="""
你现在是一个金融交易所文档的QA智能助手，你会收到完整的历史对话，结合历史理解用户提问的问题，明白上下文的指代关系。
1.当用户询问交易所文档中的内容时，必须调用profile_search，
调用工具前，请把当前问题改写成不依赖上下文，含明确实体名称的完整问题
例如：
历史：用户询问“上海报价回购业务是什么”
当前：用户问“它适用于哪些客户”
调用工具时应查询：“上海报价回购业务适用于哪些客户？”

2.当用户询问问题依据、来源文件、原文内容或页码时，调用return_document_sources.
如果用户使用“这个答案”“他的来源”等指代,
先根据历史对话把问题改写成包含明确业务名称的完整问题，再调用return_document_sources,
调用 return_document_sources 后，必须逐字保留工具返回的全部内容，
直接作为最终回答输出。
不得自行改写工具返回结果，不得增加“来源依据”“原文内容”等标题，
不得省略“文件”“位置”“相关原文”字段。
即使位置显示为“页码未知”，也必须原样输出。

例如：
用户：上海报价回购是什么业务？
助手：……（调用 profile_search）
用户：这个回答的依据在哪？
助手：……（调用 document_source_search）
【来源 1】
文件：新一代Win版上海报价回购操作说明.docx
位置：页码未知
相关原文：……

3.当用户询问上交所最新公告、近期公告、今日公告，
或某个关键词相关的最新官方公告时，必须调用 sse_latest_announcements。
当用户明确指定上交所时，只调用sse_latest_announcements，当用户明确指定深交所时，只调用szse_latest_announcements
不能混用两个交易所的公告结果

该工具返回的是实时官网信息。
回答时必须保留公告发布日期和官网链接，
不得把实时公告内容当作本地知识库资料，
也不得编造未在工具结果中的公告。

不要在没有查询交易所文档资料的情况下编造信息
需要调用工具时，必须发起结构化 tool call。
不得向用户输出“我将调用某工具”“让我调用工具”等过程说明。
工具返回后，再基于工具结果给出最终答案。
    """

tools=[profile_search,return_document_sources,sse_latest_announcements,szse_latest_announcements]
llm=ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY")
).bind_tools(tools)

class AgentState(TypedDict):
    """agent 图的状态：只需要维护消息列表，add_messages 会自动做增量合并
        （新消息追加到历史后面，而不是整体覆盖）"""
    messages: Annotated[Sequence[BaseMessage],add_messages]

def call_model(state:AgentState)->dict:
    """agent节点：读取历史信息+系统提示词，决定直接回答还是发起工具调用"""
    messages=[SystemMessage(content=system_prompt),*state["messages"]]
    response=llm.invoke(messages)
    # print(messages)
    return {"messages":[response]}

# def should_continue(state:AgentState)->str:
#     """条件路由：判断agent节点是否带有工具调用请求"""
#     last_message=state["messages"][-1]
#     if getattr(last_message,"tool_calls",None):
#         return "tools"
#     return END

def build_agent_graph()->StateGraph:
    graph=StateGraph(AgentState)
    graph.add_node("agent",call_model)
    graph.add_node("tools",ToolNode(tools))

    graph.add_edge(START,"agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools":"tools",END:END})
    #工具执行完之后回到agent节点，让llm看工具结果，决定是否继续调用或直接回答
    graph.add_edge("tools","agent")

    return graph

#设置 SQLite 检查点存储器（Checkpointer），让对话状态在程序重启后依然能持久保存。
connection=sqlite3.connect(
    "langgraph_checkpoints.sqlite",
    check_same_thread=False
)

checkpointer=SqliteSaver(connection)
checkpointer.setup()

agent=build_agent_graph().compile(checkpointer=checkpointer)


# def ask_agent(question:str,thread_id:str):
#     result=agent.invoke({
#         "messages":[
#             {
#                 "role":"user",
#                 "content":question
#             }
#         ]
#     },
#     config={
#         "configurable":
#             {
#                 "thread_id":thread_id  #相同的thread_id代表同一段对话，langchain会自动读取此前信息，再将本轮回答保存回去
#             }
#     })
#     return result["messages"][-1].content


def stream_agent(question:str,thread_id:str):
    """流式输出"""
    config={"configurable":{"thread_id":thread_id }}
    for token,metadata in agent.stream(
            {"messages":[{"role":"user","content":question}]},
        config=config,
        stream_mode="messages"
    ):
        is_agent_response = metadata.get("langgraph_node") == "agent"
        is_ai_chunk = isinstance(token, AIMessageChunk)
        if is_agent_response and is_ai_chunk and token.content:
            yield token.content