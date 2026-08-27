import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from .tools import profile_search,return_document_sources,sse_latest_announcements,szse_latest_announcements
from dotenv import load_dotenv
load_dotenv()

llm=ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY")

)

tools=[profile_search,return_document_sources,sse_latest_announcements,szse_latest_announcements]
checkpointer=InMemorySaver()

agent=create_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    system_prompt="""
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
    """
)

def ask_agent(question:str,thread_id:str):
    result=agent.invoke({
        "messages":[
            {
                "role":"user",
                "content":question
            }
        ]
    },
    config={
        "configurable":
            {
                "thread_id":thread_id  #相同的thread_id代表同一段对话，langchain会自动读取此前信息，再将本轮回答保存回去
            }
    })
    return result["messages"][-1].content