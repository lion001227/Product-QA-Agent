import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .tools import profile_search
from dotenv import load_dotenv
load_dotenv()

llm=ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,
    api_key=os.getenv("API_KEY")

)

tools=[profile_search]

agent=create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
你现在是一个金融交易所文档的QA智能助手，你的任务是帮助用户回答产品相关问题，
当用户询问交易所文档中的内容时，请调用profile_search工具进行查询，
不要在没有查询交易所文档资料的情况下编造信息
    """
)

def ask_agent(question):
    result=agent.invoke({
        "messages":[
            {
                "role":"user",
                "content":question
            }
        ]
    })
    return result["messages"][-1].content