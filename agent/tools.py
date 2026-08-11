from langchain.tools import tool


@tool
def profile_search(question:str)->str:
    """
    查询交易所文档，回答产品相关问题
    当用户咨询交易所文档相关内容时使用
    """
    from rag import rag_qa
    result=rag_qa(question)
    return result
