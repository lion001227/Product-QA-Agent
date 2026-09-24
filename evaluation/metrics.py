import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
llm=ChatOpenAI( model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"))

def calculate_tool_selection_accuracy(results):
    correct = 0
    for result in results:
        expect=set(result['expected_tool'])
        actual=set(result['tool_calls'])

        if expect == actual:
            correct += 1
    return correct/len(results)

def calculate_average_latency(results):
    avg_latency = 0
    for result in results:
        avg_latency += result['latency']
    return avg_latency / len(results)

def calculate_failure_rate(results):
    failed=sum(1 for result in results if result.get("failure",False))
    return failed/len(results)

def judge_answer(results):
    correct=0

    for result in results:
        expect=result["expected_answer"]
        actual=result["agent_answer"]
        prompt = f"""
            你是一名法官
            1.根据实际回答{expect}判断是否符合标准答案{actual}
            2.符合返回1，不符合返回0
            只返回数字0或1，不要返回其余结果
        """
        response=llm.invoke([SystemMessage(content=prompt)])
        correct+=int(response.content.strip())

    return correct/len(results)

def calculate_token_usage(results):
    total_tokens=0
    input_tokens=0
    output_tokens=0
    for result in results:
        total_tokens+=result["total_tokens"]
        input_tokens+=result["input_tokens"]
        output_tokens+=result["output_tokens"]
    return{
        "average_total_tokens":total_tokens/len(results),
        "average_input_tokens":input_tokens/len(results),
        "average_output_tokens":output_tokens/len(results),
    }

def judge_retriever_accuracy(results):
    score=0
    rag_results=[result for result in results if result["type"] in ("rag","source")]
    for rag_result in rag_results:
        prompt = f"""
            你是一个 RAG 检索评估器
            1.对retriever返回的文件{rag_result["retriever_documents"]}进行相关性评分
            2.文档相关给1分，文档部分相关根据相关性给0.1-0.9分，完全不相关给0分
            3.返回所有文档的得分的平均分
            只返回一个数字，不要返回其他内容
            """
        response=llm.invoke([SystemMessage(content=prompt)]).content
        score+=float(response)
    return score/len(rag_results)






